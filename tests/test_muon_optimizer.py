"""Tests for the Muon optimizer utilities (split_param_groups_for_muon, fallback optimizer, etc.)."""

import logging

import pytest
import torch

from musubi_tuner.optimizers.muon import FallbackSingleDeviceMuonWithAuxAdam, create_muon_optimizer
from musubi_tuner.optimizers.muon_util import (
    LayerFilter,
    MuonParamStats,
    get_default_patterns,
    print_muon_summary,
    split_param_groups_for_muon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param(shape, name, requires_grad=True):
    """Create a named parameter and return (name, param)."""
    p = torch.nn.Parameter(torch.randn(shape), requires_grad=requires_grad)
    return name, p


def _build_param_name_map(named_params):
    """Build {id(param): name} from a list of (name, param) pairs."""
    return {id(p): n for n, p in named_params}


def _make_single_group(named_params, lr=None):
    """Wrap named params into a single param-group dict."""
    group = {"params": [p for _, p in named_params]}
    if lr is not None:
        group["lr"] = lr
    return [group]


# ---------------------------------------------------------------------------
# LayerFilter
# ---------------------------------------------------------------------------


class TestLayerFilter:
    def test_exact_substring_match(self):
        f = LayerFilter("transformer_blocks")
        assert f.matches("lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight")

    def test_dot_to_underscore_match(self):
        """LoRA module names are underscore-normalized; patterns with dots should still match."""
        f = LayerFilter("encoder.block")
        assert f.matches("lora_unet_encoder_block_3_proj.lora_down.weight")

    def test_no_match(self):
        f = LayerFilter("transformer_blocks")
        assert not f.matches("lora_unet_head_proj.lora_down.weight")


# ---------------------------------------------------------------------------
# get_default_patterns
# ---------------------------------------------------------------------------


class TestGetDefaultPatterns:
    def test_exact_key(self):
        assert "blocks" in get_default_patterns("wan")

    def test_fuzzy_key(self):
        # "hunyuan_video" contains "hunyuan"
        patterns = get_default_patterns("hunyuan_video")
        assert "transformer_blocks" in patterns

    def test_unknown_falls_back(self):
        patterns = get_default_patterns("totally_unknown_model")
        assert patterns == get_default_patterns("default")


# ---------------------------------------------------------------------------
# split_param_groups_for_muon
# ---------------------------------------------------------------------------


class TestSplitParamGroups:
    """Core tests for the Muon/Adam parameter splitting logic."""

    def test_2d_in_hidden_layer_goes_to_muon(self):
        named = [_make_param((64, 32), "transformer_blocks.0.attn.to_q.weight")]
        groups = _make_single_group(named, lr=0.02)
        name_map = _build_param_name_map(named)

        result, stats = split_param_groups_for_muon(groups, name_map, hidden_layer_patterns=["transformer_blocks"], verbose=False)
        assert stats.muon_count == 1
        assert stats.adam_count == 0

    def test_1d_always_goes_to_adam(self):
        """1D params (bias, norm scales, DoRA magnitudes) should always be Adam regardless of name."""
        named = [_make_param((64,), "transformer_blocks.0.attn.bias")]
        groups = _make_single_group(named, lr=0.02)
        name_map = _build_param_name_map(named)

        result, stats = split_param_groups_for_muon(groups, name_map, hidden_layer_patterns=["transformer_blocks"], verbose=False)
        assert stats.muon_count == 0
        assert stats.adam_count == 1

    def test_non_matching_2d_goes_to_adam(self):
        named = [_make_param((64, 32), "head.proj.weight")]
        groups = _make_single_group(named, lr=0.02)
        name_map = _build_param_name_map(named)

        result, stats = split_param_groups_for_muon(groups, name_map, hidden_layer_patterns=["transformer_blocks"], verbose=False)
        assert stats.muon_count == 0
        assert stats.adam_count == 1

    def test_mixed_params_split_correctly(self):
        named = [
            _make_param((64, 32), "transformer_blocks.0.attn.to_q.weight"),
            _make_param((64,), "transformer_blocks.0.attn.bias"),
            _make_param((64, 32), "head.proj.weight"),
        ]
        groups = _make_single_group(named, lr=0.02)
        name_map = _build_param_name_map(named)

        result, stats = split_param_groups_for_muon(groups, name_map, hidden_layer_patterns=["transformer_blocks"], verbose=False)
        assert stats.muon_count == 1  # only the 2D transformer param
        assert stats.adam_count == 2  # 1D bias + non-matching 2D

    def test_muon_lr_override(self):
        named = [_make_param((64, 32), "transformer_blocks.0.weight")]
        groups = _make_single_group(named, lr=1e-4)
        name_map = _build_param_name_map(named)

        result, stats = split_param_groups_for_muon(
            groups, name_map, hidden_layer_patterns=["transformer_blocks"], muon_lr=0.02, verbose=False
        )
        muon_group = [g for g in result if g["use_muon"]][0]
        assert muon_group["lr"] == 0.02

    def test_muon_lr_preserves_group_lr_when_no_override(self):
        """When muon_lr is None, Muon groups inherit the incoming group LR."""
        named = [_make_param((64, 32), "transformer_blocks.0.weight")]
        groups = _make_single_group(named, lr=0.03)
        name_map = _build_param_name_map(named)

        result, stats = split_param_groups_for_muon(
            groups, name_map, hidden_layer_patterns=["transformer_blocks"], muon_lr=None, verbose=False
        )
        muon_group = [g for g in result if g["use_muon"]][0]
        assert muon_group["lr"] == 0.03

    def test_adam_lr_uses_explicit_default_when_muon_lr_not_set(self):
        """When muon_lr is None, incoming LR is in Muon-scale. Adam should use explicit adam_lr."""
        named = [
            _make_param((64, 32), "transformer_blocks.0.weight"),
            _make_param((64,), "transformer_blocks.0.bias"),
        ]
        groups = _make_single_group(named, lr=0.02)  # Muon-scale LR
        name_map = _build_param_name_map(named)

        result, stats = split_param_groups_for_muon(
            groups, name_map, hidden_layer_patterns=["transformer_blocks"], muon_lr=None, adam_lr=3e-4, verbose=False
        )
        adam_group = [g for g in result if not g["use_muon"]][0]
        assert adam_group["lr"] == 3e-4  # NOT 0.02

    def test_single_group_adam_uses_adam_lr_even_with_muon_lr_set(self):
        """Single group with muon_lr set: Adam subgroup uses adam_lr, not the group's base LR.

        This prevents under-training when --learning_rate is a tiny framework default (e.g. 2e-6)
        and the user sets muon_lr=0.02 expecting Adam-side to use adam_lr=3e-4.
        """
        named = [
            _make_param((64, 32), "transformer_blocks.0.weight"),
            _make_param((64,), "transformer_blocks.0.bias"),
        ]
        groups = _make_single_group(named, lr=2e-6)  # framework default, NOT intended for Adam
        name_map = _build_param_name_map(named)

        result, stats = split_param_groups_for_muon(
            groups, name_map, hidden_layer_patterns=["transformer_blocks"], muon_lr=0.02, adam_lr=3e-4, verbose=False
        )
        adam_group = [g for g in result if not g["use_muon"]][0]
        assert adam_group["lr"] == 3e-4  # adam_lr, NOT the tiny base_lr

    def test_multi_group_lora_plus_lr_preservation(self):
        """LoRA+ creates multiple groups with different LRs. Both should be preserved."""
        named_a = [_make_param((64, 32), "transformer_blocks.0.lora_down.weight")]
        named_b = [_make_param((32, 64), "transformer_blocks.0.lora_up.weight")]
        name_map = {**_build_param_name_map(named_a), **_build_param_name_map(named_b)}

        groups = [
            {"params": [named_a[0][1]], "lr": 1e-4},
            {"params": [named_b[0][1]], "lr": 5e-4},
        ]

        result, stats = split_param_groups_for_muon(
            groups, name_map, hidden_layer_patterns=["transformer_blocks"], muon_lr=0.02, verbose=False
        )
        muon_groups = [g for g in result if g["use_muon"]]
        # Both Muon groups should get the override LR
        assert all(g["lr"] == 0.02 for g in muon_groups)

    def test_multi_group_adam_subset_preserves_per_group_lr(self):
        """When LoRA+ groups have 1D params, Adam subsets preserve their group's LR."""
        named_a = [
            _make_param((64, 32), "transformer_blocks.0.lora_down.weight"),
            _make_param((64,), "transformer_blocks.0.lora_down.bias"),
        ]
        named_b = [
            _make_param((32, 64), "transformer_blocks.0.lora_up.weight"),
            _make_param((32,), "transformer_blocks.0.lora_up.bias"),
        ]
        name_map = {**_build_param_name_map(named_a), **_build_param_name_map(named_b)}

        groups = [
            {"params": [p for _, p in named_a], "lr": 1e-4},
            {"params": [p for _, p in named_b], "lr": 5e-4},
        ]

        result, stats = split_param_groups_for_muon(
            groups, name_map, hidden_layer_patterns=["transformer_blocks"], muon_lr=0.02, adam_lr=3e-4, verbose=False
        )
        adam_groups = [g for g in result if not g["use_muon"]]
        adam_lrs = sorted(g["lr"] for g in adam_groups)
        # Should preserve 1e-4 and 5e-4, NOT both be 3e-4
        assert adam_lrs == [1e-4, 5e-4]

    def test_unmatched_patterns_tracked(self):
        named = [_make_param((64, 32), "transformer_blocks.0.weight")]
        groups = _make_single_group(named)
        name_map = _build_param_name_map(named)

        _, stats = split_param_groups_for_muon(
            groups, name_map, hidden_layer_patterns=["transformer_blocks", "nonexistent_layer"], verbose=False
        )
        assert "nonexistent_layer" in stats.unmatched_patterns
        assert "transformer_blocks" in stats.matched_patterns

    def test_skip_non_grad_params(self):
        named = [_make_param((64, 32), "transformer_blocks.0.weight", requires_grad=False)]
        groups = _make_single_group(named)
        name_map = _build_param_name_map(named)

        _, stats = split_param_groups_for_muon(groups, name_map, hidden_layer_patterns=["transformer_blocks"], verbose=False)
        assert stats.muon_count == 0
        assert stats.adam_count == 0

    def test_invalid_group_raises(self):
        with pytest.raises(TypeError, match="param-group dicts"):
            split_param_groups_for_muon(
                [torch.randn(10)],  # not a dict
                {},
                verbose=False,
            )


# ---------------------------------------------------------------------------
# print_muon_summary
# ---------------------------------------------------------------------------


class TestPrintMuonSummary:
    def test_uses_logger_not_print(self, caplog):
        stats = MuonParamStats(
            muon_count=5,
            adam_count=3,
            muon_params_total=1000,
            adam_params_total=500,
            matched_patterns=["transformer_blocks"],
            unmatched_patterns=[],
        )
        with caplog.at_level(logging.INFO, logger="musubi_tuner.optimizers.muon_util"):
            print_muon_summary(stats)
        assert "Muon Parameter Assignment Summary" in caplog.text
        assert "1,000" in caplog.text


# ---------------------------------------------------------------------------
# FallbackSingleDeviceMuonWithAuxAdam
# ---------------------------------------------------------------------------


class TestFallbackMuon:
    def _make_optimizer(self, muon_params, adam_params, muon_wd=0.0, adam_wd=0.0):
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True, "lr": 0.02, "momentum": 0.95, "weight_decay": muon_wd})
        if adam_params:
            groups.append(
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": 3e-4,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": adam_wd,
                }
            )
        return FallbackSingleDeviceMuonWithAuxAdam(groups)

    def test_basic_step_runs(self):
        p_muon = torch.nn.Parameter(torch.randn(16, 8))
        p_adam = torch.nn.Parameter(torch.randn(16))

        opt = self._make_optimizer([p_muon], [p_adam])
        # Simulate a forward/backward pass
        loss = p_muon.sum() + p_adam.sum()
        loss.backward()
        opt.step()  # should not raise

    def test_grad_none_with_weight_decay_still_decays_muon(self):
        """Muon params with grad=None but weight_decay>0 should still have weight decay applied."""
        p = torch.nn.Parameter(torch.ones(16, 8))
        original_norm = p.data.norm().item()

        opt = self._make_optimizer([p], [], muon_wd=0.1)
        # Do NOT set p.grad — it stays None
        opt.step()

        # Weight decay should have reduced the parameter norm
        assert p.data.norm().item() < original_norm

    def test_grad_none_with_weight_decay_still_decays_adam(self):
        """Adam params with grad=None but weight_decay>0 should still have weight decay applied."""
        p = torch.nn.Parameter(torch.ones(16))
        original_norm = p.data.norm().item()

        opt = self._make_optimizer([], [p], adam_wd=0.1)
        # Do NOT set p.grad — it stays None
        opt.step()

        # Weight decay should have reduced the parameter norm
        assert p.data.norm().item() < original_norm

    def test_grad_none_without_weight_decay_is_noop_for_muon(self):
        """With weight_decay=0 and grad=None, Muon params should not change (zero grad → orthogonal zero ≈ no-op)."""
        p = torch.nn.Parameter(torch.ones(16, 8))
        original = p.data.clone()

        opt = self._make_optimizer([p], [], muon_wd=0.0)
        opt.step()

        # With zero grad and zero weight decay, the update should be essentially zero
        # (Newton-Schulz on a zero matrix produces zero, so p shouldn't change meaningfully)
        # Allow small numerical noise from bfloat16 Newton-Schulz
        assert torch.allclose(p.data, original, atol=1e-5)

    def test_missing_use_muon_raises(self):
        with pytest.raises(ValueError, match="use_muon"):
            FallbackSingleDeviceMuonWithAuxAdam([{"params": [torch.nn.Parameter(torch.randn(4))]}])


# ---------------------------------------------------------------------------
# create_muon_optimizer
# ---------------------------------------------------------------------------


class TestCreateMuonOptimizer:
    def test_fallback_when_no_official(self):
        groups = [
            {"params": [torch.nn.Parameter(torch.randn(16, 8))], "use_muon": True, "lr": 0.02, "momentum": 0.95},
            {"params": [torch.nn.Parameter(torch.randn(16))], "use_muon": False, "lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8},
        ]
        opt = create_muon_optimizer(groups, prefer_official=False)
        assert isinstance(opt, FallbackSingleDeviceMuonWithAuxAdam)

    def test_param_groups_accessible(self):
        """Optimizer param_groups should be directly accessible (critical for block-swap patch)."""
        p = torch.nn.Parameter(torch.randn(16, 8))
        groups = [{"params": [p], "use_muon": True, "lr": 0.02, "momentum": 0.95}]
        opt = create_muon_optimizer(groups, prefer_official=False)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["params"][0] is p


# ---------------------------------------------------------------------------
# Block-swap optimizer patch integration
# ---------------------------------------------------------------------------


class TestBlockSwapPatch:
    """Integration-style tests for the block-swap grad-move logic from zimage_train.py."""

    @staticmethod
    def _block_swap_grad_move(optimizer):
        """Reproduce the block-swap patch loop from zimage_train.py:561-568."""
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None and param.device != param.grad.device:
                    param.grad = param.grad.to(param.device, non_blocking=True)

    def test_no_crash_on_plain_optimizer(self):
        """The patch should work on a plain (non-wrapped) optimizer via .param_groups."""
        p = torch.nn.Parameter(torch.randn(16, 8))
        p.grad = torch.randn_like(p)
        opt = create_muon_optimizer(
            [{"params": [p], "use_muon": True, "lr": 0.02, "momentum": 0.95}],
            prefer_official=False,
        )
        # Should not raise — this is the H1 fix validation
        self._block_swap_grad_move(opt)

    def test_grad_device_matches_param_after_patch(self):
        """After the patch, grad.device should match param.device (both CPU here)."""
        p = torch.nn.Parameter(torch.randn(16, 8))
        p.grad = torch.randn(16, 8)  # same device (CPU)
        opt = create_muon_optimizer(
            [{"params": [p], "use_muon": True, "lr": 0.02, "momentum": 0.95}],
            prefer_official=False,
        )
        self._block_swap_grad_move(opt)
        assert p.grad.device == p.device

    def test_none_grad_skipped(self):
        """Params with None grad should be silently skipped by the patch."""
        p = torch.nn.Parameter(torch.randn(16, 8))
        # p.grad is None
        opt = create_muon_optimizer(
            [{"params": [p], "use_muon": True, "lr": 0.02, "momentum": 0.95}],
            prefer_official=False,
        )
        self._block_swap_grad_move(opt)  # should not raise

    def test_works_through_proxy_wrapper(self):
        """Block-swap patch should work through any wrapper that proxies .param_groups.

        This mirrors AcceleratedOptimizer's behavior without pulling accelerate into tests.
        """
        p = torch.nn.Parameter(torch.randn(16, 8))
        p.grad = torch.randn_like(p)
        inner_opt = create_muon_optimizer(
            [{"params": [p], "use_muon": True, "lr": 0.02, "momentum": 0.95}],
            prefer_official=False,
        )

        class OptimizerProxy:
            """Minimal stand-in for AcceleratedOptimizer's param_groups proxy."""

            def __init__(self, optimizer):
                self._optimizer = optimizer

            @property
            def param_groups(self):
                return self._optimizer.param_groups

        wrapped = OptimizerProxy(inner_opt)
        self._block_swap_grad_move(wrapped)  # should not raise
        assert p.grad.device == p.device
