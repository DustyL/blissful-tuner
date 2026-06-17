"""Tests for H2D-only block swap (LoRAStreamOffloader) and the BlockSwapConfig factory.

Ported from upstream musubi-tuner commit 9bfc5a4 ("Feat h2d only block swap").
The new offloader streams *frozen* base weights Host->Device only (no D2H), for
LoRA / LoHa / LoKr / DoRA training where the CPU copy never diverges from GPU.

Two layers are covered:
  * Env-independent: the BlockSwapConfig dataclass + from_args validation guards
    (gradient_checkpointing requirement, ring_size, inference exemption). These
    import only torch, never transformers, so they run even when the daily-build
    transformers/tokenizers pins are mid-churn.
  * CUDA-gated: a functional forward/backward *parity* smoke that drives the
    offloader exactly the way a model's forward loop does (prepare -> per-block
    wait/forward/submit -> backward), then re-prepares and re-runs. The re-prepare
    leg is the unit-scale analogue of blissful's prior-teacher no_grad restore
    (CLAUDE.md "Block Swap Invariants") for the H2D path.
"""

import argparse

import pytest
import torch
import torch.nn as nn

from musubi_tuner.modules.custom_offloading_utils import (
    BlockSwapConfig,
    LoRAStreamOffloader,
    ModelOffloader,
    create_offloader,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="LoRAStreamOffloader requires CUDA")


# --------------------------------------------------------------------------- env-independent


def _ns(**kw):
    return argparse.Namespace(**kw)


def test_config_defaults():
    cfg = BlockSwapConfig(device=torch.device("cpu"), supports_backward=True)
    assert cfg.use_pinned_memory is False
    assert cfg.h2d_only is False
    assert cfg.ring_size == 2
    assert cfg.debug is False


def test_from_args_h2d_training_requires_gradient_checkpointing():
    # H2D-only's in-place ring copy_ bumps the autograd version of weights saved for
    # backward; gradient checkpointing re-reads them at recompute time. Without it the
    # backward fails with an opaque inplace-version error, so from_args fails early.
    ns = _ns(block_swap_h2d_only=True, use_pinned_memory_for_block_swap=False, gradient_checkpointing=False)
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        BlockSwapConfig.from_args(ns, torch.device("cpu"), supports_backward=True)


def test_from_args_h2d_training_ok_with_gradient_checkpointing():
    ns = _ns(
        block_swap_h2d_only=True,
        use_pinned_memory_for_block_swap=True,
        gradient_checkpointing=True,
        block_swap_ring_size=3,
    )
    cfg = BlockSwapConfig.from_args(ns, torch.device("cpu"), supports_backward=True)
    assert cfg.h2d_only is True
    assert cfg.ring_size == 3
    assert cfg.use_pinned_memory is True


def test_from_args_inference_exempt_from_gradient_checkpointing():
    # Forward-only (inference) has no backward pass, so the version-bump constraint
    # does not apply and the gradient_checkpointing requirement must NOT fire.
    ns = _ns(block_swap_h2d_only=True, gradient_checkpointing=False)
    cfg = BlockSwapConfig.from_args(ns, torch.device("cpu"), supports_backward=False)
    assert cfg.h2d_only is True
    assert cfg.supports_backward is False


def test_from_args_ring_size_must_be_positive():
    ns = _ns(block_swap_h2d_only=True, gradient_checkpointing=True, block_swap_ring_size=0)
    with pytest.raises(ValueError, match="ring_size"):
        BlockSwapConfig.from_args(ns, torch.device("cpu"), supports_backward=True)


def test_from_args_tolerates_missing_optional_attrs():
    # Inference scripts whose parser lacks the new knobs must still build a config.
    cfg = BlockSwapConfig.from_args(_ns(), torch.device("cpu"), supports_backward=False)
    assert cfg.h2d_only is False
    assert cfg.ring_size == 2


# --------------------------------------------------------------------------- factory routing


class _Blk(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return torch.relu(self.lin(x))


def _frozen_blocks(n, d):
    blocks = nn.ModuleList(_Blk(d) for _ in range(n))
    blocks.requires_grad_(False)
    return blocks


@cuda_only
def test_create_offloader_routes_to_lora_stream_for_h2d():
    dev = torch.device("cuda")
    blocks = _frozen_blocks(6, 16).to(dev)
    cfg = BlockSwapConfig(device=dev, supports_backward=True, h2d_only=True, ring_size=2)
    off = create_offloader("test", list(blocks), 6, 3, cfg)
    assert isinstance(off, LoRAStreamOffloader)


@cuda_only
def test_create_offloader_routes_to_model_offloader_for_classic():
    dev = torch.device("cuda")
    blocks = _frozen_blocks(6, 16).to(dev)
    cfg = BlockSwapConfig(device=dev, supports_backward=True, h2d_only=False)
    off = create_offloader("test", list(blocks), 6, 3, cfg)
    assert isinstance(off, ModelOffloader)


@cuda_only
def test_h2d_rejects_trainable_base_weights():
    # The finetune guard: routing a full-FT (trainable base) model through H2D-only
    # must raise, because the H2D-only design can never write weight updates back.
    dev = torch.device("cuda")
    blocks = nn.ModuleList(_Blk(16) for _ in range(6)).to(dev)  # NOT frozen
    cfg = BlockSwapConfig(device=dev, supports_backward=True, h2d_only=True, ring_size=2)
    with pytest.raises(AssertionError, match="frozen base weights"):
        create_offloader("test", list(blocks), 6, 3, cfg)


@cuda_only
def test_h2d_accepts_blocks_after_freezing():
    # P2-1 mechanism: the same trainable blocks become acceptable once frozen. This is what the
    # WAN high-noise expert setup (wan_train_network.load_transformer) now does -- eval() +
    # requires_grad_(False) before enable_block_swap -- so dual-expert WAN + --block_swap_h2d_only
    # no longer trips the frozen-base guard at setup.
    dev = torch.device("cuda")
    blocks = nn.ModuleList(_Blk(16) for _ in range(6)).to(dev)  # trainable base
    cfg = BlockSwapConfig(device=dev, supports_backward=True, h2d_only=True, ring_size=2)
    with pytest.raises(AssertionError, match="frozen base weights"):
        create_offloader("test", list(blocks), 6, 3, cfg)
    blocks.requires_grad_(False)  # the freeze the WAN high-noise / main transformer path performs
    off = create_offloader("test", list(blocks), 6, 3, cfg)
    assert isinstance(off, LoRAStreamOffloader)


# --------------------------------------------------------------------------- functional parity


def _run_swap_forward(blocks, off, x, checkpointed=False):
    """Drive the offloader exactly like a model forward loop (flux2_models.py pattern).

    When ``checkpointed`` is set the per-block forward runs under non-reentrant
    gradient checkpointing, mirroring how every model dispatches its blocks when
    ``--gradient_checkpointing`` is on. This is mandatory for the H2D backward path:
    the ring's in-place ``copy_`` bumps the saved weight's autograd version, and only
    recompute-at-backward keeps the version consistent (see the guard test below).
    """
    h = x
    for i, blk in enumerate(blocks):
        off.wait_for_block(i)
        if checkpointed:
            h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
        else:
            h = blk(h)
        off.submit_move_blocks_forward(list(blocks), i)
    return h


@cuda_only
@pytest.mark.parametrize("use_pinned", [True, False])
def test_h2d_forward_backward_parity_and_reprepare(use_pinned):
    dev = torch.device("cuda")
    n, d = 6, 32
    torch.manual_seed(0)
    blocks = _frozen_blocks(n, d).to(dev)
    x = torch.randn(2, 8, d, device=dev)

    # ---- reference: plain forward/backward on GPU, no swap ----
    xr = x.clone().requires_grad_(True)
    h = xr
    for blk in blocks:
        h = blk(h)
    ref_out = h.detach().clone()
    h.sum().backward()
    ref_grad = xr.grad.clone()

    # ---- H2D-only swap: stream indices {1,3,5}, resident {0,2,4}, ring=2 ----
    blocks.cpu()  # prepare() places the resident blocks and pulls swap weights to CPU masters
    cfg = BlockSwapConfig(device=dev, supports_backward=True, use_pinned_memory=use_pinned, h2d_only=True, ring_size=2)
    off = create_offloader("parity", list(blocks), n, 3, cfg)
    assert off.S == 3 and off.stream_idx == [1, 3, 5]

    off.prepare_block_devices_before_forward(list(blocks))
    xs = x.clone().requires_grad_(True)
    swap_out = _run_swap_forward(blocks, off, xs, checkpointed=True)  # mirrors --gradient_checkpointing
    torch.testing.assert_close(swap_out, ref_out, rtol=0, atol=1e-5)

    swap_out.sum().backward()  # exercises the backward hooks (prefetch-behind / wait-prev)
    assert xs.grad is not None
    torch.testing.assert_close(xs.grad, ref_grad, rtol=0, atol=1e-5)

    # ---- re-prepare then forward again (cheap restore round-trip) ----
    off.prepare_block_devices_before_forward(list(blocks))
    with torch.no_grad():
        again = _run_swap_forward(blocks, off, x.clone())
    torch.testing.assert_close(again, ref_out, rtol=0, atol=1e-5)


@cuda_only
def test_h2d_prior_teacher_nograd_then_student_grad():
    """The documented crash class (CLAUDE.md "Block Swap Invariants"), in the exact order
    blissful's prior-preservation path uses it: a no_grad *teacher* forward advances the ring
    with NO backward hooks to unwind it, then restore_block_swap_after_no_grad_forward
    (re-prepare), then a *student* grad forward+backward must still succeed. This ordering
    (no_grad-first, then grad) is the one that has crashed this fork before with "mat2 is on
    cpu" / autograd-version errors -- distinct from a normal fwd+bwd step.
    """
    dev = torch.device("cuda")
    n, d = 6, 32
    torch.manual_seed(0)
    blocks = _frozen_blocks(n, d).to(dev)
    x = torch.randn(2, 8, d, device=dev)

    # reference grad (no swap)
    xr = x.clone().requires_grad_(True)
    h = xr
    for blk in blocks:
        h = blk(h)
    ref_out = h.detach().clone()
    h.sum().backward()
    ref_grad = xr.grad.clone()

    blocks.cpu()
    cfg = BlockSwapConfig(device=dev, supports_backward=True, h2d_only=True, ring_size=2)
    off = create_offloader("teacher", list(blocks), n, 3, cfg)
    off.prepare_block_devices_before_forward(list(blocks))

    # teacher forward UNDER no_grad: advances ring/slot state, NO backward hooks fire
    with torch.no_grad():
        _ = _run_swap_forward(blocks, off, x.clone())

    # restore (prepare_block_swap_before_forward) then student grad fwd+bwd
    off.prepare_block_devices_before_forward(list(blocks))
    xs = x.clone().requires_grad_(True)
    student = _run_swap_forward(blocks, off, xs, checkpointed=True)
    torch.testing.assert_close(student, ref_out, rtol=0, atol=1e-5)
    student.sum().backward()
    assert xs.grad is not None
    torch.testing.assert_close(xs.grad, ref_grad, rtol=0, atol=1e-5)


@cuda_only
def test_h2d_backward_without_gradient_checkpointing_raises_version_error():
    # Documents *why* from_args hard-requires gradient_checkpointing: without recompute,
    # the ring's in-place copy_ advances the saved weight's autograd version and backward
    # fails. This is the opaque error the from_args guard converts into an actionable message.
    dev = torch.device("cuda")
    n, d = 6, 32
    torch.manual_seed(0)
    blocks = _frozen_blocks(n, d).to(dev).cpu()
    cfg = BlockSwapConfig(device=dev, supports_backward=True, h2d_only=True, ring_size=2)
    off = create_offloader("nockpt", list(blocks), n, 3, cfg)
    off.prepare_block_devices_before_forward(list(blocks))
    xs = torch.randn(2, 8, d, device=dev, requires_grad=True)
    out = _run_swap_forward(blocks, off, xs, checkpointed=False)
    with pytest.raises(RuntimeError, match="inplace operation"):
        out.sum().backward()
