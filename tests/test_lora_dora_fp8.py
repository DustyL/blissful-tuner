# Tests for DoRA on fp8-prequantized base weights (backlog P0-2).
#
# Two distinct failure modes existed before the fix (verified empirically 2026-06-11):
#  1. CRASH: dora_weight_norm_materialized's `weight + scaling * lora_weight` raises
#     "Promotion for Float8 Types is not supported" — hit at DoRA init (_init_dora) on any
#     fp8-patched Linear (e.g. the Ideogram 4 pre-quantized shim).
#  2. SILENT WRONGNESS: get_weight_norm_efficient's `W.float()` does not crash, but a raw fp8
#     cast without the scale_weight multiply yields quantization-lattice magnitudes (~200x off
#     for real checkpoints) — wrong row norms, wrong DoRA magnitude scaling. The fp8 norm it
#     returned then ALSO crashed the downstream `magnitude / weight_norm` division.
#
# The e85284b LoRA dtype harmonization fixed only LoRAModule.forward's delta path, NOT these
# raw-weight reads — that question (backlog P0-2 step 1) is settled by these tests existing.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.networks.dora_utils import dequantize_fp8_weight, dora_weight_norm_materialized
from musubi_tuner.networks.lora import DoRALayer, LoRAInfModule, LoRAModule, LoRANetwork

E4M3_MAX = 448.0


def _quantize_per_tensor(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = w.abs().max().float() / E4M3_MAX
    return (w / scale).to(torch.float8_e4m3fn), scale.reshape(1)


def _quantize_per_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (w.abs().amax(dim=1, keepdim=True).float() / E4M3_MAX).clamp_min(1e-12)
    return (w / scale).to(torch.float8_e4m3fn), scale  # scale: (out, 1)


def _quantize_blockwise(w: torch.Tensor, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = w.shape
    blocks = w.reshape(out_features, num_blocks, in_features // num_blocks)
    scale = (blocks.abs().amax(dim=2, keepdim=True).float() / E4M3_MAX).clamp_min(1e-12)
    q = (blocks / scale).to(torch.float8_e4m3fn).reshape(out_features, in_features)
    return q, scale  # scale: (out, num_blocks, 1)


def _make_fp8_linear(out_dim=12, in_dim=8, layout="per_tensor", seed=0) -> tuple[nn.Linear, torch.Tensor]:
    """Synthetic fp8-patched Linear mirroring apply_fp8_monkey_patch: raw fp8 .weight + scale_weight
    buffer + a dequantizing forward (the monkey-patch installs the fp8-aware forward before LoRA wraps it)."""
    torch.manual_seed(seed)
    lin = nn.Linear(in_dim, out_dim, bias=False)
    true_w = lin.weight.data.clone()
    if layout == "per_tensor":
        q, scale = _quantize_per_tensor(true_w)
    elif layout == "per_row":
        q, scale = _quantize_per_row(true_w)
    else:
        q, scale = _quantize_blockwise(true_w, num_blocks=2)
    lin.weight = nn.Parameter(q, requires_grad=False)
    lin.register_buffer("scale_weight", scale)

    # Bind like apply_fp8_monkey_patch does (fp8_optimization_utils.py:487, new_forward.__get__):
    # LoRAModule._get_base_layer reads org_forward.__self__, so a plain lambda would break it.
    def _fp8_forward(self, x):
        return F.linear(x, dequantize_fp8_weight(self.weight, self.scale_weight, x.dtype))

    lin.forward = _fp8_forward.__get__(lin, type(lin))
    return lin, true_w


# --- dequantize_fp8_weight ------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["per_tensor", "per_row", "blockwise"])
def test_dequantize_recovers_true_weights(layout):
    lin, true_w = _make_fp8_linear(layout=layout)
    deq = dequantize_fp8_weight(lin.weight, lin.scale_weight)
    assert deq.dtype == torch.float32
    # e4m3 has ~3 mantissa bits; per-tensor scaling of a randn matrix stays within ~10% relative error.
    assert torch.allclose(deq, true_w, rtol=0.1, atol=0.02)


def test_dequantize_passthrough_for_non_fp8():
    w = torch.randn(4, 4, dtype=torch.bfloat16)
    assert dequantize_fp8_weight(w, None) is w


@pytest.mark.parametrize("name", ["float8_e4m3fnuz", "float8_e5m2fnuz"])
def test_fnuz_dtypes_detected_as_fp8(name):
    # The repo recognizes all four fp8 variants elsewhere (flux_utils.is_fp8); the DoRA path must too,
    # or an fnuz-prequantized base would skip dequant AND slip past the destructive-merge guards.
    from musubi_tuner.networks.dora_utils import FP8_DTYPES

    dt = getattr(torch, name, None)
    if dt is None:
        pytest.skip(f"{name} not available in this torch build")
    assert dt in FP8_DTYPES
    w = torch.randn(4, 4).to(dt)
    with pytest.raises(ValueError, match="scale_weight"):
        dequantize_fp8_weight(w, None)
    deq = dequantize_fp8_weight(w, torch.tensor([2.0]))
    assert deq.dtype == torch.float32
    assert torch.allclose(deq, w.float() * 2.0)


def test_dequantize_rejects_raw_per_row_scale_on_square_matrix():
    # A raw [out] vector broadcasts over the LAST axis: on a square weight it silently scales
    # COLUMNS instead of rows. Must refuse, not mis-scale.
    lin, _ = _make_fp8_linear(out_dim=8, in_dim=8, layout="per_row")
    bad_scale = lin.scale_weight.reshape(-1)  # (8, 1) -> raw (8,)
    with pytest.raises(ValueError, match="wrong axis"):
        dequantize_fp8_weight(lin.weight, bad_scale)
    # The properly shaped (out, 1) scale on the same weight works.
    assert dequantize_fp8_weight(lin.weight, lin.scale_weight).shape == (8, 8)


def test_dequantize_rejects_blockwise_scale_with_indivisible_blocks():
    lin, _ = _make_fp8_linear(out_dim=12, in_dim=8, layout="per_tensor")
    bad_scale = torch.rand(12, 3, 1)  # in_features=8 not divisible by num_blocks=3
    with pytest.raises(ValueError, match="unsupported scale_weight shape"):
        dequantize_fp8_weight(lin.weight, bad_scale)


def test_dequantize_fp8_without_scale_raises():
    lin, _ = _make_fp8_linear()
    with pytest.raises(ValueError, match="scale_weight"):
        dequantize_fp8_weight(lin.weight, None)


# --- weight-norm paths ----------------------------------------------------------------------------


def test_materialized_norm_fp8_matches_bf16_reference():
    lin, true_w = _make_fp8_linear()
    lora_weight = torch.randn(12, 8)
    ref = dora_weight_norm_materialized(true_w, lora_weight, 0.5)
    out = dora_weight_norm_materialized(lin.weight, lora_weight, 0.5, scale_weight=lin.scale_weight)
    assert out.dtype == lora_weight.dtype  # never fp8 — downstream magnitude/norm division would crash
    assert torch.allclose(out, ref, rtol=0.1, atol=0.02)


def test_materialized_norm_fp8_without_scale_raises():
    lin, _ = _make_fp8_linear()
    with pytest.raises(ValueError, match="scale_weight"):
        dora_weight_norm_materialized(lin.weight, torch.randn(12, 8), 0.5)


def test_efficient_norm_fp8_matches_materialized():
    lin, true_w = _make_fp8_linear()
    dl = DoRALayer(12)
    A, B = torch.randn(4, 8), torch.randn(12, 4)
    eff = dl.get_weight_norm_efficient(lin.weight, A, B, 0.5, scale_weight=lin.scale_weight)
    ref = dl.get_weight_norm_materialized(true_w, B @ A, 0.5)
    assert eff.dtype != torch.float8_e4m3fn
    assert torch.allclose(eff, ref.float(), rtol=0.1, atol=0.02)
    # The pre-fix silent failure: raw-fp8 norms were quantization-lattice values, far from truth.
    assert not torch.allclose(lin.weight.float().norm(dim=1), ref.float(), rtol=0.5)


def test_efficient_norm_non_fp8_path_unchanged():
    # Regression guard: bf16/fp32 weights keep the original behavior (norm in W.dtype, no scale needed).
    dl = DoRALayer(12)
    W, A, B = torch.randn(12, 8, dtype=torch.bfloat16), torch.randn(4, 8), torch.randn(12, 4)
    out = dl.get_weight_norm_efficient(W, A, B, 0.5)
    assert out.dtype == torch.bfloat16


# --- end-to-end: LoRAModule with use_dora=True on an fp8-patched Linear ----------------------------


def test_dora_init_on_fp8_linear_matches_bf16_reference():
    lin, true_w = _make_fp8_linear()
    torch.manual_seed(1)
    m = LoRAModule("lora_unet_test_fp8", lin, 1.0, lora_dim=4, alpha=4, use_dora=True)
    m.apply_to()  # crashed with Float8-promotion before the fix

    ref_lin = nn.Linear(8, 12, bias=False)
    ref_lin.weight.data = true_w.clone()
    torch.manual_seed(1)
    ref = LoRAModule("lora_unet_test_ref", ref_lin, 1.0, lora_dim=4, alpha=4, use_dora=True)
    ref.apply_to()

    assert torch.allclose(m.dora_layer.weight, ref.dora_layer.weight, rtol=0.1, atol=0.02)


def test_dora_forward_backward_on_fp8_linear():
    lin, true_w = _make_fp8_linear()
    torch.manual_seed(1)
    m = LoRAModule("lora_unet_test_fp8", lin, 1.0, lora_dim=4, alpha=4, use_dora=True)
    m.apply_to()

    x = torch.randn(3, 8)
    out = lin.forward(x)  # monkey-patched to LoRAModule.forward by apply_to
    assert out.shape == (3, 12)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert m.lora_down.weight.grad is not None
    assert m.lora_up.weight.grad is not None
    assert torch.isfinite(m.lora_down.weight.grad).all()

    # Output parity with a bf16-equivalent reference DoRA module (same LoRA init via same seed).
    ref_lin = nn.Linear(8, 12, bias=False)
    ref_lin.weight.data = true_w.clone()
    torch.manual_seed(1)
    ref = LoRAModule("lora_unet_test_ref", ref_lin, 1.0, lora_dim=4, alpha=4, use_dora=True)
    ref.apply_to()
    assert torch.allclose(out, ref_lin.forward(x), rtol=0.15, atol=0.05)


# --- destructive merge paths refuse fp8 bases ------------------------------------------------------
# Merging/baking into an fp8 checkpoint would write float-merged values back as fp8 without inverse
# re-scaling — silent corruption. Both destructive paths must refuse loudly; runtime forward is the
# supported fp8 route.


def test_merge_to_refuses_fp8_base():
    lin, _ = _make_fp8_linear()
    m = LoRAInfModule("lora_unet_test_fp8", lin, 1.0, lora_dim=4, alpha=4)
    weights = {
        "lora_down.weight": torch.randn(4, 8),
        "lora_up.weight": torch.randn(12, 4),
    }
    with pytest.raises(ValueError, match="fp8-prequantized"):
        m.merge_to(weights, None, None)


def test_pre_calculation_refuses_fp8_base():
    lin, _ = _make_fp8_linear()
    inf = LoRAInfModule("lora_unet_test_fp8", lin, 1.0, lora_dim=4, alpha=4)
    inf.org_module_ref = [lin]
    fake_network = SimpleNamespace(text_encoder_loras=[], unet_loras=[inf])
    with pytest.raises(ValueError, match="fp8-prequantized"):
        LoRANetwork.pre_calculation(fake_network)
