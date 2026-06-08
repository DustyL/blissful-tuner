"""Regression test for fp8 monkey-patched Linear forward being autocast-safe.

Background: fp8 Linear's patched forward (fp8_optimization_utils.fp8_linear_forward_patch,
non-scaled_mm dequant branch) was discovered 2026-06-08 to produce broken output (flat
saturated activations) when called inside a torch.autocast(bf16) context. The bug was
isolated via Ideogram 4 step-0 sample bisect at /home/dustin/output/ideogram4_parity_gate/:
- Coherent portrait with autocast off
- Flat gray with autocast(bf16) on
- LoRA wrap was irrelevant; autocast alone was sufficient and necessary.

The fix wraps the dequant branch in `torch.autocast(device_type=x.device.type, enabled=False)`
and casts all operands explicitly to x.dtype. This test validates the MATH contract that
fix must preserve, not just that the autocast context is disabled.

Contract: for the same inputs, the patched forward must produce the SAME output whether or
not the enclosing context is `torch.autocast(bf16)`. Both must match an explicit-dequant
reference: `F.linear(x, weight.to(x.dtype) * scale.to(x.dtype), bias)`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.modules.fp8_optimization_utils import fp8_linear_forward_patch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fp8 dequant autocast bug only reproduces on CUDA (autocast(cuda) is what installs the corruption).",
)


def _make_fp8_linear(in_dim: int, out_dim: int, scale_shape: tuple, *, with_bias: bool, device, dtype):
    """Build a synthetic fp8 nn.Linear with the same shape/buffer layout as apply_fp8_monkey_patch.

    Mimics: fp8_e4m3fn weight + scale_weight buffer + bound forward, exactly as
    apply_fp8_monkey_patch produces. Scale is non-unit (random in [0.5, 2.0]) so a silently
    bypassed dequant would produce visibly wrong output instead of accidentally being correct.
    """
    module = nn.Linear(in_dim, out_dim, bias=with_bias).to(device=device, dtype=dtype)

    # Build a random reference matrix at the load dtype, quantize to fp8.
    # Real fp8 monkey-patch loads pre-quantized weights from disk; here we synthesize via
    # to(float8_e4m3fn) so the test exercises the same dequant path.
    ref_weight = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * 0.1
    fp8_weight = ref_weight.to(torch.float8_e4m3fn)
    with torch.no_grad():
        module.weight = nn.Parameter(fp8_weight, requires_grad=False)

    # Non-unit scale so the dequant math actually does something detectable.
    scale = torch.rand(scale_shape, device=device, dtype=dtype) * 1.5 + 0.5
    module.register_buffer("scale_weight", scale)

    if with_bias:
        with torch.no_grad():
            module.bias.copy_(torch.randn_like(module.bias) * 0.05)

    # Bind the patched forward, matching apply_fp8_monkey_patch's binding pattern.
    def new_forward(self, x):
        return fp8_linear_forward_patch(self, x, use_scaled_mm=False, max_value=None)

    module.forward = new_forward.__get__(module, type(module))
    return module


def _reference_output(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """The math the patched forward MUST match, expressed without going through the patch."""
    scale = module.scale_weight.to(x.dtype)
    if scale.ndim < 3:
        dequant = module.weight.to(x.dtype) * scale
    else:
        out_features, num_blocks, _ = scale.shape
        dequant = module.weight.to(x.dtype).contiguous().view(out_features, num_blocks, -1)
        dequant = dequant * scale
        dequant = dequant.view(module.weight.shape)
    bias = module.bias.to(x.dtype) if module.bias is not None else None
    return F.linear(x, dequant, bias)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize(
    "scale_shape_factory,layout_name",
    [
        (lambda out_dim: (1,), "per_tensor"),  # Comfy fp8 layout (scalar -> [1] after _reshape_scale_weight)
        (lambda out_dim: (out_dim, 1), "per_row"),  # HF fp8 layout (out_dim 1-D -> [out_dim, 1])
    ],
)
def test_fp8_forward_matches_reference_no_autocast(dtype, with_bias, scale_shape_factory, layout_name):
    """Baseline: with no enclosing autocast, the patched forward must equal the explicit reference."""
    device = torch.device("cuda")
    in_dim, out_dim = 64, 128
    module = _make_fp8_linear(in_dim, out_dim, scale_shape_factory(out_dim), with_bias=with_bias, device=device, dtype=dtype)

    x = torch.randn(2, 16, in_dim, device=device, dtype=dtype) * 0.5

    actual = module(x)
    expected = _reference_output(module, x)

    assert actual.shape == expected.shape, f"{layout_name} {dtype} bias={with_bias}"
    assert actual.dtype == dtype, f"{layout_name} {dtype} bias={with_bias}: expected output dtype {dtype}, got {actual.dtype}"
    # bf16 tolerance is loose because dequant + matmul accumulates in bf16
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize(
    "scale_shape_factory,layout_name",
    [
        (lambda out_dim: (1,), "per_tensor"),
        (lambda out_dim: (out_dim, 1), "per_row"),
    ],
)
def test_fp8_forward_matches_reference_under_autocast(dtype, with_bias, scale_shape_factory, layout_name):
    """The bug-fix contract: the patched forward MUST produce the same output whether or not
    an enclosing torch.autocast(bf16) is active. Prior to the fix, this test would fail with
    flat/saturated output under autocast even though the no-autocast case worked correctly."""
    device = torch.device("cuda")
    in_dim, out_dim = 64, 128
    module = _make_fp8_linear(in_dim, out_dim, scale_shape_factory(out_dim), with_bias=with_bias, device=device, dtype=dtype)

    x = torch.randn(2, 16, in_dim, device=device, dtype=dtype) * 0.5

    with torch.autocast(device_type="cuda", dtype=dtype):
        actual_under_autocast = module(x)

    expected = _reference_output(module, x)

    assert actual_under_autocast.shape == expected.shape
    assert actual_under_autocast.dtype == dtype
    # Same tolerance as no-autocast - the fix's whole point is that the math is identical
    torch.testing.assert_close(actual_under_autocast, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    "scale_shape_factory,layout_name",
    [
        (lambda out_dim: (1,), "per_tensor"),
        (lambda out_dim: (out_dim, 1), "per_row"),
    ],
)
def test_fp8_forward_autocast_matches_no_autocast(scale_shape_factory, layout_name):
    """Cross-check: the autocast-on and autocast-off outputs must be elementwise close.
    Different from the above tests in that it doesn't go through the explicit reference path
    — it asserts the autocast invariance of the patched forward directly."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    in_dim, out_dim = 64, 128
    module = _make_fp8_linear(in_dim, out_dim, scale_shape_factory(out_dim), with_bias=True, device=device, dtype=dtype)

    x = torch.randn(2, 16, in_dim, device=device, dtype=dtype) * 0.5

    out_no_autocast = module(x)
    with torch.autocast(device_type="cuda", dtype=dtype):
        out_with_autocast = module(x)

    # These must be NUMERICALLY identical, not just close — both compute the same dequant
    # under autocast(enabled=False) regardless of the enclosing context.
    torch.testing.assert_close(out_no_autocast, out_with_autocast, rtol=0, atol=0)


def test_fp8_forward_under_autocast_is_not_flat():
    """Sanity guard: assert the output isn't a flat constant. This is the specific failure
    mode we hit pre-fix — bf16 autocast produced essentially constant outputs (collapsed
    to the safety-placeholder gray downstream). If the variance of the output across a
    diverse batch is near-zero, that's exactly the regression we're guarding against."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    in_dim, out_dim = 64, 128
    module = _make_fp8_linear(in_dim, out_dim, (out_dim, 1), with_bias=True, device=device, dtype=dtype)

    # 8 distinct batches with different scales to ensure variance
    x = torch.randn(8, 16, in_dim, device=device, dtype=dtype)
    x = x * torch.linspace(0.3, 1.5, 8, device=device, dtype=dtype).view(8, 1, 1)

    with torch.autocast(device_type="cuda", dtype=dtype):
        out = module(x)

    # If autocast were silently zeroing the fp8 dequant, all rows would be the bias term.
    # std across the batch dimension should be visibly non-zero.
    per_batch_std = out.std(dim=(1, 2))
    assert (per_batch_std > 1e-3).all(), (
        f"Output is near-flat across batch (std={per_batch_std.tolist()}). "
        "This is the regression we're guarding against: autocast(bf16) silently collapsing fp8 forward."
    )
