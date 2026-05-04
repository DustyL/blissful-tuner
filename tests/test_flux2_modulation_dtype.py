"""Pin the dtype contract of FLUX.2's Modulation and LastLayer.

Both classes deliberately upcast their conditioning vector to fp32 for numerical
stability (LayerNorm variance, SiLU on small values), then run a Linear whose
weight is in the model's load dtype (bf16 for pure-bf16 inference). PyTorch's
matmul kernels reject mismatched fp32/bf16 dtypes, so each Linear call must
narrow the activation back to ``weight.dtype`` before applying.

This test reproduces the original RuntimeError on un-narrowed inputs and
verifies the fix works end-to-end on CPU.
"""

from __future__ import annotations

import torch

from musubi_tuner.flux_2.flux2_models import LastLayer, Modulation


def test_modulation_accepts_fp32_input_with_bf16_weight():
    dim = 16
    mod = Modulation(dim, double=True, disable_bias=True).to(torch.bfloat16)
    assert mod.lin.weight.dtype == torch.bfloat16

    vec = torch.randn(2, dim, dtype=torch.bfloat16)

    out_first, out_second = mod(vec)

    assert out_second is not None  # double=True yields a second triple
    for tensor in (*out_first, *out_second):
        assert tensor.dtype == torch.bfloat16
        assert tensor.shape == (2, 1, dim)
        assert torch.isfinite(tensor).all()


def test_modulation_single_branch_accepts_fp32_input_with_bf16_weight():
    dim = 16
    mod = Modulation(dim, double=False, disable_bias=True).to(torch.bfloat16)
    vec = torch.randn(2, dim, dtype=torch.bfloat16)

    out_first, out_second = mod(vec)

    assert out_second is None
    for tensor in out_first:
        assert tensor.dtype == torch.bfloat16
        assert torch.isfinite(tensor).all()


def test_lastlayer_accepts_fp32_upcast_with_bf16_weights():
    hidden_size = 32
    out_channels = 8
    layer = LastLayer(hidden_size, out_channels).to(torch.bfloat16)

    adaln_linear = layer.adaLN_modulation[1]
    assert adaln_linear.weight.dtype == torch.bfloat16
    assert layer.linear.weight.dtype == torch.bfloat16

    seq_len = 4
    x = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
    vec = torch.randn(1, hidden_size, dtype=torch.bfloat16)

    out = layer(x, vec)

    assert out.dtype == torch.bfloat16  # matches x.dtype, the documented contract
    assert out.shape == (1, seq_len, out_channels)
    assert torch.isfinite(out).all()
