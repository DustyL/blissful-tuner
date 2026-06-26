"""GQA support in the unified attention() dispatch (added for Krea 2 = 48 query / 12 kv heads).

attention() must accept a query with more heads than key/value and produce results
identical to manually expanding k/v up to q's head count. This guards the
``repeat_interleave`` logic ported from upstream for Krea 2's grouped-query attention
on the SDPA (torch) and xformers paths. flash / sageattn / CuTE group heads natively in
their kernels, so they need no repeat — only the ``view``->``reshape`` non-contiguity fix.
That reshape path lives in the flash/sageattn/CuTE varlen branches (NOT the torch SDPA path
exercised here); it is validated separately by an ``--attn_mode flash`` generation run, since
unit-testing the varlen kernels requires a CUDA + cu_seqlens setup the kernels themselves gate.

The reference trick: run the SAME attention() twice — once with unequal heads (the GQA
branch fires and repeats internally) and once with k/v pre-expanded to equal heads (the
GQA branch is a no-op). Both must match, because ``repeat_interleave`` is exactly what the
internal expansion does.
"""

import pytest
import torch

from musubi_tuner.modules.attention import AttentionParams, attention

try:
    import xformers.ops as _xops  # noqa: F401

    _HAS_XFORMERS = True
except Exception:
    _HAS_XFORMERS = False


def _expanded_reference(q, k, v, attn_params):
    """Manually repeat_interleave k/v to q's head count (dim 2 = heads in [B, L, H, D]),
    then run attention() with now-equal heads so its enable_gqa branch is a no-op."""
    g = q.shape[2] // k.shape[2]
    k_exp = k.repeat_interleave(g, dim=2)
    v_exp = v.repeat_interleave(g, dim=2)
    return attention(q.clone(), k_exp.clone(), v_exp.clone(), attn_params)


@pytest.mark.parametrize("split", [False, True])
def test_gqa_torch_matches_expanded(split):
    """torch (SDPA) path: GQA output == head-expanded reference, split and non-split."""
    torch.manual_seed(0)
    B, L, Hq, Hkv, D = 2, 16, 8, 2, 32  # 4:1 grouping, mirrors K2's 48/12
    q = torch.randn(B, L, Hq, D, dtype=torch.float32)
    k = torch.randn(B, L, Hkv, D, dtype=torch.float32)
    v = torch.randn(B, L, Hkv, D, dtype=torch.float32)

    out_gqa = attention(q.clone(), k.clone(), v.clone(), AttentionParams.create_attention_params("torch", split))
    out_ref = _expanded_reference(q, k, v, AttentionParams.create_attention_params("torch", split))

    assert out_gqa.shape == (B, L, Hq * D)
    assert torch.isfinite(out_gqa).all()
    # repeat_interleave inside == repeat_interleave outside feeding identical SDPA inputs.
    assert torch.allclose(out_gqa, out_ref, atol=1e-5, rtol=1e-4)


def test_gqa_rejects_indivisible_heads():
    """Self-diagnosing guard: q heads not divisible by k/v heads must raise a clear error."""
    torch.manual_seed(3)
    B, L, D = 1, 8, 16
    q = torch.randn(B, L, 7, D)  # 7 not divisible by 2
    k = torch.randn(B, L, 2, D)
    v = torch.randn(B, L, 2, D)
    with pytest.raises(ValueError, match="divide the query head count"):
        attention(q, k, v)


def test_gqa_rejects_mismatched_kv_heads():
    """k and v must share a head count for the GQA expansion to be well-defined."""
    torch.manual_seed(4)
    B, L, D = 1, 8, 16
    q = torch.randn(B, L, 8, D)
    k = torch.randn(B, L, 2, D)
    v = torch.randn(B, L, 4, D)  # != k heads
    with pytest.raises(ValueError, match="equal key/value head counts"):
        attention(q, k, v)


def test_non_gqa_unchanged():
    """Equal head counts: enable_gqa is False, plain SDPA path must be unaffected."""
    torch.manual_seed(1)
    B, L, H, D = 2, 12, 4, 16
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, D)
    out = attention(q.clone(), k.clone(), v.clone())
    assert out.shape == (B, L, H * D)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not _HAS_XFORMERS, reason="xformers not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="xformers path needs CUDA")
def test_gqa_xformers_matches_expanded():
    """xformers path: GQA output == head-expanded reference (validates the dim=2 repeat)."""
    torch.manual_seed(2)
    B, L, Hq, Hkv, D = 1, 16, 8, 2, 64  # xformers needs head_dim multiple of 8
    dev = "cuda"
    q = torch.randn(B, L, Hq, D, dtype=torch.float16, device=dev)
    k = torch.randn(B, L, Hkv, D, dtype=torch.float16, device=dev)
    v = torch.randn(B, L, Hkv, D, dtype=torch.float16, device=dev)

    out_gqa = attention(q.clone(), k.clone(), v.clone(), AttentionParams.create_attention_params("xformers", False))
    out_ref = _expanded_reference(q, k, v, AttentionParams.create_attention_params("xformers", False))

    assert out_gqa.shape == (B, L, Hq * D)
    assert torch.allclose(out_gqa, out_ref, atol=1e-2, rtol=1e-2)
