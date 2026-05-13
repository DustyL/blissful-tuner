"""Parity test for the new per-tensor apply_rope vs the prior tuple form.

The optimization in perf(flux2): apply_rope per-tensor split with addcmul_
changed apply_rope's signature from `apply_rope(xq, xk, freqs_cis) -> (xq, xk)`
to `apply_rope(x, freqs_cis) -> x`. The split reduces peak memory because q
and k are processed sequentially instead of concurrently.

This test pins both the numerical equivalence and the autograd behavior:
calling the new per-tensor form on q and then k must produce outputs
within tight tolerance of what the old tuple form would have produced,
across both fp32 and bf16 dtypes, and the backward pass must populate
gradients on q and k.
"""

import unittest

import torch

from musubi_tuner.flux_2.flux2_models import apply_rope, rope


def _reference_apply_rope_tuple(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation matching the pre-optimization apply_rope.

    Kept inline rather than imported because the production code now ships
    only the per-tensor form — this is the bit we're comparing against.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def _build_inputs(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    xq = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    xk = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).expand(batch, -1)
    freqs_cis = rope(pos, head_dim, theta=10_000).unsqueeze(1)
    return xq, xk, freqs_cis


class ApplyRopeParity(unittest.TestCase):
    def test_fp32_forward_closeness(self):
        xq, xk, pe = _build_inputs(dtype=torch.float32, device=torch.device("cpu"))

        new_q = apply_rope(xq, pe)
        new_k = apply_rope(xk, pe)
        ref_q, ref_k = _reference_apply_rope_tuple(xq, xk, pe)

        self.assertTrue(
            torch.allclose(new_q, ref_q, rtol=1e-5, atol=1e-6),
            f"q parity drift (fp32): max abs diff {(new_q - ref_q).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(new_k, ref_k, rtol=1e-5, atol=1e-6),
            f"k parity drift (fp32): max abs diff {(new_k - ref_k).abs().max().item():.2e}",
        )

    def test_bf16_forward_closeness(self):
        """bf16 has narrower mantissa, so tolerance loosens accordingly.
        Both implementations cast to fp32 internally before computing —
        the bf16 difference is only at the input-cast and output-cast
        steps, so this is checking that the cast pattern hasn't shifted."""
        xq, xk, pe = _build_inputs(dtype=torch.bfloat16, device=torch.device("cpu"))

        new_q = apply_rope(xq, pe)
        new_k = apply_rope(xk, pe)
        ref_q, ref_k = _reference_apply_rope_tuple(xq, xk, pe)

        self.assertTrue(
            torch.allclose(new_q, ref_q, rtol=1e-2, atol=1e-3),
            f"q parity drift (bf16): max abs diff {(new_q - ref_q).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(new_k, ref_k, rtol=1e-2, atol=1e-3),
            f"k parity drift (bf16): max abs diff {(new_k - ref_k).abs().max().item():.2e}",
        )

    def test_backward_reaches_inputs(self):
        """The addcmul_ rewrite mutates x_out in-place — verify autograd
        still flows back to the inputs."""
        xq, xk, pe = _build_inputs(dtype=torch.float32, device=torch.device("cpu"))
        xq.requires_grad_(True)
        xk.requires_grad_(True)

        out_q = apply_rope(xq, pe)
        out_k = apply_rope(xk, pe)
        loss = out_q.sum() + out_k.sum()
        loss.backward()

        self.assertIsNotNone(xq.grad)
        self.assertIsNotNone(xk.grad)
        self.assertGreater(xq.grad.abs().sum().item(), 0.0)
        self.assertGreater(xk.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
