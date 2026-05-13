"""Regression test for Conv2d LoRA forward and backward.

Guards against the class of bug seen in xzuyn-optimizations' lora.py rewrite,
which replaced `org_forwarded + lora_up(lx) * scale` with
`org_forwarded_2d.addmm_(lx_2d, self.lora_up.weight.to(lx_2d.dtype).t(), ...)`.
That formulation calls `.t()` on the lora_up weight — which for Conv2d is 4D
(out_channels, in_channels, kH, kW) — raising `RuntimeError: t() expects a
tensor with <= 2 dimensions`.

This test exercises:
  - Conv2d LoRA forward with the canonical kernel_size=3, padding=1 shape
  - The rank_dropout `lx.dim() == 4` unsqueeze branch (Conv2d lx is 4D)
  - Backward through Conv2d LoRA producing non-zero adapter gradients

The Conv2d LoRA path is shared infrastructure — blissful-tuner's FLUX.2
adapters are Linear-only, but other architectures (HunyuanVideo, FramePack,
the VAE side of any pipeline) do use Conv2d LoRA, so a generic lora.py
regression there would silently break those trainers.
"""

import unittest

import torch

from musubi_tuner.networks.lora import LoRAModule


def _build_conv2d_lora(
    in_channels: int = 8,
    out_channels: int = 16,
    kernel_size: int = 3,
    padding: int = 1,
    lora_dim: int = 4,
    rank_dropout: float | None = None,
) -> tuple[torch.nn.Conv2d, LoRAModule]:
    base = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
    lora = LoRAModule(
        lora_name="test_conv2d",
        org_module=base,
        multiplier=1.0,
        lora_dim=lora_dim,
        alpha=lora_dim,
        rank_dropout=rank_dropout,
    )
    lora.apply_to()
    return base, lora


class Conv2dLoRARegression(unittest.TestCase):
    def test_conv2d_forward_shape_and_dtype(self):
        torch.manual_seed(0)
        base, lora = _build_conv2d_lora()
        lora.eval()

        x = torch.randn(2, 8, 16, 16)
        out = base(x)
        self.assertEqual(out.shape, (2, 16, 16, 16))
        self.assertEqual(out.dtype, x.dtype)

    def test_conv2d_backward_reaches_adapters(self):
        """Forward + backward must produce non-zero gradients on both lora_down
        and lora_up. Guards against any rewrite that detaches the adapter
        from autograd (e.g., a generic addmm_-style fast path that assumes
        2D weights and silently skips the Conv2d case)."""
        torch.manual_seed(0)
        base, lora = _build_conv2d_lora()
        lora.train()

        # Break the standard zeros-init on lora_up so the adapter path
        # carries a real gradient signal.
        with torch.no_grad():
            lora.lora_up.weight.normal_(mean=0.0, std=0.01)

        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        out = base(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(lora.lora_down.weight.grad)
        self.assertIsNotNone(lora.lora_up.weight.grad)
        self.assertGreater(lora.lora_down.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(lora.lora_up.weight.grad.abs().sum().item(), 0.0)

    def test_conv2d_rank_dropout_exercises_4d_unsqueeze(self):
        """Conv2d lx has shape (batch, lora_dim, H, W) — 4D. The rank_dropout
        mask is (batch, lora_dim) and gets `unsqueeze(-1).unsqueeze(-1)` to
        broadcast over spatial dims. This is the only path that exercises
        the `lx.dim() == 4` branch in non-split_dims LoRA, so it is a
        dedicated guard for that mask-shape logic."""
        torch.manual_seed(0)
        base, lora = _build_conv2d_lora(rank_dropout=0.1)
        lora.train()

        with torch.no_grad():
            lora.lora_up.weight.normal_(mean=0.0, std=0.01)

        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        out = base(x)
        self.assertEqual(out.shape, (2, 16, 16, 16))

        loss = out.sum()
        loss.backward()

        # Adapter grads must be non-zero — confirms the mask was broadcast
        # correctly (a broken mask shape would either raise or silently
        # zero the contribution).
        self.assertGreater(lora.lora_down.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(lora.lora_up.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
