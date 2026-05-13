"""Backward-anomaly smoke test for Flux.2 SingleStreamBlock and DoubleStreamBlock.

This is the tripwire that fires when in-place tensor mutations break autograd.
The xzuyn-optimizations branch's `img.add_(...)` / `txt.add_(...)` / `output.mul_(mod_gate)`
on residual-stream tensors causes version-counter mismatches during backward.
Per second-opinion review, gradient checkpointing did not neutralize this in a
CPU probe, so we run BOTH checkpointed and non-checkpointed forwards under
torch.autograd.set_detect_anomaly(True).

This test runs against the current (unmodified) blocks as a baseline — it
should pass. Once optimization commits land (especially step 10, the
fresh-temporary in-place arithmetic), re-running this test catches any
mutation that overreaches into residual-stream territory.

CPU-only and tiny so it's cheap to keep in the default test suite.
"""

import unittest

import torch

from musubi_tuner.flux_2.flux2_models import (
    DoubleStreamBlock,
    SingleStreamBlock,
    rope,
)
from musubi_tuner.modules.attention import AttentionParams


def _build_pe(batch: int, seq_len: int, head_dim: int, device: torch.device) -> torch.Tensor:
    """Construct a positional-encoding tensor with the shape apply_rope expects.

    rope(pos, dim, theta) returns shape (B, N, dim//2, 2, 2). Caller's apply_rope
    broadcasts this against q/k shaped (B, H, L, D).
    """
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).expand(batch, -1)
    pe = rope(pos, head_dim, theta=10_000)
    # Insert a head-dim slot for broadcast: (B, 1, N, D//2, 2, 2)
    return pe.unsqueeze(1)


def _torch_attn_params() -> AttentionParams:
    # "torch" routes to SDPA which is CPU-friendly and pure-PyTorch
    # (no flash-attn dependency that would skew anomaly behavior).
    return AttentionParams.create_attention_params("torch", split_attn=False)


class SingleStreamBlockBackwardAnomaly(unittest.TestCase):
    def _run_block(self, gradient_checkpointing: bool) -> None:
        torch.manual_seed(0)
        hidden_size = 16
        num_heads = 2
        head_dim = hidden_size // num_heads
        batch = 2
        seq_len = 8

        block = SingleStreamBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=2.0)
        if gradient_checkpointing:
            block.enable_gradient_checkpointing()
        block.train()

        x = torch.randn(batch, seq_len, hidden_size, requires_grad=True)
        pe = _build_pe(batch, seq_len, head_dim, device=x.device)
        mod_shift = torch.randn(batch, 1, hidden_size, requires_grad=True)
        mod_scale = torch.randn(batch, 1, hidden_size, requires_grad=True)
        mod_gate = torch.randn(batch, 1, hidden_size, requires_grad=True)

        with torch.autograd.set_detect_anomaly(True):
            out = block(x, pe, (mod_shift, mod_scale, mod_gate), _torch_attn_params())
            loss = out.sum()
            loss.backward()

        # Sanity: input gradient was computed (autograd actually ran).
        self.assertIsNotNone(x.grad)

    def test_no_checkpointing(self):
        self._run_block(gradient_checkpointing=False)

    def test_with_checkpointing(self):
        self._run_block(gradient_checkpointing=True)


class DoubleStreamBlockBackwardAnomaly(unittest.TestCase):
    def _run_block(self, gradient_checkpointing: bool) -> None:
        torch.manual_seed(0)
        hidden_size = 16
        num_heads = 2
        head_dim = hidden_size // num_heads
        batch = 2
        img_seq = 8
        txt_seq = 4

        block = DoubleStreamBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=2.0)
        if gradient_checkpointing:
            block.enable_gradient_checkpointing()
        block.train()

        img = torch.randn(batch, img_seq, hidden_size, requires_grad=True)
        txt = torch.randn(batch, txt_seq, hidden_size, requires_grad=True)
        pe_img = _build_pe(batch, img_seq, head_dim, device=img.device)
        pe_txt = _build_pe(batch, txt_seq, head_dim, device=img.device)

        def _mod_triple():
            return (
                torch.randn(batch, 1, hidden_size, requires_grad=True),
                torch.randn(batch, 1, hidden_size, requires_grad=True),
                torch.randn(batch, 1, hidden_size, requires_grad=True),
            )

        mod_img = (_mod_triple(), _mod_triple())
        mod_txt = (_mod_triple(), _mod_triple())

        with torch.autograd.set_detect_anomaly(True):
            out_img, out_txt = block(img, txt, pe_img, pe_txt, mod_img, mod_txt, _torch_attn_params())
            loss = out_img.sum() + out_txt.sum()
            loss.backward()

        self.assertIsNotNone(img.grad)
        self.assertIsNotNone(txt.grad)

    def test_no_checkpointing(self):
        self._run_block(gradient_checkpointing=False)

    def test_with_checkpointing(self):
        self._run_block(gradient_checkpointing=True)


if __name__ == "__main__":
    unittest.main()
