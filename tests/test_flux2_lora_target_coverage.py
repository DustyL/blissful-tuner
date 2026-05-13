"""FLUX.2-specific LoRA target coverage invariant.

The generic `tests/test_lora_target_coverage.py` proves the LoRA mechanism
itself works on a toy `AttentionBlock` that calls `self.q(x)`, etc. — but
it would still pass if someone refactored *FLUX.2's* SingleStreamBlock or
DoubleStreamBlock to do `F.linear(x, module.weight)` directly (the
xzuyn-optimizations QKV-split rewrite). That class of bug needs a
FLUX.2-specific test.

This file attaches LoRA adapters to the actual `linear1`, `linear2`, and
attention `qkv` submodules of tiny SingleStreamBlock / DoubleStreamBlock
instances, runs forward + backward, and asserts every adapter receives a
non-zero `lora_up.weight.grad`. Any adapter that doesn't has been bypassed.

CPU-only, tiny, fast.
"""

import unittest

import torch

from musubi_tuner.flux_2.flux2_models import DoubleStreamBlock, SingleStreamBlock, rope
from musubi_tuner.modules.attention import AttentionParams
from musubi_tuner.networks.lora import LoRAModule


def _wrap_with_lora(target_module: torch.nn.Linear, name: str, lora_dim: int = 4) -> LoRAModule:
    """Attach a LoRAModule to a Linear via the canonical apply_to() pattern.

    Returns the LoRAModule so the caller can inspect gradients post-backward.
    """
    lora = LoRAModule(
        lora_name=name,
        org_module=target_module,
        multiplier=1.0,
        lora_dim=lora_dim,
        alpha=lora_dim,
    )
    # Break the canonical zeros-init on lora_up so the adapter contribution is
    # non-trivial during forward — without this, `lora_up.weight.grad` would
    # still be non-zero (depends on `lora_down(x)`, not `lora_up.weight`), but
    # the forward output would equal the base model's output, weakening the
    # behavioral assertion. Perturbation makes the adapter's effect observable.
    with torch.no_grad():
        lora.lora_up.weight.normal_(mean=0.0, std=0.01)
    lora.apply_to()
    return lora


def _build_pe(batch: int, seq_len: int, head_dim: int) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).expand(batch, -1)
    return rope(pos, head_dim, theta=10_000).unsqueeze(1)


def _assert_adapter_received_gradient(test_case: unittest.TestCase, lora: LoRAModule, label: str) -> None:
    """The structural invariant: `lora_up.weight.grad` must be present and
    non-zero. If grad is None, the adapter forward never fired — i.e. the
    module was bypassed via raw-weight access or similar."""
    grad = lora.lora_up.weight.grad
    test_case.assertIsNotNone(grad, f"{label}: adapter forward did not fire — bypassed")
    test_case.assertGreater(
        grad.abs().sum().item(),
        0.0,
        f"{label}: adapter received zero gradient — output detached from loss",
    )


class SingleStreamBlockLoRACoverage(unittest.TestCase):
    """Verify LoRA flows through SingleStreamBlock's `linear1` and `linear2`."""

    def test_single_stream_block_targets_receive_gradient(self):
        torch.manual_seed(0)
        hidden_size = 16
        num_heads = 2
        head_dim = hidden_size // num_heads
        batch = 2
        seq_len = 4

        block = SingleStreamBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=2.0)
        block.train()

        # Attach LoRA to the two LoRA-target linears inside the block.
        # linear1 fuses Q/K/V/MLP; linear2 is the output projection.
        lora_linear1 = _wrap_with_lora(block.linear1, "single.linear1")
        lora_linear2 = _wrap_with_lora(block.linear2, "single.linear2")

        x = torch.randn(batch, seq_len, hidden_size, requires_grad=True)
        pe = _build_pe(batch, seq_len, head_dim)
        mod = (
            torch.randn(batch, 1, hidden_size),
            torch.randn(batch, 1, hidden_size),
            torch.randn(batch, 1, hidden_size),
        )
        attn_params = AttentionParams.create_attention_params("torch", split_attn=False)

        out = block(x, pe, mod, attn_params)
        out.sum().backward()

        _assert_adapter_received_gradient(self, lora_linear1, "single.linear1")
        _assert_adapter_received_gradient(self, lora_linear2, "single.linear2")


class DoubleStreamBlockLoRACoverage(unittest.TestCase):
    """Verify LoRA flows through DoubleStreamBlock's `img_attn.qkv`,
    `txt_attn.qkv`, `img_attn.proj`, and `txt_attn.proj`. These are exactly
    the modules the xzuyn-optimizations branch silently bypassed when it
    rewrote the forward to read `img_attn.qkv.weight` directly via
    F.linear, so each is a load-bearing canary."""

    def test_double_stream_block_targets_receive_gradient(self):
        torch.manual_seed(0)
        hidden_size = 16
        num_heads = 2
        head_dim = hidden_size // num_heads
        batch = 2
        img_seq = 6
        txt_seq = 4

        block = DoubleStreamBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=2.0)
        block.train()

        # The four "QKV-split-bypass" canaries plus the projection layers,
        # which are also LoRA targets in production FLUX.2.
        lora_img_qkv = _wrap_with_lora(block.img_attn.qkv, "double.img_attn.qkv")
        lora_txt_qkv = _wrap_with_lora(block.txt_attn.qkv, "double.txt_attn.qkv")
        lora_img_proj = _wrap_with_lora(block.img_attn.proj, "double.img_attn.proj")
        lora_txt_proj = _wrap_with_lora(block.txt_attn.proj, "double.txt_attn.proj")

        img = torch.randn(batch, img_seq, hidden_size, requires_grad=True)
        txt = torch.randn(batch, txt_seq, hidden_size, requires_grad=True)
        pe_img = _build_pe(batch, img_seq, head_dim)
        pe_txt = _build_pe(batch, txt_seq, head_dim)

        def _mod_triple():
            return (
                torch.randn(batch, 1, hidden_size),
                torch.randn(batch, 1, hidden_size),
                torch.randn(batch, 1, hidden_size),
            )

        mod_img = (_mod_triple(), _mod_triple())
        mod_txt = (_mod_triple(), _mod_triple())
        attn_params = AttentionParams.create_attention_params("torch", split_attn=False)

        out_img, out_txt = block(img, txt, pe_img, pe_txt, mod_img, mod_txt, attn_params)
        (out_img.sum() + out_txt.sum()).backward()

        _assert_adapter_received_gradient(self, lora_img_qkv, "double.img_attn.qkv")
        _assert_adapter_received_gradient(self, lora_txt_qkv, "double.txt_attn.qkv")
        _assert_adapter_received_gradient(self, lora_img_proj, "double.img_attn.proj")
        _assert_adapter_received_gradient(self, lora_txt_proj, "double.txt_attn.proj")


if __name__ == "__main__":
    unittest.main()
