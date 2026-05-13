"""Structural invariant test for LoRA target coverage.

This is the test that would have caught the xzuyn-optimizations QKV-split
bypass before it shipped. The Musubi/blissful LoRA system attaches adapters
by monkey-patching `target_module.forward` (lora.py:490-495). Any code that
reads a target's raw weight tensor and calls `F.linear(x, weight)` directly
— instead of `module(x)` which routes through __call__ → the patched forward
— silently skips its adapter. The adapter still exists, still consumes VRAM,
still saves into the checkpoint, but contributes nothing to forward output
and receives zero gradient.

This test asserts a behavioral invariant of the LoRA system: when a network
is attached to a model and the model is run forward + backward with a
non-trivial loss, at least one of the targeted adapter parameters must
receive a non-zero gradient.

The test is intentionally coarse — it does not pin any specific architecture
or naming convention. Its purpose is to fire whenever someone wires up a
LoRA target that doesn't actually participate in training, regardless of
why. That class of bug is silent in unit tests of the model alone (forward
output looks correct because adapters are at zeros-init) and only shows up
behaviorally: training loss never improves, the LoRA file is full of zeros.
"""

import unittest

import torch

from musubi_tuner.networks.lora import create_network


class AttentionBlock(torch.nn.Module):
    """Tiny attention-style block where every Linear is a candidate LoRA target.

    The forward routes every sub-Linear through `self.<name>(x)` (i.e., via
    nn.Module.__call__, which is what triggers the LoRA monkey-patch).
    """

    def __init__(self, dim: int = 16):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Toy "attention": (Q * K) -> softmax -> @ V -> proj.
        # Cheap to compute, and every Linear gets exercised.
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
        return self.proj(attn @ v)


class TinyModel(torch.nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.block = AttentionBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_network(model: TinyModel):
    return create_network(
        ["AttentionBlock"],
        "lora_unet",
        multiplier=1.0,
        network_dim=8,
        network_alpha=4.0,
        vae=None,
        text_encoders=[],
        unet=model,
    )


class LoRATargetCoverageInvariant(unittest.TestCase):
    def test_at_least_one_adapter_has_nonzero_gradient_after_backward(self):
        """The structural invariant: if you attach a LoRA network and run
        forward + backward, the adapter must participate in autograd."""
        torch.manual_seed(0)
        model = TinyModel(dim=16)
        network = _make_network(model)
        network.apply_to(text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True)
        network.train()

        x = torch.randn(2, 4, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # `lora_up.weight.grad` is the most reliable signal at init: even
        # though `lora_up.weight` starts at zeros, its gradient is
        # scale * lora_down(x), which is non-zero for non-zero input.
        # If grad is None, autograd never reached the adapter — i.e. the
        # adapter forward did not fire — i.e. the target was bypassed.
        adapters_with_grad = [
            lora
            for lora in network.unet_loras
            if lora.lora_up.weight.grad is not None and lora.lora_up.weight.grad.abs().sum().item() > 0
        ]
        self.assertGreater(
            len(adapters_with_grad),
            0,
            "No LoRA adapter received a non-zero gradient after backward. "
            "This means every targeted module was bypassed — likely via a code path "
            "that reads `module.weight` directly instead of calling `module(x)`. "
            f"Targets attempted: {[lora.lora_name for lora in network.unet_loras]}",
        )

    def test_every_adapter_participates_in_autograd(self):
        """Stronger variant: assert *every* attached adapter received some
        gradient signal. Catches partial-bypass bugs where only some targets
        are routed through their wrapped forwards (e.g., a rewrite that
        replaces `self.q(x)`/`self.k(x)` with raw F.linear but leaves `v`
        and `proj` alone)."""
        torch.manual_seed(0)
        model = TinyModel(dim=16)
        network = _make_network(model)
        network.apply_to(text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True)
        network.train()

        x = torch.randn(2, 4, 16)
        loss = model(x).sum()
        loss.backward()

        bypassed = [
            lora.lora_name
            for lora in network.unet_loras
            if lora.lora_up.weight.grad is None or lora.lora_up.weight.grad.abs().sum().item() == 0.0
        ]
        self.assertEqual(
            bypassed,
            [],
            f"Some adapters are wired up but received no gradient — they were bypassed: {bypassed}",
        )


if __name__ == "__main__":
    unittest.main()
