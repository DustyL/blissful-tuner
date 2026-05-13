"""Regression test for split_dims + rank_dropout in LoRAModule.

The split_dims rank_dropout branch (lora.py:613-620) had a stale-`lx` bug
inherited from upstream: the mask shape check `if len(lx.size()) == 3`
referenced a function-scope `lx` that was only assigned in the other
(split_dims is None) branch. Hitting the rank_dropout path with split_dims
set raises `UnboundLocalError: cannot access local variable 'lx' where it
is not associated with a value`.

This test wraps a Linear with LoRAModule(split_dims=...), enables
rank_dropout + training mode, and runs forward + backward via the
monkey-patched org_module call site.

Failure modes guarded:
  - UnboundLocalError (the original blissful-tuner bug)
  - TypeError (the xzuyn-optimizations attempted fix, which evaluated
    `len(lxs[i].dim())` — dim() returns int, so len() raises)
  - Any silent shape mismatch in the mask unsqueeze logic
"""

import unittest

import torch

from musubi_tuner.networks.lora import LoRAModule


def _build_split_dims_lora(in_features: int, split_dims: list[int], lora_dim: int = 8) -> tuple[torch.nn.Linear, LoRAModule]:
    """Create a Linear + LoRAModule(split_dims=...) and wire them via apply_to().

    Returns (base_linear, lora_module). Call base_linear(x) to route through
    the LoRA forward; the rank_dropout path fires inside it when training=True.
    """
    out_features = sum(split_dims)
    base = torch.nn.Linear(in_features, out_features, bias=False)
    lora = LoRAModule(
        lora_name="test_split_dims",
        org_module=base,
        multiplier=1.0,
        lora_dim=lora_dim,
        alpha=lora_dim,
        rank_dropout=0.1,
        split_dims=split_dims,
    )
    lora.apply_to()
    return base, lora


class SplitDimsRankDropoutRegression(unittest.TestCase):
    def test_3d_input_forward_backward_does_not_raise(self):
        """3D input (batch, seq, features) — the FLUX.2 case."""
        torch.manual_seed(0)
        in_features = 16
        split_dims = [8, 8, 8]
        base, lora = _build_split_dims_lora(in_features=in_features, split_dims=split_dims)
        lora.train()

        # LoRA's canonical zeros-init on lora_up means the adapter contribution
        # is exactly zero at init, so loss = org_forwarded.sum() carries no
        # gradient signal back to lora_down. Perturb lora_up to a small non-zero
        # value so the autograd path through both lora_down and lora_up is
        # genuinely exercised by the backward pass.
        with torch.no_grad():
            for lora_up in lora.lora_up:
                lora_up.weight.normal_(mean=0.0, std=0.01)

        x = torch.randn(2, 4, in_features, requires_grad=True)
        out = base(x)
        self.assertEqual(out.shape, (2, 4, sum(split_dims)))

        loss = out.sum()
        loss.backward()

        # Adapter parameters must receive a non-zero gradient — guards against
        # a "fix" that silently zeroes the rank_dropout output (or otherwise
        # detaches the adapter path from autograd).
        any_down_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in lora.lora_down.parameters()
        )
        any_up_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in lora.lora_up.parameters()
        )
        self.assertTrue(any_down_grad, "no gradient reached any lora_down parameter")
        self.assertTrue(any_up_grad, "no gradient reached any lora_up parameter")

    def test_2d_input_forward_does_not_raise(self):
        """2D input (batch, features) — the case where lx.dim() == 2 and
        neither unsqueeze branch is taken. Guards against an over-eager
        dim-check refactor."""
        torch.manual_seed(0)
        in_features = 16
        split_dims = [8, 8, 8]
        base, lora = _build_split_dims_lora(in_features=in_features, split_dims=split_dims)
        lora.train()

        x = torch.randn(2, in_features)
        out = base(x)
        self.assertEqual(out.shape, (2, sum(split_dims)))

    def test_eval_mode_bypasses_rank_dropout(self):
        """rank_dropout path is gated on self.training — eval() must skip
        it cleanly even with split_dims set. Pins the gate so a future
        refactor cannot always-run the rank_dropout branch."""
        torch.manual_seed(0)
        in_features = 16
        split_dims = [8, 8, 8]
        base, lora = _build_split_dims_lora(in_features=in_features, split_dims=split_dims)
        lora.eval()

        x = torch.randn(2, 4, in_features)
        out = base(x)
        self.assertEqual(out.shape, (2, 4, sum(split_dims)))


if __name__ == "__main__":
    unittest.main()
