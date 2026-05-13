"""Behavioral regression test for DoRA forward exercising _dora_delta.

A "doesn't crash" test for DoRA is insufficient — a future optimization that
silently replaces the DoRA delta path with the simple `org_forwarded + lx *
scale` LoRA formula would pass smoke tests but produce a different (and worse)
training signal. The xzuyn-optimizations branch's `addmm_` rewrite in lora.py
would do exactly this: it fuses the residual add into a single addmm_ call
that assumes the simple LoRA formula, bypassing DoRA's magnitude-vector
contribution entirely.

This test asserts a behavioral invariant: when use_dora=True, the LoRA module
output must differ from the simple-LoRA output by more than numerical noise.
The DoRA delta formula adds a `(mag_norm_scale - 1) * base_wo_bias` term that
the simple formula lacks, and a `mag_norm_scale` factor on the lora_out term —
so any DoRA forward with a non-trivial magnitude vector must measurably depart
from the simple formula.

Blissful-tuner's FLUX.2-Klein training typically runs with use_dora=True, so
silent DoRA-bypass is a load-bearing regression class to guard against.
"""

import unittest

import torch

from musubi_tuner.networks.lora import LoRAModule


def _build_dora_lora(
    in_features: int = 32,
    out_features: int = 16,
    lora_dim: int = 4,
    alpha: float = 4.0,
) -> tuple[torch.nn.Linear, LoRAModule]:
    base = torch.nn.Linear(in_features, out_features, bias=True)
    lora = LoRAModule(
        lora_name="test_dora",
        org_module=base,
        multiplier=1.0,
        lora_dim=lora_dim,
        alpha=alpha,
        use_dora=True,
    )
    lora.apply_to()
    return base, lora


class DoRADeltaPathRegression(unittest.TestCase):
    def test_dora_is_enabled_for_linear_without_split_dims_or_dropout(self):
        """Precondition check: use_dora must actually take effect for the
        canonical Linear + no-split_dims + no-dropout case. If this fails,
        the rest of this file is testing the wrong code path."""
        _, lora = _build_dora_lora()
        self.assertTrue(lora.use_dora, "use_dora should be active for Linear without split_dims/dropout")
        self.assertIsNotNone(lora.dora_layer)

    def test_dora_output_differs_from_simple_lora_formula(self):
        """The actual DoRA forward must produce output materially different
        from the simple-LoRA formula `base(x) + lora_up(lora_down(x)) * scale`.
        If the two outputs match, _dora_delta was bypassed."""
        torch.manual_seed(0)
        base, lora = _build_dora_lora()
        lora.eval()  # deterministic — no rank_dropout interference

        # Perturb lora_up away from zeros-init so the DoRA delta actually
        # contributes. At zeros-init the magnitude vector equals the base
        # row norms exactly, mag_norm_scale collapses to 1, and the delta
        # is zero — indistinguishable from the simple LoRA formula.
        with torch.no_grad():
            lora.lora_up.weight.normal_(mean=0.0, std=0.05)

        x = torch.randn(2, 32)

        # Actual DoRA forward (routed via the monkey-patched base.forward).
        dora_out = base(x)

        # Simple-LoRA reference: bypass DoRA and compute what the addmm_
        # fast path would yield. Uses the same lora_down / lora_up / scale
        # / multiplier — only the DoRA magnitude logic is omitted.
        with torch.no_grad():
            simple_lora_out = lora.org_forward(x) + lora.lora_up(lora.lora_down(x)) * lora.multiplier * lora.scale

        diff = (dora_out - simple_lora_out).abs().max().item()
        self.assertGreater(
            diff,
            1e-4,
            f"DoRA output indistinguishable from simple-LoRA formula (max abs diff={diff:.2e}). "
            "This suggests _dora_delta was not invoked.",
        )

    def test_dora_delta_is_called_during_forward(self):
        """Direct spy: replace _dora_delta with a wrapper that records each
        invocation, then run forward and assert the wrapper was called. This
        is the most direct way to detect a silent DoRA-bypass — any rewrite
        that fuses the residual add without going through _dora_delta will
        fail this test even if the output happens to match numerically by
        coincidence."""
        torch.manual_seed(0)
        base, lora = _build_dora_lora()
        lora.eval()

        with torch.no_grad():
            lora.lora_up.weight.normal_(mean=0.0, std=0.05)

        call_count = {"n": 0}
        real_dora_delta = lora._dora_delta

        def spy(*args, **kwargs):
            call_count["n"] += 1
            return real_dora_delta(*args, **kwargs)

        lora._dora_delta = spy

        x = torch.randn(2, 32)
        _ = base(x)

        self.assertEqual(
            call_count["n"],
            1,
            f"_dora_delta should be called exactly once per forward. Got {call_count['n']} calls — "
            "either DoRA was bypassed (0) or there's an unexpected re-entry (>1).",
        )


if __name__ == "__main__":
    unittest.main()
