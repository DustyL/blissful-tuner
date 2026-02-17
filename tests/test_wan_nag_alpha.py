"""Regression test for DM-01: WanAttentionBlock must forward nag_alpha to NAGCrossAttention.

The original bug was a typo — `kwargs.get("nah_alpha", 0.5)` — which silently
ignored the user-provided nag_alpha value and always used the default 0.5.
"""

import unittest

from musubi_tuner.wan.modules.model import WanAttentionBlock


class TestNAGAlphaForwarding(unittest.TestCase):
    """Verify that nag_alpha flows from WanAttentionBlock kwargs to cross_attn."""

    def _make_block(self, nag_alpha: float) -> WanAttentionBlock:
        return WanAttentionBlock(
            cross_attn_type="t2v_cross_attn",
            dim=128,
            ffn_dim=512,
            num_heads=4,
            nag_scale=1.0,  # must be set to trigger NAG path
            nag_alpha=nag_alpha,
        )

    def test_custom_nag_alpha_reaches_cross_attn(self) -> None:
        """Non-default nag_alpha must appear on the cross_attn module."""
        block = self._make_block(nag_alpha=0.3)
        self.assertAlmostEqual(block.cross_attn.nag_alpha, 0.3)

    def test_default_nag_alpha(self) -> None:
        """When nag_alpha is not provided, default 0.5 should be used."""
        block = WanAttentionBlock(
            cross_attn_type="t2v_cross_attn",
            dim=128,
            ffn_dim=512,
            num_heads=4,
            nag_scale=1.0,
        )
        self.assertAlmostEqual(block.cross_attn.nag_alpha, 0.5)

    def test_nag_alpha_not_silently_overridden(self) -> None:
        """Edge-case values must pass through, not get clamped or replaced."""
        for alpha in (0.0, 0.1, 0.9, 1.0):
            block = self._make_block(nag_alpha=alpha)
            self.assertAlmostEqual(block.cross_attn.nag_alpha, alpha, msg=f"nag_alpha={alpha} was not forwarded")


if __name__ == "__main__":
    unittest.main()
