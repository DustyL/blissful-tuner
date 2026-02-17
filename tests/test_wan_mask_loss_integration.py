"""Tests for WAN mask loss integration with 5D video tensors (TT-03).

Verifies that mask_loss works correctly with WAN-shaped video latents
(B, 16, F, H, W) including mask promotion, channel broadcasting, and
prior preservation.
"""

import argparse
import unittest

import torch

from musubi_tuner.modules.mask_loss import apply_masked_loss, apply_masked_loss_with_prior


def _wan_args(**overrides) -> argparse.Namespace:
    """Build a mask-loss args namespace with WAN-typical defaults."""
    defaults = dict(
        use_mask_loss=True,
        mask_gamma=1.0,
        mask_min_weight=0.0,
        prior_preservation_weight=0.0,
        prior_mask_threshold=None,
        normalize_per_sample=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestWanMaskLossIntegration(unittest.TestCase):
    """Mask loss integration tests using WAN-shaped 5D video tensors."""

    # --- Test 1: 16-channel video loss with 4D mask ---

    def test_16ch_video_loss_with_4d_mask(self) -> None:
        """WAN 16-channel loss (B,16,F,H,W) with 4D mask (B,F,H,W) should work."""
        torch.manual_seed(42)
        loss = torch.rand(2, 16, 5, 8, 8, dtype=torch.float32)
        mask = torch.rand(2, 5, 8, 8, dtype=torch.float32)
        args = _wan_args()

        result = apply_masked_loss(loss, mask, args=args, layout="video")

        self.assertEqual(result.ndim, 0)  # scalar
        self.assertEqual(result.dtype, torch.float32)
        self.assertGreater(result.item(), 0.0)

    # --- Test 2: 16-channel video loss with 5D mask ---

    def test_16ch_video_loss_with_5d_mask(self) -> None:
        """WAN loss with 5D mask (B,1,F,H,W) should work equivalently to 4D."""
        torch.manual_seed(42)
        loss = torch.rand(2, 16, 5, 8, 8, dtype=torch.float32)
        mask_4d = torch.rand(2, 5, 8, 8, dtype=torch.float32)
        mask_5d = mask_4d.unsqueeze(1)  # (B, 1, F, H, W)
        args = _wan_args()

        result_4d = apply_masked_loss(loss, mask_4d, args=args, layout="video")
        result_5d = apply_masked_loss(loss, mask_5d, args=args, layout="video")

        self.assertTrue(torch.allclose(result_4d, result_5d, atol=1e-6))

    # --- Test 3: mask_weights=None falls back to plain mean ---

    def test_no_mask_weights_fallback(self) -> None:
        """When mask_weights is None, loss should equal plain .mean()."""
        torch.manual_seed(42)
        loss = torch.rand(2, 16, 5, 8, 8, dtype=torch.float32)
        args = _wan_args()

        result = apply_masked_loss_with_prior(loss, None, prior_loss_unreduced=None, args=args, layout="video")
        expected = loss.float().mean()

        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    # --- Test 4: use_mask_loss=False short-circuits even with mask present ---

    def test_mask_loss_disabled_shortcircuit(self) -> None:
        """With use_mask_loss=False, mask is ignored — result equals plain mean."""
        torch.manual_seed(42)
        loss = torch.rand(1, 16, 5, 4, 4, dtype=torch.float32)
        mask = torch.zeros(1, 5, 4, 4, dtype=torch.float32)  # all-zero would normally zero loss
        args = _wan_args(use_mask_loss=False)

        result = apply_masked_loss_with_prior(loss, mask, prior_loss_unreduced=None, args=args, layout="video")
        expected = loss.float().mean()

        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    # --- Test 5: multi-channel consistency ---

    def test_multi_channel_consistency(self) -> None:
        """Weighted mean should be consistent regardless of channel count.

        A uniform mask of ones should give the same result as plain mean,
        validating that the num_channels scaling is correct for C=16.
        """
        torch.manual_seed(42)
        loss = torch.rand(2, 16, 5, 8, 8, dtype=torch.float32)
        mask = torch.ones(2, 5, 8, 8, dtype=torch.float32)
        args = _wan_args()

        result = apply_masked_loss(loss, mask, args=args, layout="video")
        expected = loss.float().mean()

        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    # --- Test 6: prior preservation with WAN shapes ---

    def test_prior_preservation_wan_shapes(self) -> None:
        """Prior preservation should add prior loss in unmasked regions."""
        # Uniform loss=1.0 everywhere, mask selects first half of spatial
        loss = torch.ones(1, 16, 5, 4, 4, dtype=torch.float32)
        prior_loss = torch.ones(1, 16, 5, 4, 4, dtype=torch.float32) * 2.0
        mask = torch.zeros(1, 5, 4, 4, dtype=torch.float32)
        mask[:, :, :2, :] = 1.0  # top half is target, bottom half is background
        args = _wan_args(prior_preservation_weight=1.0)

        result = apply_masked_loss_with_prior(loss, mask, prior_loss_unreduced=prior_loss, args=args, layout="video")

        # L_target: weighted mean of loss*mask / sum(mask) = mean over top half = 1.0
        # L_prior: weighted mean of prior*(1-mask) / sum(1-mask) * weight = mean over bottom half * 1.0 = 2.0
        # Total = 1.0 + 2.0 = 3.0
        self.assertTrue(torch.allclose(result, torch.tensor(3.0), atol=1e-4))

    # --- Test 7: gamma correction with WAN shapes ---

    def test_gamma_correction(self) -> None:
        """Gamma changes relative weights between high/low mask regions.

        With a non-uniform mask, gamma affects which regions dominate the loss.
        gamma < 1 lifts low weights (softer contrast), gamma > 1 suppresses them (sharper).
        """
        # Construct loss where high-mask region has high loss and low-mask region has low loss
        loss = torch.zeros(1, 16, 1, 1, 2, dtype=torch.float32)
        loss[:, :, :, :, 0] = 2.0  # high-loss pixel
        loss[:, :, :, :, 1] = 0.5  # low-loss pixel
        mask = torch.tensor([[[[0.9, 0.1]]]], dtype=torch.float32)  # (B=1, F=1, H=1, W=2)

        result_gamma_1 = apply_masked_loss(loss, mask, args=_wan_args(mask_gamma=1.0), layout="video")
        result_gamma_half = apply_masked_loss(loss, mask, args=_wan_args(mask_gamma=0.5), layout="video")
        result_gamma_2 = apply_masked_loss(loss, mask, args=_wan_args(mask_gamma=2.0), layout="video")

        # gamma=0.5 lifts the 0.1 weight → low-loss region contributes more → overall loss decreases
        # gamma=2.0 suppresses the 0.1 weight → high-loss region dominates more → overall loss increases
        self.assertLess(result_gamma_half.item(), result_gamma_1.item())
        self.assertGreater(result_gamma_2.item(), result_gamma_1.item())

    # --- Test 8: min_weight floor with WAN shapes ---

    def test_min_weight_floor(self) -> None:
        """mask_min_weight should prevent fully zeroed regions."""
        loss = torch.ones(1, 16, 3, 4, 4, dtype=torch.float32)
        mask = torch.zeros(1, 3, 4, 4, dtype=torch.float32)  # all-zero mask

        # Without floor: loss should be zero (or near-zero from epsilon handling)
        result_no_floor = apply_masked_loss(loss, mask, args=_wan_args(mask_min_weight=0.0), layout="video")
        # With floor: mask_min_weight=0.5 → all regions get at least 0.5 weight
        result_with_floor = apply_masked_loss(loss, mask, args=_wan_args(mask_min_weight=0.5), layout="video")

        self.assertLess(result_no_floor.item(), 0.01)
        self.assertGreater(result_with_floor.item(), 0.4)


if __name__ == "__main__":
    unittest.main()
