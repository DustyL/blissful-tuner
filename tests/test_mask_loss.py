import argparse
import unittest

import torch

from musubi_tuner.modules.mask_loss import (
    apply_masked_loss,
    apply_masked_loss_with_prior,
    validate_mask_loss_args,
)


class TestMaskedLossWithPrior(unittest.TestCase):
    def test_equivalent_to_apply_masked_loss_when_prior_disabled(self) -> None:
        torch.manual_seed(0)
        loss = torch.rand(2, 3, 2, 4, 4, dtype=torch.float32)
        mask_weights = torch.rand(2, 2, 4, 4, dtype=torch.float32)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=0.7,
            mask_min_weight=0.05,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        expected = apply_masked_loss(loss, mask_weights, args=args, layout="video")
        actual = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=torch.rand_like(loss),  # should be ignored when weight=0
            args=args,
            layout="video",
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_threshold_mode_matches_manual_split(self) -> None:
        # Two masked pixels (target), two background pixels (prior)
        loss = torch.tensor([[[[[1.0, 1.0, 1.0, 1.0]]]]])  # (B=1,C=1,F=1,H=1,W=4)
        prior_loss = torch.tensor([[[[[2.0, 2.0, 2.0, 2.0]]]]])
        mask_weights = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])  # (B=1,F=1,H=1,W=4)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.5,
            prior_mask_threshold=0.1,
            normalize_per_sample=False,
        )

        # L_target = mean([1,1]) = 1
        # L_prior  = mean([2,2]) * 0.5 = 1
        expected = torch.tensor(2.0)
        actual = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=prior_loss,
            args=args,
            layout="video",
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_mask_blur_kernel_size_changes_weighted_mean(self) -> None:
        # Hard mask boundary: without blur, only last 3 pixels contribute.
        # With blur, boundary pixels get fractional weights.
        loss = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0, 5.0]]]]])  # (B=1,C=1,F=1,H=1,W=5)
        mask_weights = torch.tensor([[[[0.0, 0.0, 1.0, 1.0, 1.0]]]])  # (B=1,F=1,H=1,W=5)

        args_no_blur = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
            mask_blur_kernel_size=0,
        )
        args_blur = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
            mask_blur_kernel_size=3,
        )

        out_no_blur = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=None, args=args_no_blur, layout="video")
        out_blur = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=None, args=args_blur, layout="video")

        # Without blur: mean([3,4,5]) = 4.0
        self.assertTrue(torch.allclose(out_no_blur, torch.tensor(4.0), rtol=0, atol=1e-6))

        # With blur (k=3): matches the Gaussian kernel that torchvision would use by default.
        # For kernel_size=3, sigma = 0.3*((1)-1)+0.8 = 0.8, kernel = exp(-x^2/(2*sigma^2)) normalized.
        kernel_size = 3
        sigma = 0.3 * (((kernel_size - 1) * 0.5) - 1.0) + 0.8
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-(coords**2) / (2.0 * (sigma**2)))
        kernel_1d = kernel_1d / kernel_1d.sum()

        a = kernel_1d[0]
        b = kernel_1d[1]
        # Because H=1 and padding is replicate, 2D blur reduces to 1D along W.
        blurred_mask = torch.tensor([0.0, a, a + b, 1.0, 1.0], dtype=torch.float32)
        expected = (loss.flatten() * blurred_mask).sum() / blurred_mask.sum()

        self.assertTrue(torch.allclose(out_blur, expected, rtol=0, atol=1e-5))

    def test_threshold_mode_prevents_target_prior_overlap(self) -> None:
        # Background would normally get mask_min_weight, but threshold-mode should zero it for target loss.
        loss = torch.tensor([[[[[1.0, 1.0, 1.0, 1.0]]]]])  # (B=1,C=1,F=1,H=1,W=4)
        prior_loss = torch.tensor([[[[[2.0, 2.0, 2.0, 2.0]]]]])
        mask_weights = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])  # (B=1,F=1,H=1,W=4)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.2,  # would floor background to 0.2
            prior_preservation_weight=1.0,
            prior_mask_threshold=0.1,
            normalize_per_sample=False,
        )

        # With overlap prevention, target only uses masked pixels => mean([1,1]) = 1
        # Prior uses background pixels => mean([2,2]) * 1 = 2
        expected = torch.tensor(3.0)
        actual = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=prior_loss,
            args=args,
            layout="video",
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_prior_mask_threshold_uses_unblurred_mask_when_blur_enabled(self) -> None:
        # If prior_mask_threshold were applied to the blurred mask, the first pixel would no longer be < threshold
        # (replicate padding + Gaussian blur will make it > 0.1). We intentionally threshold on the raw/unblurred mask.
        loss = torch.ones(1, 1, 1, 1, 3, dtype=torch.float32)
        prior_loss = torch.ones_like(loss) * 10.0
        mask_weights = torch.tensor([[[[0.0, 1.0, 1.0]]]], dtype=torch.float32)  # (B=1,F=1,H=1,W=3)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            mask_blur_kernel_size=3,
            prior_preservation_weight=1.0,
            prior_mask_threshold=0.1,
            normalize_per_sample=False,
        )

        out = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=prior_loss, args=args, layout="video")
        # L_target = 1, L_prior = 10 * 1.0
        self.assertTrue(torch.allclose(out, torch.tensor(11.0), rtol=0, atol=1e-5))

    def test_continuous_mode_matches_manual_weighted_means(self) -> None:
        # Continuous mode uses complementary masks:
        #   prior_mask = 1 - mask_processed
        # so pixels can have both target and prior contributions (a mixing coefficient).
        loss = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]])  # (B=1,C=1,F=1,H=1,W=4)
        prior_loss = torch.tensor([[[[[10.0, 20.0, 30.0, 40.0]]]]])
        mask_weights = torch.tensor([[[[1.0, 0.5, 0.0, 0.25]]]])  # (B=1,F=1,H=1,W=4)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.5,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        # L_target = sum(loss * mask) / sum(mask) = 3 / 1.75
        # L_prior  = (sum(prior_loss * (1-mask)) / sum(1-mask)) * w_prior = (70 / 2.25) * 0.5
        expected = torch.tensor((3.0 / 1.75) + (70.0 / 2.25) * 0.5)

        actual = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=prior_loss,
            args=args,
            layout="video",
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_normalize_per_sample_changes_reduction(self) -> None:
        # Sample 0 has full mask; sample 1 has zero mask.
        # Global reduction => 1.0, per-sample => (1.0 + 0.0)/2 = 0.5
        loss = torch.ones(2, 1, 1, 1, 1, dtype=torch.float32)
        mask_weights = torch.tensor([[[[1.0]]], [[[0.0]]]], dtype=torch.float32)  # (B=2,F=1,H=1,W=1)

        args_global = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )
        args_per_sample = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=True,
        )

        global_loss = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=None,
            args=args_global,
            layout="video",
        )
        per_sample_loss = apply_masked_loss_with_prior(
            loss, mask_weights, prior_loss_unreduced=None, args=args_per_sample, layout="video"
        )

        self.assertTrue(torch.allclose(global_loss, torch.tensor(1.0), rtol=0, atol=1e-6))
        self.assertTrue(torch.allclose(per_sample_loss, torch.tensor(0.5), rtol=0, atol=1e-6))

    def test_prior_term_is_zero_when_no_prior_region(self) -> None:
        loss = torch.tensor([[[[[1.0]]]]])  # (B=1,C=1,F=1,H=1,W=1)
        prior_loss = torch.tensor([[[[[100.0]]]]])
        mask_weights = torch.tensor([[[[1.0]]]])  # fully masked -> prior_mask all zeros

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=1.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        out = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=prior_loss, args=args, layout="video")
        self.assertTrue(torch.allclose(out, torch.tensor(1.0), rtol=0, atol=1e-6))

    def test_prior_term_is_zero_when_threshold_mode_has_no_background(self) -> None:
        loss = torch.tensor([[[[[1.0]]]]])  # (B=1,C=1,F=1,H=1,W=1)
        prior_loss = torch.tensor([[[[[100.0]]]]])
        mask_weights = torch.tensor([[[[1.0]]]])  # min(mask_raw)=1.0 >= threshold -> prior_mask all zeros

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=1.0,
            prior_mask_threshold=0.1,
            normalize_per_sample=False,
        )

        out = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=prior_loss, args=args, layout="video")
        self.assertTrue(torch.allclose(out, torch.tensor(1.0), rtol=0, atol=1e-6))

    def test_returns_float32_scalar(self) -> None:
        loss = torch.ones(1, 1, 1, 2, 2, dtype=torch.float16)
        mask_weights = torch.ones(1, 1, 2, 2, dtype=torch.float16)
        prior_loss = torch.zeros_like(loss)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=1.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        out = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=prior_loss,
            args=args,
            layout="video",
        )
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.ndim, 0)

    def test_bfloat16_loss_accepts_float16_mask_weights(self) -> None:
        """Regression: cache may store masks as float16 even when training uses bfloat16."""
        loss = torch.ones(1, 1, 1, 2, 2, dtype=torch.bfloat16)
        mask_weights = torch.full((1, 1, 2, 2), 0.5, dtype=torch.float16)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        out = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=None,
            args=args,
            layout="video",
        )
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertEqual(out.dtype, torch.float32)
        # Weighted mean of all-ones loss with constant 0.5 mask is still 1.0
        self.assertTrue(torch.allclose(out, torch.tensor(1.0), rtol=0, atol=1e-6))

    def test_layered_layout_matches_apply_masked_loss(self) -> None:
        loss = torch.tensor([[[[[1.0]]], [[[3.0]]]]])  # (B=1,L=2,C=1,H=1,W=1)
        mask_weights = torch.tensor([[[[[0.0]], [[1.0]], [[0.5]]]]])  # (B=1,1,F=3,H=1,W=1)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
            remove_first_image_from_target=True,
        )

        expected = apply_masked_loss(
            loss,
            mask_weights,
            args=args,
            layout="layered",
            drop_base_frame=True,
        )
        actual = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=None,
            args=args,
            layout="layered",
            drop_base_frame=True,
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-6))

    def test_raises_not_implemented_for_layered_with_prior(self) -> None:
        loss = torch.tensor([[[[[1.0]]], [[[3.0]]]]])  # (B=1,L=2,C=1,H=1,W=1)
        mask_weights = torch.tensor([[[[[0.0]], [[1.0]], [[0.5]]]]])  # (B=1,1,F=3,H=1,W=1)
        prior_loss = torch.rand_like(loss)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=1.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
            remove_first_image_from_target=True,
        )

        with self.assertRaises(NotImplementedError):
            apply_masked_loss_with_prior(
                loss,
                mask_weights,
                prior_loss_unreduced=prior_loss,
                args=args,
                layout="layered",
                drop_base_frame=True,
            )

    def test_raises_value_error_for_shape_mismatch_on_prior(self) -> None:
        loss = torch.ones(1, 3, 2, 4, 4)
        mask_weights = torch.ones(1, 2, 4, 4) * 0.5  # some mask to enable prior region
        prior_loss_wrong_shape = torch.ones(1, 3, 2, 4, 8)  # W mismatch

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=1.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        with self.assertRaises(ValueError, msg="prior_loss_unreduced shape"):
            apply_masked_loss_with_prior(
                loss,
                mask_weights,
                prior_loss_unreduced=prior_loss_wrong_shape,
                args=args,
                layout="video",
            )

    def test_multi_channel_matches_single_channel(self) -> None:
        """Verify compact mask processing gives same results regardless of channel count."""
        torch.manual_seed(42)
        mask_weights = torch.rand(2, 2, 4, 4)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=0.7,
            mask_min_weight=0.1,
            prior_preservation_weight=0.5,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        # C=1: each spatial position contributes once
        loss_c1 = torch.rand(2, 1, 2, 4, 4)
        prior_c1 = torch.rand(2, 1, 2, 4, 4)
        result_c1 = apply_masked_loss_with_prior(loss_c1, mask_weights, prior_loss_unreduced=prior_c1, args=args, layout="video")

        # C=1 replicated to C=4: each position contributes 4x but weighted mean is unchanged
        loss_c4 = loss_c1.expand(2, 4, 2, 4, 4)
        prior_c4 = prior_c1.expand(2, 4, 2, 4, 4)
        result_c4 = apply_masked_loss_with_prior(loss_c4, mask_weights, prior_loss_unreduced=prior_c4, args=args, layout="video")

        self.assertTrue(torch.allclose(result_c1, result_c4, rtol=0, atol=1e-5))

    def test_per_sample_with_prior_preservation(self) -> None:
        """Combined per-sample normalization + prior preservation."""
        # Sample 0: mostly masked (target=2, background=0.3)
        # Sample 1: mostly background (target=0.1, background=5)
        loss = torch.tensor(
            [
                [[[2.0, 2.0, 0.5, 0.5]]],  # sample 0
                [[[0.1, 3.0, 3.0, 3.0]]],  # sample 1
            ]
        ).unsqueeze(2)  # (B=2, C=1, F=1, H=1, W=4)
        prior_loss = torch.ones_like(loss) * 10.0
        mask_weights = torch.tensor(
            [
                [[[1.0, 1.0, 0.0, 0.0]]],  # sample 0: 2 target, 2 prior
                [[[1.0, 0.0, 0.0, 0.0]]],  # sample 1: 1 target, 3 prior
            ]
        )  # (B=2, F=1, H=1, W=4)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.5,
            prior_mask_threshold=0.1,
            normalize_per_sample=True,
        )

        result = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=prior_loss, args=args, layout="video")

        # Sample 0: L_target = mean([2,2]) = 2, L_prior = mean([10,10])*0.5 = 5
        # Sample 1: L_target = mean([0.1]) = 0.1, L_prior = mean([10,10,10])*0.5 = 5
        # Per-sample average: L_target = (2 + 0.1) / 2 = 1.05, L_prior = (5 + 5) / 2 = 5
        expected = torch.tensor(1.05 + 5.0)
        self.assertTrue(torch.allclose(result, expected, rtol=0, atol=1e-5))

    def test_prior_weight_per_sample_scales_per_sample_prior_term(self) -> None:
        """Per-sample prior weights should scale the per-sample prior term before batch mean."""
        # Two samples with different prior losses; only sample 0 should receive prior weight.
        loss = torch.tensor(
            [
                [[[[1.0, 999.0]]]],  # sample 0: target pixel = 1.0
                [[[[3.0, 999.0]]]],  # sample 1: target pixel = 3.0
            ],
            dtype=torch.float32,
        )  # (B=2,C=1,F=1,H=1,W=2)
        prior_loss = torch.tensor(
            [
                [[[[10.0, 10.0]]]],  # sample 0 prior value
                [[[[20.0, 20.0]]]],  # sample 1 prior value
            ],
            dtype=torch.float32,
        )
        mask_weights = torch.tensor(
            [
                [[[1.0, 0.0]]],  # sample 0: first pixel target, second pixel prior
                [[[1.0, 0.0]]],  # sample 1: same split
            ],
            dtype=torch.float32,
        )  # (B=2,F=1,H=1,W=2)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,  # ignored when prior_weight_per_sample is provided
            prior_mask_threshold=0.1,
            normalize_per_sample=True,
        )

        prior_weight_per_sample = torch.tensor([1.0, 0.0], dtype=torch.float32)
        out = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=prior_loss,
            prior_weight_per_sample=prior_weight_per_sample,
            args=args,
            layout="video",
        )

        # Target term: mean([1.0, 3.0]) = 2.0
        # Prior term:  mean([10.0*1.0, 20.0*0.0]) = 5.0
        self.assertTrue(torch.allclose(out, torch.tensor(7.0), rtol=0, atol=1e-6))

    def test_prior_weight_per_sample_requires_normalize_per_sample(self) -> None:
        loss = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32)
        prior_loss = torch.ones_like(loss)
        mask_weights = torch.ones(1, 1, 1, 1, dtype=torch.float32)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=0.1,
            normalize_per_sample=False,
        )

        with self.assertRaises(ValueError, msg="requires --normalize_per_sample"):
            apply_masked_loss_with_prior(
                loss,
                mask_weights,
                prior_loss_unreduced=prior_loss,
                prior_weight_per_sample=torch.tensor([1.0]),
                args=args,
                layout="video",
            )

    def test_all_zero_mask_emits_warning(self) -> None:
        """V1: All-zero masks should warn that training signal is zero."""
        loss = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32)
        mask_weights = torch.zeros(1, 1, 1, 1, dtype=torch.float32)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        with self.assertLogs("musubi_tuner.modules.mask_loss", level="WARNING") as cm:
            result = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=None, args=args, layout="video")

        self.assertTrue(any("All-zero mask" in m for m in cm.output), cm.output)
        self.assertTrue(torch.allclose(result, torch.tensor(0.0), atol=1e-6))

    def test_mask_weights_are_clamped_before_gamma(self) -> None:
        """Ensure out-of-range masks can't produce NaNs with fractional gamma."""
        loss = torch.tensor([[[[[1.0, 3.0]]]]], dtype=torch.float32)  # (B=1,C=1,F=1,H=1,W=2)
        # Out of range values: negative + >1.0. Should clamp to [0,1] before gamma.
        mask_weights = torch.tensor([[[[-0.5, 2.0]]]], dtype=torch.float32)  # (B=1,F=1,H=1,W=2)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=0.5,  # fractional gamma would NaN on negative values without clamping
            mask_min_weight=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        out = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=None, args=args, layout="video")
        self.assertTrue(torch.isfinite(out).all().item())
        # After clamp: mask -> [0,1], gamma keeps [0,1], weighted mean -> 3.0
        self.assertTrue(torch.allclose(out, torch.tensor(3.0), rtol=0, atol=1e-6))

    def test_prior_sample_mask_gates_threshold_overlap_and_denominator(self) -> None:
        """Per-sample prior gating must happen before threshold-mode overlap prevention."""
        # Two samples, same raw mask. We gate prior ON for sample 0, OFF for sample 1.
        # Threshold mode + mask_min_weight makes overlap-prevention observable:
        #   - With prior active: background is removed from target mask.
        #   - With prior gated off: background keeps its min_weight contribution.
        loss = torch.tensor(
            [
                [[[[1.0, 5.0]]]],  # sample 0
                [[[[1.0, 5.0]]]],  # sample 1
            ],
            dtype=torch.float32,
        )  # (B=2,C=1,F=1,H=1,W=2)
        prior_loss = torch.ones_like(loss) * 10.0
        mask_weights = torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )  # (B=2,F=1,H=1,W=2)
        prior_sample_mask = torch.tensor([True, False])

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.2,
            mask_area_scale_beta=0.0,
            prior_preservation_weight=1.0,
            prior_mask_threshold=0.1,
            normalize_per_sample=True,
        )

        out = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=prior_loss,
            prior_sample_mask=prior_sample_mask,
            args=args,
            layout="video",
        )

        # Sample 0 (prior ON): target uses only first pixel -> 1.0. Prior uses second pixel -> 10.0.
        # Sample 1 (prior OFF): target uses min_weight background -> (1 + 5*0.2) / (1 + 0.2) = 2/1.2.
        # Per-sample mean: L_target = (1 + 2/1.2)/2, L_prior = (10 + 0)/2
        expected = torch.tensor(((1.0 + (2.0 / 1.2)) / 2.0) + 5.0, dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected, rtol=0, atol=1e-5))

    def test_mask_area_scale_beta_scales_tiny_masks(self) -> None:
        """mask_area_scale_beta should scale L_target by (mask_mean ** beta)."""
        loss = torch.ones(1, 1, 1, 1, 4, dtype=torch.float32) * 2.0  # (B=1,C=1,F=1,H=1,W=4)
        mask_weights = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]], dtype=torch.float32)  # mask_mean=0.5

        args_no_scale = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            mask_area_scale_beta=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )
        args_scale = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            mask_area_scale_beta=1.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        out_no_scale = apply_masked_loss_with_prior(
            loss, mask_weights, prior_loss_unreduced=None, args=args_no_scale, layout="video"
        )
        out_scale = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=None, args=args_scale, layout="video")

        # Without scaling: strict weighted mean over masked pixels => 2.0
        self.assertTrue(torch.allclose(out_no_scale, torch.tensor(2.0), rtol=0, atol=1e-6))
        # With beta=1: scaled by mask_mean=0.5 => 1.0
        self.assertTrue(torch.allclose(out_scale, torch.tensor(1.0), rtol=0, atol=1e-6))

    def test_mask_area_scale_beta_is_per_sample_when_normalize_per_sample(self) -> None:
        # Sample 0: mask_mean=0.5, per-sample target mean=2.0
        # Sample 1: mask_mean=0.25, per-sample target mean=4.0
        # With beta=1 and per-sample normalization, scaled per-sample losses become:
        #   s0: 2.0 * 0.5 = 1.0
        #   s1: 4.0 * 0.25 = 1.0
        # => mean = 1.0
        loss = torch.tensor(
            [
                [[[[2.0, 2.0, 2.0, 2.0]]]],
                [[[[4.0, 4.0, 4.0, 4.0]]]],
            ],
            dtype=torch.float32,
        )  # (B=2,C=1,F=1,H=1,W=4)
        mask_weights = torch.tensor(
            [
                [[[1.0, 1.0, 0.0, 0.0]]],  # mean=0.5
                [[[1.0, 0.0, 0.0, 0.0]]],  # mean=0.25
            ],
            dtype=torch.float32,
        )  # (B=2,F=1,H=1,W=4)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            mask_area_scale_beta=1.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=True,
        )

        out = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced=None, args=args, layout="video")
        self.assertTrue(torch.allclose(out, torch.tensor(1.0), rtol=0, atol=1e-6))


class TestValidateMaskLossArgs(unittest.TestCase):
    def test_raises_error_for_prior_without_mask_loss(self) -> None:
        args = argparse.Namespace(
            use_mask_loss=False,
            prior_preservation_weight=1.0,
            prior_mask_threshold=None,
            mask_gamma=1.0,
            mask_min_weight=0.0,
        )

        with self.assertRaises(ValueError, msg="requires --use_mask_loss"):
            validate_mask_loss_args(args)

    def test_allows_prior_with_mask_loss(self) -> None:
        args = argparse.Namespace(
            use_mask_loss=True,
            prior_preservation_weight=1.0,
            prior_mask_threshold=None,
            mask_gamma=1.0,
            mask_min_weight=0.0,
        )

        # Should not raise
        validate_mask_loss_args(args)

    def test_raises_error_for_negative_mask_area_scale_beta(self) -> None:
        args = argparse.Namespace(
            use_mask_loss=True,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            mask_blur_kernel_size=0,
            mask_area_scale_beta=-0.1,
        )

        with self.assertRaises(ValueError, msg="--mask_area_scale_beta must be >= 0"):
            validate_mask_loss_args(args)

    def test_raises_error_for_even_mask_blur_kernel_size(self) -> None:
        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            mask_blur_kernel_size=4,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
        )

        with self.assertRaises(ValueError, msg="odd (or 0 to disable)"):
            validate_mask_loss_args(args)

    def test_prior_decay_schedule_requires_normalize_per_sample(self) -> None:
        args = argparse.Namespace(
            use_mask_loss=True,
            prior_preservation_weight=1.0,
            prior_mask_threshold=None,
            prior_decay_schedule="cosine",
            prior_decay_timestep_start=300.0,
            prior_decay_warmup_ratio=0.0,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            normalize_per_sample=False,
        )

        with self.assertRaises(ValueError, msg="requires --normalize_per_sample"):
            validate_mask_loss_args(args)

        args.normalize_per_sample = True
        validate_mask_loss_args(args)


if __name__ == "__main__":
    unittest.main()
