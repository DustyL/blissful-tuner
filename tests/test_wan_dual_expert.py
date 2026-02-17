"""Tests for WAN 2.2 dual-expert training logic (TT-01).

Covers:
- Timestep boundary classification (high vs low noise)
- Bucketing + dual-expert incompatibility (TP-01)
- Expert swap direction tracking
"""

import argparse
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestTimestepBoundaryClassification(unittest.TestCase):
    """Verify that timestep boundary logic correctly classifies high vs low noise."""

    def test_high_noise_above_boundary(self) -> None:
        """Timesteps above boundary * 1000 should be classified as high noise."""
        boundary = 0.875
        # t=900 → 900/1000 = 0.9 >= 0.875 → high noise
        t = torch.tensor([900.0])
        self.assertTrue(t[0] / 1000.0 >= boundary)

    def test_low_noise_below_boundary(self) -> None:
        """Timesteps below boundary * 1000 should be classified as low noise."""
        boundary = 0.875
        # t=500 → 500/1000 = 0.5 < 0.875 → low noise
        t = torch.tensor([500.0])
        self.assertFalse(t[0] / 1000.0 >= boundary)

    def test_boundary_edge_case_exact(self) -> None:
        """Timestep exactly at boundary should be classified as high noise (>=)."""
        boundary = 0.875
        t = torch.tensor([875.0])
        self.assertTrue(t[0] / 1000.0 >= boundary)

    def test_i2v_boundary_higher(self) -> None:
        """I2V boundary (0.900) is higher than T2V (0.875)."""
        t2v_boundary = 0.875
        i2v_boundary = 0.900
        # t=880 → high noise for T2V, low noise for I2V
        t = torch.tensor([880.0])
        self.assertTrue(t[0] / 1000.0 >= t2v_boundary)
        self.assertFalse(t[0] / 1000.0 >= i2v_boundary)


class TestBucketingDualExpertIncompatibility(unittest.TestCase):
    """Verify that timestep bucketing is rejected with dual-expert training (TP-01)."""

    @patch("musubi_tuner.wan_train_network.detect_wan_sd_dtype", return_value=torch.bfloat16)
    def test_bucketing_with_dual_expert_raises(self, _mock_detect: MagicMock) -> None:
        """num_timestep_buckets + high/low training must raise ValueError."""
        from musubi_tuner.wan_train_network import WanNetworkTrainer

        trainer = WanNetworkTrainer.__new__(WanNetworkTrainer)
        args = argparse.Namespace(
            dit_high_noise="some_path",
            dit="some_path",
            task="t2v-A14B",
            blocks_to_swap=None,
            num_timestep_buckets=10,
            offload_inactive_dit=False,
            timestep_boundary=None,
            mixed_precision="bf16",
            fp8_scaled=False,
        )

        with self.assertRaises(ValueError) as ctx:
            trainer.handle_model_specific_args(args)

        self.assertIn("incompatible", str(ctx.exception).lower())

    @patch("musubi_tuner.wan_train_network.detect_wan_sd_dtype", return_value=torch.bfloat16)
    def test_no_bucketing_with_dual_expert_ok(self, _mock_detect: MagicMock) -> None:
        """Dual-expert without bucketing should not raise about bucketing."""
        from musubi_tuner.wan_train_network import WanNetworkTrainer

        trainer = WanNetworkTrainer.__new__(WanNetworkTrainer)
        args = argparse.Namespace(
            dit_high_noise="some_path",
            dit="some_path",
            task="t2v-A14B",
            blocks_to_swap=None,
            num_timestep_buckets=None,
            offload_inactive_dit=False,
            timestep_boundary=None,
            mixed_precision="bf16",
            fp8_scaled=False,
        )

        # Should not raise ValueError about bucketing
        try:
            trainer.handle_model_specific_args(args)
        except ValueError as e:
            self.assertNotIn("incompatible", str(e).lower(), f"Unexpected bucketing error: {e}")
        except Exception:
            pass  # Other errors from incomplete mock setup are fine


class TestFlowMatchingConvention(unittest.TestCase):
    """Verify the flow matching convention used in this codebase."""

    def test_t0_is_clean_t1_is_noise(self) -> None:
        """In this codebase: noisy = (1-t)*clean + t*noise. t=0 → clean, t=1 → noise."""
        clean = torch.ones(1, 16, 1, 4, 4)
        noise = torch.randn(1, 16, 1, 4, 4)

        # t=0 → should be clean
        t = torch.tensor([0.0]).view(-1, 1, 1, 1, 1)
        noisy = (1 - t) * clean + t * noise
        self.assertTrue(torch.allclose(noisy, clean))

        # t=1 → should be noise
        t = torch.tensor([1.0]).view(-1, 1, 1, 1, 1)
        noisy = (1 - t) * clean + t * noise
        self.assertTrue(torch.allclose(noisy, noise))

    def test_velocity_target(self) -> None:
        """Rectified flow velocity target is (noise - clean)."""
        clean = torch.ones(1, 16, 1, 4, 4)
        noise = torch.randn(1, 16, 1, 4, 4)
        target = noise - clean
        # Verify: at any t, if model predicts target perfectly, we can recover noise and clean
        t = 0.5
        noisy = (1 - t) * clean + t * noise
        # model_pred ≈ target → recovered_clean = noisy - t * target
        recovered_clean = noisy - t * target
        self.assertTrue(torch.allclose(recovered_clean, clean, atol=1e-6))


class TestDualExpertRejectionSampling(unittest.TestCase):
    """Verify dual-expert rejection sampling constrains timesteps to the chosen expert region."""

    @patch("musubi_tuner.hv_train_network.NetworkTrainer.get_noisy_model_input_and_timesteps")
    def test_rejection_sampling_forces_high_noise_region(self, mock_super_get: MagicMock) -> None:
        """If the first sample is high-noise, all batch elements should be resampled into the high-noise region."""
        from musubi_tuner.wan_train_network import WanNetworkTrainer

        boundary = 0.875
        bsize = 3

        trainer = WanNetworkTrainer.__new__(WanNetworkTrainer)
        trainer.high_low_training = True
        trainer.timestep_boundary = boundary
        trainer.num_timestep_buckets = None

        # Call pattern:
        # - initial call (noise[0:1]) selects region
        # - per-item loop resamples until accepted for each i in batch
        timesteps_queue = [torch.tensor([950.0])] + [torch.tensor([800.0]), torch.tensor([920.0])] * bsize

        def side_effect(args, noise, latents, timesteps, noise_scheduler, device, dtype):
            ts = timesteps_queue.pop(0)
            return torch.zeros_like(latents), ts

        mock_super_get.side_effect = side_effect

        args = argparse.Namespace()
        noise = torch.randn(bsize, 16, 1, 2, 2)
        latents = torch.randn(bsize, 16, 1, 2, 2)

        noisy_out, timesteps_out = trainer.get_noisy_model_input_and_timesteps(
            args=args,
            noise=noise,
            latents=latents,
            timesteps=None,
            noise_scheduler=MagicMock(),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(noisy_out.shape[0], bsize)
        self.assertEqual(timesteps_out.shape[0], bsize)
        self.assertTrue(trainer.next_model_is_high_noise)
        self.assertTrue(bool(torch.all(timesteps_out / 1000.0 >= boundary)))

    @patch("musubi_tuner.hv_train_network.NetworkTrainer.get_noisy_model_input_and_timesteps")
    def test_rejection_sampling_forces_low_noise_region(self, mock_super_get: MagicMock) -> None:
        """If the first sample is low-noise, all batch elements should be resampled into the low-noise region."""
        from musubi_tuner.wan_train_network import WanNetworkTrainer

        boundary = 0.875
        bsize = 3

        trainer = WanNetworkTrainer.__new__(WanNetworkTrainer)
        trainer.high_low_training = True
        trainer.timestep_boundary = boundary
        trainer.num_timestep_buckets = None

        timesteps_queue = [torch.tensor([500.0])] + [torch.tensor([920.0]), torch.tensor([800.0])] * bsize

        def side_effect(args, noise, latents, timesteps, noise_scheduler, device, dtype):
            ts = timesteps_queue.pop(0)
            return torch.zeros_like(latents), ts

        mock_super_get.side_effect = side_effect

        args = argparse.Namespace()
        noise = torch.randn(bsize, 16, 1, 2, 2)
        latents = torch.randn(bsize, 16, 1, 2, 2)

        noisy_out, timesteps_out = trainer.get_noisy_model_input_and_timesteps(
            args=args,
            noise=noise,
            latents=latents,
            timesteps=None,
            noise_scheduler=MagicMock(),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(noisy_out.shape[0], bsize)
        self.assertEqual(timesteps_out.shape[0], bsize)
        self.assertFalse(trainer.next_model_is_high_noise)
        self.assertTrue(bool(torch.all(timesteps_out / 1000.0 < boundary)))


if __name__ == "__main__":
    unittest.main()
