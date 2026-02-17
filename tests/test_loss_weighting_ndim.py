import unittest

import torch

from musubi_tuner.hv_train_network import compute_loss_weighting_for_sd3


class _DummyScheduler:
    def __init__(self):
        # Minimal attributes required by get_sigmas()
        self.timesteps = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0], dtype=torch.float32)


class TestLossWeightingNDIM(unittest.TestCase):
    def test_sigma_sqrt_weighting_4d_broadcast(self):
        """W14: 4D latents/loss must use n_dim=4 so weighting broadcasts per-batch, not across batch."""
        sched = _DummyScheduler()
        timesteps = torch.tensor([1, 2], dtype=torch.int64)

        weighting = compute_loss_weighting_for_sd3("sigma_sqrt", sched, timesteps, device="cpu", dtype=torch.float32, n_dim=4)
        self.assertIsNotNone(weighting)
        assert weighting is not None
        self.assertEqual(weighting.ndim, 4)
        self.assertEqual(weighting.shape[0], 2)

        loss = torch.ones((2, 3, 8, 8), dtype=torch.float32)
        out = loss * weighting
        self.assertEqual(out.shape, loss.shape)

    def test_sigma_sqrt_weighting_5d_broadcast(self):
        """W14: 5D latents/loss must use n_dim=5 for video tensors."""
        sched = _DummyScheduler()
        timesteps = torch.tensor([1, 2], dtype=torch.int64)

        weighting = compute_loss_weighting_for_sd3("sigma_sqrt", sched, timesteps, device="cpu", dtype=torch.float32, n_dim=5)
        self.assertIsNotNone(weighting)
        assert weighting is not None
        self.assertEqual(weighting.ndim, 5)
        self.assertEqual(weighting.shape[0], 2)

        loss = torch.ones((2, 3, 1, 8, 8), dtype=torch.float32)
        out = loss * weighting
        self.assertEqual(out.shape, loss.shape)


if __name__ == "__main__":
    unittest.main()
