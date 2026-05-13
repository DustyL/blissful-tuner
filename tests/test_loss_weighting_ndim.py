import unittest

import torch

from musubi_tuner.hv_train_network import compute_loss_weighting_for_sd3, get_sigmas


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


class TestGetSigmasContinuousTimesteps(unittest.TestCase):
    """Regression test for get_sigmas() with continuous (non-exact-match) timesteps.

    Continuous-timestep samplers like flux2_shift produce floats that don't exact-match
    the integer schedule. After unifying both code paths on nearest-neighbor lookup,
    these must resolve to the closest scheduler index, not error or silently drop entries.
    """

    def test_float_timesteps_resolve_to_nearest_index(self):
        sched = _DummyScheduler()  # timesteps [0,1,2,3] -> sigmas [1.0, 0.5, 0.25, 0.0]
        # 0.4 -> nearest 0 -> 1.0; 1.7 -> nearest 2 -> 0.25; 2.9 -> nearest 3 -> 0.0
        timesteps = torch.tensor([0.4, 1.7, 2.9], dtype=torch.float32)

        sigma = get_sigmas(sched, timesteps, device="cpu", n_dim=1)

        self.assertEqual(sigma.shape, (3,))
        self.assertAlmostEqual(sigma[0].item(), 1.0, places=6)
        self.assertAlmostEqual(sigma[1].item(), 0.25, places=6)
        self.assertAlmostEqual(sigma[2].item(), 0.0, places=6)

    def test_exact_match_integer_timesteps_still_resolve_correctly(self):
        """Strict-superset guarantee: exact matches must keep returning the exact index."""
        sched = _DummyScheduler()
        timesteps = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        sigma = get_sigmas(sched, timesteps, device="cpu", n_dim=1)

        self.assertEqual(sigma.shape, (4,))
        for idx, expected in enumerate([1.0, 0.5, 0.25, 0.0]):
            self.assertAlmostEqual(sigma[idx].item(), expected, places=6)


if __name__ == "__main__":
    unittest.main()
