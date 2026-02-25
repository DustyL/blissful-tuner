import unittest

import torch

from musubi_tuner.modules.prior_scheduling import compute_prior_weight_per_sample


class TestPriorScheduling(unittest.TestCase):
    def test_pivot_behavior_linear(self) -> None:
        """t >= pivot should clamp to max weight; t < pivot should decay toward 0 at t=0."""
        timesteps = torch.tensor([0.0, 150.0, 300.0, 900.0], dtype=torch.float32)
        w = compute_prior_weight_per_sample(
            timesteps,
            base_weight=2.0,
            schedule="linear",
            pivot_timestep=300.0,
            global_step=100,
            warmup_steps=0,
        )
        expected = torch.tensor([0.0, 1.0, 2.0, 2.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(w, expected, rtol=0, atol=1e-6))

    def test_pivot_behavior_cosine(self) -> None:
        timesteps = torch.tensor([0.0, 75.0, 150.0, 300.0, 900.0], dtype=torch.float32)
        w = compute_prior_weight_per_sample(
            timesteps,
            base_weight=2.0,
            schedule="cosine",
            pivot_timestep=300.0,
            global_step=100,
            warmup_steps=0,
        )

        # t_norm = clamp(t,0,pivot)/pivot; for cosine schedule:
        # factor = 0.5 - 0.5*cos(pi*t_norm)
        # t=0      -> 0.0
        # t=75     -> t_norm=0.25 -> ~0.1464466
        # t=150    -> t_norm=0.5  -> 0.5
        # t>=300   -> 1.0
        expected = torch.tensor([0.0, 2.0 * 0.1464466, 1.0, 2.0, 2.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(w, expected, rtol=0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
