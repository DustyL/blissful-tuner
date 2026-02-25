import unittest

import torch

from musubi_tuner.modules.loss_utils import compute_unreduced_target_loss


class TestLossUtils(unittest.TestCase):
    def test_mse_unreduced_matches_squared_error(self) -> None:
        pred = torch.tensor([0.0, 2.0, -3.0], dtype=torch.float32)
        target = torch.zeros_like(pred)
        out = compute_unreduced_target_loss(pred, target, loss_type="mse")
        self.assertTrue(torch.allclose(out, torch.tensor([0.0, 4.0, 9.0], dtype=torch.float32)))
        self.assertEqual(out.shape, pred.shape)

    def test_huber_unreduced_matches_piecewise_definition(self) -> None:
        pred = torch.tensor([0.0, 2.0, -3.0], dtype=torch.float32)
        target = torch.zeros_like(pred)
        out = compute_unreduced_target_loss(pred, target, loss_type="huber", loss_delta=1.0)
        expected = torch.tensor([0.0, 1.5, 2.5], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected, rtol=0, atol=1e-6))
        self.assertEqual(out.shape, pred.shape)

    def test_invalid_loss_type_raises(self) -> None:
        pred = torch.zeros(1)
        with self.assertRaises(ValueError, msg="Unsupported loss_type"):
            compute_unreduced_target_loss(pred, pred, loss_type="nope")

    def test_invalid_delta_raises(self) -> None:
        pred = torch.zeros(1)
        with self.assertRaises(ValueError, msg="--loss_delta"):
            compute_unreduced_target_loss(pred, pred, loss_type="huber", loss_delta=0.0)


if __name__ == "__main__":
    unittest.main()
