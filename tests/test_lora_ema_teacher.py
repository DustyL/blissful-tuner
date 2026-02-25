import unittest

import torch

from musubi_tuner.modules.lora_ema_teacher import LoRAEmaTeacher


class _DummyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float16))
        self.b = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float16))


class TestLoRAEmaTeacher(unittest.TestCase):
    def test_apply_to_swaps_and_restores_in_place(self) -> None:
        net = _DummyNet()
        w_id = id(net.w)
        b_id = id(net.b)

        ema = LoRAEmaTeacher(decay=0.5)
        ema.init_from(net)

        # Change live params, then update EMA so EMA != live.
        with torch.no_grad():
            net.w.copy_(torch.tensor([5.0, 6.0], dtype=torch.float16))
            net.b.copy_(torch.tensor([7.0], dtype=torch.float16))
        ema.update(net)

        # EMA values should be midpoint of initial and current:
        # w: ( [1,2] + [5,6] ) / 2 = [3,4]
        # b: ( [3]   + [7]   ) / 2 = [5]
        expected_w = torch.tensor([3.0, 4.0], dtype=torch.float16)
        expected_b = torch.tensor([5.0], dtype=torch.float16)

        with ema.apply_to(net):
            self.assertTrue(torch.allclose(net.w, expected_w))
            self.assertTrue(torch.allclose(net.b, expected_b))
            # Parameters must not be replaced (graph/DDP safety).
            self.assertEqual(id(net.w), w_id)
            self.assertEqual(id(net.b), b_id)

        # After context exit, weights are restored to the live (post-update) values.
        self.assertTrue(torch.allclose(net.w, torch.tensor([5.0, 6.0], dtype=torch.float16)))
        self.assertTrue(torch.allclose(net.b, torch.tensor([7.0], dtype=torch.float16)))

    def test_update_matches_expected_ema_math(self) -> None:
        net = _DummyNet()
        ema = LoRAEmaTeacher(decay=0.9)
        ema.init_from(net)

        with torch.no_grad():
            net.w.copy_(torch.tensor([11.0, 21.0], dtype=torch.float16))

        ema.update(net)

        # ema = 0.9 * [1,2] + 0.1 * [11,21] = [2, 3.9]
        with ema.apply_to(net):
            self.assertTrue(torch.allclose(net.w, torch.tensor([2.0, 3.9], dtype=torch.float16), rtol=0, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
