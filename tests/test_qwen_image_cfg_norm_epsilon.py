import unittest

import torch

from musubi_tuner.qwen_image import qwen_image_utils


class TestQwenImageCfgNormEpsilon(unittest.TestCase):
    def test_apply_cfg_norm_avoids_nan_when_combined_pred_is_zero(self):
        cond = torch.ones((1, 2, 3), dtype=torch.float32)
        comb = torch.zeros((1, 2, 3), dtype=torch.float32)

        # Demonstrate the original failure mode: 0 * inf -> NaN.
        naive = comb * (torch.norm(cond, dim=-1, keepdim=True) / torch.norm(comb, dim=-1, keepdim=True))
        self.assertTrue(torch.isnan(naive).any().item())

        out = qwen_image_utils.apply_cfg_norm(cond, comb)
        self.assertFalse(torch.isnan(out).any().item())
        self.assertTrue(torch.all(out == 0).item())

    def test_apply_cfg_norm_preserves_norm_when_nonzero(self):
        cond = torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16)
        comb = torch.tensor([[[6.0, 8.0]]], dtype=torch.bfloat16)

        out = qwen_image_utils.apply_cfg_norm(cond, comb)

        self.assertEqual(out.dtype, comb.dtype)
        self.assertTrue(torch.allclose(out.float(), cond.float(), rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

