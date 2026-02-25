import argparse
import unittest

import numpy as np
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset.image_video_dataset import ItemInfo


class TestCacheMaskPreprocessing(unittest.TestCase):
    def test_apply_cache_mask_transforms_gamma_and_min_weight(self) -> None:
        mask = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        out = cache_latents.apply_cache_mask_transforms(mask, cache_mask_gamma=0.5, cache_mask_min_weight=0.2)

        expected_mid = (0.5**0.5) * (1.0 - 0.2) + 0.2
        expected = torch.tensor([0.2, expected_mid, 1.0], dtype=torch.float32)

        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.allclose(out, expected, rtol=0, atol=1e-6))

    def test_preprocess_contents_bakes_gamma_before_downsample(self) -> None:
        # 8x8 RGBA image with a hard "boundary": top half hair (alpha=80), bottom half background (alpha=0).
        # Downsampling to latent space (stride=8) produces a single 1x1 mask weight.
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[:4, :, 3] = 80  # hair
        rgba[4:, :, 3] = 0  # background

        item = ItemInfo(item_key="example.png", caption="", original_size=(8, 8), content=rgba)
        item.control_content = []  # preprocess_contents expects a list for image controls

        prev_gamma = cache_latents.CACHE_MASK_GAMMA
        prev_min_weight = cache_latents.CACHE_MASK_MIN_WEIGHT
        try:
            cache_latents.set_cache_mask_transform_args(argparse.Namespace(cache_mask_gamma=0.7, cache_mask_min_weight=0.0))

            _, _, _, masks = cache_latents.preprocess_contents([item])
            mask = masks[0][0]
            self.assertIsNotNone(mask)

            baked_value = float(mask.item())

            hair = 80.0 / 255.0
            expected_gamma_before = (hair**0.7) * 0.5  # half hair pixels, half background pixels
            expected_gamma_after = (hair * 0.5) ** 0.7

            # Baking gamma before downsample should match the "gamma-before" math (and be strictly smaller).
            self.assertAlmostEqual(baked_value, expected_gamma_before, delta=1e-5)
            self.assertGreater(expected_gamma_after, expected_gamma_before)
            self.assertLess(baked_value, expected_gamma_after)
        finally:
            cache_latents.set_cache_mask_transform_args(
                argparse.Namespace(cache_mask_gamma=prev_gamma, cache_mask_min_weight=prev_min_weight)
            )


if __name__ == "__main__":
    unittest.main()
