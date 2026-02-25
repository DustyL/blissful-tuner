import os
import tempfile
import unittest

import torch
from safetensors import safe_open

from musubi_tuner.dataset.image_video_dataset import (
    ItemInfo,
    save_latent_cache_flux_2,
    save_latent_cache_qwen_image,
    save_latent_cache_wan,
)


class TestMaskWeightsCacheDtype(unittest.TestCase):
    def test_wan_mask_weights_saved_as_float16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            item = ItemInfo(
                item_key="sample",
                caption="",
                original_size=(1024, 1024),
                bucket_size=(1024, 1024),
            )
            item.latent_cache_path = os.path.join(tmpdir, "sample_wan.safetensors")

            latent = torch.zeros(16, 1, 8, 8, dtype=torch.bfloat16)  # (C, F, H, W)
            mask_weights = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)  # (1, F, H, W)

            save_latent_cache_wan(
                item_info=item,
                latent=latent,
                clip_embed=None,
                image_latent=None,
                control_latent=None,
                f_indices=None,
                mask_weights=mask_weights,
            )

            with safe_open(item.latent_cache_path, framework="pt") as f:
                keys = list(f.keys())
                mask_keys = [k for k in keys if k.startswith("mask_weights_")]

                self.assertTrue(any(k.startswith("latents_") for k in keys))
                self.assertEqual(len(mask_keys), 1)

                loaded_mask = f.get_tensor(mask_keys[0])
                self.assertEqual(loaded_mask.dtype, torch.float16)
                self.assertEqual(tuple(loaded_mask.shape), tuple(mask_weights.shape))

    def test_flux2_mask_weights_saved_as_float16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            item = ItemInfo(
                item_key="sample",
                caption="",
                original_size=(1024, 1024),
                bucket_size=(1024, 1024),
            )
            item.latent_cache_path = os.path.join(tmpdir, "sample_flux2.safetensors")

            latent = torch.zeros(16, 8, 8, dtype=torch.bfloat16)  # (C, H, W)
            mask_weights = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)  # (1, 1, H, W)

            save_latent_cache_flux_2(
                item_info=item,
                latent=latent,
                control_latent=None,
                arch_full="flux_2_dev",
                mask_weights=mask_weights,
            )

            with safe_open(item.latent_cache_path, framework="pt") as f:
                keys = list(f.keys())
                mask_keys = [k for k in keys if k.startswith("mask_weights_")]

                self.assertTrue(any(k.startswith("latents_") for k in keys))
                self.assertEqual(len(mask_keys), 1)

                loaded_mask = f.get_tensor(mask_keys[0])
                self.assertEqual(loaded_mask.dtype, torch.float16)
                self.assertEqual(tuple(loaded_mask.shape), tuple(mask_weights.shape))

    def test_qwen_image_mask_weights_saved_as_float16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            item = ItemInfo(
                item_key="sample",
                caption="",
                original_size=(1024, 1024),
                bucket_size=(1024, 1024),
            )
            item.latent_cache_path = os.path.join(tmpdir, "sample_qwen.safetensors")

            latent = torch.zeros(16, 1, 8, 8, dtype=torch.bfloat16)  # (C, F/L, H, W)
            mask_weights = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)  # (1, F/L, H, W)

            save_latent_cache_qwen_image(
                item_info=item,
                latent=latent,
                control_latent=None,
                mask_weights=mask_weights,
            )

            with safe_open(item.latent_cache_path, framework="pt") as f:
                keys = list(f.keys())
                mask_keys = [k for k in keys if k.startswith("mask_weights_")]

                self.assertTrue(any(k.startswith("latents_") for k in keys))
                self.assertEqual(len(mask_keys), 1)

                loaded_mask = f.get_tensor(mask_keys[0])
                self.assertEqual(loaded_mask.dtype, torch.float16)
                self.assertEqual(tuple(loaded_mask.shape), tuple(mask_weights.shape))


if __name__ == "__main__":
    unittest.main()
