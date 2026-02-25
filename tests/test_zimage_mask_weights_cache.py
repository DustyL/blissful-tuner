import os
import tempfile
import unittest

import torch
from safetensors import safe_open

from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_z_image


class TestZImageMaskWeightsCaching(unittest.TestCase):
    def test_save_latent_cache_z_image_writes_mask_weights_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            item = ItemInfo(
                item_key="sample",
                caption="",
                original_size=(1024, 1024),
                bucket_size=(1024, 1024),
            )
            item.latent_cache_path = os.path.join(tmpdir, "sample_zi.safetensors")

            latent = torch.zeros(16, 8, 8, dtype=torch.bfloat16)
            mask_weights = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)

            save_latent_cache_z_image(item_info=item, latent=latent, mask_weights=mask_weights)

            with safe_open(item.latent_cache_path, framework="pt") as f:
                keys = list(f.keys())
                mask_keys = [k for k in keys if k.startswith("mask_weights_")]

                self.assertTrue(any(k.startswith("latents_") for k in keys))
                self.assertEqual(len(mask_keys), 1)

                loaded_mask = f.get_tensor(mask_keys[0])
                # Mask weights are bounded [0,1] and originate from 8-bit masks, so caches store them as float16
                # to reduce disk size / I/O.
                self.assertEqual(loaded_mask.dtype, torch.float16)
                self.assertEqual(tuple(loaded_mask.shape), tuple(mask_weights.shape))


if __name__ == "__main__":
    unittest.main()
