import argparse
import os
import tempfile
import unittest

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_Z_IMAGE,
    ItemInfo,
    save_latent_cache_z_image,
    scan_cache_mask_transform_metadata,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


class TestMaskCacheMetadata(unittest.TestCase):
    def test_latent_cache_writes_cache_mask_transform_metadata(self) -> None:
        prev_gamma = cache_latents.CACHE_MASK_GAMMA
        prev_min_weight = cache_latents.CACHE_MASK_MIN_WEIGHT
        try:
            cache_latents.set_cache_mask_transform_args(argparse.Namespace(cache_mask_gamma=0.7, cache_mask_min_weight=0.2))

            with tempfile.TemporaryDirectory() as tmpdir:
                item = ItemInfo(item_key="sample", caption="", original_size=(1024, 1024), bucket_size=(1024, 1024))
                item.latent_cache_path = os.path.join(tmpdir, f"sample_{ARCHITECTURE_Z_IMAGE}.safetensors")

                latent = torch.zeros(16, 8, 8, dtype=torch.bfloat16)
                mask_weights = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)
                save_latent_cache_z_image(item_info=item, latent=latent, mask_weights=mask_weights)

                with MemoryEfficientSafeOpen(item.latent_cache_path) as f:
                    md = f.metadata()

                self.assertIn("cache_mask_gamma", md)
                self.assertIn("cache_mask_min_weight", md)
                self.assertAlmostEqual(float(md["cache_mask_gamma"]), 0.7, places=6)
                self.assertAlmostEqual(float(md["cache_mask_min_weight"]), 0.2, places=6)
        finally:
            cache_latents.set_cache_mask_transform_args(
                argparse.Namespace(cache_mask_gamma=prev_gamma, cache_mask_min_weight=prev_min_weight)
            )

    def test_scan_cache_mask_metadata_finds_unique_pair(self) -> None:
        prev_gamma = cache_latents.CACHE_MASK_GAMMA
        prev_min_weight = cache_latents.CACHE_MASK_MIN_WEIGHT
        try:
            cache_latents.set_cache_mask_transform_args(argparse.Namespace(cache_mask_gamma=0.7, cache_mask_min_weight=0.2))

            with tempfile.TemporaryDirectory() as tmpdir:
                item = ItemInfo(item_key="sample", caption="", original_size=(1024, 1024), bucket_size=(1024, 1024))
                item.latent_cache_path = os.path.join(tmpdir, f"sample_{ARCHITECTURE_Z_IMAGE}.safetensors")

                latent = torch.zeros(16, 8, 8, dtype=torch.bfloat16)
                save_latent_cache_z_image(item_info=item, latent=latent, mask_weights=None)

                class _DummyDataset:
                    cache_directory = tmpdir
                    architecture = ARCHITECTURE_Z_IMAGE

                pairs, with_meta, checked = scan_cache_mask_transform_metadata([_DummyDataset()], max_files_per_dataset=8)
                self.assertEqual(checked, 1)
                self.assertEqual(with_meta, 1)
                self.assertEqual(pairs, {(0.7, 0.2)})
        finally:
            cache_latents.set_cache_mask_transform_args(
                argparse.Namespace(cache_mask_gamma=prev_gamma, cache_mask_min_weight=prev_min_weight)
            )


if __name__ == "__main__":
    unittest.main()
