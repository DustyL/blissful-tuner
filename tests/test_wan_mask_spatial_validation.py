"""Tests for WAN cache-latents mask spatial dimension validation (LC-04)."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from musubi_tuner.dataset.image_video_dataset import ItemInfo


class _DummyWanVAE:
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = torch.device("cpu") if device is None else device
        self.dtype = dtype

    def encode(self, contents: torch.Tensor) -> list[torch.Tensor]:
        bsize, _channels, frames, height, width = contents.shape
        lat_h = max(1, height // 8)
        lat_w = max(1, width // 8)
        lat_f = max(1, (frames - 1) // 4 + 1)
        return [torch.zeros(16, lat_f, lat_h, lat_w, device=self.device, dtype=self.dtype) for _ in range(bsize)]


class TestWanMaskSpatialValidation(unittest.TestCase):
    def _make_item(self, h: int, w: int, mask: np.ndarray | None) -> ItemInfo:
        item = MagicMock(spec=ItemInfo)
        item.content = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        item.item_key = "test_image"
        item.original_size = (h, w)
        item.mask_content = mask
        item.control_content = None
        return item

    @patch("musubi_tuner.wan_cache_latents.save_latent_cache_wan")
    def test_mask_mismatch_raises(self, _mock_save: MagicMock) -> None:
        from musubi_tuner.wan_cache_latents import encode_and_save_batch

        vae = _DummyWanVAE()
        mask = np.zeros((32, 32), dtype=np.uint8)
        batch = [self._make_item(64, 64, mask)]

        with self.assertRaises(ValueError) as ctx:
            encode_and_save_batch(vae, None, False, batch)
        self.assertIn("Mask spatial dimensions", str(ctx.exception))

    @patch("musubi_tuner.wan_cache_latents.save_latent_cache_wan")
    def test_mask_wrong_rank_raises(self, _mock_save: MagicMock) -> None:
        from musubi_tuner.wan_cache_latents import encode_and_save_batch

        vae = _DummyWanVAE()
        mask = np.zeros((64, 64, 1), dtype=np.uint8)
        batch = [self._make_item(64, 64, mask)]

        with self.assertRaises(ValueError) as ctx:
            encode_and_save_batch(vae, None, False, batch)
        self.assertIn("2D grayscale", str(ctx.exception))

    @patch("musubi_tuner.wan_cache_latents.save_latent_cache_wan")
    def test_matching_mask_passes(self, _mock_save: MagicMock) -> None:
        from musubi_tuner.wan_cache_latents import encode_and_save_batch

        vae = _DummyWanVAE()
        mask = np.full((64, 64), 255, dtype=np.uint8)
        batch = [self._make_item(64, 64, mask)]

        # Should not raise; dummy VAE + mocked save should allow full path to complete.
        encode_and_save_batch(vae, None, False, batch)


if __name__ == "__main__":
    unittest.main()
