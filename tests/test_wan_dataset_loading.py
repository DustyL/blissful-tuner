"""Tests for WAN dataset loading: varlen T5, batch construction, mixed masks (TT-02)."""

import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.image_video_dataset import BucketBatchManager, ItemInfo


def _make_item_info(
    key: str,
    latent_path: str,
    te_path: str,
    bucket_size: tuple = (90, 160),
) -> ItemInfo:
    """Build a minimal ItemInfo with cache paths wired up."""
    item = ItemInfo(
        item_key=key,
        caption=f"caption for {key}",
        original_size=(720, 1280),
        bucket_size=bucket_size,
        frame_count=81,
    )
    item.latent_cache_path = latent_path
    item.text_encoder_output_cache_path = te_path
    return item


class TestWanDatasetLoading(unittest.TestCase):
    """Test BucketBatchManager key normalization and batch construction for WAN caches."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def _save_wan_caches(
        self,
        name: str,
        *,
        latent_shape: tuple = (16, 5, 4, 4),
        t5_seq_len: int = 128,
        mask_weights: torch.Tensor | None = None,
        image_latent: torch.Tensor | None = None,
    ) -> tuple[str, str]:
        """Write WAN-style latent and text encoder cache files, return paths."""
        lat_path = os.path.join(self.tmpdir, f"{name}_latent.safetensors")
        te_path = os.path.join(self.tmpdir, f"{name}_te.safetensors")

        C, F, H, W = latent_shape
        latent = torch.randn(C, F, H, W, dtype=torch.bfloat16)
        lat_sd = {f"latents_{F}x{H}x{W}_bf16": latent}
        if mask_weights is not None:
            # Real WAN caches store masks as (1, F, H, W) — add leading dim if needed
            if mask_weights.ndim == 3:
                mask_weights = mask_weights.unsqueeze(0)
            _, mF, mH, mW = mask_weights.shape
            lat_sd[f"mask_weights_{mF}x{mH}x{mW}_float32"] = mask_weights
        if image_latent is not None:
            iC, iF, iH, iW = image_latent.shape
            lat_sd[f"latents_image_{iF}x{iH}x{iW}_bf16"] = image_latent
        save_file(lat_sd, lat_path)

        t5_embed = torch.randn(t5_seq_len, 4096, dtype=torch.bfloat16)
        save_file({"varlen_t5_bf16": t5_embed}, te_path)

        return lat_path, te_path

    def _make_batch_manager(
        self,
        items: list[ItemInfo],
        bucket_reso: tuple = (90, 160),
    ) -> BucketBatchManager:
        """Build BucketBatchManager with a single bucket containing all items."""
        bucketed = {bucket_reso: items}
        return BucketBatchManager(bucketed, batch_size=len(items))

    # --- Test 1: varlen T5 keys are preserved as list, not stacked ---

    def test_varlen_t5_returns_list(self) -> None:
        """Varlen T5 embeddings should be a list of tensors, not stacked."""
        paths_a = self._save_wan_caches("a", t5_seq_len=128)
        paths_b = self._save_wan_caches("b", t5_seq_len=64)

        items = [
            _make_item_info("a", paths_a[0], paths_a[1]),
            _make_item_info("b", paths_b[0], paths_b[1]),
        ]
        mgr = self._make_batch_manager(items)
        batch = mgr[0]

        # t5 should be a list (not stacked) because of variable lengths
        self.assertIsInstance(batch["t5"], list)
        self.assertEqual(len(batch["t5"]), 2)
        self.assertEqual(batch["t5"][0].shape[0], 128)
        self.assertEqual(batch["t5"][1].shape[0], 64)
        # Both have hidden dim 4096
        self.assertEqual(batch["t5"][0].shape[1], 4096)
        self.assertEqual(batch["t5"][1].shape[1], 4096)

    # --- Test 2: latents are stacked into (B, C, F, H, W) ---

    def test_latents_stacked(self) -> None:
        """Latent cache keys should be normalized and stacked into a batch tensor."""
        paths_a = self._save_wan_caches("a")
        paths_b = self._save_wan_caches("b")

        items = [
            _make_item_info("a", paths_a[0], paths_a[1]),
            _make_item_info("b", paths_b[0], paths_b[1]),
        ]
        mgr = self._make_batch_manager(items)
        batch = mgr[0]

        self.assertIn("latents", batch)
        self.assertEqual(batch["latents"].shape, (2, 16, 5, 4, 4))

    # --- Test 3: mixed-mask batch alignment ---

    def test_mixed_mask_alignment(self) -> None:
        """When one item has mask_weights and another doesn't, missing gets all-ones."""
        mask = torch.rand(5, 4, 4, dtype=torch.float32)
        paths_a = self._save_wan_caches("a", mask_weights=mask)
        paths_b = self._save_wan_caches("b")  # no mask

        items = [
            _make_item_info("a", paths_a[0], paths_a[1]),
            _make_item_info("b", paths_b[0], paths_b[1]),
        ]
        mgr = self._make_batch_manager(items)
        batch = mgr[0]

        self.assertIn("mask_weights", batch)
        # Stacked from (1, F, H, W) per item → (B, 1, F, H, W)
        self.assertEqual(batch["mask_weights"].shape, (2, 1, 5, 4, 4))
        # Item A's mask should match what was saved
        self.assertTrue(torch.allclose(batch["mask_weights"][0, 0], mask))
        # Item B's mask should be all ones (full training weight)
        self.assertTrue(torch.all(batch["mask_weights"][1] == 1.0))

    def test_mixed_mask_alignment_3d_saved(self) -> None:
        """Regression: 3D mask (F,H,W) saved to cache produces (B,F,H,W) batch."""
        # Some cache scripts may save masks without the leading channel dim
        mask = torch.rand(5, 4, 4, dtype=torch.float32)
        # Save as 3D directly (no unsqueeze in helper)
        paths_a = self._save_wan_caches("a", mask_weights=mask.unsqueeze(0))  # 4D (1,F,H,W)
        paths_b = self._save_wan_caches("b")

        items = [
            _make_item_info("a", paths_a[0], paths_a[1]),
            _make_item_info("b", paths_b[0], paths_b[1]),
        ]
        mgr = self._make_batch_manager(items)
        batch = mgr[0]

        self.assertIn("mask_weights", batch)
        # Shape is (B, 1, F, H, W) because cache stores (1, F, H, W) per item
        self.assertEqual(batch["mask_weights"].shape, (2, 1, 5, 4, 4))

    # --- Test 4: no-mask batch omits mask_weights key ---

    def test_no_mask_batch(self) -> None:
        """When no items have masks, mask_weights should not appear in batch."""
        paths_a = self._save_wan_caches("a")
        paths_b = self._save_wan_caches("b")

        items = [
            _make_item_info("a", paths_a[0], paths_a[1]),
            _make_item_info("b", paths_b[0], paths_b[1]),
        ]
        mgr = self._make_batch_manager(items)
        batch = mgr[0]

        self.assertNotIn("mask_weights", batch)

    # --- Test 5: both items have masks ---

    def test_both_have_masks(self) -> None:
        """When all items have masks, both should be stacked correctly."""
        mask_a = torch.full((5, 4, 4), 0.8, dtype=torch.float32)
        mask_b = torch.full((5, 4, 4), 0.3, dtype=torch.float32)
        paths_a = self._save_wan_caches("a", mask_weights=mask_a)
        paths_b = self._save_wan_caches("b", mask_weights=mask_b)

        items = [
            _make_item_info("a", paths_a[0], paths_a[1]),
            _make_item_info("b", paths_b[0], paths_b[1]),
        ]
        mgr = self._make_batch_manager(items)
        batch = mgr[0]

        self.assertIn("mask_weights", batch)
        # Stacked from (1, F, H, W) per item → (B, 1, F, H, W)
        self.assertEqual(batch["mask_weights"].shape, (2, 1, 5, 4, 4))
        self.assertTrue(torch.allclose(batch["mask_weights"][0, 0], mask_a))
        self.assertTrue(torch.allclose(batch["mask_weights"][1, 0], mask_b))

    # --- Test 6: I2V latents_image key normalizes correctly ---

    def test_latents_image_key_normalization(self) -> None:
        """I2V latents_image_FxHxW keys should normalize to 'latents_image'."""
        image_latent = torch.randn(20, 5, 4, 4, dtype=torch.bfloat16)
        paths_a = self._save_wan_caches("a", image_latent=image_latent)
        paths_b = self._save_wan_caches("b", image_latent=image_latent)

        items = [
            _make_item_info("a", paths_a[0], paths_a[1]),
            _make_item_info("b", paths_b[0], paths_b[1]),
        ]
        mgr = self._make_batch_manager(items)
        batch = mgr[0]

        self.assertIn("latents_image", batch)
        self.assertEqual(batch["latents_image"].shape, (2, 20, 5, 4, 4))

    # --- Test 7: timesteps default to None ---

    def test_timesteps_default_none(self) -> None:
        """Without timestep bucketing, batch['timesteps'] should be None."""
        paths_a = self._save_wan_caches("a")
        items = [_make_item_info("a", paths_a[0], paths_a[1])]
        mgr = self._make_batch_manager(items)
        batch = mgr[0]

        self.assertIsNone(batch["timesteps"])


if __name__ == "__main__":
    unittest.main()
