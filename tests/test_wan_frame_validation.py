"""Tests for WAN latent caching frame count validation (LC-01).

The WAN VAE requires T=4k+1 frame counts for clean temporal compression.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from musubi_tuner.dataset.image_video_dataset import ItemInfo


class TestFrameCountValidation(unittest.TestCase):
    """Verify T=4k+1 constraint is enforced during latent caching."""

    def _make_batch(self, num_frames: int, h: int = 64, w: int = 64) -> list[ItemInfo]:
        """Create a minimal batch with the given frame count."""
        item = MagicMock(spec=ItemInfo)
        item.content = np.random.randint(0, 255, (num_frames, h, w, 3), dtype=np.uint8)
        item.item_key = "test_video"
        item.original_size = (h, w)
        item.mask_content = None
        item.control_content = None
        return [item]

    def _make_vae(self) -> MagicMock:
        """Create a mock VAE with proper device/dtype."""
        vae = MagicMock()
        vae.device = torch.device("cpu")
        # Use bf16 so CPU autocast stays enabled (avoids noisy warnings).
        vae.dtype = torch.bfloat16
        return vae

    def test_valid_frame_counts_pass(self) -> None:
        """T=4k+1 frame counts (5, 9, 13, ..., 81) should not raise."""
        from musubi_tuner.wan_cache_latents import encode_and_save_batch

        for T in (5, 9, 13, 17, 21, 77, 81, 85):
            batch = self._make_batch(T)
            vae = self._make_vae()  # will fail at encode, but validation should pass first

            # We expect it to get past validation and fail at VAE encode
            # (since we're using a mock VAE). That's fine — the point is it doesn't
            # raise ValueError about frame count.
            try:
                encode_and_save_batch(vae, None, False, batch)
            except ValueError as e:
                self.assertNotIn("T=4k+1", str(e), f"T={T} should be valid but got: {e}")
            except Exception:
                pass  # Any non-ValueError exception means validation passed

    def test_invalid_frame_counts_raise(self) -> None:
        """Non-conforming frame counts should raise ValueError."""
        from musubi_tuner.wan_cache_latents import encode_and_save_batch

        for T in (6, 10, 14, 20, 80, 82):
            batch = self._make_batch(T)
            vae = self._make_vae()

            with self.assertRaises(ValueError, msg=f"T={T} should be rejected") as ctx:
                encode_and_save_batch(vae, None, False, batch)

            self.assertIn("T=4k+1", str(ctx.exception))

    def test_single_frame_always_passes(self) -> None:
        """Single-frame images (T=1) should bypass the video validation."""
        from musubi_tuner.wan_cache_latents import encode_and_save_batch

        batch = self._make_batch(1, h=64, w=64)
        # Single frame: contents will be unsqueeze'd to [B, 1, H, W, C]
        # which after permute is [B, C, 1, H, W]. num_frames=1, and
        # (1-1)%4 == 0, so it passes.
        vae = self._make_vae()

        try:
            encode_and_save_batch(vae, None, False, batch)
        except ValueError as e:
            self.assertNotIn("T=4k+1", str(e), f"Single frame should be valid but got: {e}")
        except Exception:
            pass  # Non-ValueError means validation passed

    def test_escape_hatch_allows_nonconforming(self) -> None:
        """--allow_nonconforming_frames should downgrade error to warning."""
        from musubi_tuner.wan_cache_latents import encode_and_save_batch

        batch = self._make_batch(10)  # 10 is not 4k+1
        vae = self._make_vae()

        # Should NOT raise ValueError when escape hatch is set
        try:
            encode_and_save_batch(vae, None, False, batch, allow_nonconforming_frames=True)
        except ValueError as e:
            self.assertNotIn("T=4k+1", str(e), f"Escape hatch should allow T=10 but got: {e}")
        except Exception:
            pass  # Non-ValueError means validation passed (went on to fail at mock VAE)


if __name__ == "__main__":
    unittest.main()
