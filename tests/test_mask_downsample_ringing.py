import re
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class TestMaskDownsampleRinging(unittest.TestCase):
    def test_lanczos_introduces_halo_vs_box_on_hard_mask(self) -> None:
        """Lanczos is a sinc-like filter and can ring on hard-edged masks.

        For loss-weight masks, this ringing shows up as non-zero "halo" weights in regions that
        should remain exactly 0 under local area averaging.

        This test uses an 8x8-aligned hard block so BOX (area average) downsampling is exact:
          - exactly one latent pixel is 255
          - all others are 0
        Lanczos introduces a halo of intermediate non-zero values around it.
        """

        w = 256
        h = 256
        scale = 8
        out_w, out_h = w // scale, h // scale

        mask = np.zeros((h, w), dtype=np.uint8)
        # 8x8 block aligned to the 8x downsample grid
        mask[h // 2 : h // 2 + scale, w // 2 : w // 2 + scale] = 255

        mask_pil = Image.fromarray(mask, mode="L")

        box = np.array(mask_pil.resize((out_w, out_h), resample=Image.BOX), dtype=np.uint8)
        lanczos = np.array(mask_pil.resize((out_w, out_h), resample=Image.LANCZOS), dtype=np.uint8)

        # BOX should preserve exact locality for this aligned block.
        self.assertEqual(int((box > 0).sum()), 1)
        self.assertEqual(int(box.max()), 255)

        # Lanczos should spread weights into a halo (non-zero where BOX is exactly zero).
        self.assertGreater(int((lanczos > 0).sum()), int((box > 0).sum()))
        leak = np.logical_and(box == 0, lanczos > 0)
        self.assertGreater(int(leak.sum()), 0)

        # Lanczos should also introduce intermediate tier values (blur), not just {0,255}.
        has_intermediate_values = bool(np.any(np.logical_and(lanczos != 0, lanczos != 255)))
        self.assertTrue(has_intermediate_values)

    def test_key_mask_downsamples_use_box_not_lanczos(self) -> None:
        """Guard against regressions: latent-space mask downsample must not use Lanczos."""

        repo_root = Path(__file__).resolve().parents[1]
        targets = [
            repo_root / "src/musubi_tuner/cache_latents.py",
            repo_root / "src/musubi_tuner/fpack_generate_video.py",
            repo_root / "src/musubi_tuner/fpack_train_network.py",
        ]

        lanczos_pat = re.compile(r"resize\(\(width\s*//\s*8,\s*height\s*//\s*8\),\s*Image\.LANCZOS\)")
        box_pat = re.compile(r"resize\(\(width\s*//\s*8,\s*height\s*//\s*8\),\s*Image\.BOX\)")

        for path in targets:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(lanczos_pat.search(text), msg=f"Found Image.LANCZOS latent mask downsample in {path}")
            self.assertIsNotNone(box_pat.search(text), msg=f"Expected Image.BOX latent mask downsample in {path}")


if __name__ == "__main__":
    unittest.main()
