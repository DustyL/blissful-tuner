import re
import unittest
from pathlib import Path


class TestMaskLossDtypeCastRegression(unittest.TestCase):
    def test_mask_weights_are_cast_to_loss_dtype_before_clamp(self) -> None:
        """Guard against regressions when cache stores masks as float16.

        If `mask_weights` is loaded as float16 but `loss` is bfloat16/float32, some CUDA kernels
        will error on mixed dtypes. We require an explicit cast to `loss.dtype` inside
        apply_masked_loss_with_prior() before any math (including clamp/gamma).
        """

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "src/musubi_tuner/modules/mask_loss.py"
        text = path.read_text(encoding="utf-8")

        cast_pat = re.compile(r"mask_weights\s*=\s*mask_weights\.to\(\s*loss\.device\s*,\s*dtype\s*=\s*loss\.dtype\s*\)")
        clamp_pat = re.compile(r"mask_raw_unblurred\s*=\s*mask_weights\.clamp\(")

        cast_match = cast_pat.search(text)
        clamp_match = clamp_pat.search(text)

        self.assertIsNotNone(cast_match, msg=f"Expected dtype cast of mask_weights in {path}")
        self.assertIsNotNone(clamp_match, msg=f"Expected clamp of mask_weights in {path}")

        # Cast must happen before clamp/gamma to avoid dtype collisions and NaNs.
        assert cast_match is not None and clamp_match is not None  # make mypy/ruff happy
        self.assertLess(cast_match.start(), clamp_match.start(), msg="Expected dtype cast before clamp")


if __name__ == "__main__":
    unittest.main()
