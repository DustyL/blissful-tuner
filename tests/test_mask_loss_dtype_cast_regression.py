import re
import unittest
from pathlib import Path


class TestMaskLossDtypeCastRegression(unittest.TestCase):
    def test_mask_weights_are_cast_to_float32_before_clamp(self) -> None:
        """Guard against regressions when cache stores masks as float16.

        Mask processing (gamma, min_weight, blur) must happen in float32 for
        numerical precision. The mask is cast to float32 early, then narrowed
        to loss.dtype only at the broadcast-multiply points to avoid mixed-
        precision collisions on CUDA.
        """

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "src/musubi_tuner/modules/mask_loss.py"
        text = path.read_text(encoding="utf-8")

        # 1. Early cast to float32 for processing precision
        cast_pat = re.compile(
            r"mask_weights\s*=\s*mask_weights\.to\(\s*device\s*=\s*loss\.device\s*,\s*dtype\s*=\s*torch\.float32\s*\)"
        )
        clamp_pat = re.compile(r"mask_raw_unblurred\s*=\s*mask_weights\.clamp\(")

        cast_match = cast_pat.search(text)
        clamp_match = clamp_pat.search(text)

        self.assertIsNotNone(cast_match, msg=f"Expected float32 dtype cast of mask_weights in {path}")
        self.assertIsNotNone(clamp_match, msg=f"Expected clamp of mask_weights in {path}")

        # Cast must happen before clamp/gamma to avoid dtype collisions and NaNs.
        assert cast_match is not None and clamp_match is not None  # make mypy/ruff happy
        self.assertLess(cast_match.start(), clamp_match.start(), msg="Expected dtype cast before clamp")

    def test_mask_narrowed_to_loss_dtype_at_multiply(self) -> None:
        """Verify mask_processed is cast to loss.dtype at the broadcast multiply point.

        Processing stays in float32, but the final multiply against the loss
        tensor must use loss.dtype to avoid mixed-precision kernel errors.
        """

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "src/musubi_tuner/modules/mask_loss.py"
        text = path.read_text(encoding="utf-8")

        # Target loss multiply should cast mask_processed to loss.dtype
        target_cast_pat = re.compile(r"loss\s*\*\s*mask_processed\.to\(\s*dtype\s*=\s*loss\.dtype\s*\)")
        self.assertIsNotNone(
            target_cast_pat.search(text),
            msg=f"Expected mask_processed.to(dtype=loss.dtype) at target loss multiply in {path}",
        )

        # Prior loss multiply should cast prior_mask to loss.dtype
        prior_cast_pat = re.compile(r"prior_loss_unreduced\s*\*\s*prior_mask\.to\(\s*dtype\s*=\s*loss\.dtype\s*\)")
        self.assertIsNotNone(
            prior_cast_pat.search(text),
            msg=f"Expected prior_mask.to(dtype=loss.dtype) at prior loss multiply in {path}",
        )


if __name__ == "__main__":
    unittest.main()
