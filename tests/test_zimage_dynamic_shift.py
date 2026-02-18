import math
import pytest

from musubi_tuner.zimage import zimage_config, zimage_utils


def test_compute_dynamic_shift_clamps_to_bounds():
    assert zimage_utils.compute_dynamic_shift(zimage_config.BASE_IMAGE_SEQ_LEN) == pytest.approx(math.exp(zimage_config.BASE_SHIFT))
    assert zimage_utils.compute_dynamic_shift(zimage_config.MAX_IMAGE_SEQ_LEN) == pytest.approx(math.exp(zimage_config.MAX_SHIFT))
    assert zimage_utils.compute_dynamic_shift(0) == pytest.approx(math.exp(zimage_config.BASE_SHIFT))
    assert zimage_utils.compute_dynamic_shift(zimage_config.MAX_IMAGE_SEQ_LEN + 1) == pytest.approx(
        math.exp(zimage_config.MAX_SHIFT)
    )


def test_compute_dynamic_shift_is_monotonic():
    # Increasing seq_len should increase shift (within the clamped range).
    lo = zimage_utils.compute_dynamic_shift(zimage_config.BASE_IMAGE_SEQ_LEN)
    mid = zimage_utils.compute_dynamic_shift((zimage_config.BASE_IMAGE_SEQ_LEN + zimage_config.MAX_IMAGE_SEQ_LEN) // 2)
    hi = zimage_utils.compute_dynamic_shift(zimage_config.MAX_IMAGE_SEQ_LEN)
    assert lo < mid < hi


def test_compute_dynamic_shift_default_1024():
    # For 1024x1024 generation, Z-Image uses 128x128 latents and patch_size=2 => 64x64 tokens => image_seq_len=4096.
    assert zimage_utils.compute_dynamic_shift(4096) == pytest.approx(math.exp(zimage_config.MAX_SHIFT))
