import pytest

from musubi_tuner.zimage import zimage_config, zimage_utils


def test_compute_dynamic_shift_clamps_to_bounds():
    assert zimage_utils.compute_dynamic_shift(zimage_config.BASE_IMAGE_SEQ_LEN) == zimage_config.BASE_SHIFT
    assert zimage_utils.compute_dynamic_shift(zimage_config.MAX_IMAGE_SEQ_LEN) == zimage_config.MAX_SHIFT
    assert zimage_utils.compute_dynamic_shift(0) == zimage_config.BASE_SHIFT
    assert zimage_utils.compute_dynamic_shift(zimage_config.MAX_IMAGE_SEQ_LEN + 1) == zimage_config.MAX_SHIFT


def test_compute_dynamic_shift_default_1024():
    # 1024 is the default 1024x1024 image seq len (32x32) for Z-Image.
    assert zimage_utils.compute_dynamic_shift(1024) == pytest.approx(0.63)
