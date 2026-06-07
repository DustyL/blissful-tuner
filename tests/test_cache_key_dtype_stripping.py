import pytest
import torch

from musubi_tuner.utils.model_utils import dtype_to_str, strip_dtype_suffix


@pytest.mark.parametrize(
    "key,expected",
    [
        # Normal single-segment dtypes (the path that already worked).
        ("t5_bfloat16", "t5"),
        ("t5_float16", "t5"),
        ("t5_float32", "t5"),
        ("ctx_seq_len_int32", "ctx_seq_len"),
        # H5 bug: multi-underscore fp8 dtype. rsplit("_", 1) would wrongly yield "i4_llm_features_float8".
        ("i4_llm_features_float8_e4m3fn", "i4_llm_features"),
        ("i4_llm_features_float8_e5m2", "i4_llm_features"),
        # latents_/mask_weights_ keep their resolution segment (the caller strips that separately, via 'x').
        ("latents_16x16_bfloat16", "latents_16x16"),
        ("mask_weights_64x64_float16", "mask_weights_64x64"),
    ],
)
def test_strip_dtype_suffix(key, expected):
    assert strip_dtype_suffix(key) == expected


def test_strip_dtype_suffix_is_longest_first():
    # 'bfloat16' must match before 'float16', else 'x_bfloat16' -> 'x_b' (a stray 'b').
    assert strip_dtype_suffix("x_bfloat16") == "x"


def test_strip_dtype_suffix_fnuz_variants_when_available():
    if hasattr(torch, "float8_e4m3fnuz"):
        assert strip_dtype_suffix("w_float8_e4m3fnuz") == "w"
        # The non-uz key must NOT be mistaken for the longer uz variant (or vice versa).
        assert strip_dtype_suffix("w_float8_e4m3fn") == "w"


def test_strip_dtype_suffix_no_dtype_unchanged():
    assert strip_dtype_suffix("ctx_seq_len") == "ctx_seq_len"
    assert strip_dtype_suffix("i4_llm_features") == "i4_llm_features"
    assert strip_dtype_suffix("plain") == "plain"


def test_strip_dtype_suffix_covers_every_cacheable_dtype():
    # Whatever dtype_to_str emits for a cacheable dtype, "{stem}_{that}" must strip cleanly back to the stem.
    for d in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.float8_e4m3fn, torch.float8_e5m2):
        assert strip_dtype_suffix(f"stem_{dtype_to_str(d)}") == "stem"
