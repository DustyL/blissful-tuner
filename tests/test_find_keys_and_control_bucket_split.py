"""Guard the control-batching ragged-stack fix.

The ImageDataset bucket key must always include per-control shapes (and counts) whenever control
data is present, not only when ``no_resize_control`` / ``control_resolution`` is set. Without this,
items with a different number of control images share a batch and the collator ragged-stacks
``latents_control_{i}`` across the batch (crash on shape mismatch, silent misalignment otherwise).

Also covers the new ``find_keys`` safetensors helper that backs the load-side bucket-key build:
returning a SORTED list so the per-control index<->shape pairing is deterministic across cache
files regardless of safetensors header order.
"""

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.utils.safetensors_utils import find_key, find_keys


def _write_st(path, tensors, metadata=None):
    save_file(tensors, str(path), metadata=metadata or {"architecture": "test"})


def test_find_keys_returns_sorted_list(tmp_path):
    """The whole point of find_keys vs find_key: deterministic ordering across files."""
    cache = tmp_path / "ordered.safetensors"
    # Write keys in a deliberately-shuffled order; sorted() must canonicalize.
    _write_st(
        cache,
        {
            "latents_control_2_4x32x32_bfloat16": torch.zeros(1),
            "latents_control_0_4x16x16_bfloat16": torch.zeros(1),
            "latents_control_1_4x24x24_bfloat16": torch.zeros(1),
            "latents_4x64x64_bfloat16": torch.zeros(1),
        },
    )
    keys = find_keys(str(cache), starts_with="latents_control_")
    assert keys == [
        "latents_control_0_4x16x16_bfloat16",
        "latents_control_1_4x24x24_bfloat16",
        "latents_control_2_4x32x32_bfloat16",
    ]


def test_find_keys_empty_when_no_match(tmp_path):
    cache = tmp_path / "no_controls.safetensors"
    _write_st(cache, {"latents_4x64x64_bfloat16": torch.zeros(1)})
    assert find_keys(str(cache), starts_with="latents_control_") == []


def test_find_keys_filters_by_both_prefix_and_suffix(tmp_path):
    cache = tmp_path / "filtered.safetensors"
    _write_st(
        cache,
        {
            "latents_control_0_4x16x16_bfloat16": torch.zeros(1),
            "latents_control_1_4x24x24_float8_e4m3fn": torch.zeros(1, dtype=torch.float8_e4m3fn),
            "siglip_features_bfloat16": torch.zeros(1),
        },
    )
    bf16_controls = find_keys(str(cache), starts_with="latents_control_", ends_with="_bfloat16")
    assert bf16_controls == ["latents_control_0_4x16x16_bfloat16"]


def test_find_key_is_still_first_match(tmp_path):
    """Sanity: pre-existing find_key behavior unchanged — returns SOME match, not necessarily sorted."""
    cache = tmp_path / "single.safetensors"
    _write_st(cache, {"latents_control_0_4x16x16_bfloat16": torch.zeros(1)})
    assert find_key(str(cache), starts_with="latents_control_") == "latents_control_0_4x16x16_bfloat16"
    assert find_key(str(cache), starts_with="latents_nope_") is None


def test_per_control_bucket_key_distinguishes_different_counts(tmp_path):
    """The bug pattern: two cache files with the SAME bucket resolution but DIFFERENT control counts
    must end up with different bucket keys. Pre-fix, only the first control shape (find_key, not
    find_keys) was appended, so items with [ctrl_0] and [ctrl_0, ctrl_1] shared a bucket key and the
    collator ragged-stacked their control latents across the batch."""
    from musubi_tuner.utils.model_utils import strip_dtype_suffix

    one_control = tmp_path / "one.safetensors"
    two_controls = tmp_path / "two.safetensors"
    _write_st(
        one_control,
        {
            "latents_4x64x64_bfloat16": torch.zeros(1),
            "latents_control_0_4x16x16_bfloat16": torch.zeros(1),
        },
    )
    _write_st(
        two_controls,
        {
            "latents_4x64x64_bfloat16": torch.zeros(1),
            "latents_control_0_4x16x16_bfloat16": torch.zeros(1),
            "latents_control_1_4x32x32_bfloat16": torch.zeros(1),
        },
    )

    def reproduce_bucket_key_build(cache_file):
        # Mirrors the (post-fix) load-side logic in ImageDataset.
        base_bucket = (64, 64)
        control_keys = find_keys(str(cache_file), starts_with="latents_control_")
        if control_keys:
            shapes = [strip_dtype_suffix(k) for k in control_keys]
            return tuple(list(base_bucket) + shapes)
        return base_bucket

    key_one = reproduce_bucket_key_build(one_control)
    key_two = reproduce_bucket_key_build(two_controls)

    assert key_one != key_two, "1-control and 2-control items must NOT share a bucket key"
    assert key_one == (64, 64, "latents_control_0_4x16x16")
    assert key_two == (64, 64, "latents_control_0_4x16x16", "latents_control_1_4x32x32")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
