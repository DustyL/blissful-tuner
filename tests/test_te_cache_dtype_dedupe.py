"""Guard the silent-corruption fix in save_text_encoder_output_cache_common.

Without dedupe-by-stripped-base, toggling text-encoder precision (e.g. --fp8_te) and re-caching
keeps BOTH dtype variants of the same logical key in the merged file. bucket.py then strips the
dtype suffix and groups them under one logical key, and the collator silently duplicates the field
per batch sample — corrupted batches with no error. The merge loop must drop existing-file keys
whose dtype-stripped base is already claimed by a fresh-write key, regardless of dtype suffix.
"""

import pytest
import torch
from safetensors.torch import safe_open

from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_common
from musubi_tuner.dataset.image_video_dataset import ItemInfo


def _make_item(tmp_path, item_key="dedupe_item"):
    item = ItemInfo(item_key=item_key, caption="test caption", original_size=(64, 64))
    item.text_encoder_output_cache_path = str(tmp_path / f"{item_key}_i4_te.safetensors")
    return item


def test_recache_at_different_dtype_supersedes_old_variant(tmp_path):
    item = _make_item(tmp_path)
    base_key = "varlen_i4_llm_features"

    # First write: bf16
    sd_bf16 = {f"{base_key}_bfloat16": torch.randn(7, 11, dtype=torch.bfloat16)}
    save_text_encoder_output_cache_common(item, sd_bf16, "ideogram4")

    # Second write at fp8: must supersede the bf16 entry, not coexist with it
    sd_fp8 = {f"{base_key}_float8_e4m3fn": torch.zeros(9, 11, dtype=torch.float8_e4m3fn)}
    save_text_encoder_output_cache_common(item, sd_fp8, "ideogram4")

    with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
        keys = list(f.keys())

    # Exactly one variant of the logical key should remain, and it must be the fresh fp8 one.
    matching = [k for k in keys if k.startswith(base_key)]
    assert matching == [f"{base_key}_float8_e4m3fn"], f"expected only the fresh fp8 variant to survive, got {matching}"


def test_distinct_logical_keys_from_prior_pass_are_preserved(tmp_path):
    """The merge loop's whole purpose: HV-style multi-pass writers (LLM pass then CLIP pass) must
    keep keys from earlier passes whose dtype-stripped base does NOT collide with the fresh write."""
    item = _make_item(tmp_path, item_key="multipass_item")

    # Pass 1: write the LLM keys
    sd_pass1 = {"llm_bfloat16": torch.randn(3, 5, dtype=torch.bfloat16)}
    save_text_encoder_output_cache_common(item, sd_pass1, "ideogram4")

    # Pass 2: write CLIP keys — distinct logical name, must coexist
    sd_pass2 = {"clipL_bfloat16": torch.randn(3, 5, dtype=torch.bfloat16)}
    save_text_encoder_output_cache_common(item, sd_pass2, "ideogram4")

    with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
        keys = set(f.keys())

    assert "llm_bfloat16" in keys, "earlier-pass distinct key was dropped — merge logic over-pruned"
    assert "clipL_bfloat16" in keys


def test_same_dtype_recache_is_idempotent(tmp_path):
    """Re-caching the same key at the same dtype must not duplicate it (sanity, was already true)."""
    item = _make_item(tmp_path, item_key="idempotent_item")
    sd = {"varlen_i4_llm_features_bfloat16": torch.randn(2, 4, dtype=torch.bfloat16)}
    save_text_encoder_output_cache_common(item, sd, "ideogram4")
    save_text_encoder_output_cache_common(item, sd, "ideogram4")

    with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
        keys = list(f.keys())
    assert keys.count("varlen_i4_llm_features_bfloat16") == 1
    assert len(keys) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
