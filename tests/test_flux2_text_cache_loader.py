"""Dataset-level FLUX.2 text cache tests that do not import the full FLUX.2 runtime."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_FLUX_2_DEV_FULL,
    BucketBatchManager,
    ItemInfo,
    save_text_encoder_output_cache_flux_2,
)


def _make_item_info(cache_path: str, caption: str = "test") -> ItemInfo:
    item = ItemInfo(item_key="test_item", caption=caption, original_size=(1024, 1024), bucket_size=(64, 64), frame_count=1)
    item.text_encoder_output_cache_path = cache_path
    return item


def test_cache_round_trip_with_ctx_seq_len():
    """New FLUX.2 caches should persist ctx_seq_len with a dtype suffix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache_flux_2_dev_te.safetensors")
        item = _make_item_info(cache_path, "a photo of a cat")

        ctx_vec = torch.randn(512, 15360, dtype=torch.bfloat16)
        ctx_seq_len = torch.tensor(73, dtype=torch.int32)

        save_text_encoder_output_cache_flux_2(item, ctx_vec, arch_full=ARCHITECTURE_FLUX_2_DEV_FULL, ctx_seq_len=ctx_seq_len)

        assert os.path.exists(cache_path)
        loaded = load_file(cache_path)

        assert "ctx_seq_len_int32" in loaded
        assert loaded["ctx_seq_len_int32"].dtype == torch.int32
        assert loaded["ctx_seq_len_int32"].item() == 73


def test_cache_round_trip_without_ctx_seq_len():
    """Old-style FLUX.2 caches without ctx_seq_len should still save cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache_flux_2_dev_te.safetensors")
        item = _make_item_info(cache_path, "a photo of a dog")

        ctx_vec = torch.randn(512, 15360, dtype=torch.bfloat16)

        save_text_encoder_output_cache_flux_2(item, ctx_vec, arch_full=ARCHITECTURE_FLUX_2_DEV_FULL)

        loaded = load_file(cache_path)
        assert "ctx_seq_len" not in loaded
        assert "ctx_seq_len_int32" not in loaded
        assert "ctx_vec_bfloat16" in loaded


@pytest.mark.parametrize("seq_len_key", ["ctx_seq_len_int32", "ctx_seq_len"])
def test_bucket_batch_manager_loads_ctx_seq_len(seq_len_key: str):
    """BucketBatchManager should surface ctx_seq_len for current and legacy cache keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        latent_path = os.path.join(tmpdir, "test_item_latent.safetensors")
        te_path = os.path.join(tmpdir, "test_item_te.safetensors")
        item = _make_item_info(te_path, "a photo of a fox")
        item.latent_cache_path = latent_path

        save_file({"latents_1x2x2_bf16": torch.randn(16, 1, 2, 2, dtype=torch.bfloat16)}, latent_path)
        save_file(
            {
                "ctx_vec_bfloat16": torch.randn(512, 15360, dtype=torch.bfloat16),
                seq_len_key: torch.tensor(73, dtype=torch.int32),
            },
            te_path,
        )

        batch = BucketBatchManager({(64, 64): [item]}, batch_size=1)[0]

        assert "ctx_seq_len" in batch
        assert batch["ctx_seq_len"].dtype == torch.int32
        assert batch["ctx_seq_len"].shape == (1,)
        assert batch["ctx_seq_len"].item() == 73
