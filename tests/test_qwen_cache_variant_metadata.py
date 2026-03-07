"""Tests for Qwen cache metadata variant correctness."""

from __future__ import annotations

import json
import struct

import torch

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_QWEN_IMAGE_EDIT_FULL,
    ARCHITECTURE_QWEN_IMAGE_FULL,
    ARCHITECTURE_QWEN_IMAGE_LAYERED_FULL,
    save_latent_cache_qwen_image,
    save_text_encoder_output_cache_qwen_image,
)


class FakeItemInfo:
    def __init__(self, cache_path: str):
        self.latent_cache_path = cache_path
        self.text_encoder_output_cache_path = cache_path
        self.item_key = "test_item"
        self.original_size = (64, 64)
        self.frame_count = None
        self.caption = "test caption"
        self.cache_mask_gamma = 1.0
        self.cache_mask_min_weight = 0.0


def _read_safetensors_architecture(path: str) -> str:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header.get("__metadata__", {}).get("architecture", "")


def test_latent_cache_default_architecture(tmp_path):
    path = str(tmp_path / "test_qi.safetensors")
    item = FakeItemInfo(path)
    save_latent_cache_qwen_image(item, torch.randn(16, 1, 8, 8), control_latent=None)
    assert _read_safetensors_architecture(path) == ARCHITECTURE_QWEN_IMAGE_FULL


def test_latent_cache_edit_architecture(tmp_path):
    path = str(tmp_path / "test_qie.safetensors")
    item = FakeItemInfo(path)
    save_latent_cache_qwen_image(
        item, torch.randn(16, 1, 8, 8), control_latent=None, architecture=ARCHITECTURE_QWEN_IMAGE_EDIT_FULL
    )
    assert _read_safetensors_architecture(path) == ARCHITECTURE_QWEN_IMAGE_EDIT_FULL


def test_latent_cache_layered_architecture(tmp_path):
    path = str(tmp_path / "test_qil.safetensors")
    item = FakeItemInfo(path)
    save_latent_cache_qwen_image(
        item, torch.randn(16, 1, 8, 8), control_latent=None, architecture=ARCHITECTURE_QWEN_IMAGE_LAYERED_FULL
    )
    assert _read_safetensors_architecture(path) == ARCHITECTURE_QWEN_IMAGE_LAYERED_FULL


def test_te_cache_edit_architecture(tmp_path):
    path = str(tmp_path / "test_qie_te.safetensors")
    item = FakeItemInfo(path)
    save_text_encoder_output_cache_qwen_image(item, torch.randn(128, 4096), architecture=ARCHITECTURE_QWEN_IMAGE_EDIT_FULL)
    assert _read_safetensors_architecture(path) == ARCHITECTURE_QWEN_IMAGE_EDIT_FULL
