from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from musubi_tuner.dataset.cache_io import save_latent_cache_ideogram4
from musubi_tuner.ideogram4.ideogram4_utils import (
    encode_pixels_to_dit_tokens,
    encode_pixels_to_latent_grid,
    grid_to_dit_tokens,
    preflight_ideogram4_latent_cache,
)


class _BoundedEncoder(nn.Module):
    """Mean in [0, 1] (so bf16 cache round-trip stays tight), large logvar (a sampler would be noisy)."""

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixels.shape
        n = batch_size * 32 * (height // 2) * (width // 2)
        mean = ((torch.arange(n, dtype=pixels.dtype) % 64) / 64.0).reshape(batch_size, 32, height // 2, width // 2)
        return torch.cat([mean, torch.full_like(mean, 5.0)], dim=1)


class _FakeAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _BoundedEncoder()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


def _item(tmp_path):
    return SimpleNamespace(
        latent_cache_path=str(tmp_path / "item_i4.safetensors"),
        original_size=(16, 16),
        frame_count=None,
        item_key="item",
    )


def test_latent_grid_matches_dit_tokens():
    ae = _FakeAutoencoder()
    pixels = torch.rand(1, 3, 16, 16)
    grid, gh, gw = encode_pixels_to_latent_grid(ae, pixels)
    tokens, gh2, gw2 = encode_pixels_to_dit_tokens(ae, pixels)

    assert grid.shape == (1, 128, gh, gw) == (1, 128, 4, 4)
    assert (gh, gw) == (gh2, gw2)
    # grid is the channel-first view of the same normalized tokens; flatten must recover them exactly.
    assert torch.allclose(grid_to_dit_tokens(grid), tokens, atol=1e-5)


def test_ideogram4_latent_cache_roundtrip(tmp_path):
    ae = _FakeAutoencoder()
    pixels = torch.rand(1, 3, 16, 16)
    grid, gh, gw = encode_pixels_to_latent_grid(ae, pixels)

    item = _item(tmp_path)
    save_latent_cache_ideogram4(item, grid[0].to(torch.bfloat16))

    # Reload exactly as blissful's shared reader does (load_file), then strip the key like bucket.py.
    sd = load_file(item.latent_cache_path)
    latent_keys = [k for k in sd if k.startswith("latents_")]
    assert latent_keys == [f"latents_{gh}x{gw}_bfloat16"]  # native key, real grid dims
    reloaded = sd[latent_keys[0]]
    assert reloaded.shape == (128, gh, gw)  # 128-ch grid, NOT re-folded to 512 (B3 would)

    # Trainer flatten reproduces the DiT tokens without re-patchify or re-norm.
    tokens_from_cache = grid_to_dit_tokens(reloaded.unsqueeze(0).float())
    tokens_direct, _, _ = encode_pixels_to_dit_tokens(ae, pixels)
    assert torch.allclose(tokens_from_cache, tokens_direct, atol=2e-2)  # atol covers the bf16 cache cast

    # Preflight accepts the freshly written, properly-flagged cache.
    preflight_ideogram4_latent_cache(item.latent_cache_path)


def test_cache_writes_contract_metadata(tmp_path):
    ae = _FakeAutoencoder()
    grid, _, _ = encode_pixels_to_latent_grid(ae, torch.rand(1, 3, 16, 16))
    item = _item(tmp_path)
    save_latent_cache_ideogram4(item, grid[0].to(torch.bfloat16))

    with safe_open(item.latent_cache_path, framework="pt") as f:
        meta = f.metadata()
    assert meta["architecture"] == "ideogram4"
    assert meta["ideogram4_latent_norm_applied"] == "true"
    assert meta["ideogram4_latent_layout"] == "grid_chw"
    assert meta["ideogram4_latent_space"] == "ideogram4_dit_tokens"


def test_preflight_rejects_unflagged_cache(tmp_path):
    # A raw / fork / stale cache that the shared reader would happily load — preflight must reject it.
    path = str(tmp_path / "raw.safetensors")
    save_file({"latents_4x4_bfloat16": torch.zeros(128, 4, 4, dtype=torch.bfloat16)}, path, metadata={"architecture": "ideogram4"})
    with pytest.raises(ValueError, match="ideogram4_latent_norm_applied"):
        preflight_ideogram4_latent_cache(path)


def test_preflight_rejects_wrong_layout(tmp_path):
    path = str(tmp_path / "flat.safetensors")
    save_file(
        {"latents_16x128_bfloat16": torch.zeros(16, 128, dtype=torch.bfloat16)},
        path,
        metadata={"ideogram4_latent_norm_applied": "true", "ideogram4_latent_layout": "BL128"},
    )
    with pytest.raises(ValueError, match="grid_chw"):
        preflight_ideogram4_latent_cache(path)


def test_grid_to_dit_tokens_rejects_wrong_shape():
    with pytest.raises(ValueError, match="128"):
        grid_to_dit_tokens(torch.zeros(1, 64, 4, 4))  # 64 != 128 channels


def test_other_arch_latent_save_has_no_ideogram_metadata_leak(tmp_path):
    # §7 permanent tripwire: a non-Ideogram latent writer (extra_metadata defaults None) must NOT gain any
    # ideogram4_ metadata. Pins the shared save_latent_cache_common change against future splice regressions.
    from musubi_tuner.dataset.cache_io import save_latent_cache_common

    item = SimpleNamespace(
        latent_cache_path=str(tmp_path / "flux2.safetensors"), original_size=(64, 64), frame_count=None, item_key="f2"
    )
    save_latent_cache_common(item, {"latents_8x8_bfloat16": torch.zeros(16, 8, 8, dtype=torch.bfloat16)}, "flux_2_dev")
    with safe_open(item.latent_cache_path, framework="pt") as f:
        meta = f.metadata()
    assert not any(k.startswith("ideogram4_") for k in meta)
    assert set(meta) == {"architecture", "width", "height", "format_version", "cache_mask_gamma", "cache_mask_min_weight"}


_GOOD_META = {
    "ideogram4_latent_norm_applied": "true",
    "ideogram4_latent_layout": "grid_chw",
    "ideogram4_latent_space": "ideogram4_dit_tokens",
}


def test_preflight_rejects_missing_latent_space(tmp_path):
    path = str(tmp_path / "nospace.safetensors")
    meta = {k: v for k, v in _GOOD_META.items() if k != "ideogram4_latent_space"}
    save_file({"latents_4x4_bfloat16": torch.zeros(128, 4, 4, dtype=torch.bfloat16)}, path, metadata=meta)
    with pytest.raises(ValueError, match="latent_space"):
        preflight_ideogram4_latent_cache(path)


def test_preflight_rejects_wrong_tensor_channels(tmp_path):
    # Metadata is perfect, but the tensor is 64-ch, not the 128-ch grid — copied/stale metadata must not pass.
    path = str(tmp_path / "badch.safetensors")
    save_file({"latents_4x4_bfloat16": torch.zeros(64, 4, 4, dtype=torch.bfloat16)}, path, metadata=_GOOD_META)
    with pytest.raises(ValueError, match="128"):
        preflight_ideogram4_latent_cache(path)


def test_preflight_rejects_key_shape_mismatch(tmp_path):
    # Key claims 8x8 but the tensor grid is 4x4.
    path = str(tmp_path / "mismatch.safetensors")
    save_file({"latents_8x8_bfloat16": torch.zeros(128, 4, 4, dtype=torch.bfloat16)}, path, metadata=_GOOD_META)
    with pytest.raises(ValueError, match="grid"):
        preflight_ideogram4_latent_cache(path)


def test_preflight_rejects_multiple_latent_tensors(tmp_path):
    path = str(tmp_path / "multi.safetensors")
    save_file(
        {
            "latents_4x4_bfloat16": torch.zeros(128, 4, 4, dtype=torch.bfloat16),
            "latents_8x8_bfloat16": torch.zeros(128, 8, 8, dtype=torch.bfloat16),
        },
        path,
        metadata=_GOOD_META,
    )
    with pytest.raises(ValueError, match="exactly one"):
        preflight_ideogram4_latent_cache(path)
