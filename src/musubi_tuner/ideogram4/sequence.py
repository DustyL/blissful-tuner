"""Shared Ideogram 4 joint-sequence construction (used by both training and generation).

Faithful to canonical pipeline_ideogram4.py::_build_packed_sequence, but built from cached Qwen3-VL text
FEATURES (not re-tokenized prompts). Per item the layout is ``[pad][text][image]``: text is LEFT-padded to
``max_text_tokens``, so the image tokens occupy a FIXED trailing slice ``[max_text : max_text+num_image]`` for
every item (image_start = max_text_tokens) — making the image scatter (x) and the DiT-output gather batch-uniform.

The DiT consumes the full sequence: ``forward(x, t, llm_features, position_ids, segment_ids, indicator)`` where
x is image-latent tokens (masked to OUTPUT_IMAGE positions) and llm_features is text features (masked to
LLM_TOKEN positions). Only OUTPUT_IMAGE positions carry a meaningful prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from musubi_tuner.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)

# Image-space tokens per side = pixels / (VAE downscale 8 * patch_size 2) = pixels / 16.
IDEOGRAM4_IMAGE_PATCH = 16


def image_grid_dims(height: int, width: int) -> tuple[int, int]:
    """Pixel (height, width) -> token grid (grid_h, grid_w). Must be divisible by 16 (VAE 8 * patch 2)."""
    if height % IDEOGRAM4_IMAGE_PATCH != 0 or width % IDEOGRAM4_IMAGE_PATCH != 0:
        raise ValueError(f"height/width must be divisible by {IDEOGRAM4_IMAGE_PATCH}, got {height}x{width}")
    return height // IDEOGRAM4_IMAGE_PATCH, width // IDEOGRAM4_IMAGE_PATCH


@dataclass
class Ideogram4Sequence:
    """Static joint-sequence conditioning. Image tokens (x) are scattered separately (build_image_input)."""

    llm_features: torch.Tensor  # (B, total, feature_dim) text features at text positions, 0 elsewhere
    position_ids: torch.Tensor  # (B, total, 3) MRoPE (t, h, w)
    segment_ids: torch.Tensor  # (B, total) sample id (1) vs padding (SEQUENCE_PADDING_INDICATOR)
    indicator: torch.Tensor  # (B, total) LLM_TOKEN_INDICATOR / OUTPUT_IMAGE_INDICATOR / 0 (pad)
    image_start: int  # first index of the image-token slice (== max_text_tokens)
    num_image_tokens: int
    grid_h: int
    grid_w: int

    @property
    def total_seq_len(self) -> int:
        return self.image_start + self.num_image_tokens


def build_ideogram4_conditioning(
    text_features: list[torch.Tensor],
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Ideogram4Sequence:
    """Build the static conditioning (llm_features / position_ids / segment_ids / indicator) for one batch.

    text_features: list of B tensors, each (L_text_i, feature_dim). grid_h/grid_w: image token grid.
    """
    if not text_features:
        raise ValueError("text_features must be non-empty")
    device = torch.device(device) if device is not None else text_features[0].device
    batch_size = len(text_features)
    feature_dim = text_features[0].shape[-1]
    num_image_tokens = grid_h * grid_w
    max_text_tokens = max(int(f.shape[0]) for f in text_features)
    total = max_text_tokens + num_image_tokens

    # Image MRoPE positions (t=0, h, w) offset to stay disjoint from the small text positions.
    h_idx = torch.arange(grid_h, device=device).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w, device=device).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    image_pos = torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET  # (num_image, 3)

    llm_features = torch.zeros(batch_size, total, feature_dim, dtype=dtype, device=device)
    position_ids = torch.zeros(batch_size, total, 3, dtype=torch.long, device=device)
    segment_ids = torch.full((batch_size, total), SEQUENCE_PADDING_INDICATOR, dtype=torch.long, device=device)
    indicator = torch.zeros(batch_size, total, dtype=torch.long, device=device)

    for b, features in enumerate(text_features):
        num_text = int(features.shape[0])
        if num_text > max_text_tokens:
            raise ValueError("internal: num_text exceeds max_text_tokens")
        offset = max_text_tokens - num_text  # left-pad so image starts at max_text_tokens for every item

        text_pos = torch.arange(num_text, device=device)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)  # text MRoPE: same index on all 3 axes

        llm_features[b, offset:max_text_tokens] = features.to(device=device, dtype=dtype)
        position_ids[b, offset:max_text_tokens] = text_pos_3d
        position_ids[b, max_text_tokens:] = image_pos
        indicator[b, offset:max_text_tokens] = LLM_TOKEN_INDICATOR
        indicator[b, max_text_tokens:] = OUTPUT_IMAGE_INDICATOR
        segment_ids[b, offset:] = 1  # text + image belong to the sample; [0:offset] stays padding

    return Ideogram4Sequence(
        llm_features=llm_features,
        position_ids=position_ids,
        segment_ids=segment_ids,
        indicator=indicator,
        image_start=max_text_tokens,
        num_image_tokens=num_image_tokens,
        grid_h=grid_h,
        grid_w=grid_w,
    )


def build_image_input(image_tokens: torch.Tensor, total_seq_len: int, image_start: int) -> torch.Tensor:
    """Scatter image-latent tokens (B, num_image, C) into x (B, total, C) at [image_start:], 0 before.

    Non-image positions are masked by the DiT (output_image_mask after input_proj), so their value is
    irrelevant; zeros keep it clean. Used per denoise step (generate) and once (train).
    """
    if image_tokens.dim() != 3:
        raise ValueError(f"image_tokens must be (B, num_image, C), got {tuple(image_tokens.shape)}")
    batch_size, num_image, channels = image_tokens.shape
    if image_start + num_image != total_seq_len:
        raise ValueError(f"image slice [{image_start}:{image_start + num_image}] does not end at total {total_seq_len}")
    x = torch.zeros(batch_size, total_seq_len, channels, dtype=image_tokens.dtype, device=image_tokens.device)
    x[:, image_start:] = image_tokens
    return x


def extract_image_tokens(sequence_output: torch.Tensor, image_start: int, num_image_tokens: int) -> torch.Tensor:
    """Gather the image-token slice (B, num_image, C) from a full-sequence DiT output (B, total, C)."""
    return sequence_output[:, image_start : image_start + num_image_tokens]
