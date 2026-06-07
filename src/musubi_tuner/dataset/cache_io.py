from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, TYPE_CHECKING

import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_FRAMEPACK_FULL,
    ARCHITECTURE_FLUX_KONTEXT_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL,
    ARCHITECTURE_IDEOGRAM4_FULL,
    ARCHITECTURE_KANDINSKY5_FULL,
    ARCHITECTURE_QWEN_IMAGE_FULL,
    ARCHITECTURE_WAN_FULL,
    ARCHITECTURE_Z_IMAGE_FULL,
)

# Leaf-module import (ideogram4/__init__ is trivial; ideogram4.constants imports nothing): safe here even
# though importing ideogram4_utils would cycle (it -> flux2_utils -> image_video_dataset -> cache_io).
from musubi_tuner.ideogram4 import constants as ideogram4_constants
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str

if TYPE_CHECKING:
    # Runtime import would be circular (image_video_dataset imports cache_io); these are only
    # referenced in string annotations (from __future__ import annotations), so TYPE_CHECKING suffices.
    from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo

import logging

logger = logging.getLogger(__name__)


# We use simple if-else approach to support multiple architectures.
# Maybe we can use a plugin system in the future.

# the keys of the dict are `<content_type>_FxHxW_<dtype>` for latents
# and `<content_type>_<dtype|mask>` for other tensors


def save_latent_cache(item_info: ItemInfo, latent: torch.Tensor):
    """HunyuanVideo architecture. HunyuanVideo doesn't support I2V and control latents"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_latent_cache_wan(
    item_info: ItemInfo,
    latent: torch.Tensor,
    clip_embed: Optional[torch.Tensor],
    image_latent: Optional[torch.Tensor],
    control_latent: Optional[torch.Tensor],
    f_indices: Optional[list[int]] = None,
    mask_weights: Optional[torch.Tensor] = None,
):
    """Wan architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if clip_embed is not None:
        sd[f"clip_{dtype_str}"] = clip_embed.detach().cpu()

    if image_latent is not None:
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if control_latent is not None:
        sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu()

    if f_indices is not None:
        dtype_str = dtype_to_str(torch.int32)
        sd[f"f_indices_{dtype_str}"] = torch.tensor(f_indices, dtype=torch.int32)

    if mask_weights is not None:
        # Save mask weights in latent space dimensions (F, H, W) as float16 to reduce cache size / I/O.
        # Mask weights originate from 8-bit masks and are clamped to [0,1], so float16 is sufficient.
        # F = number of video frames (WAN/HV) or number of layers (Qwen-Image Layered).
        # Single transfer: detach → device/dtype conversion (avoids redundant copies)
        mask_dtype_str = dtype_to_str(torch.float16)
        sd[f"mask_weights_{F}x{H}x{W}_{mask_dtype_str}"] = mask_weights.detach().to(device="cpu", dtype=torch.float16)

    save_latent_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_latent_cache_framepack(
    item_info: ItemInfo,
    latent: torch.Tensor,
    latent_indices: torch.Tensor,
    clean_latents: torch.Tensor,
    clean_latent_indices: torch.Tensor,
    clean_latents_2x: torch.Tensor,
    clean_latent_2x_indices: torch.Tensor,
    clean_latents_4x: torch.Tensor,
    clean_latent_4x_indices: torch.Tensor,
    image_embeddings: torch.Tensor,
):
    """FramePack architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    # `latents_xxx` must have {F, H, W} suffix
    indices_dtype_str = dtype_to_str(latent_indices.dtype)
    sd[f"image_embeddings_{dtype_str}"] = image_embeddings.detach().cpu()  # image embeddings dtype is same as latents dtype
    sd[f"latent_indices_{indices_dtype_str}"] = latent_indices.detach().cpu()
    sd[f"clean_latent_indices_{indices_dtype_str}"] = clean_latent_indices.detach().cpu()
    sd[f"latents_clean_{F}x{H}x{W}_{dtype_str}"] = clean_latents.detach().cpu().contiguous()
    if clean_latent_2x_indices is not None:
        sd[f"clean_latent_2x_indices_{indices_dtype_str}"] = clean_latent_2x_indices.detach().cpu()
    if clean_latents_2x is not None:
        sd[f"latents_clean_2x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_2x.detach().cpu().contiguous()
    if clean_latent_4x_indices is not None:
        sd[f"clean_latent_4x_indices_{indices_dtype_str}"] = clean_latent_4x_indices.detach().cpu()
    if clean_latents_4x is not None:
        sd[f"latents_clean_4x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_4x.detach().cpu().contiguous()

    # for key, value in sd.items():
    #     print(f"{key}: {value.shape}")
    save_latent_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_latent_cache_flux_kontext(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: torch.Tensor,
):
    """FLUX.1 Kontext architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    _, H, W = control_latent.shape
    F = 1
    sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_latent_cache_flux_2(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: Optional[list[torch.Tensor]],
    arch_full: str,
    mask_weights: Optional[torch.Tensor] = None,
):
    """Flux 2 architecture with optional mask weights for mask-weighted loss training.

    Args:
        item_info: The item info containing cache path and metadata.
        latent: The latent tensor (C, H, W).
        control_latent: Optional list of control latent tensors.
        arch_full: The full architecture name (e.g., 'flux_2_dev', 'flux_2_klein_4b').
        mask_weights: Optional mask weights for mask-weighted loss training.
    """
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"
    assert control_latent is None or all(cl.dim() == 3 for cl in control_latent), (
        "control_latent should be 3D tensor (channel, height, width) or None"
    )

    _, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, H, W = cl.shape
            sd[f"latents_control_{i}_{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    if mask_weights is not None:
        # Save mask weights in latent space dimensions (1, 1, H, W) as float16 to reduce cache size / I/O.
        # Shape matches layout="video" with F=1: (B, 1, F, H, W) -> per-item (1, 1, H, W)
        _, H, W = latent.shape
        mask_dtype_str = dtype_to_str(torch.float16)
        sd[f"mask_weights_{H}x{W}_{mask_dtype_str}"] = mask_weights.detach().to(device="cpu", dtype=torch.float16)

    save_latent_cache_common(item_info, sd, arch_full)


def save_latent_cache_ideogram4(item_info: ItemInfo, latent: torch.Tensor, arch_full: str = ARCHITECTURE_IDEOGRAM4_FULL):
    """Ideogram 4: persist the already-patchified + latent_norm'd DiT-token GRID as (128, gh, gw) under the
    native key latents_{gh}x{gw}_{dtype}, so blissful's grid-native reader loads it unchanged. The trainer
    flattens (ideogram4_utils.grid_to_dit_tokens) and must NOT patchify or latent_norm again. Metadata flags
    mark it training-ready; the shared reader ignores metadata, so an Ideogram-specific preflight
    (ideogram4_utils.preflight_ideogram4_latent_cache) enforces them before training.
    """
    assert latent.dim() == 3 and latent.shape[0] == 128, (
        f"Ideogram 4 latent must be the (128, gh, gw) token grid, got {tuple(latent.shape)}"
    )
    _, gh, gw = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{gh}x{gw}_{dtype_str}": latent.detach().cpu().contiguous()}

    extra_metadata = {
        ideogram4_constants.IDEOGRAM4_LATENT_NORM_METADATA_KEY: "true",
        ideogram4_constants.IDEOGRAM4_LATENT_LAYOUT_KEY: ideogram4_constants.IDEOGRAM4_LATENT_LAYOUT_GRID_CHW,
        ideogram4_constants.IDEOGRAM4_LATENT_SPACE_KEY: ideogram4_constants.IDEOGRAM4_LATENT_SPACE_DIT_TOKENS,
    }
    save_latent_cache_common(item_info, sd, arch_full, extra_metadata=extra_metadata)


def save_latent_cache_qwen_image(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: Optional[list[torch.Tensor]],
    mask_weights: Optional[torch.Tensor] = None,
    architecture: str = ARCHITECTURE_QWEN_IMAGE_FULL,
):
    """Qwen-Image architecture with optional mask weights for mask-weighted loss training."""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"
    assert control_latent is None or all(cl.dim() == 4 for cl in control_latent), (
        "control_latent should be 4D tensor (frame, channel, height, width) or None"
    )

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, F, H, W = cl.shape
            sd[f"latents_control_{i}_{F}x{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    if mask_weights is not None:
        # Save mask weights in latent space dimensions (1, F, H, W) as float16 to reduce cache size / I/O.
        # F = 1 for standard/Edit images; for Layered, F = number of layers (mask is expanded
        # identically across all layers — per-layer masks are not currently supported).
        # Single transfer: detach → device/dtype conversion (avoids redundant copies)
        _, F, H, W = latent.shape
        mask_dtype_str = dtype_to_str(torch.float16)
        sd[f"mask_weights_{F}x{H}x{W}_{mask_dtype_str}"] = mask_weights.detach().to(device="cpu", dtype=torch.float16)

    save_latent_cache_common(item_info, sd, architecture)


def save_latent_cache_kandinsky5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor] = None,
    control_latent: Optional[torch.Tensor] = None,
    scaling_factor: Optional[float] = None,
):
    """Kandinsky 5 architecture (image/video), with optional source/control latents for i2v/control."""
    assert latent.dim() == 3 or latent.dim() == 4, "latent should be 3D (C,H,W) or 4D (F,C,H,W) tensor"

    if latent.dim() == 4:
        _, F, H, W = latent.shape
    else:
        F, H, W = 1, latent.shape[1], latent.shape[2]
        latent = latent.unsqueeze(0)
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous().clone()}

    if image_latent is not None:
        _, F_img, H_img, W_img = image_latent.shape
        sd[f"latents_image_{F_img}x{H_img}x{W_img}_{dtype_str}"] = image_latent.detach().cpu().contiguous().clone()

    if control_latent is not None:
        _, F_ctrl, H_ctrl, W_ctrl = control_latent.shape
        sd[f"latents_control_{F_ctrl}x{H_ctrl}x{W_ctrl}_{dtype_str}"] = control_latent.detach().cpu().contiguous().clone()

    if scaling_factor is not None:
        sd["vae_scaling_factor"] = torch.tensor(float(scaling_factor))

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_latent_cache_hunyuan_video_1_5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor],
    vision_feature: Optional[torch.Tensor],
):
    """HunyuanVideo 1.5 architecture"""
    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd: dict[str, torch.Tensor] = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if image_latent is not None:
        dtype_str = dtype_to_str(image_latent.dtype)
        _, F, H, W = image_latent.shape
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if vision_feature is not None:
        dtype_str = dtype_to_str(vision_feature.dtype)
        sd[f"siglip_{dtype_str}"] = vision_feature.detach().cpu()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_latent_cache_z_image(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latents: Optional[List[torch.Tensor]] = None,
    siglip_features: Optional[List[torch.Tensor]] = None,
    mask_weights: Optional[torch.Tensor] = None,
):
    """
    Z-Image architecture cache saver.

    Standard mode (existing behavior):
        latent: [C, H, W] target image latent
        control_latents: None
        siglip_features: None

    OmniBase mode (new):
        latent: [C, H, W] target image latent
        control_latents: List of [C, H, W] control image latents
        siglip_features: List of [H_sig, W_sig, D_sig] SigLIP2 features

    Mask-weighted loss (optional):
        mask_weights: Optional mask weights for weighted loss training. Expected to be in latent space
        resolution and saved as float32 for precision.

    Args:
        item_info: Item metadata for cache path
        latent: Target image latent [C, H, W]
        control_latents: Optional list of control image latents (OmniBase)
        siglip_features: Optional list of SigLIP2 features (OmniBase)
        mask_weights: Optional mask weights tensor (see above)
    """
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    # Validate 1:1 relationship between control_latents and siglip_features if both provided
    if control_latents is not None and siglip_features is not None:
        assert len(control_latents) == len(siglip_features), (
            f"control_latents and siglip_features must have same length: "
            f"got {len(control_latents)} control latents and {len(siglip_features)} siglip features"
        )

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)

    # Base cache dict (unchanged for backward compatibility)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    # OmniBase additions (new keys only when provided)
    if control_latents is not None:
        for i, ctrl in enumerate(control_latents):
            assert ctrl.dim() == 3, f"control_latent[{i}] should be 3D tensor (channel, height, width)"
            _, ctrl_H, ctrl_W = ctrl.shape
            ctrl_dtype = dtype_to_str(ctrl.dtype)
            sd[f"latents_control_{i}_{F}x{ctrl_H}x{ctrl_W}_{ctrl_dtype}"] = ctrl.detach().cpu().contiguous()

    if siglip_features is not None:
        for i, sig in enumerate(siglip_features):
            assert sig.dim() == 3, f"siglip_features[{i}] should be 3D tensor [H, W, C]"
            sig_dtype = dtype_to_str(sig.dtype)
            sd[f"siglip_{i}_{sig_dtype}"] = sig.detach().cpu().contiguous()

    if mask_weights is not None:
        # Save mask weights in latent space dimensions as float16 to reduce cache size / I/O.
        # Common convention in this repo is per-item mask shape like (1, 1, H, W) (then stacked to (B, 1, 1, H, W)).
        mask_dtype_str = dtype_to_str(torch.float16)
        sd[f"mask_weights_{F}x{H}x{W}_{mask_dtype_str}"] = mask_weights.detach().to(device="cpu", dtype=torch.float16)

    save_latent_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_latent_cache_common(
    item_info: ItemInfo, sd: dict[str, torch.Tensor], arch_fullname: str, extra_metadata: Optional[dict[str, str]] = None
):
    metadata = {
        "architecture": arch_fullname,
        "width": f"{item_info.original_size[0]}",
        "height": f"{item_info.original_size[1]}",
        "format_version": "1.0.1",
    }
    if item_info.frame_count is not None:
        metadata["frame_count"] = f"{item_info.frame_count}"

    # Record cache-time mask preprocessing parameters for transparency and training-time safety checks.
    #
    # Store as compact, human-friendly floats (still parseable by float()).
    gamma = getattr(item_info, "cache_mask_gamma", None)
    min_weight = getattr(item_info, "cache_mask_min_weight", None)
    gamma = 1.0 if gamma is None else float(gamma)
    min_weight = 0.0 if min_weight is None else float(min_weight)
    metadata["cache_mask_gamma"] = format(gamma, ".6g")
    metadata["cache_mask_min_weight"] = format(min_weight, ".6g")

    for key, value in sd.items():
        # 1) Ensure contiguous FIRST to avoid overlapping memory issues on expanded views
        if not value.is_contiguous():
            value = value.contiguous()
            sd[key] = value

        # 2) NaN check (float/complex only - torch.isnan on int tensors varies by torch version)
        if value.dtype.is_floating_point or value.dtype.is_complex:
            if torch.isnan(value).any():
                logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replacing NaN with 0")
                value[torch.isnan(value)] = 0

    # Architecture-specific metadata (additive). Empty/None for every existing arch => byte-identical
    # metadata; only Ideogram 4 currently passes it (latent_norm_applied / latent_layout / latent_space).
    if extra_metadata:
        metadata.update(extra_metadata)

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)

    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache(item_info: ItemInfo, embed: torch.Tensor, mask: Optional[torch.Tensor], is_llm: bool):
    """HunyuanVideo architecture"""
    assert embed.dim() == 1 or embed.dim() == 2, (
        f"embed should be 2D tensor (feature, hidden_size) or (hidden_size,), got {embed.shape}"
    )
    assert mask is None or mask.dim() == 1, f"mask should be 1D tensor (feature), got {mask.shape}"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "llm" if is_llm else "clipL"
    sd[f"{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()
    if mask is not None:
        sd[f"{text_encoder_type}_mask"] = mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_text_encoder_output_cache_wan(item_info: ItemInfo, embed: torch.Tensor):
    """Wan architecture. Wan2.1 only has a single text encoder"""

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "t5"
    sd[f"varlen_{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_text_encoder_output_cache_framepack(
    item_info: ItemInfo, llama_vec: torch.Tensor, llama_attention_mask: torch.Tensor, clip_l_pooler: torch.Tensor
):
    """FramePack architecture."""
    sd = {}
    dtype_str = dtype_to_str(llama_vec.dtype)
    sd[f"llama_vec_{dtype_str}"] = llama_vec.detach().cpu()
    sd["llama_attention_mask"] = llama_attention_mask.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_text_encoder_output_cache_flux_kontext(item_info: ItemInfo, t5_vec: torch.Tensor, clip_l_pooler: torch.Tensor):
    """Flux Kontext architecture."""

    sd = {}
    dtype_str = dtype_to_str(t5_vec.dtype)
    sd[f"t5_vec_{dtype_str}"] = t5_vec.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_text_encoder_output_cache_flux_2(
    item_info: ItemInfo, ctx_vec: torch.Tensor, arch_full: str, ctx_seq_len: Optional[torch.Tensor] = None
):
    """Flux 2 architecture.

    Args:
        item_info: The item info containing cache path and metadata.
        ctx_vec: The context vector from text encoder.
        arch_full: The full architecture name (e.g., 'flux_2_dev', 'flux_2_klein_4b').
        ctx_seq_len: Optional scalar int32 tensor with the real (non-padded) token count.
    """
    sd = {}
    dtype_str = dtype_to_str(ctx_vec.dtype)
    sd[f"ctx_vec_{dtype_str}"] = ctx_vec.detach().cpu()

    if ctx_seq_len is not None:
        sd["ctx_seq_len_int32"] = ctx_seq_len.detach().cpu().to(torch.int32)

    save_text_encoder_output_cache_common(item_info, sd, arch_full)


def save_text_encoder_output_cache_qwen_image(
    item_info: ItemInfo, embed: torch.Tensor, architecture: str = ARCHITECTURE_QWEN_IMAGE_FULL
):
    """Qwen-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, architecture)


def save_text_encoder_output_cache_kandinsky5(
    item_info: ItemInfo, text_embeds: torch.Tensor, pooled_embed: torch.Tensor, attention_mask: torch.Tensor
):
    """Kandinsky 5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(text_embeds.dtype)
    sd[f"text_embeds_{dtype_str}"] = text_embeds.detach().cpu()
    dtype_str = dtype_to_str(pooled_embed.dtype)
    sd[f"pooled_embed_{dtype_str}"] = pooled_embed.detach().cpu()
    sd["attention_mask"] = attention_mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_text_encoder_output_cache_hunyuan_video_1_5(item_info: ItemInfo, embed: torch.Tensor, byt5_embed: torch.Tensor):
    """Hunyuan-Video 1.5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()
    dtype_str = dtype_to_str(byt5_embed.dtype)
    sd[f"varlen_byt5_embed_{dtype_str}"] = byt5_embed.detach().cpu()
    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_text_encoder_output_cache_z_image(item_info: ItemInfo, embed: torch.Tensor):
    """Z-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_llm_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_text_encoder_output_cache_common(item_info: ItemInfo, sd: dict[str, torch.Tensor], arch_fullname: str):
    for key, value in sd.items():
        # 1) Ensure contiguous FIRST to avoid overlapping memory issues on expanded views
        if not value.is_contiguous():
            value = value.contiguous()
            sd[key] = value

        # 2) NaN check (float/complex only - torch.isnan on int tensors varies by torch version)
        if value.dtype.is_floating_point or value.dtype.is_complex:
            if torch.isnan(value).any():
                logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replacing NaN with 0")
                value[torch.isnan(value)] = 0

    metadata = {
        "architecture": arch_fullname,
        "caption1": item_info.caption,
        "format_version": "1.0.1",
    }

    if os.path.exists(item_info.text_encoder_output_cache_path):
        # load existing cache and update metadata
        with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                if key not in sd:  # avoid overwriting by existing cache, we keep the new one
                    sd[key] = f.get_tensor(key)

        assert existing_metadata["architecture"] == metadata["architecture"], "architecture mismatch"
        if existing_metadata["caption1"] != metadata["caption1"]:
            logger.warning(f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite")
        # TODO verify format_version

        existing_metadata.pop("caption1", None)
        existing_metadata.pop("format_version", None)
        metadata.update(existing_metadata)  # copy existing metadata except caption and format_version
    else:
        text_encoder_output_dir = os.path.dirname(item_info.text_encoder_output_cache_path)
        os.makedirs(text_encoder_output_dir, exist_ok=True)

    safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


# --- blissful-tuner additions (re-homed from monolith) ---


def has_omnibase_cache(cache_path: str) -> bool:
    """
    Check if a Z-Image cache file contains OmniBase data (control latents/SigLIP features).

    Args:
        cache_path: Path to the .safetensors cache file

    Returns:
        True if cache contains OmniBase data (latents_control_* or siglip_* keys)
    """
    try:
        from safetensors import safe_open

        with safe_open(cache_path, framework="pt") as f:
            keys = f.keys()
            return any(k.startswith("latents_control_") or k.startswith("siglip_") for k in keys)
    except Exception:
        return False


def read_cache_mask_transform_metadata(cache_path: str) -> tuple[float | None, float | None]:
    """Read cache-time mask preprocessing parameters from a latent cache safetensors file.

    Returns:
        (cache_mask_gamma, cache_mask_min_weight) as floats if present, otherwise (None, None).
    """
    try:
        with safetensors_utils.MemoryEfficientSafeOpen(cache_path) as f:
            metadata = f.metadata()
    except Exception:  # noqa: BLE001
        return None, None

    gamma_str = metadata.get("cache_mask_gamma")
    min_weight_str = metadata.get("cache_mask_min_weight")
    if gamma_str is None or min_weight_str is None:
        return None, None

    try:
        return float(gamma_str), float(min_weight_str)
    except Exception:  # noqa: BLE001
        return None, None


def scan_cache_mask_transform_metadata(
    datasets: Sequence["BaseDataset"],
    *,
    max_files_per_dataset: int = 16,
) -> tuple[set[tuple[float, float]], int, int]:
    """Scan a small sample of latent cache files and collect cache-time mask transforms.

    This is intended to be called once at training startup (not per-step) to provide
    transparency and to prevent common traps (double application, prior threshold vs min_weight).

    Returns:
        (pairs, num_files_with_metadata, num_files_checked)
    """
    pairs: set[tuple[float, float]] = set()
    num_files_checked = 0
    num_files_with_metadata = 0

    for ds in datasets:
        cache_dir = getattr(ds, "cache_directory", None)
        arch = getattr(ds, "architecture", None)
        if not cache_dir or not arch:
            continue

        pattern = os.path.join(cache_dir, f"*_{arch}.safetensors")
        sampled = 0
        for cache_path in glob.iglob(pattern):
            if sampled >= max_files_per_dataset:
                break
            sampled += 1
            num_files_checked += 1

            gamma, min_weight = read_cache_mask_transform_metadata(cache_path)
            if gamma is None or min_weight is None:
                continue
            num_files_with_metadata += 1
            pairs.add((gamma, min_weight))

    return pairs, num_files_with_metadata, num_files_checked
