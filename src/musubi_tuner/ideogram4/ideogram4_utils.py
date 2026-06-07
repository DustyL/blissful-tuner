from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from tqdm import tqdm

from blissful_tuner.blissful_logger import BlissfulLogger
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.ideogram4.latent_norm import latent_denorm, latent_norm
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.device_utils import synchronize_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, get_split_weight_filenames

logger = BlissfulLogger(__name__, "green")

_FP8_DTYPES = tuple(d for d in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)) if d is not None)
IDEOGRAM4_PATCH_SIZE = 2


@dataclass
class PrequantizedFp8LoadStats:
    renamed_scales: int = 0
    dropped_comfy_quant: int = 0
    cast_float_tensors: int = 0


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


def _reshape_scale_weight(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 0:
        return scale.reshape(1)
    if scale.ndim == 1 and scale.numel() != 1:
        return scale.reshape(-1, 1)
    return scale


def convert_prequantized_fp8_tensor(
    key: str,
    value: torch.Tensor,
    compute_dtype: torch.dtype,
    stats: Optional[PrequantizedFp8LoadStats] = None,
) -> tuple[Optional[str], Optional[torch.Tensor]]:
    """Normalize one pre-quantized fp8 checkpoint tensor for Blissful's fp8 patcher."""
    if key.endswith(".comfy_quant"):
        if stats is not None:
            stats.dropped_comfy_quant += 1
        return None, None

    if key.endswith(".weight_scale"):
        if stats is not None:
            stats.renamed_scales += 1
        new_key = key.removesuffix(".weight_scale") + ".scale_weight"
        return new_key, _reshape_scale_weight(value).to(dtype=compute_dtype)

    if value.is_floating_point() and not _is_fp8_dtype(value.dtype) and value.dtype != compute_dtype:
        if stats is not None:
            stats.cast_float_tensors += 1
        value = value.to(dtype=compute_dtype)

    return key, value


def expand_weight_files(model_files: Union[str, list[str]]) -> list[str]:
    if isinstance(model_files, str):
        model_files = [model_files]

    expanded_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is None:
            expanded_files.append(model_file)
        else:
            expanded_files.extend(split_filenames)
    return expanded_files


def load_prequantized_fp8_state_dict(
    model_files: Union[str, list[str]],
    *,
    device: Union[str, torch.device] = "cpu",
    compute_dtype: torch.dtype = torch.bfloat16,
    disable_numpy_memmap: bool = False,
) -> tuple[dict[str, torch.Tensor], PrequantizedFp8LoadStats]:
    """Load Ideogram 4 pre-quantized fp8 weights into Blissful's fp8 state-dict layout."""
    device = torch.device(device)
    state_dict: dict[str, torch.Tensor] = {}
    stats = PrequantizedFp8LoadStats()
    model_files = expand_weight_files(model_files)
    logger.info(f"Loading pre-quantized Ideogram 4 fp8 model files: {model_files}")

    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as f:
            for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", unit="key", leave=False):
                value = f.get_tensor(key, device=device)
                new_key, new_value = convert_prequantized_fp8_tensor(key, value, compute_dtype, stats)
                if new_key is None or new_value is None:
                    continue
                state_dict[new_key] = new_value

    synchronize_device(device)
    logger.info(
        "Loaded pre-quantized Ideogram 4 fp8 state dict: "
        f"{len(state_dict)} tensors, {stats.renamed_scales} scales renamed, "
        f"{stats.dropped_comfy_quant} ComfyUI metadata tensors dropped, "
        f"{stats.cast_float_tensors} floating tensors cast to {compute_dtype}."
    )
    return state_dict, stats


def create_ideogram4_transformer(
    *,
    config: Optional[Ideogram4Config] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Ideogram4Transformer:
    with init_empty_weights():
        model = Ideogram4Transformer(config or Ideogram4Config())
        model.to(dtype)
        set_linear_compute_dtype(model, dtype)
    # NOTE: RoPE inv_freq stays float32 across the .to(dtype) above and any later cast/device move —
    # Ideogram4MRoPE self-heals via its _apply() override, so no explicit repair is needed here.
    return model


def _validate_fp8_linears_patched(model: torch.nn.Module) -> tuple[int, int]:
    """Ensure every fp8 Linear received a scale_weight buffer from apply_fp8_monkey_patch.

    strict=True on load catches mis-keyed/orphaned scales, but NOT the case where the shim produced
    zero scales for a genuinely fp8 checkpoint (e.g. a changed scale-key suffix) — that path assigns
    fp8 weights to un-patched Linears and crashes/garbles at the first forward instead of failing
    loudly here. Returns (patched, total_fp8); raises if any fp8 Linear lacks scale_weight.
    """
    fp8_linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and _is_fp8_dtype(m.weight.dtype)]
    patched = sum(hasattr(m, "scale_weight") for m in fp8_linears)
    if fp8_linears and patched != len(fp8_linears):
        raise RuntimeError(
            f"Ideogram 4 fp8 load: {len(fp8_linears)} fp8 Linear layers but only {patched} carry a "
            "scale_weight buffer — scale-key mismatch in the pre-quantized shim?"
        )
    return patched, len(fp8_linears)


def set_linear_compute_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.compute_dtype = dtype


def load_ideogram4_transformer(
    dit_path: Union[str, list[str]],
    *,
    dtype: torch.dtype = torch.bfloat16,
    loading_device: Union[str, torch.device] = "cpu",
    config: Optional[Ideogram4Config] = None,
    disable_numpy_memmap: bool = False,
) -> Ideogram4Transformer:
    """Load an Ideogram 4 DiT from canonical/HF or Comfy pre-quantized fp8 weights."""
    loading_device = torch.device(loading_device)
    model = create_ideogram4_transformer(config=config, dtype=dtype)
    state_dict, stats = load_prequantized_fp8_state_dict(
        dit_path,
        device=loading_device,
        compute_dtype=dtype,
        disable_numpy_memmap=disable_numpy_memmap,
    )

    apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    model.to(device=loading_device)

    patched, fp8_total = _validate_fp8_linears_patched(model)
    logger.info(
        f"Loaded Ideogram 4 transformer from {dit_path}, info={info}, fp8_stats={stats}, fp8_linears_patched={patched}/{fp8_total}"
    )
    return model


def load_ideogram4_autoencoder(
    ae_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
):
    device = torch.device(device)
    ae = flux2_utils.load_ae(ae_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    ae.to(device=device, dtype=dtype)
    ae.eval()
    return ae


def patchify_vae_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"expected VAE latents as B,C,H,W, got {tuple(latents.shape)}")
    batch_size, channels, height, width = latents.shape
    patch_size = IDEOGRAM4_PATCH_SIZE
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"latent height/width must be divisible by {patch_size}: {tuple(latents.shape)}")

    latents = latents.reshape(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size)
    latents = latents.permute(0, 3, 5, 1, 2, 4).contiguous()
    return latents.reshape(batch_size, channels * patch_size * patch_size, height // patch_size, width // patch_size)


def unpatchify_vae_latents(tokens: torch.Tensor, grid_h: Optional[int] = None, grid_w: Optional[int] = None) -> torch.Tensor:
    if tokens.ndim == 4:
        batch_size, channels, height, width = tokens.shape
        if grid_h is not None and height != grid_h:
            raise ValueError(f"token grid height mismatch: got {height}, expected {grid_h}")
        if grid_w is not None and width != grid_w:
            raise ValueError(f"token grid width mismatch: got {width}, expected {grid_w}")
        grid_h = height
        grid_w = width
        tokens = tokens.permute(0, 2, 3, 1).reshape(batch_size, grid_h * grid_w, channels)

    if tokens.ndim != 3:
        raise ValueError(f"expected tokens as B,L,C or B,C,H,W, got {tuple(tokens.shape)}")
    if grid_h is None or grid_w is None:
        raise ValueError("grid_h and grid_w are required when tokens are flattened")

    batch_size = tokens.shape[0]
    patch_size = IDEOGRAM4_PATCH_SIZE
    ae_channels = tokens.shape[-1] // (patch_size * patch_size)
    if ae_channels * patch_size * patch_size != tokens.shape[-1]:
        raise ValueError(f"token channel count must be divisible by {patch_size * patch_size}: {tuple(tokens.shape)}")

    z = tokens.reshape(batch_size, grid_h, grid_w, patch_size, patch_size, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    return z.reshape(batch_size, ae_channels, grid_h * patch_size, grid_w * patch_size)


def encode_pixels_to_raw_vae_tokens(ae: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """Encode [0, 1] pixels through raw FLUX.2 AE modules, bypassing AE BatchNorm.

    WARNING: returns PRE-``latent_norm`` tokens — NOT DiT-ready. The caller (cache writer /
    generation harness) MUST apply Ideogram ``latent_norm`` (shift/scale) before feeding these to
    the DiT, and the inverse ``latent_denorm`` before decode. A cache that stores this output
    directly as "training-ready" trains the DiT on a mis-scaled latent space (silent garbage, no
    crash). See the Phase-2 latent_norm contract in docs/plans/2026-06-07-ideogram4-fork-review.md.
    """
    pixels = pixels.to(device=ae.device, dtype=ae.dtype) * 2.0 - 1.0
    moments = ae.encoder(pixels)
    mean = torch.chunk(moments, 2, dim=1)[0]
    return patchify_vae_latents(mean)


def decode_raw_vae_tokens_to_pixels(
    ae: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    grid_h: Optional[int] = None,
    grid_w: Optional[int] = None,
) -> torch.Tensor:
    """Decode raw Ideogram VAE tokens to [0, 1] pixels, bypassing AE BatchNorm.

    WARNING: expects PRE-``latent_norm`` tokens (raw AE latent space). If you are decoding DiT
    output, apply ``latent_denorm`` FIRST, then pass the result here. Feeding normalized DiT tokens
    straight in skips the denorm and decodes the wrong distribution.
    """
    z = unpatchify_vae_latents(tokens, grid_h=grid_h, grid_w=grid_w).to(device=ae.device, dtype=ae.dtype)
    decoded = ae.decoder(z)
    return ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 0.5).clamp_(0.0, 1.0)


# Cache-writer convention: the ONLY sanctioned producer of DiT-ready (normalized) latents is
# encode_pixels_to_dit_tokens. A cache writer must stamp this key so a reader can tell training-ready
# latents from raw ones. This is a CONVENTION guard, not cryptographic proof (a tensor can't prove it
# was normalized) — but it forces the writer to consciously assert normalization and lets a reader reject
# raw caches. The real enforcement lands when ideogram4_cache_latents.py is built.
LATENT_NORM_METADATA_KEY = "ideogram4_latent_norm_applied"


def encode_pixels_to_dit_tokens(ae: torch.nn.Module, pixels: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Encode [0, 1] pixels to DiT-ready NORMALIZED tokens. The sanctioned training-ready path.

    pixels[0,1] -> raw AE encoder mean (32ch) -> patchify to (pi,pj,c) 128ch -> flatten to (B, L, 128)
    channel-last -> latent_norm. Returns ``(tokens[B, L, 128], grid_h, grid_w)``; grid dims are required to
    invert (decode_dit_tokens_to_pixels) and to build DiT position ids.

    Training convention (mean, RESOLVED): uses the VAE posterior MEAN (chunk[0]) — never a reparameterized
    sample. Confirmed against the fork's TRAINING cache path (ideogram4_cache_latents.py -> encode ->
    chunk(moments, 2)[0] at ideogram4_autoencoder.py:337). Canonical is inference-only so cannot confirm,
    but mean is the intended training-encode convention. Guarded by
    test_encode_dit_tokens_uses_posterior_mean_not_sample.
    """
    grid = encode_pixels_to_raw_vae_tokens(ae, pixels)  # (B, 128, gh, gw), pre-norm
    batch_size, channels, grid_h, grid_w = grid.shape
    tokens = grid.permute(0, 2, 3, 1).reshape(batch_size, grid_h * grid_w, channels)  # (B, L, 128) channel-last
    return latent_norm(tokens), grid_h, grid_w


def decode_dit_tokens_to_pixels(ae: torch.nn.Module, tokens: torch.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
    """Inverse of encode_pixels_to_dit_tokens: latent_denorm -> unpatchify -> raw decoder -> [0, 1] pixels.

    ``tokens`` are normalized DiT tokens (B, L, 128); denorm returns them to raw AE latent space before
    the raw decoder (which must never see normalized tokens).
    """
    raw_tokens = latent_denorm(tokens)
    return decode_raw_vae_tokens_to_pixels(ae, raw_tokens, grid_h=grid_h, grid_w=grid_w)


def assert_latent_norm_applied(metadata: dict) -> None:
    """Cache-writer guard: refuse to persist Ideogram latents that were not produced by the sanctioned
    normalized path. The writer must set ``metadata[LATENT_NORM_METADATA_KEY] == "true"`` (only warranted
    for tokens from encode_pixels_to_dit_tokens). Catches the most likely mistake: caching raw encoder
    output (encode_pixels_to_raw_vae_tokens) as if it were training-ready.
    """
    if metadata.get(LATENT_NORM_METADATA_KEY) != "true":
        raise ValueError(
            f"Ideogram 4 latent cache must set {LATENT_NORM_METADATA_KEY}='true': latents must come from "
            "encode_pixels_to_dit_tokens (latent_norm applied), not the raw VAE helpers."
        )
