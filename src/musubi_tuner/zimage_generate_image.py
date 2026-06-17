import argparse
import gc
import random
import os
import time
import copy
import math
from typing import Tuple, Optional, List, Any, Dict

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm

from musubi_tuner.utils import model_utils
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.utils.lora_utils import (
    convert_diffusers_if_needed,
    detect_network_type,
    filter_lora_state_dict,
    format_unknown_network_type_error,
)
from musubi_tuner.zimage import zimage_config, zimage_model, zimage_utils
from musubi_tuner.zimage import zimage_autoencoder
from musubi_tuner.zimage.zimage_autoencoder import AutoencoderKL


from musubi_tuner.utils.cli_compat import add_lycoris_arg, validate_lycoris_arg

from musubi_tuner.networks import lora_zimage
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.hv_generate_video import get_time_flag, save_images_grid, setup_parser_compile, synchronize_device
from musubi_tuner.wan_generate_video import merge_lora_weights

from blissful_tuner.latent_preview import LatentPreviewer

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype  # not used currently because model may be optimized


def apply_cfg_normalization(pos_pred: torch.Tensor, guided_pred: torch.Tensor, cfg_normalization: float) -> torch.Tensor:
    """Apply CFG normalization by rescaling guided prediction to not exceed the positive prediction norm.

    This matches the common "CFG normalization/rescale" behavior in official pipelines:
    - Compute norms of pos_pred and guided_pred (per-sample).
    - If guided norm exceeds (pos norm * cfg_normalization), rescale guided to that max norm.
    """
    if cfg_normalization <= 0:
        raise ValueError("cfg_normalization must be > 0")

    # Compute norms in float32 for numerical stability (avoid fp16/bf16 overflow/underflow).
    pos_f = pos_pred.float()
    guided_f = guided_pred.float()

    reduce_dims = tuple(range(1, guided_f.ndim))
    pos_norm = torch.linalg.vector_norm(pos_f, dim=reduce_dims, keepdim=True)
    guided_norm = torch.linalg.vector_norm(guided_f, dim=reduce_dims, keepdim=True)

    max_guided_norm = pos_norm * float(cfg_normalization)
    eps = 1e-6
    scale = torch.minimum(torch.ones_like(guided_norm), max_guided_norm / (guided_norm + eps))

    return guided_pred * scale.to(dtype=guided_pred.dtype)


def resolve_text_encoder_weight_dtype(args: argparse.Namespace, llm_device: torch.device) -> torch.dtype:
    """Resolve text encoder weight dtype from CLI args.

    Notes:
    - FP8 LLM weights are only meaningful on CUDA; if the text encoder runs on CPU, fall back to bf16.
    - Z-Image text encoder inference defaults to bf16 for parity with official code.
    """
    use_fp8 = bool(getattr(args, "fp8_llm", False))
    if use_fp8 and llm_device.type != "cuda":
        logger.warning("--fp8_llm requested but text encoder is not running on CUDA; falling back to bfloat16.")
        use_fp8 = False

    return torch.float8_e4m3fn if use_fp8 else torch.bfloat16


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="Z-Image inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument(
        "--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors. Default is False."
    )
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text Encoder 1 (Qwen2.5-VL) directory or path")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument(
        "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Guidance scale for classifier free guidance. Default is 0.0 (no guidance).",
    )
    parser.add_argument(
        "--cfg_truncation",
        type=float,
        default=zimage_config.DEFAULT_CFG_TRUNCATION,
        help=(
            "CFG truncation threshold in [0, 1]. When the normalized noise level (sigma) is greater than this value, "
            "CFG is disabled for that step (effective guidance scale becomes 0). "
            f"Default is {zimage_config.DEFAULT_CFG_TRUNCATION} (CFG at all steps)."
        ),
    )
    parser.add_argument(
        "--cfg_normalization",
        type=float,
        nargs="?",
        const=1.0,
        default=None,
        help=(
            "Enable CFG normalization (rescale guided prediction norm to match positive prediction norm). "
            "Pass an optional multiplier (e.g., 1.0). If specified without a value, defaults to 1.0. "
            "Default is disabled."
        ),
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="negative prompt for generation")
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[zimage_config.DEFAULT_HEIGHT, zimage_config.DEFAULT_WIDTH],
        help=f"image size as height width, must be divisible by 16 (default: {zimage_config.DEFAULT_HEIGHT} {zimage_config.DEFAULT_WIDTH})",
    )
    parser.add_argument("--infer_steps", type=int, default=25, help="number of inference steps, default is 25")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated image(s)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=None,
        help="DEPRECATED: no effect for Z-Image (accepted for CLI compatibility only).",
    )

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="Shift factor for flow matching schedulers. Default is 3.0.",
    )
    parser.add_argument(
        "--dynamic_shift",
        action="store_true",
        help=(
            "Auto-compute flow_shift from resolution using FLUX-style dynamic shifting. "
            f"Maps image seq len [{zimage_config.BASE_IMAGE_SEQ_LEN}, {zimage_config.MAX_IMAGE_SEQ_LEN}] "
            f"to mu [{zimage_config.BASE_SHIFT}, {zimage_config.MAX_SHIFT}] (clamped), then uses flow_shift = exp(mu) "
            f"(~[{math.exp(zimage_config.BASE_SHIFT):.2f}, {math.exp(zimage_config.MAX_SHIFT):.2f}]). "
            "Overrides --flow_shift."
        ),
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for language model")
    parser.add_argument("--text_encoder_cpu", action="store_true", help="Inference on CPU for Text Encoder (Qwen2.5-VL)")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "cute", "torch", "sageattn", "xformers", "sdpa"],  #  "flash2", "flash3",
        help="attention mode",
    )
    parser.add_argument(
        "--use_32bit_attention",
        action="store_true",
        help="use 32-bit precision for attention computations in DiT model even when using mixed precision (original behavior)",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned memory for block swapping, which may speed up data transfer between CPU and GPU but uses more shared GPU memory on Windows",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
        help="output type",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")

    # Blissful: latent preview during denoising
    parser.add_argument(
        "--preview_latent_every",
        type=int,
        default=None,
        help="Enable latent preview every N steps. If --preview_vae is not specified it will use latent2rgb",
    )
    parser.add_argument("--preview_vae", type=str, default=None, help="Path to TAE vae for previews")

    add_lycoris_arg(parser)
    setup_parser_compile(parser)

    # arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")
    parser.add_argument(
        "--bell",
        action="store_true",
        help="Ring bell when done. For interactive mode, ring bell on each iteration. For other modes, ring bell at the end.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")

    if args.cfg_truncation < 0.0 or args.cfg_truncation > 1.0:
        raise ValueError("--cfg_truncation must be in range [0, 1]")
    if args.cfg_normalization is not None and args.cfg_normalization <= 0:
        raise ValueError("--cfg_normalization must be > 0")

    if args.embedded_cfg_scale is not None:
        logger.warning("--embedded_cfg_scale is deprecated for Z-Image and is ignored.")

    validate_lycoris_arg(args)

    # Phase 1 hotswap rejections (see docs/plans/2026-05-04-peft-tier1c-zimage-qwenimage-hotswap.md):
    # --prepare_for_hotswap is incompatible with --prefer_lycoris (separate merge bridge),
    # with --fp8_scaled (registers per-layer scale_weight buffers that hotswap would not refresh),
    # with raw --fp8 (standard Z-Image path merges in bf16 BEFORE casting; a later hotswap into
    # already-fp8 params would discard precision and not be parity-shaped),
    # with --save_merged_model (one-shot offline workflow, not live sweep),
    # and with --latent_path (decode-only mode does not load the DiT).
    if getattr(args, "prepare_for_hotswap", False):
        if args.prefer_lycoris:
            raise ValueError(
                "--prepare_for_hotswap is incompatible with --prefer_lycoris in Z-Image Phase 1. "
                "LyCORIS uses its own merge path; omit --prepare_for_hotswap to use the standard merge."
            )
        if args.fp8_scaled:
            raise ValueError(
                "--prepare_for_hotswap is incompatible with --fp8_scaled in Z-Image Phase 1. "
                "FP8-scaled optimization registers per-layer scale_weight buffers that the hotswap "
                "path would not refresh, producing silently wrong outputs. "
                "Omit --prepare_for_hotswap to use the standard merge path with FP8."
            )
        if args.fp8:
            raise ValueError(
                "--prepare_for_hotswap is incompatible with --fp8 in Z-Image Phase 1. "
                "The standard Z-Image path merges LoRA in bf16 then casts to fp8; hotswap into "
                "already-fp8 params would not be bit-equivalent and would discard base precision before merge. "
                "Omit --prepare_for_hotswap to use the standard merge path with FP8."
            )
        if args.save_merged_model:
            raise ValueError(
                "--prepare_for_hotswap is incompatible with --save_merged_model. "
                "Hotswap is for live inference sweeps; saving the merged model is a one-shot operation."
            )
        if args.latent_path is not None and len(args.latent_path) > 0:
            raise ValueError(
                "--prepare_for_hotswap is incompatible with --latent_path. "
                "Latent-only decode mode does not load the DiT, so hotswap state cannot be prepared. "
                "Omit --prepare_for_hotswap when decoding pre-computed latents."
            )

    return args


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    # TODO common function with hv_train_network.line_to_prompt_dict
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Create dictionary of overrides
    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        # Map options to argument names
        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "image_size_width":
            args_copy.image_size[1] = value
        elif key == "image_size_height":
            args_copy.image_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int]:
    """Validate image size

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int]: (height, width)
    """
    height = args.image_size[0]
    width = args.image_size[1]

    if height % (zimage_config.ZIMAGE_VAE_SCALE_FACTOR * 2) != 0 or width % (zimage_config.ZIMAGE_VAE_SCALE_FACTOR * 2) != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

    return height, width


# region DiT model


def prepare_zimage_hotswap_state(model: "zimage_model.ZImageTransformer2DModel", args: argparse.Namespace) -> Optional[Any]:
    """Capture un-merged base for compile-friendly LoRA hotswap.

    MUST be called immediately after the Z-Image model load returns and BEFORE
    the initial LoRA merge, so the cached base never contains a LoRA delta.

    Always sets `model.hotswap_state` (None when off, state when on) so
    downstream code can rely on `getattr(model, "hotswap_state", None)`.
    """
    model.hotswap_state = None
    if not getattr(args, "prepare_for_hotswap", False):
        return None

    from musubi_tuner.utils.lora_utils import prepare_for_hotswap as _prepare_for_hotswap

    state = _prepare_for_hotswap(
        model,
        dit_paths=[args.dit],
        base_weights_paths=None,
        base_weights_multipliers=None,
        cache_in_ram=args.cache_unmerged_base,
        strict_base_hash=args.hotswap_strict_base_hash,
    )
    model.hotswap_state = state
    return state


def note_zimage_initial_loras(model: "zimage_model.ZImageTransformer2DModel", args: argparse.Namespace) -> None:
    """Record the initial active LoRA set on the hotswap state.

    No-op when hotswap is off. When on, records the LoRAs just merged via the
    standard post-load path so a later hotswap_lora() knows what's currently
    applied. Mirrors WAN convention of padding multipliers to 1.0.
    """
    if not getattr(args, "prepare_for_hotswap", False):
        return
    state = getattr(model, "hotswap_state", None)
    if state is None:
        raise RuntimeError("Z-Image hotswap_state missing after prepare; call prepare_zimage_hotswap_state first")
    lora_paths = list(args.lora_weight or [])
    multipliers = list(args.lora_multiplier or [])
    if len(multipliers) < len(lora_paths):
        multipliers = multipliers + [1.0] * (len(lora_paths) - len(multipliers))
    multipliers = multipliers[: len(lora_paths)]
    state.active_lora_paths = lora_paths
    state.active_lora_multipliers = multipliers


def load_dit_model(
    args: argparse.Namespace, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None
) -> zimage_model.ZImageTransformer2DModel:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use
        dit_weight_dtype: data type for the model weights. None for as-is

    Returns:
        zimage_model.ZImageTransformer2DModel: DiT model instance
    """
    # If LyCORIS is enabled, we will load the model to CPU and then merge LoRA weights (static method)

    prepare_for_hotswap = getattr(args, "prepare_for_hotswap", False)

    loading_device = "cpu"
    if args.blocks_to_swap == 0 and not args.prefer_lycoris:
        loading_device = device

    # Hotswap suppresses LoRA preload; we apply the initial LoRA via the post-load merge path
    # AFTER the un-merged base is captured. Same pattern as WAN/FLUX.2 hotswap.
    # See docs/plans/2026-05-04-peft-tier1c-zimage-qwenimage-hotswap.md.
    if prepare_for_hotswap:
        lora_weights_list = None
    elif not args.prefer_lycoris and args.lora_weight is not None and len(args.lora_weight) > 0:
        # load LoRA weights
        lora_weights_list = []
        for i, lora_weight in enumerate(args.lora_weight):
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            net_type = detect_network_type(lora_sd)
            if net_type == "unknown":
                raise ValueError(format_unknown_network_type_error(lora_weight))
            # Convert Diffusers-format keys to default format before merge
            lora_sd = convert_diffusers_if_needed(lora_sd)
            include_pat = args.include_patterns[i] if args.include_patterns and len(args.include_patterns) > i else None
            exclude_pat = args.exclude_patterns[i] if args.exclude_patterns and len(args.exclude_patterns) > i else None
            lora_sd = filter_lora_state_dict(lora_sd, include_pat, exclude_pat)
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled and not args.prefer_lycoris:
        loading_weight_dtype = None  # we will load weights as-is and then optimize to fp8
    elif args.prefer_lycoris:
        loading_weight_dtype = torch.bfloat16  # lycoris requires bfloat16 or float16, because it merges weights

    model = zimage_model.load_zimage_model(
        device,
        args.dit,
        args.attn_mode,
        False,
        loading_device,
        loading_weight_dtype,
        args.fp8_scaled and not args.prefer_lycoris,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
        disable_numpy_memmap=args.disable_numpy_memmap,
        use_16bit_for_attention=not args.use_32bit_attention,
    )

    # ===== HOTSWAP: capture un-merged base BEFORE the initial LoRA merge =====
    # Always sets model.hotswap_state (None or state). When hotswap is on, the LoRA
    # preload above was suppressed; we now apply the initial active LoRA set via the
    # standard post-load merge path with standard_lora_only=True. This preserves the
    # no-accumulation invariant — the cached base reflects the permanent base only.
    prepare_zimage_hotswap_state(model, args)

    if prepare_for_hotswap and args.lora_weight is not None and len(args.lora_weight) > 0:
        merge_lora_weights(
            lora_zimage,
            model,
            args.lora_weight,
            args.lora_multiplier,
            args.include_patterns,
            args.exclude_patterns,
            device,
            lycoris=False,
            save_merged_model=None,
            standard_lora_only=True,
        )
        note_zimage_initial_loras(model, args)
    # ===== END HOTSWAP =====

    # merge LoRA weights
    if args.prefer_lycoris:
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            merge_lora_weights(
                lora_zimage,
                model,
                args.lora_weight,
                args.lora_multiplier,
                args.include_patterns,
                args.exclude_patterns,
                device,
                lycoris=True,
                save_merged_model=args.save_merged_model,
                extra_unet_targets=["ZImageTransformerBlock"],
            )

        if args.fp8_scaled:
            # load state dict as-is and optimize to fp8
            state_dict = model.state_dict()

            # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
            move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
            # state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)

            from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

            # inplace optimization
            state_dict = optimize_state_dict_with_fp8(
                state_dict,
                device,
                zimage_model.FP8_OPTIMIZATION_TARGET_KEYS,
                zimage_model.FP8_OPTIMIZATION_EXCLUDE_KEYS,
                move_to_device=move_to_device,
            )
            apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)  # args.scaled_mm)

            info = model.load_state_dict(state_dict, strict=True, assign=True)
            logger.info(f"Loaded FP8 optimized weights: {info}")

    # if we only want to save the model, we can skip the rest
    if args.save_merged_model:
        if not args.prefer_lycoris:
            # Non-LyCORIS path: save was not handled by merge_lora_weights
            from musubi_tuner.utils.safetensors_utils import mem_eff_save_file

            logger.info(f"Saving merged model to {args.save_merged_model}")
            mem_eff_save_file(model.state_dict(), args.save_merged_model)
            logger.info("Merged model saved")
        return None

    if not args.fp8_scaled:
        # simple cast to dit_weight_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if dit_weight_dtype is not None:  # in case of args.fp8 and not args.fp8_scaled
            logger.info(f"Convert model to {dit_weight_dtype}")
            target_dtype = dit_weight_dtype

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    if args.blocks_to_swap > 0:
        logger.info(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        swap_config = BlockSwapConfig(device, supports_backward=False, use_pinned_memory=args.use_pinned_memory_for_block_swap)
        model.enable_block_swap(args.blocks_to_swap, swap_config)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # make sure the model is on the right device
        model.to(device)

    if args.compile:
        model = model_utils.compile_transformer(
            args, model, [model.noise_refiner, model.context_refiner, model.layers], disable_linear=args.blocks_to_swap > 0
        )

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

    return model


# endregion


def decode_latent(vae: AutoencoderKL, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    logger.info(f"Decoding image. Latent shape {latent.shape}, device {device}")
    if latent.ndim == 3:  # CHW
        latent = latent.unsqueeze(0)  # add batch dimension if not present

    latent = zimage_utils.shift_scale_latents_for_decode(latent.to(vae.dtype))

    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode(latent.to(device))  # decode to pixels, -1 to 1
    pixels = pixels.to("cpu", dtype=torch.float32)  # move to CPU and convert to float32 (bfloat16 is not supported by numpy)
    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def prepare_text_inputs(
    args: argparse.Namespace, device: torch.device, shared_models: Optional[Dict] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare text-related inputs for I2V: LLM encoding."""

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    llm_device = torch.device("cpu") if args.text_encoder_cpu else device
    if shared_models is not None:
        tokenizer = shared_models.get("tokenizer")
        text_encoder = shared_models.get("text_encoder")
        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]

        # text_encoder is on device (batched inference) or CPU (interactive inference)
    else:  # Load if not in shared_models
        llm_weight_dtype = resolve_text_encoder_weight_dtype(args, llm_device)
        tokenizer, text_encoder = zimage_utils.load_qwen3(
            args.text_encoder, dtype=llm_weight_dtype, device=llm_device, disable_mmap=True
        )

    # Store original devices to move back later if they were shared. This does nothing if shared_models is None
    text_encoder_original_device = text_encoder.device if text_encoder else None

    # Ensure text_encoder is not None before proceeding
    if not text_encoder or not tokenizer:
        raise ValueError("Text encoder or tokenizer is not loaded properly.")

    # Define a function to move models to device if needed
    # This is to avoid moving models if not needed, especially in interactive mode
    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        nonlocal shared_models

        if model_is_moved:
            return
        model_is_moved = True

        logger.info(f"Moving DiT and Text Encoder to appropriate device: {device} or CPU")
        if shared_models and "model" in shared_models:  # DiT model is shared
            if args.blocks_to_swap > 0:
                logger.info("Waiting for 5 seconds to finish block swap")
                time.sleep(5)
            model = shared_models["model"]
            model.to("cpu")
            clean_memory_on_device(device)  # clean memory on device before moving models

        text_encoder.to(llm_device)  # If text_encoder_cpu is True, this will be CPU

    logger.info("Encoding prompt with Text Encoder.")

    prompt = args.prompt

    # cache_key includes this because embed may be changed if resize_control_to_image_size is True
    cache_key = prompt

    if cache_key in conds_cache:
        embed, mask = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        embed, mask = zimage_utils.get_text_embeds(tokenizer, text_encoder, prompt)
        embed = embed.cpu()
        mask = mask.cpu()

        conds_cache[cache_key] = (embed, mask)

    negative_prompt = args.negative_prompt
    should_encode_negative = negative_prompt is not None or args.guidance_scale > 1.0
    if should_encode_negative:
        effective_negative_prompt = "" if negative_prompt is None else negative_prompt
        if negative_prompt is None:
            logger.info(
                "CFG is enabled and --negative_prompt is not provided. Encoding empty string for unconditional conditioning."
            )

        cache_key = effective_negative_prompt
        if cache_key in conds_cache:
            negative_embed, negative_mask = conds_cache[cache_key]
        else:
            move_models_to_device_if_needed()

            negative_embed, negative_mask = zimage_utils.get_text_embeds(tokenizer, text_encoder, effective_negative_prompt)
            negative_embed = negative_embed.cpu()
            negative_mask = negative_mask.cpu()

            conds_cache[cache_key] = (negative_embed, negative_mask)
    else:
        effective_negative_prompt = None
        negative_embed = None
        negative_mask = None

    if not (shared_models and "text_encoder" in shared_models):  # if loaded locally
        # There is a bug text_encoder is not freed from GPU memory when text encoder is fp8. Needs gc.collect()
        del tokenizer, text_encoder
        gc.collect()  # This may force Text Encoder to be freed from GPU memory
    else:  # if shared, move back to original device (likely CPU)
        if text_encoder:
            text_encoder.to(text_encoder_original_device)

    clean_memory_on_device(device)

    arg_c = {"embed": embed, "mask": mask, "prompt": prompt}
    arg_null = {"embed": negative_embed, "mask": negative_mask, "prompt": effective_negative_prompt}

    return arg_c, arg_null


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_text_data: Optional[Dict] = None,
) -> torch.Tensor:
    """main function for generation

    Args:
        args: command line arguments
        gen_settings: generation settings
        shared_models: dictionary containing pre-loaded models (mainly for DiT)
        precomputed_text_data: Optional dictionary with precomputed text data

    Returns:
        torch.Tensor generated latents
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context = precomputed_text_data["context"]
        context_null = precomputed_text_data["context_null"]
    else:
        logger.info("No precomputed data. Preparing image and text inputs.")
        context, context_null = prepare_text_inputs(args, device, shared_models)

    if shared_models is None or "model" not in shared_models:
        # load DiT model
        model = load_dit_model(args, device, dit_weight_dtype)

        # if we only want to save the model, we can skip the rest
        if args.save_merged_model:
            return None

        if shared_models is not None:
            shared_models["model"] = model
    else:
        # use shared model
        model: zimage_model.ZImageTransformer2DModel = shared_models["model"]
        model.move_to_device_except_swap_blocks(device)  # Handles block swap correctly
        model.prepare_block_swap_before_forward()

    # set random generator
    seed_g = torch.Generator(device="cpu" if args.cpu_noise else device)
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)
    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}")

    # image generation ######

    logger.info(f"Prompt: {context['prompt']}")

    embed = context["embed"].to(device, dtype=torch.bfloat16)
    mask = context["mask"].to(device, dtype=torch.bfloat16)
    negative_embed = context_null["embed"].to(device, dtype=torch.bfloat16) if context_null["embed"] is not None else None
    negative_mask = context_null["mask"].to(device, dtype=torch.bfloat16) if context_null["mask"] is not None else None

    # 4. Prepare latent variables
    vae_scale = zimage_config.ZIMAGE_VAE_SCALE_FACTOR * 2
    height_latent = 2 * (int(height) // vae_scale)  # divisible by 16
    width_latent = 2 * (int(width) // vae_scale)
    shape = (1, model.in_channels, height_latent, width_latent)

    latents = torch.randn(shape, generator=seed_g, device="cpu" if args.cpu_noise else device, dtype=torch.float32).to(device)
    image_sequence_length = (height_latent // model.all_patch_size[0]) * (width_latent // model.all_patch_size[0])

    preview_original_latents = None
    if args.preview_latent_every:
        preview_original_latents = latents.detach().clone()

    # The batch size is 1, so we can trim embeds as the length of the prompt
    embed, _ = zimage_utils.trim_pad_embeds_and_mask(image_sequence_length, embed, mask)
    mask = None  # No attention mask needed after trimming
    if negative_embed is not None:
        negative_embed, _ = zimage_utils.trim_pad_embeds_and_mask(image_sequence_length, negative_embed, negative_mask)
        negative_mask = None

    # 5. Prepare timesteps
    num_inference_steps = args.infer_steps
    flow_shift = args.flow_shift
    if args.dynamic_shift:
        flow_shift = zimage_utils.compute_dynamic_shift(image_sequence_length)
        logger.info(f"Dynamic shift: image_seq_len={image_sequence_length} -> flow_shift={flow_shift:.4f}")

    timesteps, sigmas = zimage_utils.get_timesteps_sigmas(num_inference_steps, flow_shift)
    timesteps = timesteps.to(device)
    sigmas = sigmas.to(device)

    previewer = None
    if args.preview_latent_every:
        os.makedirs(args.save_path, exist_ok=True)
        preview_dtype = torch.float16 if device.type == "cuda" else torch.float32
        previewer = LatentPreviewer(
            args,
            preview_original_latents,
            scheduler=None,
            device=device,
            dtype=preview_dtype,
            model_type="flux",
        )
        previewer.sigmas = sigmas

    # 6. Denoising loop
    do_cfg = args.guidance_scale > 1.0  # 0 for no CFG
    if do_cfg and negative_embed is None:
        logger.warning("CFG is enabled but negative prompt is not provided. Using unconditional generation with zeros.")
        negative_embed = torch.zeros_like(embed)
        negative_mask = None

    with tqdm(total=num_inference_steps, desc="Denoising steps") as pbar:
        for i, t in enumerate(timesteps):
            # CFG truncation: disable CFG at the noisiest steps based on sigma in [0, 1], where 1 is max noise.
            # Default cfg_truncation=1.0 means CFG is applied at all steps.
            effective_guidance_scale = args.guidance_scale
            if do_cfg and args.cfg_truncation < 1.0:
                t_normalized = sigmas[i]  # 1 -> 0
                if bool((t_normalized > args.cfg_truncation).item()):
                    effective_guidance_scale = 0.0

            timestep = t.expand(latents.shape[0])  # No effect since batch size is 1
            timestep = (1000 - timestep) / 1000  # Reverse timestep for z-image

            latent_model_input = latents.to(model.dtype)
            latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension

            # with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True), torch.no_grad():
            with torch.no_grad():
                model_out = model(latent_model_input, timestep, embed, mask)

            if do_cfg and effective_guidance_scale > 0.0:
                # with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True), torch.no_grad():
                with torch.no_grad():
                    neg_model_out = model(latent_model_input, timestep, negative_embed, negative_mask)
                noise_pred = model_out + effective_guidance_scale * (model_out - neg_model_out)

                if args.cfg_normalization is not None:
                    noise_pred = apply_cfg_normalization(model_out, noise_pred, args.cfg_normalization)
            else:
                noise_pred = model_out

            noise_pred = -noise_pred.squeeze(2)  # Remove frame dimension and invert sign (because z-image predicts negative noise)
            latents = zimage_utils.step(noise_pred.to(torch.float32), latents, sigmas, i)

            if previewer is not None and (i + 1) % args.preview_latent_every == 0:
                previewer.preview(latents, i + 1)

            pbar.update(1)

    # Only clean up shared models if they were created within this function
    if shared_models is None:
        # free memory
        del model
        synchronize_device(device)

        # wait for 5 seconds until block swap is done
        if args.blocks_to_swap > 0:
            logger.info("Waiting for 5 seconds to finish block swap")
            time.sleep(5)

        gc.collect()
        clean_memory_on_device(device)

    return latents


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    """Save latent to file

    Args:
        latent: Latent tensor
        args: command line arguments
        height: height of frame
        width: width of frame

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.guidance_scale}",
            "cfg_truncation": f"{args.cfg_truncation}",
            "cfg_normalization": f"{args.cfg_normalization}",
        }
        if args.embedded_cfg_scale is not None:
            metadata["embedded_cfg_scale"] = f"{args.embedded_cfg_scale}"
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_images(sample: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    """Save images to directory

    Args:
        sample: Image tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"
    sample = sample.unsqueeze(0).unsqueeze(2)  # CHW -> BCFHW, where B=1, C=3, F=1, H, W
    save_images_grid(sample, save_path, image_name, rescale=True, create_subdir=False)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    latent: torch.Tensor,
    device: torch.device,
    original_base_names: Optional[List[str]] = None,
) -> None:
    """save output

    Args:
        args: command line arguments
        vae: VAE model
        latent: latent tensor
        device: device to use
        original_base_names: original base names (if latents are loaded from files)
    """
    height, width = latent.shape[-2], latent.shape[-1]  # BCHW
    height *= zimage_config.ZIMAGE_VAE_SCALE_FACTOR
    width *= zimage_config.ZIMAGE_VAE_SCALE_FACTOR
    # print(f"Saving output. Latent shape {latent.shape}; pixel shape {height}x{width}")
    if args.output_type == "latent" or args.output_type == "latent_images":
        # save latent
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if vae is None:
        logger.error("VAE is None, cannot decode latents for saving video/images.")
        return

    video = decode_latent(vae, latent, device)

    if args.output_type == "images" or args.output_type == "latent_images":
        # save images
        if original_base_names is not None:
            original_name = f"_{original_base_names[0]}"
        else:
            original_name = None
        save_images(video, args, original_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    """Process multiple prompts for batch mode

    Args:
        prompt_lines: List of prompt lines
        base_args: Base command line arguments

    Returns:
        List[Dict]: List of prompt data dictionaries
    """
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse prompt line and create override dictionary
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory. VAE is NOT loaded here.
    DiT model is also NOT loaded here, handled by process_batch_prompts or generate.

    Args:
        args: Base command line arguments

    Returns:
        Dict: Dictionary of shared models (text/image encoders)
    """
    shared_models = {}
    # Load text encoders to CPU
    device = torch.device(args.device)
    llm_device = torch.device("cpu") if args.text_encoder_cpu else device
    llm_weight_dtype = resolve_text_encoder_weight_dtype(args, llm_device)
    tokenizer, text_encoder = zimage_utils.load_qwen3(args.text_encoder, dtype=llm_weight_dtype, device="cpu", disable_mmap=True)
    shared_models["tokenizer"] = tokenizer
    shared_models["text_encoder"] = text_encoder
    return shared_models


def process_batch_prompts(prompts_data: List[Dict], args: argparse.Namespace) -> None:
    """Process multiple prompts with model reuse and batched precomputation

    Args:
        prompts_data: List of prompt data dictionaries
        args: Base command line arguments
    """
    if not prompts_data:
        logger.warning("No valid prompts found")
        return

    gen_settings = get_generation_settings(args)
    dit_weight_dtype = gen_settings.dit_weight_dtype
    device = gen_settings.device

    # 1. Prepare VAE
    logger.info("Loading VAE for batch generation...")
    vae_for_batch = zimage_autoencoder.load_autoencoder_kl(args.vae, device="cpu", disable_mmap=True)
    vae_for_batch.eval()

    all_prompt_args_list = [apply_overrides(args, pd) for pd in prompts_data]  # Create all arg instances first
    for prompt_args in all_prompt_args_list:
        check_inputs(prompt_args)  # Validate each prompt's height/width

    # 2. Precompute Text Data (Text Encoder)
    logger.info("Loading Text Encoder for batch text preprocessing...")

    # Text Encoder loaded to CPU then moved to llm_device for encoding.
    llm_device = torch.device("cpu") if args.text_encoder_cpu else device
    vl_dtype = resolve_text_encoder_weight_dtype(args, llm_device)
    tokenizer_batch, text_encoder_batch = zimage_utils.load_qwen3(
        args.text_encoder, dtype=vl_dtype, device="cpu", disable_mmap=True
    )

    # Text Encoder to device for this phase
    text_encoder_batch.to(llm_device)  # Moved into prepare_text_inputs logic

    all_precomputed_text_data = []
    conds_cache_batch = {}

    logger.info("Preprocessing text and LLM/TextEncoder encoding for all prompts...")
    temp_shared_models_txt = {
        "tokenizer": tokenizer_batch,
        "text_encoder": text_encoder_batch,  # on GPU
        "conds_cache": conds_cache_batch,
    }

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Text preprocessing for prompt {i + 1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")

        # prepare_text_inputs will move text_encoders to device temporarily, and handles edit or not
        context, context_null = prepare_text_inputs(prompt_args_item, device, temp_shared_models_txt)
        text_data = {"context": context, "context_null": context_null}
        all_precomputed_text_data.append(text_data)

    # Models should be removed from device after prepare_text_inputs
    del tokenizer_batch, text_encoder_batch, temp_shared_models_txt, conds_cache_batch
    gc.collect()  # Force cleanup of Text Encoder from GPU memory
    clean_memory_on_device(device)

    # 3. Load DiT Model once
    logger.info("Loading DiT model for batch generation...")
    # Use args from the first prompt for DiT loading (LoRA etc. should be consistent for a batch)
    first_prompt_args = all_prompt_args_list[0]
    dit_model = load_dit_model(first_prompt_args, device, dit_weight_dtype)  # Load directly to target device if possible

    if first_prompt_args.save_merged_model:
        logger.info("Merged DiT model saved. Skipping generation.")
        return

    shared_models_for_generate = {"model": dit_model}  # Pass DiT via shared_models

    all_latents = []

    logger.info("Generating latents for all prompts...")
    with torch.no_grad():
        for i, prompt_args_item in enumerate(all_prompt_args_list):
            current_text_data = all_precomputed_text_data[i]
            height, width = check_inputs(prompt_args_item)  # Get height/width for each prompt

            logger.info(f"Generating latent for prompt {i + 1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
            try:
                # generate is called with precomputed data, so it won't load VAE/Text/Image encoders.
                # It will use the DiT model from shared_models_for_generate.
                # The VAE instance returned by generate will be None here.
                latent = generate(prompt_args_item, gen_settings, shared_models_for_generate, current_text_data)

                if latent is None:  # and prompt_args_item.save_merged_model:  # Should be caught earlier
                    continue

                # Save latent if needed (using data from precomputed_image_data for H/W)
                if prompt_args_item.output_type in ["latent", "latent_images"]:
                    save_latent(latent, prompt_args_item, height, width)

                all_latents.append(latent)
            except Exception as e:
                logger.error(f"Error generating latent for prompt: {prompt_args_item.prompt}. Error: {e}", exc_info=True)
                all_latents.append(None)  # Add placeholder for failed generations
                continue

    # Free DiT model
    logger.info("Releasing DiT model from memory...")
    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    del shared_models_for_generate["model"]
    del dit_model
    gc.collect()
    clean_memory_on_device(device)
    synchronize_device(device)  # Ensure memory is freed before loading VAE for decoding

    # 4. Decode latents and save outputs (using vae_for_batch)
    if args.output_type != "latent":
        logger.info("Decoding latents to videos/images using batched VAE...")
        vae_for_batch.to(device)  # Move VAE to device for decoding

        for i, latent in enumerate(all_latents):
            if latent is None:  # Skip failed generations
                logger.warning(f"Skipping decoding for prompt {i + 1} due to previous error.")
                continue

            current_args = all_prompt_args_list[i]
            logger.info(f"Decoding output {i + 1}/{len(all_latents)} for prompt: {current_args.prompt}")

            # if args.output_type is "latent_images", we already saved latent above.
            # so we skip saving latent here.
            if current_args.output_type == "latent_images":
                current_args.output_type = "images"

            # save_output expects latent to be [BCTHW] or [CTHW]. generate returns [BCTHW] (batch size 1).
            # latent[0] is correct if generate returns it with batch dim.
            # The latent from generate is (1, C, T, H, W)
            save_output(current_args, vae_for_batch, latent[0], device)  # Pass vae_for_batch

        vae_for_batch.to("cpu")  # Move VAE back to CPU

    del vae_for_batch
    clean_memory_on_device(device)


def process_interactive(args: argparse.Namespace) -> None:
    """Process prompts in interactive mode

    Args:
        args: Base command line arguments
    """
    gen_settings = get_generation_settings(args)
    device = gen_settings.device
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}  # Initialize empty cache for interactive mode

    # Load VAE for interactive mode
    logger.info("Loading VAE for interactive mode...")
    vae = zimage_autoencoder.load_autoencoder_kl(args.vae, device="cpu", disable_mmap=True)
    vae.eval()

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")

    try:
        import prompt_toolkit
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        prompt_toolkit = None

    if prompt_toolkit:
        session = prompt_toolkit.PromptSession()

        def input_line(prompt: str) -> str:
            return session.prompt(prompt)

    else:

        def input_line(prompt: str) -> str:
            return input(prompt)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in ["\x04", "\x1a"]:  # Ctrl+D or Ctrl+Z with prompt_toolkit
                    raise EOFError  # Exit on Ctrl+D or Ctrl+Z

                # Parse prompt
                prompt_data = parse_prompt_line(line)
                prompt_args = apply_overrides(args, prompt_data)

                # Generate latent
                # For interactive, precomputed data is None. shared_models contains text/image encoders.
                # generate will load VAE internally.
                latent = generate(prompt_args, gen_settings, shared_models)

                # # If not one_frame_inference, move DiT model to CPU after generation
                # if prompt_args.blocks_to_swap > 0:
                #     logger.info("Waiting for 5 seconds to finish block swap")
                #     time.sleep(5)
                # model = shared_models.get("model")
                # model.to("cpu")  # Move DiT model to CPU after generation

                # Save latent and video
                # returned_vae from generate will be used for decoding here.
                save_output(prompt_args, vae, latent[0], device)

                if args.bell:
                    print("\a")

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = torch.bfloat16  # default from Z-Image official inference code
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


def main():
    # Parse arguments
    args = parse_args()

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if args.attn_mode == "cute":
        from musubi_tuner.modules.attention import probe_cute_runtime

        ok, detail = probe_cute_runtime(device, needs_backward=False)
        if not ok:
            raise ValueError(
                f"--attn_mode cute preflight failed on this GPU: {detail}. "
                "Either install a CuTE build that supports your architecture, "
                "or use --attn_mode flash / sdpa. See docs/cute_attention.md."
            )

    if latents_mode:
        # Original latent decode mode
        original_base_names = []
        latents_list = []
        seeds = []

        for latent_path in args.latent_path:
            original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu", weights_only=True)
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"])
                    width = int(metadata["width"])
                    args.image_size = [height, width]

            seeds.append(seed)
            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

            if latents.ndim == 4:  # [BCHW]
                latents = latents.squeeze(0)  # [CHW]

            latents_list.append(latents)

        # latent = torch.stack(latents_list, dim=0)  # [N, ...], must be same shape

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]

            vae = zimage_autoencoder.load_autoencoder_kl(args.vae, device, disable_mmap=True)
            vae.eval()
            save_output(args, vae, latent, device, original_base_names)

    elif args.from_file:
        # Batch mode from file

        # Read prompts from file
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_lines = f.readlines()

        # Process prompts
        prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
        process_batch_prompts(prompts_data, args)

        if args.bell:
            print("\a")  # Bell sound

    elif args.interactive:
        # Interactive mode
        process_interactive(args)

    else:
        # Single prompt mode (original behavior)

        # Generate latent
        gen_settings = get_generation_settings(args)

        # For single mode, precomputed data is None, shared_models is None.
        # generate will load all necessary models (VAE, Text/Image Encoder, DiT).
        latent = generate(args, gen_settings)

        if latent is None:
            # --save_merged_model path: model already saved in load_dit_model, nothing else to do
            logger.info("Done!")
            return

        # Save latent and video
        vae = zimage_autoencoder.load_autoencoder_kl(args.vae, device, disable_mmap=True)
        vae.eval()
        save_output(args, vae, latent, device)

        if args.bell:
            print("\a")  # Bell sound

    logger.info("Done!")


if __name__ == "__main__":
    main()
