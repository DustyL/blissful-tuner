import argparse
from typing import Optional
from blissful_tuner.blissful_logger import BlissfulLogger
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from musubi_tuner.networks import lora
from musubi_tuner.utils.lora_utils import (
    _check_lora_base_hash,
    _read_safetensors_metadata,
    compute_base_hash,
    convert_diffusers_if_needed,
    detect_network_type,
    format_unknown_network_type_error,
    merge_nonlora_to_model,
)
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
from musubi_tuner.hunyuan_model.models import load_transformer

logger = BlissfulLogger(__name__, "green")


def _infer_architecture_from_lora_file(lora_path: str) -> Optional[str]:
    """Infer base-model architecture from a LoRA safetensors file.

    Prefers `ss_network_module` metadata when available, but falls back to key heuristics
    for foreign/partial files that omit metadata.
    """
    metadata: Optional[dict[str, str]] = None
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()

        network_module = metadata.get("ss_network_module") if metadata else None
        if network_module:
            if "lora_qwen_image" in network_module:
                return "qwen_image"
            if network_module == "networks.lora":
                return "hv"

        # Fallback: infer from key families
        keys = list(f.keys())
        if any("transformer_blocks" in k for k in keys):
            return "qwen_image"
        if any(("double_blocks" in k or "single_blocks" in k) for k in keys):
            return "hv"

    return None


def _resolve_architecture(cli_architecture: str, lora_paths: list[str] | None) -> str:
    if cli_architecture != "auto":
        return cli_architecture

    if not lora_paths:
        # Backward compatible default: this script historically merged HunyuanVideo models.
        return "hv"

    inferred = set()
    for path in lora_paths:
        arch = _infer_architecture_from_lora_file(path)
        if arch is not None:
            inferred.add(arch)

    if len(inferred) == 1:
        return inferred.pop()
    if len(inferred) == 0:
        raise ValueError(
            "Unable to infer architecture from LoRA weights (missing metadata and unrecognized keys). "
            "Pass --architecture hv|qwen_image."
        )
    raise ValueError(f"Conflicting architectures inferred from LoRA weights: {sorted(inferred)}. Pass --architecture explicitly.")


def parse_args():
    parser = argparse.ArgumentParser(description="Model merger script (merge LoRA weights into a base DiT)")

    parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint path or directory")
    parser.add_argument("--dit_in_channels", type=int, default=16, help="input channels for DiT, default is 16, skyreels I2V is 32")
    parser.add_argument(
        "--architecture",
        type=str,
        default="auto",
        choices=["auto", "hv", "qwen_image"],
        help="Base model architecture to load (auto detects from LoRA metadata when possible).",
    )
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=[1.0], help="LoRA multiplier (can specify multiple values)"
    )
    parser.add_argument("--save_merged_model", type=str, required=True, help="Path to save the merged model")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for merging"
    )
    parser.add_argument(
        "--safe_merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Check merged tensors for NaN/Inf before saving. Default: ON. "
            "Pass --no-safe_merge to restore the old unchecked behavior."
        ),
    )
    parser.add_argument("--no_safe_merge", dest="safe_merge", action="store_false", help=argparse.SUPPRESS)

    # Qwen-Image variant flags (only used when --architecture qwen_image)
    from musubi_tuner.qwen_image import qwen_image_utils

    qwen_image_utils.add_model_version_args(parser)

    return parser.parse_args()


def _pissa_base_hash_preflight(args: argparse.Namespace) -> None:
    """Validate PiSSA-tagged LoRA base hashes BEFORE loading the DiT model.

    Iterates args.lora_weight, reads each LoRA's metadata header (no tensor
    load), and for any LoRA tagged with ss_init_lora_weights="pissa*" runs
    _check_lora_base_hash with init_pissa=True and strict=True. Computes
    the base hash exactly once (cached) regardless of how many PiSSA LoRAs
    are in the merge set.

    Strict-by-default rationale (per locked Tier 2 #6b plan): offline merge
    writes a derived checkpoint, so artifact-safety failures should be loud.
    Hotswap has --no-hotswap_strict_base_hash because it's an interactive
    runtime path; merge has no such opt-out in v1. A future
    --allow_pissa_missing_base_hash flag can have its own design pass if a
    real user needs it.

    Skips entirely for non-PiSSA LoRAs to avoid noisy new behavior for
    standard merge users — pinned by tests to ensure no metadata read or
    hash compute fires when no PiSSA-tagged LoRAs are present.
    """
    lora_paths = getattr(args, "lora_weight", None) or []
    if not lora_paths:
        return

    base_hash: Optional[str] = None
    for lora_path in lora_paths:
        metadata = _read_safetensors_metadata(lora_path)
        if not metadata:
            continue
        init_lora_weights = metadata.get("ss_init_lora_weights", "")
        if not init_lora_weights.startswith("pissa"):
            continue
        if base_hash is None:
            logger.info(f"PiSSA LoRA detected ({lora_path}); computing base hash for preflight")
            base_hash = compute_base_hash([args.dit])
        _check_lora_base_hash(
            metadata,
            lora_path,
            base_hash,
            strict=True,
            init_pissa=True,
        )


def main():
    args = parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"safe_merge finite checks: {'enabled' if args.safe_merge else 'disabled'}")

    # PiSSA preflight: fail before load_transformer if any input LoRA is
    # PiSSA-tagged and the base hash doesn't match. Cheaper than discovering
    # the mismatch after a multi-GB model load.
    _pissa_base_hash_preflight(args)

    architecture = _resolve_architecture(args.architecture, args.lora_weight)
    logger.info(f"Architecture: {architecture}")

    # Load DiT model
    logger.info(f"Loading DiT model from {args.dit}")
    if architecture == "hv":
        transformer = load_transformer(args.dit, "torch", False, "cpu", torch.bfloat16, in_channels=args.dit_in_channels)
    elif architecture == "qwen_image":
        from musubi_tuner.qwen_image import qwen_image_model, qwen_image_utils

        qwen_image_utils.resolve_model_version_args(args)
        is_layered = args.model_version == "layered"
        transformer = qwen_image_model.load_qwen_image_model(
            device=device,
            dit_path=args.dit,
            attn_mode="torch",
            split_attn=False,
            zero_cond_t=args.model_version == "edit-2511",
            use_additional_t_cond=is_layered,
            use_layer3d_rope=is_layered,
            loading_device="cpu",
            dit_weight_dtype=torch.bfloat16,
            disable_numpy_memmap=getattr(args, "disable_numpy_memmap", False),
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    transformer.eval()

    # Load LoRA weights and merge
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        for i, lora_weight in enumerate(args.lora_weight):
            # Use the corresponding lora_multiplier or default to 1.0
            if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
                lora_multiplier = args.lora_multiplier[i]
            else:
                lora_multiplier = 1.0

            logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
            weights_sd = load_file(lora_weight)
            net_type = detect_network_type(weights_sd)
            if net_type == "unknown":
                raise ValueError(format_unknown_network_type_error(lora_weight))
            weights_sd = convert_diffusers_if_needed(weights_sd)
            if net_type in ("loha", "lokr", "hybrid"):
                logger.info(f"Detected {net_type} weights, using per-key-family merge")
                merged_count = merge_nonlora_to_model(transformer, weights_sd, lora_multiplier, device, safe_merge=args.safe_merge)
                if merged_count == 0:
                    raise ValueError(
                        f"No weights were merged from {lora_weight}. This usually means the LoRA architecture "
                        f"does not match the base model. Try passing --architecture explicitly."
                    )
            else:
                if architecture == "hv":
                    create_network = lora.create_arch_network_from_weights
                elif architecture == "qwen_image":
                    from musubi_tuner.networks import lora_qwen_image

                    create_network = lora_qwen_image.create_arch_network_from_weights
                else:
                    raise ValueError(f"Unsupported architecture: {architecture}")

                network = create_network(lora_multiplier, weights_sd, unet=transformer, for_inference=True)
                if getattr(network, "unet_loras", None) is not None and len(network.unet_loras) == 0:
                    raise ValueError(
                        f"No matching LoRA modules were found when applying {lora_weight}. "
                        f"This usually means the LoRA architecture does not match the base model. "
                        f"Try passing --architecture explicitly."
                    )
                logger.info("Merging LoRA weights to DiT model")
                network.merge_to(None, transformer, weights_sd, device=device, non_blocking=True, safe_merge=args.safe_merge)

            logger.info("LoRA weights merged")

    # Save the merged model
    logger.info(f"Saving merged model to {args.save_merged_model}")
    mem_eff_save_file(transformer.state_dict(), args.save_merged_model)
    logger.info("Merged model saved")


if __name__ == "__main__":
    main()
