import argparse

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from musubi_tuner.utils import model_utils

from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")


# keys of Qwen-Image state dict
QWEN_IMAGE_KEYS = [
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "txt_norm",
    "img_in",
    "txt_in",
    "transformer_blocks.*.img_mod.1",
    "transformer_blocks.*.attn.norm_q",
    "transformer_blocks.*.attn.norm_k",
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.add_k_proj",
    "transformer_blocks.*.attn.add_v_proj",
    "transformer_blocks.*.attn.add_q_proj",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.attn.to_add_out",
    "transformer_blocks.*.attn.norm_added_q",
    "transformer_blocks.*.attn.norm_added_k",
    "transformer_blocks.*.img_mlp.net.0.proj",
    "transformer_blocks.*.img_mlp.net.2",
    "transformer_blocks.*.txt_mod.1",
    "transformer_blocks.*.txt_mlp.net.0.proj",
    "transformer_blocks.*.txt_mlp.net.2",
    "norm_out.linear",
    "proj_out",
]


def convert_from_diffusers(prefix, weights_sd):
    # convert from diffusers(?) to default LoRA/LoHa/LoKr
    # Diffusers format: {"diffusion_model.module.name.lora_A.weight": weight, ...}
    # default format: {"prefix_module_name.lora_down.weight": weight, ...}

    # note: Diffusers has no alpha, so alpha is set to rank
    new_weights_sd = {}
    lora_dims = {}

    # Pass through non-dotted metadata keys (lokr_factor, use_rslora_flag, etc.)
    for key, value in weights_sd.items():
        if "." not in key:
            new_weights_sd[key] = value

    for key, weight in weights_sd.items():
        if "." not in key:
            continue  # already handled above

        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            logger.warning(f"unexpected key: {key} in diffusers format")
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_")

        if "_lora_" in new_key:
            # LoRA keys
            new_key = new_key.replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
            # support unknown format: do not replace dots but uses lora_up/lora_down/alpha
            new_key = new_key.replace("_lora_down_", ".lora_down.").replace("_lora_up_", ".lora_up.")
        else:
            # LoHa or LoKr keys
            new_key = new_key.replace("_hada_", ".hada_").replace("_lokr_", ".lokr_")

        if new_key.endswith("_alpha"):
            new_key = new_key.replace("_alpha", ".alpha")

        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims:
            if "lora_down" in new_key:
                lora_dims[lora_name] = weight.shape[0]
            elif "hada_w1_b" in new_key:
                lora_dims[lora_name] = weight.shape[0]
            elif "lokr_w2_a" in new_key:
                lora_dims[lora_name] = weight.shape[1]
            elif "lokr_w2" in new_key and "lokr_w2_b" not in new_key:
                # Full-matrix LoKr: dim is max of w2 shape
                lora_dims[lora_name] = max(weight.shape)

    # add alpha with rank (for LoRA keys that lack alpha in Diffusers format)
    for lora_name, dim in lora_dims.items():
        alpha_key = f"{lora_name}.alpha"
        if alpha_key not in new_weights_sd:
            new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return new_weights_sd


def _normalize_module_name(lora_name: str, prefix: str, arch: str | None = None) -> str:
    """Convert a lora_name (e.g. 'lora_unet_blocks_0_cross_attn_k_img') to a Diffusers module name.

    Applies architecture-specific fixups for WAN, Z-Image, and HunyuanVideo module names
    where naive underscore-to-dot replacement would be lossy.
    """
    module_name = lora_name[len(prefix) :]  # remove "lora_unet_"
    module_name = module_name.replace("_", ".")  # replace "_" with "."
    if ".cross.attn." in module_name or ".self.attn." in module_name:
        # Wan2.1 lora name to module name
        module_name = module_name.replace("cross.attn", "cross_attn")
        module_name = module_name.replace("self.attn", "self_attn")
        module_name = module_name.replace("k.img", "k_img")
        module_name = module_name.replace("v.img", "v_img")
    elif ".attention.to." in module_name or ".feed.forward." in module_name:
        # Z-Image lora name to module name
        module_name = module_name.replace("to.q", "to_q")
        module_name = module_name.replace("to.k", "to_k")
        module_name = module_name.replace("to.v", "to_v")
        module_name = module_name.replace("to.out", "to_out")
        module_name = module_name.replace("feed.forward", "feed_forward")
    else:
        # HunyuanVideo / Flux-style lora name to module name
        module_name = module_name.replace("double.blocks.", "double_blocks.")
        module_name = module_name.replace("single.blocks.", "single_blocks.")
        module_name = module_name.replace("img.", "img_")
        module_name = module_name.replace("txt.", "txt_")
        if arch not in ("flux", "flux2", "flux_kontext", "qwen_image"):
            # HunyuanVideo uses flat attributes like `img_attn_qkv`, while FLUX uses nested modules like `img_attn.qkv`.
            module_name = module_name.replace("attn.", "attn_")
    return module_name


def _resolve_arch_from_metadata(metadata: dict[str, str] | None, cli_arch: str | None) -> str | None:
    """Resolve architecture style for lossy LoRA key normalization.

    This only affects ambiguous key families that share `double_blocks`/`single_blocks` naming,
    such as FLUX.* and HunyuanVideo. For other architectures, normalization is handled by
    specific pattern branches (WAN, Z-Image) and is not affected by this value.
    """
    if cli_arch is not None and cli_arch != "auto":
        return cli_arch

    if not metadata:
        return None

    network_module = metadata.get("ss_network_module")
    if not network_module:
        return None

    # FLUX variants have nested attention modules like `img_attn.qkv`.
    if "lora_flux_2" in network_module:
        return "flux2"
    if "lora_flux" in network_module:
        return "flux_kontext"

    # Qwen-Image also uses nested modules (e.g. `attn.to_q`) rather than flat `attn_to_q`.
    if "lora_qwen_image" in network_module:
        return "qwen_image"

    # HunyuanVideo uses flat attention attributes like `img_attn_qkv`.
    if network_module == "networks.lora":
        return "hv"

    return None


def convert_to_diffusers(prefix, diffusers_prefix, weights_sd, metadata: dict[str, str] | None = None, arch: str | None = None):
    # convert from default LoRA/LoHa/LoKr to diffusers
    if diffusers_prefix is None:
        diffusers_prefix = "diffusion_model"

    arch = _resolve_arch_from_metadata(metadata, arch)
    if arch is None:
        has_ambiguous_block_keys = any(("double_blocks" in k or "single_blocks" in k) for k in weights_sd.keys())
        if has_ambiguous_block_keys:
            logger.warning(
                "LoRA metadata missing or unrecognized (ss_network_module). "
                "Ambiguous FLUX vs HunyuanVideo keys may convert incorrectly. "
                "Pass --arch flux2/flux_kontext/hv for deterministic conversion."
            )
    else:
        logger.info(f"Architecture style for key normalization: {arch}")

    # Detect network type for logging
    from musubi_tuner.utils.lora_utils import detect_network_type

    net_type = detect_network_type(weights_sd)
    logger.info(f"Detected network type: {net_type}")

    # make reverse map from LoRA name to base model module name
    lora_name_to_module_name = {}
    for key in QWEN_IMAGE_KEYS:
        if "*" not in key:
            lora_name = prefix + key.replace(".", "_")
            lora_name_to_module_name[lora_name] = key
        else:
            lora_name = prefix + key.replace(".", "_")
            for i in range(100):  # assume at most 100 transformer blocks
                lora_name_to_module_name[lora_name.replace("*", str(i))] = key.replace("*", str(i))

    # get alphas
    lora_alphas = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]  # before first dot
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = weight

    new_weights_sd = {}

    # Pass through non-dotted metadata keys (lokr_factor, use_rslora_flag, etc.)
    for key, value in weights_sd.items():
        if "." not in key:
            new_weights_sd[key] = value

    for key, weight in weights_sd.items():
        if not key.startswith(prefix):
            continue
        if "alpha" in key:
            continue

        lora_name = key.split(".", 1)[0]  # before first dot

        # Resolve module name (shared normalization for all key families)
        if lora_name in lora_name_to_module_name:
            module_name = lora_name_to_module_name[lora_name]
        else:
            module_name = _normalize_module_name(lora_name, prefix, arch=arch)

        # LoHa/LoKr keys: pass through with Diffusers-style key renaming
        if ".hada_" in key or ".lokr_" in key:
            _, param_part = key.split(".", 1)  # e.g., "hada_w1_a"
            new_key = f"{diffusers_prefix}.{module_name}.{param_part}"
            new_weights_sd[new_key] = weight
            # Copy alpha as-is (no sqrt scaling for LoHa/LoKr)
            alpha_key = f"{lora_name}.alpha"
            if alpha_key in weights_sd:
                new_alpha_key = f"{diffusers_prefix}.{module_name}.alpha"
                new_weights_sd[new_alpha_key] = weights_sd[alpha_key]
            continue

        # Standard LoRA path
        if "lora_down" in key:
            new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
            dim = weight.shape[0]
        elif "lora_up" in key:
            new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
            dim = weight.shape[1]
        else:
            logger.warning(f"unexpected key: {key} in default LoRA format")
            continue

        # scale weight by alpha
        if lora_name in lora_alphas:
            # we scale both down and up, so scale is sqrt
            scale = lora_alphas[lora_name] / dim
            scale = scale.sqrt()
            weight = weight * scale
        else:
            logger.warning(f"missing alpha for {lora_name}")

        new_weights_sd[new_key] = weight

    return new_weights_sd


def convert(input_file, output_file, target_format, diffusers_prefix, arch: str | None = None):
    logger.info(f"loading {input_file}")
    weights_sd = load_file(input_file)

    # Read source metadata (preserve all existing metadata through conversion)
    with safe_open(input_file, framework="pt") as f:
        metadata = dict(f.metadata() or {})

    logger.info(f"converting to {target_format}")
    prefix = "lora_unet_"
    if target_format == "default":
        new_weights_sd = convert_from_diffusers(prefix, weights_sd)
    elif target_format == "other":
        new_weights_sd = convert_to_diffusers(prefix, diffusers_prefix, weights_sd, metadata=metadata, arch=arch)
    else:
        raise ValueError(f"unknown target format: {target_format}")

    # Recompute hashes for the converted weights (old hashes are stale)
    model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(new_weights_sd, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    logger.info(f"saving to {output_file}")
    save_file(new_weights_sd, output_file, metadata=metadata)

    logger.info("done")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LoRA/LoHa/LoKr weights between default and other formats")
    parser.add_argument("--input", type=str, required=True, help="input model file")
    parser.add_argument("--output", type=str, required=True, help="output model file")
    parser.add_argument("--target", type=str, required=True, choices=["other", "default"], help="target format")
    parser.add_argument(
        "--diffusers_prefix", type=str, default=None, help="prefix for Diffusers weights, default is None (use `diffusion_model`)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "hv", "flux", "flux2", "flux_kontext", "qwen_image"],
        help="Architecture style for ambiguous key normalization (default: auto, inferred from metadata when possible).",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert(args.input, args.output, args.target, args.diffusers_prefix, arch=args.arch)


if __name__ == "__main__":
    main()
