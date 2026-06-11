import argparse
import logging

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_IDEOGRAM4
from musubi_tuner.ideogram4.caption_verifier import verify_caption
from musubi_tuner.ideogram4.generation import denoise_ideogram4_to_tokens
from musubi_tuner.ideogram4.ideogram4_utils import (
    decode_dit_tokens_to_pixels,
    load_ideogram4_autoencoder,
    load_ideogram4_transformer,
)
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.ideogram4.text_encoder import (
    TEXT_ENCODER_FORMATS,
    encode_prompt_to_features,
    load_ideogram4_text_encoder,
    load_ideogram4_tokenizer,
)
from musubi_tuner.networks import loha, lokr
from musubi_tuner.utils.lora_utils import detect_network_type
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _resolve_multipliers(lora_weights: list[str], lora_multipliers: list[float] | None) -> list[float]:
    """Multiplier contract for scripted A/B work (review hardening on PR #23).

    None -> 1.0 for every weight; a SINGLE value broadcasts to all weights ("try all five
    checkpoints at 0.8"); otherwise lengths must match exactly. The previous silent positional
    defaulting (3 weights + 2 multipliers -> third quietly ran at 1.0) was easy to misread in
    exactly the same-seed checkpoint-A/B use case this feature exists for.
    """
    if lora_multipliers is None:
        return [1.0] * len(lora_weights)
    if len(lora_multipliers) == 1:
        return [float(lora_multipliers[0])] * len(lora_weights)
    if len(lora_multipliers) != len(lora_weights):
        raise ValueError(
            f"--lora_multiplier count ({len(lora_multipliers)}) must match --lora_weight count "
            f"({len(lora_weights)}), or be a single value to broadcast, or be omitted (all 1.0)."
        )
    return [float(m) for m in lora_multipliers]


def apply_lora_weights_runtime(
    transformer: torch.nn.Module,
    lora_weights: list[str],
    lora_multipliers: list[float] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.nn.Module]:
    """Runtime-apply LoRA/LoHa/LoKr checkpoints to the CONDITIONAL DiT (backlog P2-7).

    Runtime apply — NOT merge — because the Ideogram 4 base is fp8-prequantized and
    networks/lora.py REFUSES destructive merges into fp8 weights by design (writing float-merged
    values back as fp8 without inverse re-scaling corrupts the checkpoint). The adapter stays live
    during denoise via the monkey-patched module forwards, exactly like sampling-during-training;
    the e85284b dtype harmonization (and its LoHa/LoKr ports) makes the no-autocast path safe.

    The unconditional DiT deliberately gets NO adapter: training never touches it (asymmetric CFG),
    so applying one at generation time would diverge from the trained regime.

    Returns the applied networks — keep the references alive until denoising is done.
    """
    multipliers = _resolve_multipliers(lora_weights, lora_multipliers)
    networks: list[torch.nn.Module] = []
    for path, multiplier in zip(lora_weights, multipliers):
        logger.info(f"Applying adapter {path} at multiplier {multiplier} (runtime, conditional DiT only)")
        # Single pass for tensors AND metadata: LoKr's factor can live only in ss_lokr_factor
        # metadata (buffer-stripped files, e.g. from the ComfyUI converter's forward path) — without
        # it, factor resolution falls back to -1 and load_state_dict fails on factorization shapes.
        weights_sd: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            for key in f.keys():
                weights_sd[key] = f.get_tensor(key)

        net_type = detect_network_type(weights_sd)
        if net_type == "lora":
            # DoRA / rsLoRA are auto-detected from the checkpoint's flag buffers inside
            # create_network_from_weights, so a v8+ DoRA checkpoint reconstructs correctly.
            network = lora_ideogram4.create_arch_network_from_weights(multiplier, weights_sd, unet=transformer, for_inference=True)
        elif net_type == "loha":
            network = loha.create_arch_network_from_weights(
                multiplier, weights_sd, unet=transformer, for_inference=True, architecture=ARCHITECTURE_IDEOGRAM4
            )
        elif net_type == "lokr":
            # lokr_factor resolved with the documented precedence: checkpoint buffer first, then the
            # ss_lokr_factor metadata this loader now actually supplies (lokr._resolve_factor).
            network = lokr.create_arch_network_from_weights(
                multiplier,
                weights_sd,
                unet=transformer,
                for_inference=True,
                architecture=ARCHITECTURE_IDEOGRAM4,
                metadata_factor=metadata.get("ss_lokr_factor"),
            )
        else:
            raise ValueError(
                f"Cannot runtime-apply '{path}': detected network type '{net_type}'. Supported: plain LoRA "
                "(incl. DoRA/rsLoRA), LoHa, LoKr. 'hybrid'/'unknown' checkpoints (e.g. post-conversion mixes) "
                "are not supported here."
            )

        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        info = network.load_state_dict(weights_sd, strict=False)
        if info.unexpected_keys:
            logger.warning(f"Adapter load: unexpected keys ignored: {sorted(info.unexpected_keys)[:5]}...")
        # Explicitly enable the live forward path: LoKrInfModule defaults enabled=False on the
        # assumption that inference flows MERGE (lokr.py "disabled by default for inference") — but
        # merge is exactly what the fp8 base forbids, so runtime-apply must flip it on. LoRA/LoHa
        # inference modules default enabled=True; calling set_enabled(True) is uniform and explicit.
        network.set_enabled(True)
        network.to(device=device, dtype=dtype)
        network.eval()
        networks.append(network)
    return networks


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ideogram 4 image generation")
    parser.add_argument("--dit", type=str, required=True, help="conditional DiT weights (fp8)")
    parser.add_argument("--unconditional_dit", type=str, required=True, help="unconditional DiT weights (fp8)")
    parser.add_argument("--vae", type=str, required=True, help="FLUX.2 VAE (flux2-vae.safetensors)")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL TE weights")
    parser.add_argument("--text_encoder_config", type=str, required=True, help="local Qwen3-VL config dir")
    parser.add_argument("--tokenizer", type=str, required=True, help="local Qwen3-VL tokenizer dir")
    parser.add_argument("--text_encoder_format", type=str, default="hf_full", choices=TEXT_ENCODER_FORMATS)
    parser.add_argument("--prompt", type=str, required=True, help="structured JSON or plain-text prompt")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"))
    parser.add_argument("--sampler_preset", type=str, default="V4_DEFAULT_20", choices=list(PRESETS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--strict_caption_verifier", action="store_true")
    parser.add_argument(
        "--lora_weight",
        type=str,
        nargs="*",
        default=None,
        help="LoRA/LoHa/LoKr checkpoint path(s), runtime-applied to the CONDITIONAL DiT only (the fp8 base "
        "cannot be merged into by design; the unconditional DiT stays adapter-free, matching training)",
    )
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        nargs="*",
        default=None,
        help="multiplier(s) for --lora_weight: omit for all-1.0, give ONE value to broadcast to every "
        "weight, or match counts exactly (mismatched counts error out)",
    )
    return parser


def main():
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = str_to_dtype(args.dtype)
    height, width = args.image_size
    preset = PRESETS[args.sampler_preset]

    # 1. Encode the prompt, then free the text encoder before loading the (large) DiTs.
    verify_caption(args.prompt, strict=args.strict_caption_verifier)  # H4: warn-only by default
    logger.info(f"Loading text encoder ({args.text_encoder_format})")
    text_encoder = load_ideogram4_text_encoder(
        args.text_encoder,
        args.text_encoder_config,
        text_encoder_format=args.text_encoder_format,
        dtype=dtype,
        loading_device=device,
    )
    tokenizer = load_ideogram4_tokenizer(args.tokenizer)
    features = encode_prompt_to_features(tokenizer, text_encoder, args.prompt, device)  # (L, 53248)
    logger.info(f"Encoded prompt -> {tuple(features.shape)}")
    del text_encoder, tokenizer
    _empty_cache(device)

    # 2. Load both DiTs, denoise (the single source of the B5 t-convention), then FREE them before loading the
    #    VAE for decode — denoise needs both DiTs, decode needs only the VAE, so freeing first keeps the peak low.
    logger.info("Loading conditional + unconditional DiTs")
    conditional = load_ideogram4_transformer(args.dit, dtype=dtype, loading_device=device)
    unconditional = load_ideogram4_transformer(args.unconditional_dit, dtype=dtype, loading_device=device)

    # P2-7: runtime-apply trained adapters to the conditional DiT. Keep the returned networks alive
    # through the denoise — they own the monkey-patched forwards.
    lora_networks = []
    if args.lora_weight:
        lora_networks = apply_lora_weights_runtime(conditional, args.lora_weight, args.lora_multiplier, device, dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    logger.info(f"Denoising: {preset.num_steps} steps, preset {args.sampler_preset}")
    z, grid_h, grid_w = denoise_ideogram4_to_tokens(
        conditional,
        unconditional,
        features,
        height=height,
        width=width,
        preset=preset,
        device=device,
        compute_dtype=dtype,
        generator=generator,
    )
    del conditional, unconditional, lora_networks
    _empty_cache(device)

    # 3. Decode via the raw VAE decoder (latent_denorm -> unpatchify -> decoder; never BN).
    autoencoder = load_ideogram4_autoencoder(args.vae, dtype=dtype, device=device)
    with torch.no_grad():
        pixels = decode_dit_tokens_to_pixels(autoencoder, z, grid_h=grid_h, grid_w=grid_w)  # (B, 3, H, W) in [0, 1]

    image = (pixels[0].clamp(0.0, 1.0).float().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(args.save_path)
    logger.info(f"Saved {args.save_path} ({width}x{height})")


if __name__ == "__main__":
    main()
