import argparse
import logging

import numpy as np
import torch
from PIL import Image

from musubi_tuner.ideogram4.caption_verifier import verify_caption
from musubi_tuner.ideogram4.generation import denoise_ideogram4_tokens
from musubi_tuner.ideogram4.ideogram4_utils import (
    decode_dit_tokens_to_pixels,
    load_ideogram4_autoencoder,
    load_ideogram4_transformer,
)
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.ideogram4.scheduler import get_schedule_for_resolution
from musubi_tuner.ideogram4.sequence import build_ideogram4_conditioning, image_grid_dims
from musubi_tuner.ideogram4.text_encoder import (
    TEXT_ENCODER_FORMATS,
    encode_prompt_to_features,
    load_ideogram4_text_encoder,
    load_ideogram4_tokenizer,
)
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    return parser


def main():
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = str_to_dtype(args.dtype)
    height, width = args.image_size
    preset = PRESETS[args.sampler_preset]
    grid_h, grid_w = image_grid_dims(height, width)

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

    sequence = build_ideogram4_conditioning([features], grid_h, grid_w, device=device, dtype=dtype)

    # 2. Load both DiTs, run the asymmetric-CFG denoise, then free them before decoding.
    logger.info("Loading conditional + unconditional DiTs")
    conditional = load_ideogram4_transformer(args.dit, dtype=dtype, loading_device=device)
    unconditional = load_ideogram4_transformer(args.unconditional_dit, dtype=dtype, loading_device=device)

    schedule = get_schedule_for_resolution((height, width), known_mean=preset.mu, std=preset.std)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    logger.info(f"Denoising: {preset.num_steps} steps, preset {args.sampler_preset}")
    z = denoise_ideogram4_tokens(
        conditional,
        unconditional,
        sequence,
        num_steps=preset.num_steps,
        guidance_schedule=preset.guidance_schedule,
        schedule=schedule,
        device=device,
        compute_dtype=dtype,
        generator=generator,
    )
    del conditional, unconditional
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
