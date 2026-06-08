import argparse
import logging

import numpy as np
import torch
from PIL import Image

from musubi_tuner.ideogram4.caption_verifier import verify_caption
from musubi_tuner.ideogram4.generation import generate_ideogram4_pixels
from musubi_tuner.ideogram4.ideogram4_utils import load_ideogram4_autoencoder, load_ideogram4_transformer
from musubi_tuner.ideogram4.sampler_configs import PRESETS
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

    # 2. Load both DiTs + the VAE, then generate via the shared helper — the single source of the validated
    #    B5 t-convention (same path as sampling-during-training). All three resident; ~21 GB peak on the 5090.
    logger.info("Loading conditional + unconditional DiTs + VAE")
    conditional = load_ideogram4_transformer(args.dit, dtype=dtype, loading_device=device)
    unconditional = load_ideogram4_transformer(args.unconditional_dit, dtype=dtype, loading_device=device)
    autoencoder = load_ideogram4_autoencoder(args.vae, dtype=dtype, device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    logger.info(f"Denoising: {preset.num_steps} steps, preset {args.sampler_preset}")
    pixels = generate_ideogram4_pixels(
        conditional,
        unconditional,
        autoencoder,
        features,
        height=height,
        width=width,
        preset=preset,
        device=device,
        compute_dtype=dtype,
        generator=generator,
    )  # (B, 3, H, W) in [0, 1]

    image = (pixels[0].clamp(0.0, 1.0).float().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(args.save_path)
    logger.info(f"Saved {args.save_path} ({width}x{height})")


if __name__ == "__main__":
    main()
