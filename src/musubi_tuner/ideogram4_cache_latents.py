import logging
from typing import List

import torch
import torch.nn.functional as F

from musubi_tuner import cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_IDEOGRAM4, ItemInfo, save_latent_cache_ideogram4
from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents(batch: List[ItemInfo]) -> torch.Tensor:
    """Stack a batch of items into pixels (B, 3, H, W) in [0, 1]."""
    contents = []
    for item in batch:
        content = item.content[0] if isinstance(item.content, list) else item.content
        contents.append(torch.from_numpy(content[..., :3]))  # RGB for the VAE; masks come from item.mask_content
    contents = torch.stack(contents, dim=0).permute(0, 3, 1, 2).contiguous()
    return contents.to(torch.float32) / 255.0


def encode_and_save_batch(autoencoder, batch: List[ItemInfo]):
    pixels = preprocess_contents(batch)  # (B, 3, H, W) in [0, 1]
    with torch.no_grad():
        # encode_pixels_to_latent_grid: raw encoder MEAN -> patchify (pi,pj,c) -> latent_norm -> (B, 128, gh, gw).
        # The grid is already DiT-ready; the trainer flattens (grid_to_dit_tokens), never re-patchifies/normalizes.
        grid, _, _ = ideogram4_utils.encode_pixels_to_latent_grid(autoencoder, pixels)

    for b, item in enumerate(batch):
        latent = grid[b].detach().to(torch.bfloat16).cpu()  # (128, gh, gw)

        # Mask-weighted loss: downsample the per-item weighted mask to the DiT-token grid (gh, gw) — the SAME
        # spatial dims as the cached latent — and stash it compact (1, 1, gh, gw). Only gamma/min_weight are
        # baked at cache time; blur/area-beta etc. stay at training time. Mirrors flux_2_cache_latents.
        mask_weights = None
        if item.mask_content is not None:
            mask = torch.from_numpy(item.mask_content).unsqueeze(0).unsqueeze(0).float() / 255.0  # (1,1,H,W)
            mask = mask.clamp_(0.0, 1.0)
            mask = cache_latents.apply_cache_mask_transforms(
                mask,
                cache_mask_gamma=float(getattr(item, "cache_mask_gamma", 1.0) or 1.0),
                cache_mask_min_weight=float(getattr(item, "cache_mask_min_weight", 0.0) or 0.0),
            )
            gh, gw = latent.shape[-2:]
            mask_weights = F.interpolate(mask, size=(gh, gw), mode="area")  # (1, 1, gh, gw)

        logger.info(
            f"Saving Ideogram 4 latent cache for {item.item_key}: {tuple(latent.shape)}"
            + (f" + mask {tuple(mask_weights.shape)}" if mask_weights is not None else "")
        )
        save_latent_cache_ideogram4(item, latent, mask_weights=mask_weights)


def main():
    parser = cache_latents.setup_parser_common()
    args = parser.parse_args()

    if args.vae is None:
        raise ValueError("--vae is required for Ideogram 4 latent caching (FLUX.2 VAE, e.g. flux2-vae.safetensors)")

    device = args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_IDEOGRAM4)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=1
        )
        return

    logger.info(f"Loading FLUX.2 VAE for Ideogram 4 from {args.vae}")
    autoencoder = ideogram4_utils.load_ideogram4_autoencoder(args.vae, device=device, dtype=vae_dtype)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(autoencoder, batch)

    # supports_alpha=False: the VAE encodes RGB (pixel alpha dropped); weighted masks come from item.mask_content
    # (mask_directory / alpha_mask), downsampled per-item in encode_and_save_batch and written as mask_weights_*.
    cache_latents.encode_datasets(datasets, encode, args, supports_alpha=False)


if __name__ == "__main__":
    main()
