import argparse
import logging
from typing import Optional

import torch

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_IDEOGRAM4, ARCHITECTURE_IDEOGRAM4_FULL
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.scheduler import get_schedule_for_resolution
from musubi_tuner.ideogram4.sequence import IDEOGRAM4_IMAGE_PATCH
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Ideogram4NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ):
        """Ideogram 4 flow-matching loss on the generate-VALIDATED convention.

        latents: cached grid (B, 128, gh, gw), already patchified + latent_norm'd (scale_shift_latents is a
        no-op). batch["i4_llm_features"]: varlen list of (L_text_i, 53248) cached Qwen3-VL features.
        """
        if getattr(args, "use_mask_loss", False):
            raise ValueError(
                "Ideogram 4 does not support --use_mask_loss yet (the latent cache writes no mask_weights). "
                "Remove --use_mask_loss, or add mask caching + token-grid mask patchify first."
            )

        device = accelerator.device
        text_features = batch["i4_llm_features"]
        grid_h, grid_w = int(latents.shape[2]), int(latents.shape[3])
        height, width = grid_h * IDEOGRAM4_IMAGE_PATCH, grid_w * IDEOGRAM4_IMAGE_PATCH

        # Sample t (cleanness coefficient) from the resolution-aware logit-normal schedule (canonical).
        schedule = get_schedule_for_resolution(
            (height, width), known_mean=args.ideogram4_timestep_mu, std=args.ideogram4_timestep_std
        )
        timesteps = schedule(torch.rand(latents.shape[0], device=device))

        model_pred, target = ideogram4_flow_matching_target(
            transformer, latents, text_features, noise, timesteps, network_dtype=network_dtype, device=device
        )
        loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target.to(network_dtype), reduction="mean")
        return loss, {}

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_IDEOGRAM4

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_IDEOGRAM4_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 7.0  # Ideogram uses asymmetric CFG at inference; unused at train time
        self.default_discrete_flow_shift = 1.0

    def scale_shift_latents(self, latents):
        # Cached Ideogram latents are already latent_norm'd (grid_to_dit_tokens flattens, never re-normalizes).
        return latents

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        # Latents are cached, so the VAE is only needed for sampling-during-training (deferred). Load the
        # FLUX.2 VAE on CPU so it does not occupy training VRAM.
        return ideogram4_utils.load_ideogram4_autoencoder(vae_path, dtype=vae_dtype, device="cpu")

    def load_transformer(
        self,
        accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        # Ideogram 4 attention is SDPA-only (Blackwell guard is active); attn_mode/split_attn are not used.
        return ideogram4_utils.load_ideogram4_transformer(
            dit_path,
            dtype=self.dit_dtype,
            loading_device=loading_device,
            disable_numpy_memmap=getattr(args, "disable_numpy_memmap", False),
        )

    def call_dit(self, *args, **kwargs):
        # Ideogram 4 computes its loss directly in process_batch (custom joint-sequence flow-matching), so the
        # base call_dit path is not used.
        raise NotImplementedError("Ideogram 4 computes loss in process_batch; call_dit is not used.")

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        raise NotImplementedError(
            "Sampling-during-training is not yet wired for Ideogram 4. Train without --sample_prompts and "
            "generate with ideogram4_generate_image.py instead."
        )

    # endregion


def ideogram4_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--ideogram4_timestep_mu", type=float, default=0.0, help="logit-normal schedule mean for training timesteps"
    )
    parser.add_argument(
        "--ideogram4_timestep_std", type=float, default=1.0, help="logit-normal schedule std for training timesteps"
    )
    # Ideogram 4 weights are always fp8 (loaded via the pre-quantized shim), so these are accepted for base-loop
    # compatibility but do not change the load path; the DiT is fp8 regardless.
    parser.add_argument("--fp8_scaled", action="store_true", help="accepted for compatibility (Ideogram DiT is fp8)")
    return parser


def main():
    parser = setup_parser_common()
    parser = ideogram4_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = Ideogram4NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
