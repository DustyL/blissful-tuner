import argparse
import logging
import os
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


def neutralize_unused_fp8_args(args) -> bool:
    """Ideogram 4 always loads its DiT as fp8 via the pre-quantized shim, so ``--fp8_base`` / ``--fp8_scaled``
    do not drive the load path (``load_transformer`` ignores ``dit_weight_dtype``). Neutralize both to False
    (warning if either was set) so the base ``fp8_scaled requires fp8_base`` assert — which fires inside
    ``_validate_args_and_init`` before our model-specific validation — cannot abort the run on a no-op flag.

    Returns True if either flag had been set (for telemetry/tests).
    """
    had = bool(getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False))
    if had:
        logger.warning(
            "Ideogram 4 loads fp8 weights via its pre-quantized shim; --fp8_base/--fp8_scaled have no effect here and are ignored."
        )
    args.fp8_base = False
    args.fp8_scaled = False
    return had


class Ideogram4NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    def _validate_args_and_init(self, args) -> bool:
        # Neutralize the no-op fp8 flags BEFORE the base fp8 assert (which lives inside the base
        # _validate_args_and_init at trainer_base.py:1443). Done here rather than in main() so it holds for
        # ANY caller of train() — CLI, GUI, or a test that constructs the trainer directly.
        neutralize_unused_fp8_args(args)
        return super()._validate_args_and_init(args)

    def _build_dataset(self, args):
        # Build the dataset, then preflight every Ideogram latent cache BEFORE the heavy TE/DiT load. The
        # shared latent reader (bucket.py) loads tensors via load_file() and never inspects safetensors
        # metadata, so a stale / raw / fork-format cache would otherwise enter training silently. This runs
        # after dataset construction (trainer_base.py:1362) and before model load (1365).
        result = super()._build_dataset(args)
        self._preflight_latent_caches(result[0])
        return result

    def _preflight_latent_caches(self, train_dataset_group) -> int:
        """Validate every ACTIVE Ideogram latent cache (norm-applied, grid layout, dit-token space, shape).

        Raises ValueError (from preflight_ideogram4_latent_cache) on the first stale / raw / fork-format cache,
        aborting before the model load. Returns the number of distinct cache files validated.

        Active set only: the cached-dataset path (image_video_dataset.py:644) skips any latent cache that has
        no paired text-encoder cache, so preflight mirrors that pairing — otherwise a stale orphan
        ``*_i4.safetensors`` (which training would ignore) could abort an otherwise-valid run.
        """
        seen: set[str] = set()
        skipped_orphans = 0
        for dataset in train_dataset_group.datasets:
            cache_dir = getattr(dataset, "cache_directory", None)
            if cache_dir is None:
                continue
            arch = getattr(dataset, "architecture", self.architecture)
            for cache_path in dataset.get_all_latent_cache_files():
                if cache_path in seen:
                    continue
                # Pairing key = item_key (basename minus the trailing _{WxH}_{arch} tokens), exactly as the
                # dataset derives it; TE cache is {item_key}_{arch}_te.safetensors in the same directory.
                tokens = os.path.basename(cache_path).split("_")
                item_key = "_".join(tokens[:-2])
                te_path = os.path.join(cache_dir, f"{item_key}_{arch}_te.safetensors")
                if not os.path.exists(te_path):
                    skipped_orphans += 1
                    continue
                seen.add(cache_path)
                ideogram4_utils.preflight_ideogram4_latent_cache(cache_path)
        msg = f"Ideogram 4 latent-cache preflight passed: {len(seen)} active cache file(s) validated."
        if skipped_orphans:
            msg += f" ({skipped_orphans} orphan latent cache(s) without a paired TE cache skipped — training ignores them.)"
        logger.info(msg)
        return len(seen)

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
        # Backstop only — the authoritative fail-fast rejection is in handle_model_specific_args (setup time,
        # before the model load). Kept here so a direct process_batch caller (test/future path) still errors.
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
        # This hook runs inside _validate_args_and_init (the FIRST phase of train()), before dataset build /
        # accelerator setup / model load — so it is the right place to fail fast on unsupported config and to
        # pin the compute dtype, rather than discovering the problem deep in the training loop.

        if getattr(args, "use_mask_loss", False):
            raise ValueError(
                "Ideogram 4 does not support --use_mask_loss yet (the latent cache writes no mask_weights). "
                "Remove --use_mask_loss, or add mask caching + token-grid mask patchify first."
            )
        if getattr(args, "gradient_checkpointing_cpu_offload", False):
            raise ValueError(
                "Ideogram 4 supports --gradient_checkpointing, but not --gradient_checkpointing_cpu_offload yet: "
                "the CPU-offload backward path needs a dedicated CUDA test before it can be trusted in training "
                "(and its return-device interaction with block swap is unvalidated). Use --gradient_checkpointing alone."
            )
        if getattr(args, "blocks_to_swap", 0):
            raise ValueError(
                "Ideogram 4 does not support --blocks_to_swap yet: the vendored modeling_ideogram4 has no "
                "block-swap hooks (base training would call transformer.enable_block_swap()). Remove the flag "
                "and train at a lower resolution, or add block swap to modeling_ideogram4."
            )
        if getattr(args, "compile", False):
            raise ValueError(
                "Ideogram 4 does not support --compile yet: it has no compile_transformer hook, so the base "
                "default raises NotImplementedError (trainer_base.py:1127) only AFTER accelerator.prepare — a "
                "confusing late crash. Remove --compile; a compile_transformer over [transformer.layers] can be "
                "added later (see docs/plans/2026-06-07-ideogram4-native-1024-gc-blockswap.md)."
            )

        # args.mixed_precision is filled from the accelerate config LATER (trainer_base.py:1516); when this
        # hook runs it is still None if the CLI omitted it. Default the omitted case to bf16 — fp32 would OOM
        # the 8B DiT AND split the loaded-model dtype (self.dit_dtype) from the training-loop dtype (the local
        # dit_dtype computed at trainer_base.py:1521). Setting it here fixes both, since both read this value.
        # Explicit --mixed_precision (fp16 / bf16 / no) is left untouched and takes precedence.
        if not args.mixed_precision:
            # WARNING (not INFO): this silently overrides the user's omitted choice and affects memory, and the
            # base loop configures the root logger so INFO from this module is suppressed. Make it visible.
            logger.warning(
                "Ideogram 4: --mixed_precision omitted; defaulting to bf16 (fp32 would OOM the 8B DiT). "
                "Pass --mixed_precision explicitly (no/fp16/bf16) to override."
            )
            args.mixed_precision = "bf16"

        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 7.0  # Ideogram uses asymmetric CFG at inference; unused at train time
        self.default_discrete_flow_shift = 1.0

    def extra_metadata(self, args) -> dict:
        # The Ideogram DiT is loaded as PRE-QUANTIZED fp8 via the shim, not the musubi fp8 path — so the base
        # loop records ss_fp8_base=False (we neutralize --fp8_base/--fp8_scaled). Record the real base-precision
        # provenance here so inventory/debug tools don't read the LoRA as trained from a full-precision base.
        metadata = super().extra_metadata(args)
        metadata["ss_ideogram4_prequantized_fp8"] = True
        metadata["ss_ideogram4_fp8_source"] = os.path.basename(args.dit) if getattr(args, "dit", None) else ""
        layout = getattr(self, "_ideogram4_fp8_layout", None)
        if layout:
            metadata["ss_ideogram4_fp8_scale_layout"] = layout  # per_row (HF) / per_tensor (Comfy) / ...
        return metadata

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
        transformer = ideogram4_utils.load_ideogram4_transformer(
            dit_path,
            dtype=self.dit_dtype,
            loading_device=loading_device,
            disable_numpy_memmap=getattr(args, "disable_numpy_memmap", False),
        )
        # Capture the real fp8 scale geometry for checkpoint provenance (see extra_metadata): the base loop
        # records ss_fp8_base=False because we neutralize the musubi fp8 flags, so the LoRA would otherwise
        # look like it was trained from a full-precision base.
        self._ideogram4_fp8_layout = ideogram4_utils.detect_fp8_scale_layout(transformer)
        return transformer

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
    # Ideogram 4 weights are always fp8 (loaded via the pre-quantized shim), so this is defined only so the base
    # loop can read args.fp8_scaled; it is neutralized (with a warning) in neutralize_unused_fp8_args(). --fp8_base
    # is already defined by the common parser and is neutralized the same way.
    parser.add_argument(
        "--fp8_scaled", action="store_true", help="no-op for Ideogram 4 (DiT is fp8 via the shim); accepted for compatibility"
    )
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
