import argparse
import gc
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
from musubi_tuner.ideogram4.caption_verifier import verify_caption
from musubi_tuner.ideogram4.generation import denoise_ideogram4_to_tokens
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.ideogram4.scheduler import get_schedule_for_resolution
from musubi_tuner.ideogram4.sequence import IDEOGRAM4_IMAGE_PATCH
from musubi_tuner.ideogram4.text_encoder import (
    TEXT_ENCODER_FORMATS,
    encode_prompt_to_features,
    load_ideogram4_text_encoder,
    load_ideogram4_tokenizer,
)
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

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
        self._sample_unconditional_dit = None  # loaded only during sampling (on_before/after_sample_images)

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

        # Sampling-during-training needs resources normal training doesn't load (Qwen3-VL encoder, the separate
        # unconditional DiT, the VAE). Require them up front — only when --sample_prompts is set — so a misconfig
        # fails before the run instead of at the first sample interval.
        if getattr(args, "sample_prompts", None):
            missing = [
                flag
                for flag, val in (
                    ("--vae", getattr(args, "vae", None)),
                    ("--unconditional_dit", getattr(args, "unconditional_dit", None)),
                    ("--text_encoder", getattr(args, "text_encoder", None)),
                    ("--text_encoder_config", getattr(args, "text_encoder_config", None)),
                    ("--tokenizer", getattr(args, "tokenizer", None)),
                )
                if not val
            ]
            if missing:
                raise ValueError(
                    f"Ideogram 4 sampling (--sample_prompts) requires {', '.join(missing)}: the Qwen3-VL encoder "
                    "(--text_encoder/--text_encoder_config/--tokenizer), the separate --unconditional_dit, and the "
                    "--vae for decode. Samples degenerate below ~1024 — set width/height=1024 in the prompt file."
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
        # Encode each UNIQUE sample prompt once through Qwen3-VL, attach CPU float32 features as
        # i4_llm_features, then free the TE so the ~8.78 GB encoder stays out of the training loop.
        logger.info(f"Ideogram 4: encoding sample prompts from {sample_prompts}")
        prompts = load_prompts(sample_prompts)
        device = accelerator.device
        text_encoder = load_ideogram4_text_encoder(
            args.text_encoder,
            args.text_encoder_config,
            text_encoder_format=args.text_encoder_format,
            dtype=self.dit_dtype,
            loading_device=device,
        )
        tokenizer = load_ideogram4_tokenizer(args.tokenizer)
        features_by_prompt: dict[str, torch.Tensor] = {}
        for prompt_dict in prompts:
            prompt = prompt_dict.get("prompt", "")
            verify_caption(prompt, strict=getattr(args, "strict_caption_verifier", False))
            if prompt not in features_by_prompt:
                feats = encode_prompt_to_features(tokenizer, text_encoder, prompt, device)  # (L, 53248)
                features_by_prompt[prompt] = feats.to(device="cpu", dtype=torch.float32)
            prompt_dict["i4_llm_features"] = features_by_prompt[prompt]
        del text_encoder, tokenizer
        clean_memory_on_device(device)
        logger.info(f"Ideogram 4: encoded {len(features_by_prompt)} unique prompt(s) for sampling")
        return prompts

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        # Ideogram uses a named preset (resolution-aware schedule + guidance schedule) and a SEPARATE
        # unconditional DiT, so the generic discrete_flow_shift / sample_steps / guidance_scale / cfg_scale /
        # negative_prompt knobs do not apply. width/height/seed (via generator) work normally.
        if sample_parameter.get("negative_prompt"):
            logger.warning(
                "Ideogram 4 ignores negative_prompt: it uses the separate unconditional DiT (asymmetric CFG), "
                "not a negative text prompt."
            )
        device = accelerator.device
        vae.to(device)  # base loads the VAE on CPU and returns it there afterward (trainer_base.py:1051)
        preset = PRESETS[args.sampler_preset]
        text_features = sample_parameter["i4_llm_features"].to(device=device, dtype=dit_dtype)
        # The unconditional DiT is loaded in on_before_sample_images; reload it if a prior prompt freed it.
        if self._sample_unconditional_dit is None:
            self._sample_unconditional_dit = ideogram4_utils.load_ideogram4_transformer(
                args.unconditional_dit, dtype=self.dit_dtype, loading_device=device
            )
        # Denoise with both DiTs, then FREE the unconditional DiT BEFORE the VAE decode. Keeping both 9.4 GB DiTs
        # resident through a 1024 decode peaks ~30 GB and OOMs a 32 GB card; freeing first caps the peak ~21 GB
        # (cond = unwrapped training transformer, LoRA ACTIVE — so samples reflect the current adapter).
        z, grid_h, grid_w = denoise_ideogram4_to_tokens(
            transformer,
            self._sample_unconditional_dit,
            text_features,
            height=height,
            width=width,
            preset=preset,
            device=device,
            compute_dtype=dit_dtype,
            generator=generator,
        )
        self._sample_unconditional_dit = None
        gc.collect()
        clean_memory_on_device(device)
        with torch.no_grad():
            pixels = ideogram4_utils.decode_dit_tokens_to_pixels(vae, z, grid_h=grid_h, grid_w=grid_w)  # (B,3,H,W) [0,1]
        return pixels.unsqueeze(2).cpu()  # (B, 3, 1, H, W) — the base saver keys on shape[2] == 1

    def on_before_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        # Load the unconditional DiT only during sampling (~9.4 GB fp8; resident all run would waste headroom).
        # Freed in on_after_sample_images, which the base runs in a `finally`.
        logger.info(f"Ideogram 4: loading unconditional DiT for sampling from {args.unconditional_dit}")
        self._sample_unconditional_dit = ideogram4_utils.load_ideogram4_transformer(
            args.unconditional_dit, dtype=self.dit_dtype, loading_device=accelerator.device
        )

    def on_after_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        # Free the unconditional DiT even if sampling raised (guard a partial/OOM'd on_before load), so VRAM
        # does not ratchet across sample intervals.
        if getattr(self, "_sample_unconditional_dit", None) is not None:
            self._sample_unconditional_dit = None
            clean_memory_on_device(accelerator.device)

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
    # Sampling-during-training (only used when --sample_prompts is set). --vae is a base arg, reused for decode.
    parser.add_argument(
        "--unconditional_dit", type=str, default=None, help="unconditional DiT weights for sampling (asymmetric CFG)"
    )
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-VL TE weights for sample-prompt encoding")
    parser.add_argument("--text_encoder_config", type=str, default=None, help="local Qwen3-VL config dir for sampling")
    parser.add_argument("--tokenizer", type=str, default=None, help="local Qwen3-VL tokenizer dir for sampling")
    parser.add_argument("--text_encoder_format", type=str, default="hf_full", choices=TEXT_ENCODER_FORMATS)
    parser.add_argument(
        "--sampler_preset",
        type=str,
        default="V4_TURBO_12",
        choices=list(PRESETS),
        help="named sampler preset for training samples (default V4_TURBO_12, a fast health check)",
    )
    parser.add_argument("--strict_caption_verifier", action="store_true", help="error (not warn) on caption-format issues")
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
