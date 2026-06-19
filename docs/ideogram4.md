# Ideogram 4

Ideogram 4 is an fp8-native text-to-image diffusion transformer with a Qwen3-VL text encoder.
Blissful Tuner provides a full LoRA / LoHa / LoKr training and inference pipeline for it, including
the blissful differentiators (mask-weighted loss, prior preservation with an EMA teacher, rich loss
diagnostics, and a ComfyUI LoRA converter).

> Status: actively developed. This doc covers the workflow and the Ideogram-specific options; for the
> shared options it links to the existing references rather than repeating them.

## Model notes

- **fp8-native weights.** Ideogram 4 is published only in quantized form (fp8 / nvfp4); there is no
  official bf16 release. Blissful loads the pre-quantized fp8 DiT directly through a dedicated
  pre-quantized state-dict normalizer and the shared monkey-patch fp8 path (`apply_fp8_monkey_patch`),
  dequantizing to the compute dtype in the forward. Because fp8 loading is automatic, `--fp8_base` /
  `--fp8_scaled` do **not** drive the Ideogram DiT load path and are neutralized (with a warning) if
  passed — do not rely on them here. You do not need a bf16 checkpoint; a "bf16" community re-upload
  is a dequant of the fp8 weights and recovers no precision.
- **Qwen3-VL text encoder.** Conditioning consumes only the Qwen3-VL *language model*; the vision
  tower is currently still loaded but unused for conditioning. The text-encoder forward drives the
  decoder layers manually to capture the raw pre-final-norm hidden states that `output_hidden_states`
  does not expose. Blissful tracks transformers `main`, and the causal-mask call adapts to the
  installed `create_causal_mask` signature (the `input_embeds`/`inputs_embeds` rename and the optional
  `cache_position`) so the manual decoder-drive stays correct across transformers versions. The manual
  drive is checked bit-for-bit against the native transformers decoder forward at the 12 intermediate
  taps by `tests/test_ideogram4_text_encoder.py::test_manual_tap_matches_native_hidden_states`.
- **VAE.** Ideogram 4 uses the FLUX.2 VAE (`flux2-vae.safetensors`).
- **Native resolution** is 1024×1024 (samples degenerate well below it).
- **Attention (batch-1 fast path).** For a single sample the block-diagonal attention mask is all-True,
  so it is elided and SDPA selects the flash backend (~26% faster training step on the settled recipe,
  measured A/B). This is a backend transition (mem-efficient → flash), so batch-1 attention numerics
  changed at that commit — runs trained before vs after it are not bitwise-comparable (no metadata flag:
  the SDPA backend already depends on torch/CUDA/hardware; the checkout boundary is the thing to track).
  batch>1 keeps the exact mask.

## Pipeline

Ideogram 4 follows the standard four-step Musubi flow. Use a fresh `cache_directory` whenever you
change mask sources or resolution.

```bash
# 1) Cache VAE latents (FLUX.2 VAE). Latents are normalized into Ideogram's DiT-token space at cache time.
python ideogram4_cache_latents.py --dataset_config config.toml --vae /path/to/flux2-vae.safetensors

# 2) Cache Qwen3-VL text-encoder outputs (the encoder weights + its local config dir + tokenizer dir)
python ideogram4_cache_text_encoder_outputs.py --dataset_config config.toml \
    --text_encoder /path/to/qwen3_vl/model.safetensors \
    --text_encoder_config /path/to/qwen3_vl_config_dir \
    --tokenizer /path/to/qwen3_vl_tokenizer_dir

# 3) Train a LoRA. With cached latents + TE, the DiT is the ONLY model loaded for plain training.
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    ideogram4_train_network.py \
    --dit /path/to/ideogram4_fp8.safetensors \
    --dataset_config config.toml \
    --network_module networks.lora_ideogram4 \
    --network_dim 32

# 4) Generate. Needs BOTH the conditional (--dit) and unconditional (--unconditional_dit) DiT.
python ideogram4_generate_image.py \
    --dit /path/to/ideogram4_fp8.safetensors \
    --unconditional_dit /path/to/ideogram4_unconditional_fp8.safetensors \
    --vae /path/to/flux2-vae.safetensors \
    --text_encoder /path/to/qwen3_vl/model.safetensors \
    --text_encoder_config /path/to/qwen3_vl_config_dir \
    --tokenizer /path/to/qwen3_vl_tokenizer_dir \
    --prompt "your prompt" \
    --save_path /path/to/out.png \
    --lora_weight /path/to/lora.safetensors
```

### Sampling during training

Sampling during training (`--sample_prompts`) additionally requires the Qwen3-VL encoder, the FLUX.2
VAE, and the separate unconditional DiT (validated up front so a misconfig fails before the run):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    ideogram4_train_network.py \
    --dit /path/to/ideogram4_fp8.safetensors \
    --dataset_config config.toml \
    --network_module networks.lora_ideogram4 --network_dim 32 \
    --sample_prompts /path/to/prompts.txt \
    --unconditional_dit /path/to/ideogram4_unconditional_fp8.safetensors \
    --vae /path/to/flux2-vae.safetensors \
    --text_encoder /path/to/qwen3_vl/model.safetensors \
    --text_encoder_config /path/to/qwen3_vl_config_dir \
    --tokenizer /path/to/qwen3_vl_tokenizer_dir
```

`networks.loha` / `networks.lokr` are also supported (Ideogram 4 is registered in the LoHa/LoKr
architecture registry). See `docs/loha_lokr.md`.

## Ideogram-specific training options

| Flag | Default | Meaning |
|------|---------|---------|
| `--ideogram4_timestep_std` | `1.0` | std of the logit-normal training-timestep schedule. Larger values push more mass into the low-noise / detail-refinement tail (e.g. `1.5` ≈ 11.6% of steps below traditional noise timestep 250 on the [0, 1000] convention, vs ≈ 3.6% at `1.0`, same median). A worthwhile A/B knob; `1.0` is the blissful default. |
| `--ideogram4_timestep_mu` | resolution-derived | known mean for the logit-normal schedule (resolution-aware by default). |
| `--ideogram4_sample_guidance` | unset | overrides the CFG scale for **training-time sample images only** (must be finite and > 0). Does not affect the trained weights. |
| `--ideogram4_fp32_timestep` | off (legacy bf16) | keep the timestep in fp32 through the DiT time-embedding (corrected conditioning). See **Timestep precision** below. |

> Note: `--fp8_base` / `--fp8_scaled` are accepted for CLI compatibility but ignored for Ideogram 4
> (fp8 loading is automatic via the pre-quantized shim). They emit a warning if set.

### Timestep precision

The DiT forward historically cast the flow-matching timestep `t` to the bf16 compute dtype before the
time embedding. Because `Ideogram4EmbedScalar` re-upcasts to fp32 and multiplies by a 1e4 sinusoidal
scale, that cast is a lossy round-trip whose error is amplified into the embedding (a standalone probe
measured ~0.79 cosine vs the fp32 embedding). `--ideogram4_fp32_timestep` keeps `t` in fp32, matching the
precision the frozen base was trained for. It changes **only** the sinusoidal computation — every
downstream tensor keeps the same dtype either way.

- **Default is legacy bf16** (behavior-preserving): every pre-2026-06 Ideogram adapter was trained *and*
  sampled under the bf16 cast, so they remain bit-consistent. A/B the fp32 regime before adopting it
  widely. The trainer flag is `--ideogram4_fp32_timestep` / `--no-ideogram4_fp32_timestep`.
- **The regime is recorded** in the LoRA metadata (`ss_ideogram4_fp32_timestep`), and is retained even
  under `--no_metadata` (it governs how the checkpoint must be run, not optional provenance).
- **Generation auto-inherits it.** `ideogram4_generate_image.py` reads that metadata and matches each
  adapter's trained regime by default, avoiding a silent train/inference mismatch. When the regime is
  genuinely ambiguous it refuses to guess and fails fast with a one-line fix:
    - a stack whose adapters disagree (one fp32, one legacy), or an fp32 adapter mixed with an
      **unstamped** adapter (which may be an old bf16 adapter *or* an upstream-native fp32 one), → error;
    - all-unstamped or all-legacy → legacy bf16 (with a warning for unstamped);
    - a malformed stamp → error.
  Pass `--ideogram4_fp32_timestep` / `--no-ideogram4_fp32_timestep` to force the regime explicitly (this
  short-circuits metadata resolution — the escape hatch for mixed/unstamped stacks and A/Bs). Note the
  intentional asymmetry: the trainer flag *defines* the regime (default legacy), the generator flag
  *inherits* it (default auto-from-metadata).

## Mask-weighted loss & prior preservation

Ideogram 4 fully supports blissful's mask-weighted loss and prior preservation (base or EMA teacher),
including per-sample prior scheduling and the raw pre-Huber/pre-mask loss diagnostics. Cache masks via
`mask_directory` or `alpha_mask` in the dataset config, and train with `--use_mask_loss` (+ the prior
flags). See `docs/MASKED_LOSS_TRAINING_GUIDE.md` for the comprehensive reference and
`docs/NETWORK_ARGS_REFERENCE.md` for the network arguments.

## ComfyUI

Convert a trained Ideogram 4 LoRA to ComfyUI format with
`src/musubi_tuner/networks/convert_ideogram4_lora_to_comfy.py` (handles LoRA / DoRA / rsLoRA / LoHa /
LoKr key remapping).
