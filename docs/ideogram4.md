# Ideogram 4

Ideogram 4 is an fp8-native text-to-image diffusion transformer with a Qwen3-VL text encoder.
Blissful Tuner provides a full LoRA / LoHa / LoKr training and inference pipeline for it, including
the blissful differentiators (mask-weighted loss, prior preservation with an EMA teacher, rich loss
diagnostics, and a ComfyUI LoRA converter).

> Status: actively developed. This doc covers the workflow and the Ideogram-specific options; for the
> shared options it links to the existing references rather than repeating them.

## Model notes

- **fp8-native weights.** Ideogram 4 is published only in quantized form (fp8 / nvfp4); there is no
  official bf16 release. Blissful loads a pre-quantized fp8 DiT directly — the loader does a cheap
  header peek and routes ComfyUI-style fp8 (per-tensor scales) and official fp8 (per-row scales)
  through the shared monkey-patch fp8 path (`apply_fp8_monkey_patch`), dequantizing to the compute
  dtype in the forward. You do **not** need a bf16 checkpoint; a "bf16" community re-upload is a
  dequant of the fp8 weights and recovers no precision.
- **Qwen3-VL text encoder.** Conditioning consumes only the Qwen3-VL *language model*; the vision
  tower is unused. The text-encoder forward drives the decoder layers manually to capture the
  raw pre-final-norm hidden states that `output_hidden_states` does not expose. Blissful tracks
  transformers `main`, and the causal-mask call adapts to the installed `create_causal_mask`
  signature (the `input_embeds`/`inputs_embeds` rename and the optional `cache_position`) so the
  manual decoder-drive stays correct across transformers versions. Native-layer parity is guarded by
  `tests/test_ideogram4_text_encoder.py`.
- **Native resolution** is 1024×1024.

## Pipeline

Ideogram 4 follows the standard four-step Musubi flow. Use a fresh `cache_directory` whenever you
change mask sources or resolution.

```bash
# 1) Cache VAE latents (already normalized into Ideogram's DiT-token latent space at cache time)
python ideogram4_cache_latents.py --dataset_config config.toml --vae /path/to/ideogram4_vae.safetensors

# 2) Cache Qwen3-VL text-encoder outputs
python ideogram4_cache_text_encoder_outputs.py --dataset_config config.toml \
    --text_encoder /path/to/qwen3_vl_text_encoder.safetensors

# 3) Train a LoRA
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    ideogram4_train_network.py \
    --dit /path/to/ideogram4_fp8.safetensors \
    --text_encoder /path/to/qwen3_vl_text_encoder.safetensors \
    --vae /path/to/ideogram4_vae.safetensors \
    --dataset_config config.toml \
    --network_module networks.lora_ideogram4 \
    --network_dim 32 \
    --fp8_scaled

# 4) Generate
python ideogram4_generate_image.py \
    --dit /path/to/ideogram4_fp8.safetensors \
    --text_encoder /path/to/qwen3_vl_text_encoder.safetensors \
    --vae /path/to/ideogram4_vae.safetensors \
    --prompt "your prompt" \
    --lora_weight /path/to/lora.safetensors
```

`networks.loha` / `networks.lokr` are also supported (Ideogram 4 is registered in the LoHa/LoKr
architecture registry). See `docs/loha_lokr.md`.

## Ideogram-specific training options

| Flag | Default | Meaning |
|------|---------|---------|
| `--ideogram4_timestep_std` | `1.0` | std of the logit-normal training-timestep schedule. Larger values push more mass into the low-noise / detail-refinement tail (e.g. `1.5` ≈ 11.6% of steps below noise-t 0.25 vs ≈ 3.6% at `1.0`, same median). A worthwhile A/B knob; `1.0` is the blissful default. |
| `--ideogram4_timestep_mu` | resolution-derived | known mean for the logit-normal schedule (resolution-aware by default). |
| `--ideogram4_sample_guidance` | unset | overrides the CFG scale for **training-time sample images only** (must be finite and > 0). Does not affect the trained weights. |
| `--fp8_scaled` | off | per-layer scaled fp8 for the DiT (recommended for Ideogram, which is fp8-native). |

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
