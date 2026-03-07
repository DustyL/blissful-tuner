# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Blissful Tuner is an extended fork of Musubi Tuner (by kohya-ss) developed by Blyss Sarania. It provides training and inference tools for LoRA models with multiple video/image generation architectures: HunyuanVideo, HunyuanVideo 1.5, Wan 2.1/2.2, FramePack, FLUX.1 Kontext, FLUX.2, Z-Image-Turbo, Qwen-Image, and Kandinsky 5.

**Key Extensions over Musubi Tuner:**
- Rich logging with beautiful console output
- Latent preview during generation
- Advanced CFG scheduling and guidance methods (CFGZero*, NAG, perpendicular negative)
- V2V/I2V/I2I inference support
- RifleX for longer videos
- Prompt wildcards and weighting
- Weighted mask loss with prior preservation for region-focused training
- EMA teacher and timestep-adaptive prior scheduling
- Muon optimizer integration
- Gradio-based training GUI
- Post-processing tools (VFI, upscaling, face restoration)

## Code Style

- **Linter/Formatter**: Ruff (configured in `pyproject.toml`)
- **Line length**: 132 characters
- **Indentation**: 4 spaces
- **Quote style**: Double quotes
- **Target Python**: 3.10+ (< 3.13)

```bash
ruff check --fix
ruff format src tests
```

Avoid broad refactors/formatting in vendored code (`src/blissful_tuner/codeformer/`, `gfpgan/`, `gimmvfi/`, `swinir/`, `esrgan/`) and Ruff-excluded upstream model directories (see `pyproject.toml` `tool.ruff.extend-exclude`).

## Python Environment

- **Virtual environment**: `/Users/dustin/blissful-tuner/venv` (NOT `.venv`)
- Always use this project's venv — do not use the system Python or create a new virtual environment
- Activate: `source /Users/dustin/blissful-tuner/venv/bin/activate`
- Run tools with: `/Users/dustin/blissful-tuner/venv/bin/python`
- **No torch/rich in shell env**: Use `python -m py_compile` for syntax checks, not import-based verification.

## Common Commands

### Installation
```bash
# pip installation
pip install -e . --group dev --group postprocess

# uv installation
uv sync --extra cu128  # or cu124, cu129, cu130

# Optional extras
uv sync --extra lycoris      # LyCORIS backend
uv sync --extra gui          # Gradio GUI
```

### Testing
```bash
# Run all tests (58 test files)
pytest tests/

# Run a specific test file
pytest tests/test_mask_loss.py

# Smoke check (compileall, no imports)
python -m compileall -q src
```

### Training Pipeline (all architectures follow this pattern)

Each architecture uses 4 scripts: cache latents, cache text encoder outputs, train, generate.

```bash
# WAN 2.2 example
python wan_cache_latents.py --dataset_config config.toml \
    --vae /path/to/Wan2.1_VAE.pth --vae_chunk_size 32 --vae_tiling
python wan_cache_text_encoder_outputs.py --dataset_config config.toml \
    --t5 /path/to/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    wan_train_network.py \
    --task t2v-A14B \
    --dit /path/to/low_noise_model.safetensors \
    --dit_high_noise /path/to/high_noise_model.safetensors \
    --dataset_config config.toml \
    --network_module networks.lora_wan \
    --network_dim 32 \
    --timestep_sampling shift \
    --discrete_flow_shift 12.0
python wan_generate_video.py \
    --task t2v-A14B \
    --dit /path/to/dit.safetensors \
    --vae /path/to/Wan2.1_VAE.pth \
    --t5 /path/to/t5.pth \
    --prompt "your prompt" \
    --video_size 720 1280 \
    --video_length 81 \
    --lora_weight /path/to/lora.safetensors
```

### All Root-Level Scripts (45 thin wrappers importing from `src/musubi_tuner/`)

```bash
# WAN 2.1/2.2
wan_cache_latents.py | wan_cache_text_encoder_outputs.py | wan_train_network.py | wan_generate_video.py

# HunyuanVideo (no separate cache scripts — uses generic wrappers or hv_train_network directly)
hv_train_network.py | hv_generate_video.py | hv_train.py  # full fine-tune

# HunyuanVideo 1.5
hv_1_5_cache_latents.py | hv_1_5_cache_text_encoder_outputs.py | hv_1_5_train_network.py | hv_1_5_generate_video.py

# FramePack
fpack_cache_latents.py | fpack_cache_text_encoder_outputs.py | fpack_train_network.py | fpack_generate_video.py

# FLUX.1 Kontext
flux_kontext_cache_latents.py | flux_kontext_cache_text_encoder_outputs.py | flux_kontext_train_network.py | flux_kontext_generate_image.py

# FLUX.2
flux_2_cache_latents.py | flux_2_cache_text_encoder_outputs.py | flux_2_train_network.py | flux_2_generate_image.py

# Qwen-Image (LoRA + full fine-tune)
qwen_image_cache_latents.py | qwen_image_cache_text_encoder_outputs.py | qwen_image_train_network.py | qwen_image_generate_image.py
qwen_image_train.py          # full fine-tune

# Z-Image-Turbo (LoRA + full fine-tune)
zimage_cache_latents.py | zimage_cache_text_encoder_outputs.py | zimage_train_network.py | zimage_generate_image.py
zimage_train.py              # full fine-tune

# Kandinsky 5
kandinsky5_cache_latents.py | kandinsky5_cache_text_encoder_outputs.py | kandinsky5_train_network.py | kandinsky5_generate_video.py

# Generic (architecture-agnostic wrappers)
cache_latents.py | cache_text_encoder_outputs.py

# Utilities
merge_lora.py                # Merge LoRA into model
convert_lora.py              # Convert LoRA format (musubi/diffusers/comfy)
lora_post_hoc_ema.py         # Post-hoc EMA merge
convert_masks_to_alpha.py    # Convert image+mask to RGBA
caption_images_by_qwen_vl.py # Auto-caption images using Qwen-VL
qwen_extract_lora.py         # Extract LoRA from Qwen models
```

## Architecture

```
blissful-tuner/
├── src/
│   ├── musubi_tuner/           # Core training/inference framework
│   │   ├── wan/                # Wan 2.1/2.2 model (configs/, modules/)
│   │   ├── hunyuan_model/      # HunyuanVideo implementation
│   │   ├── hunyuan_video_1_5/  # HunyuanVideo 1.5
│   │   ├── frame_pack/         # FramePack model
│   │   ├── flux/               # FLUX.1 Kontext
│   │   ├── flux_2/             # FLUX.2
│   │   ├── qwen_image/         # Qwen-Image
│   │   ├── zimage/             # Z-Image-Turbo
│   │   ├── kandinsky5/         # Kandinsky 5 (configs, models/, generation_utils)
│   │   ├── gui/                # Gradio-based training GUI (i18n EN/JA)
│   │   ├── optimizers/         # Custom optimizers (Muon with Newton-Schulz orthogonalization)
│   │   ├── dataset/            # Dataset handling, config parsing, caching
│   │   ├── networks/           # LoRA/LoHa/LoKr implementations + architecture registry
│   │   ├── modules/            # Shared modules (see below)
│   │   └── utils/              # Utilities (device, model loading, training helpers)
│   └── blissful_tuner/         # Blissful-specific extensions
│       ├── blissful_core.py    # Args + global behavior injection
│       ├── blissful_logger.py  # BlissfulLogger (Rich-based colored logging)
│       ├── guidance.py         # CFGZero*, NAG, perpendicular CFG
│       ├── latent_preview.py   # Real-time latent visualization
│       ├── prompt_management.py # Wildcards, weighting
│       ├── scheduling.py       # Advanced CFG scheduling
│       ├── fp8_optimization.py # FP8 quantization support
│       ├── advanced_rope.py    # RoPE positional embedding extensions
│       ├── hvw_posemb_layers.py # Positional embedding layers (used by advanced_rope)
│       ├── model_utility.py    # Model loading/conversion helpers
│       ├── common_extensions.py # Shared extension utilities (V2V, I2I noise prep)
│       ├── extract_lora.py     # Generic LoRA extraction by SVD diff between two models
│       ├── profiling.py        # VRAM profiling and tracking utilities
│       ├── utils.py            # General utility functions (random, hashing, tensor helpers)
│       ├── video_processing_common.py # BlissfulVideoProcessor (ffmpeg/PIL helpers)
│       ├── video_to_png.py     # CLI tool: extract N frames from video as PNGs
│       ├── metaview.py         # PySide6 GUI to display bt_ metadata from MKV/PNG
│       ├── facefix.py, upscaler.py, GIMMVFI.py, yolo_blur.py  # Post-processing
│       ├── taehv.py, taesd.py  # Tiny autoencoder decoders for previews
│       └── codeformer/, gfpgan/, gimmvfi/, swinir/, esrgan/   # Vendored post-processing
├── tools/                      # Standalone tools
│   ├── create_instance_masks.py               # FaceID + instance segmentation for group photos
│   ├── apply_instance_mask_to_weighted_mask.py # Multiply weighted × instance masks
│   ├── verify_flux2_architecture.py           # FLUX.2 architecture verification
│   └── summarize_tensorboard_run.py           # TensorBoard scalar run summarizer
├── tests/                      # Unit tests (pytest, 58 test files)
├── configs/                    # Training configs (DLAY, OLVA, Z-Image, Qwen examples)
├── docs/                       # Architecture guides, training references
└── *.py                        # Root-level thin wrapper scripts (45 files)
```

**Root-level `.py` files are thin wrappers** that import `main()` from `src/musubi_tuner/`. When debugging behavior, open the corresponding `src/musubi_tuner/<script>.py` implementation.

### Key Shared Modules (`src/musubi_tuner/modules/`)

| Module | Purpose |
|--------|---------|
| `mask_loss.py` | Mask-weighted loss with prior preservation (core differentiator) |
| `loss_utils.py` | `compute_unreduced_target_loss()` — MSE and Huber loss types |
| `lora_ema_teacher.py` | `LoRAEmaTeacher` dataclass for EMA tracking and in-place swap (graph-safe for torch.compile) |
| `prior_scheduling.py` | `compute_prior_weight_per_sample()` — timestep-adaptive prior weight scheduling |
| `lr_schedulers.py` | `RexLR` — Reflected Exponential LR scheduler |
| `attention.py` | Attention implementations |
| `adafactor_fused.py` | Fused Adafactor with stochastic rounding for bfloat16 |
| `custom_offloading_utils.py` | CPU offloading context manager (ThreadPoolExecutor-based) |
| `fp8_optimization_utils.py` | FP8 quantization utilities |

### Key Training Flow

1. **Dataset Config**: TOML file defines images/videos, captions, resolution bucketing, optional mask sources
2. **Latent Caching**: VAE encodes frames to latents (+ optional mask weights), stored as safetensors
3. **Text Caching**: T5 (WAN) or CLIP+LLM (HV) or Qwen2.5-VL+CLIP (K5) embeddings cached
4. **Training**: Flow matching with configurable timestep sampling, MSE/Huber loss with optional mask weighting + prior preservation
5. **LoRA Application**: Target modules vary by architecture (WanAttentionBlock for WAN, etc.)

### Training Class Hierarchy

- **`NetworkTrainer`** (`src/musubi_tuner/hv_train_network.py`): Base training class with the main `train()` loop, optimizer setup, checkpointing, and centralized mask loss application.
- **LoRA trainers** (all inherit `NetworkTrainer`): `WanNetworkTrainer`, `Kandinsky5NetworkTrainer`, `Flux2NetworkTrainer`, `QwenImageNetworkTrainer`, `ZImageNetworkTrainer`, `FramePackNetworkTrainer`, `FluxKontextNetworkTrainer`, `HunyuanVideo15NetworkTrainer`.
- **Full fine-tune trainers**:
  - `QwenImageTrainer` → inherits `QwenImageNetworkTrainer` → `NetworkTrainer`
  - `ZImageTrainer` → inherits `ZImageNetworkTrainer` → `NetworkTrainer`
  - `FineTuningTrainer` (`hv_train.py`) → **standalone class**, does NOT inherit `NetworkTrainer`. Independent 1700+ line training loop for HunyuanVideo full fine-tuning.

### WAN 2.2 Specifics

- **Dual Model Architecture**: Separate high/low noise models with boundary switching (T2V: 0.875, I2V: 0.900)
- **T5 Only**: CLIP not required (unlike WAN 2.1)
- **Tasks**: `t2v-A14B` (T2V) or `i2v-A14B` (I2V)
- **Flow Matching**: `--timestep_sampling shift`, `--discrete_flow_shift 12.0` for T2V

### Kandinsky 5 Specifics

- **Task-Config System**: `kandinsky5/configs.py` defines all model/sampling params per task — 18 configs total
- **Dual Text Encoders**: Qwen2.5-VL-7B (primary) + CLIP (secondary), unique among architectures
- **NABLA Attention**: Sparse attention for memory-efficient training/inference (`--force_nabla_attn`, `--nabla_p`)
- **Task Families**: K5-Pro (SD/HD, T2V/I2V, 5s/10s), K5-Lite (standard, distilled, nocfg, pretrain)
- **Example tasks**: `k5-pro-t2v-5s-sd`, `k5-lite-t2v-5s-sd`, `k5-lite-t2v-5s-distil-sd`, `k5-lite-t2v-5s-nocfg-sd`

### Generation Script Feature Matrix

| Feature | WAN | HV | HV1.5 | FPack | Kontext | FLUX.2 | Qwen | ZImage | K5 |
|---------|-----|----|-------|-------|---------|--------|------|--------|----|
| Latent Preview | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| CFG Schedule | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| CFGZero* | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| NAG | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Perp. Negative | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| RifleX | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| I2V | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| V2V | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wildcards | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

### Loss Computation

Loss is computed as unreduced MSE (or Huber), then processed through the centralized mask loss module (`src/musubi_tuner/modules/mask_loss.py`):

```python
loss = compute_unreduced_target_loss(model_pred, target, args)  # MSE or Huber
loss = apply_masked_loss_with_prior(loss, mask_weights, prior_loss_unreduced, args, layout, stats=stats)
```

The mask loss module handles: Gaussian blur, gamma correction, min-weight floor, area-scale beta, weighted-mean normalization, EMA teacher or base-model prior preservation, timestep-gated prior, prior decay scheduling, per-sample normalization, and Huber linear-regime telemetry.

### Mask-Weighted Loss Training

Spatial mask-weighted loss is supported for selective region training (e.g., face-focused LoRA training). See `docs/MASKED_LOSS_TRAINING_GUIDE.md` for the comprehensive reference.

**Supported Architectures:**

| Architecture | Caching | LoRA Training | Full Fine-tune |
|-------------|---------|---------------|----------------|
| WAN 2.1/2.2 | ✅ | ✅ | - |
| HunyuanVideo | ⚠️ | ✅ | - |
| Qwen-Image | ✅ | ✅ | ✅ |
| Z-Image | ✅ | ✅ | - |
| FLUX.2 | ✅ | ✅ | - |
| Kandinsky 5 | ✅ | ✅ | - |
| FLUX.1 Kontext | ❌ | ❌ | - |
| FramePack | ❌ | ❌ | - |

Notes:
- WAN supports `mask_directory` for both `image_directory` and `video_directory` datasets (one mask image per item).
- HunyuanVideo training supports applying `mask_weights` if present, but the default HV latent caching flow does not currently write `mask_weights_*` into cache files.
- FLUX.2 supports `mask_directory` and `alpha_mask` for image datasets. Masks are baked into latent cache during `flux_2_cache_latents.py`.
- Z-Image supports `mask_directory` and `alpha_mask` for image datasets. Masks are baked into latent cache during `zimage_cache_latents.py`.

**Dataset Config with Masks:**
```toml
[[datasets]]
resolution = [1024, 1024]
batch_size = 1

[[datasets.subsets]]
image_directory = "/path/to/images"
mask_directory = "/path/to/masks"  # Grayscale PNGs matched by basename
caption_extension = ".txt"
```

**Alpha Channel Masks (Alternative):**

Embed masks directly in RGBA PNGs to eliminate filename mismatch bugs:

```toml
[[datasets]]
image_directory = "/path/to/rgba_images"  # RGBA PNGs with embedded masks
cache_directory = "/path/to/cache_alpha"  # Always use fresh cache!
alpha_mask = true
mask_directory = "/path/to/masks"         # Optional fallback for non-RGBA images
require_mask = false                      # Set true to error if any image lacks mask
resolution = [1328, 1328]
```

Fallback chain:
1. `alpha_mask=true` + RGBA image → use alpha channel
2. Else if `mask_directory` has matching file → use mask file
3. Else → full-weight mask (or error if `require_mask=true`)

**Training Arguments:**
```bash
--use_mask_loss                  # Enable mask-weighted loss
--mask_gamma 1.0                 # Contrast: <1.0 softer, >1.0 sharper (default: 1.0)
--mask_min_weight 0.0            # Minimum weight for black regions (default: 0.0)
--mask_blur_kernel_size 0        # Gaussian blur kernel (odd int, 0=disabled)
--mask_blur_radius 0.0           # Gaussian blur radius
--mask_area_scale_beta 0.0       # Scale target loss by raw_mask_mean^beta (reduces gradient spikes on tiny masks)
--prior_preservation_weight 0.0  # Prior preservation weight (0 = disabled, recommended: 0.5-1.0)
--prior_mask_threshold 0.1       # Threshold mode for prior mask (optional, 0.05-0.15)
--prior_teacher_mode base        # "base" (disable LoRA) or "ema" (EMA-smoothed adapter copy)
--prior_teacher_ema_decay 0.999  # EMA decay rate (when using ema mode)
--prior_preservation_timestep_threshold 0.0  # Skip teacher below this timestep (saves compute)
--prior_decay_schedule constant  # Prior weight schedule: constant, linear, or cosine
--prior_decay_timestep_start 300 # Pivot timestep for prior decay
--prior_decay_warmup_ratio 0.0   # Warm up prior weight over fraction of training
--normalize_per_sample           # Per-sample normalization (recommended with prior preservation)
```

**Mask Format:**
- Grayscale PNG images matching training image filenames, OR alpha channel of RGBA PNGs
- White (255) = full training weight, Gray = partial, Black (0) = ignored
- Standard tiers: Face=255, Body=128, Hair=80, Background=0
- Masks are downsampled to latent space resolution during caching
- **Important:** Always use a fresh `cache_directory` when changing mask sources!

**Mask Generation Tools** (in `tools/`):
```bash
# Instance masks for group photos (isolate target person via FaceID)
python tools/create_instance_masks.py --input ./images/ --output ./instance_masks/ \
    --reference /path/to/reference_face.jpg --backend yolo

# Combine weighted + instance masks
python tools/apply_instance_mask_to_weighted_mask.py \
    --weighted-masks ./weighted_masks/ --instance-masks ./instance_masks/ --output ./final_masks/
```

### Dataset Format

**TOML Config Example:**
```toml
[[datasets]]
resolution = [720, 1280]
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets.subsets]]
video_directory = "/path/to/videos"
caption_extension = ".txt"
video_frames = 81
target_frames = 81
frame_extraction = "head"
```

**Latent Cache Format:**
- Files: `{name}_{W}x{H}_wan.safetensors`
- Keys: `latents_{F}x{H}x{W}_{dtype}`, optionally `clip_{dtype}`, `latents_image_{...}`, `latents_control_{...}`, `mask_weights_{F}x{H}x{W}_float16`

**Text Cache Format:**
- Files: `{name}_wan_te.safetensors`
- Keys: `varlen_t5_{dtype}` (variable-length T5 embeddings)

### LoRA/LoHa/LoKr Target Modules

| Architecture | Target | LoRA Module | LoHa/LoKr Module |
|-------------|--------|-------------|------------------|
| WAN | WanAttentionBlock | networks.lora_wan | networks.loha / networks.lokr |
| HunyuanVideo | DoubleStreamBlock, SingleStreamBlock | networks.lora | networks.loha / networks.lokr |
| HunyuanVideo 1.5 | MMDoubleStreamBlock | networks.lora_hv_1_5 | networks.loha / networks.lokr |
| FLUX.1 Kontext | DoubleStreamBlock, SingleStreamBlock | networks.lora_flux | networks.loha / networks.lokr |
| FLUX.2 | DoubleStreamBlock, SingleStreamBlock | networks.lora_flux_2 | networks.loha / networks.lokr |
| FramePack | HunyuanVideoTransformerBlock | networks.lora_framepack | networks.loha / networks.lokr |
| Qwen-Image | QwenImageTransformerBlock | networks.lora_qwen_image | networks.loha / networks.lokr |
| Z-Image | ZImageTransformerBlock | networks.lora_zimage | networks.loha / networks.lokr |
| Kandinsky 5 | TransformerEncoderBlock, TransformerDecoderBlock | networks.lora_kandinsky | networks.loha / networks.lokr |

LoHa and LoKr use a shared architecture registry (`networks.network_arch`) for target modules and default exclude patterns. The registry defines 13 architecture variants (including FLUX.2 Klein 4B/9B, Qwen-Image Edit/Layered). LoKr v1 is Linear-only (Conv2d layers are skipped). Factor persistence is handled via `lokr_factor` buffer + `ss_lokr_factor` metadata.

Additional network utilities in `networks/`:
- `lycoris.py` — LyCORIS adapter/bridge module
- `convert_hunyuan_video_1_5_lora_to_comfy.py` — HV1.5 LoRA → ComfyUI converter
- `convert_z_image_lora_to_comfy.py` — Z-Image LoRA → ComfyUI converter

### Memory Optimization Flags

```bash
--blocks_to_swap N        # Swap N blocks to CPU (max 39 for 14B)
--fp8_base                # FP8 precision for DiT
--fp8_t5                  # FP8 for T5 encoder
--gradient_checkpointing  # Enable gradient checkpointing
--offload_inactive_dit    # Offload inactive model (WAN 2.2)
--rope_func comfy         # VRAM-efficient rope (good with --compile)
--prefer_lycoris          # Use LyCORIS backend for LoRA merging (inference)
```

### Muon Optimizer

`src/musubi_tuner/optimizers/muon.py` provides a Muon optimizer with Newton-Schulz orthogonalization. `muon_util.py` contains `MODEL_LAYER_PATTERNS` — a per-architecture registry mapping LoRA module names to Muon-eligible parameters. Falls back to internal implementation if `muon` package is not installed.

## Documentation

### Core Guides
- `docs/wan.md` - WAN 2.1/2.2 training and inference
- `docs/hunyuan_video.md` - HunyuanVideo guide
- `docs/hunyuan_video_1_5.md` - HunyuanVideo 1.5 guide
- `docs/framepack.md` - FramePack guide
- `docs/framepack_1f.md` - FramePack 1-frame mode
- `docs/flux_2.md` - FLUX.2 training and inference
- `docs/flux_kontext.md` - FLUX.1 Kontext guide
- `docs/qwen_image.md` - Qwen Image guide
- `docs/zimage.md` - Z-Image-Turbo guide
- `docs/kandinsky5.md` - Kandinsky 5 training and inference
- `docs/wan_1f.md` - WAN 1-frame mode

### Training & Configuration
- `docs/dataset_config.md` - Dataset configuration
- `docs/advanced_config.md` - Advanced training options
- `docs/torch_compile.md` - torch.compile optimization
- `docs/LORA_TRAINING_REFERENCE.md` - LoRA training reference
- `docs/NETWORK_ARGS_REFERENCE.md` - Network arguments reference
- `docs/loha_lokr.md` - LoHa & LoKr training and inference guide
- `docs/PRODIGY_PLUS_SCHEDULEFREE_ULTIMATE_GUIDE.md` - Prodigy+ optimizer guide
- `docs/sampling_during_training.md` - Sample generation during training
- `docs/MASKED_LOSS_TRAINING_GUIDE.md` - Comprehensive mask loss and prior preservation reference
- `docs/tools.md` - Tools documentation
- `docs/DEPRECATION_NOTICES.md` - Deprecation notices and removal schedule

### Architecture References
- `docs/wan22_architecture.md` - WAN 2.2 architecture details
- `docs/flux2_architecture.md` - FLUX.2 architecture details
- `docs/qwen_image_architecture.md` - Qwen Image architecture details
- `docs/cute_attention.md` - CuTe attention documentation
- `docs/z-image-integration-reference.md` - Z-Image integration reference
- `docs/fork_analysis.md` - Fork analysis vs upstream

### Internal (plans/, planning/, reference/, archive/)
- `docs/plans/` - Implementation design documents
- `docs/planning/` - Planning notes
- `docs/reference/` - Reference materials
- `docs/archive/` - Archived documents

## Mask Loss Implementation Notes

The mask loss system (`src/musubi_tuner/modules/mask_loss.py`) is a key differentiator. These invariants must be preserved:

### Critical Design Decisions
- **Weighted-mean normalization**: Loss uses `sum(loss*w)/sum(w)`, NOT `(loss*w).mean()`. The mean approach dilutes loss based on mask coverage and produces inconsistent gradients. This is intentional and verified by comparison with other trainers.
- **Gamma/min_weight/blur at training time, not cache time**: Masks are stored as raw [0,1] values in cache. All transformations (Gaussian blur, `mask**gamma`, `mask*(1-min)+min`) are applied during training. This allows experimenting with different values without recaching.
- **Compact mask representation**: Masks are kept as `(B,1,F,H,W)` and broadcast against `(B,C,F,H,W)` loss tensors. Weight sums are multiplied by `num_channels` to compensate. This saves VRAM.
- **Prior mask non-overlap (threshold mode)**: When `prior_mask_threshold` is set, the target mask is zeroed wherever the prior mask applies (`mask_processed *= (1 - prior_mask)`), preventing double-counting.
- **EMA teacher (graph-safe)**: `LoRAEmaTeacher` uses in-place parameter swap to avoid creating new tensors, making it compatible with `torch.compile` graphs. Initialized after warmup to avoid step-0 adapter noise.

### Known Limitations
- **Prior preservation not supported for `layout="layered"`**: Raises `NotImplementedError`. Affects Qwen-Image edit mode with prior preservation.
- **HunyuanVideo caching doesn't write mask_weights**: The HV training loop can consume mask_weights, but `hv_cache_latents.py` doesn't produce them.
- **FLUX.1 Kontext and FramePack**: No mask support at all.

### Key Test Files (58 total in tests/)
- `tests/test_mask_loss.py` — Core math (gamma, min_weight, blur, prior, normalization, edge cases)
- `tests/test_wan_mask_loss_integration.py` — WAN-specific shapes and prior preservation integration
- `tests/test_wan_mask_spatial_validation.py` — Spatial dimension validation during caching
- `tests/test_mask_loss_disabled_warning.py` — Config warning when masks configured but `--use_mask_loss` off
- `tests/test_zimage_mask_weights_cache.py` — Z-Image cache round-trip
- `tests/test_lora_ema_teacher.py` — EMA teacher tracking and swap
- `tests/test_prior_scheduling.py` — Timestep-adaptive prior weight scheduling
- `tests/test_loss_utils.py` — MSE/Huber loss computation
- `tests/test_muon_optimizer.py` — Muon optimizer integration

## Blissful Logger

`BlissfulLogger` requires two arguments: `BlissfulLogger(__name__, "green")` — the color parameter (a Rich color name) is mandatory. Optional third arg `do_announce` defaults to False.

## Notes

- You have my explicit permission to use any and all available resources at your disposal to assist you with any of your tasks whether that be launching as many mutliple parallel Claude OPUS sub-agents, MCP servers, plug-ins, etc at any time.
- Please feel free to ask me preliminary or follow-up questions to achieve a better result.
- Root-level `.py` files are thin wrappers that import from `src/musubi_tuner/`
- Commit messages follow Conventional Commits: `feat:`, `fix(scope):`, `chore:`, `doc:`, `format:`
- Breaking changes may occur during development
- For issues, use this repo's issues section (not upstream Musubi Tuner)
- **NEVER open pull requests against the upstream repository (kohya-ss/musubi-tuner).** This is a personal fork. Always target `DustyL/blissful-tuner` when creating PRs. Use `--repo DustyL/blissful-tuner` with `gh pr create`.
