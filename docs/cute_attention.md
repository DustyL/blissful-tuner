# CuTE Attention Training Guide

## Overview

CuTE (“CUDA Templates”) is an attention backend exposed via `flash_attn.cute`. On Hopper/Blackwell GPUs it can be noticeably faster than FA2/SDPA, especially for long sequence lengths.

In blissful-tuner, CuTE is enabled with `--cute` (or `cute = true` in a TOML config), and is supported in:
- WAN 2.1/2.2 (`src/musubi_tuner/wan/modules/attention.py`)
- HunyuanVideo + Qwen-Image (`src/musubi_tuner/hunyuan_model/attention.py`)
- HunyuanVideo 1.5 + Z-Image + FLUX.2 (`src/musubi_tuner/modules/attention.py`)

## Requirements

- GPU: Hopper (SM 9.0+), Blackwell datacenter (SM 10.0+), or Blackwell GeForce (SM 12.0). Specifically:
  - Hopper: H100/H200 (SM 9.0)
  - Blackwell datacenter: B200 (SM 10.0), B300 (SM 10.3)
  - Blackwell GeForce: RTX 5090 (SM 12.0)
- `flash-attention` with CuTE enabled (example in this repo: `2.8.3+varlen.sm103`)
  - For SM 12.0 (RTX 5090): requires the `feat/sm120-support` branch or equivalent patches
- CuTE runtime deps:
  - `quack-kernels>=0.2.10`
  - `nvidia-cutlass-dsl>=4.4.0`
  - `apache-tvm-ffi>=0.1.5,<0.2`
  - `torch-c-dlpack-ext`

### Architecture-Specific Limitations

**Hopper (SM90):** Does **not** support variable-length (varlen) backward. In practice:
- Inference: varlen CuTE is fine (`torch.no_grad()` / no backward).
- Training: if your model uses CuTE varlen (e.g. to handle padded sequences), it will fail on SM90 during backward.

Workarounds:
- Use `--split_attn` (forces per-sample fixed-length attention).
- Or switch to `--flash-attn` / `--sage-attn` for varlen training on SM90.

**Blackwell GeForce / SM120 (RTX 5090):** Does **not** support `deterministic=True` in backward. This only affects WAN, which is the only architecture that exposes the `deterministic` parameter to CuTE. The WAN CuTE wrapper automatically detects SM120 and overrides to `deterministic=False` with a one-time warning. Datacenter Blackwell (B200/B300) and Hopper are unaffected.

Quick checks:
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "from flash_attn.cute import flash_attn_func; print('CuTE OK')"
```

## Recommended Environment

All DLAY env scripts configure CuTE caching automatically. Source the one for your architecture:
```bash
source configs/DLAY/QWEN-IMAGE/env_qwen2512.sh       # Qwen-Image
source configs/DLAY/FLUX2-KLEIN-9B/env_flux2klein9b.sh # FLUX.2 Klein-9B
source configs/DLAY/WAN22/env_wan22.sh                 # WAN 2.2
source configs/DLAY/ZIMAGE-TURBO/env_zimage_turbo.sh   # Z-Image Turbo
# ... etc.
```

These configure:
- `TORCHINDUCTOR_CACHE_DIR` (torch.compile cache)
- `CUTE_DSL_CACHE_DIR` (CuTE DSL low-level JIT cache)
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` (FA4 kernel-level cache — avoids recompiling kernel variants across runs, reducing startup from minutes to seconds)
- Allocator settings appropriate for the training configuration

If running without an env script, set these manually:
```bash
export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-$HOME/.cache/cute_dsl}"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
mkdir -p "${CUTE_DSL_CACHE_DIR}" 2>/dev/null || true
```

## How To Enable CuTE

### Option A: Use the repo launcher (DLAY/Qwen-Image example)

```bash
./configs/DLAY/QWEN-IMAGE/train_dlay_qwen2512.sh cute
```

### Option B: CLI flags (any supported training entrypoint)

Use exactly one attention backend flag:
```bash
--sdpa | --flash-attn | --flash3 | --sage-attn | --xformers | --cute
```

## Masking & Variable-Length Text (Important)

Some attention backends (FA2/CuTE/Sage) do **not** consume an explicit padding mask tensor. For correctness with padded / variable-length text:

- `src/musubi_tuner/hunyuan_model/attention.py` auto-routes:
  - `cute` → `cute_varlen` when `cu_seqlens_q` is provided (and `--split_attn` is not used)
- `src/musubi_tuner/qwen_image/qwen_image_model.py` builds and passes `cu_seqlens_*` from `txt_seq_lens` when using CuTE/FA2/Sage, ensuring padding tokens don’t contaminate attention.

If you see identity “drift” or inconsistent training when `batch_size > 1`, verify you’re using a varlen-capable backend (`--cute` / `--flash-attn` / `--sage-attn`) and that `txt_seq_lens` is present in the batch (it is for Qwen-Image caching).

## torch.compile Notes

- CuTE kernels JIT-compile independently from `torch.compile`.
- First run may be slower due to:
  - Inductor autotuning/compiles
  - CuTE kernel JIT compilation

Recommended starting point for training:
- `compile = true`
- `compile_mode = "max-autotune-no-cudagraphs"` (stable)

If using `compile_mode="max-autotune"` (CUDAGraphs enabled), keep the allocator on the native backend (the provided env script does this).

## Mask-Weighted Loss + Caching Reminder

Mask-weighted loss requires masks baked into the latent cache. If you change `mask_directory`, you must cache into a **fresh** `cache_directory` (or remove `--skip_existing`).

See: `docs/MASKED_LOSS_TRAINING_GUIDE.md`

## Troubleshooting

- `ImportError: CuTE not available`:
  - Install CuTE deps: `pip install 'quack-kernels>=0.2.10' 'nvidia-cutlass-dsl>=4.4.0' 'apache-tvm-ffi>=0.1.5,<0.2' torch-c-dlpack-ext`
  - Confirm `from flash_attn.cute import flash_attn_func` works
  - SM120 (RTX 5090): Ensure flash-attention is built with SM120 support (see `docs/fa4_sm120_reference.md`)
- Slow startup (minutes of JIT compilation):
  - Set `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` — this caches compiled FA4 kernels across runs
  - Set `CUTE_DSL_CACHE_DIR` — this caches the lower-level CuTE DSL PTX/CUBIN
  - Both are set automatically by all DLAY env scripts
- Performance worse than FA2:
  - CuTE tends to win at longer sequence lengths (often >= 1024)
  - Ensure dtype is bf16/fp16 and head_dim is supported (commonly 128 in these models)
- SM120 `deterministic=True` assert failure:
  - SM120 does not support deterministic backward in CuTE
  - The WAN CuTE wrapper auto-detects SM120 and overrides to `deterministic=False`
  - If you hit this on another architecture, pass `deterministic=False` explicitly
- Want to rollback:
  - Switch to `--flash-attn` (FA2) or `--sdpa` (PyTorch)
