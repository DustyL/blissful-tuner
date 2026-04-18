# FA4 (Flash Attention 4) Reference for Blissful Tuner

This document is a reference for Claude Code agents working on FA4/CuTE attention integration in blissful-tuner. It covers SM120 (RTX 5090) specifics, caching, and what's needed for expanding CuTE support.

## SM120 Architecture (Blackwell GeForce)

RTX 5090 = SM 12.0 (arch=120). Key facts:

- **MMA instructions:** Uses SM80-era `mma.sync.aligned.m16n8k16` (same as Ampere), NOT Hopper TMA or Blackwell UMMA
- **Shared memory:** 99 KB (vs 163 KB SM80, ~224 KB SM100)
- **FA4 kernel classes:** `FlashAttentionForwardSm120` / `FlashAttentionBackwardSm120` — thin subclasses of SM80 kernels that override `can_implement()` to check 99 KB SMEM capacity
- **Tile sizes (forward):** hdim<=64: 128x128 (48 KB), hdim>64: 128x64 (64 KB)
- **Tile sizes (backward):** m=n=64, 128 threads, 1-2 stages
- **Source:** `~/flash-attention` branch `feat/sm120-support` (cherry-picked from blake-snc/flash-attention PRs #2329, #2330, #2333, #2336)

### SM120 Feature Support Matrix

| Feature | Forward | Backward | Notes |
|---------|---------|----------|-------|
| Fixed-length attention | YES | YES | |
| Variable-length (varlen) | YES | YES | |
| Causal masking | YES | YES | |
| Sliding window (local) | YES | YES | |
| Split-KV (FlashDecoding) | YES | N/A | Decode-only |
| GQA/MQA | YES | YES | `pack_gqa=None` (auto) |
| fp16 / bf16 | YES | YES | |
| hdim 64, 96, 128 | YES | YES | |
| softmax_scale | YES | YES | |
| Block sparsity | NO | NO | Asserts at dispatch |
| score_mod (forward) | YES | — | Forward only |
| score_mod (backward) | — | NO | Asserts at dispatch |
| mask_mod (forward) | YES | — | Forward only |
| mask_mod (backward) | — | NO | Asserts at dispatch |
| Deterministic backward | — | NO | `deterministic=False` required |
| Paged KV cache | YES* | N/A | *Python-level gather, not TMA |

### SM120 Implications for Blissful Tuner

All existing CuTE wrappers in blissful-tuner work on SM120 with **no code changes** — the FA4 dispatch in `interface.py` handles architecture routing automatically. Known architecture-specific notes:

- **WAN's `deterministic` parameter:** WAN is the only architecture that exposes `deterministic` to CuTE. SM120 does not support deterministic backward. The WAN CuTE wrappers (`_cute_attention`, `_cute_attention_varlen`) include an automatic SM120 guard that detects the GPU on first call and overrides `deterministic=True` to `False` with a one-time warning. Datacenter Blackwell (B200/B300) and Hopper are unaffected.
- **SM90 varlen backward restriction:** Still applies on Hopper. SM120 does NOT have this restriction — varlen backward works fine.
- **FLUX.2:** Now fully supported via `--cute` or `cute = true`. Routes through unified attention (`src/musubi_tuner/modules/attention.py`) which already had complete CuTE support.

## JIT Compile Caching (Two Layers)

FA4 kernels are JIT-compiled at runtime. There are **two independent cache layers**:

### Layer 1: CuTe DSL Cache (`CUTE_DSL_CACHE_DIR`)

Low-level NVIDIA CuTe DSL compilation cache. Caches compiled PTX/CUBIN from the cutlass-dsl compiler.

```bash
export CUTE_DSL_CACHE_DIR="/path/to/cache/cute_dsl"
```

- Set in all DLAY env scripts (`configs/DLAY/*/env_*.sh`)

### Layer 2: FA4 Kernel Cache (`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED`)

Higher-level FA4 kernel cache. Cache key includes dtype, head_dim, causal, mask/score_mod hashes, architecture, block sizes. Stored at `/tmp/${USER}/flash_attention_cute_dsl_cache/`.

```bash
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
```

- Set in all DLAY env scripts (`configs/DLAY/*/env_*.sh`)
- Avoids re-compiling FA4 kernel variants across training runs
- Can reduce startup from minutes to seconds on subsequent runs

### Recommended Cache Setup

For any env script used with CuTE training:

```bash
# CuTE DSL low-level cache
export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-$HOME/.cache/cute_dsl}"

# FA4 kernel-level cache (avoids recompiling kernel variants)
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

# Create cache dirs
mkdir -p "${CUTE_DSL_CACHE_DIR}" 2>/dev/null || true
```

### Fast Two-Pass Test Compilation

For development/testing, FA4 supports compiling all kernels in parallel without a GPU:

```bash
# Pass 1: compile kernels (parallel, no GPU needed)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: run tests using cached kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x tests/cute/test_flash_attn.py
```

## return_lse Gap

All CuTE calls in blissful-tuner discard the log-sum-exp (LSE) return value:

```python
out, _ = _cute_attention(q, k, v, causal=causal)  # LSE discarded
```

`flash_attn_func` and `flash_attn_varlen_func` both return `(output, lse)` where `lse` has shape `(batch, nheads, seqlen)` (float32). LSE represents the log-sum-exp of the softmax denominator for each query position.

### Potential Uses

1. **Attention collapse monitoring:** Track `lse.std()` across heads/layers during training. Collapsing std indicates attention heads are becoming degenerate.
2. **Knowledge distillation:** LSE is needed to compute exact cross-entropy between teacher and student attention distributions without recomputing softmax.
3. **Debugging / diagnostics:** Sudden LSE spikes can indicate numerical instability before it manifests as NaN loss.

### Implementation Pattern

To capture LSE without breaking existing behavior:

```python
# In the @torch.compiler.disable wrappers, LSE is already returned:
out, lse = cute_flash_attn_func(q, k, v, causal=causal)
# The `_` in `out, _ = _cute_attention(...)` is where lse goes

# To expose it, either:
# 1. Add a return_lse parameter to the attention wrapper
# 2. Store lse on a context object for optional retrieval
# 3. Log lse statistics (mean, std) to training metrics when a flag is set
```

The wrappers would need a `return_lse: bool = False` parameter, and callers would need to handle the optional extra return value. All three attention files would need updating.

## CuTE Support by Architecture

### Currently Supported

| Architecture | Attention File | CuTE Wrapper | Varlen | Extra Params |
|---|---|---|---|---|
| WAN 2.1/2.2 | `src/musubi_tuner/wan/modules/attention.py` | `_cute_attention`, `_cute_attention_varlen` | Yes | `softmax_scale`, `causal`, `window_size`, `deterministic` |
| HunyuanVideo | `src/musubi_tuner/hunyuan_model/attention.py` | `_cute_attention`, `_cute_attention_varlen` | Yes | `causal` only |
| Qwen-Image | (same as HunyuanVideo) | (same) | Yes | `causal` only |
| HunyuanVideo 1.5 | `src/musubi_tuner/modules/attention.py` | `_cute_attention`, `_cute_attention_varlen` | Yes | `causal` only |
| Z-Image | (same as HV1.5) | (same) | Yes | `causal` only |
| FLUX.2 (all variants) | (same as HV1.5, unified attention) | (same) | Yes | `causal` only |
| FLUX 1.0 | (inherited from FLUX.2) | (same) | Yes | |
| FramePack | (inherited from WAN/HV) | (same) | Yes | |

### NOT Yet Supported

| Architecture | Attention File | Current Backends | Notes |
|---|---|---|---|
| **Kandinsky-5** | `src/musubi_tuner/kandinsky5/models/attention.py` | FA2, FA3, Sage, xformers, SDPA | Uses `SelfAttentionEngine` class pattern — different from other architectures |

### Kandinsky-5 CuTE Integration Notes

Kandinsky-5 uses a `SelfAttentionEngine` class with a strategy pattern — `engine` selects the attention function at init time. To add CuTE:

1. Add CuTE import block (same pattern as other attention files)
2. Add a `@torch.compiler.disable` wrapped `_cute_attention` helper
3. Add `"cute"` to the `SelfAttentionEngine.__init__()` engine choices
4. Kandinsky-5 attention is **fixed-length only** (no varlen/cu_seqlens) — simpler integration
5. No `causal` parameter is used (all attention is non-causal)
6. The `@_maybe_compile` decorator pattern means CuTE needs a `@torch.compiler.disable` wrapper to prevent Dynamo tracing (same reason as other architectures)
7. Head dimension is 128 across all Kandinsky-5 attention — compatible with SM120

## All Architecture Head Dimensions

All supported diffusion architectures in blissful-tuner use **hdim=128**, which is within SM120's supported range (64, 96, 128).

## Known Issues

### pack_gqa cutlass-dsl 4.4.1 Bug

`pack_gqa=True` forced on MHA (where num_q_heads == num_kv_heads) triggers MLIR layout mismatch in cutlass-dsl 4.4.1:
```
ValueError: Operation creation failed [...] unable to compute crd2idx
```

**Workaround:** Use `pack_gqa=None` (the default), which auto-detects and skips packing when heads are equal. This is NOT SM120-specific — it affects all architectures.

Blissful-tuner does not set `pack_gqa` in any CuTE call, so the default (`None`) is always used. No action needed.

### CUDA Launch Ordering

During rapid sequential SM120 kernel launches, asynchronous CUDA illegal memory access errors can appear. Setting `CUDA_LAUNCH_BLOCKING=1` serializes launches and resolves this. This was observed during testing, not during normal training.

## FA4 API Quick Reference

```python
from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

# Fixed-length
out, lse = flash_attn_func(
    q,                      # (batch, seqlen, nheads, headdim)
    k,                      # (batch, seqlen, nheads_k, headdim)
    v,                      # (batch, seqlen, nheads_k, headdim)
    softmax_scale=None,     # default: 1/sqrt(headdim)
    causal=False,
    window_size=(None, None),  # (left, right) or (None, None) for full
    deterministic=False,       # SM120: must be False
    # Advanced (not used in blissful-tuner):
    # score_mod, mask_mod, block_sparse_tensors, num_splits, pack_gqa,
    # m_block_size, n_block_size, num_threads
)

# Variable-length
out, lse = flash_attn_varlen_func(
    q,                      # (total_q, nheads, headdim)
    k,                      # (total_k, nheads_k, headdim)
    v,                      # (total_k, nheads_k, headdim)
    cu_seqlens_q,           # (batch+1,) int32
    cu_seqlens_k,           # (batch+1,) int32
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),
    deterministic=False,
)
```

Tensor layout: `(batch, seqlen, num_heads, head_dim)` — last dim contiguous, 16-byte aligned. Dtypes: fp16 or bf16.

## File Locations

### Blissful Tuner Attention Files
- `src/musubi_tuner/wan/modules/attention.py` — WAN CuTE wrappers (most parameters)
- `src/musubi_tuner/hunyuan_model/attention.py` — HunyuanVideo + Qwen-Image CuTE wrappers
- `src/musubi_tuner/modules/attention.py` — HV1.5 + Z-Image + FLUX.2 CuTE wrappers (uses `AttentionParams` dataclass)
- `src/musubi_tuner/kandinsky5/models/attention.py` — Kandinsky-5 (NO CuTE yet)

### Flash Attention 4 Source (patched)
- `~/flash-attention/flash_attn/cute/interface.py` — Public API, SM120 dispatch logic
- `~/flash-attention/flash_attn/cute/flash_fwd_sm120.py` — SM120 forward kernel
- `~/flash-attention/flash_attn/cute/flash_bwd_sm120.py` — SM120 backward kernel
- `~/flash-attention/flash_attn/cute/flash_fwd.py` — SM80 base forward (split-KV refactored)

### Existing Documentation
- `docs/cute_attention.md` — User-facing CuTE training guide (includes SM120 GPU requirements)
- `configs/DLAY/*/env_*.sh` — All DLAY env scripts set `CUTE_DSL_CACHE_DIR` and `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED`
