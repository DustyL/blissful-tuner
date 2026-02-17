# Z-Image LoRA Pipeline Audit — Action Plan

**Created**: 2026-02-17
**Revised**: 2026-02-17 (v5 — Phase 5 implemented)
**Source**: Comprehensive audit of Z-Image training/inference pipeline against `docs/z-image-integration-reference.md` v2.3
**Status**: In Progress — Phases 1–5 complete, Phase 6+ pending

---

## Table of Contents

- [Phase 1: Critical Bugs](#phase-1-critical-bugs)
- [Phase 2: Medium Bugs](#phase-2-medium-bugs)
- [Phase 3: Reference Document Corrections](#phase-3-reference-document-corrections)
- [Phase 4: Warnings — Medium Priority](#phase-4-warnings--medium-priority)
- [Phase 5: Warnings — Low Priority](#phase-5-warnings--low-priority)
- [Phase 6: Improvements](#phase-6-improvements)
- [Confirmed Correct (No Action)](#confirmed-correct-no-action-needed)

---

## Phase 1: Critical Bugs

### BUG-1: LyCORIS merge silently no-ops for Z-Image (module targeting) ✅ DONE

- **Files**:
  - `src/musubi_tuner/zimage_generate_image.py:28,301-313` (call site)
  - `src/musubi_tuner/wan_generate_video.py:841-852` (`merge_lora_weights` LyCORIS branch)
  - `src/musubi_tuner/networks/lycoris.py:221-225` (auto-detect logic)
- **Severity**: CRITICAL — `--prefer_lycoris` merges silently apply zero weights

**Root Cause (corrected from v1)**:

The original audit flagged the `lora_qwen_image` import as the bug. That is **cosmetically wrong** (should be `lora_zimage`), but it is NOT the root cause: in the `lycoris=True` branch of `merge_lora_weights()` (`wan_generate_video.py:841`), the `lora_module` argument is **never used**. The LyCORIS path calls `create_network_from_weights()` directly.

The real problem is twofold:

**1. Hardcoded WAN-specific `extra_unet_targets`** (`wan_generate_video.py:850`):
```python
# In merge_lora_weights(), lycoris=True branch:
lycoris_net, _ = create_network_from_weights(
    multiplier=lora_multiplier,
    file=None,
    weights_sd=weights_sd,
    unet=model,
    text_encoder=None,
    vae=None,
    for_inference=True,
    extra_unet_targets=["WanAttentionBlock"],  # <-- HARDCODED WAN-ONLY
)
```

`"WanAttentionBlock"` is passed regardless of which architecture is calling. Z-Image models contain `ZImageTransformerBlock`, not `WanAttentionBlock`, so LyCORIS module discovery finds zero matches.

**2. Auto-detect only knows WAN** (`lycoris.py:221-225`):
```python
# Auto-detect only fires when extra_unet_targets is None:
if extra_unet_targets is None and unet is not None:
    module_names = {m.__class__.__name__ for m in unet.modules()}
    if "WanAttentionBlock" in module_names:
        extra_unet_targets = ["WanAttentionBlock"]
    # No check for ZImageTransformerBlock or any other arch!
```

Even if the hardcoded `["WanAttentionBlock"]` were removed, the auto-detect fallback only knows about WAN.

**3. Silent no-op** — there is no logging or error when zero modules are matched during LyCORIS merge. The merge completes "successfully" with nothing applied.

**Fix (multi-part)**:

A. Fix the import (cosmetic but correct):
```python
# zimage_generate_image.py:28
from musubi_tuner.networks import lora_zimage  # was: lora_qwen_image
```

B. Make `merge_lora_weights()` accept architecture-specific targets:
```python
# wan_generate_video.py — change merge_lora_weights signature:
def merge_lora_weights(
    lora_module, model, lora_weights, lora_multipliers,
    include_patterns, exclude_patterns, device,
    lycoris=False, save_merged_model=None,
    extra_unet_targets=None,  # <-- NEW: caller passes arch-specific targets
):
    # ...
    if lycoris:
        lycoris_net, _ = create_network_from_weights(
            ...,
            extra_unet_targets=extra_unet_targets,  # <-- USE CALLER'S VALUE
        )
```

Then in `zimage_generate_image.py`:
```python
merge_lora_weights(
    lora_zimage, model, ...,
    lycoris=True,
    extra_unet_targets=["ZImageTransformerBlock"],  # <-- Z-IMAGE SPECIFIC
)
```

C. Extend `lycoris.py` auto-detect to know Z-Image (and ideally all architectures):
```python
if extra_unet_targets is None and unet is not None:
    module_names = {m.__class__.__name__ for m in unet.modules()}
    for candidate in ["WanAttentionBlock", "ZImageTransformerBlock",
                       "Flux2DoubleStreamBlock", "QwenImageTransformerBlock"]:
        if candidate in module_names:
            extra_unet_targets = [candidate]
            break
```

D. **Add merge observability guardrail** — after any LyCORIS merge, log matched module count:
```python
lycoris_net, _ = create_network_from_weights(...)
matched = sum(1 for _ in lycoris_net.unet_loras)  # or equivalent count
if matched == 0:
    logger.error(f"LyCORIS merge matched 0 modules in model. "
                 f"Check extra_unet_targets={extra_unet_targets}")
else:
    logger.info(f"LyCORIS merge matched {matched} modules")
lycoris_net.merge_to(None, model, weights_sd, dtype=None, device=device)
```

**Verification**: Run Z-Image generation with `--prefer_lycoris --lora_weight <path>` and confirm the log shows matched modules > 0, and the output differs from un-LoRA'd generation.

---

### BUG-5: `--save_merged_model` broken without `--prefer_lycoris` ✅ DONE

- **File**: `src/musubi_tuner/zimage_generate_image.py:312,339-340`
- **Severity**: CRITICAL — silent data loss (user thinks save worked, but no file is written)

**Current code flow:**
```python
# Line 301-313: LyCORIS path — save happens inside merge_lora_weights()
if args.prefer_lycoris:
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        merge_lora_weights(
            ...,
            save_merged_model=args.save_merged_model,  # save happens at wan_generate_video.py:868
        )

# Line 339-340: ALWAYS returns None when save_merged_model is set
if args.save_merged_model:
    return None  # <-- returns without saving in non-LyCORIS path!
```

**Problem**: When a user runs:
```bash
python zimage_generate_image.py --save_merged_model out.safetensors --lora_weight lora.safetensors
# (without --prefer_lycoris)
```

The non-LyCORIS LoRA merge happens earlier via `load_safetensors_with_lora_and_fp8()` (line 286), which correctly merges LoRA into the model weights. But the save at line 339 just returns `None` — it never actually writes the file. The save only works inside the `merge_lora_weights()` call at `wan_generate_video.py:868`, which is only reached in the LyCORIS branch.

**Fix**: Add the save before returning:
```python
if args.save_merged_model:
    if not args.prefer_lycoris:
        # Non-LyCORIS path: save was not handled by merge_lora_weights
        from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
        logger.info(f"Saving merged model to {args.save_merged_model}")
        mem_eff_save_file(model.state_dict(), args.save_merged_model)
        logger.info("Merged model saved")
    return None
```

**Verification**: Run with `--save_merged_model out.safetensors` (no `--prefer_lycoris`) and verify the file is created.

---

### ~~BUG-2: `--lora_multiplier` crashes when not explicitly set~~ (RESOLVED)

- **Status**: ~~CRITICAL~~ **Already fixed**. All generation scripts already use `default=None`.
- **Verified**: `grep` across all `*_generate_*.py` confirms every script uses `default=None` for `--lora_multiplier`.
- The audit agent was working from a stale read.

---

## Phase 2: Medium Bugs

### BUG-3: FP16 safety disabled during gradient checkpointing ✅ DONE

- **File**: `src/musubi_tuner/zimage/zimage_model.py`
- **Line**: 131 (in `FeedForward.forward`)
- **Severity**: Medium — only affects `--mixed_precision fp16` with gradient checkpointing (not the recommended bf16 path)

**Current code:**
```python
def forward(self, x, apply_fp16_downscale=False):
    if self.training and self.gradient_checkpointing:
        return checkpoint(self._forward, x, use_reentrant=False)  # apply_fp16_downscale NOT passed
    else:
        return self._forward(x, apply_fp16_downscale)
```

**Problem**: `checkpoint()` is called with only `x`, so `_forward` receives `apply_fp16_downscale=False` (the default). The caller at line 279 passes `apply_fp16_downscale=True`, but this is lost during checkpointing. The `÷32` FP16 overflow protection in the FFN output is silently disabled.

**Fix**:
```python
def forward(self, x, apply_fp16_downscale=False):
    if self.training and self.gradient_checkpointing:
        return checkpoint(self._forward, x, apply_fp16_downscale, use_reentrant=False)
    else:
        return self._forward(x, apply_fp16_downscale)
```

**Verification**: Syntax check with `python -m py_compile`. Functional testing requires fp16 training with gradient checkpointing.

---

### BUG-4: Default image size 256x256 instead of 1024x1024 ✅ DONE

- **File**: `src/musubi_tuner/zimage_generate_image.py`
- **Line**: 80
- **Severity**: Medium — usability issue producing unexpectedly poor output

**Current code:**
```python
parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="image size, height and width")
```

**Problem**: `zimage_config.py` defines `DEFAULT_HEIGHT = DEFAULT_WIDTH = 1024`, but the CLI defaults to 256x256 (Phase 1 pre-training resolution). Users who forget `--image_size` get tiny images.

**Fix** (use config constants to avoid drift):
```python
from musubi_tuner.zimage import zimage_config
# ...
parser.add_argument(
    "--image_size", type=int, nargs=2,
    default=[zimage_config.DEFAULT_HEIGHT, zimage_config.DEFAULT_WIDTH],
    help="image size as height width, must be divisible by 16 (default: 1024 1024)"
)
```

---

### BUG-6: `--flash3` accepted but raises ValueError at runtime ✅ DONE

- **Files**:
  - `src/musubi_tuner/zimage_train.py:182-183` (parses `--flash3` → `attn_mode = "flash3"`)
  - `src/musubi_tuner/modules/attention.py:257-259` (no `"flash3"` handler)
- **Severity**: Medium — runtime crash, but clear error message

**Current behavior**: `zimage_train.py` accepts `--flash3` and sets `attn_mode = "flash3"`. The shared `attention()` function at `attention.py:257-259` has:
```python
else:
    raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")
```

The attention module handles `"torch"`, `"xformers"`, `"sageattn"`, and `"flash"` — but NOT `"flash3"`. Users who pass `--flash3` get a `ValueError` on the first forward pass.

**Fix options** (pick one):
1. **Remove `--flash3` from Z-Image scripts**: If FA3 support is not planned, remove the flag to prevent confusion.
2. **Map `flash3` → `flash`**: If the FA3 codepath is the same as FA2 (which it often is in PyTorch ≥2.5):
   ```python
   if attn_mode == "flash3":
       logger.warning("flash3 not separately supported for Z-Image, falling back to flash")
       attn_mode = "flash"
   ```
3. **Implement FA3 support**: Add a `"flash3"` branch to `attention.py` (largest scope, deferred to Phase 6).

---

## Phase 3: Reference Document Corrections

### DOC-1: Sandwich-Norm ordering claim is wrong ✅ DONE

- **File**: `docs/z-image-integration-reference.md`
- **Location**: Section 8.1 comparison table (Sandwich-Norm row), Section 8.3 gap item 7

**Problem**: The comparison table marks Blissful Tuner as `norm2 pre-attention (**incorrect**)`. The actual code at `zimage_model.py:274-276` correctly applies `norm2` post-output:

```python
# Actual code (CORRECT):
attn_out = self.attention(self.attention_norm1(x) * scale_msa, ...)
x = x + gate_msa * self.attention_norm2(clamp_fp16(attn_out))  # norm2 wraps OUTPUT
```

**Fix**: Update comparison table row to `norm2 **post-output** (correct)`. Remove gap item 7 from Section 8.3.

---

### DOC-2: FinalLayer normalization claim is wrong ✅ DONE

- **File**: `docs/z-image-integration-reference.md`
- **Location**: Section 8.1 comparison table (FinalLayer norm row), Section 8.3 gap item 8

**Problem**: The comparison table says Blissful Tuner uses "RMSNorm" for FinalLayer. The actual code at `zimage_model.py:308`:

```python
# Actual code (CORRECT):
self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
```

**Fix**: Update comparison table to show `nn.LayerNorm` for Blissful Tuner. Remove gap item 8 from Section 8.3.

---

### DOC-3: `docs/zimage.md` incorrectly says Z-Image doesn't support control images ✅ DONE

- **File**: `docs/zimage.md:93`
- **Current text**: "Z-Image does not support control images, so only target image latents are cached."

**Problem**: `src/musubi_tuner/zimage_cache_latents.py` (lines 78-198) explicitly implements OmniBase mode with control latent caching + SigLIP2 feature caching. The code has `--image_encoder` support and caches `latents_control_{i}` keys.

**Fix** (pick one):
1. Update to distinguish standard vs OmniBase: "In standard T2I mode, only target image latents are cached. OmniBase mode additionally caches control latents and SigLIP2 features (see `--image_encoder`)."
2. Or remove the sentence entirely since it's misleading given the current codebase.

---

## Phase 4: Warnings — Medium Priority

### WARN-1: No mask-weighted loss support for Z-Image ✅ DONE

- **Files**: `src/musubi_tuner/zimage_cache_latents.py`, `src/musubi_tuner/dataset/image_video_dataset.py`
- **Severity**: Medium — feature gap vs WAN, FLUX.2, Qwen-Image

**Problem**: Z-Image latent caching did not bake `mask_weights_*` into the latent cache, so `--use_mask_loss` would fail even if the dataset produced `item.mask_content` (via `alpha_mask=true` and/or `mask_directory`).

**Context**: The CLAUDE.md mask support table does not list Z-Image, so this may be intentionally deferred. The training script (`zimage_train_network.py`) inherits mask loss application from the base `NetworkTrainer`, so if mask weights were cached, they would work automatically.

**Fix implemented**:
1. Added `mask_weights` parameter to `save_latent_cache_z_image` in `image_video_dataset.py`
2. Added mask downsampling + saving in `encode_and_save_batch` in `zimage_cache_latents.py` (uses `item.mask_content`)
3. Updated `CLAUDE.md` mask support table to include Z-Image

---

### WARN-2: No CFG truncation or normalization in inference ✅ DONE

- **File**: `src/musubi_tuner/zimage_generate_image.py`
- **Lines**: 593 (comment acknowledges gap), 604-610 (denoising loop)
- **Severity**: Medium — official pipeline supports both features

**CFG truncation** (official behavior):
```python
# When t_normalized > cfg_truncation, guidance_scale is forced to 0.0
# Default: cfg_truncation=1.0 (CFG at all steps)
# Lower values (e.g., 0.8) disable CFG for noisiest 20% of timesteps
```

**CFG normalization** (official behavior):
```python
# Rescales CFG prediction to match positive prediction's norm
# Recommended: True for photorealism, False for stylistic output
ori_pos_norm = torch.linalg.vector_norm(pos)
new_pos_norm = torch.linalg.vector_norm(pred)
max_new_norm = ori_pos_norm * float(cfg_normalization)
if new_pos_norm > max_new_norm:
    pred = pred * (max_new_norm / new_pos_norm)
```

**Fix implemented**: Added `--cfg_truncation` (default 1.0) and `--cfg_normalization` (default disabled) CLI args and wired both into the denoising loop.

---

## Phase 5: Warnings — Low Priority

### WARN-3: `--embedded_cfg_scale` is dead code ✅ DONE

- **File**: `src/musubi_tuner/zimage_generate_image.py`
- **Lines**: 85-86 (parser), 662 (metadata only)
- **Fix implemented**: Kept the CLI arg for compatibility, but deprecated it (warns if provided) and defaults it to `None` so it doesn't silently suggest behavior that doesn't exist.

---

### WARN-4: `print()` instead of `logger.info()` ✅ DONE

- **File**: `src/musubi_tuner/zimage/zimage_model.py`
- **Lines**: 468, 477, 489, 491, 497, 503

**Current**: `print(f"Z-Image: Gradient checkpointing enabled. CPU offload: {cpu_offload}")`
**Fix**: `logger.info(f"Gradient checkpointing enabled. CPU offload: {cpu_offload}")`

The `logger` is already imported and initialized at line 37.

---

### WARN-5: `del` statements in `_forward` (torch.compile tradeoff) ✅ DONE

- **File**: `src/musubi_tuner/zimage/zimage_model.py`
- **Lines**: 270, 275, 277, 281, 664, 697-699

```python
del adaln_input    # line 270
del scale_msa      # line 275
del gate_msa       # line 277
del scale_mlp, gate_mlp  # line 281
```

**Context**: In CPython, `del name` can reduce refcounts earlier, which is not purely a no-op — it can trigger earlier memory reclamation. However, these statements may cause `torch.compile` graph breaks (dynamo cannot trace through `del`). The CLAUDE.md notes prior `del` cleanup was done for the attention module.

**Fix implemented**: Keep the early-free behavior in eager mode, but guard the `del` statements under `if not torch.compiler.is_compiling(): ...` to reduce graph-break risk when using `torch.compile`.

---

### WARN-6: Nested gradient checkpointing (redundant) ✅ DONE

- **File**: `src/musubi_tuner/zimage/zimage_model.py`
- **Lines**: 248-258 (enable on children), 300 (block-level checkpoint)

**Problem**: `enable_gradient_checkpointing` enables checkpointing on the block AND its children (FFN, attention). The block's `forward` wraps the entire `_forward` in `checkpoint()`, while FFN and attention also individually wrap in `checkpoint()`. This creates nested checkpointing — the inner checkpoints are unnecessary when the outer one already handles recomputation.

**Fix implemented**: Disable inner (attention/FFN) checkpointing when block-level checkpointing is enabled to avoid redundant nested recomputation.

---

### WARN-7: RoPE cache kept on CPU ✅ DONE

- **File**: `src/musubi_tuner/zimage/zimage_model.py`
- **Lines**: 349 (creation), 354 (per-call GPU transfer)

**Current**: RoPE frequencies are precomputed on CPU, then indexed + `.to(device)` on every forward pass.
**Official**: Cache is moved to GPU on first call and stays there.

**Impact**: Small perf hit per forward pass. Could cause torch.compile graph breaks or RoPE cache state pollution (noted in diffusers tests).

**Fix implemented**: Cache `freqs_cis` on the active device and reuse it (no per-call `.to(device)` transfers).

---

### WARN-8: Unconditional weight tying for Qwen3-8B ✅ DONE

- **File**: `src/musubi_tuner/zimage/zimage_utils.py`
- **Line**: 114

```python
sd["lm_head.weight"] = sd["model.embed_tokens.weight"]  # unconditional
```

**Problem**: Qwen3-8B has `tie_word_embeddings=False` but weights are force-tied. No functional impact (lm_head is never used for text encoding — only `hidden_states[-2]` is extracted).

**Fix implemented**: Only force-tie for 4B; for 8B, preserve `lm_head.weight` when present and only synthesize a fallback weight when missing (to satisfy strict loading).

---

### WARN-9: `--fp8_llm` ignored in interactive/batch mode ✅ DONE

- **File**: `src/musubi_tuner/zimage_generate_image.py`
- **Lines**: 779-780 (load_shared_models), 816 (batch path)

**Problem**: `--fp8_llm` works in single-prompt mode but `load_shared_models` hardcodes `torch.bfloat16` for text encoder. This is a "flag lies" issue — the user expects FP8 but gets bf16.

**Fix implemented**: Dtype selection is now centralized and used consistently in single/batch/interactive modes. Also adds a warning + fallback to bf16 when the text encoder runs on CPU (FP8 is CUDA-only).

---

## Phase 6: Improvements

### IMPROV-1: Expose CFG truncation and normalization

- **File**: `src/musubi_tuner/zimage_generate_image.py`
- **Scope**: Add CLI args + wire into denoising loop
- **See**: WARN-2 for implementation details

---

### IMPROV-2: Dynamic shift computation

- **Files**: `src/musubi_tuner/zimage/zimage_config.py` (constants exist), `src/musubi_tuner/zimage/zimage_utils.py` (add function)

The constants exist but no function uses them:
```python
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15
```

**Implementation**:
```python
def compute_shift(image_seq_len):
    mu = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN) * (image_seq_len - BASE_IMAGE_SEQ_LEN) + BASE_SHIFT
    return mu
```

Add `--dynamic_shift` flag to generation script that auto-computes shift from resolution.

---

### IMPROV-3: Optional noise_refiner LoRA targeting

- **Files**: `src/musubi_tuner/networks/lora_zimage.py`, `src/musubi_tuner/networks/network_arch.py`
- **Context**: Diffusers applies LoRA to noise_refiner layers; Blissful Tuner excludes them via `r".*(_modulation|_refiner).*"` pattern

**Implementation**: Add `--include_refiner` network arg that removes `_refiner` from exclude pattern:
```python
exclude_patterns = [r".*_modulation.*"]  # Always exclude modulation
if not include_refiner:
    exclude_patterns.append(r".*_refiner.*")
```

---

### IMPROV-4: Mask-weighted loss for Z-Image

- **See**: WARN-1 for full implementation scope
- **Files**: `zimage_cache_latents.py`, `image_video_dataset.py`, CLAUDE.md

---

### IMPROV-5: Empty-string negative prompts

- **File**: `src/musubi_tuner/zimage_generate_image.py`
- **Line**: ~588

**Current**: Uses `torch.zeros_like(embed)` for negative conditioning when no negative prompt specified.
**Official**: Encodes empty string `""` through the text encoder (includes chat template tokens).

**Impact**: May produce slightly different CFG results. The empty-string approach preserves the text encoder's token structure.

---

### IMPROV-6: Latent preview in generation

- **File**: `src/musubi_tuner/zimage_generate_image.py`
- **Context**: Other Blissful Tuner architectures have real-time latent preview during denoising. Z-Image denoising loop (lines 591-615) lacks preview hooks.

---

### IMPROV-7: Regression test for LoRA multiplier defaults

- **File**: `tests/` (new test)
- **Context**: BUG-2 (the lora_multiplier crash) was already fixed across all scripts, but there is no test to prevent regression.

**Implementation**: A lightweight test that simulates argparse with `--lora_weight` but no `--lora_multiplier` for each generation script and asserts no crash during argument processing.

---

## Confirmed Correct (No Action Needed)

These items were verified correct across all sources:

| Item | Location | Verification |
|------|----------|-------------|
| Flow matching timestep reversal | `zimage_train_network.py:317` | `(1000.0 - timesteps) / 1000.0` |
| Flow matching target | `zimage_train_network.py:338` | `latents - noise` |
| Inference output negation | `zimage_generate_image.py:612` | `-noise_pred.squeeze(2)` |
| CFG formula | `zimage_generate_image.py:608` | `pos + scale * (pos - neg)` matches official |
| Latent encode normalization | `zimage_train_network.py:270` | `(raw - 0.1159) * 0.3611` |
| Latent decode denormalization | `zimage_utils.py:23` | `(latents / 0.3611) + 0.1159` |
| AdaLN block: 4 outputs, no activation | `zimage_model.py:243,269` | Bare Linear, chunk(4) |
| AdaLN FinalLayer: SiLU + 1 output | `zimage_model.py:310-313` | Sequential(SiLU, Linear) |
| TimestepEmbedder mid_size=1024 | `zimage_model.py:423` | Matches reference |
| Tanh gating | `zimage_model.py:271` | `gate_msa.tanh()` |
| Scale formula: `1 + scale` (no shift) | `zimage_model.py:272` | Matches all sources |
| QK-Norm before RoPE | `zimage_model.py:181-188` | norm_q/norm_k → apply_rotary_emb |
| Sandwich-Norm post-output | `zimage_model.py:274-276` | `norm2(attn_out)` = correct |
| FinalLayer nn.LayerNorm | `zimage_model.py:308` | Not RMSNorm |
| Text encoder: hidden_states[-2] | `zimage_utils.py:213-218` | Penultimate layer |
| Qwen2Tokenizer (not Qwen3) | `zimage_utils.py:8,160` | Correct tokenizer |
| Chat template enable_thinking=True | `zimage_utils.py:186-192` | Matches official |
| Max seq length 512 | `zimage_config.py:28` | Correct |
| VAE float32 always | `zimage_autoencoder.py:488` | force_upcast equivalent |
| VAE latent_channels=16 | `zimage_config.py:55` | Correct (not SD1.x's 4) |
| VAE encode + decode | `zimage_autoencoder.py:389-400` | Both implemented |
| Scaling factor 0.3611, shift 0.1159 | `zimage_config.py:58-59` | Matches HF config |
| Target modules: ZImageTransformerBlock | `lora_zimage.py` | Correct |
| Exclude: _modulation + _refiner | `network_arch.py:90-93` | Correct pattern |
| 5D frame dimension | `zimage_train_network.py:293,335` | unsqueeze(2)/squeeze(2) |
| Dynamic shift constants | `zimage_config.py:46-49` | BASE_SHIFT=0.5, MAX_SHIFT=1.15 |
| Sigma schedule with Flux-style shift | `zimage_utils.py:256-269` | Correct formula |
| Resolution divisible by 16 check | `zimage_generate_image.py:234` | VAE_SCALE * 2 = 16 |
| RMSNorm float32 upcasting | `zimage_model.py:96-99` | Deliberate stability improvement |
| OmniBase caching (SigLIP2 + control) | `zimage_cache_latents.py:78-198` | Correct cache keys |
| FFN hidden dim: int(dim/3*8) = 10240 | `zimage_model.py:235` | Correct formula |
| `--lora_multiplier` defaults | All `*_generate_*.py` | All use `default=None` (already fixed) |

---

## Execution Order

Recommended order for addressing items:

1. **Phase 1** (Critical bugs: BUG-1, BUG-5) — Fix immediately; BUG-1 is multi-file, BUG-5 is localized
2. **Phase 2** (Medium bugs: BUG-3, BUG-4, BUG-6) — Fix alongside Phase 1
3. **Phase 3** (Doc corrections: DOC-1, DOC-2, DOC-3) — Quick fixes, prevents future confusion
4. **Phase 4** (Medium warnings) — Feature gaps, implement as time allows
5. **Phase 5** (Low warnings) — Code quality, batch during cleanup pass
6. **Phase 6** (Improvements) — Enhancement backlog

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2026-02-17 | Initial audit from 4 parallel sub-agents |
| v2 | 2026-02-17 | Human review corrections: BUG-1 re-root-caused (LyCORIS targeting, not import), BUG-2 marked resolved (already fixed), BUG-5 added (--save_merged_model), BUG-6 added (--flash3), DOC-3 added (zimage.md control image claim), IMPROV-7 added (regression test), WARN-5 reframed as tradeoff, BUG-4 fix uses config constants |
| v3 | 2026-02-17 | Phase 1 & 2 implemented: BUG-1 (lycoris.py auto-detect extended to 11 block types, merge_lora_weights gains extra_unet_targets param + observability guardrail, Z-Image import fixed + explicit targets), BUG-5 (non-LyCORIS save path added), BUG-3 (apply_fp16_downscale passed through checkpoint), BUG-4 (uses config constants), BUG-6 (flash3→flash fallback with warning) |

---

*Generated from audit of Z-Image pipeline against `docs/z-image-integration-reference.md` v2.3 (4 sources: HF configs, official GitHub, technical report, diffusers repository). Revised after human review with corrected root causes and additional findings.*
