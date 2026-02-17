# WAN 2.2 Training Pipeline Audit

Comprehensive audit of the WAN 2.2 training pipeline in blissful-tuner, conducted incrementally against the architecture reference in `docs/wan22_architecture.md`.

## Audit Rounds

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Latent Caching + Text Encoder Caching | COMPLETE |
| 2 | Training Pipeline (WanNetworkTrainer + base NetworkTrainer) | COMPLETE |
| 3 | LoRA Module + WAN Model Code | COMPLETE |
| 4 | Generation / Inference | COMPLETE |
| 5 | Dataset Config + Mask Loss | COMPLETE |
| 6 | Tests Coverage | COMPLETE |

---

## Round 1: Latent Caching + Text Encoder Caching

### Files Reviewed
- `wan_cache_latents.py` (root wrapper)
- `src/musubi_tuner/wan_cache_latents.py` (implementation)
- `wan_cache_text_encoder_outputs.py` (root wrapper)
- `src/musubi_tuner/wan_cache_text_encoder_outputs.py` (implementation)
- `src/musubi_tuner/wan/modules/vae.py` (VAE integration)
- `src/musubi_tuner/wan/modules/t5.py` (T5 integration)
- `src/musubi_tuner/wan/modules/tokenizers.py` (tokenizer handling)
- `src/musubi_tuner/dataset/image_video_dataset.py` (dataset + cache format)
- `src/musubi_tuner/dataset/config_utils.py` (config parsing)
- `src/musubi_tuner/wan/configs/*.py` (WAN configs)

### Positive Observations

1. **VAE normalization values match architecture reference exactly** (`vae.py:664-701` vs `wan22_architecture.md` latent normalization section).
2. **Variable-length T5 caching is well-implemented** — embeddings are trimmed to actual token count before saving (`t5.py:520-526`), avoiding wasted disk space.
3. **Batch reconstruction correctly handles varlen keys** — `image_video_dataset.py:1254-1280` distinguishes `varlen_` keys from fixed-length, avoiding incorrect stacking.
4. **Mask weights saved as float32** for precision (`image_video_dataset.py:467`), preventing bf16 quantization artifacts on mask boundaries.
5. **NaN protection in cache saving** (`image_video_dataset.py:752-755`) prevents corrupt caches.
6. **Mixed-mask batches handled** — `BucketBatchManager.__getitem__` pads missing masks with ones for proper batch alignment.
7. **T5 attention correctly omits 1/√d_k scaling** (`t5.py:115-118`), faithful to T5 paper.
8. **No double normalization** — VAE normalizes during `encode()`, training loop uses cached latents directly (verified: no latent normalization in `wan_train_network.py`).

### Issues Found

---

#### LC-01: No T=4k+1 Frame Count Validation

**Severity**: HIGH
**Category**: Gap / Missing Validation
**File**: `src/musubi_tuner/wan_cache_latents.py:45-60`

**Problem**: The WAN VAE requires video frame counts to satisfy `T = 4k + 1` (e.g., 81, 85, 89) for clean temporal compression. The caching code does not validate this constraint. If a non-conforming frame count is provided:
- The main latent encoding will produce a floor-divided result (undefined behavior)
- The I2V mask construction (pre-fix) could fail with a **cryptic reshape error** if `(4 + num_frames - 1) % 4 != 0`

**Current Code**:
```python
contents = torch.stack([torch.from_numpy(item.content) for item in batch])
# ... no frame count validation before VAE encoding
```

**Suggested Fix** (upgraded from warning to hard error per review — warning allows semantically broken caches):
```python
F = contents.shape[2]
if F > 1 and (F - 1) % 4 != 0:
    if args.allow_nonconforming_frames:
        logger.warning(f"Video frame count {F} does not satisfy T=4k+1. Proceeding anyway (--allow_nonconforming_frames).")
    else:
        raise ValueError(
            f"Video frame count {F} does not satisfy T=4k+1 constraint for WAN VAE "
            f"(expected 5, 9, 13, ..., 77, 81, 85, ...). Use --allow_nonconforming_frames to override."
        )
```
Alternative: auto-pad to next valid 4k+1 by repeating the last frame, with explicit log message.

---

#### LC-02: I2V Mask Construction Undocumented and Non-Obvious

**Severity**: MEDIUM
**Category**: Documentation / Code Clarity
**File**: `src/musubi_tuner/wan_cache_latents.py:87-103`

**Problem** (pre-fix): The I2V 36-channel conditioning construction used a clever but undocumented pixel→latent temporal reshape trick. It worked for valid `T=4k+1` lengths, but was difficult to read and tightly coupled to the frame-count constraint.

**Fix (LC-06)**: Construct the mask directly from `lat_f` (latent frames), which is already known after VAE encoding:

```python
msk = torch.zeros(B, 4, lat_f, lat_h, lat_w, dtype=vae.dtype, device=vae.device)
msk[:, :, 0] = 1  # First latent frame is conditioned (known image)
```

This is equivalent for the standard WAN I2V training cache path (first-frame conditioning), is easier to understand, and removes the fragile `view(... // 4, 4, ...)` reshape.

---

#### LC-03: CLIP/WAN Version Confusion for I2V Caching

**Severity**: MEDIUM
**Category**: Inconsistency / UX
**File**: `src/musubi_tuner/wan_cache_latents.py:266-267, 295-297`

**Problem**: The `--clip` flag doubles as an I2V mode selector (`if args.clip is not None: args.i2v = True`), and uses `wan_i2v_14B` (WAN 2.1) config for CLIP dtype. WAN 2.2 I2V does NOT use CLIP. A user training WAN 2.2 I2V who mistakenly provides `--clip` will get WAN 2.1-style caching without any error.

**Architecture Reference** (`wan22_architecture.md`):
> The image conditioning is done via extra latent channels, *not* via CLIP tokens. `image_encoder: [null, null]` in Diffusers confirms this.

**Fix**: Added an optional `--task` argument to `wan_cache_latents.py` and a task-aware validation step that raises a `ValueError` if `--clip` is provided for WAN 2.2 tasks (e.g. `i2v-A14B`, `t2v-A14B`). This prevents accidentally generating WAN 2.1-style CLIP caches for WAN 2.2.

**Tests**: `tests/test_wan_clip_guard_cache_latents.py`

---

#### LC-04: Missing Mask Spatial Dimension Validation

**Severity**: MEDIUM
**Category**: Gap / Silent Failure
**File**: `src/musubi_tuner/wan_cache_latents.py:138-146`

**Problem**: Mask content is loaded and downsampled to latent space without validating that mask dimensions match the corresponding image/video dimensions. A dimension mismatch produces silently incorrect mask weights.

**Current Code**:
```python
if item.mask_content is not None:
    mask = torch.from_numpy(item.mask_content).unsqueeze(0).unsqueeze(0)  # 1, 1, H, W
    mask = mask.float() / 255.0
    mask = F.interpolate(mask, size=(lat_h, lat_w), mode="area")  # no validation!
```

**Impact**: Wrong mask weights lead to incorrect loss weighting — training silently degrades.

**Fix**: Added strict validation that `mask_content` is a 2D grayscale mask and that its spatial dimensions match the bucketed content dimensions before encoding. Mismatches now raise a `ValueError` with a clear message instead of silently producing misaligned `mask_weights`.

**Tests**: `tests/test_wan_mask_spatial_validation.py`

---

#### TC-01: Missing Context Length / Attention Mask in Text Cache

**Severity**: MEDIUM
**Category**: Gap
**File**: `src/musubi_tuner/dataset/image_video_dataset.py:780-788`

**Problem**: WAN text encoder caches save variable-length T5 embeddings but do NOT save attention masks or context lengths. During training, reconstructing masks requires inferring lengths from tensor shapes. By contrast, HunyuanVideo caching (`save_text_encoder_output_cache`, line 763-777) explicitly saves masks.

**Current Code**:
```python
def save_text_encoder_output_cache_wan(item_info, embed):
    sd = {}
    sd[f"varlen_{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()
    # ❌ No mask or context_len saved
```

**Comparison** (HV does save masks):
```python
def save_text_encoder_output_cache(item_info, embed, mask, is_llm):
    sd[f"{text_encoder_type}_mask"] = mask.detach().cpu()  # ✅
```

**Impact**: Not a correctness bug (lengths can be inferred from variable-length tensor shapes), but an efficiency and consistency gap. If the training loop ever needs explicit masks (e.g., for cross-attention mask optimization), they'd need to be reconstructed.

**Suggested Fix**: Optionally save `context_len` as an integer tensor alongside the embedding.

---

#### TC-02: Empty Caption Handling Not Logged

**Severity**: LOW
**Category**: Improvement
**File**: `src/musubi_tuner/wan_cache_text_encoder_outputs.py:23`

**Problem**: Empty captions (`""`) are silently processed by the T5 tokenizer, producing very short embeddings (just special tokens). No warning is logged, making it hard to debug training issues caused by empty captions.

**Fix**: Added a warning during WAN text encoder output caching when empty/whitespace captions are encountered.

**Tests**: `tests/test_wan_text_cache_empty_captions_warning.py`

---

#### LC-05: Redundant dtype Casts with Uncertain Comments

**Severity**: LOW
**Category**: Code Clarity
**File**: `src/musubi_tuner/wan_cache_latents.py:47, 80, 106, 180`

**Problem**: Multiple lines cast latents to `vae.dtype` after encoding within `torch.amp.autocast(dtype=vae.dtype)`. The comment "we are not sure if this is correct" adds uncertainty. Since autocast already produces outputs in the target dtype, these casts are likely no-ops.

```python
latent = latent.to(vae.dtype)  # convert to bfloat16, we are not sure if this is correct
```

**Suggested Fix**: Verify VAE output dtype and either remove the casts or replace the uncertain comment with a definitive one.

---

#### TC-03: T5 Attention "No Scaling" Undocumented in Code

**Severity**: NOTE
**Category**: Documentation
**File**: `src/musubi_tuner/wan/modules/t5.py:115-118`

**Problem**: The T5 attention correctly omits `1/√d_k` scaling per the T5 paper, but has no code comment explaining this deliberate omission. Developers may mistake it for a bug.

```python
# T5 does not use 1/sqrt(d_k) scaling (absorbed into relative attention bias)
attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
```

---

#### TC-04: FP8 T5 Accelerator Usage Undocumented

**Severity**: NOTE
**Category**: Documentation
**File**: `src/musubi_tuner/wan_cache_text_encoder_outputs.py:60-61`

**Problem**: When `--fp8_t5` is set, an `accelerator` with `mixed_precision="bf16"` is created, then `accelerator.autocast()` wraps the T5 forward pass. The interaction between FP8 quantized weights and bf16 autocast is not explained.

---

## Round 2: Training Pipeline (WanNetworkTrainer + Base NetworkTrainer)

### Files Reviewed
- `src/musubi_tuner/wan_train_network.py` (WanNetworkTrainer — dual-expert switching, I2V forward, timestep sampling)
- `src/musubi_tuner/hv_train_network.py` (base NetworkTrainer — main train loop, loss computation, optimizer, sampling)
- `src/musubi_tuner/wan/modules/model.py` (WanModel DiT — forward pass, patch embedding, RoPE)
- `src/musubi_tuner/modules/scheduling_flow_match_discrete.py` (FlowMatchDiscreteScheduler)
- `src/musubi_tuner/wan/utils/fm_solvers.py` (DPM++ inference scheduler)
- `src/musubi_tuner/wan/utils/fm_solvers_unipc.py` (UniPC inference scheduler)
- `src/musubi_tuner/networks/lora_wan.py` (WAN LoRA module)
- `src/musubi_tuner/networks/lora.py` (base LoRA implementation)
- `src/musubi_tuner/modules/mask_loss.py` (mask loss + prior preservation)

### Positive Observations

1. **Dual-expert state dict swapping is well-designed** — uses `assign=True` for efficient weight transfer, handles both block-swap and offload cases, verifies key integrity with `strict=True` + assertions (`wan_train_network.py:599-640`).
2. **LoRA correctly stays separate during expert swap** — LoRA params live in a separate `LoRANetwork` module, not in the base model's `state_dict()`. Expert switching swaps base weights only. One set of LoRA params is trained across both experts.
3. **Prior model context manager is elegant** — disables LoRA without mode switching, keeps model in `train()` mode (`hv_train_network.py:478-517`). Matches OneTrainer's approach.
4. **Fail-fast mask validation on first step** — checks mask weights early, preventing wasted training time (`hv_train_network.py:2408-2416`).
5. **Smart prior preservation skip** — skips teacher forward when prior_mask would be zero, saving VRAM/time (`hv_train_network.py:2446-2458`).
6. **Mask loss module is exemplary** — proper separation of concerns, comprehensive validation, clear error messages, numerical stability via float32 sums, flexible video/layered layouts (`modules/mask_loss.py`).
7. **Comprehensive timestep sampling** — supports 8 methods (uniform, sigmoid, shift variants, logsnr, qinglong hybrid) with clean parameterization (`hv_train_network.py:974-1089`).
8. **Flow matching loss formulation is correct** — `target = noise - latents` matches rectified flow velocity. Timestep shift formulas are identical across training (`sd3_time_shift`) and inference (DPM++, UniPC) schedulers.

### Issues Found

---

#### TP-01: Timestep Bucketing Should Error with Dual-Expert, Not Warn

**Severity**: MEDIUM
**Category**: UX / Correctness Risk
**File**: `src/musubi_tuner/wan_train_network.py:88-91`

**Problem**: The code warns that `num_timestep_buckets` doesn't work well with high/low models training, but doesn't explain why or disable it. The rejection sampling loop (lines 567-585) retries up to 100 times per batch item to find timesteps matching the current expert's range, defeating the purpose of bucketing and potentially skewing the timestep distribution.

**Current Code**:
```python
if args.num_timestep_buckets is not None:
    logger.warning(
        "num_timestep_buckets is not working well with high and low models training"
    )
```

**Impact**: Users may ignore the warning. High rejection rates slow training and change the effective timestep distribution.

**Suggested Fix**: Change to `raise ValueError(...)` explaining the incompatibility, or at minimum improve the warning with actionable guidance.

---

#### TP-02: Rejection Sampling for Dual-Expert Timesteps is Wasteful

**Severity**: MEDIUM
**Category**: Design Concern / Performance
**File**: `src/musubi_tuner/wan_train_network.py:567-585`

**Problem**: When a batch's first sample determines the active expert (high/low noise), remaining samples must match via rejection sampling — retry up to 100 times per sample. For boundary=0.875, ~12.5% of timesteps fall in the high-noise range, so finding a high-noise timestep has ~12.5% acceptance rate per try.

**Impact**: Not a correctness bug (produces correct conditional distribution), but wasteful. Typical rejection count: ~8 for high-noise, ~1.1 for low-noise. Worst case: fallback at line 586-591 clamps to boundary.

**Suggested Fix** (REVISED — original "uniform in range" fix was incorrect; it silently changes the timestep distribution for non-uniform samplers like shift/logsnr/qinglong):

**Option A (safest)**: Keep rejection sampling but vectorize — sample N candidates per item at once, take first valid, increase N on failure. Preserves `p(t | t in region)` for all sampler types.

**Option B**: Only use direct sampling when `--timestep_sampling uniform`. For non-uniform samplers, implement truncated CDF/inverse-CDF for the specific sampler.

**Option C (bigger design change)**: Stop forcing one expert per batch — split batch into high/low sub-batches and do two forwards. Removes rejection complexity but changes memory/performance tradeoffs.

---

#### TP-03: FP8 Norm Preservation Not Validated for WAN 2.2

**Severity**: MEDIUM
**Category**: Gap / Validation
**File**: `src/musubi_tuner/wan_train_network.py:68-75`

**Problem**: WAN 2.2 uses `WanLayerNorm` and `WanRMSNorm` which compute in FP32. The FP8 quantization code excludes keys matching `"norm"`, but there's no explicit post-FP8 check that WAN-specific norms and modulation layers remain in FP32.

**Impact**: If FP8 accidentally quantizes norms, training diverges with NaN gradients.

**Suggested Fix**: Add a validation step after FP8 application to verify no norm/modulation parameter was quantized.

---

#### TP-04: Gradient Accumulation + Expert Swap Edge Case

**Severity**: MEDIUM
**Category**: Potential Bug / Edge Case
**File**: `src/musubi_tuner/wan_train_network.py:654` + `src/musubi_tuner/hv_train_network.py:2420`

**Problem**: With `gradient_accumulation_steps > 1`, consecutive micro-batches may trigger expert swap (if timesteps cross the boundary). Gradients computed on expert A accumulate alongside gradients from expert B, but LoRA optimizer sees both as one optimization step. While LoRA params are shared across experts (so this isn't strictly wrong), it creates an unusual optimization landscape.

**Impact**: Rare (requires batch to straddle boundary), but could cause training instability with large accumulation steps.

**Suggested Fix**: Document limitation: "gradient accumulation > 1 with dual-expert is supported but may produce mixed-expert gradient steps." Or detect and log when accumulation spans an expert swap.

---

#### TP-05: Block Swap + Dual-Expert Interaction Undertested

**Severity**: MEDIUM
**Category**: Gap / Testing
**File**: `src/musubi_tuner/wan_train_network.py:84-87, 612-638`

**Problem**: Block swapping with dual-expert has special code paths (lines 630-638) that swap the full state dict while blocks are partially on CPU. The `offload_inactive_dit` guard exists (line 84-87), but the block-swap path itself lacks validation that both experts' block layouts match.

**Impact**: Untested combinations may cause weight misalignment or silent corruption.

**Suggested Fix**: Add integration test for `--blocks_to_swap N --dit_high_noise`. Log expert swap events with block device info.

---

#### TP-06: LoRA + Dual-Expert Behavior Undocumented

**Severity**: LOW
**Category**: Documentation
**File**: `src/musubi_tuner/wan_train_network.py:503-537`

**Problem**: The dual-expert LoRA training model — one set of LoRA params trained across both high/low noise experts via base model swapping — is entirely undocumented. Users may wonder if they're training separate LoRAs per expert.

**Impact**: User confusion. The current behavior (shared LoRA) is correct but non-obvious.

**Suggested Fix**: Add documentation in `docs/wan.md` explaining that dual-expert LoRA uses one trainable parameter set applied to whichever expert is active for each batch.

---

#### TP-07: Prior Preservation + Expert Swap Interaction Undocumented

**Severity**: LOW
**Category**: Documentation
**File**: `src/musubi_tuner/hv_train_network.py:2460-2476`

**Problem**: Prior preservation's teacher forward pass with dual-expert works correctly (same expert, LoRA disabled), but this interaction is not documented. The code path: main `call_dit()` swaps to correct expert → prior `call_dit()` sees `current == next` so no swap → correct expert with LoRA off.

**Impact**: Developer confusion only. Code is correct.

**Suggested Fix**: Add inline comment explaining prior + expert swap interaction.

---

#### TP-08: Timestep Convention Inconsistency Across Codebase

**Severity**: LOW
**Category**: Code Clarity / Documentation
**File**: Multiple

**Problem**: Timesteps are represented in three formats:
- Math: `t ∈ [0, 1]` (flow matching)
- Scheduler: `timesteps ∈ [1, 1000]` (after `t * 1000 + 1`)
- Boundary comparison: `t / 1000.0` (converts back to [0, 1])

**Impact**: Cognitive load for developers. Not a correctness issue.

**Suggested Fix**: Add inline comments at each conversion point, and a section in the architecture doc explaining the representation chain.

---

#### TP-09: Guidance Scale Tuple Ordering Not Validated

**Severity**: LOW
**Category**: UX
**File**: `src/musubi_tuner/wan_train_network.py:231`, `wan/configs/wan_t2v_A14B.py:42`

**Problem**: Config uses `sample_guide_scale = (3.0, 4.0)` with comment "(low noise, high noise)". The inference code uses `[0]` for low noise. There's no validation that the tuple has exactly 2 elements, and some doc sections use reversed ordering.

**Suggested Fix**: Add assertion in `handle_model_specific_args` for dual-expert: `assert len(config.sample_guide_scale) == 2`.

---

#### TP-10: torch.compile Key Patching for Expert Swap is Fragile

**Severity**: LOW
**Category**: Maintainability
**File**: `src/musubi_tuner/wan_train_network.py:602-610`

**Problem**: The `patch_fn` manually inserts `._orig_mod.` into state dict keys for torch.compile compatibility. This relies on torch.compile's internal naming convention, which could change.

**Impact**: Minor — `strict=True` assertions will catch any breakage immediately.

---

#### TP-11: normalize_per_sample Default Could Auto-Enable

**Severity**: NOTE
**Category**: UX
**File**: `src/musubi_tuner/modules/mask_loss.py:300`

**Problem**: `normalize_per_sample` defaults to False but docs recommend True when prior preservation is enabled. The recommended mode isn't the default.

**Suggested Fix**: Auto-enable when `prior_preservation_weight > 0` and user hasn't explicitly set the flag.

---

#### TP-12: Missing EMA for LoRA Training

**Severity**: NOTE
**Category**: Feature Request
**File**: N/A

**Problem**: No native Exponential Moving Average support for LoRA weights during training. Workaround exists via `lora_post_hoc_ema.py` for post-hoc merge.

**Impact**: Minor — post-hoc EMA is available but less convenient than online EMA.

---

## Round 3: LoRA Module + WAN Model Code

### Files Reviewed
- `src/musubi_tuner/networks/lora_wan.py` (WAN-specific LoRA wrapper)
- `src/musubi_tuner/networks/lora.py` (base LoRA implementation)
- `src/musubi_tuner/networks/network_arch.py` (architecture registry)
- `src/musubi_tuner/wan/modules/model.py` (WanModel, WanAttentionBlock, RoPE, modulation)
- `src/musubi_tuner/wan/modules/attention.py` (flash_attention, variable-length sequences)
- `src/musubi_tuner/wan/configs/wan_t2v_A14B.py`, `wan_i2v_A14B.py`, `__init__.py`

### Positive Observations

1. **LoRA target module selection is correct** — `WanAttentionBlock` captures all QKV/O projections and FFN layers (10 Linear per block for T2V, 12 for WAN 2.1 I2V).
2. **Exclude pattern is comprehensive** — correctly excludes `patch_embedding`, `text_embedding`, `time_embedding`, `time_projection`, `norm`, `head`. Modulation is inherently excluded (it's a `nn.Parameter`, not a module).
3. **No list-based batching issue** — WAN converts list inputs to batched tensor at `model.py:1005` before passing through blocks. All Linear layers receive standard `[B, L, C]` tensors.
4. **Per-token time embedding (WAN 2.2) is correctly implemented** — `t` expanded to `[B, seq_len]`, processed through sinusoidal + MLP, producing `[B, seq_len, 6, dim]` modulation.
5. **3D RoPE frequency splits are mathematically consistent** — init produces `[22, 21, 21]` complex freqs, matching apply function's expected split on `c=d//2=64`.
6. **FP32 norms preserved** — both `WanRMSNorm` and `WanLayerNorm` use `.float()` before normalization and `.type_as(x)` after.
7. **`freqs` stored as plain attribute** (not buffer) to preserve complex128 dtype during `model.to(dtype)`. Manual device movement at forward time.
8. **LoRA initialization correct** — Kaiming uniform for `lora_down`, zeros for `lora_up` (identity at init).
9. **I2V config correct** — `in_dim=36` for I2V, `in_dim=16` for T2V, matching the 36-channel conditioning.

### Issues Found

---

#### DM-01: NAG Alpha Parameter Typo — User Setting Always Ignored

**Severity**: HIGH
**Category**: Bug / Typo
**File**: `src/musubi_tuner/wan/modules/model.py:535`

**Problem**: Line 535 has `kwargs.get("nah_alpha", 0.5)` — a typo of `"nah_alpha"` instead of `"nag_alpha"`. The generation script passes `"nag_alpha"` (correctly) at `wan_generate_video.py:681`, but it never matches the mistyped key. Users' `--nag_alpha` CLI flag is silently ignored; the hardcoded default of 0.5 is always used.

**Current Code**:
```python
nag_alpha = kwargs.get("nah_alpha", 0.5)  # ❌ "nah" instead of "nag"
```

**Impact**: NAG (Negative Attention Guidance) alpha parameter is non-configurable. Users who set `--nag_alpha 0.3` or any value get 0.5 silently.

**Suggested Fix**:
```python
nag_alpha = kwargs.get("nag_alpha", 0.5)
```

---

## Round 4: Generation / Inference

### Files Reviewed
- `src/musubi_tuner/wan_generate_video.py` (main generation — denoising loop, scheduler, CFG, dual-expert, I2V, LoRA loading)
- `src/musubi_tuner/wan/utils/fm_solvers.py` (DPM++ scheduler)
- `src/musubi_tuner/wan/utils/fm_solvers_unipc.py` (UniPC scheduler)
- `src/blissful_tuner/guidance.py` (parse_scheduled_cfg, perpendicular negative, CFGZero*)

### Positive Observations

1. **Dual-expert inference boundary logic is correct** — `(t / 1000.0) >= timestep_boundary` correctly selects high-noise expert for early denoising steps.
2. **I2V 36-channel construction is correct** — 4 mask + 16 image + 16 noise = 36 channels, matching config `in_dim=36`.
3. **Flow shift applied consistently** — all schedulers (DPM++, UniPC, FlowMatch, LCM) receive same `shift` parameter.
4. **LoRA loading is robust** — auto-detects format, converts Diffusers if needed, supports LyCORIS backend.
5. **V2V/I2I noise preparation is correct** — uses `prepare_v2v_noise()` / `prepare_i2i_noise()` from common extensions.
6. **NAG context forwarding works** — `nag_context` correctly passed through model calls.
7. **Batch and interactive modes well-designed** — efficient model reuse with pre-encoded text contexts.

### Issues Found

---

#### GN-01: Hardcoded 5-Second Sleep for Block Swap Synchronization

**Severity**: MEDIUM
**Category**: Fragile Synchronization
**File**: `src/musubi_tuner/wan_generate_video.py:1600-1601`

**Problem**: When switching from high-noise to low-noise model with `blocks_to_swap > 0`, a hardcoded `time.sleep(5)` is used as synchronization. This is fragile — too short on slow systems, wasteful on fast ones.

**Suggested Fix**: Replace with `synchronize_device()` (already imported at line 43 of `wan_generate_video.py`). This is device-agnostic and consistent with the rest of the codebase:
```python
synchronize_device(device)
```

---

#### GN-02: CFG Skip Mode Silently Ignored Without cfg_apply_ratio

**Severity**: LOW
**Category**: UX / Silent Ignore
**File**: `src/musubi_tuner/wan_generate_video.py:1509`

**Problem**: If user sets `--cfg_skip_mode early` without `--cfg_apply_ratio`, the condition `cfg_skip_mode != "none" and cfg_apply_ratio is not None` is False, so the skip mode is silently ignored. CFG falls through to default "always apply" behavior without warning.

**Suggested Fix**: Add warning when `cfg_skip_mode != "none"` but `cfg_apply_ratio is None`.

---

#### GN-03: perp_neg and cfgzerostar Mutual Exclusion Not Validated

**Severity**: LOW
**Category**: UX
**File**: `src/musubi_tuner/wan_generate_video.py:1653-1660`

**Problem**: `--perp_neg` and `--cfgzerostar_scaling` are mutually exclusive via `elif` but no upfront validation warns users. `perp_neg` silently takes precedence.

**Suggested Fix**: Add upfront check raising `ValueError` if both are set.

---

#### GN-04: Lazy Loading Model Cleanup Incomplete

**Severity**: LOW
**Category**: Resource Management
**File**: `src/musubi_tuner/wan_generate_video.py:1694-1697`

**Problem**: After lazy loading, only the current active model is deleted. Other models in the list may remain in VRAM.

---

## Round 5: Dataset Config + Mask Loss

### Files Reviewed
- `src/musubi_tuner/dataset/config_utils.py` (dataset config parsing)
- `src/musubi_tuner/dataset/image_video_dataset.py` (dataset loading, batch construction, cache format)
- `src/musubi_tuner/modules/mask_loss.py` (mask loss + prior preservation)

### Positive Observations

1. **Mask loss module remains exemplary** — proper shape validation, compact broadcasting, per-sample normalization, float32 accumulation, flexible video/layered layouts.
2. **WAN latent cache format correctly supported** — all required keys (`latents_*`, `mask_weights_*`, `latents_image_*`, `clip_*`) properly handled.
3. **Varlen T5 correctly preserved as list** — not stacked, maintaining per-sample variable lengths for WAN's cross-attention.
4. **Mixed mask batches handled** — missing masks filled with all-ones for proper batch alignment.
5. **Centralized mask loss in base trainer** — all architectures inherit consistent behavior.

### Issues Found

---

#### DS-01: No Warning When mask_directory Set Without use_mask_loss

**Severity**: LOW
**Category**: UX / Configuration
**File**: `src/musubi_tuner/dataset/config_utils.py`

**Problem**: If user sets `mask_directory` in TOML but forgets `--use_mask_loss`, masks are cached but never used in training. No warning is emitted.

**Fix**: Added a training startup warning when the dataset blueprint configures mask sources (`mask_directory`, `alpha_mask`, or `require_mask`) but `--use_mask_loss` is disabled. This prevents silently ignoring masks during training.

**Tests**: `tests/test_mask_loss_disabled_warning.py`

---

## Round 6: Tests Coverage

### Files Reviewed
- All 19 test files in `tests/`

### Test Coverage Summary

| Area | Coverage | Notes |
|------|----------|-------|
| LoRA conversion/naming (WAN) | Good (9 tests) | Roundtrip, prefix stripping, v_img naming |
| Mask loss (generic) | Excellent (16 tests) | Prior preservation, per-sample, shapes |
| WAN dual-expert training | **None** | Critical gap |
| WAN dataset loading | **None** | No varlen T5 or batch tests |
| WAN mask loss integration | **None** | No video layout integration test |
| WAN I2V conditioning | **None** | No I2V cache/training test |

### Critical Test Gaps

---

#### TT-01: No WAN Dual-Expert Training Tests

**Severity**: HIGH (test gap)
**Category**: Testing
**Description**: No tests verify timestep boundary logic, model swapping during training, or offload behavior for WAN 2.2 dual-expert mode.

---

#### TT-02: No WAN Dataset Loading Tests

**Severity**: HIGH (test gap)
**Category**: Testing
**Description**: No tests verify varlen T5 cache loading, batch construction, or frame count validation for WAN.

---

#### TT-03: No WAN Mask Loss Integration Test

**Severity**: MEDIUM (test gap)
**Category**: Testing
**Description**: Generic mask loss tests exist but no test verifies WAN trainer correctly applies mask loss to `(B, C, F, H, W)` video tensors.

---

## Consolidated Issue List

| # | Severity | Round | Area | Summary | Status |
|---|----------|-------|------|---------|--------|
| DM-01 | HIGH | 3 | Model | NAG alpha parameter typo (`"nah_alpha"` → `"nag_alpha"`) — user setting always ignored | FIXED |
| LC-01 | HIGH | 1 | Latent Cache | No T=4k+1 frame count validation — cryptic reshape error on non-conforming videos | FIXED |
| LC-02 | MEDIUM | 1 | Latent Cache | I2V mask construction undocumented, misleading shape comment at line 77 | FIXED (via LC-06) |
| LC-03 | MEDIUM | 1 | Latent Cache | CLIP/WAN version confusion — no guard against `--clip` with WAN 2.2 A14B tasks | FIXED |
| LC-04 | MEDIUM | 1 | Latent Cache | Missing mask spatial dimension validation — silent failure on mismatched masks | FIXED |
| TC-01 | MEDIUM | 1 | Text Cache | No attention mask / context_len saved in WAN text cache (unlike HV) | OPEN |
| TP-01 | MEDIUM | 2 | Training | Timestep bucketing should error with dual-expert, not warn | FIXED |
| TP-02 | MEDIUM | 2 | Training | Rejection sampling for dual-expert timesteps is wasteful | TRIAGED (documented) |
| TP-03 | MEDIUM | 2 | Training | FP8 norm preservation not validated for WAN 2.2 | OPEN |
| TP-04 | MEDIUM | 2 | Training | Gradient accumulation + expert swap edge case | OPEN |
| TP-05 | MEDIUM | 2 | Training | Block swap + dual-expert interaction undertested | OPEN |
| GN-01 | MEDIUM | 4 | Generation | Hardcoded 5s sleep for block swap sync — fragile | FIXED |
| TT-01 | HIGH | 6 | Tests | No WAN dual-expert training tests (critical gap) | OPEN |
| TT-02 | HIGH | 6 | Tests | No WAN dataset loading tests (varlen T5, batch) | OPEN |
| TT-03 | MEDIUM | 6 | Tests | No WAN mask loss integration test (video layout) | OPEN |
| TC-02 | LOW | 1 | Text Cache | Empty captions silently processed without logging | FIXED |
| LC-05 | LOW | 1 | Latent Cache | Redundant dtype casts with uncertain "we are not sure" comments | OPEN |
| TP-06 | LOW | 2 | Training | LoRA + dual-expert behavior undocumented | OPEN |
| TP-07 | LOW | 2 | Training | Prior preservation + expert swap interaction undocumented | OPEN |
| TP-08 | LOW | 2 | Training | Timestep convention inconsistency (0-1 vs 1-1000) | OPEN |
| TP-09 | LOW | 2 | Training | Guidance scale tuple ordering not validated | OPEN |
| TP-10 | LOW | 2 | Training | torch.compile key patching for expert swap is fragile | OPEN |
| GN-02 | LOW | 4 | Generation | CFG skip mode silently ignored without cfg_apply_ratio | OPEN |
| GN-03 | LOW | 4 | Generation | perp_neg and cfgzerostar mutual exclusion not validated | OPEN |
| GN-04 | LOW | 4 | Generation | Lazy loading model cleanup incomplete | OPEN |
| DS-01 | LOW | 5 | Dataset | No warning when mask_directory set without use_mask_loss | FIXED |
| TC-03 | NOTE | 1 | Text Cache | T5 attention no-scaling not documented in code comments | OPEN |
| TC-04 | NOTE | 1 | Text Cache | FP8 T5 accelerator interaction undocumented | OPEN |
| TP-11 | NOTE | 2 | Training | normalize_per_sample default could auto-enable with prior preservation | OPEN |
| TP-12 | NOTE | 2 | Training | Missing EMA for LoRA training (feature request) | OPEN |
| TP-13 | MEDIUM | R | Training | WAN 2.2 training defaults are easy-misuse traps (`--timestep_sampling sigma`, `--discrete_flow_shift 1.0`) | FIXED |
| TP-14 | LOW | R | Training | Block swap + dual-expert: no runtime validation that both experts have identical block structure | OPEN |
| LC-06 | LOW | R | Latent Cache | I2V mask could be constructed directly in latent space from `lat_f` instead of pixel→latent reshape trick | FIXED |
| DOC-01 | HIGH | R | Architecture Doc | I2V pseudocode in `wan22_architecture.md` has inverted mask convention and wrong image conditioning | FIXED |
| CF-01 | NOTE | R | Design | Cache format evolution: proposed fixes (TC-01, LC-01 auto-pad) change on-disk format — need compatibility stance | OPEN |

### Review-Round Issues (added post-audit from reviewer feedback)

---

#### TP-13: WAN 2.2 Training Defaults Are Easy-Misuse Traps

**Severity**: MEDIUM
**Category**: UX / Defaults
**File**: `src/musubi_tuner/hv_train_network.py:3043, 3061`

**Problem**: The base `NetworkTrainer` argparser defaults `--timestep_sampling` to `"sigma"` and `--discrete_flow_shift` to `1.0`. WAN 2.2 T2V training requires `--timestep_sampling shift` with `--discrete_flow_shift 12.0`. A user who follows the generic docs without the WAN-specific section will silently train with the wrong timestep schedule, producing a LoRA that doesn't match the inference distribution.

**Impact**: Silently degraded training quality. Difficult to debug because the model still produces output, just worse.

**Fix**: Added WAN 2.2 task-aware warnings in `WanNetworkTrainer.handle_model_specific_args` when the user is likely on base defaults (e.g. `--timestep_sampling sigma`, `--discrete_flow_shift 1.0`) or has selected shift sampling but left the shift value at the default.

**Tests**: `tests/test_wan22_timestep_defaults_warning.py`

---

#### TP-14: Block Swap + Dual-Expert Runtime Structure Validation

**Severity**: LOW
**Category**: Gap / Defensive
**File**: `src/musubi_tuner/wan_train_network.py:612-638`

**Problem**: `swap_high_low_weights()` transfers state dict keys between experts, relying on `strict=True` to catch mismatches. But this only verifies key names match — it doesn't validate that both models have the same number of blocks, block layout, or block ordering. A future model variant with asymmetric experts would silently corrupt weights.

**Suggested Fix**: Add a one-time startup assertion that both models' block counts and key sets are identical.

---

#### LC-06: I2V Mask Could Be Constructed Directly in Latent Space

**Severity**: LOW
**Category**: Simplification
**File**: `src/musubi_tuner/wan_cache_latents.py:64-70`

**Problem**: The current I2V mask construction uses a pixel→latent reshape trick (repeat first frame 4x, view, transpose) that is mathematically correct but obscure. Since `lat_f` (latent frames) is already known from `latent.shape`, the mask could be constructed directly:

```python
lat_f = latent.shape[2]  # Already computed from VAE output
msk = torch.zeros(B, 4, lat_f, lat_h, lat_w, dtype=vae.dtype, device=vae.device)
msk[:, :, 0] = 1  # First latent frame = 1, rest = 0
```

This is equivalent, more readable, and eliminates the fragile `view(1, shape[1]//4, ...)` that depends on `T=4k+1`.

---

#### DOC-01: I2V Pseudocode Inverted Mask Convention

**Severity**: HIGH
**Category**: Documentation Correctness
**File**: `docs/wan22_architecture.md:1181-1195`

**Problem**: The architecture doc's I2V pseudocode has two errors:

1. **Mask convention is inverted**: Doc says `mask[:,:,0] = 0` (first frame = 0, rest = 1). Actual code: `msk[:, 1:] = 0` (first frame = 1, rest = 0). Both the codebase and Diffusers use first-frame-is-1.

2. **Image conditioning is wrong**: Doc says `image_latent.repeat(...) * mask` (repeat image to all frames, then mask). Actual code: pad image with zeros to full video length, then VAE-encode as a video. The zero-padding produces naturally decayed latents, not just masked copies.

**Impact**: The architecture doc serves as truth reference for the audit. Wrong pseudocode creates recurring false positives/negatives.

---

#### CF-01: Cache Format Evolution Compatibility Stance

**Severity**: NOTE
**Category**: Design / Process
**File**: N/A

**Problem**: Several proposed fixes change the on-disk cache format:
- TC-01: Would add `context_len` to text encoder cache
- LC-01 auto-pad: Would change frame counts in latent caches

There is no documented compatibility stance: Can old caches be used after code updates? Should cache files be versioned? Is re-caching always expected?

**Suggested Fix**: Document a stance in code comments or a brief section in `docs/dataset_config.md`:
- Option A: "Always re-cache after code changes" (simplest, current implicit behavior)
- Option B: Add version key to cache files, validate on load
- Option C: Read-time migration (detect old format, convert in memory)

---

### Agent False Positives (Verified Correct)

#### Round 1 False Positives (3)

1. **"I2V temporal mask uses pixel-space `num_frames`"** — The mask construction uses a reshape trick `(4 + T - 1) / 4` that implicitly converts pixel frames to latent frames. For T=81: (4 + 80) / 4 = 21 latent frames. **Correct.**

2. **"I2V image latent encoding uses wrong frame count"** — The VAE naturally compresses 81 pixel frames to 21 latent frames. The `y[:, :, :num_frames]` slice (where num_frames=81 but dim is 21) is a safe no-op in PyTorch. **Correct.**

3. **"Double normalization risk"** — VAE normalizes during `encode()`. Training code does NOT re-normalize (verified: no `latents_mean`/`latent_mean` in `wan_train_network.py`). **Correct, no double normalization.**

#### Round 2 False Positives (5)

4. **"Boundary logic inverted (both agents)"** — Both agents confused flow matching's forward-time convention with DDPM's reverse convention. In this codebase: `noisy_model_input = (1 - t) * latents + t * noise` → t=0 is clean, t=1 is noise. So `t/1000 >= 0.875` correctly identifies high-noise timesteps. Boundary check at `wan_train_network.py:564` is **correct**.

5. **"Flow matching convention reversed (base agent)"** — The code uses `(1-t)*clean + t*noise` and `target = noise - clean`. This is the "reverse" convention (clean→noise) vs academic standard (noise→clean), but is mathematically equivalent. The scheduler's `reverse=True` handles the ODE direction. **Correct, self-consistent convention.**

6. **"I2V mask shape mismatch (WanNetworkTrainer agent)"** — Agent claimed `[B, 20, F, H, W]` in cache mismatches training expectations. In reality: cache stores `[B, 20, F, H, W]` (4 mask + 16 image latent), passed as `y=image_latents` to `WanModel.forward()`. PyTorch tensor iteration over batch dim works like a list. `torch.cat([u, v], dim=0)` per element: `[16, F, H, W]` + `[20, F, H, W]` = `[36, F, H, W]`, matching `patch_embedding`'s `in_channels=36`. **Correct.**

7. **"Timestep shift formula mismatch (WanNetworkTrainer agent)"** — Agent claimed training and inference use different shift formulas. Verified: all three schedulers (FlowMatchDiscrete, DPM++, UniPC non-dynamic) use identical formula: `shift * t / (1 + (shift - 1) * t)`. **Correct, no mismatch.**

8. **"LoRA weight sharing ambiguity (base agent)"** — Agent questioned whether LoRA swaps with base model during expert switching. Verified: `LoRAModule.apply_to()` replaces `org_module.forward` but LoRA params live in separate `LoRANetwork` module, NOT in base model's `state_dict()`. Expert swap only touches base params. **Correct behavior, just undocumented.**

#### Round 4 False Positives (3)

9. **"CFG schedule indexing off-by-one"** — Agent claimed `(i + 1) in scale_per_step` creates a shift. Verified: `parse_scheduled_cfg` returns 1-indexed step numbers (user-facing: "step 1" = first step). Code consistently uses `i + 1` for both the check and the scale lookup at line 1642. **Correct, intentional 1-indexing.**

10. **"CFG skip mode crashes without cfg_apply_ratio"** — Agent claimed `TypeError` crash. Verified: line 1509 checks `cfg_apply_ratio is not None` before entering the block. Falls through to default behavior silently. **Not a crash, just a UX issue (downgraded to LOW).**

11. **"One-frame contiguity issue"** — Agent suggested `.contiguous()` should be called before slicing. Current order (slice then contiguous) is actually optimal — slicing produces a small view, then contiguous copies only the small tensor. Reversing the order would make the entire large tensor contiguous before discarding most of it. **Current code is correct and efficient.**

---
