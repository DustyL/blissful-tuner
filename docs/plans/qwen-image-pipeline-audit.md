# Qwen-Image Pipeline Audit — Consolidated Issue List

**Date**: 2026-02-17
**Last updated**: 2026-02-17 (post-review revision)
**Status**: All Phases Complete

This document consolidates all findings from the Qwen-Image training pipeline audit.
Issues are prioritized: P0 = will crash/corrupt, P1 = incorrect behavior, P2 = edge case/improvement, P3 = documentation only.

---

## Methodology

**Phase 1** (COMPLETE): Core training, caching, and inference audits via focused sub-agents
**Phase 2** (COMPLETE): LoRA module + architecture registry, dataset integration, test coverage
**Phase 3** (COMPLETE): Cross-architecture consistency, documentation updates, action plan

---

## Phase 1 Findings — Training Core

### T1. Edit Model Control Image Silent Fallback [P1]
**Location**: `src/musubi_tuner/qwen_image_train_network.py:475`
**Status**: CONFIRMED
**Issue**: When `is_edit=True` but no control images are found in batch (the while-loop at line 468-473 finds no `latents_control_{i}` keys), the code silently falls back to T2I mode without any warning. Users have no way to know their edit training has degraded — potentially wasting days of GPU time.
**Fix**: Default to `raise ValueError()` when `num_control_images == 0` and `is_edit` is True. Add `--allow_edit_fallback_to_t2i` escape hatch for users who intentionally want mixed-mode training.

### T2. Missing `require_mask_weights_if_enabled()` Validation [P2]
**Location**: `src/musubi_tuner/qwen_image_train_network.py` (call_dit method)
**Status**: CONFIRMED
**Issue**: The Qwen-Image trainer does not call `require_mask_weights_if_enabled()` to validate that mask_weights exist in the batch when `--use_mask_loss` is enabled. The base trainer at `hv_train_network.py:2407` does this. Without it, a user who enables `--use_mask_loss` but forgets to recache latents with masks will get a cryptic error downstream.
**Fix**: Add the validation call in the training loop before `call_dit`, or ensure the base class already covers it (it does at line 2407 — verify this code path is reached for Qwen-Image).

### T3. No Validation of Control Image Count for Layered [P2]
**Location**: `src/musubi_tuner/qwen_image_train_network.py:467-475`
**Status**: CONFIRMED
**Issue**: The while-loop counting control images has no upper bound and no validation that layered models require exactly 1 control image. Malformed cache with non-sequential indices (e.g., `latents_control_0`, `latents_control_2` missing `_1`) will silently stop early.
**Fix**: Add validation: `if args.is_layered and num_control_images != 1: raise ValueError(...)`.

### T4. `qwen_shift` Timestep Sampling Correctness [P2 — needs verification]
**Location**: `src/musubi_tuner/hv_train_network.py:1022-1023`
**Status**: NEEDS VERIFICATION
**Issue**: The `qwen_shift` sampling uses `(h // 2) * (w // 2)` with parameters `x1=256, y1=0.5, x2=8192, y2=0.9`. The h,w refer to unpacked latent dimensions, and the `// 2` accounts for 2x2 packing. This matches the inference formula, but should be verified against official training code if available.
**Impact**: If wrong, timestep distribution will be skewed during training.

---

## Phase 1 Findings — Caching Pipeline

### C1. Missing Control Images Causes Prompt↔Image Misalignment [P0]
**Location**: `src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py:44-47,68-71`
**Status**: CONFIRMED
**Issue**: When an Edit model item has no control content, the code logs a warning and skips the item with `continue` (line 47). This removes the item from the `images` list but NOT from `prompts` (line 36: `prompts = [item.caption for item in batch]`). At line 70, `images[i]` is indexed by batch position, causing either:
- `IndexError` if `continue` made `images` shorter than `batch`
- Silent prompt↔image↔cache misalignment if earlier items are skipped (image N gets paired with prompt N+K)
**Fix**: Remove the `continue` pattern entirely. For Edit models, `raise ValueError()` when control images are missing — skipping is never safe because it desynchronizes parallel lists. Remove the `print()` at lines 69-71 as well (see C11).

### C2. `prompt_template_encode_start_idx` May Be Wrong for Edit Mode [P2 — needs verification]
**Location**: `src/musubi_tuner/qwen_image/qwen_image_utils.py:338,397-399`
**Status**: NEEDS VERIFICATION
**Issue**: T2I uses `prompt_template_encode_start_idx = 34`, Edit uses `= 64`. These indices control how much of the system prompt prefix is stripped. The Edit system prompt is longer (264 chars vs 153 for T2I), so the drop index difference seems correct, but token positions (not character positions) should be verified against the actual tokenizer output.
**Impact**: Wrong drop index means either leaked system tokens or truncated user content in embeddings.
**Fix**: Consider computing the drop index dynamically: tokenize the template prefix once and derive the index from token positions rather than hardcoding 34/64. This survives tokenizer/template drift and makes testing straightforward.

### C3. Alpha Mask Priority Unclear in Logs [P2]
**Location**: `src/musubi_tuner/dataset/image_video_dataset.py:2499-2518`
**Status**: CONFIRMED
**Issue**: The fallback chain (RGBA alpha > mask_directory > full-weight) is correctly implemented but only documented in code comments. No summary log is emitted during caching to show how many items used each mask source.
**Fix**: Add a summary log after caching showing mask source distribution.

### C4. Empty Mask Not Validated During Caching [P2]
**Location**: `src/musubi_tuner/qwen_image_cache_latents.py:146`
**Status**: CONFIRMED (related to V1)
**Issue**: If `item.mask_content` is a valid numpy array but all zeros (fully black mask), it passes through caching without warning. During training, `mask_loss.py:368` returns zero loss silently (the V1 finding). Catching this at cache time would surface the problem earlier.
**Fix**: Add validation during caching: `if mask_content.sum() == 0: logger.warning(...)`.

### C5. Mask Resolution Mismatch Not Detected [P2]
**Location**: `src/musubi_tuner/qwen_image_cache_latents.py:154-155`
**Status**: CONFIRMED
**Issue**: If a mask has a drastically different aspect ratio from the target image (e.g., user provides wrong mask file), the `F.interpolate` call succeeds silently but produces spatially misaligned weights.
**Fix**: Add warning if mask aspect ratio differs from target by >10%.

### C6. Multi-Control-Image Alignment [P2 — needs verification]
**Location**: `src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py:52-59`
**Status**: NEEDS VERIFICATION
**Issue**: For Edit-2509/2511 with multiple control images, the text encoder processes all images (creating `Picture 1: <image>`, `Picture 2: <image>` prefixes), but latent caching encodes each control separately. During training, these need to be correctly matched. Verify alignment.

### C7. Control Image Resolution Downsampled to 384x384 [P3 — doc gap]
**Location**: `src/musubi_tuner/qwen_image/qwen_image_utils.py:903`
**Status**: CONFIRMED
**Issue**: Control images are resized to `CONDITION_IMAGE_RESOLUTION` (384x384) before being passed to the VL processor for text encoding. This is correct behavior but undocumented in user-facing docs.
**Fix**: Document in `qwen_image.md`.

### C8. Cache Key Format F-Dimension Semantics [P3 — doc gap]
**Location**: `src/musubi_tuner/dataset/image_video_dataset.py:597`
**Status**: CONFIRMED
**Issue**: `mask_weights_{F}x{H}x{W}_float32` uses `F` which means "frames" for video and "layers" for layered models. Code comments should clarify this dual meaning.

### C9. No Per-Layer Masks for Layered Models [P3 — doc gap]
**Location**: `src/musubi_tuner/qwen_image_cache_latents.py:158`
**Status**: CONFIRMED
**Issue**: Single mask is expanded across all layers via `.expand(-1, lat_l, -1, -1)`. Users cannot specify different masks per layer. Document this as a known limitation.

### C10. VAE Scale Factor Hardcoded [P3]
**Location**: `src/musubi_tuner/qwen_image/qwen_image_utils.py:30`
**Status**: CONFIRMED
**Issue**: `VAE_SCALE_FACTOR = 8` is hardcoded. If a future Qwen variant uses a different VAE compression ratio, this will silently produce wrong dimensions.

### C11. Debug `print()` Statements in Cache Script [P2]
**Location**: `src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py:69-71`
**Status**: CONFIRMED
**Issue**: Raw `print()` statements output per-item debug info during caching. On large datasets this pollutes logs and slows execution.
**Fix**: Replace with `logger.debug()` or gate behind `--debug`.

---

## Phase 1 Findings — Inference Pipeline

### I1. CFG Implementation Differs from Official Edit-2509/2511 [P1]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:866-910`
**Status**: CONFIRMED
**Issue**: The official Edit-2509/2511 pipeline uses a dual-CFG system: `guidance_scale=1.0` (disabling standard CFG) + `true_cfg_scale=4.0` (enabling true CFG). The current implementation uses only `--guidance_scale` as a single parameter for all models. While the generation result may still work (it's effectively standard CFG), the behavior differs from the official pipeline.
**Fix**: Keep `--guidance_scale` for backwards compatibility. Add `--true_cfg_scale`. Implement model-version-specific defaults: Edit-2509/2511 → `guidance_scale=1.0, true_cfg_scale=4.0`; T2I → `true_cfg_scale=4.0`. Make explicit in `--help` which parameter is used when.

### I2. `zero_cond_t` Semantics Misunderstood in Docs [P1 — corrected]
**Location**: `src/musubi_tuner/qwen_image/qwen_image_model.py:1332-1348,982-1008`
**Status**: HYPOTHESIS CORRECTED — behavior verified against code
**Issue**: Previous hypothesis said `zero_cond_t` zeros the timestep for the CFG unconditional pass. This is **wrong**. The actual implementation:
1. `timestep = torch.cat([timestep, timestep * 0], dim=0)` — doubles the timestep batch (real + zero) at line 1336
2. `temb = self.time_text_embed(timestep, ...)` — produces two sets of modulation parameters
3. `timestep_zero_index = base_len` — the packed sequence length of the base/noise tokens (line 1348)
4. In `_modulate()` at lines 987-1008: `x[:, :timestep_zero_index]` gets real-timestep modulation, `x[:, timestep_zero_index:]` gets zero-timestep modulation
So `zero_cond_t` applies zero-timestep conditioning specifically to the **control/reference token segment** within a single forward pass, differentiating the noise path from the appearance path. This happens identically in both conditional and unconditional CFG passes.
**Fix**: Update `docs/qwen_image_architecture.md` to accurately describe this as "intra-sequence conditional timestep split" rather than "CFG-related unconditional zeroing". Verify whether this behavior matches the official Qwen-Image Edit-2511 implementation.

### I3. CFG Normalization Always-On [P2]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:908-910`
**Status**: CONFIRMED (deliberate per line 95-96 comment)
**Issue**: CFG normalization is always applied: `noise_pred = comb_pred * (cond_norm / noise_norm)`. This is a deliberate simplification (line 95: "CFG normalization is always enabled for Musubi Tuner"), but the official Layered pipeline defaults to `cfg_normalize=False`.
**Fix**: Add `--cfg_normalize` toggle, defaulting False for Layered.

### I4. Negative Prompt Default `None` Crashes All Models [P0]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:87,645-652` → `qwen_image_utils.py:345,349`
**Status**: CONFIRMED — verified crash path
**Issue**: `--negative_prompt` defaults to `None` (line 87). The negative prompt is **always** encoded unconditionally at lines 645-652 — there is no CFG guard. `get_embeds(None, images)` calls `get_qwen_prompt_embeds(tokenizer, text_encoder, None)`, where line 345 leaves `None` as-is (`isinstance(None, str)` is False), then line 349 iterates: `[template.format(e) for e in None]` → `TypeError: 'NoneType' object is not iterable`. This crashes for **every model version** (T2I, Edit, Layered), not just Edit.
**Fix**: Normalize at argparse level: `default=""` (empty string). Or normalize immediately after parsing: `if args.negative_prompt is None: args.negative_prompt = " " if args.is_edit else ""`.

### I5. Default Inference Steps Too Low [P2]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:118`
**Status**: CONFIRMED
**Issue**: Default `infer_steps=25`. Official defaults are 50 (T2I) and 40 (Edit-2509/2511).
**Fix**: Set model-version-specific defaults after `resolve_model_version_args()`. Be careful not to break users relying on 25 steps for speed — consider only changing when the user hasn't explicitly set `--infer_steps`.

### I6. `automatic_prompt_lang_for_layered` Crashes Without Control Image [P2]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:624-628`
**Status**: CONFIRMED
**Issue**: If `--automatic_prompt_lang_for_layered en` is set but no `--control_image_path` provided, `get_image_caption()` receives `None` for the images parameter and crashes.
**Fix**: Add validation: `if images is None: raise ValueError(...)`.

### I7. Missing Edit Control Image Assertion Clarity [P2]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:691`
**Status**: CONFIRMED
**Issue**: The assertion `assert args.is_edit and args.control_image_path is not None or not args.is_edit` is valid but confusing. Simplify to `if args.is_edit: assert args.control_image_path is not None`.

### I8. `calculate_shift()` Dead Code Has Wrong Defaults [P2]
**Location**: `src/musubi_tuner/qwen_image/qwen_image_model.py:57-67`
**Status**: CONFIRMED
**Issue**: The `calculate_shift()` function in model.py has defaults `max_seq_len=4096, max_shift=1.15` which don't match official config (8192, 0.9). The actual inference uses `calculate_shift_qwen_image()` from `qwen_image_utils.py` which passes correct constants. This is dead code but a latent footgun.
**Fix**: Update defaults or remove the function if unused.

### I9. RCM Absolute Threshold Range Misleading [P3]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:175-176`
**Status**: CONFIRMED
**Issue**: Help text says "0.01 to 0.1 for absolute threshold" but latent values have magnitude ~3.0, so these values would mask almost everything.
**Fix**: Update help text with more appropriate range.

### I10. Batch Decode Assumes batch_size=1 [P3]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:524-553`
**Status**: CONFIRMED
**Issue**: `decode_latent()` has comments and dimension handling that only work for batch_size=1.
**Fix**: Add `assert latent.shape[0] == 1` for clarity.

### I11. No `guidance_embeds` Support [P3 — future-proofing]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:811`
**Status**: CONFIRMED
**Issue**: The generation script hardcodes `guidance = None`. Current Qwen-Image models have `guidance_embeds=false` in their config, so this is correct today. However, if future variants enable guidance embeddings, the script will silently ignore them.
**Fix**: Add conditional support: read `guidance_embeds` from model config and construct the guidance tensor when enabled.

### I12. No Automatic Prompt Enhancement [P3 — feature gap]
**Location**: Generation pipeline (no specific line)
**Status**: CONFIRMED
**Issue**: The official Qwen-Image pipeline includes a sophisticated prompt enhancement system (language detection, category-specific rewriting via Qwen-Plus API for T2I, Qwen-VL-Max for editing). The blissful-tuner generation script uses prompts as-is, which may produce lower-quality results compared to official demos.
**Fix**: Document this limitation in `qwen_image.md`. Recommend users manually enhance prompts or use external prompt rewriting tools.

### I13. CFG Normalization Missing Epsilon Guard [P2]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:910`
**Status**: CONFIRMED
**Issue**: `noise_pred = comb_pred * (cond_norm / noise_norm)` divides by `noise_norm` without epsilon. If `noise_norm` contains any zero elements (degenerate input or early timesteps), this produces NaN that propagates through the rest of the denoising loop. The same pattern appears in training sampling.
**Fix**: `noise_pred = comb_pred * (cond_norm / (noise_norm + 1e-8))`. Apply to both inference and training paths.

### I14. Embed Cache Key Missing Behavior-Affecting Flags [P1]
**Location**: `src/musubi_tuner/qwen_image_generate_image.py:632,646`
**Status**: CONFIRMED
**Issue**: `conds_cache` keys include `(prompt, control_image_paths, (width, height))` but not `model_version` or `resize_control_to_image_size`. In interactive/from-file usage where these can vary between prompts, the cache can reuse embeddings computed with the wrong model version template or control image preprocessing, producing silently incorrect results.
**Fix**: Include `model_version` and `resize_control_to_image_size` in the cache key tuple.

---

## Phase 1 Findings — Training Core (Addendum)

### T5. `mask_min_weight` vs Prior Preservation Warning Not Prominent [P3]
**Location**: `src/musubi_tuner/modules/mask_loss.py:117-122`
**Status**: CONFIRMED
**Issue**: When both `--prior_preservation_weight > 0` and `--mask_min_weight > 0` are set, the code logs a warning that the floor dilutes prior preservation effectiveness. However, this warning is only emitted during validation and may be missed by users scrolling through startup logs.
**Impact**: Users training with both flags may get suboptimal results without understanding why — the non-zero floor forces the model to learn the target at reduced strength in masked-out regions, conflicting with the prior loss.
**Fix**: Make the warning more prominent (e.g., include in the mask loss configuration banner) or consider forcing `mask_min_weight=0.0` when prior preservation is active.

---

## Validated Findings (from previous session)

### V1. All-Zero Mask Warning Missing [P2]
**Location**: `src/musubi_tuner/modules/mask_loss.py:368`
**Status**: CONFIRMED — also see C4 for caching-time detection
**Issue**: If a user provides an all-black mask, `target_weight_sum < 1e-8` causes zero loss silently. No warning logged.
**Fix**: Log a warning when this occurs so users know their training signal is effectively zero.

### V2. Prior Preservation Not Implemented for Layered Layout [P2 — doc gap]
**Location**: `src/musubi_tuner/modules/mask_loss.py:307`
**Status**: CONFIRMED — explicit `NotImplementedError`
**Issue**: Code correctly raises `NotImplementedError`, but this limitation is not documented in `qwen_image.md` or `MASKED_LOSS_TRAINING_GUIDE.md`.
**Fix**: Document this limitation.

### V3. Gamma/Min-Weight Pipeline Clarification [P3 — doc gap]
**Location**: `mask_loss.py:326-338` (training) vs caching scripts
**Status**: CONFIRMED
**Issue**: Raw [0.0, 1.0] masks are cached; gamma correction and min-weight are applied at training time. This is correct (allows changing params without re-caching) but underdocumented.
**Fix**: Add to `qwen_image.md` mask loss section and `MASKED_LOSS_TRAINING_GUIDE.md`.

### V6. Alpha Mask Conflict with Layered Model [P2 — doc gap]
**Location**: Documentation only
**Status**: CONFIRMED
**Issue**: Layered model requires RGBA images where alpha = layer transparency. If `alpha_mask=true`, the alpha channel is hijacked for loss weighting, conflicting with its semantic purpose.
**Fix**: Document that separate `mask_directory` should be used for layered models.

---

## Invalidated Findings

| Finding | Agent | Reason |
|---------|-------|--------|
| Training P0: Mask dimension mismatch with `remove_first_image_from_target` | Training | `drop_base_frame=True` in `mask_loss._prepare_tensors()` correctly slices the mask before dimension check |
| Inference P0-2: Layered shift formula uses `control_latent.shape[1]` | Inference | Correct — `control_latent.shape[1]` gives per-image packed size (e.g., 4096 for 1024x1024), which is the right basis for shift. Using `latents.shape[1]` would include all layers. |
| Inference P0-3: `img_shapes` nested list structure wrong | Inference | Intentional design — `QwenEmbedRope.forward()` at model.py:321 does `video_fhw = video_fhw[0]` to unwrap outer list |
| Inference P2-12: `inpainting_shape` typo | Inference | Not found in codebase via grep — agent hallucinated or was fixed |
| Previous V4: Layer count batch crashes | Previous | Dataset validates `num_targets` consistency at `image_video_dataset.py:1442-1452` |
| Caching P0-1: Mask shape mismatch for layered | Caching | Agent self-corrected — semantic confusion only, not a crash |
| Caching P0-2: Control latent shape assertion | Caching | Agent self-corrected — working as intended |
| LoRA: Qwen-Image 2512 variant missing from registry | LoRA | No such variant in codebase — 3 entries correctly cover all variants |
| LoRA: LoHa/LoKr architecture kwarg not passed | LoRA | Trainer passes `architecture=self.architecture`; LoHa/LoKr extract via `kwargs.get()` |
| LoRA: LoKr fails on Conv2d in Qwen-Image | LoRA | `QwenImageTransformerBlock` contains only `Linear` modules |
| LoRA: Weight save/load dtype handling | LoRA | Delegates to `lora.py` which handles dtype properly |
| LoRA: Inference-time LoRA loading fails | LoRA | Both paths (direct merge + LyCORIS) work correctly |

---

## Phase 2 Findings — LoRA / Architecture / Tests

### L1. `exclude_mod` Regex Never Matches Qwen-Image Modulation Layers [P0]
**Location**: `src/musubi_tuner/networks/lora_qwen_image.py:47`, `src/musubi_tuner/networks/network_arch.py:78,83,88`
**Status**: CONFIRMED
**Issue**: The regex `r".*(_mod_).*"` requires `_mod_` (underscore both sides), but actual model paths use `img_mod.` and `txt_mod.` (dot after "mod", e.g. `transformer_blocks.0.img_mod.1`). The regex `fullmatch()` never matches, so the default `exclude_mod=True` is completely broken. Every Qwen-Image LoRA trained with this codebase has included modulation layers regardless of setting.
**Fix**: Change regex to `r".*\.(img_mod|txt_mod)\..*"` in all three locations. Ensure the pattern matches the linear submodule index (e.g., `img_mod.1.weight`). Also update `docs/qwen_image.md` which currently claims modulation layers are excluded by default — that claim is false today.

### L2. `merge_lora.py` Cannot Merge Qwen-Image LoRAs [P0]
**Location**: `src/musubi_tuner/merge_lora.py:13,43,65`
**Status**: CONFIRMED
**Issue**: `merge_lora.py` imports `load_transformer` from `hunyuan_model` and uses `lora.create_arch_network_from_weights()` (HunyuanVideo-specific, targeting `MMDoubleStreamBlock`/`MMSingleStreamBlock`). When used with a Qwen-Image LoRA, no modules match and the merge silently produces an unchanged model.
**Fix**: Add `--architecture` argument or auto-detect from LoRA metadata (`ss_network_module`), then dispatch to correct model loader and LoRA module. Add a defensive check: if zero modules matched or zero weights were applied, log a prominent error instead of silently saving an unchanged model.

### L3. `convert_lora.py` Doesn't Detect `lora_qwen_image` in Metadata [P1]
**Location**: `src/musubi_tuner/convert_lora.py:136-163`
**Status**: CONFIRMED
**Issue**: `_resolve_arch_from_metadata` checks for `lora_flux_2`, `lora_flux`, `networks.lora` but not `networks.lora_qwen_image`. The lookup table catches current keys, but future extensions would fall through to incorrect HV/FLUX normalization.
**Fix**: Add `if "lora_qwen_image" in network_module: return "qwen_image"`.

### L4. `convert_lora.py` `--arch` CLI Missing `qwen_image` Choice [P1]
**Location**: `src/musubi_tuner/convert_lora.py:307`
**Status**: CONFIRMED
**Issue**: `choices=["auto", "hv", "flux", "flux2", "flux_kontext"]` omits `qwen_image`. No manual override possible.
**Fix**: Add `"qwen_image"` to choices.

### L5. `lora_qwen_image.py` Uses Standard Logger [P2]
**Location**: `src/musubi_tuner/networks/lora_qwen_image.py:10`
**Status**: CONFIRMED
**Issue**: Uses `logging.getLogger(__name__)` instead of `BlissfulLogger(__name__, "green")`. Inconsistent log formatting.

### Test Coverage Summary [P1 — systemic gap]
**Status**: CONFIRMED
**Issue**: Qwen-Image has the weakest test coverage of any supported architecture despite being the most complex (3 model variants, unique CFG normalization, layered RGBA). Only 2 of 15 test files have direct Qwen-Image coverage (arch registry + unknown format rejection). Zero tests for:
- `pack_latents`/`unpack_latents` roundtrip
- `calculate_shift_qwen_image` vs `calculate_shift` divergent defaults
- `resolve_model_version_args` validation
- `call_dit` img_shapes construction (3 modes)
- `remove_first_image_from_target` integration
- Edit mode fallback (no control images)
- CFG normalization formula

**Recommended new test files**: `tests/test_qwen_image_utils.py` (pure functions), `tests/test_qwen_image_training.py` (mock-based integration). See full agent report for detailed test signatures.

---

## Summary Statistics

| Priority | Count | Needs Verification |
|----------|-------|--------------------|
| P0 | 4 | 0 |
| P1 | 6 | 0 |
| P2 | 16 | 3 |
| P3 | 10 | 0 |
| Invalidated | 12 | — |
| **Total Active** | **36** | **3** |

### Priority Breakdown

**P0 (Will crash/corrupt) — 4 confirmed:**
- I4: `negative_prompt=None` crashes ALL model versions (inference)
- C1: `continue` pattern misaligns prompts↔images↔caches (caching)
- L1: `exclude_mod` regex broken — modulation layers always trained (LoRA)
- L2: `merge_lora.py` silently fails for Qwen-Image (LoRA merge)

**P1 (Incorrect behavior) — 6 confirmed:**
- T1: Edit model silent fallback — error-by-default recommended (training)
- I1: CFG differs from official Edit-2509/2511 (inference)
- I2: `zero_cond_t` docs describe wrong semantics — corrected to intra-sequence split (docs/inference)
- I14: Embed cache key missing `model_version` and `resize_control_to_image_size` (inference)
- L3: `convert_lora.py` missing Qwen-Image detection (LoRA convert)
- L4: `convert_lora.py` missing `qwen_image` CLI choice (LoRA convert)

**P2 (Edge case/improvement) — 16 confirmed + 3 needs verification:**
- T2, T3, T4, C2, C3, C4, C5, C6, C11, I3, I5, I6, I7, I8, I13, V1, V2, V6, L5

**P3 (Documentation/minor) — 10 confirmed:**
- C7, C8, C9, C10, I9, I10, I11, I12, T5, V3

---

## Phase 3: Action Plan

### Sprint 1 — Critical Crash Fixes (P0)
**Goal**: Fix all bugs that crash or silently corrupt outputs.
**Definition of Done**: Each fix has at least 1 regression test, manual repro confirms crash is gone.

1. **I4: Fix negative_prompt=None crash** (1 file, ~3 lines)
   - `qwen_image_generate_image.py:87`: Change `default=None` to `default=""`, or normalize after parsing
   - For Edit models, default to `" "` (single space) per official pipeline
   - **Regression test**: "qwen_image_generate_image defaults do not crash" (mock-based, verify `get_embeds` receives valid string)

2. **C1: Fix prompt↔image misalignment in cache script** (1 file, ~10 lines)
   - `qwen_image_cache_text_encoder_outputs.py:44-47`: Replace `continue` with `raise ValueError()` for Edit models
   - Remove raw `print()` at lines 69-71 → `logger.debug()` (fixes C11 too)
   - **Regression test**: "Edit caching raises on missing control images"

3. **L1: Fix `exclude_mod` regex** (3 files, 1-line each)
   - `networks/lora_qwen_image.py:47`: `r".*(_mod_).*"` → `r".*\.(img_mod|txt_mod)\..*"`
   - `networks/network_arch.py:78,83,88`: same change in all 3 Qwen-Image entries
   - Update `docs/qwen_image.md` — current claim that modulation layers are excluded is false
   - **Regression test**: Verify regex matches actual model paths like `transformer_blocks.0.img_mod.1.weight`

4. **L2: Make `merge_lora.py` architecture-aware**
   - Add `--architecture` arg (auto-detect from LoRA metadata `ss_network_module`)
   - Dispatch to correct model loader + LoRA module for each architecture
   - Add defensive check: if zero modules matched / zero weights applied → error, don't save
   - Scope: larger change — may warrant its own PR

### Sprint 2 — Behavioral Fixes (P1)
**Goal**: Fix incorrect behaviors users will encounter.
**Definition of Done**: Each fix verified with test, `--help` updated where relevant.

5. **T1: Edit model training error-by-default**
   - `qwen_image_train_network.py:475`: `raise ValueError()` when `num_control_images == 0` and `is_edit`
   - Add `--allow_edit_fallback_to_t2i` flag as explicit escape hatch
   - **Regression test**: "Edit training cannot proceed without control latents unless explicitly allowed"

6. **I1: Dual-CFG for Edit-2509/2511**
   - Keep `--guidance_scale` for backwards compatibility
   - Add `--true_cfg_scale` parameter
   - Model-version-specific defaults: Edit-2509/2511 → `guidance_scale=1.0, true_cfg_scale=4.0`; T2I → `true_cfg_scale=4.0`
   - Explicit in `--help` which parameter is used when

7. **I2: Correct `zero_cond_t` documentation**
   - Update `docs/qwen_image_architecture.md` to describe intra-sequence timestep split (NOT CFG unconditional zeroing)
   - Verify behavior matches official Qwen-Image Edit-2511 pipeline

8. **I14: Fix embed cache key**
   - Add `model_version` and `resize_control_to_image_size` to cache key tuple at lines 632, 646

9. **L3 + L4: Add Qwen-Image to `convert_lora.py`**
   - Add `"qwen_image"` to `--arch` choices
   - Add `lora_qwen_image` detection in `_resolve_arch_from_metadata`

### Sprint 3 — Test Coverage (co-located with Sprint 1-2 regression tests)
**Goal**: Establish baseline test coverage for Qwen-Image.
**Definition of Done**: All new tests pass, coverage report shows Qwen-Image pure functions covered.

10. Create `tests/test_qwen_image_utils.py` — pure function tests:
    - `pack_latents`/`unpack_latents` roundtrip
    - `calculate_shift_qwen_image` vs `calculate_shift` defaults
    - `resolve_model_version_args` validation

11. Create `tests/test_qwen_image_training.py` — mock-based integration:
    - `call_dit` img_shapes construction (3 modes)
    - `remove_first_image_from_target` integration
    - Edit mode fallback behavior (regression for T1)

### Sprint 4 — Edge Cases + Polish (P2)
**Goal**: Improve robustness and defaults.

12. Inference hardening:
    - I5: Model-version-specific default steps (50/40) — only apply when user hasn't explicitly set `--infer_steps`
    - I3: Add `--cfg_normalize` toggle, default False for Layered
    - I13: Add epsilon to CFG normalization: `cond_norm / (noise_norm + 1e-8)` in both inference and training
    - I6: Validate `--automatic_prompt_lang_for_layered` requires `--control_image_path`
    - I7: Simplify Edit assertion

13. Caching hardening:
    - C4: Empty mask warning at cache time
    - C5: Mask aspect ratio mismatch warning
    - C3: Mask source distribution summary log
    - C2: Consider computing `prompt_template_encode_start_idx` dynamically from tokenizer

14. Training validation:
    - T2: Add `require_mask_weights_if_enabled()` call or verify base class path
    - T3: Validate control image count for Layered (exactly 1)

15. Dead code: update `calculate_shift()` defaults in model.py or remove (I8)

### Sprint 5 — Documentation (P3)
**Goal**: Fill documentation gaps.

16. Document in `qwen_image.md`: mask loss for Qwen-Image, alpha mask vs Layered conflict, control image 384px downsample, prior preservation limitation for Layered
17. Document in `MASKED_LOSS_TRAINING_GUIDE.md`: gamma/min-weight pipeline, mask_min_weight vs prior interaction
18. Code comments: cache key F-dimension semantics, per-layer mask limitation, shift formula differences
19. Correct `qwen_image_architecture.md` zero_cond_t description (see I2)

---

## Audit Status: COMPLETE
**Date completed**: 2026-02-17
**Last revision**: 2026-02-17 (priority reclassifications, 4 new findings, corrected I2 hypothesis, strengthened fix recommendations)
**Phases**: All 3 phases complete
**Total findings**: 36 active (4 P0, 6 P1, 16 P2, 10 P3) + 12 invalidated
**Agent reports archived**: Training (a0374ce), Caching (a3cd502), Inference (a776980), LoRA (a942e20), Tests (a5ea998)

### Revision Log
- **v1** (2026-02-17): Initial audit — 32 findings
- **v2** (2026-02-17): Post-review revision:
  - Upgraded I4 (negative_prompt=None) from P2 → P0: crashes all model versions, not just Edit
  - Upgraded C1 (missing control images) from P1 → P0: `continue` pattern causes prompt↔image misalignment or IndexError
  - Added I13 (CFG-norm epsilon guard, P2)
  - Added I14 (embed cache key missing flags, P1)
  - Added C11 (debug print statements, P2)
  - Corrected I2 hypothesis: `zero_cond_t` is intra-sequence timestep split for control tokens, NOT CFG unconditional zeroing
  - Strengthened T1 fix: error-by-default with `--allow_edit_fallback_to_t2i` escape hatch
  - Strengthened L2 fix: zero-match defensive check
  - Strengthened C2 fix: consider computing drop index dynamically from tokenizer
  - Added Definition of Done per sprint, co-located regression tests with fixes
  - Fixed status banner contradiction (header vs footer)
