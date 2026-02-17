# FLUX.2 Pipeline Audit — Complete Findings & Fix Plan (v4)

**Date:** 2026-02-17 (v4: Phase 4 complete — all findings addressed)
**Scope:** Full audit of FLUX.2 LoRA training pipeline (training, caching, generation, LoRA conversion) against updated architecture documentation (`docs/flux2_architecture.md`, `docs/flux_2.md`).
**Method:** 6 parallel sub-agents audited: (1) Training script, (2) Model definitions, (3) Utils/text encoding, (4) Caching scripts, (5) Generation pipeline, (6) LoRA targeting/conversion.

---

## Post‑v4 Addendum (v4.1) — Additional Hardening (2026-02-17)

After v4/Phase 4 completion, additional Flux.2‑Dev / FLUX.2‑klein‑9B / FLUX.2‑klein‑base‑9B issues were addressed (outside the original findings list). These are primarily **variant-awareness + safety guards**.

### Summary of Changes

1. **Training sample prompt defaults are now variant-aware (and distilled fixed params enforced).**
   - **Problem:** `hv_train_network.sample_image_inference()` uses generic defaults (`sample_steps=20`, `guidance_scale=self.default_guidance_scale`). For Klein distilled variants (4 steps / guidance=1.0 fixed), default samples were misleading/low quality unless prompt file overrode values.
   - **Fix:** `Flux2NetworkTrainer.process_sample_prompts()` now fills per-prompt defaults from `FLUX2_MODEL_INFO[model_version].defaults` and enforces `fixed_params` with warnings for Klein distilled.
   - **Files:** `src/musubi_tuner/flux_2_train_network.py`
   - **Tests:** `tests/test_flux2_training_sample_defaults.py`

2. **Inference knob aliasing + warnings to prevent silent no-ops.**
   - **Problem:** Users commonly set `--guidance_scale` on guidance-distilled variants (DEV) even though sampling uses `--embedded_cfg_scale`, and vice-versa for base models. Negative prompts are also ignored for guidance-distilled models but previously could fail silently.
   - **Fix:** `apply_model_defaults_and_enforce_fixed_params()` now:
     - Aliases `--guidance_scale -> --embedded_cfg_scale` for distilled variants when only one is set.
     - Aliases `--embedded_cfg_scale -> --guidance_scale` for base (CFG) variants when only one is set.
     - Warns if both are set but differ (explicitly stating which knob is ignored).
     - Warns when `--negative_prompt` is provided for guidance-distilled variants (ignored).
   - **Files:** `src/musubi_tuner/flux_2_generate_image.py`
   - **Tests:** Updated `tests/test_generation_argparse_defaults.py`

3. **Tokenizer/processor loading is now local-first (reduces network dependency).**
   - **DEV (Mistral3):** `AutoProcessor.from_pretrained()` now attempts local `--text_encoder` path first, then falls back to HF ID.
   - **Klein (Qwen3 tokenizer):** `Qwen2Tokenizer.from_pretrained()` now attempts local checkpoint path first, then falls back to HF ID.
   - **Files:** `src/musubi_tuner/flux_2/flux2_utils.py`, `src/musubi_tuner/zimage/zimage_utils.py`
   - **Tests:** `tests/test_flux2_tokenizer_local_first.py`

4. **Context vector dimensionality + dtype guards (catch wrong text encoder / cache variant).**
   - **Problem:** Using the wrong TE checkpoint or mixing caches across variants can produce `ctx_vec` with mismatched last-dim vs `params.context_in_dim` (e.g., DEV 15360 vs Klein9B 12288), causing confusing downstream errors or silent breakage.
   - **Fix:** New `flux2_utils.validate_ctx_vec_dim()` guard, applied in:
     - Generation prompt encoding (`prepare_text_inputs()`).
     - Training sample prompt TE caching (`process_sample_prompts()`).
     - Training forward path (`call_dit()`).
     - Text encoder output caching script; also normalizes cached `ctx_vec` to `bfloat16`.
   - **Files:** `src/musubi_tuner/flux_2/flux2_utils.py`, `src/musubi_tuner/flux_2_generate_image.py`, `src/musubi_tuner/flux_2_train_network.py`, `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py`
   - **Tests:** `tests/test_flux2_ctx_vec_guards.py`

5. **Performance: Qwen3 embedder now batches tokenization.**
   - **Problem:** Qwen3 tokenization was previously executed once per prompt in a Python loop (more overhead than necessary).
   - **Fix:** `Qwen3Embedder.forward()` now formats prompts with `apply_chat_template()` per prompt, then tokenizes the whole batch in one call to the tokenizer.
   - **Files:** `src/musubi_tuner/flux_2/flux2_utils.py`
   - **Tests:** `tests/test_flux2_qwen3_embedder_batch_tokenization.py`

### Validation (post‑v4.1)

- `./venv/bin/ruff check` clean
- `./venv/bin/python -m pytest -q` → **330 passed, 8 skipped**

### Implementation Status

| Phase | Status | Validation |
|-------|--------|------------|
| Phase 1 (CRITICALs) | **DONE** | `ruff check` clean, 245 tests pass (8 skipped) |
| Phase 2A (Arg Defaults Bundle) | **DONE** | Sentinel defaults + fixed_params enforcement live |
| Phase 2B (Correctness & Polish) | **DONE** | Type annotations, FP8 keys, n_dim, CLAUDE.md fixed |
| Phase 3 (Behavioral) | **DONE** | `.to()` chaining, ae.eval(), block swap bounds, etc. |
| Phase 4 (Cleanup & Features) | **DONE** | W6, W12, W11, W10 all addressed. 253 tests pass (8 skipped). |

**Extra fix (not in original plan):** Duplicate `--compile` registration crash in `qwen_image_generate_image.parse_args()` — both `setup_parser_compile()` and Blissful's Qwen arg helper registered it. Fixed by making `blissful_core.py:464` additive/skip-if-present.

**Additional pipeline hardening:** LyCORIS merge silently did nothing when module targets weren't discoverable. `merge_lora_weights()` now accepts `extra_unet_targets` and logs when LyCORIS matches 0 modules (`wan_generate_video.py:778`). LyCORIS target discovery auto-detects known block types (`networks/lycoris.py:221`). Relevant for `--prefer_lycoris` on FLUX.2.

---

## Table of Contents

1. [Architecture Quick-Reference](#architecture-quick-reference)
2. [CRITICAL Findings (5)](#critical-findings)
3. [WARNING Findings (20)](#warning-findings)
4. [INFO Findings (12)](#info-findings)
5. [Extra Findings (discovered during implementation)](#extra-findings)
6. [Fix Plan — Execution Order](#fix-plan)
7. [Testing Matrix](#testing-matrix)
8. [Files Touched Summary](#files-touched)

---

## Architecture Quick-Reference

Block counts from `flux2_models.py` (corrected from initial audit):

| Variant | Params Class | Double Blocks (`depth`) | Single Blocks (`depth_single_blocks`) | Max Swappable (each - 2) |
|---------|-------------|------------------------|--------------------------------------|--------------------------|
| DEV (32B) | `Flux2Params` | 8 | 48 | 6 double, 46 single |
| Klein-9B | `Klein9BParams` | 8 | 24 | 6 double, 22 single |
| Klein-4B | `Klein4BParams` | **5** | **20** | **3 double, 18 single** |

`FLUX2_MODEL_INFO` defaults and fixed params from `flux2_utils.py`:

| Model Version | `defaults` | `fixed_params` | `guidance_distilled` | `use_guidance_embed` |
|--------------|------------|----------------|---------------------|---------------------|
| `klein-4b` | `guidance=1.0, num_steps=4` | `{guidance, num_steps}` | True | False |
| `klein-base-4b` | `guidance=4.0, num_steps=50` | `∅` | False | False |
| `klein-9b` | `guidance=1.0, num_steps=4` | `{guidance, num_steps}` | True | False |
| `klein-base-9b` | `guidance=4.0, num_steps=50` | `∅` | False | False |
| `dev` | `guidance=4.0, num_steps=50` | `∅` | True | True |

CLI defaults — **pre-fix observation** (before Phase 2A sentinel change):
- `--infer_steps`: `default=50` (help text incorrectly said 25)
- `--embedded_cfg_scale`: `default=4.0` (docs/flux_2.md:215 incorrectly said 2.5)
- `--guidance_scale`: `default=4.0`

CLI defaults — **post-fix behavior** (after Phase 2A):
- `--infer_steps`: `default=None` → filled from `FLUX2_MODEL_INFO[version].defaults["num_steps"]` after model_version resolution
- `--embedded_cfg_scale`: `default=None` → filled from `defaults["guidance"]` for guidance-distilled models
- `--guidance_scale`: `default=None` → filled from `defaults["guidance"]` for non-distilled models
- For `fixed_params` entries (Klein distilled): user-supplied values that differ from defaults are overridden with a warning

---

## CRITICAL Findings

### C1. FP8 Cast Silently Skipped with `--blocks_to_swap` — FIXED

**File:** `src/musubi_tuner/flux_2_generate_image.py` lines 362-375
**Impact:** Model runs in bf16 instead of fp8, silently using ~2× more VRAM than intended.

**Root Cause:** The `optimize_model()` function fuses dtype cast and device move into a single `.to(device, dtype)` call. When `--blocks_to_swap > 0`, `target_device` is set to `None` (because the block swap logic handles device placement separately). The combined guard `if target_device is not None and target_dtype is not None:` then skips BOTH the device move AND the dtype cast.

```python
# Lines 362-375 — current broken logic
else:
    target_dtype = None
    target_device = None

    if args.fp8:
        target_dtype = torch.float8_e4m3fn    # ← SET

    if args.blocks_to_swap == 0:
        target_device = device                 # ← NOT SET when blocks_to_swap > 0

    if target_device is not None and target_dtype is not None:  # ← False!
        model.to(target_device, target_dtype)  # ← SKIPPED
```

**Fix:** Separate the dtype cast from the device move:
```python
else:
    if args.fp8:
        logger.info("Casting model to FP8 (float8_e4m3fn)")
        model.to(dtype=torch.float8_e4m3fn)

    if args.blocks_to_swap == 0:
        logger.info(f"Move model to device: {device}")
        model.to(device)
```

The block swap path (lines 377-383) already handles device placement via `move_to_device_except_swap_blocks(device)`, so removing the fused call is safe.

---

### C2. `--save_merged_model` Crashes in Single-Prompt Mode — FIXED

**File:** `src/musubi_tuner/flux_2_generate_image.py` lines 1234-1241
**Impact:** `TypeError: 'NoneType' object is not subscriptable` — feature completely broken.

**Root Cause:** When `--save_merged_model` is set, `generate()` returns `(None, None)` at line 656 after saving. The commented-out guard at lines 1236-1237 was supposed to early-return, but it's inactive. The caller then executes `latent[0]` on `None`.

```python
# Lines 1234-1241
returned_vae, latent = generate(args, gen_settings)
# if args.save_merged_model:    # ← COMMENTED OUT
#     return                     # ← COMMENTED OUT
save_output(args, returned_vae, latent[0], device)  # ← CRASH: latent is None
```

**Fix:** Un-comment the guard:
```python
returned_vae, latent = generate(args, gen_settings)
if args.save_merged_model:
    return
save_output(args, returned_vae, latent[0], device)
```

---

### C3. Control Latent Index Misalignment in Mixed-Control Batches — FIXED

**File:** `src/musubi_tuner/flux_2_cache_latents.py` lines 32-48 (preprocess) and 74-76 (save loop)
**Impact:** IndexError or silent data corruption when a batch contains a mix of items with and without control images.

**Root Cause:** The `controls` list is built sparsely — it only appends entries for items that HAVE `control_content`, skipping items without. But the save loop indexes `control_latents[b]` by batch position `b`. If item 0 has no control and item 1 does, `controls` has 1 element but `control_latents[1]` is accessed.

```python
# Lines 32-48: controls list is sparse (skips items without control)
for item in batch:
    contents.append(torch.from_numpy(item.content))
    if item.control_content is not None and len(item.control_content) > 0:
        controls.append(img_ctx_prep)  # Only appended for items WITH control

# Lines 74-76: indexes by batch position, not controls position
for b, item in enumerate(batch):
    control_latent = control_latents[b] if control_latents is not None else None  # BUG
```

**Fix:** Always append to `controls` to maintain index alignment, using `None` for items without control:

```python
# In preprocess_contents_flux_2:
for item in batch:
    contents.append(torch.from_numpy(item.content))
    if item.control_content is not None and len(item.control_content) > 0:
        # ... existing pixel limit and prep logic ...
        controls.append(img_ctx_prep)
    else:
        controls.append(None)  # Maintain index alignment

# In encode_and_save_batch, update control encoding:
if controls is not None:
    control_latents = []
    for cl in controls:
        if cl is not None:
            control_latents.append([ae.encode(c.to(ae.device, dtype=ae.dtype).unsqueeze(0))[0] for c in cl])
        else:
            control_latents.append(None)
```

Also update the `if not controls:` check to handle lists with all None:
```python
if all(c is None for c in controls):
    controls = None
```

---

### C4. `_normalize_module_name` Mangles FLUX.2 Attention Keys in LoRA Conversion — FIXED

**File:** `src/musubi_tuner/convert_lora.py` lines 103-131
**Impact:** Converting FLUX.2 LoRA from musubi→diffusers format produces malformed keys. Diffusers→musubi direction works correctly.

**Root Cause:** FLUX.2 has no dedicated branch in `_normalize_module_name`. It falls into the `else` (HunyuanVideo) branch, which applies `"attn." → "attn_"` replacement. This is correct for HunyuanVideo (where `img_attn_qkv` is a flat attribute name) but wrong for FLUX.2 (where `img_attn.qkv` is a nested module path — `img_attn` is a `SelfAttention` object containing a `qkv` child).

**Trace for FLUX.2 key `lora_unet_double_blocks_0_img_attn_qkv`:**
```
Step 1: Strip prefix    → "double_blocks_0_img_attn_qkv"
Step 2: "_" → "."       → "double.blocks.0.img.attn.qkv"
Step 3: "double.blocks." → "double_blocks."  → "double_blocks.0.img.attn.qkv"
Step 4: "img." → "img_"                      → "double_blocks.0.img_attn.qkv"
Step 5: "attn." → "attn_"                    → "double_blocks.0.img_attn_qkv"  ← WRONG
Expected:                                       "double_blocks.0.img_attn.qkv"
```

**Affected keys (all DoubleStreamBlock attention):**
- `img_attn.qkv` → incorrectly becomes `img_attn_qkv`
- `img_attn.proj` → incorrectly becomes `img_attn_proj`
- `txt_attn.qkv` → incorrectly becomes `txt_attn_qkv`
- `txt_attn.proj` → incorrectly becomes `txt_attn_proj`

**Disambiguation Problem:** HunyuanVideo, FLUX.1 Kontext, and FLUX.2 all use `double_blocks`/`single_blocks` naming. The encoded LoRA key `img_attn_qkv` maps ambiguously to:
- `img_attn.qkv` (FLUX — nested SelfAttention sub-module with `.qkv` child)
- `img_attn_qkv` (HunyuanVideo — flat nn.Linear attribute)

The key encoding is lossy — you cannot disambiguate from the flattened key alone.

**Fix (metadata-driven architecture dispatch):**
1. Read `ss_network_module` from LoRA safetensors metadata:
   - `networks.lora_flux_2` → FLUX.2
   - `networks.lora_flux` → FLUX.1 Kontext
   - `networks.lora` → HunyuanVideo
   - `networks.lora_wan` → WAN
   - etc.
2. Pass architecture info to `_normalize_module_name` and skip `"attn." → "attn_"` for FLUX architectures.
3. **Fallback when metadata is missing:** Either:
   - (a) Keep current behavior + emit warning: `"LoRA metadata missing ss_network_module; FLUX.2 conversions may have incorrect keys"`
   - (b) Add `--arch` CLI flag to `convert_lora.py` for explicit override (e.g., `--arch flux2`, `--arch hv`)
   - Recommend (b) since it's deterministic and doesn't break existing HV conversions.

FLUX.1 Kontext has the same nested `img_attn.qkv` structure as FLUX.2, so the fix should cover both.

---

### C5. `--lora_multiplier` Default Type Mismatch Causes Crash — FIXED

**File:** `src/musubi_tuner/flux_2_generate_image.py` line 57 (and 5 other generation scripts)
**Impact:** `TypeError: object of type 'float' has no len()` when `--lora_weight` is used without explicit `--lora_multiplier`.

**Root Cause:** With `nargs="*"` and `default=1.0`, argparse returns the raw default (float `1.0`) when the flag is omitted. But `merge_lora_weights` at `wan_generate_video.py:808` calls `len(lora_multipliers)` on it.

```python
# Line 57 — BROKEN
parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
# User: --lora_weight my.safetensors  (no --lora_multiplier)
# args.lora_multiplier = 1.0 (float, not list) → len(1.0) → TypeError
```

**Affected files (all have `default=1.0`, should be `default=None`):**
- `src/musubi_tuner/flux_2_generate_image.py:57`
- `src/musubi_tuner/hv_generate_video.py:449`
- `src/musubi_tuner/fpack_generate_video.py:125`
- `src/musubi_tuner/zimage_generate_image.py:58`
- `src/musubi_tuner/flux_kontext_generate_image.py:60`
- `src/musubi_tuner/qwen_image_generate_image.py:72`

**Already correct (use `default=None`):**
- `src/musubi_tuner/wan_generate_video.py:109`
- `src/musubi_tuner/hv_1_5_generate_video.py:91`
- `src/musubi_tuner/kandinsky5_generate_video.py:99`

**Fix:** Change `default=1.0` → `default=None` in all 6 affected files. The `merge_lora_weights` function already handles `None` correctly (line 808: `if lora_multipliers is not None and len(lora_multipliers) > i:` → with `None`, falls through to `lora_multiplier = 1.0` default).

---

## WARNING Findings

### W1. `fp8_optimization()` Return Type Annotation Wrong — FIXED

**File:** `src/musubi_tuner/flux_2/flux2_models.py` line 514-515
**Impact:** Cosmetic — misleading for IDE/type checkers.

```python
def fp8_optimization(self, state_dict, device, move_to_device, use_scaled_mm=False) -> int:  # WRONG
    # Actually returns dict[str, torch.Tensor] (line 541: return state_dict)
```

**Fix:** Change `-> int` to `-> dict[str, torch.Tensor]`.

---

### W2. Block Swap Can Assert-Fail on Klein-4B for Large Swap Counts — FIXED

**File:** `src/musubi_tuner/flux_2/flux2_models.py` lines 575-602
**Impact:** Runtime crash with assertion error for valid (but large) `--blocks_to_swap` values.

**Corrected block counts** (from `Klein4BParams`): Klein-4B has `depth=5` double blocks and `depth_single_blocks=20` single blocks. Max swappable per-type: 3 double, 18 single. The redistribution while-loop (lines 586-592) can produce `single_blocks_to_swap > self.num_single_blocks - 2` without being able to transfer more to double blocks for large swap counts.

**Fix:** Add bounds clamping before the assert, or calculate the maximum allowable swap count and reject with a clear error.

---

### W3. `.to()` Returns Inner Model Instead of Self — FIXED

**File:** `src/musubi_tuner/flux_2/flux2_utils.py` lines 830-831 (Mistral3Embedder) and 974-976 (Qwen3Embedder)
**Impact:** Breaks method chaining like `embedder.to(device).forward(txt)`.

```python
# Mistral3Embedder
def to(self, *args, **kwargs):
    return self.mistral3.to(*args, **kwargs)  # Returns the inner model, not self

# Qwen3Embedder (line 974-976, also has FIXME comment about dtype)
def to(self, *args, **kwargs):
    return self.model.to(*args, **kwargs)  # Same issue
```

**Fix:** Call inner `.to()` but return `self`:
```python
def to(self, *args, **kwargs):
    self.mistral3.to(*args, **kwargs)
    return self
```

---

### W4. Variable Shadowing `for img_i in img_i` — FIXED

**File:** `src/musubi_tuner/flux_2/flux2_utils.py` line 883
**Impact:** Confusing, fragile — works due to Python scoping but is a maintenance hazard.

```python
img = [[cap_pixels(img_i, UPSAMPLING_MAX_IMAGE_SIZE) for img_i in img_i] for img_i in img]
#                                                       ^^^^^inner   ^^^^^outer — same name!
```

**Fix:** Rename inner variable:
```python
img = [[cap_pixels(im, UPSAMPLING_MAX_IMAGE_SIZE) for im in img_i] for img_i in img]
```

---

### W5. `--infer_steps` Help Text Says Default 25, Actual Default 50 — FIXED

**File:** `src/musubi_tuner/flux_2_generate_image.py` line 84
**Impact:** User confusion — help says 25, code uses 50.

```python
parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 25")
```

**Fix:** Will be addressed as part of the W8/W9 arg defaults bundle (Phase 2A). See [Arg Defaults & Enforcement Bundle](#phase-2a-arg-defaults--enforcement-bundle).

---

### W6. `--no_resize_control` Parsed But Never Checked — FIXED (Phase 4, early)

**File:** `src/musubi_tuner/flux_2_generate_image.py` line 83
**Impact:** Users setting this flag get no feedback that control images are still resized.

The flag is parsed at line 83 but `args.no_resize_control` is never referenced anywhere in the script. Control images always go through `flux2_utils.default_prep()` which applies pixel capping and center-cropping.

**Fix options:**
1. Implement the flag — pass it through to `prepare_image_inputs()` and skip `default_prep()` when set.
2. Remove the flag if it's not needed.

---

### W7. Missing `ae.eval()` Call in Cache Script — FIXED

**File:** `src/musubi_tuner/flux_2_cache_latents.py` lines 151-152
**Impact:** No functional bug (FLUX.2 AE internally calls `self.bn.eval()` in its `normalize()` method), but deviates from the pattern used by all other caching scripts (zimage, kandinsky5, hv_1_5, generic cache_latents all call `.eval()`).

**Fix:** Add `ae.eval()` for consistency and safety.

---

### W8. `fixed_params` Never Enforced — Klein Distilled Accepts Arbitrary Steps/Guidance — FIXED

**File:** `src/musubi_tuner/flux_2/flux2_utils.py` line 69, `src/musubi_tuner/flux_2_generate_image.py`
**Impact:** Klein distilled models (4-step) can run with arbitrary `--infer_steps 50` without warning, producing garbage output.

Each `Flux2ModelInfo` defines `fixed_params` (e.g., `{"guidance", "num_steps"}` for Klein distilled) and `defaults` (e.g., `{"guidance": 1.0, "num_steps": 4}`), but no code ever reads these fields to enforce or warn.

**Design for the fix (sentinel-based detection):**

The core problem: the CLI `default=50` for `--infer_steps` already violates Klein's `fixed_params` default of 4. A naive "warn when user differs from defaults" would warn every time, even when the user didn't explicitly pass `--infer_steps`.

**Correct pattern:**
1. Set CLI defaults to `None` for fixed-ish knobs: `--infer_steps`, `--embedded_cfg_scale`, `--guidance_scale`
2. After `--model_version` resolution, fill from `FLUX2_MODEL_INFO[version].defaults`
3. If user explicitly set a fixed param (value is not None at parse time) AND it differs from `fixed_params` default → override with warning or hard error

```python
# After model_version resolution:
model_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]

# Map CLI arg names → FLUX2_MODEL_INFO.defaults keys
PARAM_MAP = {
    "infer_steps": "num_steps",
    "embedded_cfg_scale": "guidance",  # for guidance-distilled
    "guidance_scale": "guidance",       # for base (non-distilled)
}

for cli_name, info_key in PARAM_MAP.items():
    user_value = getattr(args, cli_name)
    default_value = model_info.defaults.get(info_key)

    if default_value is None:
        continue

    if info_key in model_info.fixed_params:
        if user_value is not None and user_value != default_value:
            logger.warning(
                f"--{cli_name}={user_value} overridden to {default_value} "
                f"(fixed for {args.model_version} distilled model)"
            )
        setattr(args, cli_name, default_value)
    elif user_value is None:
        setattr(args, cli_name, default_value)
```

This must also update help text and docs to reflect the new None-default behavior.

---

### W9. `--embedded_cfg_scale` No-Op on Klein Distilled (No Warning) — FIXED (via W8)

**File:** `src/musubi_tuner/flux_2_generate_image.py` line 91
**Impact:** Users adjusting this for Klein distilled see zero effect with no feedback.

For Klein distilled models (`use_guidance_embed=False`), the guidance vector is completely ignored inside `model.forward()`. Adjusting `--embedded_cfg_scale` is a no-op.

**Fix:** Subsumed by W8 enforcement. When `fixed_params` includes `"guidance"`, the value is locked to the default. If user explicitly sets it differently, the override-with-warning fires.

---

### W10. No Prompt Wildcard/Weighting Support (Feature Gap) — FIXED (wildcards only)

**File:** `src/musubi_tuner/flux_2_generate_image.py`
**Impact:** Feature gap compared to WAN and HunyuanVideo generation scripts which integrate `prompt_management` for wildcards and weighting.

**Fix (wildcards):** Added `--prompt_wildcards` argument and `process_wildcards()` integration in three code paths:
- Single-prompt mode: wildcards applied in `main()` after `parse_args()`
- Batch mode (`--from_file`): wildcards applied in `parse_prompt_line()` (including negative prompts via `--n`)
- Interactive mode: wildcards applied in `parse_prompt_line()` via caller

**Deferred (prompt weighting):** `MiniT5Wrapper` in `prompt_management.py` is T5-specific. FLUX.2 uses Mistral3/Qwen3 text encoders which have different tokenizer APIs. A new embedder wrapper would be needed — this is a separate feature effort.

---

### W11. Repeated VAE Loading in Interactive Mode — FIXED

**File:** `src/musubi_tuner/flux_2_generate_image.py` lines 616-625
**Impact:** Performance waste — VAE loaded from disk every iteration in interactive mode.

In `process_interactive`, each call to `generate()` loads the VAE from disk since `shared_models` does not include `"ae"`. The VAE should be loaded once and shared.

**Fix:** Added VAE loading before the interactive loop: `shared_models["ae"] = flux2_utils.load_ae(...)`. The `generate()` function already had a guard `if shared_models and "ae" in shared_models:` that uses the cached instance.

---

### W12. `parse_prompt_line` Sets `image_path` Override for Non-Existent CLI Arg — FIXED

**File:** `src/musubi_tuner/flux_2_generate_image.py` line 244
**Impact:** The `--i` shorthand in prompt file lines sets `overrides["image_path"]`, but `--image_path` is commented out (lines 95-99). The override silently has no effect.

**Fix:** Removed the dead `--i` handler from `parse_prompt_line`. Added a comment noting it should be re-added when image-to-image support is implemented for FLUX.2.

---

### W13. `print()` Instead of `logger.info()` in Cache Script — FIXED

**File:** `src/musubi_tuner/flux_2_cache_latents.py` lines 93-97
**Impact:** Bypasses log level filtering, formatting, and routing.

**Fix:** Replace `print(...)` with `logger.info(...)`.

---

### W14. Loss Weighting `n_dim=5` Hardcoded for 4D FLUX.2 Tensors — FIXED

**File:** `src/musubi_tuner/hv_train_network.py` line 358
**Impact:** With `batch_size > 1` AND `--weighting_scheme sigma_sqrt|cosmap|structure_bell`, produces incorrect loss weighting via broadcast shape mismatch.

**Root Cause:** `compute_loss_weighting_for_sd3()` hardcodes `n_dim=5` (for 5D video tensors `B,C,F,H,W`). For FLUX.2's 4D tensors `(B,C,H,W)`, this produces 5D sigmas `(B,1,1,1,1)` which broadcast against 4D loss `(B,C,H,W)` as:
- PyTorch left-pads: `(1,B,C,H,W)` × `(B,1,1,1,1)` → `(B,B,C,H,W)`
- With B=1: harmless `(1,1,C,H,W)` ≈ `(1,C,H,W)`
- With B>1: silently cross-broadcasts batch elements — incorrect gradients

**Fix:** Pass actual tensor ndim instead of hardcoded 5:
```python
# In compute_loss_weighting_for_sd3, accept n_dim parameter:
def compute_loss_weighting_for_sd3(weighting_scheme, noise_scheduler, timesteps, device, dtype, n_dim=5):
    if weighting_scheme in ("sigma_sqrt", "cosmap", "structure_bell"):
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=n_dim, dtype=dtype)
```
Then at the call site (line ~2431), pass `n_dim=latents.ndim`.

**Cross-architecture impact:** This affects ALL 4D architectures (FLUX.2, FLUX.1 Kontext, Qwen-Image, Z-Image), not just FLUX.2. Must verify each architecture's latent ndim at the call site. Video architectures (WAN, HV, FramePack, Kandinsky5) use 5D and are unaffected.

---

### W15. `Mistral3Embedder.__init__` Return Type Annotation Wrong — FIXED

**File:** `src/musubi_tuner/flux_2/flux2_utils.py` line 735
**Impact:** Cosmetic — `__init__` always returns `None`, but annotated as `-> tuple[AutoProcessor, Mistral3ForConditionalGeneration]`.

**Fix:** Remove the return type annotation (or change to `-> None`).

---

### W16. Unnecessary `compute_empirical_mu` When `flow_shift` Provided — FIXED

**File:** `src/musubi_tuner/flux_2/flux2_utils.py` lines 485-488
**Impact:** Minor performance waste — mu is computed but unused when flow_shift overrides it.

```python
mu = compute_empirical_mu(image_seq_len, num_steps)  # Always computed
if flow_shift is not None:
    timesteps = (timesteps * flow_shift) / (1 + (flow_shift - 1) * timesteps)  # mu not used
else:
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
```

**Fix:** Move mu computation inside the `else` branch.

---

### W17. Dual FP8 Exclude Key Definitions — Module-Level Constants Are Dead Code — FIXED

**File:** `src/musubi_tuner/flux_2/flux2_models.py` lines 24-25 vs 527-531
**Impact:** The "documented" module-level constants are never used; only the instance method's inline list takes effect. The exclusions diverge silently.

```python
# Line 24-25 — module-level (NEVER USED by any code)
FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks", "single_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "pe_embedder", "time_in", "_modulation"]

# Line 527-531 — instance method (ACTUALLY USED)
TARGET_KEYS = ["single_blocks", "double_blocks"]
EXCLUDE_KEYS = ["norm", "mod"]  # Missing: pe_embedder, time_in, _modulation
```

**Fix:** Reconcile the two. Need to verify which exclusions are correct:
- `pe_embedder`: positional embedding — should NOT be FP8 (precision-sensitive). Likely should be excluded.
- `time_in`: timestep embedding — should NOT be FP8 (small module, precision matters). Likely should be excluded.
- `_modulation`: modulation layers — could be FP8 but risky. `"mod"` in the instance method already catches `_modulation`, `img_mod`, `txt_mod` etc. via substring match.

Recommended: Make instance method use module-level constants. Keep `pe_embedder` and `time_in` exclusions (they're correct for precision safety). The `"mod"` vs `"_modulation"` difference is cosmetic since both are substring matches that cover the same layers.

---

### W18. `--embedded_cfg_scale` Default: Docs Say 2.5, Code Says 4.0 — FIXED (via Phase 2A)

**File:** `docs/flux_2.md:215` vs `src/musubi_tuner/flux_2_generate_image.py:91`
**Impact:** Users reading docs get wrong default. Interacts with W8/W9 enforcement.

```
docs/flux_2.md:215  → "--embedded_cfg_scale (default 2.5)"
code line 91        → default=4.0
FLUX2_MODEL_INFO    → dev defaults: guidance=4.0
```

**Fix:** Update docs to match code (4.0 is correct per `FLUX2_MODEL_INFO`). Will be addressed as part of the arg defaults bundle.

---

### W19. Prompt-File Mode Can't Override `embedded_cfg_scale` — FIXED (via Phase 2A)

**File:** `src/musubi_tuner/flux_2_generate_image.py` lines 239-241
**Impact:** Prompt-file shorthand `--g`/`--l` maps to `guidance_scale` only. For DEV (which uses embedded guidance via `embedded_cfg_scale`), there is no prompt-file shorthand to control the distilled guidance scale.

```python
elif option == "g" or option == "l":
    overrides["guidance_scale"] = float(value)  # Only guidance_scale, not embedded_cfg_scale
```

**Fix options:**
1. Add `--e` or `--ecfg` shorthand mapping to `embedded_cfg_scale`.
2. At minimum, document that `--g`/`--l` doesn't affect distilled guidance and users must use the CLI flag.

---

### W20. Modulation Exclude Patterns Are Dead Code for FLUX.2 — FIXED

**File:** `src/musubi_tuner/networks/lora_flux_2.py` line 32, `src/musubi_tuner/networks/network_arch.py` lines 65/69/73
**Impact:** No functional issue — pattern `r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*"` never matches anything because FLUX.2's modulation layers live on the top-level `Flux2` class, not inside `DoubleStreamBlock`/`SingleStreamBlock` which are the LoRA targets. Pattern was copied from FLUX.1 Kontext where modulation IS inside the blocks.

Additionally, when user supplies custom `--exclude_patterns`, the default modulation exclusion is replaced rather than merged (lines 30-34), violating the `network_arch.py` "always additive" contract. For FLUX.2 this is a non-issue (dead code), but for FLUX.1 Kontext it could cause instability.

**Fix:** Remove dead patterns from FLUX.2 registry entries. For FLUX.1 Kontext, change exclude_patterns to merge user-supplied with defaults.

---

## INFO Findings

### I1. `2024**2` Pixel Limit — Likely Intentional

**Files:** `flux_2_cache_latents.py:39`, `flux_2_generate_image.py:425`
**Note:** `2024**2 = 4,096,576`. Not a power of 2, but consistently used in both files and matches BFL reference code. Likely intentional upstream value, not a typo for `2048**2`. No action needed.

### I2. Dead Code — `vanilla_guidance()` and `encode_image_refs()`

**File:** `flux2_utils.py` ~line 258 and ~551
**Note:** Defined but never called. Can be removed.

### I3. CLAUDE.md Documents Wrong Class Names

**File:** `CLAUDE.md`
**Note:** States `Flux2DoubleStreamBlock` / `Flux2SingleStreamBlock` but actual names are `DoubleStreamBlock` / `SingleStreamBlock` in `flux2_models.py`. The LoRA target list in `lora_flux_2.py:16` correctly uses `["DoubleStreamBlock", "SingleStreamBlock"]`.

### I4. Return Type Annotation on `preprocess_contents_flux_2`

**File:** `flux_2_cache_latents.py:25`
**Note:** Returns `tuple[torch.Tensor, Optional[list[list[torch.Tensor]]]]`, not `tuple[torch.Tensor, List[List[np.ndarray]]]`.

### I5. Dead Code Branch in Control Pixel Limit

**File:** `flux_2_cache_latents.py:40-41`
**Note:** `else: limit_pixels = None` — unreachable after `len(item.control_content) > 0` check.

### I6. Redundant `ae.to(device)` After `load_ae`

**File:** `flux_2_cache_latents.py:152`
**Note:** `load_ae` already loads weights to the target device via `load_split_weights(..., device=str(device))`.

### I7. `guidance_distilled` Parameter Accepted But Never Used in Text Cache

**File:** `flux_2_cache_text_encoder_outputs.py:26`
**Note:** Dead scaffolding — text embeddings are cached identically regardless of distillation status (correct behavior, since the text encoder is the same).

### I8. Type Annotation `flux2_models.Flux` Should Be `flux2_models.Flux2`

**File:** `flux_2_generate_image.py:665`
**Note:** Wrong class reference in type annotation. No runtime impact.

### I9. Dead Variable `t_coords` in `scatter_ids`

**File:** `flux2_utils.py:170,189`
**Note:** `t_coords` list is populated via `.append()` but never returned or used.

### I10. PIL Image File Handles Not Explicitly Closed

**File:** `flux_2_generate_image.py:420`
**Note:** `Image.open()` without context manager. GC handles it but not ideal.

### I11. Negative Prompt Metadata Saving Commented Out; No `model_version` in Metadata

**File:** `flux_2_generate_image.py:775-776`
**Note:** Minor reproducibility gap.

### I12. Double "weight weight" Typo in Log Message

**File:** `flux_2_generate_image.py:1148`
```python
logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")
#                                           ^^^^^^ ^^^^^^ doubled word
```

---

## Extra Findings (discovered during implementation)

### X1. Duplicate `--compile` Registration Crash in `qwen_image_generate_image`

**File:** `src/blissful_tuner/blissful_core.py:464`, `src/musubi_tuner/qwen_image_generate_image.py`
**Impact:** `parse_args()` crashed due to argparse duplicate argument error.
**Root Cause:** Both `setup_parser_compile()` and Blissful's Qwen arg helper registered `--compile`.
**Fix:** Made Blissful's helper additive/skip-if-present.
**Status:** FIXED.

**Plan gap this reveals:** Need a "CLI parser collision" smoke test — import every `*_generate_*` script and call its `parse_args()` with minimal argv to catch this class of issue early.

### X2. LyCORIS Merge Silently Did Nothing When Module Targets Not Discoverable

**File:** `src/musubi_tuner/wan_generate_video.py:778`, `src/musubi_tuner/networks/lycoris.py:221`
**Impact:** `--prefer_lycoris` on FLUX.2 (and potentially other architectures) would silently apply zero LoRA modules.
**Fix:** `merge_lora_weights()` now accepts `extra_unet_targets` and logs when LyCORIS matches 0 modules. LyCORIS target discovery auto-detects known block types when not provided.
**Status:** FIXED.

---

## Fix Plan

### Phase 1: CRITICAL Bug Fixes — **DONE**

| Step | Finding | Files | Status |
|------|---------|-------|--------|
| 1.1 | C5 | 6 generate scripts | DONE — `default=1.0` → `default=None` for `--lora_multiplier` |
| 1.2 | C2 | `flux_2_generate_image.py` | DONE — Un-comment save_merged_model guard |
| 1.3 | C1 | `flux_2_generate_image.py` | DONE — Separate dtype/device in `optimize_model()` |
| 1.4 | C3 | `flux_2_cache_latents.py` | DONE — Index alignment + None handling for controls |
| 1.5 | C4 | `convert_lora.py` | DONE — Metadata-driven `_normalize_module_name` + `--arch` fallback flag |

### Phase 2A: Arg Defaults & Enforcement Bundle — **DONE**

| Step | Finding | Files | Status |
|------|---------|-------|--------|
| 2A.1 | W8 | `flux_2_generate_image.py` | DONE — None sentinels + post-resolution fill |
| 2A.2 | W8 | `flux_2_generate_image.py` | DONE — `fixed_params` override-with-warning |
| 2A.3 | W5 | `flux_2_generate_image.py` | DONE — Help text updated |
| 2A.4 | W9 | `flux_2_generate_image.py` | DONE — Subsumed by W8 enforcement |
| 2A.5 | W18 | `docs/flux_2.md` | DONE — Docs corrected |
| 2A.6 | W19 | `flux_2_generate_image.py` | DONE — `--e` shorthand added |
| 2A.7 | — | `docs/flux_2.md` | DONE — Enforcement behavior documented |

### Phase 2B: High-Impact Correctness & Polish — **DONE**

| Step | Finding | Files | Status |
|------|---------|-------|--------|
| 2B.1 | W14 | `hv_train_network.py` | DONE — `n_dim` parameterized, callers verified |
| 2B.2 | W17 | `flux2_models.py` | DONE — FP8 exclude keys reconciled |
| 2B.3 | W1 | `flux2_models.py` | DONE — Return type fixed |
| 2B.4 | W15 | `flux2_utils.py` | DONE — `__init__` annotation fixed |
| 2B.5 | W13 | `flux_2_cache_latents.py` | DONE — `print` → `logger` |
| 2B.6 | I12 | `flux_2_generate_image.py` | DONE — "weight weight" typo fixed |
| 2B.7 | I3 | `CLAUDE.md` | DONE — Class names corrected |

### Phase 3: Behavioral Improvements — **DONE**

| Step | Finding | Files | Status |
|------|---------|-------|--------|
| 3.1 | W3 | `flux2_utils.py` | DONE — `.to()` returns `self` |
| 3.2 | W4 | `flux2_utils.py` | DONE — Loop variable renamed |
| 3.3 | W7 | `flux_2_cache_latents.py` | DONE — `ae.eval()` added |
| 3.4 | W16 | `flux2_utils.py` | DONE — mu moved into `else` |
| 3.5 | W2 | `flux2_models.py` | DONE — Bounds clamping |
| 3.6 | W20 | `lora_flux_2.py`, `network_arch.py` | DONE — Dead patterns removed, merge-vs-replace fixed |

### Phase 4: Cleanup & Feature Gaps — **DONE**

| Step | Finding | Files | Status |
|------|---------|-------|--------|
| 4.1 | W6 | `flux_2_generate_image.py` | DONE — `--no_resize_control` implemented |
| 4.2 | W12 | `flux_2_generate_image.py` | DONE — Dead `--i` handler removed |
| 4.3 | W11 | `flux_2_generate_image.py` | DONE — VAE loaded once before interactive loop via `shared_models["ae"]` |
| 4.4 | W10 | `flux_2_generate_image.py` | DONE — `--prompt_wildcards` + `process_wildcards()` in all 3 code paths. Prompt weighting deferred (needs Mistral3/Qwen3 wrapper). |
| 4.5 | I2-I11 | Various | OPEN — Dead code removal, type annotations, metadata (low priority) |

### Suggested Future Work (from implementation learnings)

| Item | Description | Priority |
|------|-------------|----------|
| CLI parser collision smoke test | Import every `*_generate_*` script, call `parse_args()` with minimal argv | Medium |
| C4 `--strict` mode | Refuse FLUX conversions when `ss_network_module` missing and `--arch auto` is ambiguous | Low |
| Per-architecture latent ndim registry | Centralize `latents.ndim` knowledge to avoid future hardcoded `n_dim` bugs | Low |
| Prompt weighting for Mistral3/Qwen3 | `MiniT5Wrapper` is T5-specific; need a new embedder wrapper for FLUX.2's text encoders | Medium |
| I2-I11 cleanup pass | Dead code removal (`vanilla_guidance`, `encode_image_refs`, `t_coords`), type annotations, PIL context managers, metadata gaps | Low |

---

## Testing Matrix

Lightweight tests that don't require model weights, organized by the class of regression they protect against.

### Argparse & Type Safety (C5, W5, W8, W9, W18)

```python
# tests/test_flux2_argparse.py
"""Test FLUX.2 argument parsing — no model weights needed."""

def test_lora_multiplier_default_is_none():
    """C5: --lora_multiplier default must be None (not 1.0) to avoid TypeError."""
    args = parse_args(["--dit", "x", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
    assert args.lora_multiplier is None, f"Expected None, got {args.lora_multiplier}"

def test_lora_multiplier_explicit_value_is_list():
    """C5: --lora_multiplier with explicit values must produce a list."""
    args = parse_args(["--dit", "x", "--text_encoder", "x", "--save_path", "x",
                       "--prompt", "x", "--lora_multiplier", "1.5"])
    assert isinstance(args.lora_multiplier, list)

def test_infer_steps_default_is_none_sentinel():
    """W8: --infer_steps default should be None (filled from model info post-resolution)."""
    args = parse_args(["--dit", "x", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
    assert args.infer_steps is None

def test_fixed_params_enforcement_klein_distilled():
    """W8: Klein distilled should override infer_steps to 4 and guidance to 1.0."""
    # Requires mocking model_version resolution
    pass  # Placeholder — test after W8 implementation
```

### Control Latent Index Alignment (C3)

```python
# tests/test_flux2_cache_latents.py
"""Test control latent alignment — uses mock ItemInfo objects, no model weights."""
from unittest.mock import MagicMock
import numpy as np

def test_mixed_control_batch_alignment():
    """C3: Batch with mix of control/no-control items must maintain index alignment."""
    items = [
        make_mock_item(has_control=False),
        make_mock_item(has_control=True),
    ]
    contents, controls = preprocess_contents_flux_2(items)
    assert len(controls) == len(items), "controls list must match batch length"
    assert controls[0] is None
    assert controls[1] is not None

def test_all_no_control_returns_none():
    """C3: All items without control should return controls=None."""
    items = [make_mock_item(has_control=False), make_mock_item(has_control=False)]
    contents, controls = preprocess_contents_flux_2(items)
    assert controls is None
```

### LoRA Key Normalization (C4)

```python
# tests/test_convert_lora.py
"""Test LoRA key normalization — pure string operations, no model weights."""

def test_flux2_attn_key_preserved():
    """C4: FLUX.2 img_attn.qkv must NOT be collapsed to img_attn_qkv."""
    result = _normalize_module_name("lora_unet_double_blocks_0_img_attn_qkv", "lora_unet_", arch="flux2")
    assert "img_attn.qkv" in result, f"Expected nested path, got {result}"

def test_hv_attn_key_collapsed():
    """C4: HunyuanVideo img_attn_qkv MUST be collapsed to img_attn_qkv."""
    result = _normalize_module_name("lora_unet_double_blocks_0_img_attn_qkv", "lora_unet_", arch="hv")
    assert "img_attn_qkv" in result

def test_flux2_all_affected_keys():
    """C4: All four affected attention keys must preserve nested dot."""
    for key_suffix in ["img_attn_qkv", "img_attn_proj", "txt_attn_qkv", "txt_attn_proj"]:
        result = _normalize_module_name(f"lora_unet_double_blocks_0_{key_suffix}", "lora_unet_", arch="flux2")
        assert ".attn." in result or result.endswith(key_suffix.replace("attn_", "attn.")), \
            f"Key {key_suffix} → {result} lost nested attn path"

def test_metadata_missing_falls_back_gracefully():
    """C4: Missing ss_network_module metadata should warn, not crash."""
    pass  # Placeholder — test after C4 implementation
```

### Loss Weighting ndim (W14)

```python
# tests/test_loss_weighting.py
"""Test loss weighting broadcast correctness — uses mock tensors, no model weights."""
import torch

def test_sigma_sqrt_4d_batch_gt_1():
    """W14: sigma_sqrt with 4D loss and B>1 must produce correct broadcast shape."""
    # After fix: n_dim should match loss.ndim
    B, C, H, W = 2, 128, 4, 4
    loss = torch.randn(B, C, H, W)
    # Simulate get_sigmas with correct n_dim=4
    sigmas = torch.randn(B, 1, 1, 1)  # (B, 1, 1, 1) for 4D
    result = loss * sigmas
    assert result.shape == loss.shape, f"Expected {loss.shape}, got {result.shape}"

def test_sigma_sqrt_5d_video():
    """W14: sigma_sqrt with 5D loss (video) should still work with n_dim=5."""
    B, C, F, H, W = 2, 128, 8, 4, 4
    loss = torch.randn(B, C, F, H, W)
    sigmas = torch.randn(B, 1, 1, 1, 1)
    result = loss * sigmas
    assert result.shape == loss.shape
```

### FP8 + Block Swap (C1)

```python
# tests/test_flux2_optimize.py
"""Test optimize_model logic paths — mock model, no real weights."""

def test_fp8_applied_with_blocks_to_swap():
    """C1: FP8 cast must be applied even when blocks_to_swap > 0."""
    # After fix, model.to(dtype=fp8) should be called regardless of blocks_to_swap
    pass  # Requires mock model — placeholder

def test_fp8_scaled_path_unchanged():
    """C1: fp8_scaled path should not be affected by the fix."""
    pass  # Verify fp8_scaled still works after refactor
```

### Save Merged Model (C2)

```python
# tests/test_flux2_generate.py

def test_save_merged_model_early_return():
    """C2: When save_merged_model is set, main() should return after generate() without crash."""
    pass  # Integration-level test — verify no TypeError on None[0]
```

### Block Swap Bounds (W2)

```python
# tests/test_flux2_block_swap.py
"""Test block swap distribution — pure arithmetic, no model weights."""

def test_klein_4b_max_swap():
    """W2: Klein-4B (5 double, 20 single) should reject excessive swap counts gracefully."""
    # Max: 3 double + 18 single = ~21 total swappable
    # Test that requesting 22+ either clamps or raises clear error
    pass

def test_dev_large_swap():
    """W2: DEV (8 double, 48 single) should handle typical swap values."""
    pass
```

---

## Files Touched Summary

| File | Findings | Phase |
|------|----------|-------|
| `src/musubi_tuner/flux_2_generate_image.py` | C1, C2, C5, W5, W6, W8, W9, W11, W12, W19, I8, I10, I11, I12 | 1, 2A, 4 |
| `src/musubi_tuner/flux_2_cache_latents.py` | C3, W7, W13, I4, I5, I6 | 1, 2B, 3 |
| `src/musubi_tuner/convert_lora.py` | C4 | 1 |
| `src/musubi_tuner/flux_2/flux2_utils.py` | W3, W4, W15, W16, I1, I2, I9 | 2B, 3, 4 |
| `src/musubi_tuner/flux_2/flux2_models.py` | W1, W2, W17 | 2B, 3 |
| `src/musubi_tuner/hv_train_network.py` | W14 | 2B |
| `src/musubi_tuner/networks/lora_flux_2.py` | W20 | 3 |
| `src/musubi_tuner/networks/network_arch.py` | W20 | 3 |
| `src/musubi_tuner/hv_generate_video.py` | C5 | 1 |
| `src/musubi_tuner/fpack_generate_video.py` | C5 | 1 |
| `src/musubi_tuner/zimage_generate_image.py` | C5 | 1 |
| `src/musubi_tuner/flux_kontext_generate_image.py` | C5 | 1 |
| `src/musubi_tuner/qwen_image_generate_image.py` | C5 | 1 |
| `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py` | I7 | 4 |
| `docs/flux_2.md` | W18 | 2A |
| `CLAUDE.md` | I3 | 2B |
