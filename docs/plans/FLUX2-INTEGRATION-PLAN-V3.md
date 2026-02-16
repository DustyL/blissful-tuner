# FLUX.2 Upstream Merge Plan (Revised v3 - Final)

## Overview

Integrate upstream musubi-tuner FLUX.2 refactoring into blissful-tuner while preserving custom features (masked loss training, BlissfulLogger).

**Upstream source:** `/Users/dustin/musubi-tuner` (commit 594f19f)
**Blissful target:** `/Users/dustin/blissful-tuner` (HEAD: bc21c5c)

## Key Changes in Upstream

| Area | Upstream (New) | Blissful (Current) |
|------|----------------|-------------------|
| Architecture constants | Short codes: `f2d`, `f2k4b`, `f2k9b` | Single: `f2` |
| Model info structure | `@dataclass Flux2ModelInfo` | Plain dict |
| Model version names | `dev`, `klein-4b`, `klein-base-4b`, `klein-9b`, `klein-base-9b` | `flux.2-klein-4b`, etc. |
| Text encoder | Qwen3 for Klein, Mistral3 for dev | Mistral3 only |
| Cache filenames | `*_f2d.safetensors`, `*_f2k4b.safetensors` | `*_f2.safetensors` |
| Control config keys | `no_resize_control`, `control_resolution` | `flux_kontext_no_resize_control`, etc. |

## Custom Features to Preserve

1. **Masked loss** (`flux_2_cache_latents.py`, `flux_2_train_network.py`, `modules/mask_loss.py`)
2. **Dataset mask keys** (`config_utils.py`): `mask_directory`, `alpha_mask`, `require_mask`
3. **BlissfulLogger** in `image_video_dataset.py`

---

## Execution Phases

### Phase 0: Pre-Merge Inventory (Informational)

Generate file-truth diffs for each critical file:

```bash
set -euo pipefail

# For each file, see exact differences to reconcile
for f in \
  src/musubi_tuner/dataset/image_video_dataset.py \
  src/musubi_tuner/dataset/config_utils.py \
  src/musubi_tuner/flux_2/flux2_models.py \
  src/musubi_tuner/flux_2/flux2_utils.py \
  src/musubi_tuner/flux_2_cache_latents.py \
  src/musubi_tuner/flux_2_cache_text_encoder_outputs.py \
  src/musubi_tuner/flux_2_train_network.py \
  src/musubi_tuner/flux_2_generate_image.py \
  src/musubi_tuner/modules/mask_loss.py \
  src/musubi_tuner/utils/sai_model_spec.py; do
  echo "=== $f ==="
  # git diff --no-index exits 1 when there are differences; ignore it.
  (git diff --no-index /Users/dustin/blissful-tuner/$f /Users/dustin/musubi-tuner/$f 2>/dev/null || true) | head -100 || true
done
```

**Verify dependencies:**
```bash
python -c "from transformers import Qwen3ForCausalLM, Qwen2Tokenizer; print('Qwen3 OK')"
python -c "from transformers import Mistral3ForConditionalGeneration; print('Mistral3 OK')"
```

---

### Phase 1: Architecture Constants & Control Key Migration

**File:** `src/musubi_tuner/dataset/image_video_dataset.py`

1. Replace single `ARCHITECTURE_FLUX_2` with multi-variant constants:
```python
ARCHITECTURE_FLUX_2_DEV = "f2d"
ARCHITECTURE_FLUX_2_DEV_FULL = "flux_2_dev"
ARCHITECTURE_FLUX_2_KLEIN_4B = "f2k4b"
ARCHITECTURE_FLUX_2_KLEIN_4B_FULL = "flux_2_klein_4b"
ARCHITECTURE_FLUX_2_KLEIN_9B = "f2k9b"
ARCHITECTURE_FLUX_2_KLEIN_9B_FULL = "flux_2_klein_9b"
```

2. Update `RESOLUTION_STEPS` dict with new architecture keys

3. **Control key migration** in `ImageDataset`/`VideoDataset`:
   - Both `flux_kontext_no_resize_control` AND `qwen_image_edit_no_resize_control` converge into single `no_resize_control` boolean
   - `qwen_image_edit_control_resolution` → `control_resolution`
   - Update bucket-key logic to use: `if self.no_resize_control or self.control_resolution is not None:`

**PRESERVE:** BlissfulLogger integration

**Grep gate after Phase 1:**
```bash
set -euo pipefail

# Ensure old control keys are not used outside config deprecation handling
if rg "flux_kontext_no_resize_control|qwen_image_edit_no_resize_control|qwen_image_edit_control_resolution" \
     src/musubi_tuner/dataset/ --glob '!*test*' \
  | rg -v "^src/musubi_tuner/dataset/config_utils\\.py:"; then
  echo "Found deprecated control keys outside config_utils.py"
  exit 1
fi
```

---

### Phase 2: Model Definitions

**File:** `src/musubi_tuner/flux_2/flux2_models.py`

1. Add FP8 optimization constants:
```python
FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks", "single_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "pe_embedder", "time_in", "_modulation"]
```

2. Add/verify `Klein4BParams`, `Klein9BParams` dataclasses
3. Ensure `Flux2Params` matches upstream

---

### Phase 3: Model Utilities

**File:** `src/musubi_tuner/flux_2/flux2_utils.py`

1. Add `@dataclass Flux2ModelInfo`:
```python
@dataclass
class Flux2ModelInfo:
    params: Flux2Params
    defaults: dict[str, float | int]
    fixed_params: set[str]
    guidance_distilled: bool
    architecture: str        # Short code for cache filenames (f2d, f2k4b)
    architecture_full: str   # Full name for metadata (flux_2_dev)
    qwen_variant: Optional[str] = None  # None=Mistral3, "4B"/"8B"=Qwen3
```

2. Build `FLUX2_MODEL_INFO` with canonical keys only (`dev`, `klein-4b`, etc.)

3. Add back-compat via argparse type mapping (NOT dict aliases):
```python
MODEL_VERSION_ALIASES = {
    "flux.2-dev": "dev",
    "flux.2-klein-4b": "klein-4b",
    "flux.2-klein-base-4b": "klein-base-4b",
    "flux.2-klein-9b": "klein-9b",
    "flux.2-klein-base-9b": "klein-base-9b",
}

def resolve_model_version(v: str) -> str:
    return MODEL_VERSION_ALIASES.get(v, v)

def add_model_version_args(parser):
    parser.add_argument(
        "--model_version",
        type=resolve_model_version,  # Accepts aliases, converts to canonical
        choices=list(FLUX2_MODEL_INFO.keys()),
        default="dev",
    )
```

4. **Reuse** `zimage_utils.load_qwen3()` for Qwen3 text encoder loading (already exists)

5. Update function signatures:
   - `load_text_embedder(model_version_info: Flux2ModelInfo, ...)`
   - `load_flow_model(device, model_version_info: Flux2ModelInfo, ...)`

**Compile gate after this phase:**
```bash
python -m compileall -q src/musubi_tuner/flux_2/
```

**Grep gate for dict-style access:**
```bash
set -euo pipefail

# Ensure no old dict-style FLUX2 model info access remains (bracket or .get() patterns)
if rg 'FLUX2_MODEL_INFO\[.*\]\["' src/musubi_tuner/; then
  echo "Found dict-style bracket access to FLUX2_MODEL_INFO[...]"
  exit 1
fi
if rg 'FLUX2_MODEL_INFO\[.*\]\.get\(' src/musubi_tuner/; then
  echo "Found dict-style .get() access to FLUX2_MODEL_INFO[...]"
  exit 1
fi
```

---

### Phase 4: Config Utilities

**File:** `src/musubi_tuner/dataset/config_utils.py`

1. **Rename control keys in config schema** - exact touchpoints:
   - **Dataclasses**: Remove `flux_kontext_no_resize_control`, `qwen_image_edit_no_resize_control`, `qwen_image_edit_control_resolution` from `ImageDatasetParams`, `VideoDatasetParams`, `SubsetParams` (or equivalent)
   - **Add**: `no_resize_control: bool = False`, `control_resolution: Optional[tuple[int, int]] = None`
   - **ConfigSanitizer key sets**: Update any `allowed_keys` or `known_keys` sets to use new names
   - **generate_dataset_group_by_blueprint**: Update YAML/TOML rendering to use new keys
   - Old key strings must exist **only** in `DEPRECATED_KEY_MAP`

2. Add key normalization that applies to `[general]`, `[[datasets]]`, and `[[datasets.subsets]]`.

**Placement:** Place helpers after the module `logger = ...` line and call `normalize_deprecated_keys_in_user_config(config)` just before returning in `load_user_config()` (applies to both TOML and JSON code paths).

```python
from typing import Any

DEPRECATED_KEY_MAP = {
    "flux_kontext_no_resize_control": "no_resize_control",
    "qwen_image_edit_no_resize_control": "no_resize_control",
    "qwen_image_edit_control_resolution": "control_resolution",
}

def normalize_deprecated_keys_in_section(section: dict[str, Any], *, section_name: str) -> None:
    for old_key, new_key in DEPRECATED_KEY_MAP.items():
        if old_key not in section:
            continue

        if new_key in section:
            logger.warning(
                f"Deprecated config key '{old_key}' is ignored because '{new_key}' is already set in {section_name}."
            )
            section.pop(old_key, None)
            continue

        section[new_key] = section.pop(old_key)
        logger.warning(f"Deprecated config key '{old_key}' found in {section_name}; use '{new_key}' instead.")


def normalize_deprecated_keys_in_user_config(config: dict[str, Any]) -> None:
    general = config.get("general")
    if isinstance(general, dict):
        normalize_deprecated_keys_in_section(general, section_name="general")

    datasets = config.get("datasets")
    if not isinstance(datasets, list):
        return

    for dataset_idx, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            continue

        normalize_deprecated_keys_in_section(dataset, section_name=f"datasets[{dataset_idx}]")

        subsets = dataset.get("subsets")
        if not isinstance(subsets, list):
            continue

        for subset_idx, subset in enumerate(subsets):
            if isinstance(subset, dict):
                normalize_deprecated_keys_in_section(
                    subset,
                    section_name=f"datasets[{dataset_idx}].subsets[{subset_idx}]",
                )


# In load_user_config(), after parsing the file into `config` and before returning it:
# normalize_deprecated_keys_in_user_config(config)
```

3. **PRESERVE** blissful mask keys in subset config:
```python
mask_directory: Optional[str] = None
alpha_mask: bool = False
require_mask: bool = False
```

**Grep gate after Phase 4:**
```bash
set -euo pipefail

# Verify old control key strings appear only once each (in DEPRECATED_KEY_MAP)
for k in flux_kontext_no_resize_control qwen_image_edit_no_resize_control qwen_image_edit_control_resolution; do
  n="$( (rg -n "$k" src/musubi_tuner/dataset/config_utils.py || true) | wc -l | tr -d ' ')"
  test "$n" -eq 1 || { echo "Unexpected count for $k: $n"; exit 1; }
done

# Broader scan: ensure deprecated keys don't appear elsewhere in src/musubi_tuner/
if rg "flux_kontext_no_resize_control|qwen_image_edit_no_resize_control|qwen_image_edit_control_resolution" \
     src/musubi_tuner/ --glob '!*test*' \
  | rg -v "^src/musubi_tuner/dataset/config_utils\\.py:"; then
  echo "Found deprecated control keys outside config_utils.py"
  exit 1
fi
```

---

### Phase 5: Cache Save Functions

**File:** `src/musubi_tuner/dataset/image_video_dataset.py`

1. Update `save_latent_cache_flux_2()`:
```python
def save_latent_cache_flux_2(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: Optional[list[torch.Tensor]],
    arch_full: str,  # NEW: e.g., "flux_2_dev" (for metadata)
    mask_weights: Optional[torch.Tensor] = None,  # PRESERVE
):
    # Filename uses SHORT arch code from item_info.architecture (e.g., f2d)
    # Metadata uses arch_full (e.g., flux_2_dev)
```

2. Update `save_text_encoder_output_cache_flux_2()` similarly

---

### Phase 6: Cache Latents Script

**File:** `src/musubi_tuner/flux_2_cache_latents.py`

1. Use `model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]`
2. Pass `arch_full=model_version_info.architecture_full` to save functions
3. **ALIGN:** Accept `--vae_dtype` argument (upstream supports it, blissful currently rejects)
4. **PRESERVE:** Mask processing logic

---

### Phase 7: Cache Text Encoder Script

**File:** `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py`

1. Use `model_version_info`
2. Update `load_text_embedder()` call
3. Use FP8 autocast handling:
```python
autocast_dtype = torch.bfloat16 if text_embedder.dtype.itemsize == 1 else text_embedder.dtype
```
4. Pass `arch_full` to save function

---

### Phase 8: Mask Loss Utility

**File:** `src/musubi_tuner/modules/mask_loss.py`

Update `apply_masked_loss()` to accept 4D loss tensors `(B,C,H,W)` for FLUX.2 compatibility.

**Key insight:** Current mask_weights semantics are `(B, F, H, W)` → `unsqueeze(1)` → `(B, 1, F, H, W)`.
For 4D loss support, only unsqueeze the loss tensor (treat as F=1), leave mask handling unchanged.
If mask dimensions don't match, let it error correctly.

**Important:** Do not squeeze anything back; `apply_masked_loss()` returns a scalar.

```python
# In apply_masked_loss() (after the early return for unmasked path),
# relax the strict ndim check like this:
if loss.ndim == 4:
    # FLUX.2 produces per-image loss (B, C, H, W); treat it as F=1 for layout='video'
    if layout != "video":
        raise ValueError("4D loss is only supported for layout='video'")
    loss = loss.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
elif loss.ndim != 5:
    raise ValueError(f"Expected loss to be 4D or 5D, got {loss.ndim}D: {tuple(loss.shape)}")
```

This allows `flux_2_train_network.py` to stay closer to upstream (no 5D expansion hack).

**Compile gate:**
```bash
python -m compileall -q src/musubi_tuner/modules/
```

---

### Phase 9: Training Script

**File:** `src/musubi_tuner/flux_2_train_network.py`

1. Add `self.model_version_info` in `handle_model_specific_args()`:
```python
self.model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
self.default_discrete_flow_shift = None
```

2. Update `architecture` and `architecture_full_name` properties

3. Update `process_sample_prompts()`:
   - Select Mistral3 or Qwen3 based on `model_version_info.qwen_variant`
   - Add negative prompt handling for non-guidance-distilled models

4. Update `do_inference()` for guidance_distilled branching

5. Update `load_transformer()` to pass `model_version_info`

6. **Remove** 5D tensor expansion in `call_dit()` (now handled by Phase 8)

**Compile gate:**
```bash
python -m compileall -q src/musubi_tuner/flux_2_train_network.py
```

---

### Phase 10: Inference Script

**File:** `src/musubi_tuner/flux_2_generate_image.py`

1. Use `model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]`
2. Use `model_version_info.guidance_distilled` for inference branching
3. Use `model_version_info.qwen_variant` for text encoder selection
4. Update all hardcoded architecture references

---

### Phase 11: Model Spec

**File:** `src/musubi_tuner/utils/sai_model_spec.py`

Add imports and handling for new FLUX.2 architecture variants.

---

## Critical Files Summary

| # | File | Key Changes |
|---|------|-------------|
| 1 | `dataset/image_video_dataset.py` | Arch constants, control key migration, save functions |
| 2 | `flux_2/flux2_models.py` | FP8 constants, model params |
| 3 | `flux_2/flux2_utils.py` | Dataclass, Qwen3, back-compat shims |
| 4 | `dataset/config_utils.py` | Key deprecation, preserve mask keys |
| 5 | `flux_2_cache_latents.py` | model_version_info, vae_dtype, preserve masks |
| 6 | `flux_2_cache_text_encoder_outputs.py` | model_version_info, FP8 handling |
| 7 | `modules/mask_loss.py` | 4D loss tensor support |
| 8 | `flux_2_train_network.py` | Instance methods, remove 5D hack |
| 9 | `flux_2_generate_image.py` | Inference updates |
| 10 | `utils/sai_model_spec.py` | New arch constants |

---

## Verification Plan

### 1. Syntax/Import/Lint Check
```bash
python -m compileall -q src
ruff check src/musubi_tuner/flux_2/ src/musubi_tuner/flux_2_*.py src/musubi_tuner/dataset/ src/musubi_tuner/modules/mask_loss.py src/musubi_tuner/utils/sai_model_spec.py
python -c "from musubi_tuner.flux_2 import flux2_utils, flux2_models; print('OK')"
python -c "from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer; print('OK')"
python -c "from musubi_tuner.flux_2_generate_image import main; print('OK')"
```

### 1b. Non-FLUX2 Smoke Check (control key paths)
```bash
# Verify control-key changes don't break non-FLUX2 paths
python -m compileall -q src/musubi_tuner/dataset
python flux_kontext_train_network.py --help > /dev/null
python qwen_image_train_network.py --help > /dev/null
python qwen_image_generate_image.py --help > /dev/null
```

### 2. Model Version & Back-Compat Check
```bash
python flux_2_train_network.py --help | grep -A5 model_version
# Should show: dev, klein-4b, klein-base-4b, klein-9b, klein-base-9b (NO aliases)

# Test back-compat accepts old names
python -c "
from musubi_tuner.flux_2.flux2_utils import resolve_model_version, FLUX2_MODEL_INFO
assert resolve_model_version('flux.2-klein-4b') == 'klein-4b'
assert resolve_model_version('flux.2-klein-base-4b') == 'klein-base-4b'
assert 'klein-4b' in FLUX2_MODEL_INFO
print('Back-compat OK')
"
```

### 3. Cache Test with Metadata Verification (Latents)
```bash
# Cache latents (dev model)
python flux_2_cache_latents.py \
    --dataset_config test_config.toml \
    --vae /path/to/ae.sft \
    --model_version dev

# Verify filename uses SHORT code, metadata uses FULL name
python -c "
from safetensors import safe_open
import glob

cache_file = glob.glob('/path/to/cache/*_f2d.safetensors')[0]
with safe_open(cache_file, framework='pt') as f:
    meta = f.metadata()
    assert meta.get('architecture') == 'flux_2_dev', f'Got: {meta}'
    print(f'Latent cache OK: {cache_file}')
"
```

### 4. Mask Presence Check (if using masks)
```bash
python -c "
from safetensors import safe_open
cache_file = '/path/to/cache/sample_f2d.safetensors'
with safe_open(cache_file, framework='pt') as f:
    keys = f.keys()
    mask_keys = [k for k in keys if 'mask_weights' in k]
    print(f'Mask keys: {mask_keys}')
    assert len(mask_keys) > 0, 'mask_weights not found!'
"
```

### 5. Training Smoke Test (dev model)
```bash
accelerate launch --mixed_precision bf16 flux_2_train_network.py \
    --model_version dev \
    --dit /path/to/flux2-dev.safetensors \
    --text_encoder /path/to/mistral3.safetensors \
    --vae /path/to/ae.sft \
    --dataset_config test.toml \
    --network_module networks.lora_flux_2 \
    --network_dim 32 \
    --max_train_steps 5 \
    --use_mask_loss
```

### 6. Inference Test (dev model)
```bash
python flux_2_generate_image.py \
    --model_version dev \
    --dit /path/to/flux2-dev.safetensors \
    --vae /path/to/ae.sft \
    --text_encoder /path/to/mistral3.safetensors \
    --prompt "test prompt" \
    --save_path /tmp/test_dev.png
```

### 7. Klein Text Encoder Cache Test (if Qwen3 available)
```bash
# Cache text encoder outputs for Klein
python flux_2_cache_text_encoder_outputs.py \
    --dataset_config test_config.toml \
    --text_encoder /path/to/qwen3-4b.safetensors \
    --model_version klein-4b

# Verify TE cache filename and metadata
python -c "
from safetensors import safe_open
import glob

te_cache = glob.glob('/path/to/cache/*_f2k4b_te.safetensors')[0]
with safe_open(te_cache, framework='pt') as f:
    meta = f.metadata()
    assert meta.get('architecture') == 'flux_2_klein_4b', f'Got: {meta}'
    print(f'TE cache OK: {te_cache}')
"
```

### 8. Klein Inference Smoke Test (if Qwen3 available)
```bash
python flux_2_generate_image.py \
    --model_version klein-4b \
    --dit /path/to/flux2-klein-4b.safetensors \
    --vae /path/to/ae.sft \
    --text_encoder /path/to/qwen3-4b.safetensors \
    --prompt "test prompt" \
    --save_path /tmp/test_klein.png
```

---

## Rollback Strategy

```bash
# Restore individual files
git checkout bc21c5c -- <file>

# Full rollback
git reset --hard bc21c5c
```

---

## Breaking Changes & Migration

1. **Latent cache files:** `*_f2.safetensors` → `*_f2d.safetensors`, `*_f2k4b.safetensors`, etc. (re-cache required)
2. **TE cache files:** `*_f2_te.safetensors` → `*_f2d_te.safetensors`, `*_f2k4b_te.safetensors`, etc. (re-cache required)
3. **Model versions:** `flux.2-*` → `dev`, `klein-*` (old names accepted via alias)
4. **Config keys:** `flux_kontext_no_resize_control` → `no_resize_control` (warning shown)
5. **Text encoder:** Klein models need Qwen3 checkpoints

---

## Done Criteria

All phases complete when:
- [ ] Phases 1-11 executed (Phase 0 is informational prep)
- [ ] `python -m compileall -q src` passes
- [ ] `ruff check` passes on modified files
- [ ] All grep gates pass (no old patterns remain)
- [ ] Import checks pass
- [ ] Non-FLUX2 smoke check passes (control key paths)
- [ ] Back-compat test passes
- [ ] Latent cache metadata verification passes
- [ ] Training smoke test completes 5 steps
- [ ] Dev inference test produces output image
- [ ] Klein TE cache verification passes (if Qwen3 available)
- [ ] Klein inference test produces output image (if Qwen3 available)

---

## Follow-Up Tasks (Optional)

- [ ] Update blissful docs/config examples that mention `flux.2-*` to use new canonical names
- [ ] Update docs that reference old control keys (`flux_kontext_no_resize_control`, etc.)
