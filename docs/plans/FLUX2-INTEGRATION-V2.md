# FLUX.2 Upstream Merge Plan (Revised v2)

## Overview

Integrate upstream musubi-tuner FLUX.2 refactoring into blissful-tuner while preserving custom features (masked loss training, BlissfulLogger).

**Upstream source:** `/Users/dustin/musubi-tuner` (commit 594f19f)
**Blissful target:** `/Users/dustin/blissful-tuner` (HEAD: bc21c5c)

## Key Changes in Upstream

| Area | Upstream (New) | Blissful (Current) |
|------|----------------|-------------------|
| Architecture constants | Short codes: `f2d`, `f2k4b`, `f2k9b` | Single: `f2` |
| Model info structure | `@dataclass Flux2ModelInfo` | Plain dict |
| Model version names | `dev`, `klein-4b`, `klein-9b` | `flux.2-klein-4b`, etc. |
| Text encoder | Qwen3 for Klein, Mistral3 for dev | Mistral3 only |
| Cache filenames | `*_f2d.safetensors`, `*_f2k4b.safetensors` | `*_f2.safetensors` |
| Control config keys | `no_resize_control`, `control_resolution` | `flux_kontext_no_resize_control`, etc. |

## Custom Features to Preserve

1. **Masked loss** (`flux_2_cache_latents.py`, `flux_2_train_network.py`, `modules/mask_loss.py`)
2. **Dataset mask keys** (`config_utils.py`): `mask_directory`, `alpha_mask`, `require_mask`
3. **BlissfulLogger** in `image_video_dataset.py`

---

## Execution Phases

### Phase 0: Pre-Merge Inventory

Generate file-truth diffs for each critical file:

```bash
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
  src/musubi_tuner/utils/sai_model_spec.py; do
  echo "=== $f ===" && git diff --no-index /Users/dustin/blissful-tuner/$f /Users/dustin/musubi-tuner/$f 2>/dev/null | head -100
done
```

**Verify dependencies:**
```bash
python -c "from transformers import Qwen3ForCausalLM, Qwen2Tokenizer; print('Qwen3 OK')"
python -c "from transformers import Mistral3ForConditionalGeneration; print('Mistral3 OK')"
```

---

### Phase 1: Architecture Constants

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

3. Migrate control config parameter names in `ImageDataset`/`VideoDataset`:
   - `flux_kontext_no_resize_control` → `no_resize_control`
   - `qwen_image_edit_no_resize_control` → (same)
   - `qwen_image_edit_control_resolution` → `control_resolution`

**PRESERVE:** BlissfulLogger integration

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
    "flux.2-klein-9b": "klein-9b",
    # etc.
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

4. Add `Qwen3Embedder` class and `load_qwen3()` function

5. Update function signatures:
   - `load_text_embedder(model_version_info: Flux2ModelInfo, ...)`
   - `load_flow_model(device, model_version_info: Flux2ModelInfo, ...)`

**Compile gate after this phase:**
```bash
python -m compileall -q src/musubi_tuner/flux_2/
```

---

### Phase 4: Config Utilities

**File:** `src/musubi_tuner/dataset/config_utils.py`

1. Add key normalization in `load_user_config()`:
```python
DEPRECATED_KEY_MAP = {
    "flux_kontext_no_resize_control": "no_resize_control",
    "qwen_image_edit_no_resize_control": "no_resize_control",
    "qwen_image_edit_control_resolution": "control_resolution",
}

def normalize_deprecated_keys(config_dict):
    for old, new in DEPRECATED_KEY_MAP.items():
        if old in config_dict:
            warnings.warn(f"'{old}' is deprecated, use '{new}'", DeprecationWarning)
            config_dict[new] = config_dict.pop(old)
```

2. **PRESERVE** blissful mask keys in subset config:
```python
mask_directory: Optional[str] = None
alpha_mask: bool = False
require_mask: bool = False
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

Update `apply_masked_loss()` to accept both 4D `(B,C,H,W)` and 5D `(B,C,F,H,W)` tensors:
```python
def apply_masked_loss(loss, mask_weights, layout="video"):
    # If 4D input (B,C,H,W), treat as F=1 for FLUX.2 compatibility
    if loss.ndim == 4:
        loss = loss.unsqueeze(2)  # B,C,H,W -> B,C,1,H,W
        if mask_weights.ndim == 4:
            mask_weights = mask_weights.unsqueeze(2)
    # ... rest of existing logic
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
| 7 | `modules/mask_loss.py` | 4D+5D tensor support |
| 8 | `flux_2_train_network.py` | Instance methods, remove 5D hack |
| 9 | `flux_2_generate_image.py` | Inference updates |
| 10 | `utils/sai_model_spec.py` | New arch constants |

---

## Verification Plan

### 1. Syntax/Import Check
```bash
python -m compileall -q src
python -c "from musubi_tuner.flux_2 import flux2_utils, flux2_models; print('OK')"
python -c "from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer; print('OK')"
python -c "from musubi_tuner.flux_2_generate_image import main; print('OK')"
```

### 2. Model Version & Back-Compat Check
```bash
python flux_2_train_network.py --help | grep -A5 model_version
# Should show: dev, klein-4b, klein-base-4b, klein-9b, klein-base-9b (NO aliases)

# Test back-compat accepts old names
python -c "
from musubi_tuner.flux_2.flux2_utils import resolve_model_version, FLUX2_MODEL_INFO
assert resolve_model_version('flux.2-klein-4b') == 'klein-4b'
assert 'klein-4b' in FLUX2_MODEL_INFO
print('Back-compat OK')
"
```

### 3. Cache Test with Metadata Verification
```bash
# Cache latents (dev model)
python flux_2_cache_latents.py \
    --dataset_config test_config.toml \
    --vae /path/to/ae.sft \
    --model_version dev

# Verify filename uses SHORT code, metadata uses FULL name
python -c "
import safetensors.safe_open
import glob

cache_file = glob.glob('/path/to/cache/*_f2d.safetensors')[0]
with safetensors.safe_open(cache_file, framework='pt') as f:
    meta = f.metadata()
    assert meta.get('architecture') == 'flux_2_dev', f'Got: {meta}'
    print(f'Cache OK: {cache_file}')
"
```

### 4. Mask Presence Check (if using masks)
```bash
python -c "
import safetensors.safe_open
cache_file = '/path/to/cache/sample_f2d.safetensors'
with safetensors.safe_open(cache_file, framework='pt') as f:
    keys = f.keys()
    mask_keys = [k for k in keys if 'mask_weights' in k]
    print(f'Mask keys: {mask_keys}')
    assert len(mask_keys) > 0, 'mask_weights not found!'
"
```

### 5. Training Smoke Test
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

### 6. Inference Test
```bash
python flux_2_generate_image.py \
    --model_version dev \
    --dit /path/to/flux2-dev.safetensors \
    --vae /path/to/ae.sft \
    --text_encoder /path/to/mistral3.safetensors \
    --prompt "test prompt" \
    --save_path /tmp/test.png
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

1. **Cache files:** `*_f2.safetensors` → `*_f2d.safetensors` (re-cache required)
2. **Model versions:** `flux.2-*` → `dev`, `klein-*` (old names accepted via alias)
3. **Config keys:** `flux_kontext_no_resize_control` → `no_resize_control` (warning shown)
4. **Text encoder:** Klein models need Qwen3 checkpoints

---

## Done Criteria

All phases complete when:
- [ ] All 11 phases executed
- [ ] `python -m compileall -q src` passes
- [ ] Import checks pass
- [ ] Back-compat test passes
- [ ] Cache metadata verification passes
- [ ] Training smoke test completes 5 steps
- [ ] Inference test produces output image
