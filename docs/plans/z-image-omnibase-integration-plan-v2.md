# Z-Image OmniBase Integration Plan v2.1

**Version**: 2.1 (Final Clarifications)
**Created**: 2026-01-27
**Status**: ⏸️ PAUSED - Awaiting OmniBase Model Release
**Estimated Total Effort**: 10-14 hours (MVP), +4-6 hours (optional enhancements)

---

## Current Status (2026-01-28)

### Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Configuration & Detection | ✅ Complete | SigLIP2 loading, grid conversion, detection functions |
| Phase 2: Dataset & Caching | ✅ Complete | `save_latent_cache_z_image`, `has_omnibase_cache`, `zimage_cache_latents.py` OmniBase support |
| Phase 3: Model Architecture | ⏸️ Blocked | Requires OmniBase checkpoint with `siglip_embedder.*` weights |
| Phase 4: Training Integration | ⏸️ Blocked | Requires Phase 3 |
| Phase 5: Testing & Validation | ⏸️ Blocked | Requires Phases 3-4 |

### Blocker: OmniBase Model Not Yet Released

As of 2026-01-28, the Z-Image model family release status is:

| Model | HuggingFace | ModelScope | Status |
|-------|-------------|------------|--------|
| Z-Image (Base) | ✅ Released | ✅ Released | **Just released 2026-01-27** |
| Z-Image-Turbo | ✅ Released | ✅ Released | Available |
| **Z-Image-Omni-Base** | ❌ To be released | ❌ To be released | **BLOCKER** |
| Z-Image-Edit | ❌ To be released | ❌ To be released | Not needed for MVP |

**Phase 3-4 require the OmniBase checkpoint** which contains `siglip_embedder.*` weights for the SigLIP2 feature projection. Without this checkpoint, we cannot:
- Test model loading with OmniBase detection
- Implement `_forward_omni()` path (architecture may differ from technical report)
- Run end-to-end OmniBase training

### Phase 2 Verification Checklist (Run on CUDA Machine)

When OmniBase releases OR to verify caching pipeline works:

```bash
# 0) Preconditions
#    - Use a fresh cache_directory when changing control_directory/SigLIP settings
#    - Keep multiple_target=false for Z-Image caching

# 1) Static sanity
ruff check src/musubi_tuner/zimage_cache_latents.py src/musubi_tuner/zimage/zimage_utils.py src/musubi_tuner/dataset/image_video_dataset.py
python3 -m compileall -q src/musubi_tuner/zimage_cache_latents.py src/musubi_tuner/zimage src/musubi_tuner/dataset

# 2A) Standard caching (no control) - TESTABLE NOW
uv run python zimage_cache_latents.py --dataset_config <config_without_control> --vae <vae_path> --batch_size 1 --num_workers 1
# Expect: only latents_... keys in *_zi.safetensors

# 2B) Control present, no SigLIP - TESTABLE NOW
# (dataset has control_directory, run without --image_encoder)
# Expect: warning about missing --image_encoder
# Expect: latents_control_0_... but NO siglip_0_...

# 2C) Full OmniBase cache - TESTABLE NOW (features cached but no model to consume them)
uv run python zimage_cache_latents.py --dataset_config <config_with_control> --vae <vae_path> \
    --image_encoder google/siglip2-base-patch16-256 --batch_size 1 --num_workers 1
# Expect: latents_control_0_... AND siglip_0_... keys

# 3) Inspect cache file
python3 - <<'PY'
import glob
from safetensors import safe_open
p = sorted(glob.glob('PATH_TO_CACHE_DIR/*_zi.safetensors'))[0]
print('file:', p)
with safe_open(p, framework='pt') as f:
    print('metadata:', f.metadata())
    for k in sorted(f.keys()):
        t = f.get_tensor(k)
        print(k, tuple(t.shape), t.dtype)
PY
# Expected (full OmniBase mode):
#   latents_1xHxW_* with shape [C,H,W] (C=16 for Z-Image)
#   latents_control_0_1xHxW_* shape [C,H,W]
#   siglip_0_* shape like [16,16,1152] for patch16-256

# 4) Cache text encoder outputs
uv run python zimage_cache_text_encoder_outputs.py --dataset_config <config> --text_encoder <te_path> \
    --batch_size 1 --num_workers 1
# Confirm matching *_zi_te.safetensors for every *_zi.safetensors

# 5) Dataset-load smoke test (no model needed)
python3 - <<'PY'
import argparse
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_Z_IMAGE

cfg = 'PATH_TO_DATASET_TOML'
user_config = config_utils.load_user_config(cfg)
args = argparse.Namespace(debug_dataset=False)
bp = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_Z_IMAGE)
dg = config_utils.generate_dataset_group_by_blueprint(bp.dataset_group)
for ds in dg.datasets:
    for batch_key, batch in ds.retrieve_latent_cache_batches(1):
        print(f"batch_key={batch_key}, items={len(batch)}")
        for item in batch[:1]:
            print(f"  item_key={item.item_key}")
            print(f"  latent_cache_path={item.latent_cache_path}")
        break
PY
```

### Next Steps (When OmniBase Releases)

1. Download OmniBase checkpoint from HuggingFace/ModelScope
2. Verify checkpoint contains `siglip_embedder.*` keys
3. Compare architecture against our Phase 3 implementation plan
4. Implement Phase 3 (model architecture extensions)
5. Implement Phase 4 (training integration)
6. Full end-to-end testing

---

## Corrections from v1

This version addresses the following issues identified in v1:

| Issue | v1 Problem | v2 Fix |
|-------|------------|--------|
| AdaLN modulation | Incorrectly specified 6 values | Fixed: 4 values (scale_msa, gate_msa, scale_mlp, gate_mlp) |
| SigLIP CLS handling | Wrong grid calculation | Fixed: grid^2 + 1, drop CLS, reshape (grid, grid, C) |
| Missing file | Dataset saver not listed | Added: `image_video_dataset.py` modifications |
| Cache filename | Used `_zimage.safetensors` | Fixed: `_zi.safetensors` via `ARCHITECTURE_Z_IMAGE` |
| argparse footgun | `store_true` + `default=True` | Fixed: Auto-detect without flag |
| Model loading | No state dict detection | Added: Detect `siglip_embedder.*` keys |
| Training integration | Proposed new `train_step()` | Fixed: Extend `ZImageNetworkTrainer.call_dit` |
| OmniBase specifics | Hand-wavy dual-branch | Fixed: Explicit t_clean=1, RoPE coords, text approach |
| transformers version | Said >=4.51.0 | Fixed: Already >=4.56.1 in pyproject.toml |

**v2.1 Additional Clarifications:**

| Clarification | Details |
|---------------|---------|
| RoPE temporal coords | T_start = cap_seq_len + 1; ref at T_start, target at T_start + 1 |
| Noise mask location | Segment-level in trainer, per-token built by model |
| call_dit return | Must return `(model_pred, target)` tuple like standard path |
| MVP scope | **1 control image per target** (multi-control is follow-up) |
| SigLIP2 loading | Handle `subfolder="image_encoder"` layout |
| Model loading location | Detection in `zimage_model.py:load_zimage_model`, shape-based inference |
| Modulation coverage | noise_refiner + final_layer + main layers (all need per-token selection) |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [MVP Slice (Phases 1-4)](#2-mvp-slice-phases-1-4)
3. [Phase 1: Configuration & Detection](#3-phase-1-configuration--detection)
4. [Phase 2: Dataset & Caching Extensions](#4-phase-2-dataset--caching-extensions)
5. [Phase 3: Model Architecture Extensions](#5-phase-3-model-architecture-extensions)
6. [Phase 4: Training Integration](#6-phase-4-training-integration)
7. [Phase 5: Testing & Validation](#7-phase-5-testing--validation)
8. [Follow-Up Plan (Optional)](#8-follow-up-plan-optional)
9. [Risk Assessment](#9-risk-assessment)

---

## 1. Architecture Overview

### 1.1 OmniBase Dual-Branch Design

OmniBase uses **per-token modulation selection** to handle reference (clean) and target (noisy) images differently:

```
Input Sequence: [caption_tokens | ref_img_tokens | target_img_tokens]
                                 ↓
Noise Mask:     [       0       |       0        |         1        ]
                                 ↓
                         ┌──────┴──────┐
                         │             │
                    t_clean = 1    t_noisy = t
                         │             │
                         ↓             ↓
                   AdaLN(t_clean)  AdaLN(t_noisy)
                         │             │
                         └──────┬──────┘
                                ↓
                    select_per_token(noise_mask)
                                ↓
                         4 modulation values:
                    (scale_msa, gate_msa, scale_mlp, gate_mlp)
```

### 1.2 Key Design Decisions

**Time Conditioning:**
- Reference images: `t_clean = 1.0` (fully denoised state)
- Target image: `t_noisy = current_timestep` (varies during training)
- Selection via `noise_mask`: 0 = clean, 1 = noisy

**RoPE Coordinates:**
- Reference and target images share **same spatial RoPE** (H, W coordinates)
- **Temporal coordinate scheme** (avoids collision with caption):
  ```
  T_start = cap_seq_len + 1  # Image tokens start after caption
  Reference image: T = T_start
  Target image:    T = T_start + 1  # Unit offset from reference
  ```
- This ensures: (1) no collision with caption positions, (2) spatial alignment between ref/target, (3) temporal distinction for the model

**Text Conditioning:**
- **Single instruction embedding** per sample (not per-image)
- Instruction describes the edit operation (e.g., "change hair color to red")
- Caption tokens prepended to image sequence with shared RoPE temporal coord

**AdaLN Modulation (4 values, NOT 6):**
```python
# Blissful Tuner's actual implementation (zimage_model.py:269)
scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation[0](adaln_input).chunk(4, dim=-1)
```

**Modules Requiring Per-Token Modulation in OmniBase:**
```
noise_refiner     - modulation=True  → NEEDS per-token selection
context_refiner   - modulation=False → No timestep modulation (skip)
main layers       - modulation=True  → NEEDS per-token selection
final_layer       - has adaLN_modulation → NEEDS per-token selection (scale only)
```

Note: `final_layer.adaLN_modulation` produces only `scale` (not scale+gate), so its per-token handling is simpler.

### 1.3 Model Loading Detection

To maintain backward compatibility, detect OmniBase from state dict:

```python
def should_enable_omnibase(state_dict: dict) -> bool:
    """Detect OmniBase capability from checkpoint keys."""
    return any(k.startswith("siglip_embedder.") for k in state_dict.keys())
```

---

## 2. MVP Slice (Phases 1-4)

**Goal**: Enable OmniBase LoRA training with minimal changes, maximum compatibility.

**MVP Scope:**
1. Extend cache format with control latents + SigLIP features
2. Auto-detect OmniBase from cached control keys in dataset
3. Extend `call_dit` to handle OmniBase forward pass
4. Return `(model_pred, target)` tuple for loss computation (same contract as standard)

**MVP Constraint: 1 Control Image Per Target**
- The current dataset pipeline for `ARCHITECTURE_Z_IMAGE` enforces single control image
- Cache format supports `latents_control_0` only in MVP
- Multi-control (variable `latents_control_i` keys per sample) requires additional dataset pipeline work and is deferred to follow-up

**Explicitly Out of Scope (Follow-Up Plan):**
- Attention backend abstraction (FA2/FA3)
- OmniBase inference/generation
- Logit-normal timestep sampling
- Prompt enhancement module
- Multi-control images per target (requires dataset pipeline changes)

---

## 3. Phase 1: Configuration & Detection

**Effort**: 30-45 minutes
**Risk**: Low
**Files Modified**: 2

### 3.1 Changes to `zimage_config.py`

```python
# ADD: OmniBase/SigLIP2 configuration
DEFAULT_TRANSFORMER_SIGLIP_FEAT_DIM = 1152  # SigLIP2 hidden size

# ADD: OmniBase time conditioning
OMNIBASE_T_CLEAN = 1.0  # Reference images use t=1 (fully denoised)

# VERIFY existing values match official config (no changes needed if correct):
# DEFAULT_TRANSFORMER_DIM = 3840
# DEFAULT_TRANSFORMER_N_LAYERS = 30
# etc.
```

### 3.2 Changes to `zimage_utils.py`

```python
# ADD: SigLIP2 integration (transformers>=4.56.1 already required)
import math
from typing import Optional, Tuple, Any

try:
    from transformers import (
        Siglip2VisionModel,
        Siglip2Processor,
    )
    SIGLIP2_AVAILABLE = True
except ImportError:
    Siglip2VisionModel = None
    Siglip2Processor = None
    SIGLIP2_AVAILABLE = False


def load_siglip2_encoder(
    encoder_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load SigLIP2 vision encoder and processor.

    Args:
        encoder_path: Path to SigLIP2 checkpoint. Supports layouts:
            - Flat HF directory (config.json at root)
            - Subfolder layout (image_encoder/ subdirectory)
            - Direct model file path

    Returns:
        Tuple of (vision_model, processor) or (None, None) if unavailable
    """
    if not SIGLIP2_AVAILABLE:
        logger.warning("SigLIP2 not available in transformers. Skipping.")
        return None, None

    import os

    # Try common layouts
    paths_to_try = [
        encoder_path,                                    # Flat HF directory
        os.path.join(encoder_path, "image_encoder"),    # Subfolder layout
    ]

    for path in paths_to_try:
        if not os.path.exists(path):
            continue
        try:
            # Try loading processor (may not exist for all checkpoints)
            processor = None
            try:
                processor = Siglip2Processor.from_pretrained(path)
            except Exception:
                # Fallback: create processor from vision config
                from transformers import SiglipImageProcessor
                processor = SiglipImageProcessor.from_pretrained(path)

            vision_model = Siglip2VisionModel.from_pretrained(
                path,
                torch_dtype=dtype,
            ).to(device)
            vision_model.eval()
            logger.info(f"Loaded SigLIP2 encoder from {path}")
            return vision_model, processor
        except Exception as e:
            logger.debug(f"Failed to load from {path}: {e}")
            continue

    logger.error(f"Failed to load SigLIP2 from any path variant of {encoder_path}")
    return None, None


def siglip_last_hidden_to_grid(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Convert SigLIP2 last_hidden_state to spatial grid.

    SigLIP2 output shape: [num_tokens, C] where num_tokens = grid^2 + 1 (with CLS)
    or grid^2 (without CLS, depending on model variant).

    Returns: [grid_h, grid_w, C] spatial grid (CLS token dropped if present)
    """
    num_tokens, channels = last_hidden_state.shape

    # Case 1: Perfect square (no CLS token)
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size == num_tokens:
        return last_hidden_state.view(grid_size, grid_size, channels)

    # Case 2: grid^2 + 1 (CLS token at position 0)
    grid_size = int(math.sqrt(num_tokens - 1))
    if grid_size * grid_size == num_tokens - 1:
        # Drop CLS token (first token), reshape remaining
        patch_tokens = last_hidden_state[1:]  # [grid^2, C]
        return patch_tokens.view(grid_size, grid_size, channels)

    raise ValueError(
        f"Cannot reshape {num_tokens} tokens to square grid. "
        f"Expected grid^2 or grid^2+1 tokens."
    )


def should_enable_omnibase(state_dict: dict) -> bool:
    """
    Detect OmniBase capability from checkpoint state dict keys.

    Returns True if checkpoint contains SigLIP embedder weights.
    """
    return any(k.startswith("siglip_embedder.") for k in state_dict.keys())
```

### 3.3 Acceptance Criteria

- [ ] `SIGLIP2_AVAILABLE` correctly reflects transformers capability
- [ ] `load_siglip2_encoder` handles missing checkpoints gracefully
- [ ] `siglip_last_hidden_to_grid` correctly handles both CLS and no-CLS cases
- [ ] `should_enable_omnibase` detects OmniBase checkpoints

---

## 4. Phase 2: Dataset & Caching Extensions

**Effort**: 1.5-2 hours
**Risk**: Medium
**Files Modified**: 2

### 4.1 Changes to `image_video_dataset.py`

```python
# MODIFY: save_latent_cache_z_image to support OmniBase

def save_latent_cache_z_image(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latents: Optional[List[torch.Tensor]] = None,
    siglip_features: Optional[List[torch.Tensor]] = None,
):
    """
    Z-Image architecture cache saver.

    Standard mode (existing behavior):
        latent: [C, H, W] target image latent
        control_latents: None
        siglip_features: None

    OmniBase mode (new):
        latent: [C, H, W] target image latent
        control_latents: List of [C, H, W] control image latents
        siglip_features: List of [H_sig, W_sig, D_sig] SigLIP features
    """
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    C, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)

    # Base cache dict (unchanged for backward compatibility)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    # OmniBase additions (new keys only)
    if control_latents is not None:
        for i, ctrl in enumerate(control_latents):
            assert ctrl.dim() == 3, f"control_latent[{i}] should be 3D"
            ctrl_C, ctrl_H, ctrl_W = ctrl.shape
            ctrl_dtype = dtype_to_str(ctrl.dtype)
            sd[f"latents_control_{i}_{F}x{ctrl_H}x{ctrl_W}_{ctrl_dtype}"] = ctrl.detach().cpu().contiguous()

    if siglip_features is not None:
        for i, sig in enumerate(siglip_features):
            assert sig.dim() == 3, f"siglip_features[{i}] should be 3D [H, W, C]"
            sig_dtype = dtype_to_str(sig.dtype)
            sd[f"siglip_{i}_{sig_dtype}"] = sig.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


# ADD: Helper to detect OmniBase from cached file
def has_omnibase_cache(cache_path: str) -> bool:
    """Check if a Z-Image cache file contains OmniBase data."""
    try:
        with safetensors.safe_open(cache_path, framework="pt") as f:
            keys = f.keys()
            return any(k.startswith("latents_control_") or k.startswith("siglip_") for k in keys)
    except Exception:
        return False
```

### 4.2 Changes to `zimage_cache_latents.py`

```python
# ADD: OmniBase caching support

def setup_parser():
    parser = argparse.ArgumentParser()
    # ... existing arguments ...

    # NEW: OmniBase arguments
    parser.add_argument(
        "--image_encoder",
        type=str,
        default=None,
        help="Path to SigLIP2 encoder for OmniBase I2I caching. "
             "If not provided, control images cached without SigLIP features.",
    )
    # NOTE: No --i2v or --auto_i2v flags.
    # OmniBase mode is auto-detected from dataset config (control_directory present).
    return parser


def encode_and_save_batch(
    # ... existing parameters ...
    control_images: Optional[List[Image.Image]] = None,
    vision_model=None,
    processor=None,
):
    """
    Encode and cache latents for Z-Image.

    OmniBase mode activated when control_images is not None.
    """
    # Encode target image (existing logic)
    target_latent = encode_image_to_latent(vae, image, ...)

    control_latents = None
    siglip_features = None

    # OmniBase: encode control images
    if control_images is not None:
        control_latents = []
        siglip_features = [] if vision_model is not None else None

        for ctrl_img in control_images:
            # VAE encode control image
            ctrl_latent = encode_image_to_latent(vae, ctrl_img, ...)
            control_latents.append(ctrl_latent)

            # SigLIP feature extraction (if encoder available)
            if vision_model is not None and processor is not None:
                sig_feat = extract_siglip_features(vision_model, processor, ctrl_img, device)
                siglip_features.append(sig_feat)

    # Save cache
    save_latent_cache_z_image(
        item_info=item,
        latent=target_latent,
        control_latents=control_latents,
        siglip_features=siglip_features,
    )


def extract_siglip_features(
    vision_model,
    processor,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Extract SigLIP2 features and convert to spatial grid."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vision_model(**inputs)
        last_hidden = outputs.last_hidden_state[0]  # [num_tokens, C]

    # Convert to spatial grid
    return siglip_last_hidden_to_grid(last_hidden)
```

### 4.3 Dataset Configuration

```toml
# Standard Z-Image (unchanged)
[[datasets]]
resolution = [1024, 1024]
batch_size = 1

[[datasets.subsets]]
image_directory = "/path/to/images"
caption_extension = ".txt"


# OmniBase Z-Image (new: control_directory)
[[datasets]]
resolution = [1024, 1024]
batch_size = 1

[[datasets.subsets]]
image_directory = "/path/to/target_images"
control_directory = "/path/to/control_images"  # NEW: Triggers OmniBase mode
caption_extension = ".txt"
# Filenames must match: target/foo.png ↔ control/foo.png
```

### 4.4 Cache File Format

```
# Standard mode (unchanged)
{basename}_{w}x{h}_zi.safetensors:
├── latents_1x{H}x{W}_{dtype}     # Target latent

# OmniBase mode - MVP (single control)
{basename}_{w}x{h}_zi.safetensors:
├── latents_1x{H}x{W}_{dtype}              # Target latent
├── latents_control_0_1x{H}x{W}_{dtype}    # Control image latent
└── siglip_0_{dtype}                        # SigLIP features [H_sig, W_sig, 1152]

# OmniBase mode - Future (multi-control, OUT OF MVP SCOPE)
# Would require dataset pipeline changes to guarantee consistent key sets
# {basename}_{w}x{h}_zi.safetensors:
# ├── latents_control_0_...
# ├── latents_control_1_...  # Variable count per sample
# └── ...
```

### 4.5 Acceptance Criteria

- [ ] Standard caching produces identical files (regression test)
- [ ] OmniBase caching adds new keys without breaking existing keys
- [ ] `has_omnibase_cache()` correctly detects OmniBase caches
- [ ] Missing control images produce clear error messages

---

## 5. Phase 3: Model Architecture Extensions

**Effort**: 2-3 hours
**Risk**: High (core model changes)
**Files Modified**: 1

### 5.1 Design Principles

1. **Lazy initialization**: OmniBase modules only created if `siglip_feat_dim` provided
2. **State dict detection**: Auto-detect OmniBase from checkpoint keys
3. **Separate forward paths**: `_forward_standard` unchanged, `_forward_omni` new
4. **4-value modulation**: Maintain compatibility with existing AdaLN

### 5.2 Changes to `zimage_model.py`

```python
# ADD: Helper function for per-token modulation selection
def select_per_token(
    noisy_mod: torch.Tensor,
    clean_mod: torch.Tensor,
    noise_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Select modulation values per token based on noise mask.

    Args:
        noisy_mod: [B, 4*D] modulation for noisy tokens (t=t_current)
        clean_mod: [B, 4*D] modulation for clean tokens (t=1.0)
        noise_mask: [B, N] where 1=noisy (target), 0=clean (reference)

    Returns:
        [B, N, 4*D] per-token modulation
    """
    B, N = noise_mask.shape
    D = noisy_mod.shape[-1]

    # Expand to sequence length
    noisy_expanded = noisy_mod.unsqueeze(1).expand(B, N, D)  # [B, N, 4*D]
    clean_expanded = clean_mod.unsqueeze(1).expand(B, N, D)  # [B, N, 4*D]

    # Select per token: mask=1 → noisy, mask=0 → clean
    mask = noise_mask.unsqueeze(-1).expand(B, N, D)  # [B, N, 4*D]
    return torch.where(mask == 1, noisy_expanded, clean_expanded)


# MODIFY: ZImageTransformer2DModel.__init__
class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        # ... existing parameters ...
        siglip_feat_dim: Optional[int] = None,  # NEW: None disables OmniBase
    ):
        super().__init__()
        # ... existing initialization ...

        # NEW: OmniBase components (lazy init)
        self.siglip_feat_dim = siglip_feat_dim
        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=self.norm_eps),
                nn.Linear(siglip_feat_dim, self.dim, bias=False),
            )
            # 2 refiner layers for SigLIP features
            self.siglip_refiner = nn.ModuleList([
                ZImageTransformerBlock(
                    dim=self.dim,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    norm_eps=self.norm_eps,
                    qk_norm=self.qk_norm,
                    modulation=False,  # No timestep modulation for refiner
                )
                for _ in range(2)
            ])
            self.siglip_pad_token = nn.Parameter(torch.zeros(1, self.dim))
            nn.init.normal_(self.siglip_pad_token, std=0.02)
            logger.info(f"OmniBase enabled: siglip_feat_dim={siglip_feat_dim}")


# MODIFY: ZImageTransformerBlock._forward for OmniBase support
class ZImageTransformerBlock(nn.Module):
    def _forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        # NEW: OmniBase per-token modulation
        per_token_mod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with optional per-token modulation (OmniBase).

        Standard mode: adaln_input provided, per_token_mod is None
        OmniBase mode: per_token_mod provided [B, N, 4*D], adaln_input ignored
        """
        if self.modulation:
            if per_token_mod is not None:
                # OmniBase: per-token modulation already computed
                # Shape: [B, N, 4*D] → chunk into 4 values
                scale_msa, gate_msa, scale_mlp, gate_mlp = per_token_mod.chunk(4, dim=-1)
            else:
                # Standard: compute modulation from adaln_input
                assert adaln_input is not None
                mod = self.adaLN_modulation[0](adaln_input)  # [B, 4*D]
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=-1)

            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        else:
            scale_msa = scale_mlp = 1.0
            gate_msa = gate_mlp = 1.0

        # Rest of forward unchanged...
        # (attention with scale_msa, gate_msa)
        # (FFN with scale_mlp, gate_mlp)


# ADD: OmniBase forward path
class ZImageTransformer2DModel(nn.Module):
    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        t: torch.Tensor,
        cap_feats: Union[torch.Tensor, List[torch.Tensor]],
        cap_mask: Optional[torch.Tensor] = None,
        # NEW: OmniBase parameters (MVP: single control)
        control_latent: Optional[torch.Tensor] = None,
        siglip_feat: Optional[torch.Tensor] = None,
        segment_noise_mask: Optional[torch.Tensor] = None,  # Segment-level, NOT per-token
    ) -> torch.Tensor:
        """
        Forward supporting standard and OmniBase modes.

        Standard mode (backward compatible):
            x: [B, C, F, H, W] target latent
            control_latent, siglip_feat, segment_noise_mask: None

        OmniBase mode (MVP: single control):
            x: [B, C, F, H, W] target latent (noised by trainer)
            control_latent: [B, C, F, H, W] single control latent (clean)
            siglip_feat: [B, H_sig, W_sig, D_sig] SigLIP features for control
            segment_noise_mask: [2] tensor e.g. [0, 1] meaning [control=clean, target=noisy]
                Model expands this to per-token mask based on sequence construction.
        """
        if control_latent is not None:
            return self._forward_omni(x, t, cap_feats, cap_mask, control_latent, siglip_feat, segment_noise_mask)
        else:
            return self._forward_standard(x, t, cap_feats, cap_mask)

    def _forward_standard(self, x, t, cap_feats, cap_mask):
        """Original forward - NO CHANGES."""
        # ... existing implementation exactly as-is ...
        pass

    def _forward_omni(self, target_x, t, cap_feats, cap_mask, control_latent, siglip_feat, segment_noise_mask):
        """
        OmniBase forward with dual-branch time conditioning.

        Time conditioning:
            - Reference images: t_clean = 1.0 (fully denoised)
            - Target image: t_noisy = t (current timestep)

        RoPE coordinates (avoids caption collision):
            T_start = cap_seq_len + 1
            Control image: T = T_start
            Target image: T = T_start + 1
        """
        if self.siglip_feat_dim is None:
            raise RuntimeError("OmniBase forward called but model was not initialized with siglip_feat_dim")

        # Compute time embeddings for both branches
        t_noisy = t  # Current training timestep
        t_clean = torch.ones_like(t)  # Reference images are "fully denoised"

        adaln_noisy = self.t_embedder(t_noisy * self.t_scale)  # [B, adaln_dim]
        adaln_clean = self.t_embedder(t_clean * self.t_scale)  # [B, adaln_dim]

        # Build unified sequence and compute per-token noise mask
        # The model constructs per-token mask here, NOT the trainer
        x, per_token_noise_mask, freqs, attn_mask = self._build_omni_sequence(
            target_x, control_latent, siglip_feat, cap_feats, cap_mask, segment_noise_mask
        )

        # Apply noise refiner (modulation=True, needs per-token selection)
        for layer in self.noise_refiner:
            mod_noisy = layer.adaLN_modulation[0](adaln_noisy)
            mod_clean = layer.adaLN_modulation[0](adaln_clean)
            per_token_mod = select_per_token(mod_noisy, mod_clean, per_token_noise_mask)
            x = layer(x, freqs, per_token_mod=per_token_mod, attn_params=...)

        # Apply context refiner (modulation=False, no per-token needed)
        for layer in self.context_refiner:
            x = layer(x, freqs, attn_params=...)

        # Apply main layers (modulation=True, needs per-token selection)
        for layer in self.layers:
            mod_noisy = layer.adaLN_modulation[0](adaln_noisy)
            mod_clean = layer.adaLN_modulation[0](adaln_clean)
            per_token_mod = select_per_token(mod_noisy, mod_clean, per_token_noise_mask)
            x = layer(x, freqs, per_token_mod=per_token_mod, mask=attn_mask)

        # Apply final layer (has adaLN_modulation, needs per-token selection)
        # FinalLayer uses scale only (not scale+gate), so simpler handling
        scale_noisy = 1.0 + self.final_layer.adaLN_modulation(adaln_noisy)
        scale_clean = 1.0 + self.final_layer.adaLN_modulation(adaln_clean)
        per_token_scale = select_per_token(scale_noisy, scale_clean, per_token_noise_mask)
        x = self.final_layer.norm_final(x) * per_token_scale.unsqueeze(1)
        x = self.final_layer.linear(x)

        # Extract only target image prediction
        target_pred = self._extract_target_prediction(x, target_seq_start, target_seq_end)
        return target_pred

    def _build_omni_sequence(self, target_x, control_latent, siglip_feat, cap_feats, cap_mask, segment_noise_mask):
        """
        Build unified sequence for OmniBase and compute per-token noise mask.

        Sequence order: [caption | siglip | control_img | target_img]

        Returns:
            x: [B, total_seq_len, dim] unified sequence
            per_token_noise_mask: [B, total_seq_len] where 0=clean, 1=noisy
            freqs: RoPE frequencies
            attn_mask: Attention mask (if needed)
        """
        # ... sequence construction with proper RoPE T coordinates ...
        # T_start = cap_seq_len + 1
        # Control tokens: T = T_start (all same temporal coord)
        # Target tokens: T = T_start + 1 (unit offset)

        # Build per-token mask from segment mask
        # segment_noise_mask = [0, 1] means [control=clean, target=noisy]
        # Expand to actual token counts:
        # - caption: 0 (always clean)
        # - siglip: 0 (always clean)
        # - control_img tokens: segment_noise_mask[0] = 0
        # - target_img tokens: segment_noise_mask[1] = 1
        pass
```

### 5.3 Model Loading with OmniBase Detection

**Location**: `src/musubi_tuner/zimage/zimage_model.py:load_zimage_model` (line 757)

```python
# MODIFY: load_zimage_model in zimage_model.py (NOT zimage_utils.py)

def load_zimage_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[List[float]] = None,
    disable_numpy_memmap: bool = False,
    use_16bit_for_attention: bool = False,
) -> ZImageTransformer2DModel:
    """Load Z-Image model with auto OmniBase detection."""

    # Peek at state dict keys to detect OmniBase (before full load)
    from safetensors import safe_open
    with safe_open(dit_path, framework="pt") as f:
        keys = list(f.keys())

    # Shape-based inference of siglip_feat_dim
    siglip_feat_dim = None
    siglip_key = next((k for k in keys if k.startswith("siglip_embedder.")), None)
    if siglip_key is not None:
        # Infer dimension from weight shape (siglip_embedder.1.weight is [dim, siglip_feat_dim])
        with safe_open(dit_path, framework="pt") as f:
            if "siglip_embedder.1.weight" in keys:
                weight = f.get_tensor("siglip_embedder.1.weight")
                siglip_feat_dim = weight.shape[1]  # Input dimension
                logger.info(f"Detected OmniBase checkpoint, siglip_feat_dim={siglip_feat_dim}")
            else:
                # Fallback to default if shape not determinable
                siglip_feat_dim = zimage_config.DEFAULT_TRANSFORMER_SIGLIP_FEAT_DIM
                logger.info(f"Detected OmniBase checkpoint, using default siglip_feat_dim={siglip_feat_dim}")

    # Create model with detected architecture
    model = create_model(
        attn_mode, split_attn, dit_weight_dtype,
        use_16bit_for_attention=use_16bit_for_attention,
        siglip_feat_dim=siglip_feat_dim,  # NEW parameter
    )

    # ... rest of existing loading logic unchanged ...
```

**Also modify `create_model`** to accept `siglip_feat_dim`:
```python
def create_model(
    attn_mode: str,
    split_attn: bool,
    dtype: Optional[torch.dtype],
    use_16bit_for_attention: bool = False,
    siglip_feat_dim: Optional[int] = None,  # NEW
) -> ZImageTransformer2DModel:
    with init_empty_weights():
        model = ZImageTransformer2DModel(
            # ... existing params ...
            siglip_feat_dim=siglip_feat_dim,  # Pass through
        )
    return model
```

### 5.4 Acceptance Criteria

- [ ] Standard checkpoints load with `siglip_feat_dim=None` (no OmniBase modules)
- [ ] OmniBase checkpoints auto-detected and load with SigLIP modules
- [ ] `_forward_standard` produces identical output to before
- [ ] `_forward_omni` correctly applies per-token modulation
- [ ] Existing LoRAs work unchanged

---

## 6. Phase 4: Training Integration

**Effort**: 2-3 hours
**Risk**: Medium
**Files Modified**: 1

### 6.1 Design: Extend `call_dit`, Not New Function

Training integration happens in `ZImageNetworkTrainer.call_dit` (line 258), not a new function.

### 6.2 Changes to `zimage_train_network.py`

```python
class ZImageNetworkTrainer(NetworkTrainer):

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        model: zimage_model.ZImageTransformer2DModel = accelerator.unwrap_model(transformer)
        bsize = latents.shape[0]

        # Detect OmniBase mode from cached control latents
        omnibase_mode = "latents_control_0" in batch or any(
            k.startswith("latents_control_") for k in batch.keys()
        )

        if omnibase_mode:
            return self._call_dit_omnibase(
                args, accelerator, model, latents, batch, noise,
                noisy_model_input, timesteps, network_dtype
            )
        else:
            # Existing standard implementation (unchanged)
            return self._call_dit_standard(
                args, accelerator, model, latents, batch, noise,
                noisy_model_input, timesteps, network_dtype
            )

    def _call_dit_standard(self, args, accelerator, model, latents, batch, noise,
                           noisy_model_input, timesteps, network_dtype):
        """Original call_dit implementation - MOVE existing code here unchanged."""
        # ... existing implementation exactly as-is ...
        pass

    def _call_dit_omnibase(self, args, accelerator, model, latents, batch, noise,
                           noisy_model_input, timesteps, network_dtype):
        """
        OmniBase training: reference images stay clean, target gets noised.

        Key differences from standard:
        1. Control latents loaded from cache (not noised)
        2. SigLIP features loaded from cache
        3. Segment-level mask passed to model (model builds per-token mask)
        4. Returns (model_pred, target) tuple like standard path

        MVP: Single control image (latents_control_0 only)
        """
        bsize = latents.shape[0]

        # Load control latent (clean, no noise) - MVP: single control only
        control_latent = None
        siglip_feat = None

        if "latents_control_0" in batch:
            ctrl = batch["latents_control_0"].to(accelerator.device, dtype=network_dtype)
            # Add frame dimension if needed
            if ctrl.dim() == 4:  # [B, C, H, W]
                ctrl = ctrl.unsqueeze(2)  # [B, C, 1, H, W]
            control_latent = ctrl

            # Load SigLIP features if present
            if "siglip_0" in batch:
                siglip_feat = batch["siglip_0"].to(accelerator.device, dtype=network_dtype)

        # Target latent (noised as usual)
        noisy_target = noisy_model_input.unsqueeze(2)  # [B, C, 1, H, W]

        # Segment-level mask: [0, 1] meaning [control=clean, target=noisy]
        # Model will expand this to per-token mask when building unified sequence
        segment_mask = torch.tensor([0, 1], device=accelerator.device)  # NOT per-token!

        # Caption features (same as standard)
        llm_embed = batch["llm_embed"]

        # Call model in OmniBase mode
        model_pred = model(
            x=noisy_target,
            t=timesteps,
            cap_feats=llm_embed,
            cap_mask=None,
            control_latent=control_latent,       # Single control (MVP)
            siglip_feat=siglip_feat,             # Single siglip (MVP)
            segment_noise_mask=segment_mask,     # Segment-level, NOT per-token
        )

        # Compute target for loss (same as standard path)
        # target = latents - noise (velocity target for flow matching)
        target = latents - noise

        # Return (model_pred, target) - SAME CONTRACT as standard path
        return model_pred, target
```

### 6.3 Training Arguments

```python
# No new arguments needed - OmniBase auto-detected from cache
# However, add validation:

def validate_omnibase_args(args, dataset):
    """Warn if OmniBase data detected but model might not support it."""
    has_control_data = any(has_omnibase_cache(item.latent_cache_path) for item in dataset)
    if has_control_data:
        logger.info("OmniBase data detected in cache. Ensure model supports OmniBase or use --omnibase_model flag.")
```

### 6.4 Acceptance Criteria

- [ ] Standard training unchanged (no OmniBase keys → standard path)
- [ ] OmniBase detected from cache keys automatically
- [ ] Control latents stay clean (no noise added)
- [ ] Target latent noised normally
- [ ] Noise mask correctly identifies target tokens
- [ ] Loss computed only on target prediction

---

## 7. Phase 5: Testing & Validation

**Effort**: 3-4 hours
**Risk**: N/A (validation)

### 7.1 Regression Test Matrix

| Test | Standard | OmniBase |
|------|----------|----------|
| Config loading | ✓ | ✓ |
| Model init (no siglip) | ✓ | N/A |
| Model init (with siglip) | N/A | ✓ |
| State dict detection | ✓ | ✓ |
| Checkpoint loading | ✓ | ✓ |
| Forward pass shapes | ✓ | ✓ |
| Backward pass | ✓ | ✓ |
| Caching (standard) | ✓ | N/A |
| Caching (OmniBase) | N/A | ✓ |
| Training step | ✓ | ✓ |
| LoRA application | ✓ | ✓ |

### 7.2 Test Commands

```bash
# Test 1: Standard Z-Image regression
python zimage_cache_latents.py --dataset_config test_standard.toml --vae /path/to/vae
python zimage_train_network.py --dit /path/to/turbo_dit --dataset_config test_standard.toml \
    --network_module networks.lora_zimage --network_dim 32

# Test 2: OmniBase caching
python zimage_cache_latents.py --dataset_config test_omnibase.toml --vae /path/to/vae \
    --image_encoder /path/to/siglip2

# Test 3: OmniBase training
python zimage_train_network.py --dit /path/to/omnibase_dit --dataset_config test_omnibase.toml \
    --network_module networks.lora_zimage --network_dim 32
```

### 7.3 Validation Checkpoints

- [ ] Phase 1: SigLIP utils work, detection correct
- [ ] Phase 2: Caching produces correct files, backward compatible
- [ ] Phase 3: Model loads both checkpoint types correctly
- [ ] Phase 4: Training runs end-to-end for both modes
- [ ] Phase 5: All regression tests pass

---

## 8. Follow-Up Plan (Optional)

These enhancements are **out of scope for MVP** and should be separate PRs:

### 8.1 Attention Backend Abstraction

Add FA2/FA3 support with registry pattern (from official Z-Image).

**Estimated effort**: 4-6 hours
**Benefit**: ~2x inference speedup

### 8.2 OmniBase Inference

Implement generation with control images.

**Estimated effort**: 6-8 hours
**Benefit**: Complete OmniBase feature set

### 8.3 Logit-Normal Timestep Sampling

Match official training methodology.

**Estimated effort**: 1-2 hours
**Benefit**: Potentially better training efficiency

### 8.4 Prompt Enhancement Module

Integrate VLM-based prompt enhancement.

**Estimated effort**: 4-6 hours
**Benefit**: Better prompt understanding

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing LoRAs | Low | **Critical** | State dict detection, separate forward paths |
| Memory increase (OmniBase disabled) | Low | Medium | Lazy module init |
| Incorrect modulation (4 vs 6) | **Fixed in v2** | Critical | Use existing 4-value pattern |
| Cache format incompatibility | Low | Medium | New keys only, existing preserved |
| Training instability | Medium | Medium | Match official t_clean=1 approach |

### 9.1 Critical Invariants

1. **AdaLN always produces 4 values**: `scale_msa, gate_msa, scale_mlp, gate_mlp`
2. **siglip_feat_dim=None**: No OmniBase modules created, no memory overhead
3. **State dict detection**: Must correctly identify Turbo vs OmniBase checkpoints
4. **Existing cache files**: Must remain loadable and functional

---

## Appendix: File Change Summary

| File | Location | Changes |
|------|----------|---------|
| `zimage_config.py` | `src/musubi_tuner/zimage/` | +3 lines (siglip dim, t_clean) |
| `zimage_utils.py` | `src/musubi_tuner/zimage/` | +80 lines (SigLIP2 loading, detection) |
| `zimage_model.py` | `src/musubi_tuner/zimage/` | +200 lines (OmniBase forward, select_per_token, create_model param) |
| `image_video_dataset.py` | `src/musubi_tuner/dataset/` | +40 lines (extended saver, detection helper) |
| `zimage_cache_latents.py` | `src/musubi_tuner/` | +60 lines (control image caching) |
| `zimage_train_network.py` | `src/musubi_tuner/` | +60 lines (OmniBase call_dit path) |
| **Total** | | **~440 lines** |

---

*Plan v2.1 - Corrected based on actual codebase review and expert feedback. Ready for implementation.*
