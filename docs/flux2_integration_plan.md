# FLUX.2 Integration Plan for Blissful Tuner

## Overview

This document outlines the plan to integrate FLUX.2 (DEV and klein-9B) training support from the [scenario-labs/musubi-tuner flux2 branch](https://github.com/scenario-labs/musubi-tuner/tree/flux2) into Blissful Tuner.

**Status**: Work in Progress
**Branch**: `flux2-integration`
**Worktree**: `/Users/dustin/blissful-tuner-flux2`

---

## Source Analysis

### Files from scenario-labs/flux2 Branch

| File | Size | Description |
|------|------|-------------|
| `src/musubi_tuner/flux_2/flux2_models.py` | 38KB | Core model definitions (Flux2, DoubleStreamBlock, SingleStreamBlock, AutoEncoder) |
| `src/musubi_tuner/flux_2/flux2_utils.py` | 21KB | Loading utilities, text encoding, scheduling |
| `src/musubi_tuner/networks/lora_flux_2.py` | 1.7KB | LoRA network module |
| `src/musubi_tuner/flux_2_train_network.py` | 12KB | Training loop extending NetworkTrainer |
| `src/musubi_tuner/flux_2_cache_latents.py` | ~90B | Thin wrapper script |
| `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py` | ~100B | Thin wrapper script |
| `src/musubi_tuner/flux_2_generate_image.py` | ~90B | Thin wrapper script |

### Modified Files Requiring Merge

| File | Changes |
|------|---------|
| `src/musubi_tuner/dataset/image_video_dataset.py` | Added FLUX.2 architecture constants, cache functions, resolution steps |

---

## Integration Steps

### Phase 1: Core Module Integration (Additive)

These files are purely additive and can be copied directly:

```
[ ] Copy flux_2/ directory to src/musubi_tuner/
    - flux2_models.py
    - flux2_utils.py

[ ] Copy flux_2_*.py scripts to src/musubi_tuner/
    - flux_2_train_network.py
    - flux_2_cache_latents.py
    - flux_2_cache_text_encoder_outputs.py
    - flux_2_generate_image.py

[ ] Copy root wrapper scripts to project root
    - flux_2_train_network.py
    - flux_2_cache_latents.py
    - flux_2_cache_text_encoder_outputs.py
    - flux_2_generate_image.py

[ ] Copy networks/lora_flux_2.py to src/musubi_tuner/networks/
```

### Phase 2: Dataset Module Merge (Requires Careful Merging)

The `image_video_dataset.py` file has diverged between branches:

**Blissful Tuner Changes to Preserve:**
- BlissfulLogger integration (line 27-29)
- Duplicate removal in `glob_images()` (line 98-100)
- Mask-weighted loss support
- Alpha channel mask support
- Any other Blissful-specific enhancements

**FLUX.2 Changes to Integrate:**
```python
# Add at line ~78 (after other ARCHITECTURE_* constants)
ARCHITECTURE_FLUX_2 = "f2"
ARCHITECTURE_FLUX_2_FULL = "flux_2"

# Add RESOLUTION_STEPS_FLUX_2 constant (similar to other architectures)
RESOLUTION_STEPS_FLUX_2 = 64  # Based on VAE downscaling

# Add to RESOLUTION_STEPS dict (~line 608)
ARCHITECTURE_FLUX_2: RESOLUTION_STEPS_FLUX_2,

# Add save_latent_cache_flux_2 function (~line 322)
def save_latent_cache_flux_2(item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]]):
    ...

# Add save_text_encoder_output_cache_flux_2 function (~line 503)
def save_text_encoder_output_cache_flux_2(item_info: ItemInfo, m3_vec: torch.Tensor):
    ...

# Add ARCHITECTURE_FLUX_2 case in dataset loading (~line 1817)
elif self.architecture == ARCHITECTURE_FLUX_2:
    ...
```

### Phase 3: Blissful Tuner Enhancements

After basic integration, apply Blissful Tuner improvements:

```
[ ] Add BlissfulLogger to flux_2 modules
    - Replace standard logging with BlissfulLogger
    - Add rich console output

[ ] Add latent preview support
    - Integrate latent_preview.py into flux_2_generate_image.py

[ ] Add guidance enhancements
    - CFGZero* support
    - NAG (Normalized Attention Guidance)
    - Perpendicular negative guidance

[ ] Add prompt management
    - Wildcard support
    - Prompt weighting

[ ] Add mask-weighted loss support
    - Integrate mask handling into flux_2 cache/training
    - Add alpha channel mask support
```

### Phase 4: Testing & Validation

```
[ ] Latent caching test
    - Cache latents for test images
    - Verify cache file format matches expected structure

[ ] Text encoder caching test
    - Test with both Mistral3 (DEV) and Qwen3 (klein)
    - Verify multi-layer extraction (layers 10, 20, 30)

[ ] Training test
    - Run short training (10 steps) to verify forward/backward pass
    - Check memory usage with block swapping

[ ] Inference test
    - Generate test images with trained LoRA
    - Verify quality and prompt adherence
```

---

## Technical Details

### Architecture Constants

```python
# To add to image_video_dataset.py
ARCHITECTURE_FLUX_2 = "f2"
ARCHITECTURE_FLUX_2_FULL = "flux_2"

# Resolution step based on VAE
# VAE has 4 down blocks with stride 2 each = 16x downscale
# Plus 2x2 patchification in transformer
# Total: 16 * 2 = 32 for spatial, but latent channels = 32
RESOLUTION_STEPS_FLUX_2 = 32  # Verify during testing
```

### LoRA Target Modules

```python
# From lora_flux_2.py
FLUX_2_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]

# Excluded patterns (modulators and norms)
exclude_patterns = [
    r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*",
    r".*(norm).*"
]
```

### Model Variants

| Variant | Params | Double Blocks | Single Blocks | Heads | Text Encoder |
|---------|--------|---------------|---------------|-------|--------------|
| DEV | ~60B | 8 | 48 | 48 | Mistral3/Pixtral |
| klein-9B | ~9B | 8 | 24 | 32 | Qwen3 |

### Text Encoding

FLUX.2 uses VLM-based text encoders with multi-layer extraction:

```python
OUTPUT_LAYERS = [10, 20, 30]
MAX_LENGTH = 512

# Embeddings from layers 10, 20, 30 are concatenated
# DEV: 3 * 5120 = 15360 joint_attention_dim
# klein: 3 * 4096 = 12288 joint_attention_dim
```

---

## File Structure After Integration

```
blissful-tuner/
├── flux_2_cache_latents.py          # Root wrapper
├── flux_2_cache_text_encoder_outputs.py
├── flux_2_generate_image.py
├── flux_2_train_network.py
├── src/
│   └── musubi_tuner/
│       ├── flux_2/                   # NEW: FLUX.2 module
│       │   ├── __init__.py
│       │   ├── flux2_models.py       # Model definitions
│       │   └── flux2_utils.py        # Utilities
│       ├── flux_2_cache_latents.py   # NEW: Cache script
│       ├── flux_2_cache_text_encoder_outputs.py
│       ├── flux_2_generate_image.py
│       ├── flux_2_train_network.py   # NEW: Training loop
│       ├── dataset/
│       │   └── image_video_dataset.py # MODIFIED: Add FLUX.2 support
│       └── networks/
│           └── lora_flux_2.py        # NEW: LoRA module
└── docs/
    ├── flux2_architecture.md         # Architecture reference
    └── flux2_integration_plan.md     # This document
```

---

## Git Workflow

### Current Setup

```bash
# Main repo
cd /Users/dustin/blissful-tuner
git remote -v
# origin    -> blissful-tuner fork
# scenario-labs -> https://github.com/scenario-labs/musubi-tuner.git

# Worktree for development
cd /Users/dustin/blissful-tuner-flux2
git branch
# * flux2-integration (tracking scenario-labs/flux2)
```

### Development Workflow

```bash
# Work in worktree
cd /Users/dustin/blissful-tuner-flux2

# Make changes and commit
git add .
git commit -m "feat(flux2): description"

# Sync with scenario-labs if needed
git fetch scenario-labs flux2
git rebase scenario-labs/flux2

# When ready to merge into blissful-tuner main
cd /Users/dustin/blissful-tuner
git merge flux2-integration --no-ff
```

### Keeping Upstream Sync

```bash
# Fetch latest from scenario-labs
git fetch scenario-labs flux2

# In worktree, rebase if needed
cd /Users/dustin/blissful-tuner-flux2
git rebase scenario-labs/flux2
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dataset module merge conflicts | Medium | High | Careful line-by-line merge, preserve Blissful features |
| Breaking existing architectures | Low | High | Comprehensive testing of WAN, HV, etc. after merge |
| Text encoder compatibility | Medium | Medium | Test with actual Mistral3/Qwen3 models |
| Memory issues with large models | Medium | Medium | Verify block swapping works correctly |

---

## Dependencies

### Python Packages

FLUX.2 may require:
- `transformers >= 4.56.0` (for Qwen3/Mistral3 support)
- `diffusers >= 0.37.0` (for AutoencoderKLFlux2)
- `einops` (already present)
- `torch >= 2.0` (already present)

### Model Weights

| Model | Source | Notes |
|-------|--------|-------|
| FLUX.2-DEV | BFL | ~60B params, Mistral3 text encoder |
| FLUX.2-klein-9B | BFL | ~9B params, Qwen3 text encoder |
| VAE | BFL | AutoencoderKLFlux2, 32 latent channels |

---

## Next Steps

1. **Immediate**: Cherry-pick FLUX.2-specific files into worktree with Blissful enhancements
2. **Short-term**: Merge dataset module changes preserving Blissful features
3. **Medium-term**: Add Blissful-specific enhancements (latent preview, guidance)
4. **Long-term**: Full testing and documentation

---

## References

- [FLUX.2 Architecture Documentation](./flux2_architecture.md)
- [scenario-labs/musubi-tuner flux2 branch](https://github.com/scenario-labs/musubi-tuner/tree/flux2)
- [Black Forest Labs FLUX.2 Announcement](https://bfl.ai)
- [Blissful Tuner Main Documentation](../README.md)
