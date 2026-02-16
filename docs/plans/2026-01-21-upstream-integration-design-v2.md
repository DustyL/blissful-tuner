# Upstream Musubi-Tuner Integration Design (v2)

**Date**: 2026-01-21
**Version**: 2.0 (Revised after code review)
**Status**: Design Review
**Author**: Claude (with Dustin)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-21 | Initial design |
| 2.0 | 2026-01-21 | Major corrections after code review - fixed Phase 1 scope, updated Phase 3 estimates, added missing components |

---

## Executive Summary

This document outlines the integration plan for upstream musubi-tuner changes into blissful-tuner. The integration covers three main areas:

1. **Phase 1**: Qwen-Image Bug Fixes (4 remaining fixes from commit 045eed5)
2. **Phase 2**: LoHa Network Module (new PEFT method)
3. **Phase 3**: FLUX.2 Architecture Support (major new feature)

**Revised Scope Estimate**: ~18 files modified/added, **~4,200 lines of code**

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Phase 1: Qwen-Image Bug Fixes](#phase-1-qwen-image-bug-fixes)
3. [Phase 2: LoHa Network Module](#phase-2-loha-network-module)
4. [Phase 3: FLUX.2 Integration](#phase-3-flux2-integration)
5. [Testing Strategy](#testing-strategy)
6. [Risk Assessment](#risk-assessment)
7. [Implementation Checklist](#implementation-checklist)

---

## Current State Analysis

### Already Integrated from Upstream

| Commit | Description | File | Status |
|--------|-------------|------|--------|
| `b3edf9e` | varlen attention encoder_hidden_states metadata fix | `qwen_image_model.py` | ✅ Integrated |
| `fc0e691` | img_shapes list-of-lists + remove_first_image_from_target | `qwen_image_train_network.py` | ✅ Integrated |
| `d50909f` | z-image diffusers key mapping | `convert_lora.py` | ✅ Integrated |
| `ab32f89` | Mu calculation fix using control_latent shape | `qwen_image_generate_image.py:795` | ✅ Integrated |
| `6bcf3e5` | Control image loading for layered (VAE move + assertion) | `qwen_image_generate_image.py:1223` | ✅ Integrated |

### Pending from Upstream

| Commit | Description | Priority | Files Affected |
|--------|-------------|----------|----------------|
| `045eed5` | Batch generation fixes (4 specific issues) | Critical | `qwen_image_generate_image.py` |
| `92ef4ee` | LoHa network module | Medium | `networks/loha.py`, `hv_train_network.py` |
| `737f5a8` | FLUX.2-dev training support | High | ~16 files, ~3900 LOC |
| `d5f1ca7` | FLUX.2 formatting and readme updates | High | Included in above |

### Blissful-Tuner Customizations to Preserve

- **BlissfulLogger** integration (replaces standard logging)
- **Rich console output** (RichHelpFormatter, rich tracebacks)
- **Prompt management** (wildcards via `process_wildcards()`)
- **Power seed** (`power_seed()` for seed handling)
- **Blissful args** (`add_blissful_qwen_args`, `parse_blissful_args`)
- **LoRA conversion** (auto-detect and convert diffusers format)
- **Compile flags** (`disable_linear_for_compile` handling)

---

## Phase 1: Qwen-Image Bug Fixes

### Scope Clarification

**Previous incorrect assessment**: Listed 9 items including fixes already integrated.

**Actual remaining work**: 4 specific issues from commit `045eed5`:

| Issue | Location | Status |
|-------|----------|--------|
| `--only` prompt lines support | `parse_prompt_line()` | **MISSING** |
| VL processor in `load_shared_models()` | Line 1166 | **MISSING** |
| VL processor in `process_batch_prompts()` | Line 1205 | **MISSING** |
| `latent[0]` bug in save_output | Line 1325 | **MISSING** |

### 1.1 Prompt Lines Starting with `--` (No Prompt Text)

**File**: `src/musubi_tuner/qwen_image_generate_image.py`
**Location**: `parse_prompt_line()` function (~line 217)

#### Problem

Users cannot specify option-only lines in batch files (e.g., `--w 512 --h 512 --d 123` without a prompt). The current implementation assumes every line starts with a prompt.

#### Current Code (Broken)

```python
def parse_prompt_line(line: str, prompt_wildcards: Optional[str] = None) -> Dict[str, Any]:
    parts = line.split(" --")
    prompt = parts[0].strip()
    if prompt_wildcards is not None:
        prompt = process_wildcards(prompt, prompt_wildcards)
    overrides = {"prompt": prompt}
    overrides["control_image_path"] = []

    for part in parts[1:]:  # Skips first element assuming it's prompt
        ...
```

#### Required Fix (Preserving Blissful Wildcards)

```python
def parse_prompt_line(line: str, prompt_wildcards: Optional[str] = None) -> Dict[str, Any]:
    if line.strip().startswith("--"):  # No prompt, only options
        parts = (" " + line.strip()).split(" --")
        prompt = None
    else:
        parts = line.split(" --")
        prompt = parts[0].strip()
        if prompt_wildcards is not None:
            prompt = process_wildcards(prompt, prompt_wildcards)
        parts = parts[1:]  # CRITICAL: Remove prompt from parts for iteration

    overrides = {} if prompt is None else {"prompt": prompt}
    overrides["control_image_path"] = []

    for part in parts:  # Now iterates over options only
        ...
```

### 1.2 VL Processor Loading for Layered Mode

**File**: `src/musubi_tuner/qwen_image_generate_image.py`

#### Problem

The VL processor is only loaded for Edit mode, but Layered mode with `automatic_prompt_lang_for_layered` also requires it.

#### Location 1: `load_shared_models()` (~line 1166)

```python
# CURRENT (line 1166)
if args.is_edit:
    vl_processor = qwen_image_utils.load_vl_processor()
    shared_models["vl_processor"] = vl_processor

# REQUIRED
if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None):
    vl_processor = qwen_image_utils.load_vl_processor()
    shared_models["vl_processor"] = vl_processor
```

#### Location 2: `process_batch_prompts()` (~line 1205)

```python
# CURRENT (line 1205)
vl_processor_batch = qwen_image_utils.load_vl_processor() if args.is_edit else None

# REQUIRED
vl_processor_batch = (
    qwen_image_utils.load_vl_processor()
    if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None)
    else None
)
```

### 1.3 Latent Shape Bug in Batch Generation

**File**: `src/musubi_tuner/qwen_image_generate_image.py`
**Location**: `process_batch_prompts()` (~line 1325)

#### Problem

The `save_output()` call incorrectly indexes `latent[0]`, but `generate()` now returns different shapes:
- Non-layered: `[B, C, 1, H, W]` (BC1HW)
- Layered: `[B, L, C, H, W]` (BLCHW)

The `[0]` indexing breaks batch processing.

#### Current Code (Broken)

```python
# Lines 1323-1325
# save_output expects latent to be [BCTHW] or [CTHW]. generate returns [BCTHW] (batch size 1).
# latent[0] is correct if generate returns it with batch dim.
# The latent from generate is (1, C, T, H, W)
save_output(current_args, vae_for_batch, latent[0], device)  # Pass vae_for_batch
```

#### Required Fix

```python
# save_output expects latent to be [BCTHW] or [CTHW].
# generate returns BC1HW for non-layered (backward compatibility) or BLCHW for layered
save_output(current_args, vae_for_batch, latent, device)  # Pass vae_for_batch
```

---

## Phase 2: LoHa Network Module

### 2.1 Overview

LoHa (Low-rank Hadamard Product) is an alternative PEFT method from the LyCORIS family:

```
LoRA:  ΔW = lora_down @ lora_up
LoHa:  ΔW = (W1_a @ W1_b) ⊙ (W2_a @ W2_b)  # Hadamard product
```

### 2.2 New File

**File**: `src/musubi_tuner/networks/loha.py` (~767 lines)

Copy from upstream with BlissfulLogger adaptation:

```python
# UPSTREAM (lines 11-17)
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# BLISSFUL ADAPTATION
from blissful_tuner.blissful_logger import BlissfulLogger
logger = BlissfulLogger(__name__, "green")
```

**Alternative approach** (recommended by reviewer): Create a logging adapter so BlissfulLogger can intercept standard logging calls, reducing fork maintenance burden.

### 2.3 Training Script Changes

**File**: `src/musubi_tuner/hv_train_network.py`

Add `architecture=self.architecture` parameter at **3 locations**:

#### Location 1: LoRA merging (~line 1793)

```python
module = network_module.create_arch_network_from_weights(
    multiplier, weights_sd, unet=transformer, for_inference=True,
    architecture=self.architecture  # ADD
)
```

#### Location 2: dim_from_weights (~line 1809)

```python
network, _ = network_module.create_arch_network_from_weights(
    1, weights_sd, unet=transformer,
    architecture=self.architecture  # ADD
)
```

#### Location 3: create_arch_network (~line 1823)

```python
network = network_module.create_arch_network(
    ...
    neuron_dropout=args.network_dropout,
    architecture=self.architecture,  # ADD
    **net_kwargs,
)
```

### 2.4 Additional Call Sites to Check

The reviewer correctly identified other files that call `create_arch_network_from_weights`:

| File | Line | Needs Review |
|------|------|--------------|
| `hv_generate_video.py` | 714 | Check if architecture param needed |
| `wan_generate_video.py` | 865 | Check if architecture param needed |
| `merge_lora.py` | 51 | Check if architecture param needed |

**Note**: These may work without the parameter if they always use the default LoRA module (not LoHa). However, for consistency and future-proofing, consider adding the parameter.

---

## Phase 3: FLUX.2 Integration

### 3.1 Revised Scope

| Metric | Previous Estimate | Actual |
|--------|-------------------|--------|
| Lines of Code | ~1,200 | **~3,911** |
| Files | ~15 | **~16** |

### 3.2 New Files Required

#### Core Model Files

| File | Lines | Description |
|------|-------|-------------|
| `src/musubi_tuner/flux_2/__init__.py` | ~5 | Package init |
| `src/musubi_tuner/flux_2/flux2_models.py` | ~1,500 | Transformer, VAE, blocks |
| `src/musubi_tuner/flux_2/flux2_utils.py` | ~800 | Loading, encoding, scheduling |

#### Training/Inference Scripts

| File | Lines | Description |
|------|-------|-------------|
| `src/musubi_tuner/flux_2_train_network.py` | ~450 | Training loop |
| `src/musubi_tuner/flux_2_cache_latents.py` | ~150 | Latent caching |
| `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py` | ~100 | Text encoder caching |
| `src/musubi_tuner/flux_2_generate_image.py` | ~700 | Generation/inference |

#### Network Module

| File | Lines | Description |
|------|-------|-------------|
| `src/musubi_tuner/networks/lora_flux_2.py` | ~100 | LoRA for FLUX.2 |

#### Root Wrappers

| File | Description |
|------|-------------|
| `flux_2_train_network.py` | Thin wrapper |
| `flux_2_cache_latents.py` | Thin wrapper |
| `flux_2_cache_text_encoder_outputs.py` | Thin wrapper |
| `flux_2_generate_image.py` | Thin wrapper |

### 3.3 Dataset Module Modifications

**File**: `src/musubi_tuner/dataset/image_video_dataset.py`

#### New Constants (CORRECTED)

```python
# Add after existing ARCHITECTURE_* constants (~line 78)
ARCHITECTURE_FLUX_2 = "f2"
ARCHITECTURE_FLUX_2_FULL = "flux_2"

# CORRECTED: Use 16, not 32!
RESOLUTION_STEPS_FLUX_2 = 16  # Matches upstream exactly

# Add to RESOLUTION_STEPS dict (~line 608)
ARCHITECTURE_FLUX_2: RESOLUTION_STEPS_FLUX_2,
```

**WARNING**: Using 32 would silently break bucket selection and cache compatibility with upstream.

#### Cache Functions (CORRECTED KEY NAMING)

```python
def save_latent_cache_flux_2(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: Optional[list[torch.Tensor]] = None
):
    """Save FLUX.2 latent cache."""
    ...

def save_text_encoder_output_cache_flux_2(
    item_info: ItemInfo,
    ctx_vec: torch.Tensor  # CORRECTED: ctx_vec, not m3_vec
):
    """Save FLUX.2 text encoder outputs."""
    # Cache key should be ctx_vec_*, not m3_vec_*
    ...
```

### 3.4 Model Spec Metadata (MISSING FROM v1)

**File**: `src/musubi_tuner/utils/sai_model_spec.py`

Add FLUX.2 architecture constants and mapping:

```python
# Add import (~line 19)
from musubi_tuner.dataset.image_video_dataset import (
    ...
    ARCHITECTURE_FLUX_2,
)

# Add constants (~line 76)
ARCH_FLUX_2 = "Flux.2-dev"

# Add implementation URL (~line 94)
IMPL_FLUX_2 = "https://github.com/black-forest-labs/flux2"

# Add to get_arch_and_impl() (~line 171)
elif architecture == ARCHITECTURE_FLUX_2:
    arch = ARCH_FLUX_2
    impl = IMPL_FLUX_2
```

### 3.5 Timestep Sampling: flux2_shift (MISSING FROM v1)

**File**: `src/musubi_tuner/hv_train_network.py`

FLUX.2 requires a new timestep sampling option. Add to:

#### Argparse choices (~line 2720 in upstream)

```python
parser.add_argument(
    "--timestep_sampling",
    type=str,
    default="sigmoid",
    choices=[
        "uniform", "sigmoid", "shift", "flux_shift",
        "flux2_shift",  # ADD THIS
    ],
    ...
)
```

#### Sampling logic (~line 825, 861)

```python
# Add condition check (~line 825)
if (
    args.timestep_sampling in ("shift", "flux_shift")
    or args.timestep_sampling == "flux2_shift"  # ADD THIS
):
    ...

# Add sampling case (~line 861)
elif args.timestep_sampling == "flux2_shift":
    # FLUX.2 specific shift calculation
    ...
```

### 3.6 Dependencies (CORRECTED)

**File**: `pyproject.toml`

```toml
# CORRECTED: transformers needs bumping, diffusers is NOT required for Flux.2 VAE
[project.dependencies]
transformers = ">=4.56.1"  # Was >=4.46.0, Flux.2 requires newer version
# Note: Flux.2 uses its own flux2_models.AutoEncoder, NOT diffusers
```

**Important**: The upstream Flux.2 implementation does NOT use `diffusers.AutoencoderKLFlux2`. It has its own VAE implementation in `flux2_models.py`.

### 3.7 Batch Size Limitation (NEW)

**Critical UX constraint** discovered in upstream:

```python
# flux_2_train_network.py line 287
assert bsize == 1, "Flux 2 can't be trained with higher batch size since ref images may different size and number"
```

**Impact**: When using reference images, batch size MUST be 1. This significantly affects:
- Training performance (no batching possible with refs)
- Dataset configuration recommendations
- User documentation

### 3.8 Blissful-Tuner Customization Strategy

**Reviewer recommendation**: Instead of replacing every `logging` call in upstream files with BlissfulLogger, consider:

1. **Option A**: Create a logging handler/adapter that routes stdlib logging through BlissfulLogger
2. **Option B**: Only customize at script entry points (argparse, main functions), leave library code closer to upstream

**Benefits**:
- Reduces merge conflicts with future upstream updates
- Easier maintenance
- Consistent behavior

**Suggested approach**:

```python
# In script entry points only
import logging
from blissful_tuner.blissful_logger import BlissfulLogger

# Configure root logger to use BlissfulLogger handler
blissful_handler = BlissfulLogger.create_handler()
logging.root.addHandler(blissful_handler)
```

---

## Testing Strategy

### Phase 1 Testing (Qwen-Image)

```bash
# Test 1: Batch generation with --from_file (including --only lines)
cat > test_prompts.txt << 'EOF'
A portrait photo --w 1024 --h 1024 --d 42
--w 512 --h 512 --d 123
Another prompt with options --w 768 --h 768
EOF

python qwen_image_generate_image.py \
    --from_file test_prompts.txt \
    --dit /path/to/dit \
    --vae /path/to/vae \
    --text_encoder /path/to/te

# Verify: All 3 lines processed correctly, no IndexError

# Test 2: Layered generation with automatic prompt
python qwen_image_generate_image.py \
    --is_layered \
    --automatic_prompt_lang_for_layered en \
    --control_image_path /path/to/control.png \
    ...

# Verify: VL processor loads without error
```

### Phase 2 Testing (LoHa)

```bash
# Test 1: LoHa training
accelerate launch hv_train_network.py \
    --network_module networks.loha \
    --network_dim 8 \
    --max_train_steps 10 \
    ...

# Verify: "create LoHa network" in logs, correct architecture targets

# Test 2: Verify architecture parameter propagation
# Check logs show correct target modules for the architecture being trained
```

### Phase 3 Testing (FLUX.2)

```bash
# Test 1: Latent caching
python flux_2_cache_latents.py \
    --dataset_config test_config.toml \
    --vae /path/to/flux2_vae.safetensors

# Verify: Cache files created with correct naming

# Test 2: Text encoder caching
python flux_2_cache_text_encoder_outputs.py \
    --dataset_config test_config.toml \
    --text_encoder /path/to/mistral3.safetensors

# Verify: Cache files contain ctx_vec_* keys (not m3_vec)

# Test 3: Training with flux2_shift
accelerate launch --mixed_precision bf16 flux_2_train_network.py \
    --dit /path/to/flux2-dev.safetensors \
    --timestep_sampling flux2_shift \
    --max_train_steps 10 \
    ...

# Verify: No argparse error, training proceeds

# Test 4: Training with reference images (batch=1 constraint)
# Verify: Assertion triggers if batch_size > 1 with refs
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Qwen-Image fixes break existing workflows | Low | High | Test existing Edit mode after changes |
| LoHa architecture parameter breaks other trainers | Low | Medium | Verify all trainer scripts pass architecture |
| FLUX.2 dataset merge conflicts | Medium | High | Line-by-line merge, preserve Blissful features |
| Breaking existing architectures (WAN, HV) | Low | Critical | Smoke tests on all architectures after merge |
| **FLUX.2 batch=1 with refs limitation** | High | Medium | **Document prominently, consider guardrails** |
| **flux2_shift missing causes argparse failure** | High | High | **Include in initial implementation** |
| **RESOLUTION_STEPS mismatch (32 vs 16)** | N/A (fixed) | Critical | Use upstream value: 16 |
| **Cache key naming mismatch** | N/A (fixed) | Medium | Use upstream naming: ctx_vec |
| Dependency version conflicts | Medium | Medium | Bump transformers to >=4.56.1 |

---

## Implementation Checklist

### Phase 1: Qwen-Image Bug Fixes (4 items)

- [ ] **1.1** Fix `parse_prompt_line()` to support `--only` lines (preserve wildcards)
- [ ] **1.2** Fix `load_shared_models()` VL processor for layered+auto_prompt
- [ ] **1.3** Fix `process_batch_prompts()` VL processor for layered+auto_prompt
- [ ] **1.4** Fix `save_output()` latent shape (remove `[0]` indexing)
- [ ] **1.5** Test batch generation with mixed prompt types
- [ ] **1.6** Test layered generation with automatic_prompt_lang

### Phase 2: LoHa Network Module (6 items)

- [ ] **2.1** Copy `networks/loha.py` from upstream
- [ ] **2.2** Apply BlissfulLogger integration (or create adapter)
- [ ] **2.3** Add `architecture` param to `hv_train_network.py` location 1 (line ~1793)
- [ ] **2.4** Add `architecture` param to `hv_train_network.py` location 2 (line ~1809)
- [ ] **2.5** Add `architecture` param to `hv_train_network.py` location 3 (line ~1823)
- [ ] **2.6** Audit other `create_arch_network` call sites (hv_generate, wan_generate, merge_lora)
- [ ] **2.7** Test LoHa training on HunyuanVideo
- [ ] **2.8** Test LoHa with other architectures (WAN, Qwen-Image)

### Phase 3: FLUX.2 Integration (22 items)

#### Core Files
- [ ] **3.1** Create `src/musubi_tuner/flux_2/` directory
- [ ] **3.2** Copy `flux_2/__init__.py`
- [ ] **3.3** Copy `flux_2/flux2_models.py` (apply minimal Blissful adaptations)
- [ ] **3.4** Copy `flux_2/flux2_utils.py` (apply minimal Blissful adaptations)
- [ ] **3.5** Copy `networks/lora_flux_2.py`

#### Scripts
- [ ] **3.6** Copy `flux_2_train_network.py` with Blissful enhancements
- [ ] **3.7** Copy `flux_2_cache_latents.py` with Blissful enhancements
- [ ] **3.8** Copy `flux_2_cache_text_encoder_outputs.py` with Blissful enhancements
- [ ] **3.9** Copy `flux_2_generate_image.py` with Blissful enhancements
- [ ] **3.10** Create root wrapper scripts (4 files)

#### Dataset Module
- [ ] **3.11** Add `ARCHITECTURE_FLUX_2` constant
- [ ] **3.12** Add `RESOLUTION_STEPS_FLUX_2 = 16` (NOT 32!)
- [ ] **3.13** Add `save_latent_cache_flux_2()` function
- [ ] **3.14** Add `save_text_encoder_output_cache_flux_2()` function (use `ctx_vec` key)
- [ ] **3.15** Add FLUX.2 case to dataset loading logic

#### Supporting Changes
- [ ] **3.16** Add FLUX.2 to `sai_model_spec.py` (ARCH, IMPL, mapping)
- [ ] **3.17** Add `flux2_shift` to `hv_train_network.py` argparse choices
- [ ] **3.18** Add `flux2_shift` timestep sampling logic
- [ ] **3.19** Update `pyproject.toml`: `transformers>=4.56.1`

#### Testing
- [ ] **3.20** Test latent caching
- [ ] **3.21** Test text encoder caching (verify ctx_vec keys)
- [ ] **3.22** Test training with flux2_shift
- [ ] **3.23** Test batch=1 assertion with reference images
- [ ] **3.24** Test generation

### Post-Integration (4 items)

- [ ] **4.1** Run smoke tests on existing architectures (WAN, HV, Qwen-Image, FramePack)
- [ ] **4.2** Update CLAUDE.md with FLUX.2 information
- [ ] **4.3** Update docs/flux2_integration_plan.md status
- [ ] **4.4** Consider regression test for batch generation latent shapes

---

## Appendix A: Corrected File Diffs

### qwen_image_generate_image.py (Phase 1)

```diff
# parse_prompt_line (~line 217)
 def parse_prompt_line(line: str, prompt_wildcards: Optional[str] = None) -> Dict[str, Any]:
+    if line.strip().startswith("--"):  # No prompt, only options
+        parts = (" " + line.strip()).split(" --")
+        prompt = None
+    else:
         parts = line.split(" --")
         prompt = parts[0].strip()
         if prompt_wildcards is not None:
             prompt = process_wildcards(prompt, prompt_wildcards)
-    overrides = {"prompt": prompt}
+        parts = parts[1:]
+
+    overrides = {} if prompt is None else {"prompt": prompt}
     overrides["control_image_path"] = []
-    for part in parts[1:]:
+    for part in parts:

# load_shared_models (~line 1166)
-    if args.is_edit:
+    if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None):
         vl_processor = qwen_image_utils.load_vl_processor()

# process_batch_prompts (~line 1205)
-    vl_processor_batch = qwen_image_utils.load_vl_processor() if args.is_edit else None
+    vl_processor_batch = (
+        qwen_image_utils.load_vl_processor()
+        if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None)
+        else None
+    )

# save_output call (~line 1325)
-            save_output(current_args, vae_for_batch, latent[0], device)
+            save_output(current_args, vae_for_batch, latent, device)
```

### hv_train_network.py (Phase 2 + Phase 3)

```diff
# LoHa architecture parameter (3 locations)
  network_module.create_arch_network_from_weights(
      multiplier, weights_sd, unet=transformer, for_inference=True,
+     architecture=self.architecture
  )

# flux2_shift argparse (~line 2720)
  choices=[
      "uniform", "sigmoid", "shift", "flux_shift",
+     "flux2_shift",
  ],

# flux2_shift logic (~line 825)
  if (
      args.timestep_sampling in ("shift", "flux_shift")
+     or args.timestep_sampling == "flux2_shift"
  ):
```

---

## Appendix B: Upstream Reference

- **Upstream repository**: `~/musubi-tuner` (synced to `d5f1ca7`)
- **FLUX.2 commits**: `737f5a8` (main implementation), `d5f1ca7` (formatting)
- **LoHa commit**: `92ef4ee`
- **Qwen-Image fix commit**: `045eed5`

### Key Upstream Files for Reference

| Purpose | Upstream Path |
|---------|---------------|
| FLUX.2 models | `src/musubi_tuner/flux_2/flux2_models.py` |
| FLUX.2 utils | `src/musubi_tuner/flux_2/flux2_utils.py` |
| Dataset constants | `src/musubi_tuner/dataset/image_video_dataset.py:595-608` |
| Model spec | `src/musubi_tuner/utils/sai_model_spec.py:76,94,171-173` |
| LoHa module | `src/musubi_tuner/networks/loha.py` |

---

*Document revised after thorough code review. All claims verified against actual source code.*
