# Upstream Musubi-Tuner Integration Design

**Date**: 2026-01-21
**Status**: Design Review
**Author**: Claude (with Dustin)

---

## Executive Summary

This document outlines the integration plan for upstream musubi-tuner changes into blissful-tuner. The integration covers three main areas:

1. **Phase 1**: Qwen-Image Bug Fixes (3 remaining fixes)
2. **Phase 2**: LoHa Network Module (new PEFT method)
3. **Phase 3**: FLUX.2 Architecture Support (major new feature)

**Estimated Scope**: ~15 files modified/added, ~1200 lines of code

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

| Commit | Description | File |
|--------|-------------|------|
| `b3edf9e` | varlen attention encoder_hidden_states metadata fix | `qwen_image_model.py` |
| `fc0e691` | img_shapes list-of-lists + remove_first_image_from_target | `qwen_image_train_network.py` |
| `d50909f` | z-image diffusers key mapping | `convert_lora.py` |

### Pending from Upstream

| Commit | Description | Priority |
|--------|-------------|----------|
| `045eed5` | Batch generation degradation fix for Qwen-Image | Critical |
| `ab32f89` | Mu calculation fix using control_latent shape | Critical |
| `6bcf3e5` | Control image loading for layered generation | Critical |
| `92ef4ee` | LoHa network module | Medium |
| `737f5a8` | FLUX.2-dev training support | High |
| `d5f1ca7` | FLUX.2 formatting and readme updates | High |

### Blissful-Tuner Customizations to Preserve

- **BlissfulLogger** integration (replaces standard logging)
- **Rich console output** (RichHelpFormatter, rich tracebacks)
- **Prompt management** (wildcards via `process_wildcards()`)
- **Power seed** (`power_seed()` for seed handling)
- **Blissful args** (`add_blissful_qwen_args`, `parse_blissful_args`)
- **LoRA conversion** (auto-detect and convert diffusers format)
- **Compile flags** (`disable_linear_for_compile` vs `blocks_to_swap > 0`)

---

## Phase 1: Qwen-Image Bug Fixes

### 1.1 Batch Generation Degradation Fix

**Upstream Commit**: `045eed5`
**File**: `src/musubi_tuner/qwen_image_generate_image.py`
**Severity**: Critical

#### Problem

When using `--from_file` batch mode, the generation output handling incorrectly passed `latent[0]` to `save_output()`, breaking the expected tensor shape.

#### Changes Required

**Location 1: `parse_prompt_line()` function (~line 221)**

The upstream version restores support for prompt lines that start with `--` (no prompt text, only options). Current blissful-tuner version breaks this.

```python
# CURRENT (blissful-tuner) - BROKEN for --only lines
def parse_prompt_line(line: str, prompt_wildcards: Optional[str] = None) -> Dict[str, Any]:
    parts = line.split(" --")
    prompt = parts[0].strip()
    if prompt_wildcards is not None:
        prompt = process_wildcards(prompt, prompt_wildcards)
    overrides = {"prompt": prompt}
    ...

# NEEDED (merge upstream fix while preserving wildcards)
def parse_prompt_line(line: str, prompt_wildcards: Optional[str] = None) -> Dict[str, Any]:
    if line.strip().startswith("--"):  # No prompt, only options
        parts = (" " + line.strip()).split(" --")
        prompt = None
    else:
        parts = line.split(" --")
        prompt = parts[0].strip()
        if prompt_wildcards is not None:
            prompt = process_wildcards(prompt, prompt_wildcards)
        parts = parts[1:]  # Important: remove prompt from parts

    overrides = {} if prompt is None else {"prompt": prompt}
    ...

    for part in parts:  # Now iterates correctly
        ...
```

**Location 2: `load_shared_models()` function (~line 1150)**

Add layered mode check for VL processor loading:

```python
# CURRENT
if args.is_edit:
    vl_processor = qwen_image_utils.load_vl_processor()

# NEEDED
if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None):
    vl_processor = qwen_image_utils.load_vl_processor()
```

**Location 3: `process_batch_prompts()` function (~line 1189)**

Same VL processor fix:

```python
# CURRENT
vl_processor_batch = qwen_image_utils.load_vl_processor() if args.is_edit else None

# NEEDED
vl_processor_batch = (
    qwen_image_utils.load_vl_processor()
    if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None)
    else None
)
```

**Location 4: `process_batch_prompts()` - save_output call (~line 1310)**

Fix latent shape handling:

```python
# CURRENT - BROKEN
save_output(current_args, vae_for_batch, latent[0], device)

# NEEDED
# generate returns BC1HW for non-layered or BLCHW for layered
save_output(current_args, vae_for_batch, latent, device)
```

### 1.2 Mu Calculation Fix for Layered Mode

**Upstream Commit**: `ab32f89`
**File**: `src/musubi_tuner/qwen_image_generate_image.py`
**Severity**: Critical

#### Problem

In layered generation mode, the `mu` parameter for the flow matching scheduler was incorrectly calculated using `image_seq_len` (which includes both image AND control latents) instead of just the control latent shape.

#### Changes Required

**Location: `generate()` function (~line 774)**

```python
# CURRENT - INCORRECT
if not args.is_layered:
    mu = qwen_image_utils.calculate_shift_qwen_image(image_seq_len)
else:
    base_seqlen = 256 * 256 / 16 / 16
    mu = (image_seq_len / base_seqlen) ** 0.5  # WRONG: uses combined length

# NEEDED - CORRECT
if not args.is_layered:
    mu = qwen_image_utils.calculate_shift_qwen_image(image_seq_len)
else:
    base_seqlen = 256 * 256 / 16 / 16
    mu = (control_latent.shape[1] / base_seqlen) ** 0.5  # CORRECT: only control
```

#### Why This Matters

The `mu` parameter controls the timestep shifting in flow matching. Using incorrect sequence length produces wrong noise schedules, leading to degraded generation quality in layered mode.

### 1.3 Control Image Loading for Layered Generation

**Upstream Commit**: `6bcf3e5`
**File**: `src/musubi_tuner/qwen_image_generate_image.py`
**Severity**: Critical

#### Problem

Control images were only loaded for Edit mode, not for Layered mode, causing layered generation to fail.

#### Changes Required

**Location 1: `process_batch_prompts()` (~line 1202)**

```python
# CURRENT
if args.is_edit:
    vae_for_batch.to(device)
    for i, prompt_args_item in enumerate(all_prompt_args_list):
        ...

# NEEDED
if args.is_edit or args.is_layered:
    vae_for_batch.to(device)
    for i, prompt_args_item in enumerate(all_prompt_args_list):
        ...
```

**Location 2: Improved error message (~line 698)**

```python
# CURRENT
assert (not args.is_layered) or (control_latents is not None and len(control_latents) == 1), \
    "Qwen-Image-Layered supports only one control image."

# NEEDED (more informative)
assert (not args.is_layered) or (control_latents is not None and len(control_latents) == 1), \
    f"Qwen-Image-Layered supports only one control image: got {len(control_latents) if control_latents is not None else 0}"
```

---

## Phase 2: LoHa Network Module

### 2.1 Overview

LoHa (Low-rank Hadamard Product) is an alternative to LoRA from the LyCORIS family. It uses Hadamard product of two low-rank matrices instead of simple matrix multiplication:

```
ΔW = (W1_a @ W1_b) ⊙ (W2_a @ W2_b)  # Hadamard product
```

vs LoRA:
```
ΔW = lora_down @ lora_up  # Matrix multiplication
```

### 2.2 Benefits

- Potentially better expressiveness with similar parameter counts
- Different optimization landscape may help with certain training scenarios
- Compatible with existing training infrastructure

### 2.3 New Files

**File**: `src/musubi_tuner/networks/loha.py` (767 lines)

Core components:
- `LoHaModule` - Training module with rank/module dropout support
- `LoHaInfModule` - Inference module (inherits from LoHaModule)
- `LoHaNetwork` - Network container managing all modules
- `create_arch_network()` - Architecture-aware network creation
- `create_network_from_weights()` - Load network from saved weights

#### Key Implementation Details

```python
class LoHaModule(torch.nn.Module):
    def __init__(self, lora_name, org_module, multiplier=1.0, lora_dim=4, alpha=1, ...):
        # Four factorized matrices for Hadamard product
        self.hada_w1_a = nn.Parameter(torch.empty(out_dim, lora_dim))
        self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, flatten_in_dim))
        self.hada_w2_a = nn.Parameter(torch.empty(out_dim, lora_dim))
        self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, flatten_in_dim))

    def _compute_diff_weight(self):
        # Hadamard product of two low-rank reconstructions
        diff_weight = (w1_a @ w1_b) * (w2_a @ w2_b)  # Element-wise multiply
        return diff_weight, scale
```

#### Supported Architectures

The module includes architecture-specific target mappings:

```python
TARGET_REPLACE_MODULES = {
    ARCHITECTURE_FRAMEPACK: lora_framepack.FRAMEPACK_TARGET_REPLACE_MODULES,
    ARCHITECTURE_FLUX_KONTEXT: lora_flux.FLUX_KONTEXT_TARGET_REPLACE_MODULES,
    ARCHITECTURE_HUNYUAN_VIDEO: lora_hunyuan.HUNYUAN_TARGET_REPLACE_MODULES,
    ARCHITECTURE_QWEN_IMAGE: lora_qwen_image.QWEN_IMAGE_TARGET_REPLACE_MODULES,
    ARCHITECTURE_QWEN_IMAGE_EDIT: lora_qwen_image.QWEN_IMAGE_TARGET_REPLACE_MODULES,
    ARCHITECTURE_WAN: lora_wan.WAN_TARGET_REPLACE_MODULES,
}
```

### 2.4 Training Script Changes

**File**: `src/musubi_tuner/hv_train_network.py`

Three locations need `architecture=self.architecture` parameter:

**Location 1: LoRA merging (~line 1793)**
```python
module = network_module.create_arch_network_from_weights(
    multiplier, weights_sd, unet=transformer, for_inference=True,
    architecture=self.architecture  # ADD THIS
)
```

**Location 2: dim_from_weights (~line 1809)**
```python
network, _ = network_module.create_arch_network_from_weights(
    1, weights_sd, unet=transformer,
    architecture=self.architecture  # ADD THIS
)
```

**Location 3: create_arch_network (~line 1823)**
```python
network = network_module.create_arch_network(
    ...
    neuron_dropout=args.network_dropout,
    architecture=self.architecture,  # ADD THIS
    **net_kwargs,
)
```

### 2.5 Blissful-Tuner Adaptations

When integrating, replace standard logging with BlissfulLogger:

```python
# UPSTREAM
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# BLISSFUL
from blissful_tuner.blissful_logger import BlissfulLogger
logger = BlissfulLogger(__name__, "green")
```

---

## Phase 3: FLUX.2 Integration

### 3.1 Overview

FLUX.2 is a new architecture from Black Forest Labs with significant differences from FLUX.1:

| Aspect | FLUX.1 | FLUX.2 |
|--------|--------|--------|
| VAE Latent Channels | 16 | 32 |
| Text Encoder | T5 + CLIP (dual) | VLM (Mistral3/Qwen3) |
| Loss | Noise prediction (ε) | Velocity prediction (v) |
| Model Variants | dev, schnell | dev (60B), klein (9B) |

### 3.2 New Files Required

#### Core Model Files

| File | Description | Lines |
|------|-------------|-------|
| `src/musubi_tuner/flux_2/__init__.py` | Package init | ~5 |
| `src/musubi_tuner/flux_2/flux2_models.py` | Transformer, VAE, blocks | ~1500 |
| `src/musubi_tuner/flux_2/flux2_utils.py` | Loading, encoding, scheduling | ~800 |

#### Training/Inference Scripts

| File | Description |
|------|-------------|
| `src/musubi_tuner/flux_2_train_network.py` | Training loop (~500 lines) |
| `src/musubi_tuner/flux_2_cache_latents.py` | Latent caching script |
| `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py` | Text encoder caching |
| `src/musubi_tuner/flux_2_generate_image.py` | Generation/inference |

#### Root Wrappers

| File | Description |
|------|-------------|
| `flux_2_train_network.py` | Thin wrapper |
| `flux_2_cache_latents.py` | Thin wrapper |
| `flux_2_cache_text_encoder_outputs.py` | Thin wrapper |
| `flux_2_generate_image.py` | Thin wrapper |

#### Network Module

| File | Description |
|------|-------------|
| `src/musubi_tuner/networks/lora_flux_2.py` | LoRA for FLUX.2 (~100 lines) |

### 3.3 Dataset Module Modifications

**File**: `src/musubi_tuner/dataset/image_video_dataset.py`

#### New Constants

```python
# Add after existing ARCHITECTURE_* constants (~line 78)
ARCHITECTURE_FLUX_2 = "f2"
ARCHITECTURE_FLUX_2_FULL = "flux_2"

# Add resolution steps constant
RESOLUTION_STEPS_FLUX_2 = 32  # 16× VAE downscale + considerations

# Add to RESOLUTION_STEPS dict (~line 608)
ARCHITECTURE_FLUX_2: RESOLUTION_STEPS_FLUX_2,
```

#### New Cache Functions

```python
def save_latent_cache_flux_2(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: Optional[list[torch.Tensor]] = None
):
    """Save FLUX.2 latent cache with optional control/reference images."""
    ...

def save_text_encoder_output_cache_flux_2(
    item_info: ItemInfo,
    m3_vec: torch.Tensor
):
    """Save FLUX.2 text encoder outputs (Mistral3 multi-layer extraction)."""
    ...
```

#### Dataset Loading Case

```python
# Add to dataset loading logic (~line 1817)
elif self.architecture == ARCHITECTURE_FLUX_2:
    # Load FLUX.2 specific cache format
    ...
```

### 3.4 Model Architecture Details

#### Transformer Structure

```
FLUX.2-DEV:
├── Double-Stream Blocks (×8)
│   ├── Image stream: to_q, to_k, to_v, to_out.0
│   ├── Text stream: add_q_proj, add_k_proj, add_v_proj, to_add_out
│   └── Separate FFNs for each stream
│
└── Single-Stream Blocks (×48)
    ├── Fused QKV+MLP: to_qkv_mlp_proj
    └── Shared modulation
```

#### LoRA Targets

```python
FLUX_2_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]

# Excluded patterns (modulators and norms)
exclude_patterns = [
    r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*",
    r".*(norm).*"
]
```

#### Text Encoding

FLUX.2 extracts from 3 intermediate VLM layers:

```python
OUTPUT_LAYERS = [10, 20, 30]  # For 40-layer Mistral3

# Extract and concatenate
embeddings = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
# Shape: [B, 3, seq_len, hidden_size]

prompt_embeds = embeddings.permute(0, 2, 1, 3).reshape(B, seq_len, joint_attention_dim)
# Final: [B, seq_len, 15360] for DEV (3 × 5120)
```

### 3.5 Blissful-Tuner Enhancements

After basic integration, apply Blissful customizations:

1. **BlissfulLogger** - Replace all standard logging
2. **Rich console** - Add RichHelpFormatter to argparse
3. **Latent preview** - Integrate `latent_preview.py`
4. **Guidance enhancements** - CFGZero*, NAG, perpendicular CFG
5. **Prompt management** - Wildcards, weighting
6. **Power seed** - Enhanced seed handling

### 3.6 Dependencies

Verify/add to `pyproject.toml`:

```toml
[project.dependencies]
transformers = ">=4.56.0"  # For Mistral3/Qwen3 support
diffusers = ">=0.37.0"     # For AutoencoderKLFlux2
```

---

## Testing Strategy

### Phase 1 Testing (Qwen-Image)

```bash
# Test 1: Batch generation with --from_file
echo "A portrait photo --w 1024 --h 1024 --d 42" > test_prompts.txt
echo "--w 512 --h 512 --d 123" >> test_prompts.txt  # No prompt line
python qwen_image_generate_image.py \
    --from_file test_prompts.txt \
    --dit /path/to/dit \
    --vae /path/to/vae \
    --text_encoder /path/to/te

# Test 2: Layered generation
python qwen_image_generate_image.py \
    --is_layered \
    --control_image_path /path/to/control.png \
    --prompt "test prompt" \
    ...

# Verify: Check that mu value logged matches expected calculation
```

### Phase 2 Testing (LoHa)

```bash
# Test 1: LoHa training (short run)
accelerate launch hv_train_network.py \
    --network_module networks.loha \
    --network_dim 8 \
    --max_train_steps 10 \
    ...

# Test 2: LoHa inference/merge
python merge_lora.py \
    --lora_weight /path/to/loha.safetensors \
    ...

# Verify: Check that network creation succeeds with architecture parameter
```

### Phase 3 Testing (FLUX.2)

```bash
# Test 1: Latent caching
python flux_2_cache_latents.py \
    --dataset_config test_config.toml \
    --vae /path/to/flux2_vae.safetensors

# Test 2: Text encoder caching
python flux_2_cache_text_encoder_outputs.py \
    --dataset_config test_config.toml \
    --text_encoder /path/to/mistral3.safetensors

# Test 3: Training (short run)
accelerate launch --mixed_precision bf16 flux_2_train_network.py \
    --dit /path/to/flux2-dev.safetensors \
    --max_train_steps 10 \
    ...

# Test 4: Generation
python flux_2_generate_image.py \
    --dit /path/to/flux2-dev.safetensors \
    --prompt "test image" \
    ...
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Qwen-Image fixes break existing functionality | Low | High | Test existing workflows after each fix |
| LoHa architecture parameter breaks other trainers | Low | Medium | Verify all trainer scripts pass architecture |
| FLUX.2 dataset merge conflicts | Medium | High | Line-by-line merge, preserve Blissful features |
| Breaking existing architectures (WAN, HV) | Low | Critical | Run smoke tests on all architectures |
| Memory issues with FLUX.2 large models | Medium | Medium | Test block swapping thoroughly |
| Dependency version conflicts | Low | Medium | Check transformers/diffusers versions |

---

## Implementation Checklist

### Phase 1: Qwen-Image Bug Fixes

- [ ] **1.1** Apply `parse_prompt_line()` fix preserving wildcards
- [ ] **1.2** Apply `load_shared_models()` VL processor fix
- [ ] **1.3** Apply `process_batch_prompts()` VL processor fix
- [ ] **1.4** Apply `save_output()` latent shape fix
- [ ] **1.5** Apply mu calculation fix for layered mode
- [ ] **1.6** Apply control image loading for layered mode
- [ ] **1.7** Improve error message for layered assertion
- [ ] **1.8** Test batch generation
- [ ] **1.9** Test layered generation

### Phase 2: LoHa Network Module

- [ ] **2.1** Copy `networks/loha.py` from upstream
- [ ] **2.2** Apply BlissfulLogger integration to loha.py
- [ ] **2.3** Add architecture parameter to `hv_train_network.py` (3 locations)
- [ ] **2.4** Test LoHa training
- [ ] **2.5** Test LoHa inference/merge

### Phase 3: FLUX.2 Integration

- [ ] **3.1** Create `flux_2/` directory structure
- [ ] **3.2** Copy `flux2_models.py` with Blissful adaptations
- [ ] **3.3** Copy `flux2_utils.py` with Blissful adaptations
- [ ] **3.4** Copy `lora_flux_2.py` with Blissful adaptations
- [ ] **3.5** Copy training script with Blissful enhancements
- [ ] **3.6** Copy cache scripts with Blissful enhancements
- [ ] **3.7** Copy generation script with Blissful enhancements
- [ ] **3.8** Create root wrapper scripts
- [ ] **3.9** Merge dataset module changes (preserve Blissful features)
- [ ] **3.10** Add FLUX.2 constants to image_video_dataset.py
- [ ] **3.11** Add cache functions to image_video_dataset.py
- [ ] **3.12** Update pyproject.toml dependencies if needed
- [ ] **3.13** Test latent caching
- [ ] **3.14** Test text encoder caching
- [ ] **3.15** Test training
- [ ] **3.16** Test generation

### Post-Integration

- [ ] Run smoke tests on existing architectures (WAN, HV, Qwen-Image)
- [ ] Update CLAUDE.md with FLUX.2 information
- [ ] Update docs/flux2_integration_plan.md status
- [ ] Commit changes with appropriate message

---

## Appendix A: File Diff Summary

### qwen_image_generate_image.py Changes

```diff
# parse_prompt_line - restore --only lines support
+ if line.strip().startswith("--"):
+     parts = (" " + line.strip()).split(" --")
+     prompt = None
+ else:
      parts = line.split(" --")
      prompt = parts[0].strip()
+     parts = parts[1:]

# load_shared_models - add layered check
- if args.is_edit:
+ if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None):

# process_batch_prompts - add layered check (2 locations)
- if args.is_edit:
+ if args.is_edit or args.is_layered:

# generate - fix mu calculation
- mu = (image_seq_len / base_seqlen) ** 0.5
+ mu = (control_latent.shape[1] / base_seqlen) ** 0.5

# save_output call - fix latent indexing
- save_output(current_args, vae_for_batch, latent[0], device)
+ save_output(current_args, vae_for_batch, latent, device)
```

### hv_train_network.py Changes

```diff
# 3 locations - add architecture parameter
  network_module.create_arch_network_from_weights(
      multiplier, weights_sd, unet=transformer, for_inference=True,
+     architecture=self.architecture
  )
```

---

## Appendix B: Source References

- Upstream repository: `~/musubi-tuner` (synced to `d5f1ca7`)
- FLUX.2 worktree: `~/blissful-tuner-flux2`
- Existing documentation:
  - `docs/flux2_architecture.md`
  - `docs/flux2_integration_plan.md`

---

*Document generated as part of brainstorming session for upstream integration.*
