# Muon Optimizer Integration Plan for Blissful-Tuner

**Version:** 3.0
**Date:** 2026-01-27
**Status:** Draft - For Review (Updated with Critical Fixes)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Competitive Analysis: OneTrainer's Implementation](#3-competitive-analysis-onetrainers-implementation)
4. [Critical Implementation Issues (v3.0 Fixes)](#4-critical-implementation-issues-v30-fixes)
5. [Architecture Overview](#5-architecture-overview)
6. [Integration Strategy](#6-integration-strategy)
7. [Implementation Details](#7-implementation-details-updated-for-v30)
8. [Configuration Interface](#8-configuration-interface-v30---corrected-format)
9. [Testing Plan](#9-testing-plan)
10. [Risk Assessment](#10-risk-assessment)
11. [Timeline & Phases](#11-timeline--phases)
12. [Open Questions](#12-open-questions-v30---most-resolved)
13. [Validation Checklist](#13-validation-checklist-before-declaring-success)

---

## 1. Executive Summary

### What is Muon?

**Muon** (MomentUm Orthogonalized by Newton-schulz) is a novel optimizer that replaces each gradient update with the nearest orthogonal matrix via Newton-Schulz iteration. Key characteristics:

- **Memory Efficient**: Only requires momentum buffer (same as SGD)
- **Compute Efficient**: <1% FLOP overhead via 5 Newton-Schulz iterations
- **bfloat16 Stable**: Unlike many second-order methods, runs stably in bfloat16
- **High Learning Rate**: Uses LR ~0.02 (spectral norm units) vs AdamW's ~3e-4

### Why Integrate?

| Benefit | Impact for LoRA Training |
|---------|-------------------------|
| Orthogonalized updates | Prevents rank collapse in low-rank matrices |
| Spectral norm LR | Equal update magnitude across all singular values |
| Minimal memory overhead | Critical for GPU-constrained training |
| Fast per-step time | 142ms vs SOAP's 301ms (from benchmarks) |

### Integration Complexity

**Moderate** - Muon requires parameter group separation (2D weights → Muon, others → AdamW), which differs from the current single-optimizer-per-group approach but is well-supported by the existing infrastructure.

---

## 2. Background & Motivation

### Current Optimizer Landscape in Blissful-Tuner

The codebase supports:
- **AdamW / AdamW8bit**: Default choice, well-understood
- **Adafactor**: Memory-efficient, relative_step support
- **ProdigyPlusScheduleFree**: Adaptive learning rate
- **RMSprop**: Proven effective per community testing
- **Any torch.optim.* optimizer**: Via dynamic loading

### Gap Analysis

| Need | Current Solution | With Muon |
|------|-----------------|-----------|
| Orthogonalized updates | None | Native support |
| Spectral-norm-based LR | Manual tuning | Built-in |
| Fast second-order benefits | SOAP (too slow/memory-hungry) | Muon (<1% overhead) |
| LoRA rank preservation | Hope for the best | Mathematically guaranteed |

### Use Case: Qwen-Image LoRA Training

For the target use case (269-image persona LoRA):
- Small batch sizes (1-8) → Muon works well (unlike SOAP)
- Memory-constrained → Muon adds only 1x params overhead
- 2D LoRA weights → Perfect match for Muon's specialization

---

## 3. Competitive Analysis: OneTrainer's Implementation

> **Key Insight:** OneTrainer's Muon implementation reveals that for LoRA training, almost ALL trainable params are 2D matrices (lora_down, lora_up). Pure dimensionality filtering would assign everything to Muon, leaving nothing for AdamW. **Layer-name filtering provides semantic control** over *which* LoRA modules get Muon's orthogonalization.

### 3.1 Parameter Separation: Dimensionality vs Layer-Name Filtering

**Our Original Plan (Dimensionality-Only):**
```python
if param.ndim == 2:  # 2D weights → Muon
    muon_params.append(param)
else:
    adamw_params.append(param)
```

**OneTrainer's Approach (Layer-Name + Dimensionality):**
```python
# Model-specific layer patterns
default_patterns = {
    'FLUX/SD3/SANA': ['transformer_blocks', 'encoder.block'],
    'SD1.5/SDXL/UNet': ['block', 'text_model.encoder.layers'],
    'Qwen': ['layers', 'blocks'],  # Inferred
}

# Two-stage filtering
if any(pattern in param_name for pattern in filters) and param.ndim != 1:
    return 'muon'
return 'adam'
```

**Why This Matters:**
| Scenario | Dimensionality-Only | Layer-Name Filtering |
|----------|--------------------|--------------------|
| LoRA on attention layers | ✅ Muon | ✅ Muon |
| LoRA on projection layers | ✅ Muon (maybe unwanted) | ❌ AdamW (controlled) |
| Text encoder LoRA | ✅ Muon | ✅/❌ Configurable |
| Mixed architectures | No control | Fine-grained control |

### 3.2 OneTrainer's Dependency Strategy

OneTrainer uses the **official Muon package** directly:
```
# requirements-global.txt
-e git+https://github.com/KellerJordan/Muon.git@f90a42b#egg=muon-optimizer
```

This gives them:
- `MuonWithAuxAdam` (distributed)
- `SingleDeviceMuonWithAuxAdam` (non-distributed)

**Trade-off Analysis:**

| Approach | Pros | Cons |
|----------|------|------|
| Official package | Automatic updates, less maintenance | External dependency, less control |
| Custom implementation | Full control, custom logging | Must maintain Newton-Schulz code |
| Hybrid (thin wrapper) | Best of both | Moderate complexity |

**Recommendation Update:** Consider **Option C (Hybrid)** - use official package but wrap with blissful-tuner-specific features.

### 3.3 Advanced Features in OneTrainer

OneTrainer's `MUON_ADV` variant (via `adv_optm` package) includes experimental features:

| Feature | Purpose | Default |
|---------|---------|---------|
| `normuon_variant` | Normalized Muon variant | False |
| `low_rank_ortho` | Low-rank orthogonalization for memory savings | False |
| `accelerated_ns` | Accelerated Newton-Schulz | False |
| `orthogonal_gradient` | OrthoGrad method | False |
| `approx_mars` | Approximate MARS method | False |
| `rms_rescaling` | RMS-based rescaling | False |

These are all disabled by default but available for experimentation.

### 3.4 Per-Component Learning Rate Configuration

OneTrainer supports separate LRs for different text encoders:
```python
# muon_util.py
te1_adam_lr = optimizer_config.muon_te1_adam_lr
te2_adam_lr = optimizer_config.muon_te2_adam_lr

if original_name in ('text_encoder', 'text_encoder_1', ...):
    adam_lr = te1_adam_lr if te1_adam_lr is not None else base_adam_lr
```

This is useful for:
- SDXL (two text encoders)
- SD3/FLUX (multiple text encoders)
- Models where TE and DiT benefit from different LRs

### 3.5 Default Values Comparison

| Parameter | OneTrainer (Basic) | OneTrainer (ADV) | Our Original Plan |
|-----------|-------------------|------------------|-------------------|
| `muon_lr` | Global LR | Global LR | 0.02 (hardcoded) |
| `momentum` | 0.95 | 0.95 | 0.95 |
| `adam_lr` | 3e-4 | **1e-6** (very conservative) | 3e-4 |
| `weight_decay` | 0.0 | 0.0 | 0.0 |

**Notable:** OneTrainer's advanced variant uses **1e-6** for Adam LR, which is extremely conservative.

### 3.6 Warning System

OneTrainer includes helpful warnings for misconfiguration:
```python
if adam_params_count == 0:
    print("WARNING: 100% of trainable parameters are assigned to Muon.")

if unused_filters:
    print(f"WARNING: The following hidden layer patterns did not match...")
```

### 3.7 Key Takeaways for Our Implementation

1. **Layer-name filtering is more important than dimensionality alone**
   - Add `--muon_hidden_layers` parameter for pattern specification

2. **Consider using official Muon package**
   - Reduces maintenance burden
   - `pip install git+https://github.com/KellerJordan/Muon.git`

3. **Don't hardcode Muon LR**
   - Use global `learning_rate` as default, allow override with `muon_lr`

4. **Add useful warnings**
   - Warn if 100% or 0% of params assigned to Muon
   - Warn if layer patterns don't match any params

5. **Preserve metadata on param groups**
   - Restore `initial_lr`, `name`, `optim_type` for scheduler/logging compatibility

---

## 4. Critical Implementation Issues (v3.0 Fixes)

This section addresses critical issues identified during technical review that would cause the v2.0 plan to fail.

### 4.1 Issue: Synthetic Names Break Layer Filtering

**Problem:** The v2.0 plan proposed building synthetic names like `group{i}.param{j}`, which completely breaks layer-name filtering because there's no semantic information to match against.

**Root Cause:** By the time we reach `get_optimizer()`, the `trainable_params` are just flat lists - the original layer names are lost.

**Solution:** Build a `param_id → full_name` map from `network.named_parameters()` *before* calling `get_optimizer()`, at `src/musubi_tuner/hv_train_network.py:1912` where we still have the `network` object in scope.

```python
# At line ~1912, before get_optimizer call:
trainable_params, lr_descriptions = network.prepare_optimizer_params(...)

# NEW: Build param_id -> name mapping for Muon layer filtering
param_name_map = {id(p): name for name, p in network.named_parameters() if p.requires_grad}

# Pass to get_optimizer
optimizer_name, optimizer_args, optimizer, ... = self.get_optimizer(
    args, trainable_params, param_name_map=param_name_map  # NEW param
)
```

### 4.2 Issue: LoRA Names Use Underscores, Not Dots

**Problem:** LoRA module names are created via `.replace(".", "_")` in `src/musubi_tuner/networks/lora.py:800`. OneTrainer-style dot patterns like `text_model.encoder.layers` won't match LoRA names like `text_model_encoder_layers`.

**Solution:** Normalize patterns to use underscores OR support dual matching:

```python
def normalize_pattern(pattern: str) -> str:
    """Normalize pattern for LoRA name matching."""
    return pattern.replace(".", "_")

# In LayerFilter:
def matches(self, param_name: str) -> bool:
    # Match either original pattern or underscore-normalized version
    normalized_pattern = self.pattern.replace(".", "_")
    return (
        self.pattern in param_name or
        normalized_pattern in param_name or
        fnmatch.fnmatch(param_name, f"*{self.pattern}*") or
        fnmatch.fnmatch(param_name, f"*{normalized_pattern}*")
    )
```

### 4.3 Issue: Official Muon Param-Group Schema is Strict

**Problem:** `SingleDeviceMuonWithAuxAdam` asserts exact key sets for each param group. You can't pass extra metadata keys like `optim_type` at construction time.

```python
# Official muon.py asserts:
assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])  # for Muon
assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])  # for Adam
```

**Solution:** Add metadata *after* optimizer creation (OneTrainer's approach):

```python
# Create optimizer with strict schema
optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

# Add metadata for logging AFTER creation
for i, group in enumerate(optimizer.param_groups):
    group["optim_type"] = "muon" if group.get("use_muon") else "adam"
    group["name"] = f"group_{i}"  # Or preserved name from param_name_map
```

### 4.4 Issue: LR Semantics Are Ambiguous

**Problem:** The plan implies `args.learning_rate` could default both Muon LR and Adam LR, which is almost certainly wrong given their ~100x scale difference.

**Solution:** Explicit LR semantics:

| Argument | Meaning | Default |
|----------|---------|---------|
| `--learning_rate` | **Muon LR** (when optimizer_type is Muon/MuonWithAdamW) | 0.02 |
| `--muon_adam_lr` | Aux Adam LR for non-hidden params | 3e-4 |
| `--muon_lr` | Override Muon LR (optional) | Same as `learning_rate` |

```python
# In get_optimizer for Muon:
muon_lr = getattr(args, "muon_lr", None) or args.learning_rate
adam_lr = getattr(args, "muon_adam_lr", 3e-4)
```

**Documentation must clarify:** "When using Muon, `--learning_rate` is in spectral norm units (~0.02), not Adam units (~3e-4)."

### 4.5 Issue: TOML Config Format Doesn't Match Loader

**Problem:** The v2.0 plan shows `[optimizer.args]` tables, but `read_config_from_file` flattens only top-level tables. Nested tables won't work.

**Current Loader Behavior (`src/musubi_tuner/hv_train_network.py:3131`):**
```python
# Only flattens top-level tables, not nested ones
```

**Solution:** Use the existing flat format that works:

```toml
# CORRECT - works with current loader
optimizer_type = "MuonWithAdamW"
learning_rate = 0.02
muon_adam_lr = 3e-4
optimizer_args = ["muon_momentum=0.95", "muon_weight_decay=0.01"]

# INCORRECT - won't work
[optimizer]
optimizer_type = "MuonWithAdamW"
[optimizer.args]
muon_momentum = 0.95  # This won't feed into args.optimizer_args
```

### 4.6 Issue: LoRA+ Multi-Group Compatibility

**Problem:** `prepare_optimizer_params` can emit multiple groups (e.g., "unet" and "unet plus") with different LRs. If we collapse everything into one Muon group, we silently break LoRA+ ratios.

**Solution:** Split *each* incoming group into {muon_subgroup, adam_subgroup} while preserving that group's LR:

```python
def split_param_groups_for_muon(
    trainable_params: List[Dict],
    param_name_map: Dict[int, str],
    layer_filters: List[LayerFilter],
    muon_lr_scale: float = 1.0,  # Relative to group's base LR
) -> List[Dict]:
    """
    Split each param group into Muon and Adam subgroups.
    Preserves LoRA+ LR ratios.
    """
    result_groups = []

    for group in trainable_params:
        base_lr = group.get("lr", 0.02)
        group_params = group["params"]

        muon_params = []
        adam_params = []

        for p in group_params:
            param_name = param_name_map.get(id(p), "")
            if should_use_muon(param_name, p, layer_filters):
                muon_params.append(p)
            else:
                adam_params.append(p)

        if muon_params:
            result_groups.append({
                "params": muon_params,
                "lr": base_lr * muon_lr_scale,  # Preserve relative scaling
                "use_muon": True,
                "momentum": 0.95,
                "weight_decay": 0.0,
            })

        if adam_params:
            result_groups.append({
                "params": adam_params,
                "lr": adam_lr,  # Aux Adam LR (Adam units, not Muon units)
                "use_muon": False,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.0,
            })

    return result_groups
```

**Additional gotcha (logging):** After splitting groups, the number of optimizer `param_groups` increases, but `lr_descriptions`
returned by `prepare_optimizer_params()` still reflects the *pre-split* groups. Any logging code that assumes
`len(lr_descriptions) == len(optimizer.param_groups)` must be updated to avoid index errors.

**Fix:** In step logging, only use `lr_descriptions` if its length matches `len(lr_scheduler.get_last_lr())`; otherwise fall back to
`param_group` metadata (`name`, `optim_type`) or generic `group{i}` labels.

### 4.7 Resolved Open Questions

Based on technical review:

| Question | Resolution |
|----------|------------|
| Qwen-Image patterns | `["transformer_blocks"]` - tight and semantically correct |
| Adam LR default | 3e-4 (standard), expose `muon_adam_lr` with 1e-6 as advanced option |
| Advanced features | **Defer** - validate basic Muon first before adding normuon/low_rank_ortho |
| Named parameters | Build `param_id → full_name` map from `network.named_parameters()` at line 1912 |

---

## 5. Architecture Overview

### 5.1 Muon's Core Algorithm

```
Input: gradient G, momentum buffer M, β=0.95

1. M ← β*M + (1-β)*G                    # Standard momentum update
2. U ← G + β*(G - M)                    # Nesterov-style look-ahead (optional)
3. U_orth ← NewtonSchulz5(U)            # Orthogonalize via NS iteration
4. scale ← √(max(1, rows/cols))         # Spectral norm scaling
5. return U_orth * scale
```

### 5.2 Newton-Schulz Iteration (5 steps)

```python
def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimized coefficients
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT  # Work with shorter dimension
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # Normalize
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
```

### 5.3 Parameter Group Architecture (Updated)

Muon requires splitting parameters with **preserved LoRA+ group structure**:

```
┌─────────────────────────────────────────────────────────────┐
│                    All Trainable Parameters                  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   2D Weight Matrices    │     │   Embeddings, Biases, 1D    │
│   (LoRA down/up)        │     │   (LayerNorm, etc.)         │
│                         │     │                             │
│   → Use MUON            │     │   → Use AdamW               │
│   → LR: 0.02            │     │   → LR: 3e-4                │
│   → momentum: 0.95      │     │   → betas: (0.9, 0.95)      │
└─────────────────────────┘     └─────────────────────────────┘
```

---

## 6. Integration Strategy

### Option A: Standalone Muon Module

Create a new optimizer module that reimplements Muon with blissful-tuner conventions.

**Pros:**
- Full control over implementation
- No external dependencies
- Can add custom logging/features

**Cons:**
- Must maintain Newton-Schulz code
- Miss upstream improvements

### Option B: Direct Dynamic Loading

Use existing infrastructure to load official Muon from external package.

```bash
pip install git+https://github.com/KellerJordan/Muon.git@f90a42b#egg=muon-optimizer
--optimizer_type muon.SingleDeviceMuonWithAuxAdam
```

**Pros:**
- No code changes to optimizer
- Automatic upstream updates
- Minimal maintenance

**Cons:**
- No blissful-tuner-specific integration
- Can't add custom logging/features
- User must install separate package

### Option C: Hybrid Approach (Recommended - Updated)

Use the **official Muon package** as the core optimizer, wrapped with:
1. Layer-name-based parameter filtering (like OneTrainer)
2. Blissful-tuner-specific logging
3. Warning system for misconfiguration
4. Model-specific default patterns

**Pros:**
- Automatic upstream updates for Newton-Schulz core
- Full control over parameter separation logic
- Custom logging and warnings
- Follows proven OneTrainer patterns

**Cons:**
- External dependency (official Muon package)
- Moderate complexity

**Recommendation: Option C (Hybrid)** - Provides the best balance of maintainability and features, validated by OneTrainer's production use.

---

## 7. Implementation Details (Updated for v3.0)

### 7.1 New Files to Create

#### File 1: `src/musubi_tuner/optimizers/muon_util.py`

This file handles parameter filtering and separation (inspired by OneTrainer's approach):

```python
"""
Muon Optimizer Utilities for Blissful-Tuner

Provides layer-name-based parameter filtering for intelligent Muon/AdamW separation.
Based on patterns from OneTrainer's muon_util.py.
"""

import fnmatch
import warnings
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass


# Model-specific default patterns for hidden layers (where Muon is most beneficial)
MODEL_LAYER_PATTERNS: Dict[str, List[str]] = {
    # Diffusion Transformers
    "flux": ["transformer_blocks", "single_transformer_blocks", "encoder.block"],
    "sd3": ["transformer_blocks", "encoder.block"],
    "wan": ["blocks", "transformer"],
    "hunyuan": ["transformer_blocks", "single_transformer_blocks"],
    "framepack": ["transformer_blocks"],

    # Text Encoders
    "t5": ["encoder.block", "layers"],
    "clip": ["text_model.encoder.layers"],

    # Vision Models
    "qwen": ["layers", "blocks", "visual.blocks"],
    "qwen_image": ["transformer_blocks"],

    # Generic fallback (matches most transformer architectures)
    "default": ["layers", "blocks", "transformer_blocks", "encoder.block"],
}


@dataclass
class MuonParamStats:
    """Statistics about parameter assignment."""
    muon_count: int
    adam_count: int
    muon_params_total: int
    adam_params_total: int
    matched_patterns: List[str]
    unmatched_patterns: List[str]


class LayerFilter:
    """Filter for matching parameter names against patterns.

    Handles both dot-notation (OneTrainer style) and underscore-notation
    (Blissful-Tuner LoRA names use .replace(".", "_")).
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pattern_underscore = pattern.replace(".", "_")  # For LoRA name matching
        self._compiled = f"*{pattern}*"
        self._compiled_underscore = f"*{self.pattern_underscore}*"

    def matches(self, param_name: str) -> bool:
        """Check if parameter name matches this filter's pattern.

        Matches against both original pattern and underscore-normalized version
        to handle LoRA's name normalization (.replace(".", "_")).
        """
        return (
            self.pattern in param_name or
            self.pattern_underscore in param_name or
            fnmatch.fnmatch(param_name, self._compiled) or
            fnmatch.fnmatch(param_name, self._compiled_underscore)
        )


def get_default_patterns(model_type: str) -> List[str]:
    """Get default layer patterns for a model type."""
    model_type_lower = model_type.lower()

    # Try exact match first
    if model_type_lower in MODEL_LAYER_PATTERNS:
        return MODEL_LAYER_PATTERNS[model_type_lower]

    # Try partial match
    for key, patterns in MODEL_LAYER_PATTERNS.items():
        if key in model_type_lower or model_type_lower in key:
            return patterns

    # Fall back to default
    return MODEL_LAYER_PATTERNS["default"]


def classify_parameter(
    param_name: str,
    param: "torch.nn.Parameter",
    layer_filters: List[LayerFilter],
) -> str:
    """
    Classify a parameter as 'muon' or 'adam' based on layer name and dimensionality.

    Logic (matching OneTrainer):
    1. If param is 1D (bias, layer norm, etc.) → always 'adam'
    2. If param name matches any layer filter pattern AND is not 1D → 'muon'
    3. Otherwise → 'adam'
    """
    # 1D params always use Adam (biases, layer norms, scales)
    if param.ndim == 1:
        return "adam"

    # Check if param name matches any hidden layer pattern
    for f in layer_filters:
        if f.matches(param_name):
            return "muon"

    # Default to Adam for unmatched 2D params
    return "adam"


def split_param_groups_for_muon(
    trainable_params: List[Dict[str, Any]],
    param_name_map: Dict[int, str],
    hidden_layer_patterns: Optional[List[str]] = None,
    model_type: str = "default",
    muon_lr: Optional[float] = None,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adam_lr: float = 3e-4,
    adam_betas: Tuple[float, float] = (0.9, 0.95),
    adam_eps: float = 1e-8,
    adam_weight_decay: float = 0.0,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], MuonParamStats]:
    """
    Split param groups into Muon and AdamW subgroups, preserving LoRA+ structure.

    This function takes the output of prepare_optimizer_params() and splits each
    group into Muon-eligible and Adam-eligible subgroups. This preserves LoRA+
    LR ratios and other per-group settings.

    Args:
        trainable_params: List of param group dicts from prepare_optimizer_params()
        param_name_map: Dict mapping id(param) -> full layer name (from network.named_parameters())
        hidden_layer_patterns: Custom patterns for hidden layers (if None, uses model_type defaults)
        model_type: Model type for default pattern lookup (e.g., "flux", "wan", "qwen")
        muon_lr: Learning rate for Muon (if None, uses group's base LR)
        muon_momentum: Momentum for Muon
        muon_weight_decay: Weight decay for Muon
        adam_lr: Learning rate for AdamW (fallback if group has no LR)
        adam_betas: Beta coefficients for AdamW
        adam_eps: Epsilon for AdamW
        adam_weight_decay: Weight decay for AdamW
        verbose: Whether to print warnings

    Returns:
        Tuple of (param_groups, stats) - groups are ready for SingleDeviceMuonWithAuxAdam
    """
    # Determine patterns to use
    if hidden_layer_patterns is None:
        patterns = get_default_patterns(model_type)
    else:
        patterns = hidden_layer_patterns

    # Create filters
    filters = [LayerFilter(p) for p in patterns]

    # Track statistics
    all_muon_params = []
    all_adam_params = []
    matched_patterns = set()
    result_groups = []

    # Process each incoming group (preserves LoRA+ structure)
    for group in trainable_params:
        group_params = group["params"] if isinstance(group, dict) else [group]
        base_lr = group.get("lr") if isinstance(group, dict) else None

        muon_params = []
        adam_params = []

        for param in group_params:
            if not param.requires_grad:
                continue

            # Look up real layer name from map
            param_name = param_name_map.get(id(param), "")

            classification = classify_parameter(param_name, param, filters)

            if classification == "muon":
                muon_params.append(param)
                all_muon_params.append(param)
                # Track which patterns matched
                for f in filters:
                    if f.matches(param_name):
                        matched_patterns.add(f.pattern)
            else:
                adam_params.append(param)
                all_adam_params.append(param)

        # Create Muon subgroup for this group (preserves group's LR scaling)
        if muon_params:
            muon_group = {
                "params": muon_params,
                "use_muon": True,
                "lr": muon_lr if muon_lr is not None else (base_lr or 0.02),
                "momentum": muon_momentum,
                "weight_decay": muon_weight_decay,
            }
            result_groups.append(muon_group)

        # Create Adam subgroup for this group (use explicit Adam LR in Adam units)
        if adam_params:
            adam_group = {
                "params": adam_params,
                "use_muon": False,
                "lr": adam_lr,
                "betas": adam_betas,
                "eps": adam_eps,
                "weight_decay": adam_weight_decay,
            }
            result_groups.append(adam_group)

    # Calculate unmatched patterns
    unmatched_patterns = [p for p in patterns if p not in matched_patterns]

    # Build stats
    stats = MuonParamStats(
        muon_count=len(all_muon_params),
        adam_count=len(all_adam_params),
        muon_params_total=sum(p.numel() for p in all_muon_params),
        adam_params_total=sum(p.numel() for p in all_adam_params),
        matched_patterns=list(matched_patterns),
        unmatched_patterns=unmatched_patterns,
    )

    # Emit warnings
    if verbose:
        total_count = stats.muon_count + stats.adam_count
        if total_count > 0:
            muon_pct = 100 * stats.muon_count / total_count
            if muon_pct == 100:
                warnings.warn(
                    "100% of trainable parameters are assigned to Muon. "
                    "This is unusual - consider checking your hidden_layer_patterns."
                )
            elif muon_pct == 0:
                warnings.warn(
                    "0% of trainable parameters are assigned to Muon. "
                    "All parameters will use AdamW. Check if your hidden_layer_patterns "
                    f"match your model architecture. Patterns: {patterns}"
                )

        if unmatched_patterns:
            warnings.warn(
                f"The following hidden layer patterns did not match any parameters: "
                f"{unmatched_patterns}"
            )

    return result_groups, stats


def print_muon_summary(stats: MuonParamStats) -> None:
    """Print a summary of Muon parameter assignment."""
    total_params = stats.muon_params_total + stats.adam_params_total
    muon_pct = 100 * stats.muon_params_total / total_params if total_params > 0 else 0

    print(f"\n{'='*60}")
    print("Muon Parameter Assignment Summary")
    print(f"{'='*60}")
    print(f"  Muon parameters:  {stats.muon_count:,} tensors ({stats.muon_params_total:,} params, {muon_pct:.1f}%)")
    print(f"  AdamW parameters: {stats.adam_count:,} tensors ({stats.adam_params_total:,} params, {100-muon_pct:.1f}%)")
    print(f"  Matched patterns: {stats.matched_patterns}")
    if stats.unmatched_patterns:
        print(f"  Unmatched patterns: {stats.unmatched_patterns}")
    print(f"{'='*60}\n")
```

#### File 2: `src/musubi_tuner/optimizers/muon.py`

The core optimizer implementation (using official Muon package with fallback):

```python
"""
Muon Optimizer for Blissful-Tuner

Provides MuonWithAdamW optimizer that uses:
- Muon (Newton-Schulz orthogonalization) for hidden layer 2D weights
- AdamW for all other parameters (biases, layer norms, embeddings)

Uses the official Muon package when available, with a built-in fallback.
"""

import torch
from typing import List, Dict, Any, Tuple, Optional

__all__ = [
    "MuonWithAdamW",
    "create_muon_optimizer",
    "MUON_AVAILABLE",
]

# Try to import official Muon package
MUON_AVAILABLE = False
try:
    from muon import SingleDeviceMuonWithAuxAdam as OfficialMuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    OfficialMuonWithAuxAdam = None


# ============================================================================
# Fallback Implementation (used when official package not installed)
# ============================================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses optimized quintic coefficients (3.4445, -4.7750, 2.0315) that maximize
    slope at zero for fast convergence.
    """
    assert G.ndim >= 2, "Input must be at least 2D"
    a, b, c = (3.4445, -4.7750, 2.0315)

    original_dtype = G.dtype
    X = G.bfloat16()

    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(original_dtype)


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute Muon update for a single parameter."""
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer

    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)

    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5

    return update.view(original_shape)


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: Tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Standard Adam update computation."""
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])

    bias_correction1 = 1 - betas[0] ** step
    bias_correction2 = 1 - betas[1] ** step

    exp_avg_corrected = exp_avg / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2

    return exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)


class FallbackMuonWithAdamW(torch.optim.Optimizer):
    """
    Fallback implementation of MuonWithAdamW when official package unavailable.
    """

    def __init__(self, param_groups: List[Dict[str, Any]]):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each param_group must specify 'use_muon'")

            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)

        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )

                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])

                    p.add_(update.view(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1

                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])

                    p.add_(update, alpha=-group["lr"])

        return loss


# ============================================================================
# Public API
# ============================================================================

def create_muon_optimizer(
    param_groups: List[Dict[str, Any]],
    use_official: bool = True,
) -> torch.optim.Optimizer:
    """
    Create a MuonWithAdamW optimizer.

    Uses the official Muon package when available, otherwise falls back
    to the built-in implementation.

    Args:
        param_groups: Parameter groups with 'use_muon' flag
        use_official: Whether to prefer the official package (default: True)

    Returns:
        Optimizer instance
    """
    if use_official and MUON_AVAILABLE:
        # Convert param_groups to official format
        official_groups = []
        for group in param_groups:
            official_group = {k: v for k, v in group.items()}
            official_groups.append(official_group)
        return OfficialMuonWithAuxAdam(official_groups)
    else:
        if use_official and not MUON_AVAILABLE:
            import warnings
            warnings.warn(
                "Official Muon package not found. Using built-in fallback. "
                "Install with: pip install git+https://github.com/KellerJordan/Muon.git"
            )
        return FallbackMuonWithAdamW(param_groups)


# Alias for convenience
MuonWithAdamW = FallbackMuonWithAdamW
```

#### File 3: `src/musubi_tuner/optimizers/__init__.py`

```python
"""
Custom optimizers for Blissful-Tuner.
"""

from .muon import (
    MuonWithAdamW,
    create_muon_optimizer,
    MUON_AVAILABLE,
    zeropower_via_newtonschulz5,
    muon_update,
)

from .muon_util import (
    split_param_groups_for_muon,
    get_default_patterns,
    print_muon_summary,
    MuonParamStats,
    MODEL_LAYER_PATTERNS,
)

__all__ = [
    # Optimizer
    "MuonWithAdamW",
    "create_muon_optimizer",
    "MUON_AVAILABLE",
    "zeropower_via_newtonschulz5",
    "muon_update",
    # Utilities
    "split_param_groups_for_muon",
    "get_default_patterns",
    "print_muon_summary",
    "MuonParamStats",
    "MODEL_LAYER_PATTERNS",
]
```

### 7.2 Modifications to Existing Files

#### Modification 1: `src/musubi_tuner/hv_train_network.py`

**Location 1:** Before `get_optimizer()` call (~line 1912)

Build the param_name_map while we still have access to the network:

```python
# At line ~1912, BEFORE get_optimizer call:
trainable_params, lr_descriptions = network.prepare_optimizer_params(
    text_encoder_lr=0, unet_lr=args.learning_rate
)

# NEW: Build param_id -> name mapping for Muon layer filtering
param_name_map = None
if args.optimizer_type.lower() in ("muon", "muonwithadamw"):
    param_name_map = {id(p): name for name, p in network.named_parameters() if p.requires_grad}

# Pass to get_optimizer (add param_name_map parameter)
optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(
    args, trainable_params, param_name_map=param_name_map  # NEW parameter
)
```

**Location 2:** `get_optimizer()` method signature (~line 457)

Update signature to accept param_name_map:

```python
def get_optimizer(
    self,
    args: argparse.Namespace,
    trainable_params: List[Dict[str, Any]],
    param_name_map: Optional[Dict[int, str]] = None,  # NEW
) -> Tuple[str, str, torch.optim.Optimizer, ...]:
```

**Location 3:** `get_optimizer()` Muon handling (~line 520)

Add Muon handling before the generic fallback:

```python
# Add after AdamW handling (line ~523) and before generic fallback

elif case_insensitive_optimizer_type in ("muon", "muonwithadamw"):
    from musubi_tuner.optimizers.muon import create_muon_optimizer, MUON_AVAILABLE
    from musubi_tuner.optimizers.muon_util import (
        split_param_groups_for_muon,
        print_muon_summary,
    )

    if param_name_map is None:
        raise ValueError(
            "Muon optimizer requires param_name_map for layer filtering. "
            "Ensure network.named_parameters() is captured before get_optimizer()."
        )

    # Extract Muon-specific args from optimizer_kwargs
    muon_lr = optimizer_kwargs.pop("muon_lr", None)  # None = use group's base LR
    muon_momentum = optimizer_kwargs.pop("muon_momentum", 0.95)
    muon_weight_decay = optimizer_kwargs.pop("muon_weight_decay", 0.0)
    muon_adam_lr = getattr(args, "muon_adam_lr", None) or optimizer_kwargs.pop("muon_adam_lr", 3e-4)

    # Layer pattern configuration
    hidden_layer_patterns = getattr(args, "muon_hidden_layers", None)
    if hidden_layer_patterns and isinstance(hidden_layer_patterns, str):
        hidden_layer_patterns = [p.strip() for p in hidden_layer_patterns.split(",")]

    model_type = getattr(args, "muon_model_type", "default")

    # Split param groups (preserves LoRA+ structure)
    param_groups, stats = split_param_groups_for_muon(
        trainable_params,
        param_name_map,
        hidden_layer_patterns=hidden_layer_patterns,
        model_type=model_type,
        muon_lr=muon_lr or args.learning_rate,  # Use global LR if not specified
        muon_momentum=muon_momentum,
        muon_weight_decay=muon_weight_decay,
        adam_lr=muon_adam_lr,
        adam_betas=optimizer_kwargs.pop("adam_betas", (0.9, 0.95)),
        adam_eps=optimizer_kwargs.pop("adam_eps", 1e-8),
        adam_weight_decay=optimizer_kwargs.pop("adam_weight_decay", 0.0),
    )

    print_muon_summary(stats)

    # Create optimizer (uses official package if available)
    optimizer = create_muon_optimizer(param_groups)

    # Add metadata AFTER creation (official Muon has strict schema)
    for i, group in enumerate(optimizer.param_groups):
        group["optim_type"] = "muon" if group.get("use_muon") else "adam"
        group["name"] = f"{'muon' if group.get('use_muon') else 'adam'}_{i}"
```

#### Modification 2: New Arguments

**Location:** `hv_train_network.py` argument parser (~line 2672)

Add Muon-specific arguments:

```python
# Muon optimizer arguments
parser.add_argument(
    "--muon_hidden_layers",
    type=str,
    default=None,
    help="Comma-separated layer patterns for Muon (e.g., 'layers,blocks,transformer'). "
         "If not specified, uses model-type-specific defaults.",
)
parser.add_argument(
    "--muon_model_type",
    type=str,
    default="default",
    choices=["flux", "sd3", "wan", "hunyuan", "qwen", "qwen_image", "default"],
    help="Model type for default Muon layer patterns",
)
```

Update `--optimizer_type` help text:

```python
parser.add_argument(
    "--optimizer_type",
    type=str,
    default="",
    help="Optimizer type: AdamW, AdamW8bit, Adafactor, Muon, MuonWithAdamW, "
         "or full module path (e.g., torch.optim.RMSprop)",
)
```

#### Modification 3: Logging Enhancements

**Location:** `hv_train_network.py` in logging section (~line 438)

```python
# Add Muon-specific logging
if "muon" in opt_type.lower():
    for i, param_group in enumerate(optimizer.param_groups):
        optim_type = param_group.get("optim_type", "unknown")
        logs[f"lr/{optim_type}_{i}"] = param_group["lr"]
        if param_group.get("use_muon"):
            logs[f"muon/momentum_{i}"] = param_group.get("momentum", 0.95)
```

---

## 8. Configuration Interface (v3.0 - Corrected Format)

> **Important:** The TOML config loader flattens only top-level tables. Nested tables like `[optimizer.args]` won't work. Use the flat format below.

### 8.1 TOML Configuration Examples

#### Basic MuonWithAdamW (Recommended)

```toml
# CORRECT - flat format that works with current config loader
optimizer_type = "MuonWithAdamW"
learning_rate = 0.02              # This is the MUON LR (spectral norm units)
muon_adam_lr = 3e-4               # Aux Adam LR for non-hidden params
muon_model_type = "qwen_image"    # Uses patterns: ["transformer_blocks"]
lr_scheduler = "cosine"
lr_warmup_steps = 100             # Critical for Muon stability
```

#### MuonWithAdamW with Custom Layer Patterns

```toml
optimizer_type = "MuonWithAdamW"
learning_rate = 0.015             # Muon LR (conservative for diffusion)
muon_adam_lr = 3e-4               # Adam LR for non-hidden params
muon_hidden_layers = "transformer_blocks,blocks"  # Custom comma-separated patterns
optimizer_args = ["muon_momentum=0.95", "muon_weight_decay=0.01"]
```

#### MuonWithAdamW with Very Conservative Adam (OneTrainer ADV style)

```toml
optimizer_type = "MuonWithAdamW"
learning_rate = 0.02              # Muon LR
muon_adam_lr = 1e-6               # Very conservative for non-hidden layers
muon_model_type = "wan"
optimizer_args = ["muon_momentum=0.95", "adam_betas=(0.9,0.95)", "adam_eps=1e-8"]
lr_scheduler = "cosine"
lr_warmup_steps = 100
```

#### Command Line Equivalent

```bash
accelerate launch wan_train_network.py \
    --optimizer_type MuonWithAdamW \
    --learning_rate 0.02 \
    --muon_adam_lr 3e-4 \
    --muon_model_type wan \
    --muon_hidden_layers "transformer_blocks,blocks" \
    --optimizer_args "muon_momentum=0.95" "muon_weight_decay=0.01" \
    --lr_scheduler cosine \
    --lr_warmup_steps 100
```

### 8.2 Example Training Config: Qwen-Image with Muon

```toml
# dlay_qwen2512_lora_muon_v1.toml
# Flat format - all settings at top level

# Model
pretrained_model_name_or_path = "/path/to/Qwen2.5-VL-7B"
vae_path = "/path/to/vae"

# Network
network_module = "networks.lora_qwen"
network_dim = 32
network_alpha = 16

# Optimizer - Muon with AdamW auxiliary
optimizer_type = "MuonWithAdamW"
learning_rate = 0.015              # Muon LR (spectral norm units, conservative for diffusion)
muon_adam_lr = 3e-4                # Adam LR for non-hidden params
muon_model_type = "qwen_image"     # Uses patterns: ["transformer_blocks"]
optimizer_args = ["muon_momentum=0.95", "muon_weight_decay=0.01", "adam_betas=(0.9,0.95)"]

# Scheduler - warmup is CRITICAL for Muon
lr_scheduler = "cosine"
lr_warmup_steps = 100

# Training
max_train_steps = 2000
mixed_precision = "bf16"
gradient_checkpointing = true
max_grad_norm = 1.0

# Logging
logging_dir = "./logs/muon_experiment"
log_with = "tensorboard"

# Saving
output_dir = "./outputs/muon_lora"
save_every_n_steps = 500
```

### 8.3 LR Semantics (v3.0 Clarification)

**Critical:** When using Muon, learning rates are NOT the same scale as AdamW!

| Argument | Scale | Meaning | Default |
|----------|-------|---------|---------|
| `--learning_rate` | Muon scale (~0.02) | Main Muon LR (spectral norm units) | 0.02 |
| `--muon_adam_lr` | Adam scale (~3e-4) | Aux Adam LR for non-hidden params | 3e-4 |
| `--muon_lr` (via optimizer_args) | Muon scale | Override Muon LR | Same as `learning_rate` |

**Documentation must clarify:** *"When using Muon/MuonWithAdamW, `--learning_rate` is in spectral norm units (~0.02), NOT Adam units (~3e-4). Use `--muon_adam_lr` to set the auxiliary Adam learning rate."*

### 8.4 Model-Specific Default Layer Patterns (v3.0 Updated)

These patterns are automatically used when `muon_model_type` is specified.

**Note:** Patterns are matched against **underscore-normalized** LoRA names (e.g., `transformer_blocks` matches `lora_unet_transformer_blocks_0_lora_down`).

| Model Type | Default Patterns | Notes |
|------------|-----------------|-------|
| `flux` | `transformer_blocks`, `single_transformer_blocks` | FLUX DiT architecture |
| `sd3` | `transformer_blocks` | SD3 architecture |
| `wan` | `blocks`, `transformer` | WAN 2.1/2.2 DiT |
| `hunyuan` | `transformer_blocks`, `single_transformer_blocks` | HunyuanVideo |
| `qwen_image` | `transformer_blocks` | Qwen-Image (tight, semantically correct) |
| `default` | `transformer_blocks`, `blocks` | Generic fallback |

**For Qwen-Image specifically:** The backbone uses `transformer_blocks` (see `src/musubi_tuner/qwen_image/qwen_image_model.py:1175`). Use `["transformer_blocks"]` as the tight, semantically correct pattern.

### 8.5 Hyperparameter Recommendations (v3.0)

| Parameter | Diffusion LoRA | Notes |
|-----------|---------------|-------|
| `learning_rate` (Muon) | 0.01 - 0.02 | Start conservative |
| `muon_adam_lr` | 3e-4 (standard) or 1e-6 (conservative) | For non-hidden params |
| `muon_momentum` | 0.95 | Rarely needs tuning |
| `muon_weight_decay` | 0.01 | Optional, AdamW-style |
| `warmup_steps` | 100-200 | **CRITICAL for stability** |
| `lr_scheduler` | cosine | Recommended |

### 8.6 Comparison: Our Config vs OneTrainer

| Feature | Blissful-Tuner (v3.0) | OneTrainer |
|---------|----------------------|------------|
| Layer pattern config | `--muon_hidden_layers` | `muon_hidden_layers` |
| Model type presets | `--muon_model_type` | Built-in per model |
| Default Muon LR | `--learning_rate` | Global `learning_rate` |
| Default Adam LR | `--muon_adam_lr` (3e-4) | 3e-4 (basic), 1e-6 (adv) |
| LoRA+ preservation | **Yes** (split per group) | Yes |
| Name normalization | Underscore matching | Dot patterns |
| Advanced features | Deferred | normuon, low_rank_ortho, etc. |
| Official package | Yes (with fallback) | Yes |

---

## 9. Testing Plan

### 9.1 Unit Tests

#### Test File: `tests/test_muon_optimizer.py`

```python
import torch
import pytest
from musubi_tuner.optimizers.muon import (
    zeropower_via_newtonschulz5,
    muon_update,
    SingleDeviceMuon,
    MuonWithAdamW,
    get_muon_param_groups,
)


class TestNewtonSchulz:
    def test_output_shape_preserved(self):
        G = torch.randn(64, 128)
        result = zeropower_via_newtonschulz5(G)
        assert result.shape == G.shape

    def test_tall_matrix(self):
        G = torch.randn(256, 64)  # rows > cols
        result = zeropower_via_newtonschulz5(G)
        assert result.shape == G.shape

    def test_bfloat16_stability(self):
        G = torch.randn(64, 64, dtype=torch.bfloat16)
        result = zeropower_via_newtonschulz5(G)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_orthogonality_approximation(self):
        G = torch.randn(32, 32)
        result = zeropower_via_newtonschulz5(G)
        # Check approximate orthogonality: U @ U.T ≈ I
        product = result @ result.T
        identity = torch.eye(32)
        # Allow for the US'V^T approximation
        assert torch.allclose(product, identity, atol=0.5)


class TestMuonUpdate:
    def test_basic_update(self):
        grad = torch.randn(64, 128)
        momentum = torch.zeros_like(grad)
        result = muon_update(grad, momentum)
        assert result.shape == grad.shape

    def test_conv_filter_handling(self):
        grad = torch.randn(64, 32, 3, 3)  # Conv filter
        momentum = torch.zeros_like(grad)
        result = muon_update(grad, momentum)
        assert result.shape == grad.shape


class TestSingleDeviceMuon:
    def test_basic_step(self):
        params = [torch.randn(64, 128, requires_grad=True)]
        optimizer = SingleDeviceMuon(params, lr=0.02)

        # Simulate gradient
        params[0].grad = torch.randn_like(params[0])

        # Step should not raise
        optimizer.step()

    def test_weight_decay(self):
        params = [torch.randn(64, 128, requires_grad=True)]
        original_norm = params[0].norm().item()

        optimizer = SingleDeviceMuon(params, lr=0.02, weight_decay=0.1)
        params[0].grad = torch.zeros_like(params[0])  # Zero gradient
        optimizer.step()

        # Weight should decay
        assert params[0].norm().item() < original_norm


class TestMuonWithAdamW:
    def test_mixed_param_groups(self):
        muon_param = torch.randn(64, 128, requires_grad=True)
        adamw_param = torch.randn(64, requires_grad=True)  # 1D

        param_groups = [
            {"params": [muon_param], "use_muon": True, "lr": 0.02},
            {"params": [adamw_param], "use_muon": False, "lr": 3e-4},
        ]

        optimizer = MuonWithAdamW(param_groups)

        muon_param.grad = torch.randn_like(muon_param)
        adamw_param.grad = torch.randn_like(adamw_param)

        optimizer.step()  # Should not raise


class TestLayerNameFiltering:
    """Tests for OneTrainer-style layer name filtering."""

    def test_pattern_matching(self):
        from musubi_tuner.optimizers.muon_util import LayerFilter

        f = LayerFilter("transformer_blocks")
        assert f.matches("model.transformer_blocks.0.attn.weight")
        assert f.matches("transformer_blocks.5.mlp.weight")
        assert not f.matches("model.embed.weight")

    def test_separate_params_with_patterns(self):
        from musubi_tuner.optimizers.muon_util import separate_params_for_muon

        # Create dummy params with realistic names
        params = [
            ("lora.transformer_blocks.0.lora_down", torch.randn(64, 768, requires_grad=True)),
            ("lora.transformer_blocks.0.lora_up", torch.randn(768, 64, requires_grad=True)),
            ("lora.embed.lora_down", torch.randn(64, 768, requires_grad=True)),  # Not hidden layer
            ("lora.bias", torch.randn(768, requires_grad=True)),  # 1D, always Adam
        ]

        groups, stats = separate_params_for_muon(
            iter(params),
            hidden_layer_patterns=["transformer_blocks"],
        )

        # Should have 2 transformer params in Muon, 2 others in Adam
        assert stats.muon_count == 2
        assert stats.adam_count == 2

    def test_model_type_defaults(self):
        from musubi_tuner.optimizers.muon_util import get_default_patterns

        flux_patterns = get_default_patterns("flux")
        assert "transformer_blocks" in flux_patterns

        wan_patterns = get_default_patterns("wan")
        assert "blocks" in wan_patterns

        qwen_patterns = get_default_patterns("qwen_image")
        assert "layers" in qwen_patterns

    def test_warning_on_no_muon_params(self):
        from musubi_tuner.optimizers.muon_util import separate_params_for_muon
        import warnings

        # Params that don't match any pattern
        params = [
            ("lora.other.weight", torch.randn(64, 768, requires_grad=True)),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            groups, stats = separate_params_for_muon(
                iter(params),
                hidden_layer_patterns=["transformer_blocks"],  # Won't match
            )
            assert len(w) >= 1
            assert "0% of trainable parameters" in str(w[0].message)
```

### 9.2 Integration Tests

```python
class TestMuonTrainingIntegration:
    """Integration tests with actual training loop."""

    def test_lora_training_convergence(self):
        """Verify Muon can train a simple LoRA."""
        # Create dummy LoRA-like parameters
        down = torch.randn(64, 768, requires_grad=True)
        up = torch.randn(768, 64, requires_grad=True)

        optimizer = SingleDeviceMuon([down, up], lr=0.02)

        # Simple optimization target
        target = torch.randn(768, 768)

        losses = []
        for _ in range(100):
            output = up @ down
            loss = ((output - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0] * 0.5

    def test_gradient_clipping_compatibility(self):
        """Verify Muon works with gradient clipping."""
        params = [torch.randn(64, 128, requires_grad=True)]
        optimizer = SingleDeviceMuon(params, lr=0.02)

        params[0].grad = torch.randn_like(params[0]) * 100  # Large gradient

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        # Step should work normally
        optimizer.step()
```

### 9.3 Manual Testing Checklist

- [ ] Train Qwen-Image LoRA with Muon for 500 steps, verify loss decreases
- [ ] Compare loss curves: Muon vs AdamW on same dataset
- [ ] Verify VRAM usage is comparable to AdamW (only momentum buffer overhead)
- [ ] Test with gradient checkpointing enabled
- [ ] Test with mixed precision (bf16)
- [ ] Test warmup schedule works correctly
- [ ] Generate images with Muon-trained LoRA, verify quality

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Muon unstable for diffusion training | Medium | High | Start with conservative LR, extensive warmup |
| Memory issues with large models | Low | Medium | Muon uses minimal extra memory |
| Incompatible with existing features | Low | Medium | Thorough integration testing |
| Performance regression | Low | Low | Benchmark against AdamW baseline |

### 10.2 Known Limitations

1. **2D Parameters Only**: Muon only benefits 2D weight matrices. 1D params (biases, layer norms) must use AdamW.

2. **Learning Rate Scale**: Muon's LR (0.02) is ~100x higher than AdamW. Users must adjust expectations.

3. **No Proven Diffusion Track Record**: Muon was developed for LLMs. Diffusion application is experimental.

4. **Warmup Critical**: Muon can be unstable without proper warmup (100+ steps recommended).

### 10.3 Rollback Plan

If issues arise:
1. Muon is isolated in `optimizers/muon.py` - can be removed without affecting other code
2. Users can always fall back to AdamW with same training config
3. No changes to core training loop required

---

## 11. Timeline & Phases

### Phase 1: Core Implementation (1-2 hours)

- [ ] Create `src/musubi_tuner/optimizers/muon.py`
- [ ] Create `src/musubi_tuner/optimizers/__init__.py`
- [ ] Add Muon handling to `get_optimizer()` in `hv_train_network.py`
- [ ] Basic unit tests

### Phase 2: Integration & Testing (2-3 hours)

- [ ] Update argument parser documentation
- [ ] Create example TOML configs
- [ ] Integration tests with training loop
- [ ] Manual testing with Qwen-Image LoRA

### Phase 3: Documentation & Polish (1 hour)

- [ ] Update `docs/advanced_config.md` with Muon section
- [ ] Add Muon to CLAUDE.md optimizer reference
- [ ] Create example training config for each architecture

### Phase 4: Experimental Validation (Ongoing)

- [ ] Run comparison experiments: Muon vs AdamW vs RMSprop
- [ ] Tune hyperparameters for diffusion use case
- [ ] Gather user feedback

---

## 12. Open Questions (v3.0 - Most Resolved)

### Resolved Questions

Based on OneTrainer's production implementation, the following questions are now resolved:

| Question | OneTrainer's Answer | Our Decision |
|----------|--------------------|--------------|
| Default optimizer name | MuonWithAuxAdam | Use `MuonWithAdamW` as primary |
| Default Muon LR | Global learning_rate | Use global LR, allow override via `muon_lr` |
| Parameter separation | Layer-name + dimensionality | Implement layer-name filtering |
| Official package | Yes | Use with fallback implementation |
| Distributed support | Optional class selection | Skip initially (SingleDevice sufficient) |

### Remaining Questions for Your Review

1. **Layer Pattern Defaults for Qwen-Image**:
   - Current proposal: `["layers", "blocks"]`
   - Should we add more patterns specific to Qwen's architecture?

2. **Conservative vs Aggressive Adam LR**:
   - OneTrainer basic: 3e-4
   - OneTrainer ADV: 1e-6 (very conservative)
   - Which should be our default?

3. **Advanced Features Priority**:
   Should we implement any of OneTrainer's advanced features in Phase 2?
   - [ ] `normuon_variant` - Normalized Muon
   - [ ] `low_rank_ortho` - Memory-efficient orthogonalization
   - [ ] `accelerated_ns` - Faster Newton-Schulz
   - [ ] Per-text-encoder LR configuration

4. **Warning Verbosity**:
   - OneTrainer prints warnings for 100% Muon assignment and unmatched patterns
   - Should these be warnings, info logs, or errors?

5. **LoRA+ Compatibility**:
   - Should we support different LRs for `lora_up` vs `lora_down` within the Muon group?
   - OneTrainer does NOT implement this

### Technical Decision Points

6. **Fallback Behavior**:
   - If official Muon package unavailable, should we:
     a. Use built-in fallback silently?
     b. Warn user and use fallback?
     c. Error and require package installation?
   - Current proposal: **Option B** (warn + fallback)

7. **Named Parameters Access**: **RESOLVED in v3.0**
   - Build `param_id → full_name` map from `network.named_parameters()` at line 1912
   - Pass map to `get_optimizer()` as new parameter
   - No synthetic names needed

---

## 13. Validation Checklist (Before Declaring Success)

Before merging Muon integration, complete this validation:

### 13.1 Basic Functionality Tests

- [ ] LoRA job (200-500 steps) completes without NaN/Inf
- [ ] Muon parameters assigned to correct layers (check stats output)
- [ ] AdamW parameters assigned to non-hidden layers
- [ ] LoRA+ LR ratios preserved across Muon/Adam split

### 13.2 Comparison Experiments (Same Dataset, Same Seed)

| Metric | AdamW Baseline | Muon |
|--------|---------------|------|
| Loss curve stability | ✅ / ❌ | ✅ / ❌ |
| Training speed (step time) | ___ ms | ___ ms |
| Sample quality (fixed prompt set) | ___ | ___ |

### 13.3 Compatibility Tests

- [ ] **DoRA enabled** (1D params exist) - Adam handles them correctly
- [ ] **LoRA+ enabled** (multi-group LRs) - ratios preserved
- [ ] **bf16 mixed precision** - Muon's internal `.bfloat16()` works
- [ ] **fp16 mixed precision** - potential hardware constraint (test)
- [ ] **gradient checkpointing** - no interference
- [ ] **block swapping** - no interference

### 13.4 Documentation Verification

- [ ] LR semantics clearly documented (Muon scale vs Adam scale)
- [ ] `muon_model_type` patterns documented per architecture
- [ ] Warmup requirement emphasized
- [ ] Example configs work with actual config loader

---

## Appendix A: Muon vs AdamW Theoretical Comparison

| Aspect | AdamW | Muon |
|--------|-------|------|
| Update normalization | Per-element (second moment) | Per-matrix (spectral norm) |
| Memory | 2x params (m, v) | 1x params (momentum) |
| Compute | O(n) | O(n) + O(k³) for k×k blocks |
| LR interpretation | Raw magnitude | Spectral norm per step |
| Handles rank collapse | No | Yes (orthogonalization) |
| bfloat16 safe | With care | Natively stable |

## Appendix B: Reference Implementations

### Official Muon
- Repository: https://github.com/KellerJordan/Muon
- Main file: `muon.py`
- Install: `pip install git+https://github.com/KellerJordan/Muon.git@f90a42b#egg=muon-optimizer`

### OneTrainer (Production Reference)
- Repository: https://github.com/Nerogar/OneTrainer
- Key files:
  - `modules/util/optimizer/muon_util.py` - Parameter separation logic
  - `modules/util/create.py` - Optimizer instantiation
  - `modules/util/config/TrainConfig.py` - Configuration schema
- Notable features:
  - Layer-name-based filtering with model-specific patterns
  - Advanced variant with normuon, low_rank_ortho, etc.
  - Per-text-encoder LR configuration

## Appendix C: Quick Reference - Config Cheatsheet

### Minimal MuonWithAdamW Config
```toml
optimizer_type = "MuonWithAdamW"
learning_rate = 1e-4
lr_warmup_steps = 100
```

### Full Featured Config
```toml
optimizer_type = "MuonWithAdamW"
learning_rate = 1e-4
muon_model_type = "qwen_image"
optimizer_args = [
    "muon_lr=0.015",
    "muon_momentum=0.95",
    "muon_weight_decay=0.01",
    "adamw_lr=3e-4",
    "adamw_betas=(0.9,0.95)",
]
lr_scheduler = "cosine"
lr_warmup_steps = 100
```

### Command Line Quick Start
```bash
--optimizer_type MuonWithAdamW \
--learning_rate 1e-4 \
--muon_model_type qwen_image \
--lr_scheduler cosine \
--lr_warmup_steps 100
```

---

**End of Plan v3.0**

*Updated with critical implementation fixes:*
- *Param name mapping (not synthetic names)*
- *Underscore normalization for LoRA names*
- *LoRA+ group preservation*
- *Strict official Muon schema handling*
- *Corrected TOML format*
- *Explicit LR semantics*

*Please review and approve before implementation begins.*
