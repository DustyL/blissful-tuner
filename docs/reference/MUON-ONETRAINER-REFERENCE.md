# Muon Optimizer Reference: Lessons from OneTrainer

**Purpose:** Reference documentation for implementing Muon optimizer in blissful-tuner, based on analysis of OneTrainer's production implementation.

**Source Repository:** https://github.com/Nerogar/OneTrainer
**Analysis Date:** 2026-01-27

---

## Table of Contents

1. [Algorithm Overview](#1-algorithm-overview)
2. [OneTrainer Architecture](#2-onetrainer-architecture)
3. [Parameter Separation Strategy](#3-parameter-separation-strategy)
4. [Model-Specific Layer Patterns](#4-model-specific-layer-patterns)
5. [Configuration Options](#5-configuration-options)
6. [Code Patterns](#6-code-patterns)
7. [Advanced Variants](#7-advanced-variants)
8. [Integration Points](#8-integration-points)
9. [Default Values Reference](#9-default-values-reference)
10. [Warnings and Edge Cases](#10-warnings-and-edge-cases)

---

## 1. Algorithm Overview

### Core Muon Algorithm

From the official implementation (https://github.com/KellerJordan/Muon):

```python
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute orthogonalization.
    Produces US'V^T where S' ~ Uniform(0.5, 1.5).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimized quintic coefficients
    X = G.bfloat16()

    # Work with shorter dimension for efficiency
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize to spectral norm <= 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # 5 Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
```

### Why Muon Works for LoRA

| Property | Benefit for LoRA |
|----------|------------------|
| Orthogonalized updates | Prevents rank collapse in low-rank matrices |
| Spectral norm LR | Equal update magnitude across singular values |
| Minimal memory | Only momentum buffer (1x params vs AdamW's 2x) |
| bfloat16 stable | Safe for mixed precision training |

---

## 2. OneTrainer Architecture

### File Structure

```
modules/util/
├── optimizer/
│   └── muon_util.py          # Parameter separation logic
├── optimizer_util.py          # Default parameters, optimizer config
├── create.py                  # Optimizer instantiation (lines 1235-1389)
├── enum/
│   └── Optimizer.py           # MUON, MUON_ADV, ADAMUON_ADV enums
└── config/
    └── TrainConfig.py         # Configuration dataclass

modules/ui/
├── MuonAdamWindow.py          # UI for auxiliary Adam config
└── OptimizerParamsWindow.py   # General optimizer params UI
```

### Dependencies

```
# requirements-global.txt
adv_optm==2.1.0                                              # Advanced optimizers (Muon_adv, Adamuon_adv)
-e git+https://github.com/KellerJordan/Muon.git@f90a42b#egg=muon-optimizer  # Official Muon
```

### Optimizer Variants

| Enum Value | Class Used | Description |
|------------|------------|-------------|
| `MUON` | `SingleDeviceMuonWithAuxAdam` / `MuonWithAuxAdam` | Basic Muon + AdamW hybrid |
| `MUON_ADV` | `Muon_adv` from adv_optm | Advanced variant with many options |
| `ADAMUON_ADV` | `Adamuon_adv` from adv_optm | Adam-Muon hybrid |

---

## 3. Parameter Separation Strategy

### OneTrainer's Approach (muon_util.py)

OneTrainer uses **layer-name-based filtering**, not just dimensionality:

```python
def get_optim_type(param_name: str, p: torch.nn.Parameter) -> str:
    """Applies the simplified rule hierarchy to a single parameter."""
    # Rule 1: Check against the layer name filters first
    if any(f.matches(param_name) for f in filters) and p.ndim != 1:
        return 'muon'

    # Rule 2: For everything else, use Adam
    return 'adam'
```

### Key Insight: Two-Stage Filtering

1. **Layer name must match** one of the "hidden layer" patterns
2. **AND** parameter must not be 1D (excludes biases, LayerNorm scales)

### Parameter Group Splitting (muon_util.py:107-165)

```python
def split_parameters_for_muon(
    parameters: list[dict],
    layer_key_fn: dict[int, str],
    config: TrainConfig,
) -> tuple[list[dict], bool]:
    """
    Splits parameter groups into 'muon' and 'adam' subgroups.
    """
    final_param_groups = []
    for group in parameters:
        muon_params = [p for p in group['params']
                       if p.requires_grad and layer_key_fn.get(id(p)) == 'muon']
        adam_params = [p for p in group['params']
                       if p.requires_grad and layer_key_fn.get(id(p)) != 'muon']

        if muon_params:
            muon_group = group.copy()
            muon_group['params'] = muon_params
            muon_group['optim_type'] = 'muon'
            final_param_groups.append(muon_group)

        if adam_params:
            adam_group = group.copy()
            adam_group['params'] = adam_params
            adam_group['optim_type'] = 'adam'
            # Apply Adam-specific LR
            adam_group['lr'] = compute_adam_lr(group, config)
            final_param_groups.append(adam_group)

    return final_param_groups, has_adam_params
```

---

## 4. Model-Specific Layer Patterns

### Default Patterns by Model Type (muon_util.py:31-56)

```python
match model.model_type:
    # UNet-based models (SD 1.5, 2.x, SDXL, Stable Cascade, Würstchen)
    case ModelType.STABLE_DIFFUSION_15 | ModelType.STABLE_DIFFUSION_XL_10_BASE | ...:
        default_patterns = [
            'block',                      # UNet blocks
            'text_model.encoder.layers',  # CLIP text encoder layers
        ]

    # DiT/Transformer-based models (SD3, FLUX, Sana, PixArt, Chroma, Qwen)
    case ModelType.STABLE_DIFFUSION_3 | ModelType.FLUX_DEV_1 | ModelType.SANA | ...:
        default_patterns = [
            'transformer_blocks',  # Main transformer blocks
            'encoder.block',       # T5 text encoder blocks
        ]

    # HiDream
    case ModelType.HI_DREAM_FULL:
        default_patterns = [
            'caption_projection',
            'double_stream_blocks',
            'single_stream_blocks',
        ]

    # Z-Image
    case ModelType.Z_IMAGE:
        default_patterns = [
            'layers',
            'refiner',
        ]
```

### User Override via Configuration

```python
# Users can override with custom patterns:
if config.optimizer.muon_hidden_layers is not None:
    patterns_list = [p.strip() for p in config.optimizer.muon_hidden_layers.split(',')]
    filters = [ModuleFilter(p, use_regex=config.optimizer.muon_adam_regex) for p in patterns_list]
```

### Suggested Patterns for Qwen-Image

Based on typical Qwen architecture and OneTrainer's QWEN pattern:

```python
# For Qwen-based models:
default_patterns = [
    'transformer_blocks',  # Main transformer
    'encoder.block',       # Text encoder
]

# For LoRA specifically, you might use:
lora_patterns = [
    'lora',  # All LoRA modules
    # Or more specific:
    'lora_down',
    'lora_up',
]
```

---

## 5. Configuration Options

### Basic Muon Config (optimizer_util.py:154-164)

```python
Optimizer.MUON: {
    "momentum": 0.95,
    "weight_decay": 0.0,
    "MuonWithAuxAdam": True,           # Enable hybrid mode
    "muon_hidden_layers": None,         # Custom layer patterns (comma-separated)
    "muon_adam_regex": False,           # Use regex for pattern matching
    "muon_adam_lr": 3e-4,               # LR for Adam parameters
    "muon_te1_adam_lr": None,           # Override LR for text_encoder_1
    "muon_te2_adam_lr": None,           # Override LR for text_encoder_2
    "muon_adam_config": None,           # Full Adam config dict
}
```

### Advanced Muon Config (optimizer_util.py:590-618)

```python
Optimizer.MUON_ADV: {
    "beta1": 0.9,
    "cautious_wd": False,
    "weight_decay": 0.0,
    "accelerated_ns": False,            # Accelerated Newton-Schulz
    "ns_steps": 5,                      # Newton-Schulz iterations
    "low_rank_ortho": False,            # Low-rank orthogonalization
    "ortho_rank": 128,                  # Rank for low-rank ortho
    "rms_rescaling": True,              # RMS-based rescaling
    "nnmf_factor": False,               # Factored optimizer mode
    "stochastic_rounding": True,
    "compile": False,                   # torch.compile
    "fused_back_pass": False,
    "MuonWithAuxAdam": True,
    "muon_hidden_layers": None,
    "muon_adam_regex": False,
    "muon_adam_lr": 1e-6,               # Note: Much lower than basic!
    "muon_te1_adam_lr": None,
    "muon_te2_adam_lr": None,
    "nesterov": True,
    "Simplified_AdEMAMix": False,
    "alpha_grad": 100.0,
    "normuon_variant": True,            # Normalized Muon
    "beta2_normuon": 0.95,
    "normuon_eps": 1e-8,
    "orthogonal_gradient": False,       # OrthoGrad method
    "approx_mars": False,               # Approximate MARS
    "muon_adam_config": None,
}
```

### Auxiliary Adam Configuration (MuonAdamWindow.py)

```python
MUON_AUX_ADAM_DEFAULTS = {
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.0,
}
```

---

## 6. Code Patterns

### Optimizer Instantiation (create.py:1339-1389)

```python
case Optimizer.MUON:
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

    # Split parameters into Muon and Adam groups
    params_for_optimizer, ___ = split_parameters_for_muon(parameters, layer_key_fn, config)

    final_param_groups = []
    for group in params_for_optimizer:
        is_muon = group.get('optim_type') == 'muon'

        if is_muon:
            final_group = {
                'params': group['params'],
                'lr': group['lr'],
                'use_muon': True,
                'momentum': optimizer_config.momentum or 0.95,
                'weight_decay': optimizer_config.weight_decay or 0.0,
            }
        else:  # is adam
            adam_config = optimizer_config.muon_adam_config or {}
            final_group = {
                'params': group['params'],
                'lr': group['lr'],
                'use_muon': False,
                'betas': (adam_config.get('beta1', 0.9), adam_config.get('beta2', 0.95)),
                'eps': adam_config.get('eps', 1e-10),
                'weight_decay': adam_config.get('weight_decay', 0.0),
            }
        final_param_groups.append(final_group)

    # Select class based on distributed mode
    OptimizerClass = MuonWithAuxAdam if multi.world_size() > 1 else SingleDeviceMuonWithAuxAdam
    optimizer = OptimizerClass(param_groups=final_param_groups)

    # Restore metadata for framework compatibility
    for i, group in enumerate(optimizer.param_groups):
        original_group = params_for_optimizer[i]
        group['initial_lr'] = original_group.get('initial_lr', original_group['lr'])
        group['name'] = original_group.get('name')
        group['optim_type'] = original_group.get('optim_type')
```

### Layer Key Function Building (muon_util.py:11-105)

```python
def build_muon_adam_key_fn(model: BaseModel, config: TrainConfig) -> dict[int, str]:
    """
    Creates a mapping from parameter id to optimizer type ('muon' or 'adam').
    """
    param_map: dict[int, str] = {}

    # Build filters from config or defaults
    if config.optimizer.muon_hidden_layers is not None:
        patterns_list = [p.strip() for p in config.optimizer.muon_hidden_layers.split(',')]
        filters = [ModuleFilter(p, use_regex=config.optimizer.muon_adam_regex) for p in patterns_list]
    else:
        filters = [ModuleFilter(p, use_regex=False) for p in get_default_patterns(model.model_type)]

    def get_optim_type(param_name: str, p: torch.nn.Parameter) -> str:
        if any(f.matches(param_name) for f in filters) and p.ndim != 1:
            return 'muon'
        return 'adam'

    # Iterate through model modules
    for module_prefix, module in vars(model).items():
        if isinstance(module, LoRAModuleWrapper):
            for lora_module in module.lora_modules.values():
                full_prefix = lora_module.prefix
                for param_name, p in lora_module.named_parameters():
                    if p.requires_grad:
                        full_param_name = f"{full_prefix}.{param_name}"
                        param_map[id(p)] = get_optim_type(full_param_name, p)
        elif isinstance(module, torch.nn.Module):
            for param_name, p in module.named_parameters():
                if p.requires_grad:
                    full_param_name = f"{module_prefix}.{param_name}"
                    param_map[id(p)] = get_optim_type(full_param_name, p)

    return param_map
```

### ModuleFilter Pattern Matching (ModuleFilter.py)

```python
class ModuleFilter:
    """Filter module names using substring or regex matching."""

    def __init__(self, pattern: str, use_regex: bool = False):
        self._pattern = pattern.strip()
        self._used = False
        self._compiled = None

        if use_regex and self._pattern:
            self._compiled = re.compile(self._pattern)

    def matches(self, module_name: str) -> bool:
        if not self._pattern:
            return True  # Empty pattern matches all

        if self._compiled:
            is_match = self._compiled.search(module_name) is not None
        else:
            is_match = self._pattern in module_name

        if is_match:
            self._used = True
        return is_match

    def was_used(self) -> bool:
        return self._used
```

---

## 7. Advanced Variants

### MUON_ADV Features

| Feature | Config Key | Description |
|---------|------------|-------------|
| Normalized Muon | `normuon_variant` | Adds normalization to Muon updates |
| Low-rank Ortho | `low_rank_ortho`, `ortho_rank` | Memory-efficient orthogonalization |
| Accelerated NS | `accelerated_ns` | Faster Newton-Schulz convergence |
| OrthoGrad | `orthogonal_gradient` | Orthogonal gradient projection |
| Approx MARS | `approx_mars` | Approximate MARS method |
| Nesterov | `nesterov` | Nesterov momentum (default: True) |
| Stochastic Rounding | `stochastic_rounding` | Better bfloat16 weight updates |
| Factored Mode | `nnmf_factor` | Memory-efficient state factorization |

### ADAMUON_ADV

Hybrid that combines Adam's adaptive learning with Muon's orthogonalization:

```python
Optimizer.ADAMUON_ADV: {
    "beta1": 0.95,          # Higher than typical Adam
    "beta2": 0.95,          # Second moment coefficient
    "eps": 1e-8,
    "nesterov": False,      # Disabled by default (unlike MUON_ADV)
    # ... plus all MUON_ADV features
}
```

---

## 8. Integration Points

### Where Muon Connects in Training Pipeline

```
TrainConfig
    ↓
optimizer_util.init_model_parameters()
    ├─→ build_muon_adam_key_fn()     # Creates param→optim_type mapping
    └─→ create.create_optimizer()     # Instantiates optimizer
            ├─→ split_parameters_for_muon()  # Splits param groups
            └─→ SingleDeviceMuonWithAuxAdam() # Creates optimizer
                    ↓
            Training Loop (unchanged)
                    ↓
            optimizer.step()  # Muon handles routing internally
```

### Key Integration in optimizer_util.py:52-73

```python
def init_model_parameters(
    model: BaseModel,
    parameters: NamedParameterGroupCollection,
    train_device: torch.device,
):
    model.parameters = parameters
    multi.broadcast_parameters(parameters.parameters(), train_device)

    # Build layer key function if using Muon with auxiliary Adam
    layer_key_fn = None
    if model.train_config.optimizer.MuonWithAuxAdam:
        print("INFO: Creating layer keys for MuonWithAuxAdam.")
        layer_key_fn = build_muon_adam_key_fn(model, model.train_config)

    model.optimizer = create.create_optimizer(
        parameters, model.optimizer_state_dict, model.train_config, layer_key_fn
    )
```

### Parameter Group Name Handling (optimizer_util.py:81-91)

```python
# Update param_group_mapping to include optimizer type suffix
if model.optimizer is not None and any('optim_type' in g for g in model.optimizer.param_groups):
    new_param_group_mapping = []
    for group in model.optimizer.param_groups:
        original_name = group.get('name')
        optim_type = group.get('optim_type', 'unknown')
        unique_name = f"{original_name}_{optim_type}"  # e.g., "unet_lora_muon"
        new_param_group_mapping.append(unique_name)
    model.param_group_mapping = new_param_group_mapping
```

---

## 9. Default Values Reference

### Complete Defaults Table

| Parameter | MUON (Basic) | MUON_ADV | ADAMUON_ADV | Notes |
|-----------|--------------|----------|-------------|-------|
| momentum/beta1 | 0.95 | 0.9 | 0.95 | Momentum coefficient |
| beta2 | N/A | N/A | 0.95 | Second moment (Adamuon only) |
| weight_decay | 0.0 | 0.0 | 0.0 | Decoupled weight decay |
| ns_steps | 5 (internal) | 5 | 5 | Newton-Schulz iterations |
| nesterov | True (internal) | True | False | Nesterov momentum |
| MuonWithAuxAdam | True | True | True | Enable hybrid mode |
| muon_adam_lr | 3e-4 | 1e-6 | 1e-6 | LR for Adam params |
| stochastic_rounding | N/A | True | True | bfloat16 rounding |
| rms_rescaling | N/A | True | True | RMS-based scaling |
| normuon_variant | N/A | True | True | Normalized Muon |

### Auxiliary Adam Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| beta1 | 0.9 | Standard Adam |
| beta2 | 0.999 (basic) / 0.95 (adv) | Second moment |
| eps | 1e-8 (basic) / 1e-10 (optimizer) | Numerical stability |
| weight_decay | 0.0 | Can override per-group |

---

## 10. Warnings and Edge Cases

### Warning: All Parameters Assigned to Muon (muon_util.py:95-98)

```python
if adam_params_count == 0:
    print("\n[MuonWithAuxAdam] WARNING: 100% of trainable parameters are assigned to Muon.")
    print("Consider disabling 'MuonWithAuxAdam' in your configuration since the auxiliary "
          "AdamW optimizer is not being used.")
```

**When this happens:** All trainable parameters match the hidden layer patterns and are 2D.

**Resolution:** Either:
1. Set `MuonWithAuxAdam=False` to use pure Muon
2. Adjust `muon_hidden_layers` to be more selective

### Warning: Unused Filter Patterns (muon_util.py:99-102)

```python
if config.optimizer.muon_hidden_layers is not None:
    unused_filters = [f._pattern for f in filters if not f.was_used()]
    if unused_filters:
        print(f"WARNING: The following hidden layer patterns did not match any parameters: {unused_filters}")
```

**When this happens:** User-specified patterns don't match any parameter names.

**Resolution:** Check parameter names in model and adjust patterns.

### Edge Case: Empty Parameter Groups

The splitting logic handles cases where a group might have all Muon or all Adam params:

```python
if muon_params:
    # Only create Muon group if there are Muon params
    final_param_groups.append(muon_group)

if adam_params:
    # Only create Adam group if there are Adam params
    final_param_groups.append(adam_group)
```

### Edge Case: LoRA Module Iteration

OneTrainer specifically handles LoRA modules differently to get correct full parameter names:

```python
if isinstance(module, LoRAModuleWrapper):
    for lora_module in module.lora_modules.values():
        # Use the LoRA module's prefix for full parameter name
        full_prefix = lora_module.prefix
        for param_name, p in lora_module.named_parameters():
            full_param_name = f"{full_prefix}.{param_name}"
            # This ensures patterns like "transformer_blocks" match correctly
```

---

## Appendix A: Quick Implementation Checklist

Based on OneTrainer's approach:

- [ ] Install official Muon package or implement Newton-Schulz
- [ ] Create parameter separation function with layer-name filtering
- [ ] Define model-specific default layer patterns
- [ ] Support user override of layer patterns
- [ ] Handle separate LR for auxiliary Adam
- [ ] Add warnings for edge cases (all Muon, unused patterns)
- [ ] Preserve param_group metadata after optimizer creation
- [ ] Test with your target model architecture

## Appendix B: External Resources

- **Official Muon Repository:** https://github.com/KellerJordan/Muon
- **adv_optm Package:** Contains Muon_adv, Adamuon_adv with extended features
- **OneTrainer Source:** https://github.com/Nerogar/OneTrainer
- **Muon Paper/Blog:** Search for "Muon optimizer" by Keller Jordan

---

*Document generated from OneTrainer codebase analysis. Last updated: 2026-01-27*
