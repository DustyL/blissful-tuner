# T-LoRA: Timestep-Dependent Low-Rank Adaptation — Technical Reference

> **Purpose:** Comprehensive reference for evaluating T-LoRA integration into Blissful Tuner.
> **Sources:** T-LoRA paper (arXiv:2507.05964v2), official repo (ControlGenAI/T-LoRA), LyCORIS integration (PR #277), and Blissful Tuner codebase analysis.
> **Date:** 2026-02-13

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [Architecture & Implementation](#4-architecture--implementation)
5. [LyCORIS Integration](#5-lycoris-integration)
6. [Experimental Results](#6-experimental-results)
7. [Blissful Tuner Integration Assessment](#7-blissful-tuner-integration-assessment)
8. [Integration vs LyCORIS Passthrough](#8-integration-vs-lycoris-passthrough)
9. [Open Questions & Risks](#9-open-questions--risks)
10. [Recommendation](#10-recommendation)

---

## 1. Executive Summary

**T-LoRA** ("Timestep-aware LoRA") is an adaptation algorithm designed specifically for diffusion model fine-tuning that addresses the overfitting problem in few-shot (especially single-image) customization. It was published by researchers at AIRI/HSE University (arXiv:2507.05964, July 2025, revised January 2026).

**Core innovations:**

| Component | What It Does | Why It Matters |
|-----------|-------------|----------------|
| **Timestep-dependent rank masking** | Binary mask reduces active LoRA ranks at high (noisy) timesteps | Constrains capacity where overfitting originates while preserving detail capacity at low timesteps |
| **Ortho-LoRA (SVD initialization)** | Initializes P/Q from SVD of random matrix, using bottom singular vectors | Ensures rank dimensions are truly independent, making masking meaningful |
| **Residual base subtraction** | `ΔW = P·diag(λ·mask)·Q − P_base·diag(λ_base·mask)·Q_base` | Guarantees zero LoRA delta at initialization despite non-zero P/Q init |

**Key result:** T-LoRA achieves nearly identical concept fidelity (Image Similarity) to standard LoRA while dramatically improving text alignment (Text Similarity) — effectively reducing overfitting to backgrounds, poses, and training-image-specific artifacts. A single-image T-LoRA even outperforms standard LoRA trained on 2-3 images in text alignment.

**Paper:** <https://arxiv.org/abs/2507.05964>
**Official repo:** <https://github.com/ControlGenAI/T-LoRA> (also cloned at `/Users/dustin/T-LoRA`)
**LyCORIS PR:** #277 (merged), 3 commits by bghira

---

## 2. Problem Statement & Motivation

### The Overfitting Dilemma in Few-Shot Fine-Tuning

When fine-tuning diffusion models for personalization with limited images (especially single-image), overfitting manifests as:
- Background elements from training images "leaking" into outputs
- Pose/position memorization instead of generalization
- Reduced diversity and flexibility
- Poor text-prompt alignment (reproducing training image rather than following new prompts)

### The Timestep Insight

The paper's motivational experiment fine-tunes SD-XL with LoRA over fixed timestep intervals:

| Timestep Range | Behavior | Role |
|---------------|----------|------|
| **High (800–1000)** | Rapid overfitting, memorizes poses/backgrounds | Essential for shape and proportions |
| **Middle (500–800)** | Richer context, better fine detail | Shape information lost |
| **Low (0–500)** | Best text alignment, most diverse | Concept fidelity suffers |

**The dilemma:** High timesteps are simultaneously the most damaging (causing overfitting) AND the most necessary (for structural coherence). Simply omitting them causes substantial loss of concept fidelity.

**Solution:** A timestep-sensitive strategy that provides full training capacity at low timesteps while constraining capacity at high timesteps.

### Theoretical Grounding

Different diffusion timesteps serve distinct roles:
- High timesteps form coarse features (Choi et al. 2022)
- Middle timesteps produce perceptually rich content
- Low timesteps remove residual noise
- High timesteps contribute most to image diversity; inadequate representation during this stage cannot be recovered later (Chang et al. 2023; Gao et al. 2023)

---

## 3. Mathematical Formulation

### Standard LoRA Baseline

```
W̃ = W + BA    where A ∈ ℝ^{r×m}, B ∈ ℝ^{n×r}
```

### Vanilla T-LoRA (Rank Masking Only)

```
W̃_t = W + B · M_t · A

M_t = M_{r(t)} = diag(1, 1, ..., 1, 0, 0, ..., 0) ∈ ℝ^{r×r}
                       ├── r(t) ──┤  ├── r-r(t) ──┤
```

### Rank Schedule (Linear, Inverse Proportional to Timestep)

```
r(t) = ⌊(r - r_min) · (T - t) / T⌋ + r_min
```

| Variable | Meaning |
|----------|---------|
| `r` | Maximum LoRA rank |
| `r_min` | Minimum rank at noisiest timestep (recommended: 50% of `r`) |
| `T` | Maximum timestep (e.g., 1000 for DDPM) |
| `t` | Current timestep |

**Example (r=64, r_min=32, T=1000):**
- t=0 (clean): r(0) = 64 (full rank)
- t=250: r(250) = 56
- t=500: r(500) = 48
- t=750: r(750) = 40
- t=1000 (noisiest): r(1000) = 32 (minimum rank)

### Ortho-LoRA: SVD-Based Initialization

Standard LoRA has a hidden problem: at higher ranks, the **effective rank** (rank capturing 95% of singular value sum) is much lower than the specified rank. This means many rank dimensions are linearly dependent — and masking out linearly dependent columns has no effect.

**Ortho-LoRA initialization procedure:**

1. Generate random matrix `R ∈ ℝ^{n×m}` sampled from `N(0, 1/r)`
2. Compute SVD: `R = UΣV^T`
3. Take the **last** (smallest) `r` components:
   - `A_init = V^T[-r:]` (bottom singular vectors)
   - `B_init = U[:, -r:]` (bottom singular vectors)
   - `S_init = Σ[-r:]` (bottom singular values)

**Why "last" components?** Higher singular values correlate strongly with overfitting. Using the smallest singular vectors of a random matrix provides:
- Full orthogonality from initialization (no regularization needed)
- Non-trivial singular values (unlike bottom components of the actual weight matrix `W`, which can be near-zero)
- Each rank dimension captures genuinely independent information

### Residual Base Subtraction ("LoRA Trick")

**Problem:** Ortho-LoRA requires non-zero initialization of B (from SVD), but standard LoRA requires `BA = 0` at init to preserve pretrained behavior.

**Solution — algebraic rearrangement:**

```
W̄ = (W − B_init · S_init · A_init) + B · S · A
      ├──── adjusted base W' ─────┤   ├─ LoRA ─┤
```

At initialization (`B = B_init`, `S = S_init`, `A = A_init`):
```
W̄ = W − B_init·S_init·A_init + B_init·S_init·A_init = W  ✓
```

### Full T-LoRA (Combined)

```
W̄ = W − B_init · S_init · M_t · A_init  +  B · S · M_t · A
```

The mask `M_t` appears in both the base subtraction and the trained LoRA path, ensuring `ΔW = 0` at initialization for any timestep.

### Generalized Rank Schedule (with alpha exponent)

The SDXL implementation supports a power-curve exponent:

```
r(t) = ⌊((T - t) / T)^α · (r_max - r_min)⌋ + r_min
```

| α value | Behavior |
|---------|----------|
| α = 1.0 | Linear (default, paper-recommended) |
| α < 1.0 | More rank at higher timesteps (concave) |
| α > 1.0 | Less rank at higher timesteps (convex) |

---

## 4. Architecture & Implementation

### 4.1 Official T-LoRA Repository Structure

```
T-LoRA/
├── train.py                    # SDXL training entry point
├── inference.py                # SDXL inference entry point
├── tlora/
│   ├── model/
│   │   ├── lora.py             # Core: 3 LoRA layer variants
│   │   ├── pipeline_sdxl.py    # T-LoRA inference pipeline
│   │   └── utils_sdxl.py       # Training param utilities
│   ├── trainer_sdxl.py         # SDXL trainer classes
│   ├── inferencer_sdxl.py      # SDXL inference
│   └── data/dataset_sdxl.py    # DreamBooth dataset
└── flux/
    ├── run.py                  # FLUX training entry point
    ├── train.py                # FLUX training loop
    ├── model/
    │   ├── lora.py             # FLUX LoRA + attention processor
    │   └── pipeline.py         # T-LoRA FLUX pipeline
    └── ...
```

### 4.2 Three Layer Variants

| Class | Registry Key | Init Source | Use Case |
|-------|-------------|-------------|----------|
| `LoRALinearLayer` | `"lora"` | Random normal / zero | Vanilla LoRA + optional masking |
| `OrthogonalLoRALinearLayer` | `"ortho_lora"` | SVD of random matrix R | **Primary T-LoRA** (SDXL) |
| `LOrthogonalLoRALinearLayer` | `"lortho_lora"` | SVD of actual weight W | Layer-specific orthogonal init |

### 4.3 Forward Pass — OrthogonalLoRALinearLayer

```python
def forward(self, hidden_states, mask=None):
    if mask is None:
        mask = torch.ones((1, self.rank))

    # Trained path
    q_hidden = self.q_layer(x) * self.lambda_layer * mask
    p_hidden = self.p_layer(q_hidden)

    # Base path (frozen) — subtracted for residual
    result = p_hidden - self.base_p(
        self.base_q(x) * self.base_lambda * mask
    )
    return result
```

Output: `P_curr(λ_curr · Q_curr(x) · mask) − P_base(λ_base · Q_base(x) · mask)`

### 4.4 Mask Computation & Passing

**During training** (SDXL path):
```python
def get_mask_by_timestep(self, timestep, max_timestep, max_rank, min_rank=1, alpha=1):
    r = int(((max_timestep - timestep) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    sigma_mask = torch.zeros((1, self.config.lora_rank))
    sigma_mask[:, :r] = 1.0
    return sigma_mask
```

The mask is passed to the UNet via `cross_attention_kwargs={"sigma_mask": sigma_mask}`, which diffusers pipes through to the custom attention processors.

**During inference:** The same mask computation happens at each denoising step in the custom pipeline.

### 4.5 FLUX-Specific Notes

- FLUX uses only Vanilla T-LoRA (no Ortho-LoRA needed)
- Reason: LoRA adapters trained on MM-DiT (FLUX architecture) are **already full-rank** — orthogonal initialization is unnecessary
- Uses Prodigy optimizer with lr=1.0 (vs Adam with lr=1e-4 for SDXL)
- The `--tlora` flag toggles timestep masking; `--min_rank` controls minimum rank

### 4.6 Training Hyperparameters

| Parameter | SD-XL | FLUX-1.dev |
|-----------|-------|-----------|
| Optimizer | Adam | Prodigy |
| Learning rate | 1e-4 | 1.0 |
| Betas | (0.9, 0.999) | — |
| Weight decay | 1e-4 | — |
| Training steps | 800 (T-LoRA), 500 (LoRA) | 500 (T-LoRA), 400 (LoRA) |
| Batch size | 1 | 1 |
| r_min | 50% of r | 50% of r |
| sig_type | "last" | N/A (vanilla only) |

**Note:** T-LoRA requires ~60% more training steps than LoRA due to non-zero initialization requiring more time to diverge from the init point.

---

## 5. LyCORIS Integration

### 5.1 Module Registration

T-LoRA is registered at multiple levels in LyCORIS:

| Registration Point | Key | Value |
|-------------------|-----|-------|
| `modules/__init__.py` | MODULE_LIST | `TLoraModule` |
| `wrapper.py` network_module_dict | `"tlora"` | `TLoraModule` |
| `config_sdk.py` ALGO_REGISTRY | `"tlora"` | `AlgoSpec(...)` |
| Top-level `__init__.py` | exports | `TLoraModule`, `set_timestep_mask`, `get_timestep_mask`, `clear_timestep_mask`, `compute_timestep_mask` |

Usage: `algo=tlora` in LyCORIS config or `--network_args algo=tlora` in kohya-based training.

### 5.2 Key Differences from Official Repo

| Feature | Official T-LoRA | LyCORIS T-LoRA |
|---------|----------------|----------------|
| Module types | Linear only | Linear + Conv1d/2d/3d |
| Architecture | Monolithic attention processors | Per-layer modular wrapping |
| SVD source | Random matrix only (OrthogonalLoRA) or weight only (LOrthogonal) | Both via `use_data_init` flag |
| Base storage | `copy.deepcopy()` of nn.Linear modules | `register_buffer()` (more memory-efficient) |
| Mask passing | Explicit via `cross_attention_kwargs` | Module-level global dict `_timestep_mask_storage` |
| Dropout | None | dropout, rank_dropout, module_dropout |
| Weight merge | Not supported | Full merge_to(), onfly_merge() support |
| Max norm | Not supported | `apply_max_norm()` on lambda_layer |
| Scaling | None | Standard `alpha / lora_dim` |
| State dict | Standard PyTorch | Custom serialization (q_layer, p_layer, lambda_layer, alpha only) |

### 5.3 Timestep Mask Flow in LyCORIS

LyCORIS uses a **module-level global storage pattern** instead of explicit function arguments:

```python
# Module-level storage (in lycoris/modules/tlora.py)
_timestep_mask_storage: dict[int, torch.Tensor] = {}

# Before model forward (training framework's responsibility):
from lycoris.modules.tlora import set_timestep_mask, compute_timestep_mask

mask = compute_timestep_mask(
    timestep=current_timestep,
    max_timestep=1000,
    max_rank=lora_dim,
    min_rank=1,
    alpha=1.0,
)
set_timestep_mask(mask, group_id=0)

# Inside each TLoraModule.forward():
mask = get_timestep_mask(self.mask_group_id)  # reads from global storage
```

**Multi-network support:** `mask_group_id` allows different T-LoRA networks to use different masks.

### 5.4 LyCORIS Configuration Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `algo` | str | — | Must be `"tlora"` |
| `dim` | int | 4 | Maximum LoRA rank |
| `alpha` | float | 1 | Scale = alpha/dim |
| `sig_type` | str | `"principal"` | SVD component selection: `"principal"`, `"last"`, `"middle"` |
| `use_data_init` | bool | True | True = SVD of original weights; False = SVD of random matrix |
| `mask_group_id` | int | 0 | Group ID for multi-network mask scenarios |
| `bypass_mode` | bool | False | Use `Y = WX + ΔWX` instead of `Y = (W+ΔW)X` (auto-enabled for quantized layers) |
| `dropout` | float | 0.0 | Standard dropout |
| `rank_dropout` | float | 0.0 | Rank-wise dropout |
| `module_dropout` | float | 0.0 | Module-level dropout |

### 5.5 LyCORIS Implementation Quality Notes

- **3 commits, just merged** (Feb 12, 2026) — very fresh code
- **Zero test coverage** — TLoraModule is missing from the test module list in `test/module.py`
- **Conv bug was caught and fixed** in commit `ee78542` (wrong kw_dict for down_op)
- **State dict loading fix** in commit `e50775d` (base buffers were being overwritten on load, breaking residual subtraction)
- The `sig_type` default differs: LyCORIS uses `"principal"` vs the paper's recommended `"last"`

---

## 6. Experimental Results

### 6.1 T-LoRA vs Standard LoRA (SD-XL)

| Method | r=4 IS/TS | r=8 IS/TS | r=16 IS/TS | r=32 IS/TS | r=64 IS/TS |
|--------|-----------|-----------|------------|------------|------------|
| LoRA | .890/.250 | .897/.249 | .900/.243 | .901/.238 | .901/.232 |
| Vanilla T-LoRA | .894/.259 | .892/.261 | .902/.256 | .904/.248 | .902/.240 |
| T-LoRA | .899/.255 | .897/.260 | .897/.260 | .899/.259 | .900/.256 |

**Key findings:**
- T-LoRA improves Text Similarity (TS) at ALL ranks while maintaining Image Similarity (IS) within 0.003
- Standard LoRA's TS **degrades** as rank increases (0.250→0.232), while T-LoRA maintains stable TS
- **T-LoRA's advantage grows with rank** — at r=64, TS improves by +0.024 (0.256 vs 0.232)

### 6.2 T-LoRA vs Other Methods (SD-XL, rank 64)

| Method | DINO-IS | IS | TS |
|--------|---------|----|----|
| **T-LoRA-64** | 0.802 | 0.900 | **0.256** |
| LoRA-64 | 0.808 | 0.901 | 0.232 |
| OFT-32 | 0.804 | 0.901 | 0.247 |
| OFT-16 | 0.802 | 0.899 | 0.212 |
| GSOFT-64 | 0.806 | 0.901 | 0.247 |
| GSOFT-32 | 0.804 | 0.901 | 0.212 |
| SVDiff | 0.414 | 0.753 | 0.295 |

T-LoRA achieves the best TS among all methods that actually preserve the concept (SVDiff has high TS but catastrophically low IS).

### 6.3 User Study (1800 human assessments)

| T-LoRA-64 vs | Concept (T-LoRA wins) | Text (T-LoRA wins) | Overall (T-LoRA wins) |
|-------------|----------------------|--------------------|-----------------------|
| LoRA-64 | 39.3% | **71.0%** | **67.3%** |
| OFT-32 | 52.5% | **58.3%** | **63.5%** |
| GSOFT-64 | 49.0% | **61.5%** | **60.3%** |
| Ortho-LoRA-64 | 50.3% | **58.5%** | **59.3%** |
| Vanilla T-LoRA-64 | 51.7% | **60.7%** | **60.3%** |

Humans significantly prefer T-LoRA overall despite LoRA winning on concept preservation (due to overfitting — it literally reproduces the training image).

### 6.4 Multi-Image Results

| Method | 1 img TS | 2 img TS | 3 img TS |
|--------|----------|----------|----------|
| LoRA-64 | 0.232 | 0.243 | 0.251 |
| T-LoRA-64 | **0.256** | **0.262** | 0.263 |

**T-LoRA trained on 1 image surpasses LoRA trained on 2-3 images in text similarity.**

### 6.5 FLUX-1.dev Results

Only Vanilla T-LoRA (no Ortho-LoRA needed since FLUX LoRA matrices are already full-rank):
- T-LoRA achieves better prompt alignment and more consistent ambiance than standard LoRA
- The benefit is more modest than on SD-XL, likely because FLUX already has better rank utilization

### 6.6 Key Ablation Findings

1. **r_min = 50% of r** is optimal (25% loses concept fidelity, 75% reduces overfitting benefit)
2. **"last" SVD components from random R** are optimal for initialization (top components overfit, actual-weight bottom components are too small)
3. Ortho-LoRA is essential at high ranks (r≥32) where standard LoRA has low effective rank, making masking ineffective
4. For FLUX/MM-DiT, Ortho-LoRA is unnecessary — LoRA is already full-rank on these architectures

---

## 7. Blissful Tuner Integration Assessment

### 7.1 Current LoRA Architecture

Blissful Tuner has a clean, layered LoRA architecture:

- **`src/musubi_tuner/networks/lora.py`** (~1354 lines): Core `LoRAModule`, `LoRAInfModule`, `DoRALayer`, `LoRANetwork`
- **Architecture wrappers** (`lora_wan.py`, `lora_flux_2.py`, etc.): Thin wrappers specifying target modules
- **`networks/lycoris.py`**: Adapter bridge to external LyCORIS library
- **Module loading**: `--network_module` imports via `importlib`, expecting `create_network()` / `create_network_from_weights()`

### 7.2 LoRA Forward Pass Signature Constraint

The fundamental integration challenge:

```python
# Current LoRA forward (replaces nn.Linear.forward):
def forward(self, x):
    org_forwarded = self.org_forward(x)
    lx = self.lora_down(x)
    lx = self.lora_up(lx)
    return org_forwarded + lx * self.multiplier * self.scale
```

This takes **only** `x` — no timestep. But the `LoRANetwork` has lifecycle hooks:

```python
network.on_step_start()  # Called before each training step (line 2406)
network.on_epoch_start()  # Called before each epoch
```

### 7.3 Timestep Injection Approaches

**Approach A: Network-level state (recommended)**
```python
# In training loop, after sampling timesteps:
accelerator.unwrap_model(network).set_timestep(timesteps)

# In TLoRAModule.forward():
mask = self.network.get_current_mask()
```

**Approach B: Module-level global storage (LyCORIS pattern)**
```python
from networks.tlora import set_timestep_mask, compute_timestep_mask
mask = compute_timestep_mask(timestep, max_timestep, lora_dim, min_rank)
set_timestep_mask(mask)
# All T-LoRA modules read from global storage in forward()
```

**Approach C: Extend on_step_start() to accept timesteps**
```python
# Modify training loop:
accelerator.unwrap_model(network).on_step_start(timesteps=timesteps)
```

### 7.4 Required Changes for Native Implementation

#### New Files
- `src/musubi_tuner/networks/tlora.py` — Core T-LoRA module + network (~400-600 lines)
- Architecture wrappers (e.g., `tlora_wan.py`, `tlora_flux_2.py`) — Thin wrappers (~60 lines each)

#### Training Loop Changes (`hv_train_network.py`)
Minimal — add timestep broadcasting after line 2414:
```python
timesteps = ...  # already sampled
accelerator.unwrap_model(network).set_timestep(timesteps)
```
This could be done generically in `on_step_start()` or as a separate call.

#### Key Module Design

```python
class TLoRAModule(nn.Module):
    def __init__(self, lora_name, org_module, multiplier, lora_dim, alpha,
                 sig_type="last", use_data_init=False, min_rank_ratio=0.5):
        # P/Q projections (replacing lora_up/lora_down)
        self.q_layer = nn.Linear(in_dim, lora_dim, bias=False)
        self.p_layer = nn.Linear(lora_dim, out_dim, bias=False)
        self.lambda_layer = nn.Parameter(torch.ones(1, lora_dim))

        # SVD initialization
        self._initialize_svd(org_module.weight, sig_type, use_data_init)

        # Frozen base copies for residual subtraction
        self.register_buffer("base_q", self.q_layer.weight.data.clone())
        self.register_buffer("base_p", self.p_layer.weight.data.clone())
        self.register_buffer("base_lambda", self.lambda_layer.data.clone())

    def forward(self, x):
        mask = self._get_current_mask()  # from network-level state

        # Trained path
        q_out = self.q_layer(x) * self.lambda_layer * mask
        p_out = self.p_layer(q_out)

        # Base path (frozen) — subtracted
        with torch.no_grad():
            base_q_out = F.linear(x, self.base_q) * self.base_lambda * mask
            base_p_out = F.linear(base_q_out, self.base_p)

        orig = self.org_forward(x)
        return orig + (p_out - base_p_out) * self.multiplier * self.scale
```

#### Checkpoint Format

Additional keys per module:
- `{name}.q_layer.weight` (replaces `lora_down.weight`)
- `{name}.p_layer.weight` (replaces `lora_up.weight`)
- `{name}.lambda_layer` (new — diagonal scaling)
- `{name}.alpha` (same as standard LoRA)
- Network-level: `use_tlora_flag` buffer for auto-detection

#### Inference Considerations

**Static weight merging is NOT possible** for T-LoRA because the LoRA contribution varies per timestep. Options:
1. **Keep hooks active during inference** (no merge) — correct but slower
2. **Precompute merged weights for discrete timestep bins** — approximation
3. **Per-step merge/unmerge** — expensive but exact

For Blissful Tuner's inference scripts, option 1 is simplest and most correct.

### 7.5 Architecture Compatibility

| Architecture | Timestep Access | Ortho-LoRA Needed? | Integration Difficulty |
|-------------|----------------|-------------------|----------------------|
| WAN 2.1/2.2 | `t` in `WanModel.forward()` | Likely yes (UNet-like) | Low |
| HunyuanVideo | `timestep` in transformer | Likely yes | Low |
| FLUX.2 | `timestep` in Flux2Model | **No** (MM-DiT already full-rank) | Low |
| FLUX.1 Kontext | `timestep` in FluxModel | **No** | Low |
| FramePack | `timestep` in transformer | TBD | Medium |
| Qwen-Image | `timestep` in QwenImage | TBD | Medium |
| Z-Image | `timestep` in model | TBD | Medium |

All architectures have timesteps available at the model level. The `on_step_start()` hook in the training loop is architecture-independent.

### 7.6 Compatibility with Existing Features

| Feature | Compatible? | Notes |
|---------|------------|-------|
| DoRA | ⚠️ Needs work | DoRA magnitude computation would need timestep-aware delta |
| RS-LoRA | ✅ | Just affects alpha/dim scaling |
| LoRA+ | ✅ | Just affects learning rates |
| Prior preservation | ✅ | `set_enabled(False)` disables entire module |
| Mask-weighted loss | ✅ | Orthogonal — mask loss operates on MSE, T-LoRA on LoRA structure |
| Gradient checkpointing | ✅ | Standard torch mechanism |
| fp8_base | ✅ | T-LoRA operates on LoRA params, not base weights |
| torch.compile | ⚠️ TBD | Dynamic mask shape could cause recompilation |

---

## 8. Integration vs LyCORIS Passthrough

### Option A: Native Implementation

**Pros:**
- Full control over implementation quality and optimization
- Can deeply integrate with Blissful Tuner's training loop (timestep passing, loss integration)
- Consistent UX with existing LoRA args (`--network_dim`, `--network_alpha`, etc.)
- Can optimize for video architectures (WAN, HV) which T-LoRA hasn't been tested on
- Better inference integration (Blissful Tuner's custom pipelines)

**Cons:**
- ~500-800 lines of new code to write and maintain
- Need to implement for all supported architectures
- No upstream bug fixes from LyCORIS

### Option B: LyCORIS Passthrough (via `networks/lycoris.py`)

**Pros:**
- Already have a LyCORIS adapter bridge
- T-LoRA implementation is there and merged
- Less code to maintain

**Cons:**
- LyCORIS T-LoRA has **zero tests** and is 1 day old (3 commits, 2 were bug fixes)
- `sig_type` default differs from paper recommendation (`"principal"` vs `"last"`)
- Timestep mask must be set externally via `set_timestep_mask()` — training loop integration is our responsibility either way
- Global storage pattern is less clean than network-level state
- LyCORIS T-LoRA doesn't support the `alpha_rank_scale` power exponent
- No video architecture testing
- Dependency on external library's API stability

### Option C: Hybrid — Use LyCORIS for Experimentation, Native for Production

Start with LyCORIS passthrough to validate T-LoRA on video architectures. If results are promising, implement natively for production quality.

---

## 9. Open Questions & Risks

### Untested Territory

1. **Video diffusion models:** T-LoRA has only been tested on SD-XL and FLUX-1.dev (both image-only). Behavior on video architectures (WAN, HV, FramePack) is completely unknown.
2. **Flow matching:** T-LoRA was designed for DDPM-style diffusion. WAN 2.2 uses flow matching with `--timestep_sampling shift` and `--discrete_flow_shift 12.0`. The rank schedule formula may need adaptation for flow matching timestep distributions.
3. **Dual-model architecture:** WAN 2.2's dual high/low noise model with boundary switching may interact with rank masking in unexpected ways.
4. **Multi-image training:** T-LoRA's primary benefit is for single-image customization. With larger datasets (common for video LoRA), the overfitting problem is less severe, potentially reducing T-LoRA's advantage.
5. **Training cost:** T-LoRA requires ~60% more steps. For video training (already expensive), this is significant.

### Technical Risks

1. **SVD initialization cost:** Computing SVD for every adapted layer adds initialization overhead. For large models (14B WAN), this could be significant.
2. **Mask-dependent gradients:** Only unmasked rank dimensions receive gradients at each step. This means each rank dimension is trained on a subset of timesteps, potentially leading to uneven convergence.
3. **torch.compile compatibility:** Dynamic mask shapes may trigger recompilation. Could be mitigated with fixed-size masks using zeros.
4. **Inference overhead:** No static merge means T-LoRA inference is always slower than merged LoRA. For video generation (many frames), this matters.

### LyCORIS-Specific Risks

1. **Fresh code (1 day old):** Two of three commits were bug fixes. More bugs are likely.
2. **No tests:** Zero test coverage in LyCORIS test suite.
3. **Default mismatch:** LyCORIS defaults to `sig_type="principal"` but the paper recommends `"last"`.
4. **Memory leaks:** Global `_timestep_mask_storage` dict is never explicitly cleaned in training loops.

---

## 10. Recommendation

### Assessment: Worth Investigating, Not Production-Ready Yet

**T-LoRA addresses a real problem** — single-image/few-shot overfitting is a genuine pain point in LoRA training. The paper's results are compelling, showing clear improvements in text alignment without sacrificing concept fidelity.

**However, several factors suggest waiting:**

1. **Untested on video:** All results are image-only. Video training has different dynamics (temporal coherence, longer training, larger datasets).
2. **LyCORIS implementation is immature:** 1 day old, 2/3 commits were bug fixes, zero tests.
3. **Flow matching adaptation needed:** The rank schedule assumes DDPM-style timesteps. Adaptation for flow matching (used by WAN 2.2, FLUX.2) needs research.
4. **Training cost:** 60% more steps is significant for video workflows.

### Suggested Path Forward

1. **Short-term (now):** This document serves as the reference. No code changes needed.
2. **Experimentation phase:** Try LyCORIS passthrough with T-LoRA on a WAN 2.2 image LoRA to validate the concept on our architecture. Key things to test:
   - Does the rank schedule work with flow matching timestep distributions?
   - What `r_min` ratio works for video models?
   - Is Ortho-LoRA needed for WAN (14B model)?
   - How does training time scale?
3. **If promising:** Implement natively in Blissful Tuner with proper architecture support, video-aware rank scheduling, and integration with mask-weighted loss.
4. **If not:** The technique remains image-focused and we can revisit if the paper gets follow-up work on video.

### Key Decision Points

- **If overfitting is a user pain point** (especially single-subject video LoRAs): Prioritize experimentation
- **If users mostly train with larger datasets**: Lower priority, as T-LoRA's benefit diminishes
- **If LyCORIS integration is needed anyway**: T-LoRA comes "for free" through the existing adapter

---

## Appendix A: Key File Locations

| File | Location |
|------|----------|
| T-LoRA paper (markdown) | `/Users/dustin/Downloads/T-LORA.md` |
| Official T-LoRA repo | `/Users/dustin/T-LoRA/` |
| LyCORIS repo | `~/LyCORIS/` |
| LyCORIS T-LoRA module | `~/LyCORIS/lycoris/modules/tlora.py` |
| LyCORIS T-LoRA exports | `~/LyCORIS/lycoris/__init__.py` (set_timestep_mask, etc.) |
| Blissful LoRA core | `/Users/dustin/blissful-tuner/src/musubi_tuner/networks/lora.py` |
| Blissful LyCORIS adapter | `/Users/dustin/blissful-tuner/src/musubi_tuner/networks/lycoris.py` |
| Blissful training loop | `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py` |

## Appendix B: Paper Citation

```bibtex
@article{soboleva2025tlora,
  title={T-LoRA: Single Image Diffusion Model Customization Without Overfitting},
  author={Soboleva, Vera and Alanov, Aibek and Kuznetsov, Andrey and Sobolev, Konstantin},
  journal={arXiv preprint arXiv:2507.05964},
  year={2025}
}
```

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **IS (Image Similarity)** | CLIP ViT-B/32 cosine similarity between generated and training images |
| **DINO-IS** | Same as IS but using DINO embeddings |
| **TS (Text Similarity)** | CLIP cosine similarity between prompt and generated image |
| **r(t)** | Active rank at timestep t |
| **r_min** | Minimum rank at maximum noise timestep |
| **Ortho-LoRA** | SVD-based orthogonal initialization ensuring independent rank dimensions |
| **Residual base subtraction** | Technique ensuring zero LoRA delta at init despite non-zero P/Q |
| **sig_type** | Which singular vectors to use: "principal" (top), "last" (bottom), "middle" |
| **alpha_rank_scale** | Power exponent for rank schedule curve (1.0 = linear) |
