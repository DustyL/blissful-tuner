# Network Configuration: `network_args` Reference

**For musubi-tuner / blissful-tuner LoRA training**

This document covers all `network_args` options available when configuring LoRA networks.

---

## How `network_args` Works

This repo passes `--network_args` as key/value strings (e.g., `use_dora=True`). In training scripts, each entry is split on `=` and forwarded into the network module's `create_arch_network(..., **net_kwargs)` / `create_network(..., **kwargs)`.

**Example:**
```toml
[network]
network_module = "networks.lora_wan"
network_dim = 128
network_alpha = 64
network_args = ["use_rslora=True", "use_dora=True", "loraplus_lr_ratio=8"]
```

**Command-line equivalent:**
```bash
--network_args "use_rslora=True" "use_dora=True" "loraplus_lr_ratio=8"
```

---

## Table of Contents

1. [RS-LoRA](#rs-lora-use_rslora)
2. [DoRA](#dora-use_dora)
3. [LoRA Init](#lora-init-init_lora_weights)
4. [Rank / Alpha Patterns](#rank--alpha-patterns-rank_pattern--alpha_pattern)
5. [LoRA+](#lora-loraplus_lr_ratio)
6. [Dropout Options](#dropout-options)
7. [Conv-Specific Settings](#conv-specific-settings)
8. [Module Selection Patterns](#module-selection-patterns)
9. [Architecture-Specific Options](#architecture-specific-options)
10. [Utility Options](#utility-options)
11. [Quick Reference Table](#quick-reference-table)
12. [Known-Good Combinations](#known-good-combinations)

---

## RS-LoRA (`use_rslora`)

**Option:** `use_rslora=True|False` (default: `False`)

### What it does

Changes LoRA scaling from standard to rank-stabilized:

| Mode | Scaling Formula | alpha=0 Behavior |
|------|-----------------|------------------|
| Standard LoRA | `scale = alpha / r` | scale = 1.0 |
| RS-LoRA | `scale = alpha / sqrt(r)` | scale = 1.0 |

### When to use

- Training with **higher ranks** (dim > 32) where gradient magnitudes vary significantly
- When comparing runs with **different `network_dim`** values (RS-LoRA normalizes the effect)
- When you want **more stable training dynamics** across layers

### Technical notes / gotchas

| Topic | Detail |
|-------|--------|
| **Weight compatibility** | RS-LoRA vs standard LoRA is a semantic difference. Loading weights with the wrong setting produces incorrect scaling. The implementation enforces this with a **hard error** on mismatch. |
| **alpha=0 persistence** | When `alpha=0` under RS-LoRA, the saved alpha becomes `sqrt(r)` so that `alpha/sqrt(r)=1`. External tools that ignore `use_rslora_flag` and assume `alpha/r` will misinterpret these weights. |
| **Regularization** | `scale_weight_norms` (max-norm regularization) uses RS-LoRA scaling when enabled. |
| **Flag storage** | Weights include a network-level `use_rslora_flag` buffer for unambiguous detection on load. |
| **Suspicious alpha hint** | If loading fails due to flag mismatch and >50% of alphas equal `sqrt(dim)`, the error message hints that RS-LoRA may have been used. |

### Example

```toml
network_args = ["use_rslora=True"]
```

**Scaling comparison (dim=128, alpha=64):**
- Standard: `64/128 = 0.5`
- RS-LoRA: `64/sqrt(128) = 64/11.31 ≈ 5.66`

---

## DoRA (`use_dora`)

**Option:** `use_dora=True|False` (default: `False`)

### What it does

Enables **DoRA (Weight-Decomposed Low-Rank Adaptation)**:

- Adds a **per-layer magnitude vector** (`dora_layer.weight`)
- Applies magnitude normalization using `||W + ΔW||` (row-wise L2 norm, detached)
- Uses **PEFT-style bias handling** (bias not scaled by the magnitude term)

**DoRA formula:**
```
weight_norm = ||W + scaling * (B @ A)||_row  (detached)
mag_norm_scale = magnitude / weight_norm
delta = (mag_norm_scale - 1) * base_wo_bias + mag_norm_scale * lora_out * scaling
```

### Supported layers / limitations

| Constraint | Behavior |
|------------|----------|
| **Linear layers only** | DoRA enabled only for `torch.nn.Linear` |
| **Conv layers** | DoRA disabled (all Conv types, including Conv1x1) |
| **split_dims** | DoRA disabled when qkv-style ModuleLists are used |
| **dropout > 0** | DoRA disabled for that module |
| **rank_dropout > 0** | DoRA disabled for that module |
| **module_dropout** | Allowed with DoRA |

When DoRA is disabled for a module, it **silently falls back to standard LoRA**. A summary is logged at network creation:
```
DoRA enabled on 45 modules, disabled on: 12 non-Linear, 3 dropout
```

### When to use

- When standard LoRA produces **inconsistent quality** across layers
- For **fine-grained control** over weight magnitudes
- Training **style or identity LoRAs** where magnitude decomposition helps

### Technical notes / gotchas

| Topic | Detail |
|-------|--------|
| **Saved weights** | Includes network-level `use_dora_flag` and per-module `dora_layer.weight` tensors. |
| **Fallback detection** | For older/external weights without the flag, DoRA is detected by scanning for `dora_layer.weight` keys. |
| **Merging** | Magnitude is read from the weights file. If missing for a DoRA-enabled module, this is treated as an error (prevents silent “all-ones” magnitudes). |
| **Mismatch: network expects DoRA, weights lack it** | **Hard error** (to avoid uninitialized magnitudes). |
| **Mismatch: network doesn't expect DoRA, weights have it** | **Warning** (treats as standard LoRA; DoRA magnitudes ignored; `use_dora_flag` is not allowed to flip to `True`). |
| **Memory-efficient norm** | Uses expanded-norm formula without materializing B@A during forward pass. |
| **`lora_multiplier=0` semantics** | Treated as a true no-op (DoRA included): base output/weights are unchanged. |

### Example

```toml
network_args = ["use_dora=True"]
```

---

## LoRA Init (`init_lora_weights`)

**Option:** `init_lora_weights=kaiming|orthogonal|pissa|pissa_niter_<N>|true` (default: `kaiming`)

### What it does

Selects the initialization scheme for standard LoRA `lora_down` / `lora_up` weights:

| Value | Behavior |
|-------|----------|
| `kaiming` | Current default: `lora_down` uses Kaiming uniform, `lora_up` starts at zero |
| `true` | Alias for `kaiming`, matching PEFT's default-style config spelling |
| `orthogonal` | QR-based PEFT orthogonal init: both matrices start nonzero while `lora_up @ lora_down == 0` |
| `pissa` | Principal Singular values & Singular vectors Adaptation: SVD the base weight, seed `lora_A` / `lora_B` with the principal-rank-r components, and **residualize the base weight** so the live model produces identical output at step 0 but training updates the principal directions directly. Single-DiT only in v1; see "PiSSA safety contract" below. |
| `pissa_niter_<N>` | PiSSA with Newton-Schulz approximate SVD via `torch.svd_lowrank(weight, r, niter=N)`. Faster than full SVD for big weights (3072×3072+ attention projections, 4096×16384 MLPs). Niter values follow PEFT convention; example `pissa_niter_5`. The speed/accuracy tradeoff for diffusion DiT weights is not characterized in this codebase yet — values are PEFT-compatible spellings, not tuned blissful-tuner knobs. |

`orthogonal` is recommended mainly for higher ranks (`network_dim >= 16`). It is valid at any even rank, but odd ranks raise because the algorithm splits an orthogonal matrix into even/odd row groups.

`pissa` empirically gives faster convergence than random init at the same rank because the principal-r SVD components are a strong starting point — diffusion DiT weights have heavy-tailed singular value distributions where the top-r captures most of the signal.

### Technical notes / gotchas

| Topic | Detail |
|-------|--------|
| **Default compatibility** | Omitted `init_lora_weights` keeps the historical Kaiming+zero behavior. |
| **Even rank required** | `orthogonal` requires even `network_dim` for Linear targets. Use an even rank or `init_lora_weights=kaiming`. |
| **Conv2d targets (orthogonal)** | Conv2d LoRA modules fall back to Kaiming init with a counted warning. |
| **Conv2d targets (PiSSA)** | Conv2d **hard-rejects** for `pissa*` — see "PiSSA safety contract" below for why. |
| **split_dims (orthogonal)** | QKV split modules apply orthogonal init independently per split. |
| **split_dims (PiSSA)** | QKV split **hard-rejects** for `pissa*`. |
| **Saved weights** | Safetensors metadata records `ss_init_lora_weights=<scheme>`. For `pissa*`, also records `ss_base_sha256` (the file hash of the base DiT trained against — see Tier 2 #6a write infrastructure). No tensor flag is needed because init does not change load-time merge math, but PiSSA-tagged LoRAs receive stricter base-hash validation at hotswap and merge time (see "PiSSA safety contract"). |
| **Scope** | `orthogonal` and `pissa*` are standard-LoRA-only. Neither is applied to LoHa/LoKr/LyCORIS modules. |
| **Base mutation (PiSSA only)** | PiSSA mutates `org_module.weight.data` in place at LoRAModule init time, replacing the original weight with the residual `W - scale * lora_B @ lora_A`. Forward equivalence at step 0 holds because `residual + scale * lora_B @ lora_A == original`. This in-place mutation is what couples PiSSA to the exact base file used at training time. |

### Examples

```toml
# Standard kaiming (default)
network_args = ["init_lora_weights=kaiming"]

# Orthogonal for higher-rank LoRA
network_args = ["init_lora_weights=orthogonal"]

# PiSSA with full SVD
network_args = ["init_lora_weights=pissa"]

# PiSSA with Newton-Schulz approximate SVD (faster on big DiT weights)
network_args = ["init_lora_weights=pissa_niter_5"]
```

### PiSSA safety contract

PiSSA mutates the base weights at LoRAModule init time and the math is load-bearingly coupled to the exact base bytes used at training time (residualized base assumption breaks under any other base). The contract closes ten distinct failure modes across training, hotswap, and offline merge surfaces. **Detection is metadata-driven** — `ss_init_lora_weights.startswith("pissa")` is the canonical predicate; a fully metadata-stripped PiSSA-trained safetensors has no detectable PiSSA signal and falls through to the non-PiSSA back-compat path.

| User action | Detection point | Outcome |
|---|---|---|
| `init_lora_weights=pissa*` + Conv2d target module | `LoRAModule.__init__` | Hard reject before allocation |
| `init_lora_weights=pissa*` + `split_dims=...` (QKV split) | `LoRAModule.__init__` | Hard reject before allocation |
| `init_lora_weights=pissa*` + `use_dora=True` | `validate_pissa_training_args` (training start) AND `LoRAModule.__init__` (defense in depth) | Hard reject before model load |
| `init_lora_weights=pissa*` + WAN dual-expert (`--dit_high_noise`) | `validate_pissa_training_args` (training start) | Hard reject before model load — blocked on Tier 2 #6a-2 follow-up for per-expert metadata keys |
| `init_lora_weights=pissa*` + `--resume` (checkpoint resume) | `validate_pissa_training_args` (training start) | Hard reject before model load — re-residualization footgun |
| `init_lora_weights=pissa*` + fp8 base tensor | `_init_pissa_lora_pair` | TypeError before residual mutation (LoRA module is being constructed by then; not a pre-allocation reject) |
| **Hotswap** PiSSA-tagged LoRA + matching `ss_base_sha256` | `_check_lora_base_hash(init_pissa=True)` | Silent OK |
| **Hotswap** PiSSA-tagged LoRA + mismatched `ss_base_sha256` | `_check_lora_base_hash(init_pissa=True)` | Hard reject **always** (math is provably wrong; no `--no-hotswap_strict_base_hash` opt-out for this case) |
| **Hotswap** PiSSA-tagged LoRA + missing `ss_base_sha256` + `--hotswap_strict_base_hash` (default ON) | `_check_lora_base_hash(init_pissa=True)` | Hard reject with "base-hash metadata appears stripped" |
| **Hotswap** PiSSA-tagged LoRA + missing `ss_base_sha256` + `--no-hotswap_strict_base_hash` | `_check_lora_base_hash(init_pissa=True)` | Warn with same message, proceed |
| **Offline merge** (`merge_lora.py`) PiSSA-tagged LoRA + mismatched `ss_base_sha256` | `_pissa_base_hash_preflight` (before `load_transformer`) | Raises always |
| **Offline merge** PiSSA-tagged LoRA + missing `ss_base_sha256` | `_pissa_base_hash_preflight` (before `load_transformer`) | Raises always — **strict-by-default for offline merge**, no opt-out flag in v1 because the output is a persisted derived checkpoint |

**Hotswap vs offline merge asymmetry:** Hotswap is interactive runtime, so missing-hash warnings can be downgraded via `--no-hotswap_strict_base_hash`. Offline merge writes a persisted derived checkpoint, so both missing and mismatched cases raise unconditionally — there is no `--no-allow_pissa_missing_base_hash` flag in v1.

### When NOT to use PiSSA (empirical bounds, 2026-05-06)

PiSSA's correctness contract holds (see safety table above), but a real-weights training experiment narrowed when reaching for it actually helps. The implementation is *correct and available*; it is *empirically dispreferred for this specific configuration class*. Users on different optimizers or different adapter mixes should not infer from this section that PiSSA is broadly worse than kaiming.

**Disrecommended configuration:** `init_lora_weights=pissa*` together with **ProdigyPlusScheduleFree** as the optimizer, when **DoRA (`use_dora=True`) was previously load-bearing in the same pipeline**.

**Evidence (2026-05-05 v10-PiSSA-rank32 vs v9-rank32, FLUX.2-Klein-9B persona LoRA, 10000 steps each, identical dataset/seed/scale/rank):**

- `loss/average` final: **+7.0%** worse on v10 (PiSSA + no DoRA): 0.10529 → 0.11270
- `masked_loss/target` final: **+24.8%** worse on v10: 0.07968 → 0.09948 (the unsmoothed in-mask reconstruction component, where identity-relevant features live; this gap is much larger than the headline `loss/average` because the in-mask region is where DoRA's per-output-channel magnitude lever has the most leverage)
- Late-slope `[8001..10000]`: v9 still descending at −0.00081/1k; v10 essentially flat at −0.000029/1k. v9 had headroom; v10 saturated early.
- Prodigy `lr/d/unet plus` first-final-step: v9 step 88, **v10 step 47** at a final value 4.3% of v9's. The optimizer concluded "geometry near-optimal" *prematurely* under PiSSA's principal-direction init, locking in small-step exploration before the LoRA had room to discover the channel-magnitude structure DoRA would have provided.
- Qualitative: at adversarial seed+prompt combinations (e.g. formal-event with a seed where the base model leans feminine), v10's likeness LoRA captured facial features but did not stabilize gender-leaning channel priors. v9 corrected the same seed by step 1000 and held; v10 briefly corrected at step 1000 then drifted back to feminine presentation by step 5000+. This was *invisible to `loss/average` averaging* but obvious in cross-prompt-cross-seed sample inspection.

**Why this is narrower than "PiSSA + adaptive optimizers is bad":**

The mechanism evidenced by v10 is specific to **Prodigy's `d`-discovery interaction with PiSSA's principal-direction init**, not to adaptive optimizers in general. AdamW + hand-tuned LR + PiSSA, Adafactor + PiSSA, Lion + PiSSA — these are untested in this codebase and would behave differently. Prodigy's `d` is auto-discovered from gradient norms over recent steps; PiSSA puts initial LoRA gradients in a low-magnitude regime that triggers Prodigy's "near-optimal, freeze `d`" heuristic prematurely. AdamW's adaptivity is per-parameter momentum smoothing — a different mechanism with no direct parallel.

**Why the DoRA-loss component matters separately:**

PiSSA forbids DoRA in v1 (the math doesn't compose without PEFT's `pissa_decompose_dora` utility, deferred to a hypothetical Tier 2 #6d). For pipelines where DoRA is load-bearing — empirically true for likeness-class LoRAs on small datasets per v10's adversarial-seed evidence — adopting PiSSA forces a two-variable swap: PiSSA on, DoRA off. The +24.8% `masked_loss/target` gap is the *net* of (PiSSA-gain − DoRA-loss); since PiSSA's gain was at-best-neutral on early descent (per the d-freeze evidence), the gap is best attributed to the DoRA loss. **This is a hypothesis, not a proven mechanism** — clean attribution requires the PiSSA × DoRA 2×2 that's only possible after `pissa_decompose_dora` ships.

**Practical guidance:**

| Your pipeline | Recommendation |
|---|---|
| ProdigyPlusScheduleFree + DoRA-load-bearing likeness LoRA (DLAY-class) | Stay on `kaiming + use_dora=True`. PiSSA is empirically worse here. |
| Non-Prodigy optimizer (AdamW + hand-tuned LR, etc.) | PiSSA is *not characterized* under your conditions — the v10 evidence does not generalize. Run a small A/B if curious. |
| You don't use DoRA today | The PiSSA-no-DoRA combination is what v10 tested. The DoRA-loss component of v10's penalty does not apply to you, but the Prodigy-d-freeze component still does (under Prodigy specifically). |
| Fixed-LR optimizer + small dataset + low rank + short training | This is PiSSA's strongest theoretical case (PEFT paper regime). Untested in this codebase but plausibly useful. |

The PiSSA implementation is not deprecated. It will continue to be supported and the safety contract will be honored. This subsection exists so users don't repeat the v10 experiment unnecessarily on equivalent pipelines.

---

## Rank / Alpha Patterns (`rank_pattern` / `alpha_pattern`)

**Options:**
- `rank_pattern=<dict[str, int]>` (default: unset)
- `alpha_pattern=<dict[str, int|float]>` (default: unset)

### What they do

Override LoRA rank and alpha per target module at fresh network creation time. Patterns are regexes matched with `re.fullmatch()` against the module's dotted `original_name`, for example `block.attn.to_q`, `double_blocks.0.img_attn.qkv`, or `single_blocks.12.linear1`.

| Option | Effect |
|--------|--------|
| `rank_pattern` | Overrides `network_dim` for matching modules |
| `alpha_pattern` | Overrides `network_alpha` for matching modules that receive LoRA |

Non-matching modules continue to use `network_dim` / `network_alpha`. If multiple patterns match the same module, the first pattern in the dict wins. Python preserves dict insertion order, so put more specific rules before broader fallback rules when you want them to win.

### Technical notes / gotchas

| Topic | Detail |
|-------|--------|
| **Fresh creation only** | Patterns are used when creating a new LoRA. When loading existing weights, the ranks and alphas in the state dict are authoritative. |
| **Regex target** | Match against dotted `original_name`, not saved LoRA key names. Use `.*attn\.to_q` rather than `lora_unet_block_attn_to_q`. |
| **Match semantics** | `re.fullmatch()` + first-match-wins. Metadata records `ss_rank_pattern_match_semantics=original_name_fullmatch_first_match` for forward compatibility. |
| **Input format** | Both JSON-style dict strings and Python literal dict strings are accepted. Values must be positive; ranks must be integers. |
| **Conv2d targets** | `rank_pattern` can target Conv2d modules. There is no separate `conv_rank_pattern` / `conv_alpha_pattern` in this phase. |
| **Orthogonal init** | With `init_lora_weights=orthogonal`, each resolved rank must be even. Errors name the module and matching pattern. |
| **Scope** | Pattern overrides are standard-LoRA-only. They are rejected for LoHa/LoKr/LyCORIS module classes. |
| **Saved weights** | Safetensors metadata records compact JSON strings in `ss_rank_pattern` and `ss_alpha_pattern` when used. |

### Examples

```toml
# Higher rank on early double blocks, lower rank elsewhere
network_args = [
  "rank_pattern={'.*double_blocks\\.[0-7]\\..*': 32, '.*single_blocks\\..*': 16}",
  "alpha_pattern={'.*double_blocks\\.[0-7]\\..*': 32, '.*single_blocks\\..*': 16}",
]
```

```toml
# Fullmatch means this targets any dotted name ending in attn.to_q
network_args = ["rank_pattern={'.*attn\\.to_q': 16}"]
```

---

## LoRA+ (`loraplus_lr_ratio`)

**Option:** `loraplus_lr_ratio=<float>` (default: unset/disabled)

### What it does

Applies a **learning-rate multiplier** to LoRA-B (up) parameters during optimizer param group creation:

| Parameter Group | Learning Rate |
|-----------------|---------------|
| LoRA-A (down) | `learning_rate` |
| LoRA-B (up) | `learning_rate * loraplus_lr_ratio` |

### When to use

- When you want **faster convergence** without changing other hyperparameters
- Typical ratios: **4-16** (original paper recommends 16, but 4-8 is often more stable)

### Technical notes

- Implemented via `LoRANetwork.set_loraplus_lr_ratio()` and applied in `prepare_optimizer_params()`
- Only affects LoRA parameters, not text encoder or other trainable params

### Example

```toml
[network]
network_args = ["loraplus_lr_ratio=8"]

[optimizer]
learning_rate = 5e-5
# Effective: LoRA-A = 5e-5, LoRA-B = 4e-4
```

---

## Dropout Options

### `rank_dropout`

**Option:** `rank_dropout=<float>` (default: None/disabled)

Applies dropout to the **rank dimension** during training. Randomly zeros out entire rank slices.

**DoRA interaction:** If `rank_dropout > 0`, DoRA is **disabled** for that module.

```toml
network_args = ["rank_dropout=0.1"]  # 10% rank dropout
```

### `module_dropout`

**Option:** `module_dropout=<float>` (default: None/disabled)

Applies dropout to the **entire LoRA module output** during training. With probability `module_dropout`, the LoRA contribution is zeroed (only base model output used).

**DoRA interaction:** `module_dropout` is **allowed** with DoRA.

```toml
network_args = ["module_dropout=0.1"]  # 10% module dropout
```

### Dropout + DoRA compatibility

| Dropout Type | DoRA Compatible |
|--------------|-----------------|
| `dropout` (neuron) | ❌ DoRA disabled |
| `rank_dropout` | ❌ DoRA disabled |
| `module_dropout` | ✅ DoRA allowed |

**Note:** `dropout=0.0` and `rank_dropout=0.0` are treated as disabled (DoRA remains enabled).

---

## Conv-Specific Settings

### `conv_dim`

**Option:** `conv_dim=<int>` (default: same as `network_dim`)

Sets a **separate rank** for Conv2d layers. Useful when you want different capacity for convolutions vs linear layers.

```toml
network_args = ["conv_dim=32"]  # Conv layers use rank 32
```

### `conv_alpha`

**Option:** `conv_alpha=<float>` (default: same as `network_alpha`)

Sets a **separate alpha** for Conv2d layers.

```toml
network_args = ["conv_dim=32", "conv_alpha=16"]
```

---

## Module Selection Patterns

Control which submodules receive LoRA adapters using regex patterns.

### `exclude_patterns`

**Option:** `exclude_patterns=<python-literal-list-of-regex>`

A list of regex patterns matched against the module's original dotted name (before `.` → `_` conversion) using `re.fullmatch()`. Patterns must match the **entire** module name, not just a prefix — use `.*` anchors (e.g., `'.*attn.*'` not `'attn'`). If a module matches an exclude pattern, it is **skipped** unless it also matches an include pattern.

**Example:**
```toml
network_args = ["exclude_patterns=['.*(img_mod|txt_mod|modulation).*']"]
```

### `include_patterns`

**Option:** `include_patterns=<python-literal-list-of-regex>`

A list of regex patterns that **force inclusion** even if excluded (acts as an override).

**Example - Only train attention Q/K/V:**
```toml
network_args = [
  "exclude_patterns=['.*attn.*']",
  "include_patterns=['.*attn\\.to_q.*', '.*attn\\.to_k.*', '.*attn\\.to_v.*']",
]
```

### Pattern application order

1. Architecture-specific default excludes are **always applied** (additive, cannot be removed via user patterns)
2. User-supplied `exclude_patterns` are appended to the defaults
3. For each module: check exclude list → if matches, mark for exclusion
4. Check `include_patterns` → if matches, override exclusion (use this to selectively re-enable a default-excluded module)

### Default exclusions

Each architecture has safety excludes for layers that cause instability if trained (norm, modulation, embeddings). For example, HunyuanVideo excludes:
```
.*(img_mod|txt_mod|modulation).*
```

These defaults are always active. To train a default-excluded module, add it to `include_patterns` rather than trying to remove the default exclude. (Exception: Qwen-Image's `exclude_mod=False` disables the modulation default entirely — see below.)

> **LoKr-specific args:** For LoKr's `factor` option and other LoKr/LoHa-specific configuration, see `docs/loha_lokr.md`.

---

## Architecture-Specific Options

### Qwen-Image: `exclude_mod`

**Option:** `exclude_mod=True|False` (default: `True`)

**Module:** `networks.lora_qwen_image`

Controls whether modulation layers (`img_mod`, `txt_mod`) are excluded from LoRA.

| Value | Behavior |
|-------|----------|
| `True` (default) | Modulation layers excluded (standard for style/concept LoRAs) |
| `False` | Modulation layers included (recommended for **persona/identity** LoRAs) |

**Example - Include modulation for identity training:**
```toml
[network]
network_module = "networks.lora_qwen_image"
network_args = ["exclude_mod=False"]
```

---

## Utility Options

### `verbose`

**Option:** `verbose=True|False` (default: `False`)

Prints detailed information about the LoRA network during creation, including:
- All modules that receive LoRA adapters
- Modules that were excluded and why
- Parameter counts per module

```toml
network_args = ["verbose=True"]
```

---

## Quick Reference Table

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_rslora` | bool | `False` | RS-LoRA scaling (`alpha/sqrt(r)`) |
| `use_dora` | bool | `False` | DoRA magnitude decomposition (Linear only) |
| `init_lora_weights` | str | `kaiming` | LoRA init scheme: `kaiming`, `orthogonal`, `pissa`, `pissa_niter_<N>`, or `true` alias |
| `rank_pattern` | dict | None | Per-module rank overrides matched with `re.fullmatch()` on dotted module names |
| `alpha_pattern` | dict | None | Per-module alpha overrides using the same matching semantics as `rank_pattern` |
| `loraplus_lr_ratio` | float | None | LoRA-B learning rate multiplier |
| `rank_dropout` | float | None | Dropout on rank dimension (disables DoRA) |
| `module_dropout` | float | None | Dropout on entire module (DoRA OK) |
| `conv_dim` | int | `network_dim` | Separate rank for Conv2d layers |
| `conv_alpha` | float | `network_alpha` | Separate alpha for Conv2d layers |
| `exclude_patterns` | list | `[]` | Regex patterns to exclude modules |
| `include_patterns` | list | `[]` | Regex patterns to force-include modules |
| `exclude_mod` | bool | `True` | Exclude modulation layers (Qwen-Image only) |
| `verbose` | bool | `False` | Print detailed network info |

---

## Known-Good Combinations

### RS-LoRA only
```toml
network_args = ["use_rslora=True"]
```
Best for: Higher-rank training (dim > 32), cross-rank comparisons.

### DoRA only
```toml
network_args = ["use_dora=True"]
```
Best for: Linear-layer focused training without dropout.

### RS-LoRA + DoRA
```toml
network_args = ["use_rslora=True", "use_dora=True"]
```
Best for: Maximum expressiveness with stable scaling.

### LoRA+ + RS-LoRA
```toml
network_args = ["use_rslora=True", "loraplus_lr_ratio=8"]
```
Best for: Faster convergence with stable high-rank training.

### Orthogonal init + RS-LoRA
```toml
network_args = ["init_lora_weights=orthogonal", "use_rslora=True"]
```
Best for: Higher-rank standard LoRA training where you want nonzero initial LoRA-A and LoRA-B while preserving zero initial delta.

### Per-layer rank shaping
```toml
network_args = [
  "rank_pattern={'.*double_blocks\\.[0-7]\\..*': 32, '.*single_blocks\\..*': 16}",
  "alpha_pattern={'.*double_blocks\\.[0-7]\\..*': 32, '.*single_blocks\\..*': 16}",
]
```
Best for: Allocating more capacity to selected block ranges while keeping the global `network_dim` as a fallback.

### Full combo (RS-LoRA + DoRA + LoRA+)
```toml
network_args = ["use_rslora=True", "use_dora=True", "loraplus_lr_ratio=8"]
```
Best for: High-quality persona/identity LoRAs with faster training.

### Persona/Identity (Qwen-Image)
```toml
network_args = ["use_rslora=True", "loraplus_lr_ratio=8", "exclude_mod=False"]
```
Best for: Character/person LoRAs where modulation layers help capture identity.

### Style training with module dropout
```toml
network_args = ["use_dora=True", "module_dropout=0.1"]
```
Best for: Style LoRAs with regularization (module_dropout is DoRA-compatible).

### Selective layer training
```toml
network_args = [
  "exclude_patterns=['.*single_blocks.*']",
  "use_rslora=True",
]
```
Best for: Training only double blocks (or vice versa).

---

## Troubleshooting

| Error/Warning | Cause | Solution |
|---------------|-------|----------|
| `RS-LoRA flag mismatch` (hard error) | Loading RS-LoRA weights without `use_rslora=True` or vice versa | Match `use_rslora` to how weights were trained |
| `DoRA flag mismatch: network expects DoRA but weights lack it` (hard error) | Weights don't contain DoRA magnitudes | Remove `use_dora=True` or use DoRA-trained weights |
| `DoRA flag mismatch: weights contain DoRA but network doesn't expect it` (warning) | Ignoring DoRA magnitudes | Add `use_dora=True` if you want DoRA behavior |
| `DoRA magnitude appears uninitialized` (warning) | Called `get_weight()` before loading weights | Ensure `load_state_dict()` is called first |
| `DoRA disabled for X modules` (info) | Conv layers, dropout, or split_dims | Expected behavior; those modules use standard LoRA |
| `Orthogonal LoRA init requires even rank` | `init_lora_weights=orthogonal` with an odd Linear rank | Use an even `network_dim` or set `init_lora_weights=kaiming` |
| `Conv2d LoRA modules fell back to kaiming init` | Conv2d targets cannot use the Linear-only orthogonal algorithm | Expected behavior; Linear targets still use orthogonal init |
| `rank_pattern regex ... did not compile` | Invalid regex in `rank_pattern` / `alpha_pattern` | Fix the regex string; patterns are compiled with Python `re` |
| `rank_pattern value ... must be a positive int` | Rank override is zero, negative, bool, or non-integer | Use a positive integer rank |
| `resolved to rank ... via rank_pattern ... orthogonal requires even rank` | Pattern selected an odd rank while using orthogonal init | Change that pattern to an even rank or use `init_lora_weights=kaiming` |

---

## Changelog

### 2026-05-06
- Added empirical-bounds subsection ("When NOT to use PiSSA") to the PiSSA section, recording the v10-PiSSA-rank32 vs v9-rank32 negative-result evidence (FLUX.2-Klein-9B persona LoRA, 10000 steps, ProdigyPlusScheduleFree). Narrows the disrecommendation to the specific configuration class evidenced (Prodigy + DoRA-load-bearing pipelines), explicitly does not generalize to all adaptive optimizers. Implementation remains supported; safety contract unchanged.

### 2026-05-05
- Added PiSSA LoRA init (`init_lora_weights=pissa` and `init_lora_weights=pissa_niter_<N>`) with PEFT-parity SVD math, in-place base residualization, ten-failure-mode safety contract (see "PiSSA safety contract"), `ss_base_sha256` hash-coupled validation at hotswap and offline merge time, and fail-fast rejects for incompatible combinations (Conv2d, split_dims, DoRA, WAN dual-expert, `--resume`, fp8 base).

### 2026-05-04
- Added static `rank_pattern` / `alpha_pattern` per-module overrides with fullmatch/first-match semantics and safetensors metadata.

### 2026-05-03
- Added orthogonal LoRA init (`init_lora_weights=orthogonal`) with `true` alias compatibility and metadata persistence.

### 2026-01-16
- Added RS-LoRA (`use_rslora`) with flag mismatch handling and suspicious alpha hints
- Added DoRA (`use_dora`) with Linear-only constraints and dropout interaction
- Added dropout options with DoRA compatibility notes
- Added Conv-specific settings (`conv_dim`, `conv_alpha`)
- Added architecture-specific options (`exclude_mod` for Qwen-Image)
- Added troubleshooting section

---

*Document created: 2026-01-16*
*For blissful-tuner LoRA training*
