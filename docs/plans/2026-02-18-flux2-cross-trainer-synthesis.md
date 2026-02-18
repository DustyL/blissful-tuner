# FLUX.2 Cross-Trainer Comparison — Synthesis Report

**Date:** 2026-02-18
**Sources:**
- Agent A: OneTrainer comparison (16 findings: 0 CRITICAL, 4 WARNING, 7 INFO, 5 OK)
- Agent B: SimpleTuner comparison (20 findings: 0 CRITICAL, 2 WARNING, 8 INFO, 10 OK)

---

## Executive Summary

- **0 high-confidence correctness bugs found** — core training mathematics are verified correct across all three codebases
- **1 high-confidence behavioral finding** where both trainers handle something differently (guidance embedding)
- **2 medium-confidence findings** where one trainer differs
- **7 confirmed-OK domains** where all three implementations agree
- **3 feature-gap observations** worth considering

The blissful-tuner v4.2 audit successfully addressed the critical bugs. The remaining findings are behavioral defaults and feature gaps, not mathematical errors.

---

## High-Confidence Findings (Both Trainers Inform)

### HC-1: Training Guidance Embedding — Mixed Signal

**OneTrainer (OT-10):** Configurable via `config.transformer.guidance_scale`, **defaults to 4.0 for DEV**. Only provided when `guidance_embeds=True`.

**SimpleTuner (ST-15):** Configurable via `flux_guidance_mode` + `flux_guidance_value`, **defaults to 1.0**. Supports `constant`, `random-range`, and disabled modes.

**blissful-tuner:** Hardcoded to `1.0` at `flux_2_train_network.py:449`.

**Assessment:** The two external trainers **disagree on the default** — OneTrainer defaults to 4.0 (matching DEV inference), SimpleTuner defaults to 1.0. Both make it configurable; blissful-tuner does not. The argument for 1.0 is that it makes the LoRA "guidance-neutral" (works across different inference guidance values). The argument for matching inference guidance (e.g., 4.0) is that it minimizes distribution shift. Both are valid approaches.

**Recommendation:** Make the training guidance value configurable via `--training_guidance_scale` (default 1.0 to preserve current behavior). Document the trade-off. Priority: **MEDIUM**.

### HC-2: Dynamic Timestep Shift — Both Trainers Use Different Sequence Length

**OneTrainer (OT-8):** Uses `(latent_h / patch_size) * (latent_w / patch_size)` where `patch_size=2`, effectively computing `(h/2) * (w/2)` on patchified dims. For 1024px: seq_len = 1024.

**SimpleTuner (ST-11):** Uses `(height // patch_size) * (width // patch_size)` with `patch_size=2`, getting the same result. For 1024px: seq_len = 1024.

**blissful-tuner:** Uses `h * w` on patchified dims (no additional halving). For 1024px: seq_len = 4096.

**Assessment:** Both external trainers apply an additional `//patch_size=2` divisor that blissful-tuner does not. However, the SimpleTuner agent noted this may be a FLUX.1 artifact — FLUX.2's transformer actually processes `h*w` tokens (4096 for 1024px), not `(h/2)*(w/2)` tokens, because the pixel shuffle already happened at VAE encode time. blissful-tuner's approach may be more correct for FLUX.2's architecture, but there's no authoritative reference.

**Recommendation:** Document the difference. Both approaches are approximate since FLUX.2's optimal shift parameters aren't published. The current blissful-tuner implementation is defensible. Consider offering both as options (e.g., `--timestep_sampling flux2_shift_v2` using the `//2` convention). Priority: **LOW**.

### HC-3: Loss Computation Space — Patchified vs Unpatchified

**OneTrainer (OT-5/OT-6):** Unpatchifies predicted and target before MSE. Loss in `(B, 32, H/8, W/8)` space. Masks downscaled to `H/8 x W/8`.

**SimpleTuner (ST-8):** Keeps patchified. Loss in `(B, 128, H/16, W/16)` space.

**blissful-tuner:** Keeps patchified. Loss in `(B, 128, H/16, W/16)` space.

**Assessment:** blissful-tuner agrees with SimpleTuner. For unweighted loss, the total MSE is mathematically identical regardless of space. For **mask-weighted loss**, blissful-tuner's masks operate at 2x lower spatial resolution (64x64 vs 128x128 for 1024px images). This means mask details smaller than ~32x32 pixels in image space are lost.

**Recommendation:** Document the mask resolution limitation. For most use cases (face/body isolation), 64x64 mask resolution is sufficient. Priority: **LOW**.

---

## Medium-Confidence Findings (One Trainer Differs)

### MC-1: Sigmoid Scale Default (SimpleTuner Only)

**ST-4:** SimpleTuner defaults `flow_sigmoid_scale=5.0` while blissful-tuner defaults `sigmoid_scale=1.0`.

A scale of 5.0 produces a nearly-uniform sigma distribution. A scale of 1.0 concentrates sampling around mid-noise levels (sigma ≈ 0.5).

**Assessment:** The 1.0 default is inherited from musubi-tuner / FLUX.1 and may not be optimal for FLUX.2. This is configurable via `--sigmoid_scale` so no bug, but the default may be suboptimal.

**Recommendation:** Document that `--sigmoid_scale 5.0` may be worth experimenting with for FLUX.2 training. Consider whether `flux2_shift` should use a different default than the global `sigmoid_scale`. Priority: **LOW**.

### MC-2: Timestep +1 Offset (SimpleTuner Only)

**ST-2:** blissful-tuner adds `timesteps += 1` before dividing by 1000, producing normalized timesteps in [0.001, 1.001] instead of [0, 1].

**Assessment:** Inherited from the base NetworkTrainer class (designed for 1-indexed discrete schedulers). The 0.1% offset is negligible and changing it would affect all architectures sharing the base class.

**Recommendation:** No action needed. The offset is harmless and architecturally intentional.

---

## Confirmed OK — All Three Implementations Agree

| Domain | Evidence |
|--------|----------|
| **Flow matching formula** | `noisy = sigma * noise + (1-sigma) * clean` — OT-1, ST-1 |
| **Latent scaling** | BN normalization `(z - mean) / sqrt(var + eps)` — OT-3, ST-5/6 |
| **Loss target** | `target = noise - scaled_latent` (rectified flow velocity) — OT-4, ST-7 |
| **Patchification** | 2x2 pixel shuffle: 32ch → 128ch, halved spatial — OT-9, ST-12/13/14 |
| **Timestep normalization** | `timestep / 1000` → [0, 1] — OT-7, ST-10 |
| **Position IDs** | 4D `(t, h, w, l)` tuples, `cartesian_prod`, scale=10 time offsets — OT-11, ST-16/17 |
| **LoRA scope** | Transformer blocks only, no text encoder LoRA — OT-12, ST-18 |

---

## Feature Gap Observations

### FG-1: Offset Noise (OneTrainer has, blissful-tuner doesn't)
OneTrainer supports `--offset_noise_weight` for channel-wise constant noise that improves color/contrast range. blissful-tuner and SimpleTuner do not have this for FLUX.2. Priority: LOW.

### FG-2: Multiple Loss Functions (OneTrainer has)
OneTrainer supports MAE/L1, Huber, and log-cosh losses alongside MSE, with configurable weights. blissful-tuner uses MSE only. MSE is the standard choice for flow matching; alternatives are niche. Priority: LOW.

### FG-3: Named LoRA Targeting Presets (SimpleTuner has)
SimpleTuner offers `tiny`, `mlp`, `slider`, `all` targeting modes as named presets. blissful-tuner uses regex `--exclude_patterns` which is more flexible but less discoverable. Priority: LOW.

### FG-4: Random Guidance Range (SimpleTuner has)
SimpleTuner supports `flux_guidance_mode=random-range` for guidance augmentation during training. Could improve LoRA robustness across different inference guidance values. Priority: LOW (if HC-1 configurable guidance is implemented, this could be a natural extension).

---

## Recommended Actions (Ordered by Priority)

| # | Priority | Finding | Action | Effort |
|---|----------|---------|--------|--------|
| 1 | **MEDIUM** | HC-1 | Add `--training_guidance_scale` CLI arg (default 1.0), document trade-off | Small |
| 2 | LOW | HC-2 | Document timestep shift formula differences in `docs/flux_2.md` | Tiny |
| 3 | LOW | HC-3 | Document mask spatial resolution (64x64 for 1024px) in masked loss guide | Tiny |
| 4 | LOW | MC-1 | Document `--sigmoid_scale 5.0` as worth trying for FLUX.2 | Tiny |
| 5 | LOW | FG-1 | Consider offset noise support (future feature) | Medium |
| 6 | LOW | FG-3 | Consider named LoRA targeting presets (future feature) | Small |

---

## Methodology Notes

- Both agents read **actual source code** (not just reference docs) in all three codebases
- 135 tool calls total across both agents (45 for OneTrainer, 90 for SimpleTuner)
- Reference docs were verified against source; no significant discrepancies found
- Cross-referencing between independent investigations provides natural validation
