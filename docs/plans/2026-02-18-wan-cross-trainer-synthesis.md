# WAN Cross-Trainer Comparison — Synthesis Report

**Date:** 2026-02-18
**Sources:**
- Agent A: ai-toolkit comparison (22 findings: 0 CRITICAL, 4 WARNING, 8 INFO, 10 OK)
- Agent B: SimpleTuner comparison (20 findings: 0 CRITICAL, 4 WARNING, 8 INFO, 8 OK)

---

## Executive Summary

- **0 correctness bugs found** — core training mathematics verified correct across all three codebases
- **2 high-confidence architectural findings** where both trainers reveal design trade-offs in blissful-tuner
- **6 confirmed-OK domains** where all three implementations agree on core math
- **3 design-choice observations** worth documenting

The WAN pipeline is mathematically sound. The most significant findings are architectural: the dual-stage training approach (rejection sampling vs round-robin) and the unified LoRA save format.

---

## High-Confidence Findings (Both Trainers Inform)

### HC-WAN-1: Dual-Stage Training Imbalance — Rejection Sampling Undertrain High-Noise Expert

**ai-toolkit (AT-WAN-06, WARNING):** Uses deterministic round-robin via `switch_boundary_every`. Every N steps, alternates between high-noise [0.875, 1.0] and low-noise [0, 0.875] ranges. Both experts receive equal training time.

**SimpleTuner (ST-WAN-08, INFO):** Trains one stage at a time via `model_flavour` selection. User decides how many steps to allocate per stage.

**blissful-tuner:** Uses rejection sampling. The first random timestep determines which expert trains for the entire batch. With boundary=0.875 and shift sampling, high-noise region has ~12.5% probability mass → high-noise expert gets ~7x fewer training updates.

**Assessment:** Both external trainers give users explicit control over expert training balance. blissful-tuner's natural distribution weighting means the high-noise expert is significantly undertrained. This may be intentional (train proportional to timestep frequency) or an oversight.

**Recommendation:** Consider adding an optional `--equal_expert_training` or `--switch_boundary_every N` flag for deterministic alternation. Document the ~7:1 training ratio imbalance in `docs/wan.md`. Priority: **MEDIUM**.

### HC-WAN-2: No Split LoRA Save for WAN 2.2 Dual-Expert

**ai-toolkit (AT-WAN-08, WARNING):** Saves separate `_high_noise.safetensors` and `_low_noise.safetensors` files with key renaming. Users can apply different LoRAs per expert at inference.

**SimpleTuner (ST-WAN-07, OK):** Trains one stage at a time (separate runs), so each produces its own LoRA file naturally.

**blissful-tuner:** Uses weight-swap architecture — a single model trains both experts. LoRA weights are unified. Cannot be split per expert.

**Assessment:** This is a fundamental architectural consequence of the weight-swap design. Since both experts share the same module names, LoRA updates from both noise ranges are accumulated into the same parameters. The LoRA "learns" a compromise that works across both ranges.

**Recommendation:** Document this as a known characteristic. Users who need per-expert LoRAs can train with `--dit_high_noise` only (freezing one expert). Priority: **LOW** (documentation).

---

## Confirmed OK — All Three Implementations Agree

| Domain | Evidence |
|--------|----------|
| **Flow matching formula** | `x_t = (1-t)*x_0 + t*noise` — AT-WAN-01, ST-WAN-01 |
| **Loss target** | `v = noise - latents` — AT-WAN-02, ST-WAN-02 |
| **VAE normalization** | Per-channel `(latent - mean) * (1/std)` with identical 16-channel stats — AT-WAN-09, ST-WAN-04 |
| **Shift formula** | `sigma_shifted = shift*sigma / (1 + (shift-1)*sigma)` — AT-WAN-12, ST-WAN-09 |
| **5D tensor handling** | Correct broadcasting for `(B, C, T, H, W)` video — AT-WAN-03, ST-WAN-13 |
| **Text encoding** | UMT5, 512 tokens, variable-length embeddings — AT-WAN-17/18/19, ST-WAN-15/16 |

---

## Medium-Confidence Findings

### MC-WAN-1: Training Shift Value Defaults Diverge

**ai-toolkit (AT-WAN-11):** Bakes `shift=5.0` into WAN 2.2 scheduler config.

**SimpleTuner (ST-WAN-10):** Uses sigmoid sampling + `flow_schedule_shift=3` by default.

**blissful-tuner:** Defaults to `shift=1.0` but recommends `--discrete_flow_shift 12.0` for T2V-A14B and `5.0` for I2V-A14B via warnings.

**Assessment:** Three different defaults (1.0, 3.0, 5.0). blissful-tuner's recommended 12.0 for T2V is the most aggressive. All are configurable. The right value is empirical.

**Recommendation:** Document shift recommendations per task. Priority: **LOW**.

### MC-WAN-2: Frame Count Constraint (SimpleTuner Only)

**ST-WAN-11 (WARNING):** SimpleTuner enforces `frames % 8 == 1` (stride 8). blissful-tuner enforces `(frames - 1) % 4 == 0` (stride 4, T=4k+1). The WAN VAE temporal compression is 4x, so blissful-tuner's constraint is correct and more permissive.

**Recommendation:** blissful-tuner is correct here. No action needed. Priority: N/A.

### MC-WAN-3: LoRA Targeting — Agents Disagree on Who's Wider

**ai-toolkit (AT-WAN-20):** Says ai-toolkit targets all Linear layers (broader), blissful-tuner targets `WanAttentionBlock` (narrower).

**SimpleTuner (ST-WAN-18):** Says SimpleTuner targets only `to_k/q/v/out.0` (4 projections, narrower), blissful-tuner targets all Linear in `WanAttentionBlock` (broader, ~10-11 per block including FFN).

**Assessment:** blissful-tuner is in the middle — broader than SimpleTuner (includes FFN), narrower than ai-toolkit's full-model targeting. This is the established convention inherited from musubi-tuner.

**Recommendation:** Document the scope. Priority: **LOW**.

---

## Feature Gap Observations

### FG-WAN-1: Noise Augmentation Stack (ai-toolkit has, blissful-tuner doesn't)
ai-toolkit has a 5-stage noise augmentation pipeline: dynamic noise offset, noise multiplier, signal correction noise, random noise shift, random noise multiplier. blissful-tuner has none. Signal correction noise (`noise += latents * randn(B,C,1,1) * scale`) is novel and could be beneficial. Priority: LOW.

### FG-WAN-2: Loss Weighting (ai-toolkit has)
ai-toolkit supports min-SNR weighting, learnable SNR, and bell-shaped timestep weighting. blissful-tuner has sigma_sqrt, cosmap, and structure_bell (different but comparable). Priority: LOW.

### FG-WAN-3: Prompt Cleaning (SimpleTuner has)
SimpleTuner applies ftfy-based Unicode normalization and HTML unescaping to prompts. blissful-tuner only does whitespace normalization. Priority: LOW.

### FG-WAN-4: WAN 2.2 T2V Multi-Stage (blissful-tuner advantage)
blissful-tuner supports both T2V (0.875) and I2V (0.900) multi-stage with task-specific configs. SimpleTuner only supports I2V multi-stage. Priority: N/A (blissful-tuner already has this).

---

## Recommended Actions (Ordered by Priority)

| # | Priority | Finding | Action | Effort |
|---|----------|---------|--------|--------|
| 1 | **MEDIUM** | HC-WAN-1 | Add optional `--switch_boundary_every` for deterministic expert alternation | Medium |
| 2 | LOW | HC-WAN-2 | Document unified LoRA save limitation for WAN 2.2 | Tiny |
| 3 | LOW | MC-WAN-1 | Document shift recommendations per task | Tiny |
| 4 | LOW | MC-WAN-3 | Document LoRA targeting scope | Tiny |
| 5 | LOW | FG-WAN-1 | Consider signal correction noise feature | Medium |

---

## Cross-Architecture Pattern Update (All 4 Architectures)

| Pattern | FLUX.2 | Qwen Image | Z-Image | WAN |
|---------|--------|-----------|---------|-----|
| Timestep +1 offset | Yes | Both flag | Yes | Both note |
| Wider LoRA scope | Yes | Yes | Yes | Middle (narrower than AT, wider than ST) |
| MSE-only loss | Yes | Yes | Yes | Yes |
| No offset/augmentation noise | Yes | Yes | Yes | Yes (AT has 5-stage stack) |
| Training guidance | Hardcoded 1.0 | N/A | N/A | N/A |
| Sign convention | N/A | N/A | **Non-standard** | Standard (OK) |
| Dual-stage imbalance | N/A | N/A | N/A | **~7:1 ratio** |
| Unified LoRA | N/A | N/A | N/A | **No split save** |

---

## Methodology Notes

- Both agents read actual source code across all three codebases (139 tool calls total)
- WAN is the most complex architecture compared due to dual-stage MOE, 5D video tensors, temporal VAE compression, and I2V conditioning
- Cross-validation between ai-toolkit and SimpleTuner provides confidence on correctness findings
- The dual-stage architectural findings are WAN-specific and don't apply to other architectures
