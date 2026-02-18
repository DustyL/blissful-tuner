# Z-Image Cross-Trainer Comparison — Synthesis Report

**Date:** 2026-02-18
**Sources:**
- Agent A: OneTrainer comparison (13 findings: 2 CRITICAL, 2 WARNING, 2 INFO, 7 OK)
- Agent B: SimpleTuner comparison (14 findings: 0 CRITICAL, 3 WARNING, 6 INFO, 5 OK)

---

## Executive Summary

- **0 correctness bugs** — training produces correct gradients, but via a non-standard convention
- **1 high-confidence maintenance concern** where both trainers flag the same issue (output negation / target inversion)
- **6 confirmed-OK domains** where all three implementations agree on core math
- **2 feature-gap observations** (turbo assistant LoRA, CFG Zero\*)

The Z-Image pipeline is **mathematically correct** for current MSE-only training. The most notable finding is a non-standard sign convention that both external trainers handle differently from blissful-tuner.

---

## High-Confidence Finding (Both Trainers Flag)

### HC-ZI-1: Non-Standard Sign Convention in Training — Both Trainers Negate Output, blissful-tuner Inverts Target

**OneTrainer (OT-ZI-2, CRITICAL):**
```python
predicted_flow = -torch.stack(output_list, dim=0).squeeze(dim=2)  # NEGATED
flow = latent_noise - scaled_latent_image                          # noise - image
```

**SimpleTuner (ST-ZI-2, WARNING):**
```python
noise_pred = -noise_pred  # NEGATED
target = noise - latents   # noise - latents
```

**blissful-tuner:**
```python
model_pred = model_pred.squeeze(2)     # NOT negated
target = latents - noise               # image - noise (INVERTED)
```

**Assessment:** Both external trainers negate the transformer output and use the standard target (`noise - latents`). blissful-tuner does NOT negate the output but compensates by inverting the target (`latents - noise`). Since `MSE(-a, b) == MSE(a, -b)`, the training loss and gradients are **mathematically identical**.

However, this creates:
1. **Asymmetry between training (no negation) and inference (negation)** — a maintenance hazard
2. **Fragility for future loss functions** — Huber, L1, or perceptual losses would break the equivalence
3. **Divergence from the official Z-Image convention** — both external trainers follow the standard
4. **Potential confusion** when comparing training diagnostics across frameworks

The inference path is correct in all three codebases (all negate the output).

**Recommendation:** Two-line fix to align with standard convention:
1. Negate output: `model_pred = -model_pred.squeeze(2)` (in `call_dit()`)
2. Standard target: `target = noise - latents` (instead of `latents - noise`)

Priority: **MEDIUM** — not a bug today, but a maintenance hazard that blocks future loss function additions.

---

## Confirmed OK — All Three Implementations Agree

| Domain | Evidence |
|--------|----------|
| **Timestep inversion** | `(1000 - t) / 1000` in all code paths — OT-ZI-1, ST-ZI-1 |
| **Flow matching noise** | `sigma * noise + (1-sigma) * latents` — OT-ZI-5, ST-ZI-3 |
| **Latent scaling** | `(latents - 0.1159) * 0.3611` / inverse — OT-ZI-6, ST-ZI-5 |
| **Text encoding** | Qwen3, `enable_thinking=True`, `hidden_states[-2]`, 512 tokens — OT-ZI-8, ST-ZI-7/8 |
| **LoRA exclusions** | Both exclude modulation layers; blissful-tuner excludes refiners by default — OT-ZI-10, ST-ZI-9 |
| **Default flow shift** | 3.0 in all three — OT-ZI-6, ST-ZI-14 |

---

## Medium-Confidence Findings

### MC-ZI-1: RoPE axes_lens First Axis Differs (SimpleTuner Only)

**ST-ZI-11:** blissful-tuner uses `axes_lens=[1536, 512, 512]` while SimpleTuner's Python default is `[1024, 512, 512]`. The first axis defines the maximum position for the sequence/frame RoPE dimension. Since actual positions stay well below 514, this is harmless. blissful-tuner's `1536` likely comes from the original Z-Image source code.

**Recommendation:** Verify against the official `config.json`. Priority: **LOW**.

### MC-ZI-2: Turbo Assistant LoRA Not Supported (SimpleTuner Only)

**ST-ZI-10:** SimpleTuner loads a pre-trained assistant LoRA adapter (`ostris/zimage_turbo_training_adapter`) for turbo model training, considering it mandatory. blissful-tuner has no equivalent.

**Recommendation:** Document that Z-Image-Turbo LoRA training may benefit from the assistant adapter. Priority: **LOW**.

### MC-ZI-3: CFG Zero\* Not Wired for Z-Image Inference (SimpleTuner Only)

**ST-ZI-13:** SimpleTuner uses CFG Zero\* by default for Z-Image inference. blissful-tuner has CFG Zero\* implemented in `guidance.py` for other architectures but hasn't wired it into Z-Image generation.

**Recommendation:** Wire existing `guidance.py` CFG Zero\* into Z-Image generation script. Priority: **LOW**.

---

## Cross-Architecture Pattern Update

The Z-Image comparison reveals a pattern unique among the three architectures studied:

| Pattern | FLUX.2 | Qwen Image | Z-Image |
|---------|--------|-----------|---------|
| Training guidance | Hardcoded 1.0 | N/A | N/A (no guidance embed) |
| Timestep +1 offset | Yes | Yes | Yes (shared base class) |
| Output negation | N/A | N/A | **Non-standard** (target inverted) |
| LoRA scope wider | Yes | Yes | Yes (consistent design) |
| MSE-only loss | Yes | Yes | Yes (but especially risky here due to sign convention) |

The output negation finding is Z-Image-specific because Z-Image is the only architecture where the transformer output needs negation. FLUX.2 and Qwen Image don't have this quirk.

---

## Recommended Actions (Ordered by Priority)

| # | Priority | Finding | Action | Effort |
|---|----------|---------|--------|--------|
| 1 | **MEDIUM** | HC-ZI-1 | Negate output + standard target in `call_dit()` (two-line fix) | Tiny |
| 2 | LOW | MC-ZI-1 | Verify RoPE axes_lens against official config.json | Tiny |
| 3 | LOW | MC-ZI-2 | Document turbo assistant LoRA recommendation | Tiny |
| 4 | LOW | MC-ZI-3 | Wire CFG Zero\* into Z-Image inference | Small |

---

## Methodology Notes

- Both agents read actual source code across all three codebases (65 tool calls total)
- The non-standard sign convention was flagged independently by both agents, giving high confidence
- All inference paths verified correct across all three codebases
- Cross-architecture patterns tracked across FLUX.2, Qwen Image, and Z-Image comparisons
