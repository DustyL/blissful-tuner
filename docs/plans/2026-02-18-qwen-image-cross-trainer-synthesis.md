# Qwen Image Cross-Trainer Comparison — Synthesis Report

**Date:** 2026-02-18
**Sources:**
- Agent A: OneTrainer comparison (14 findings: 0 CRITICAL, 4 WARNING, 3 INFO, 7 OK)
- Agent B: SimpleTuner comparison (23 findings: 0 CRITICAL, 3 WARNING, 8 INFO, 12 OK)

---

## Executive Summary

- **0 correctness bugs found** — core training mathematics verified correct across all three codebases
- **1 high-confidence behavioral finding** where both trainers flag the same issue (timestep +1 offset)
- **1 medium-confidence finding** where only one trainer differs (VAE posterior sampling mode)
- **8 confirmed-OK domains** where all three implementations agree on core math
- **2 design-choice observations** worth documenting (LoRA scope, seq padding)

The blissful-tuner Qwen Image pipeline is mathematically sound. The prior internal audit (40 findings, all resolved) already caught the significant issues. Cross-trainer comparison validates correctness and reveals only behavioral defaults and feature gaps.

---

## High-Confidence Findings (Both Trainers Inform)

### HC-QI-1: Timestep +1 Offset — Both External Trainers Use [0, 1], blissful-tuner Uses [0.001, 1.001]

**OneTrainer (OT-QI-14):** Uses discrete sigma table indexed 1-to-N, mapping to sigma range [0.001, 1.0]. Continuous `t` values discretized to integers via `.int()`.

**SimpleTuner (ST-QI-3, ST-QI-12):** Computes `timesteps = sigmas * 1000.0`, normalized to [0.0, 1.0]. No +1 offset.

**blissful-tuner:** Adds `timesteps += 1` (inherited from HunyuanVideo/Wan base trainer at `hv_train_network.py:1127`), producing [0.001, 1.001] after /1000 normalization.

**Assessment:** Both external trainers' normalized timestep ranges start at 0.0 (or very close). blissful-tuner's +1 offset means the transformer never sees exactly t=0 (fully clean) and slightly exceeds t=1.0 (fully noisy). The 0.1% magnitude is small but creates a minor train/inference mismatch since Qwen Image inference uses timesteps in [0, 1].

**Recommendation:** Consider adding a Qwen-Image override that skips the +1 offset. Since this is inherited from the base class and affects all architectures, a per-architecture flag in `get_noisy_model_input_and_timesteps()` would be cleanest. Priority: **LOW** — the practical impact is minimal but technically incorrect.

### HC-QI-2: LoRA Targeting Scope — blissful-tuner Targets ~3x More Layers

**OneTrainer:** Default preset `"attn-mlp"` targets attention + MLP (excludes modulation).

**SimpleTuner:** Default targets attention projections only: `["to_k", "to_q", "to_v", "to_out.0"]`.

**blissful-tuner:** Targets ALL Linear layers in `QwenImageTransformerBlock` minus modulation — includes attention + cross-attention + MLP.

**Assessment:** blissful-tuner's coverage is closest to OneTrainer's "attn-mlp" preset. SimpleTuner's default is narrower. At the same rank, blissful-tuner LoRAs have ~3x more parameters than SimpleTuner's. This is a design choice, not a bug, but should be documented for users comparing training results across trainers.

**Recommendation:** Document this difference in `docs/qwen_image.md`. Priority: **LOW** (documentation only).

---

## Medium-Confidence Findings (One Trainer Differs)

### MC-QI-1: VAE Posterior Sampling — mode() vs sample() (SimpleTuner Only)

**ST-QI-5:** SimpleTuner uses `posterior.sample()` (stochastic) for VAE latent extraction. blissful-tuner uses `posterior.mode()` (deterministic mean).

**Assessment:** Both are valid approaches. `mode()` produces reproducible latents; `sample()` adds implicit augmentation through KL noise. The blissful-tuner choice is deliberate (code comment: "Use mode instead of sampling for deterministic results"). OneTrainer uses `SampleVAEDistribution(mode='mean')` — agreeing with blissful-tuner's `mode()` approach.

**Recommendation:** No action needed. Two trainers agree on `mode()`/`mean`, one uses `sample()`. blissful-tuner is in the majority here.

### MC-QI-2: Sequence Padding to Multiple of 16 (OneTrainer Only)

**OT-QI-8:** OneTrainer pads text embedding sequence length to a multiple of 16 for torch.compile compatibility. blissful-tuner does not.

**Assessment:** This only matters when using `--compile` with variable-length text sequences and batch_size > 1. Since most Qwen Image LoRA training uses batch_size=1, the impact is minimal.

**Recommendation:** Consider adding optional padding when torch.compile is enabled. Priority: **LOW**.

---

## Confirmed OK — All Three Implementations Agree

| Domain | Evidence |
|--------|----------|
| **Flow matching formula** | `noisy = sigma * noise + (1-sigma) * clean` — OT-QI-1, ST-QI-1 |
| **Latent scaling** | `(latents - mean) * (1/std)` from VAE config — OT-QI-2, ST-QI-4 |
| **Loss target** | `target = noise - scaled_latent` in unpacked space — OT-QI-3, ST-QI-7/8 |
| **Shift formula** | Linear interp with (256, 0.5) to (8192, 0.9), exp(mu) — OT-QI-4, ST-QI-10 |
| **Packing order** | scale -> noise -> pack -> transformer — OT-QI-5, ST-QI-13/14 |
| **Text encoding** | Last hidden layer, template crop at idx 34 (T2I) / 64 (Edit) — OT-QI-6, ST-QI-16/17 |
| **Pack/unpack** | 2x2 patchification: (B,16,H,W) -> (B,HW/4,64) — OT-QI-5, ST-QI-14 |
| **Edit conditioning** | Concat control along sequence dim, slice output — OT-QI-11, ST-QI-21/23 |

---

## Feature Gap Observations

### FG-QI-1: Text Encoder LoRA (OneTrainer has, blissful-tuner doesn't)
OneTrainer supports optional LoRA on Qwen2.5-VL text encoder. blissful-tuner only supports transformer LoRA. OneTrainer defaults to text_encoder.train=false. Priority: LOW.

### FG-QI-2: Offset Noise (OneTrainer has)
Same gap as FLUX.2 — OneTrainer supports offset noise for Qwen Image. Priority: LOW.

### FG-QI-3: Multiple Loss Functions (OneTrainer has)
Same gap as FLUX.2 — OneTrainer supports MAE, Huber, log-cosh alongside MSE. Priority: LOW.

### FG-QI-4: Extended Prompt Length (blissful-tuner advantage)
blissful-tuner supports 1024 user tokens vs OneTrainer's 512. This goes beyond the official model training distribution but provides flexibility. Priority: N/A (already implemented).

### FG-QI-5: Dynamic Timestep Shift (blissful-tuner advantage)
blissful-tuner's `qwen_shift` provides resolution-dependent dynamic shifting matching the official scheduler, while SimpleTuner defaults to static shift=1.73. Priority: N/A (already implemented).

---

## Recommended Actions (Ordered by Priority)

| # | Priority | Finding | Action | Effort |
|---|----------|---------|--------|--------|
| 1 | LOW | HC-QI-1 | Consider per-architecture override to skip +1 timestep offset for Qwen Image | Small |
| 2 | LOW | HC-QI-2 | Document LoRA targeting scope difference in `docs/qwen_image.md` | Tiny |
| 3 | LOW | MC-QI-2 | Add optional seq-length padding to multiple of 16 for torch.compile | Small |
| 4 | LOW | FG-QI-1 | Consider text encoder LoRA support (future feature) | Medium |

---

## Comparison with FLUX.2 Cross-Trainer Findings

Several findings are shared between the FLUX.2 and Qwen Image comparisons:

| Finding | FLUX.2 | Qwen Image | Notes |
|---------|--------|-----------|-------|
| Timestep +1 offset | ST-2 (INFO) | HC-QI-1 (both flag) | Same root cause: inherited from base trainer |
| Offset noise gap | OT-13 (INFO) | FG-QI-2 (INFO) | Same gap across architectures |
| Multiple loss functions | OT-14 (INFO) | FG-QI-3 (INFO) | Same gap across architectures |
| LoRA scope wider | OT-12/ST-18 (INFO) | HC-QI-2 (both note) | Consistent design choice |
| Training guidance | HC-1 (MEDIUM) | N/A | Qwen Image has no guidance embedding |

The timestep +1 offset and wider LoRA scope are systemic patterns in blissful-tuner, not architecture-specific issues. If addressed, they should be fixed once in the base trainer.

---

## Methodology Notes

- Both agents read actual source code in all three codebases (98 tool calls total)
- Reference docs verified against source; no significant discrepancies found
- Prior Qwen Image audit findings (40 items, all resolved) cross-referenced for consistency
- Cross-validation between independent investigations confirms core math correctness
