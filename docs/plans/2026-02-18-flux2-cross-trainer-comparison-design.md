# FLUX.2 Cross-Trainer Comparison — Design Document

**Date:** 2026-02-18
**Goal:** Compare blissful-tuner's FLUX.2 LoRA training pipeline against OneTrainer and SimpleTuner to identify correctness bugs, regressions, missed improvements, or optimizations.
**Context:** The blissful-tuner FLUX.2 pipeline went through a 37-finding audit (v4.2, 2026-02-17). This cross-trainer comparison validates post-audit correctness and discovers issues the internal audit couldn't catch.

---

## Approach

**Parallel Agent Teams by Trainer** — two independent investigation agents, each comparing one external trainer against blissful-tuner's actual source code.

### Agent A: OneTrainer vs blissful-tuner

**Source locations:**
- OneTrainer: `~/OneTrainer/`
- blissful-tuner: `~/blissful-tuner/`
- Reference doc: `~/blissful-tuner/ONETRAINER_FLUX2_LORA_PIPELINE.md`

### Agent B: SimpleTuner vs blissful-tuner

**Source locations:**
- SimpleTuner: `~/SimpleTuner/`
- blissful-tuner: `~/blissful-tuner/`
- Reference doc: `~/SimpleTuner/FLUX2_PIPELINE_REFERENCE.md`

### Synthesis Phase

Merge findings from both agents. Cross-reference where two trainers agree (high confidence signal) vs disagree. Produce final documentation.

---

## Comparison Domains (Priority Order)

Each agent investigates these correctness-critical areas:

### 1. Flow Matching Formula (CRITICAL)
- Noise addition: `noisy = sigma * noise + (1 - sigma) * clean` — verify exact formulation
- Sigma computation from timesteps
- Whether sigma = t/num_train_timesteps (linear) or other schedule

### 2. Latent Scaling (CRITICAL)
- OneTrainer uses VAE batch norm stats: `(latents - bn.running_mean) / sqrt(bn.running_var)`
- SimpleTuner uses `scaling_factor = 1.0` (no rescaling)
- What does blissful-tuner do? Are these approaches equivalent?

### 3. Loss Target Construction (CRITICAL)
- `target = noise - scaled_latent` vs `target = noise - latent`
- Is the target computed pre- or post-patchification?
- Is loss computed in packed or unpacked space?

### 4. Timestep Normalization (HIGH)
- `timestep / 1000` before transformer input
- Resolution-dependent shift (mu calculation)
- Shift formula: `shifted = shift * t / (1 + (shift - 1) * t)` — verify parameters

### 5. Patchification / Packing Order (HIGH)
- When does 2x2 pixel-shuffle packing happen relative to noise addition?
- OneTrainer: patchify -> scale -> noise -> pack -> transformer
- SimpleTuner: pack latents with position IDs -> noise
- blissful-tuner: verify order

### 6. Guidance Embedding During Training (MEDIUM)
- What guidance value is used during training?
- SimpleTuner: always 1.0 during training
- OneTrainer: `config.transformer.guidance_scale` (configurable)
- Conditional inclusion based on `guidance_embeds` config

### 7. Position ID Construction (MEDIUM)
- 4D coordinate tuples (t, h, w, l)
- Image tokens: spatial coordinates
- Text tokens: sequential coordinates
- Any differences in construction between trainers?

### 8. LoRA Targeting Scope (MEDIUM)
- Which modules are targeted by default
- Exclusion patterns
- Whether text encoder LoRA is supported/applicable

---

## Output Format

Each agent produces a structured findings report:

```markdown
## Finding [ID]: [Short Title]
**Severity:** CRITICAL / WARNING / INFO / OK
**Domain:** [which comparison domain]
**What [other trainer] does:** [description with file:line references]
**What blissful-tuner does:** [description with file:line references]
**Discrepancy:** [what differs and why it matters, or "None — implementations agree"]
**Recommendation:** [action needed, if any]
```

### Severity Definitions
- **CRITICAL**: Mathematical incorrectness that would produce wrong training results
- **WARNING**: Behavioral difference that could affect training quality or user experience
- **INFO**: Style/approach difference with no correctness impact
- **OK**: Implementations agree — document for confidence

---

## Success Criteria

1. All 8 comparison domains investigated for both external trainers
2. Findings cross-referenced: areas where both external trainers agree differently from blissful-tuner are flagged as high-confidence issues
3. Final documentation written to `docs/plans/` with actionable recommendations
4. No false positives from reference doc inaccuracies (agents verify against actual source)
