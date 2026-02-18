# FLUX.2 Cross-Trainer Comparison — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compare blissful-tuner's FLUX.2 LoRA training pipeline against OneTrainer and SimpleTuner to identify correctness bugs, regressions, or missed improvements.

**Architecture:** Two parallel investigation agents (one per external trainer) read actual source code across all three codebases, produce structured findings reports. A synthesis phase cross-references findings where both external trainers agree differently from blissful-tuner (high-confidence signals).

**Tech Stack:** Claude Code subagents (general-purpose type), file reading, structured markdown output

---

## Task 1: Launch Agent A — OneTrainer vs blissful-tuner (parallel with Task 2)

**Purpose:** Deep-read OneTrainer's FLUX.2 implementation and compare against blissful-tuner across 8 correctness domains.

**Step 1: Dispatch Agent A as a background Task agent**

Launch a `general-purpose` subagent with the following prompt (run in background, since it's long-running):

```
You are investigating OneTrainer's FLUX.2 LoRA training pipeline and comparing it
against blissful-tuner's implementation. Your goal is to find correctness bugs,
regressions, or missed improvements in blissful-tuner by reading actual source code
in both codebases.

## Reference Documents
- Read first: /Users/dustin/blissful-tuner/ONETRAINER_FLUX2_LORA_PIPELINE.md
- Audit context: /Users/dustin/blissful-tuner/docs/plans/2026-02-17-flux2-pipeline-audit-fixes.md

## Key Files to Read

### OneTrainer (~/OneTrainer/)
- modules/modelSetup/BaseFlux2Setup.py — forward pass (predict method), latent processing
- modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py — flow matching formula, sigma, timesteps
- modules/modelSetup/mixin/ModelSetupNoiseMixin.py — noise generation, offset noise
- modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py — loss computation, target construction
- modules/modelSetup/Flux2LoRASetup.py — LoRA configuration, target modules
- modules/model/Flux2Model.py — model structure, patchify/unpatchify/scale/pack operations
- modules/dataLoader/Flux2BaseDataLoader.py — data loading, VAE encoding pipeline
- modules/modelSampler/Flux2Sampler.py — inference loop (for reference)

### blissful-tuner (~/blissful-tuner/)
- src/musubi_tuner/flux_2_train_network.py — Flux2NetworkTrainer, call_dit()
- src/musubi_tuner/hv_train_network.py — NetworkTrainer base class, training loop (lines ~2400-2520 for loss), call_dit (line ~1749), compute_loss_weighting_for_sd3 (line ~350)
- src/musubi_tuner/flux_2/flux2_models.py — Flux2 model class, patchify/unpatchify/scale/pack
- src/musubi_tuner/flux_2/flux2_utils.py — text encoding (Mistral3Embedder, Qwen3Embedder), timestep shift, guidance
- src/musubi_tuner/flux_2/__init__.py — imports and utilities
- src/musubi_tuner/flux_2_cache_latents.py — latent caching pipeline
- src/musubi_tuner/networks/lora_flux_2.py — LoRA target modules, exclusion patterns
- src/musubi_tuner/networks/network_arch.py — architecture registry
- src/musubi_tuner/modules/mask_loss.py — mask-weighted loss

## Comparison Domains (investigate in this order)

### Domain 1: Flow Matching Formula (CRITICAL)
Read the EXACT noise addition formula in both codebases. Verify:
- Is it `noisy = sigma * noise + (1 - sigma) * clean`?
- How is sigma computed from the timestep? Is it `t / num_train_timesteps`?
- Are there any differences in the formula?

### Domain 2: Latent Scaling (CRITICAL)
OneTrainer uses VAE batch norm stats: `(latents - bn.running_mean) / sqrt(bn.running_var)`.
- Does blissful-tuner do the same? Read the actual code.
- Where does scaling happen relative to patchification and noise addition?
- Is the scaled version used in the loss target?

### Domain 3: Loss Target Construction (CRITICAL)
- What is the exact loss target? `noise - scaled_latent` or `noise - latent`?
- Is the target computed pre- or post-patchification?
- Is loss MSE computed in packed (sequence) space or unpacked (spatial) space?
- Does OneTrainer unpack before loss? Does blissful-tuner?

### Domain 4: Timestep Normalization (HIGH)
- Verify timestep division by 1000 before passing to transformer
- Compare resolution-dependent shift formulas (mu calculation, base_shift, max_shift)
- Check timestep sampling distributions available

### Domain 5: Patchification / Packing Order (HIGH)
- Trace the EXACT order: raw_latent -> patchify -> scale -> noise -> pack -> transformer
- Verify this order is consistent between both codebases
- Pay special attention to whether noise is added BEFORE or AFTER patchification

### Domain 6: Guidance Embedding During Training (MEDIUM)
- What guidance value does OneTrainer use during training?
- What does blissful-tuner use?
- Is it conditional on guidance_embeds config?

### Domain 7: Position ID Construction (MEDIUM)
- Compare coordinate tensor construction (t, h, w, l format)
- Image IDs vs text IDs
- Any differences in axis ordering or indexing?

### Domain 8: LoRA Targeting Scope (MEDIUM)
- Compare default target modules
- Compare exclusion patterns
- Note any modules one trainer targets that the other doesn't

## Output Format

Write your findings to: /Users/dustin/blissful-tuner/docs/plans/2026-02-18-onetrainer-comparison-findings.md

Use this format for EACH finding:

## Finding [OT-N]: [Short Title]
**Severity:** CRITICAL / WARNING / INFO / OK
**Domain:** [1-8]
**What OneTrainer does:** [description with exact file:line references]
**What blissful-tuner does:** [description with exact file:line references]
**Discrepancy:** [what differs and why it matters, or "None — implementations agree"]
**Recommendation:** [action needed, if any]

IMPORTANT: For every domain, produce at least one finding (even if "OK — implementations agree").
Include a summary table at the top with all findings by severity.
```

**Step 2: Verify Agent A launched successfully**

Check the background task output file exists and agent has started reading files.

---

## Task 2: Launch Agent B — SimpleTuner vs blissful-tuner (parallel with Task 1)

**Purpose:** Deep-read SimpleTuner's FLUX.2 implementation and compare against blissful-tuner across 8 correctness domains.

**Step 1: Dispatch Agent B as a background Task agent**

Launch a `general-purpose` subagent with the following prompt (run in background):

```
You are investigating SimpleTuner's FLUX.2 LoRA training pipeline and comparing it
against blissful-tuner's implementation. Your goal is to find correctness bugs,
regressions, or missed improvements in blissful-tuner by reading actual source code
in both codebases.

## Reference Documents
- Read first: /Users/dustin/SimpleTuner/FLUX2_PIPELINE_REFERENCE.md
- Audit context: /Users/dustin/blissful-tuner/docs/plans/2026-02-17-flux2-pipeline-audit-fixes.md

## Key Files to Read

### SimpleTuner (~/SimpleTuner/)
- simpletuner/helpers/models/flux2/model.py — Flux2 class, model_predict() (lines ~788-911), model constants
- simpletuner/helpers/models/flux2/__init__.py — pack_latents, unpack_latents (lines 22-91), conditioning packing
- simpletuner/helpers/models/flux2/transformer.py — transformer architecture details
- simpletuner/helpers/models/flux2/pipeline.py — Flux2Pipeline, inference
- simpletuner/helpers/models/common.py — prepare_batch (lines ~4275-4434), loss (lines ~4593-4771), sigma sampling (lines ~3661-3711)
- simpletuner/helpers/training/custom_schedule.py — flow schedule shift (lines ~443-478)
- simpletuner/helpers/training/collate.py — batch collation (lines ~554-1053)
- simpletuner/helpers/training/trainer.py — training loop (lines ~4909-5032)

### blissful-tuner (~/blissful-tuner/)
- src/musubi_tuner/flux_2_train_network.py — Flux2NetworkTrainer, call_dit()
- src/musubi_tuner/hv_train_network.py — NetworkTrainer base class, training loop (lines ~2400-2520 for loss), call_dit (line ~1749), compute_loss_weighting_for_sd3 (line ~350)
- src/musubi_tuner/flux_2/flux2_models.py — Flux2 model class, patchify/unpatchify/scale/pack
- src/musubi_tuner/flux_2/flux2_utils.py — text encoding (Mistral3Embedder, Qwen3Embedder), timestep shift, guidance
- src/musubi_tuner/flux_2/__init__.py — imports and utilities
- src/musubi_tuner/flux_2_cache_latents.py — latent caching pipeline
- src/musubi_tuner/networks/lora_flux_2.py — LoRA target modules, exclusion patterns
- src/musubi_tuner/networks/network_arch.py — architecture registry
- src/musubi_tuner/modules/mask_loss.py — mask-weighted loss

## Comparison Domains (investigate in this order)

### Domain 1: Flow Matching Formula (CRITICAL)
Read the EXACT noise addition formula in both codebases. Verify:
- Is it `noisy = sigma * noise + (1 - sigma) * clean`?
- SimpleTuner reference says `noisy_latents = (1 - sigmas) * latents + sigmas * noise`
- How is sigma computed? Sigmoid distribution? Uniform? Beta?
- Compare all available sigma sampling strategies

### Domain 2: Latent Scaling (CRITICAL)
SimpleTuner reference says `AUTOENCODER_SCALING_FACTOR = 1.0` (no rescaling).
- Does blissful-tuner apply any latent scaling? Read the actual VAE encode/decode path.
- Does SimpleTuner's VAE (AutoencoderKLFlux2) internally handle scaling via batch norm?
- Are the approaches equivalent despite appearing different?

### Domain 3: Loss Target Construction (CRITICAL)
SimpleTuner reference: `target = noise - latents` (velocity field).
- Is this pre- or post-patchification in SimpleTuner?
- Compare exact loss reduction: `loss.mean(dim=spatial_dims).mean(dim=batch)`
- Does blissful-tuner use the same reduction?

### Domain 4: Timestep Normalization (HIGH)
- SimpleTuner: `timesteps = sigmas * 1000.0` then `timesteps / 1000.0` to transformer
- Verify the resolution-dependent shift formula matches
- Compare sigmoid scale parameter (SimpleTuner default: 5.0)

### Domain 5: Patchification / Packing Order (HIGH)
- SimpleTuner packs latents with position IDs THEN adds noise
- Or does noise happen before packing?
- Trace the exact order in both codebases
- SimpleTuner: `LATENT_CHANNEL_COUNT = 128` (32 * 4 from pixel shuffle)

### Domain 6: Guidance Embedding During Training (MEDIUM)
- SimpleTuner: always 1.0 during training (reference doc)
- Verify in actual source
- Does blissful-tuner hardcode 1.0 or make it configurable?

### Domain 7: Position ID Construction (MEDIUM)
- SimpleTuner uses 4D tuples (t, h, w, l)
- Reference images get TIME-OFFSET position IDs
- Compare with blissful-tuner's coordinate construction

### Domain 8: LoRA Targeting Scope (MEDIUM)
- SimpleTuner default targets: ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.to_qkv_mlp_proj"]
- Compare with blissful-tuner's targeting (which uses block-level targets for custom LoRA)
- Note any targeting modes SimpleTuner has that blissful-tuner doesn't (e.g., "tiny", "mlp", "slider")

## Output Format

Write your findings to: /Users/dustin/blissful-tuner/docs/plans/2026-02-18-simpletuner-comparison-findings.md

Use this format for EACH finding:

## Finding [ST-N]: [Short Title]
**Severity:** CRITICAL / WARNING / INFO / OK
**Domain:** [1-8]
**What SimpleTuner does:** [description with exact file:line references]
**What blissful-tuner does:** [description with exact file:line references]
**Discrepancy:** [what differs and why it matters, or "None — implementations agree"]
**Recommendation:** [action needed, if any]

IMPORTANT: For every domain, produce at least one finding (even if "OK — implementations agree").
Include a summary table at the top with all findings by severity.
```

**Step 2: Verify Agent B launched successfully**

Check the background task output file exists and agent has started reading files.

---

## Task 3: Monitor Agent Progress

**Purpose:** Periodically check both agents' progress while they run in background.

**Step 1:** After ~2 minutes, tail both output files to verify agents are making progress.

**Step 2:** After ~5 minutes, check if either agent has completed.

**Step 3:** Wait for both agents to complete before proceeding to Task 4.

---

## Task 4: Synthesis — Cross-Reference Findings

**Purpose:** Merge findings from both agents, cross-reference, and produce final documentation.

**Depends on:** Task 1 and Task 2 both complete.

**Step 1: Read both findings reports**

- Read `/Users/dustin/blissful-tuner/docs/plans/2026-02-18-onetrainer-comparison-findings.md`
- Read `/Users/dustin/blissful-tuner/docs/plans/2026-02-18-simpletuner-comparison-findings.md`

**Step 2: Cross-reference findings**

For each of the 8 domains, classify:

| Pattern | Confidence | Action |
|---------|-----------|--------|
| Both trainers agree, differ from blissful-tuner | **HIGH** | Likely bug in blissful-tuner — investigate |
| One trainer differs, other agrees with blissful-tuner | **MEDIUM** | Design choice difference — document |
| Both trainers differ from each other AND blissful-tuner | **LOW** | Multiple valid approaches — note |
| All three agree | **CONFIRMED OK** | No action needed |

**Step 3: Write synthesis document**

Write to: `/Users/dustin/blissful-tuner/docs/plans/2026-02-18-flux2-cross-trainer-synthesis.md`

Structure:
```markdown
# FLUX.2 Cross-Trainer Comparison — Synthesis Report

## Executive Summary
- N high-confidence findings (both trainers agree, blissful-tuner differs)
- N medium-confidence findings (one trainer differs)
- N confirmed-OK areas

## High-Confidence Findings (Both Trainers Agree)
[findings where both OneTrainer and SimpleTuner do the same thing differently from blissful-tuner]

## Medium-Confidence Findings (One Trainer Differs)
[findings where only one external trainer differs]

## Confirmed OK
[areas where all three implementations agree]

## Recommended Actions
[prioritized list of changes to investigate/implement]
```

**Step 4: Commit synthesis document**

```bash
git add docs/plans/2026-02-18-onetrainer-comparison-findings.md \
        docs/plans/2026-02-18-simpletuner-comparison-findings.md \
        docs/plans/2026-02-18-flux2-cross-trainer-synthesis.md
git commit -m "doc(plans): add FLUX.2 cross-trainer comparison findings and synthesis"
```

---

## Task 5: Review and Next Steps

**Purpose:** Present findings to user and determine action items.

**Step 1:** Summarize the synthesis report for the user, highlighting:
- Any CRITICAL findings that need immediate investigation
- WARNING findings that may affect training quality
- Feature gaps worth considering

**Step 2:** Ask user which findings to act on, if any.
