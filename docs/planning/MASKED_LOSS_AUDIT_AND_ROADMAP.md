# Blissful Tuner: Masked Loss & Prior Preservation Audit & Enhancement Roadmap

**Date:** February 25, 2026  
**Context:** This document consolidates a deep architectural review of the Blissful Tuner masked loss and prior preservation system (`src/musubi_tuner/modules/mask_loss.py` and related training loops). It outlines critical bug fixes, high-ROI feature enhancements, and notes on existing mathematical robustness.

**Agent Instructions:** Use this document as a blueprint for modifying the training loops (e.g., `hv_train_network.py`, `wan_train_network.py`) and the mask loss module.
*   **Important:** Remember to add any new CLI arguments (e.g., `--mask_area_scale_beta`, `--ema_teacher`) to the `add_mask_loss_args` function in `mask_loss.py` or the respective training script's parser.
 *   **ALSO:**I have done my best to verify the contents in Section 3 as correct to avoid introducing regressions in already-solved edge cases, but of course you are welcome to review them yourself to confirm their veracity.

---

## 1. Critical Bugs & Edge Cases (Priority: Immediate)

These issues affect the correctness of the training loop under specific configurations (specifically `batch_size > 1`, temporal videos, and teacher stochasticity).

### A. Timestep Gating Logic for `batch_size > 1`
**The Issue:** In the training loop, the timestep gating logic for the teacher forward pass currently uses an `all()` condition (e.g., `if all(timesteps > threshold)`). If a batch contains timesteps `[900, 800, 200, 100]` with a threshold of `300`, the condition evaluates to `False`, skipping the teacher pass for the *entire batch*. 
**The Fix:** 
1. Evaluate the threshold condition per-sample to create a boolean mask (`do_prior_mask`).
2. Trigger the teacher pass if `any()` sample meets the condition.
3. Multiply the `prior_loss_unreduced` tensor by the broadcasted boolean mask to zero out the prior loss for samples below the threshold.

*Pseudocode Implementation (in `train_network.py` variants):*
```python
# Create boolean mask for the batch
do_prior_mask = timesteps >= args.prior_preservation_timestep_threshold # Shape: (B,)
run_teacher = do_prior_mask.any().item()

if run_teacher and args.prior_preservation_weight > 0:
    # ... setup teacher pass ...
    with torch.no_grad(), disable_lora():
        teacher_pred = network(noisy_latents, timesteps, ...)
    
    prior_loss_unreduced = F.mse_loss(model_pred, teacher_pred, reduction="none")
    
    # Zero out prior loss for batch items below the threshold
    # Reshape to broadcast against (B, C, F, H, W) or (B, C, H, W)
    b_shape =[-1] + [1] * (prior_loss_unreduced.ndim - 1)
    broadcast_mask = do_prior_mask.view(*b_shape).to(prior_loss_unreduced.dtype)
    prior_loss_unreduced = prior_loss_unreduced * broadcast_mask
else:
    prior_loss_unreduced = None

```

### B. State Management During Teacher Forward Pass

**The Issue:**: The teacher forward pass runs under `torch.no_grad()` and with adapters disabled, but the base network remains in `.train()` mode. If any future architecture uses Dropout or tracks running statistics, (e.g., BatchNorm, though rare in DiTs), the teacher targets will become stochastic/noisy, corrupting the distillation process.
**The Fix:** Explicitly toggle `.eval()` and `.train()`  around the teacher pass.


*Pseudocode Implementation:*
```python
if run_teacher and args.prior_preservation_weight > 0:
    network.eval() # <-- ADD THIS
    with torch.no_grad(), disable_lora():
        teacher_pred = network(noisy_latents, timesteps, ...)
    network.train() # <-- ADD THIS
```


### C. Prevent Gradient Explosion on Tiny Masks (Area-Aware Scaling)

**The Issue:** The mathematically correct weighted mean `sum(loss * w) / sum(w)` prevents loss dilution, but it causes a severe optimization hazard for very small masks. If a mask covers only 1% of the image, the gradient applied to those pixels is scaled up by 100x, which can destabilize AdamW and cause NaNs.

**The Fix:** Implement an explicit area-scaling exponent (e.g., `--mask_area_scale_beta`). 
*   Let `a = clamp(mask_area / total_area, eps, 1.0)`.
*   Multiply the final masked mean by `a ** beta`.
*   `beta = 0.0` (Default): Strict weighted mean (current behavior, area independent).
*   `beta = 1.0`: Strict global mean (safe, but dilutes).
*   `beta = 0.5`: A safe middle-ground that prevents gradient explosions on tiny masks while still prioritizing the subject.

*Pseudocode Implementation:*
```python

# Calculate area ratio
eps = 1e-8
m = mask_processed.float()
A = m.sum(dim=reduce_dims, keepdim=True)
N = torch.tensor(m.numel() // m.shape[0], device=m.device, dtype=m.dtype) # total pixels per sample
area_ratio = (A / N).clamp_min(eps)

# Calculate weighted mean
target_loss_weighted = loss * m
per_sample_target = target_loss_weighted.sum(dim=reduce_dims, keepdim=True) / A.clamp_min(eps)

# Apply Area Scaling Beta (if args.mask_area_scale_beta > 0)
beta = getattr(args, "mask_area_scale_beta", 0.0)
if beta > 0.0:
    per_sample_target = per_sample_target * (area_ratio ** beta)

L_target = per_sample_target.mean()

```

### D. The Spatiotemporal Video Masking Limitation
**The Issue:** Currently, the codebase globs a single static 2D image mask and broadcasts it across all frames of a video tensor. If the subject moves outside the bounds of this static mask during the video, the teacher prior will forcefully attempt to overwrite the subject with the background, causing severe ghosting and identity collapse.
**The Fix:**  Modify `VideoDataset` and `wan_cache_latents.py` to support 3D Spatiotemporal Masking. Allow `mask_directory` to accept either an .mp4 file containing a moving mask, or a subdirectory of sequential mask frames that map 1:1 to the video's frame count.


### E. Resolve Qwen-Image Layered Layout Limitations
**The Issue:** Qwen-Image's `layout="layered"` operates on separate spatial pathways (foreground/background/details) rather than a flat tensor, which currently triggers a `NotImplementedError` for prior preservation.
**The Fix:** Develop a custom `apply_masked_loss_with_prior` pathway specifically for the layered layout. This requires partitioning the 5D mask tensor (B, L, 1, H, W) to map against the specific foreground/background streams of the Qwen architecture, computing the prior loss exclusively against the background layer's output from the teacher.

## 2. High-ROI Architectural Enhancements (Priority: Next Release)

### A. Timestep-Adaptive Prior Weight Scheduling
**The Concept:** Hallucinations/phantom limbs form at high noise levels ($t > 500$) where structural composition is determined. At low noise levels ($t < 300$), the model refines details and lighting. Applying a rigid $w_{prior}$ at low noise prevents the background from naturally adjusting its lighting/shadows to match the new LoRA subject.
**The Feature:** Introduce `--prior_decay_schedule` (e.g., `constant`, `cosine`, `linear`). Scale `args.prior_preservation_weight` dynamically based on the current normalized timestep.
*   **Warmup:** Linearly scale $w_{prior}$ from 0 to its maximum over the first 5-10% of steps to avoid fighting early chaotic LoRA gradients.
*   **High Noise:** $w_{prior}$ is at maximum (strictly preserve background structure).
*   **Low Noise:** $w_{prior}$ decays toward 0 (allow background lighting to harmonize with the subject).

### B. EMA Teacher (Exponential Moving Average)
**The Concept:** Using the completely frozen base model as a teacher causes a stylistic clash if the user is fine-tuning a heavily stylized LoRA (e.g., an anime style). 
**The Feature:** Maintain an Exponential Moving Average (EMA) copy of the model weights (base + LoRA). During the teacher pass, use the EMA weights instead of completely disabling the LoRA. 
*   **Implementation Note:** Initialize the EMA model outside the main step loop. Freeze the teacher as the base model for the warmup period, and only begin accumulating and using the EMA *after* the warmup phase is complete.

### C. Huber Loss for Boundary Robustness (Claude Insight)
**The Issue:** Sharp transitions in mask weights (e.g., from 1.0 to 0.0) can cause the MSE loss to spike unnaturally at the mask boundaries, leading to edge artifacts or "halos."
**The Fix:** Allow the target loss to utilize Huber loss instead of MSE. Huber acts as a shock-absorber for boundary spikes (quadratic for small errors, linear for large ones).
*   **Implementation:** Ensure `apply_masked_loss_with_prior` respects `--loss_type huber` if passed from the main training loop.

### D. Mask-Aware Timestep Sampling
**The Concept:** If a user mask only covers 5% of the frame (e.g., face-only training), sampling high-noise timesteps is computationally inefficient because the overall image structure is already defined by the prior. 
**The Feature:** In the `BucketBatchManager`, calculate the average mask coverage percentage for a batch. If the coverage is very small, bias the timestep sampler heavily toward lower noise levels (where high-frequency detail learning occurs). This will drastically accelerate convergence for localized features like faces.
**Implementation Note:** Use PyTorch Beta distributions to dynamically shift the timestep probability curve. For small masks (detail-focused), sample t using torch.distributions.Beta(2, 5). For large masks (structure-focused), use Beta(5, 2).

### E. Frequency-Aware Masking (Timestep-Dependent Blur)
**The Concept:** At high noise levels, the model needs to understand how the subject blends into the background globally. At low noise levels, it should strictly focus on the subject's pores/textures.
**The Feature:** Implement a `--dynamic_mask_blur` flag. The `mask_blur_kernel_size` becomes a function of $t$. At high noise, apply a massive blur to the mask (soft boundaries). At low noise, reduce the blur to 0 (hard boundaries).

### F. Stochastic Full-Image Regularization (Claude/OneTrainer Insight)
**The Issue:** A strict, deterministic teacher prior can sometimes over-constrain the background, preventing the global lighting of the image from harmonizing.
**The Fix:** Implement `--unmasked_probability <float>`. For example, at `0.05`, 5% of the time the system completely bypasses `apply_masked_loss_with_prior` and calculates standard global loss on the ground-truth image. This provides occasional "global harmonization" signals.

---

## 3. Consdiered Correct

*   **Weighted-Mean Normalization:** The implementation `sum(loss * w) / sum(w)` (scaled by `num_channels`) is correct. I do not recommend changing this to a standard `.mean()` reduction.
*   **Threshold Mode Overlap Conflict:** The concern that `min_weight` will cause conflicting gradients with the prior mask is already handled. The code explicitly does `mask_processed = mask_processed * (1 - prior_mask)`, ensuring mutually exclusive optimization zones.
*   **Division by Zero (NaNs):** The code already utilizes `.clamp_min(1e-8)` and `valid_target` boolean checks to safely handle empty or near-empty mask sums.
*   **Float32 Precision Casting:** The concern about bfloat16 precision loss during massive tensor summation is already handled. `target_loss_weighted.sum(dim=reduce_dims, dtype=torch.float32)` safely performs the summation in float32. *(Agent Note: For global reductions on massive video tensors, it is acceptable/encouraged to perform this specific summation in `float64` before casting back to `float32` to prevent accumulator swamping).*
---

## 4. Documentation Updates

*   **Latent Space Blur Radius:** Update `docs/MASKED_LOSS_TRAINING_GUIDE.md` to explicitly warn users that `mask_blur_kernel_size` operates in *latent space*. Because WAN and FLUX use a spatial compression factor of $8\times$, a `kernel_size=3` blurs a $24 \times 24$ pixel radius in real space. Users should be advised to use small values.
*   Tiny Mask Warnings: Document the new --mask_area_scale_beta. Warn users that if they are training on tiny masks (e.g., distant faces), they should set --mask_area_scale_beta 0.5 to prevent exploding gradients/NaNs.
* Threshold vs. Blur Interaction: Explicitly state that --prior_mask_threshold is calculated using the raw, unblurred mask. Therefore, adjusting --mask_blur_kernel_size or --mask_gamma changes the target learning softness, but does not shift the hard boundary of the teacher's background prior.
* In  `MASKED_LOSS_TRAINING_GUIDE.md`, under the "Why this matters" section, consider explicitly stating: "Unlike other implementations that calculate priors with active adapters, Blissful Tuner temporarily disables LoRA adapters during the teacher pass. This guarantees a pristine, untainted base-model prior, completely preventing recursive self-hallucination."
*  **Spatiotemporal Video Warning:** Explicitly state: *"Warning: Video datasets currently use a single static 2D mask for all frames. The subject MUST remain relatively stationary within the masked region. If the subject walks out of the mask, the prior preservation loss will attempt to erase them."*


## 5. Long-Term Research Frontiers (Experimental)

### Contrastive Spatial Decoupling (InfoNCE Loss)
**The Concept:** Currently, prior preservation acts as a passive anchor (preventing the background from changing). A contrastive loss could *actively* force the network to decouple the subject from the environment.
**The Implementation:** Extract the intermediate hidden states (features) from the DiT blocks. Apply the mask to separate the "Foreground Features" and "Background Features". Apply an InfoNCE (contrastive) loss to minimize the cosine similarity between the foreground and background representations. This explicitly teaches the network that the subject is a distinct entity from its surrounding environment, attacking the "phantom limb" problem at the manifold representation level.

### Cross-Attention Localization Penalty
**The Concept:** Prevent the trigger word from "looking" at the background.
**The Implementation:** Extract the cross-attention map for the subject token (e.g., from layer 8 of the DiT). Apply a penalty: `leakage_loss = (attention_map * (1 - mask)).sum()`. This teaches the attention heads to strictly bind the concept to the spatial region, killing phantom limbs at the attention level.

### Self-Perceptual Prior Loss (LPL)
**The Concept:** MSE on noise predictions can over-smooth backgrounds. 
**The Implementation:** Instead of computing MSE on the final output tensor for the prior, extract the intermediate feature representations from a middle DiT block for both the Teacher and Student. Apply the `prior_mask` to these feature maps and calculate MSE. This enforces structural/perceptual identity without demanding exact pixel parity.