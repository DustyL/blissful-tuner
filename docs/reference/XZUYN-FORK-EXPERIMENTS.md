# xzuyn Fork Experimental Features

**Fork URL:** https://github.com/xzuyn/musubi-tuner
**Local Clone:** `/Users/dustin/musubi-tuner-forks/xzuyn`
**Last Reviewed:** 2026-01-30
**Status:** Monitoring for maturity

This document tracks interesting experimental features from xzuyn's musubi-tuner fork that may be candidates for future integration into blissful-tuner once they are more fleshed out and validated.

---

## Branch 1: `custom_flux2`

**Purpose:** Alternative timestep sampling strategy for FLUX.2 training that matches the inference schedule distribution.

**Latest Commit:** `abb3c65` (2026-01-30)

### Concept

Instead of using the standard `flux2_shift` approach (sigmoid with dynamic mu), this method:

1. Pre-computes all sigma values from the official FLUX.2 scheduler for step counts 10-100
2. Creates a "candidate pool" of valid timesteps
3. Samples from this pool using kernel density estimation (KDE) with Gaussian smoothing

### Key Implementation Details

```python
# Official FLUX.2 empirical coefficients (from Black Forest Labs)
A1, B1 = 8.73809524e-05, 1.89833333  # optimal μ for 10 steps
A2, B2 = 0.00016927, 0.45666666      # optimal μ for 200 steps

def flux2_scheduler(num_steps, image_seq_len):
    if image_seq_len > 4300:  # ~1049x1049 resolution
        mu = float(A2 * image_seq_len + B2)
    else:
        m_10 = A1 * image_seq_len + B1
        m_200 = A2 * image_seq_len + B2
        a = (m_200 - m_10) / 190.0  # slope between 10 and 200 steps
        mu = float(a * num_steps + (m_200 - 200.0 * a))
    return math.exp(mu) / (math.exp(mu) + (1 / torch.linspace(1, 0, num_steps + 1) - 1))

# Aggregate sigmas from step counts 10-100
for n in range(10, 101):
    mixed_list.extend(flux2_scheduler(n, seq_len))
candidates = unique(mixed_list)

# KDE sampling
centers = random_choice(candidates, batch_size)
bandwidth = (candidates.max() - candidates.min()) / (len(candidates) / 4)
t = centers + randn(batch_size) * bandwidth
t = clamp(t, 0.001, 0.999)
```

### Comparison with Current `flux2_shift`

| Aspect | blissful-tuner `flux2_shift` | xzuyn `custom_flux2` |
|--------|------------------------------|----------------------|
| Mu calculation | `get_lin_function(y1=0.5, y2=1.15)(h * w)` | Official BFL formula with dual linear segments |
| Sampling method | Sigmoid of normal with shift | KDE over pre-computed inference sigmas |
| Resolution awareness | Uses `h * w` directly | Uses `h * w / 256` (seq_len) |
| Step count awareness | No | Yes (aggregates from 10-100 steps) |
| Distribution shape | Continuous shifted sigmoid | Empirical from inference schedules |

### Optimizations

- Per-resolution caching via `self.flux2_candidates_cache[seq_len]`
- Avoids recomputation for same-resolution batches

### Source Reference

- Official FLUX.2 sampling: https://github.com/black-forest-labs/flux2/blob/b56ac61450f56ea7d32374c2fa54e77a262067f6/src/flux2/sampling.py#L240-L266

### Maturity Assessment

| Indicator | Status |
|-----------|--------|
| Active development | Yes (commits same day as review) |
| Bug fixes in progress | Yes (step range 4→10 correction) |
| Documentation | Partial (inline comments added) |
| Validated results | Unknown |
| Production ready | **No** |

### Integration Considerations

- Would add new `--timestep_sampling custom_flux2` option
- Requires adding caching infrastructure to trainer class
- Should validate that seq_len calculation matches FLUX.2 architecture
- Need to compare training results against baseline `flux2_shift`

---

## Branch 2: `n_timesteps_per_step`

**Purpose:** Compute loss over multiple timesteps per optimization step to improve gradient signal and timestep coverage.

**Latest Commit:** `359439a` (2026-01-30)

### Concept

Standard flow matching training computes loss at a single random timestep per step. This approach:

1. Runs multiple forward passes with different timesteps per step
2. Accumulates gradients across all timesteps
3. Performs optimizer step after all timestep gradients are accumulated

### Key Implementation Details

**New argument:**
```bash
--n_timesteps_per_step N  # default=1
```

**Effective timestep coverage:**

| Config | Forward passes/step | Timesteps seen/step |
|--------|---------------------|---------------------|
| BS1 GA1 n=1 | 1 | 1 |
| BS1 GA1 n=2 | 2 | 2 |
| BS1 GA2 n=2 | 4 | 4 |
| BS1 GA4 n=4 | 16 | 16 |
| BS2 GA4 n=4 | 16 | 32 |

**Memory-efficient backward:**
```python
noise = torch.randn_like(latents)  # Single noise sample (reused)
for _ in range(args.n_timesteps_per_step):
    noisy_model_input, timesteps = get_noisy_model_input_and_timesteps(...)
    model_pred, target = call_dit(...)
    loss_i = mse_loss(model_pred, target)

    # Scale and backward immediately (frees activation memory)
    scaled_loss = loss_i.mean() / float(args.n_timesteps_per_step)
    accelerator.backward(scaled_loss)

    total_loss += loss_i.mean().detach().item()  # For logging only
```

### Design Decisions

**Noise reuse (current approach):**
- Same noise tensor used for all timesteps within a step
- Rationale: "How well does the model denoise this specific noise pattern at different noise levels?"
- More consistent gradient signal across timestep dimension
- Author's note: "i think reusing the noise might work better, but i'm not 100% sure yet"

**Per-timestep backward:**
- Avoids keeping all intermediate activations in memory
- Gradients accumulate in-place via PyTorch autograd
- Loss scaling (`/ n_timesteps`) ensures gradient magnitude is correct

### Metadata Logging

Saves `ss_n_timesteps_per_step` to LoRA metadata for reproducibility.

### Practical Implications for Video Training

| Aspect | Impact |
|--------|--------|
| Training time | `n × longer` per step |
| VRAM usage | Similar (backward frees memory between timesteps) |
| Timestep coverage | Better coverage of noise schedule per step |
| Effective "batch diversity" | Higher (more timesteps per optimizer step) |
| Learning rate | May need adjustment |

### Example Equivalent Configs

```bash
# Traditional approach:
--batch_size 1 --gradient_accumulation_steps 4 --n_timesteps_per_step 1
# Result: 4 timesteps per optimizer step

# Multi-timestep approach:
--batch_size 1 --gradient_accumulation_steps 2 --n_timesteps_per_step 2
# Result: 4 timesteps per optimizer step (same total, different distribution)

# Aggressive multi-timestep:
--batch_size 1 --gradient_accumulation_steps 4 --n_timesteps_per_step 4
# Result: 16 timesteps per optimizer step
```

### Maturity Assessment

| Indicator | Status |
|-----------|--------|
| Active development | Yes (changed day of review) |
| Uncertainty expressed | Yes ("i'm not 100% sure yet") |
| Memory efficiency | Good (per-timestep backward) |
| Logging/metadata | Good |
| Validated results | Unknown |
| Production ready | **No** |

### Integration Considerations

- Relatively straightforward to add to training loop
- Need to determine optimal noise strategy (reuse vs fresh)
- Should benchmark training time impact for WAN 2.2 14B
- May require learning rate schedule adjustments
- Could interact with gradient accumulation in unexpected ways

---

## Future Actions

### Monitoring

- [ ] Check fork periodically for new commits/branches
- [ ] Watch for any published training results or comparisons
- [ ] Monitor if features get merged upstream to kohya-ss/musubi-tuner

### Validation Before Integration

For `custom_flux2`:
- [ ] Compare training curves against baseline `flux2_shift`
- [ ] Validate seq_len calculation for FLUX.2 architecture
- [ ] Test across multiple resolutions

For `n_timesteps_per_step`:
- [ ] Benchmark training time overhead for video models
- [ ] Compare noise reuse vs fresh noise
- [ ] Test learning rate adjustments
- [ ] Validate gradient scaling is correct with accelerate

### Commands to Sync Fork

```bash
cd /Users/dustin/musubi-tuner-forks/xzuyn
git fetch origin
git log --oneline origin/custom_flux2 -5
git log --oneline origin/n_timesteps_per_step -5
```

---

## Changelog

| Date | Event |
|------|-------|
| 2026-01-30 | Initial documentation created from code review |
