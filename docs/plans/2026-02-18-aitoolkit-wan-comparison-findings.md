# AI-Toolkit vs Blissful-Tuner: WAN 2.1/2.2 Training Pipeline Comparison

**Date:** 2026-02-18
**Codebases compared:**
- ai-toolkit: `/Users/dustin/ai-toolkit/` (ostris/ai-toolkit)
- blissful-tuner: `/Users/dustin/blissful-tuner/` (DustyL/blissful-tuner)

---

## Summary Table

| ID | Title | Severity | Domain |
|----|-------|----------|--------|
| AT-WAN-01 | Flow matching formula agreement | OK | 1: Flow Matching |
| AT-WAN-02 | Loss target (velocity prediction) agreement | OK | 1: Flow Matching |
| AT-WAN-03 | 5D tensor handling in noise addition | OK | 1: Flow Matching |
| AT-WAN-04 | Timestep +1 offset in blissful-tuner | INFO | 1: Flow Matching |
| AT-WAN-05 | Dual-stage boundary routing agreement | OK | 2: Dual-Stage Architecture |
| AT-WAN-06 | Stage switching: deterministic vs rejection sampling | WARNING | 2: Dual-Stage Architecture |
| AT-WAN-07 | Dual-model memory: weight swap vs two-model wrapper | INFO | 2: Dual-Stage Architecture |
| AT-WAN-08 | No split LoRA save in blissful-tuner | WARNING | 2: Dual-Stage Architecture |
| AT-WAN-09 | VAE per-channel normalization agreement | OK | 3: VAE Latent Normalization |
| AT-WAN-10 | VAE returns mu only (no reparameterize) | OK | 3: VAE Latent Normalization |
| AT-WAN-11 | Training shift mismatch: 5.0 (ai-toolkit) vs user-specified (blissful-tuner) | WARNING | 4: Timestep Sampling |
| AT-WAN-12 | Shift formula agreement | OK | 4: Timestep Sampling |
| AT-WAN-13 | Blissful-tuner has richer timestep sampling strategies | INFO | 4: Timestep Sampling |
| AT-WAN-14 | Loss computation agreement (MSE + velocity target) | OK | 5: Loss Computation |
| AT-WAN-15 | Blissful-tuner has structured mask loss; ai-toolkit has simpler mask | INFO | 5: Loss Computation |
| AT-WAN-16 | ai-toolkit loss weighting features not in blissful-tuner | INFO | 5: Loss Computation |
| AT-WAN-17 | UMT5 text encoder: same model, different wrappers | OK | 6: Text Encoding |
| AT-WAN-18 | Text length agreement (512) | OK | 6: Text Encoding |
| AT-WAN-19 | Variable-length T5 embeddings agreement | OK | 6: Text Encoding |
| AT-WAN-20 | LoRA targeting scope: WanAttentionBlock vs all Linear layers | WARNING | 7: LoRA Targeting |
| AT-WAN-21 | No noise augmentation stack in blissful-tuner | INFO | 8: Noise Augmentation |
| AT-WAN-22 | Signal correction noise absent from blissful-tuner | INFO | 8: Noise Augmentation |

---

## Finding AT-WAN-01: Flow Matching Formula Agreement
**Severity:** OK
**Domain:** 1: Flow Matching Formula

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/samplers/custom_flowmatch_sampler.py:91-102`:
```python
def add_noise(self, original_samples, noise, timesteps):
    t_01 = (timesteps / 1000).to(original_samples.device)
    noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
    return noisy_model_input
```

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1125`:
```python
noisy_model_input = (1 - t) * latents + t * noise
```
Where `t` is sampled in [0, 1] range and timesteps = t * 1000 (line 1123).

**Discrepancy:** None -- implementations agree. Both use `x_t = (1-t)*x_0 + t*noise` where t is in [0, 1]. ai-toolkit normalizes timesteps from [0, 1000] to [0, 1]; blissful-tuner samples t directly in [0, 1] for the shift/sigmoid/uniform paths.

**Recommendation:** None.

---

## Finding AT-WAN-02: Loss Target (Velocity Prediction) Agreement
**Severity:** OK
**Domain:** 1: Flow Matching Formula

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/models/wan21/wan21.py:674-681`:
```python
def get_loss_target(self, *args, **kwargs):
    noise = kwargs.get('noise')
    batch = kwargs.get('batch')
    return (noise - batch.latents).detach()
```

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan_train_network.py:851`:
```python
target = noise - latents
```

**Discrepancy:** None -- both compute velocity as `v = noise - latents`. This is the derivative of the linear interpolation ODE: d(x_t)/dt = noise - x_0.

**Recommendation:** None.

---

## Finding AT-WAN-03: 5D Tensor Handling in Noise Addition
**Severity:** OK
**Domain:** 1: Flow Matching Formula

**What ai-toolkit does:**
The `add_noise` in `custom_flowmatch_sampler.py:91-102` uses element-wise broadcasting. The `t_01` tensor is shaped `(B,)` or `(B, 1)` after the division. When multiplied with the 5D `original_samples` tensor `(B, C, T, H, W)`, standard PyTorch broadcasting handles this correctly because trailing dimensions are broadcast-expanded.

The base model's `add_noise` at `/Users/dustin/ai-toolkit/toolkit/models/base_model.py:759-776` chunks per batch element and calls the scheduler's `add_noise` per-element, so the t_01 shape is always `(1,)` against a `(1, C, T, H, W)` tensor -- no 5D issue.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1124`:
```python
t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
noisy_model_input = (1 - t) * latents + t * noise
```

Explicitly reshapes `t` to match 5D or 4D latent dimensions before the interpolation.

**Discrepancy:** None -- both correctly handle 5D (video) tensors. Blissful-tuner is more explicit about the reshape, which is arguably better practice for clarity.

**Recommendation:** None.

---

## Finding AT-WAN-04: Timestep +1 Offset in Blissful-Tuner
**Severity:** INFO
**Domain:** 1: Flow Matching Formula

**What ai-toolkit does:**
Timesteps are sampled from `linspace(1000, 1, N)` (line 117 of `custom_flowmatch_sampler.py`), producing values in [1, 1000]. With shift applied, the range is modified via `sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)`, but timesteps never go below 1.

**What blissful-tuner does:**
For the shift/sigmoid/uniform paths (`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1123-1127`):
```python
timesteps = t * 1000.0
...
timesteps += 1  # 1 to 1000
```
`t` is sampled in [0, 1], scaled to [0, 1000], then 1 is added, producing [1, 1001]. In practice, `t=1.0` is essentially impossible from sigmoid/shift sampling, so the effective range is (1, ~1000].

For the sigma-based path (line 1143):
```python
timesteps = noise_scheduler.timesteps[indices].to(device=device)  # 1 to 1000
```
Uses pre-computed timesteps from the scheduler which starts at `sigmas * num_train_timesteps` with sigmas from linspace(1, 0, N+1).

**Discrepancy:** Minor. The `+1` in blissful-tuner's shift/uniform/sigmoid path ensures timesteps are never exactly 0, matching the [1, 1000] convention. Both codebases avoid t=0 (pure clean signal). The conventions agree in effect.

**Recommendation:** None needed -- this is a standard convention difference with no practical impact.

---

## Finding AT-WAN-05: Dual-Stage Boundary Routing Agreement
**Severity:** OK
**Domain:** 2: WAN 2.2 Dual-Stage Architecture

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/extensions_built_in/diffusion_models/wan22/wan22_14b_model.py:124-128`:
```python
def forward(self, hidden_states, timestep, ...):
    with torch.no_grad():
        if timestep.float().mean().item() > self.boundary:
            t_name = "transformer_1"
        else:
            t_name = "transformer_2"
```
With `boundary = 0.875 * 1000 = 875` for T2V and `0.9 * 1000 = 900` for I2V.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan_train_network.py:137-150`:
```python
self.timestep_boundary = (
    args.timestep_boundary if args.timestep_boundary is not None else self.config.boundary
)
if self.timestep_boundary > 1:
    self.timestep_boundary /= 1000.0
```
And boundary decision at line 647:
```python
high_noise = sample_timesteps[0] / 1000.0 >= self.timestep_boundary
```

Config values: T2V boundary = 0.875 (`wan_t2v_A14B.py:41`), I2V boundary = 0.9 (from config).

**Discrepancy:** None -- both use the same boundary values (0.875 for T2V, 0.9 for I2V). ai-toolkit uses `>` (strictly greater), blissful-tuner uses `>=` (greater or equal). For continuous timestep sampling, the probability of hitting exactly 875 is essentially zero, so this is not a practical concern.

**Recommendation:** None.

---

## Finding AT-WAN-06: Stage Switching: Deterministic Round-Robin vs Rejection Sampling
**Severity:** WARNING
**Domain:** 2: WAN 2.2 Dual-Stage Architecture

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py:2048-2055` and `/Users/dustin/ai-toolkit/jobs/process/BaseSDTrainProcess.py:1194-1205`:

ai-toolkit uses a deterministic round-robin approach. The `switch_boundary_every` parameter controls how many steps to spend in each stage. Every N steps, it advances `current_boundary_index`, which constrains the timestep sampling range to [boundary_max, boundary_min] for the current stage. This means:
- Step 1-10: sample timesteps only from [875, 1000] (high-noise model)
- Step 11-20: sample timesteps only from [0, 875] (low-noise model)
- ...and so on

The timestep sampling within each stage range is handled by clipping `min_noise_steps`/`max_noise_steps` indices.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan_train_network.py:627-684`:

Blissful-tuner uses rejection sampling. It first samples a single timestep to determine which expert to use for the entire batch, then rejection-samples additional timesteps to fill the batch:
```python
# First sample determines expert choice
noisy_model_input, sample_timesteps = super().get_noisy_model_input_and_timesteps(...)
high_noise = sample_timesteps[0] / 1000.0 >= self.timestep_boundary
self.next_model_is_high_noise = high_noise

# Rejection sampling for remaining batch elements
for i in range(bsize):
    for _ in range(num_max_calls):
        noisy_model_input, ts_i = super().get_noisy_model_input_and_timesteps(...)
        if (high_noise and ts_i[0] / 1000.0 >= self.timestep_boundary) or
           (not high_noise and ts_i[0] / 1000.0 < self.timestep_boundary):
            break
```

**Discrepancy:** Significant design difference.

1. **ai-toolkit**: Deterministic switching ensures equal training time per stage (modulo `switch_boundary_every`). Each step trains only one expert, and the transition is predictable.

2. **blissful-tuner**: The first random sample determines the expert for the entire batch. For shift sampling with the default boundary of 0.875, the high-noise region [0.875, 1.0] has only 12.5% of the probability mass (even less with shift applied), so high-noise training is much rarer. The comment in the code acknowledges this: "~8 retries avg for high-noise (12.5% acceptance), ~1.1 for low-noise (87.5%)."

3. **Impact**: blissful-tuner's approach naturally respects the timestep distribution (high-noise steps are rarer because they naturally occur less), while ai-toolkit's approach gives equal time to both stages regardless of their natural probability. Neither is strictly "correct" -- they represent different training philosophies. However, the high-noise expert in blissful-tuner will receive approximately 7x fewer training updates than the low-noise expert per epoch, which may lead to undertrained high-noise LoRA weights.

**Recommendation:** Consider adding an optional `--equal_expert_training` flag that alternates between high and low noise regions (similar to ai-toolkit's `switch_boundary_every`), ensuring both experts receive equal training updates. The current rejection sampling approach is valid but users should be aware of the imbalanced training distribution. Document the expected training imbalance in `docs/wan.md`.

---

## Finding AT-WAN-07: Dual-Model Memory: Weight Swap vs Two-Model Wrapper
**Severity:** INFO
**Domain:** 2: WAN 2.2 Dual-Stage Architecture

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/extensions_built_in/diffusion_models/wan22/wan22_14b_model.py:69-161`:

Uses a `DualWanTransformer3DModel` wrapper that holds both transformers as `nn.Module` submodules. Both exist in memory simultaneously (or one on CPU if `low_vram=True`). The wrapper routes forward calls based on timestep mean.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan_train_network.py:571-614` and `686-766`:

Uses a single `WanModel` instance with a `dit_inactive_state_dict` dictionary holding the other expert's weights. On expert switch, it does a full `load_state_dict` swap:
```python
state_dict = model.state_dict()
model.load_state_dict(self.dit_inactive_state_dict, strict=True, assign=True)
self.dit_inactive_state_dict = state_dict
```
Optionally offloads to CPU between swaps (`--offload_inactive_dit`).

**Discrepancy:** Different design, same effect. blissful-tuner's approach has higher swap overhead (full state dict copy) but lower peak memory (only one model's parameters on GPU at a time, plus one state dict on CPU). ai-toolkit holds both models on GPU by default, using more VRAM but with instant switching.

**Recommendation:** None -- this is a deliberate design choice. blissful-tuner's approach is better for memory-constrained setups.

---

## Finding AT-WAN-08: No Split LoRA Save in Blissful-Tuner
**Severity:** WARNING
**Domain:** 2: WAN 2.2 Dual-Stage Architecture

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/extensions_built_in/diffusion_models/wan22/wan22_14b_model.py:444-537`:

When `split_multistage_loras=True` (default), saves separate LoRA files:
- `model_high_noise.safetensors` with `.transformer_1.` removed from keys
- `model_low_noise.safetensors` with `.transformer_2.` removed from keys

On load, re-adds the transformer prefix to reconstruct the combined state dict. This allows users to apply different high/low noise LoRAs at inference.

**What blissful-tuner does:**
Blissful-tuner uses a single model and swaps state dicts, so all LoRA weights are created against the same `WanModel` module hierarchy. The LoRA is saved as a single file. There is no mechanism to split LoRA weights into separate high-noise and low-noise files.

Since both experts share the same module names (the WanModel is reused), the LoRA weights trained during high-noise steps and low-noise steps are interleaved across the same parameter set. This means:
1. A single LoRA file captures training from both experts
2. The LoRA cannot be applied selectively to just one expert at inference

**Discrepancy:** blissful-tuner's single-model weight-swap architecture means it trains a single LoRA that must work for both noise ranges, while ai-toolkit trains separate LoRAs per expert. This is a fundamental architectural difference.

**Recommendation:** This is a known design limitation of the weight-swap approach. To support split LoRA saving would require tracking which expert produced which gradient updates, which is complex with the current architecture. Document this limitation in `docs/wan.md` so users understand that WAN 2.2 LoRAs from blissful-tuner are unified (not split per expert), and clarify that this may affect quality compared to split-expert LoRAs.

---

## Finding AT-WAN-09: VAE Per-Channel Normalization Agreement
**Severity:** OK
**Domain:** 3: VAE Latent Normalization

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/models/wan21/wan21.py:644-652`:
```python
latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
latents = (latents - latents_mean) * latents_std
```

Mean values from `/Users/dustin/ai-toolkit/toolkit/models/wan21/autoencoder_kl_wan.py:977-1012`:
```
mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
std  = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
```

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan/modules/vae.py:664-702`:
```python
self.mean = torch.tensor(mean, ...)
self.std = torch.tensor(std, ...)
self.scale = [self.mean, 1.0 / self.std]
```

And the encoding at lines 531-569:
```python
def encode(self, x, scale):
    ...
    mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
```

The mean and std values at lines 664-699 are **identical** to ai-toolkit's.

**Discrepancy:** None -- both use the exact same per-channel mean/std normalization formula `(latent - mean) * (1/std)` with identical 16-channel statistics.

**Recommendation:** None.

---

## Finding AT-WAN-10: VAE Returns mu Only (No Reparameterize)
**Severity:** OK
**Domain:** 3: VAE Latent Normalization

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/models/wan21/wan21.py:642`:
```python
latents = self.vae.encode(images).latent_dist.sample()
```
Uses the diffusers-style VAE that samples from the distribution (mu + sigma * randn). This adds stochasticity via the reparameterization trick.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan/modules/vae.py:531-569`:
```python
def encode(self, x, scale):
    ...
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
    return mu  # returns mu only, no sampling
```

Blissful-tuner returns only the mean `mu`, not a sample from the distribution. This is the standard approach for caching latents (no randomness in the cache).

**Discrepancy:** Technically different (deterministic mu vs sampled), but this is the standard practice for latent caching pipelines. The stochasticity from VAE sampling is negligible compared to the diffusion noise, and caching deterministic mu is standard in Musubi/Kohya-style trainers. ai-toolkit also caches latents deterministically in its caching path.

**Recommendation:** None -- this is standard and correct.

---

## Finding AT-WAN-11: Training Shift Value Mismatch
**Severity:** WARNING
**Domain:** 4: Timestep Sampling & Shift

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/extensions_built_in/diffusion_models/wan22/wan22_5b_model.py:53-57`:
```python
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "use_dynamic_shifting": False,
}
```
Both WAN 2.2 5B and 14B use `shift=5.0` for training (the 14B model imports this same config at `/Users/dustin/ai-toolkit/extensions_built_in/diffusion_models/wan22/wan22_14b_model.py:28`).

WAN 2.1 base uses `shift=3.0` (`/Users/dustin/ai-toolkit/toolkit/models/wan21/wan21.py:82`).

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:3071-3073`:
```python
parser.add_argument("--discrete_flow_shift", type=float, default=1.0, ...)
```
Default shift is **1.0** (no shift). The WAN 2.2 recommended values are in `wan_t2v_A14B.py:39`:
```python
t2v_A14B.sample_shift = 12.0  # for INFERENCE, not training
```

The `wan_train_network.py:62-92` warning system alerts users about suboptimal defaults:
```python
if timestep_sampling == "sigma":
    rec = f" (recommended: --timestep_sampling shift --discrete_flow_shift {recommended_flow_shift})"
```
Where `recommended_flow_shift = 12.0` for T2V and `5.0` for I2V.

**Discrepancy:** Significant difference in default behavior.

1. **ai-toolkit** bakes `shift=5.0` into the scheduler for both WAN 2.2 variants. Users don't need to specify it.
2. **blissful-tuner** defaults to `shift=1.0` (effectively no shift) and requires users to explicitly set `--timestep_sampling shift --discrete_flow_shift 12.0` for WAN 2.2 T2V. The default `--timestep_sampling sigma` path does not use the shift at all.
3. **The recommended values differ**: ai-toolkit uses 5.0 for training; blissful-tuner recommends 12.0 for T2V. The 12.0 value matches the official Alibaba inference shift (`sample_shift = 12.0`), but the ai-toolkit author chose 5.0 for training (perhaps because the WAN 2.2 paper or early experiments suggested a lower training shift). The optimal training shift is not necessarily the same as the inference shift.

**Recommendation:** Consider either:
(a) Auto-detecting the task and setting an appropriate default shift when `--timestep_sampling shift` is selected with a WAN 2.2 task, or
(b) Documenting more prominently that `--discrete_flow_shift 12.0` (or 5.0) should be used for WAN 2.2 training, not the default 1.0.

The discrepancy between 5.0 (ai-toolkit) and 12.0 (blissful-tuner recommendation) deserves investigation -- ideally benchmarking both values. The original Wan team uses shift=12 for T2V inference.

---

## Finding AT-WAN-12: Shift Formula Agreement
**Severity:** OK
**Domain:** 4: Timestep Sampling & Shift

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/samplers/custom_flowmatch_sampler.py:161`:
```python
sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
```

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/modules/scheduling_flow_match_discrete.py:181`:
```python
def sd3_time_shift(self, t: torch.Tensor):
    return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)
```

And in the training timestep sampling at `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1031`:
```python
t = (t * shift) / (1 + (shift - 1) * t)
```

**Discrepancy:** None -- the shift formula is identical in all three locations: `sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)`. This is the standard SD3/flow matching shift formula.

**Recommendation:** None.

---

## Finding AT-WAN-13: Blissful-Tuner Has Richer Timestep Sampling Strategies
**Severity:** INFO
**Domain:** 4: Timestep Sampling & Shift

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/samplers/custom_flowmatch_sampler.py:107-219`:
Supports: `linear`, `sigmoid`, `shift` (flux_shift/lumina2_shift), `lognorm_blend`, `weighted`

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:974-1090`:
Supports: `sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift`, `flux2_shift`, `qwen_shift`, `logsnr`, `qinglong_flux`, `qinglong_qwen`

**Discrepancy:** Blissful-tuner has more sampling strategies (logsnr, qinglong hybrid, architecture-specific shifts). ai-toolkit has `lognorm_blend` and `weighted` which blissful-tuner lacks. Both have the critical ones (sigmoid, shift) needed for WAN 2.2.

**Recommendation:** None -- both have adequate coverage for WAN 2.2. The `logsnr` strategy in blissful-tuner follows the recent LDM literature (arXiv:2411.14793v3) and could be beneficial for WAN training.

---

## Finding AT-WAN-14: Loss Computation Agreement
**Severity:** OK
**Domain:** 5: Loss Computation

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py:769`:
```python
loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
```
Where `target = (noise - batch.latents).detach()` (from `get_loss_target`).

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:2491`:
```python
loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
```
Where `target = noise - latents` (from `wan_train_network.py:851`).

**Discrepancy:** None -- both compute unreduced MSE between model prediction and velocity target (noise - latents). The only minor difference is that ai-toolkit casts to `.float()` (fp32) for the loss computation, while blissful-tuner uses `network_dtype` which could be bf16. Computing loss in bf16 is slightly less numerically stable but is standard practice in mixed-precision training.

**Recommendation:** None -- the numerical difference is negligible in practice.

---

## Finding AT-WAN-15: Blissful-Tuner Has Structured Mask Loss
**Severity:** INFO
**Domain:** 5: Loss Computation

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py:802-812`:
```python
if len(noise_pred.shape) == 5:
    mask_multiplier = mask_multiplier.unsqueeze(2)
    mask_multiplier = mask_multiplier.repeat(1, 1, noise_pred.shape[2], 1, 1)
loss = loss * mask_multiplier
```
With optional `inverted_mask_prior` for prior preservation (`lines 535-558`). The mask is a simple multiplier resized to latent spatial dimensions.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:2510-2517`:
```python
loss = apply_masked_loss_with_prior(
    loss, mask_weights, prior_loss_unreduced=prior_loss_unreduced,
    args=args, layout=layout, drop_base_frame=drop_base_frame,
)
```
Uses a dedicated mask loss module (`/Users/dustin/blissful-tuner/src/musubi_tuner/modules/mask_loss.py`) with:
- Gamma correction (`--mask_gamma`)
- Minimum weight floor (`--mask_min_weight`)
- Weighted-mean normalization
- Prior preservation with LoRA-disabled teacher forward pass
- Per-sample normalization
- Threshold-based prior masks

**Discrepancy:** Blissful-tuner's mask loss system is significantly more sophisticated. ai-toolkit's approach is a simple multiplicative mask; blissful-tuner adds gamma correction, min-weight floors, proper weighted-mean normalization, and structured prior preservation.

**Recommendation:** None -- this is a blissful-tuner advantage. The structured mask loss with prior preservation is well-documented in `docs/MASKED_LOSS_TRAINING_GUIDE.md`.

---

## Finding AT-WAN-16: ai-toolkit Loss Weighting Features Not in Blissful-Tuner
**Severity:** INFO
**Domain:** 5: Loss Computation

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py:850-860`:
- **Min-SNR gamma**: `apply_snr_weight(loss, timesteps, scheduler, gamma)` (train_tools.py)
- **Learnable SNR**: `apply_learnable_snr_gos(loss, timesteps, self.snr_gos)` (train_tools.py)
- **Bell-shaped timestep weighting (BSMNTW)**: Gaussian-weighted loss favoring mid-range timesteps
- **Wavelet loss**: Frequency-domain loss (`loss_type="wavelet"`)
- **Pixel-space loss**: Decode-then-compare approach

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:2438-2440`:
```python
weighting = compute_loss_weighting_for_sd3(
    args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype, n_dim=latents.ndim
)
```
Supports: `sigma_sqrt`, `cosmap`, `structure_bell` weighting schemes (SD3-style). Does not have min-SNR, learnable SNR, or wavelet/pixelspace losses.

**Discrepancy:** ai-toolkit has more loss weighting options. However, for standard WAN 2.2 training, most users use unweighted MSE, making these advanced options rarely needed.

**Recommendation:** Low priority. Min-SNR gamma could be worth adding for WAN training as it's well-studied in the literature, but it's not critical.

---

## Finding AT-WAN-17: UMT5 Text Encoder: Same Model, Different Wrappers
**Severity:** OK
**Domain:** 6: Text Encoding - UMT5

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/models/wan21/wan21.py:421`:
```python
tokenizer, text_encoder = get_umt5_encoder(model_path=te_path, ...)
```
Uses diffusers' `UMT5EncoderModel` loaded via HuggingFace format, with a standard `AutoTokenizer`.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan/modules/t5.py:457-528`:
Uses a custom `T5EncoderModel` class wrapping a `umt5_xxl(encoder_only=True)` model loaded from a raw `.pth` or `.safetensors` weight file. The tokenizer is `HuggingfaceTokenizer(name="google/umt5-xxl", seq_len=text_len)`.

**Discrepancy:** Different loading mechanisms (HuggingFace format vs raw weights), but the underlying model architecture is the same UMT5-XXL encoder. Both produce the same embeddings given the same input.

**Recommendation:** None.

---

## Finding AT-WAN-18: Text Length Agreement (512)
**Severity:** OK
**Domain:** 6: Text Encoding - UMT5

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/models/wan21/wan21.py:593`:
```python
prompt_embeds, _ = self.pipeline.encode_prompt(prompt, ..., max_sequence_length=512, ...)
```

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan/configs/shared_config.py:11`:
```python
wan_shared_cfg.text_len = 512
```
And `/Users/dustin/blissful-tuner/src/musubi_tuner/wan/modules/t5.py:478`:
```python
self.text_len = text_len  # 512
```

**Discrepancy:** None -- both use 512 max sequence length.

**Recommendation:** None.

---

## Finding AT-WAN-19: Variable-Length T5 Embeddings Agreement
**Severity:** OK
**Domain:** 6: Text Encoding - UMT5

**What ai-toolkit does:**
The diffusers pipeline produces fixed-length embeddings with attention masks. During training, the full 512-length tensor is passed to the model.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/wan/modules/t5.py:522-528`:
```python
def __call__(self, texts, device):
    ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = self.model(ids, mask)
    return [u[:v] for u, v in zip(context, seq_lens)]
```
Returns variable-length embeddings (truncated to actual token count). These are then padded to `text_len` in the WanModel forward pass (`model.py:1093`):
```python
context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
```

**Discrepancy:** Functionally equivalent. blissful-tuner stores variable-length embeddings in the cache (saving disk space), then pads to 512 at training time. ai-toolkit stores full 512-length embeddings. The model sees the same padded tensors.

**Recommendation:** None.

---

## Finding AT-WAN-20: LoRA Targeting Scope Difference
**Severity:** WARNING
**Domain:** 7: LoRA Targeting

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/toolkit/lora_special.py:338-402`:

Targets all modules whose parent class name is in `target_lora_modules`:
- WAN 2.1: `["WanTransformer3DModel"]`
- WAN 2.2 (both stages): `["DualWanTransformer3DModel"]`
- WAN 2.2 (single stage): `["WanTransformer3DModel"]`

Within those module classes, it targets **all `nn.Linear` layers** (and optionally Conv2d). This includes:
- Attention QKV, output projections
- FFN layers
- **Embedding projections** (condition_embedder, etc.)
- Any other Linear layer in the transformer

Filtering: The `transformer_only` flag + `get_transformer_block_names()` returning `['blocks']` means only layers inside `blocks.*` are targeted, excluding top-level projections. But this depends on whether `transformer_only` is enabled.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/networks/lora_wan.py:13-34`:
```python
WAN_TARGET_REPLACE_MODULES = ["WanAttentionBlock"]
```
With explicit exclusion pattern:
```python
exclude_patterns.append(r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*")
```

Only targets Linear layers **inside `WanAttentionBlock`** instances. This includes:
- Self-attention QKV (`self_attn.q/k/v`), output projection (`self_attn.o`)
- Cross-attention QKV, output projection
- FFN layers (two Linear layers)
- **Excludes**: modulation parameters, normalization layers, embedding projections, the final head

**Discrepancy:** The scope differs significantly:

1. **ai-toolkit** (without `transformer_only`): Targets all Linear layers in the entire transformer, potentially including condition embedder projections, time embeddings, etc.
2. **ai-toolkit** (with `transformer_only` + `blocks` filter): Similar scope to blissful-tuner but might include additional Linear layers not in WanAttentionBlock if any exist.
3. **blissful-tuner**: Precisely targets WanAttentionBlock only, with explicit norm/embedding exclusions.

The key practical difference: blissful-tuner's approach is more conservative and avoids LoRA on embeddings/projections outside the attention blocks. This is the standard approach for WAN LoRA (matching the original Musubi Tuner design) and is generally recommended to avoid instability.

**Recommendation:** None urgently needed -- blissful-tuner's conservative targeting is the established best practice for WAN LoRA. The broader targeting in ai-toolkit could potentially capture more model capacity but at higher risk of training instability. Users of ai-toolkit who want blissful-tuner-compatible LoRAs should ensure consistent targeting.

---

## Finding AT-WAN-21: No Noise Augmentation Stack in Blissful-Tuner
**Severity:** INFO
**Domain:** 8: Noise Augmentation

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/jobs/process/BaseSDTrainProcess.py:1281-1332`:

5-stage noise augmentation pipeline:
1. **Dynamic noise offset** (line 1288): `noise += latents.mean(dim=(2,3), keepdim=True) / 2`
2. **Noise multiplier** (line 1307): `noise *= noise_multiplier`
3. **Signal correction noise** (line 1309-1317): `noise += latents * randn(B,C,1,1) * scale`
4. **Random noise shift** (line 1319-1327): `noise += randn(B,C,1,1) * shift`
5. **Random noise multiplier** (line 1329-1331): `noise *= exp(randn(s) * sigma)`

All are optional and disabled by default except noise_multiplier (default 1.0, no-op).

**What blissful-tuner does:**
No noise augmentation pipeline exists. The noise is simply `torch.randn_like(latents)` at line 2431 of `hv_train_network.py`, with no modifications before use.

**Discrepancy:** blissful-tuner has no noise augmentation capabilities. This is not necessarily a bug -- most WAN training does not use these features, and they are all optional in ai-toolkit.

**Recommendation:** Low priority. These features are experimental in ai-toolkit and not widely used for WAN training. If desired, signal correction noise (Finding AT-WAN-22) is the most interesting feature to consider porting.

---

## Finding AT-WAN-22: Signal Correction Noise Absent from Blissful-Tuner
**Severity:** INFO
**Domain:** 8: Noise Augmentation

**What ai-toolkit does:**
`/Users/dustin/ai-toolkit/jobs/process/BaseSDTrainProcess.py:1309-1317`:
```python
if self.train_config.do_signal_correction_noise:
    batch_noise = latents.clone().to(noise.device, dtype=noise.dtype)
    scn_scale = torch.randn(
        batch_noise.shape[0], batch_noise.shape[1], 1, 1,
        device=batch_noise.device, dtype=batch_noise.dtype
    ) * self.train_config.signal_correction_noise_scale
    batch_noise = batch_noise * scn_scale
    noise = noise + batch_noise
```
Math: `noise_augmented = noise + latents * N(0, scale)` per-channel

Note: the `scn_scale` shape is `(B, C, 1, 1)` which is 4D, but WAN latents are 5D `(B, C, T, H, W)`. This would broadcast correctly but applies the same per-channel scale across all temporal frames, which may or may not be intentional for video.

**What blissful-tuner does:**
No signal correction noise implementation.

**Discrepancy:** This is a novel technique in ai-toolkit that adds a scaled version of the clean latent to the noise, effectively creating signal-correlated noise. The theory is that this helps the model learn to better separate signal from noise.

**Recommendation:** Could be worth investigating as an optional training feature (`--signal_correction_noise_scale`). However, this is experimental and its effectiveness for WAN specifically has not been widely validated. Low priority.

---

## Overall Assessment

The two codebases implement the WAN 2.1/2.2 training pipeline with the same core math:
- Flow matching formula: `x_t = (1-t)*x_0 + t*noise` -- **AGREE**
- Loss target: `v = noise - latents` -- **AGREE**
- VAE normalization: `(latent - mean) * (1/std)` with identical statistics -- **AGREE**
- Shift formula: `sigma_shifted = shift*sigma / (1 + (shift-1)*sigma)` -- **AGREE**

The key **differences** worth attention:

1. **AT-WAN-06 (WARNING)**: Stage switching strategy differs fundamentally. ai-toolkit uses deterministic round-robin; blissful-tuner uses probability-weighted rejection sampling. This leads to ~7x fewer high-noise training steps in blissful-tuner. Consider adding an equal-training mode.

2. **AT-WAN-08 (WARNING)**: No split LoRA save capability in blissful-tuner for WAN 2.2 dual-expert models. The single-model weight-swap architecture precludes separate high/low noise LoRA files.

3. **AT-WAN-11 (WARNING)**: Default training shift differs (ai-toolkit: 5.0 baked in; blissful-tuner: user must specify, recommends 12.0 for T2V). The optimal training shift value needs benchmarking.

4. **AT-WAN-20 (WARNING)**: LoRA targeting scope is narrower in blissful-tuner (WanAttentionBlock only vs potentially all Linear layers). This is the conservative and standard approach.

No **CRITICAL** bugs were found. The core training mathematics are sound and agree between both implementations.
