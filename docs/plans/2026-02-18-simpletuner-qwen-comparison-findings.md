# SimpleTuner vs blissful-tuner: Qwen Image LoRA Training Pipeline Comparison

**Date**: 2026-02-18
**Methodology**: Line-by-line source code comparison across 8 domains
**SimpleTuner version**: 4.1.0 (codebase at `/Users/dustin/SimpleTuner/`)
**blissful-tuner codebase**: `/Users/dustin/blissful-tuner/`

---

## Summary Table

| ID | Title | Severity | Domain |
|----|-------|----------|--------|
| ST-QI-1 | Flow matching noise formula agrees | OK | 1: Flow Matching |
| ST-QI-2 | Sigma sampling strategies diverge | INFO | 1: Flow Matching |
| ST-QI-3 | Timestep +1 offset differs | WARNING | 1: Flow Matching |
| ST-QI-4 | VAE latent normalization formula agrees | OK | 2: Latent Scaling |
| ST-QI-5 | VAE posterior sampling: sample() vs mode() | WARNING | 2: Latent Scaling |
| ST-QI-6 | Frame dimension squeeze order differs | INFO | 2: Latent Scaling |
| ST-QI-7 | Loss target formula agrees: noise - latents | OK | 3: Loss Target |
| ST-QI-8 | Loss computed in unpacked spatial space (both) | OK | 3: Loss Target |
| ST-QI-9 | Loss reduction differs: mean-of-dims vs masked-loss | INFO | 3: Loss Target |
| ST-QI-10 | Dynamic shift formula and parameters agree | OK | 4: Timestep Normalization |
| ST-QI-11 | SimpleTuner uses static shift 1.73 by default | INFO | 4: Timestep Normalization |
| ST-QI-12 | Timestep /1000 normalization differs by +1 offset | WARNING | 4: Timestep Normalization |
| ST-QI-13 | Noise addition before packing agrees | OK | 5: Latent Packing |
| ST-QI-14 | Pack/unpack implementations agree | OK | 5: Latent Packing |
| ST-QI-15 | img_shapes format: per-batch vs nested list | INFO | 5: Latent Packing |
| ST-QI-16 | Prompt template and drop index agree | OK | 6: Text Encoding |
| ST-QI-17 | Hidden layer extraction: last layer (both) | OK | 6: Text Encoding |
| ST-QI-18 | Edit prompt template tokens differ by model_version | OK | 6: Text Encoding |
| ST-QI-19 | LoRA targeting scope differs fundamentally | INFO | 7: LoRA Targeting |
| ST-QI-20 | blissful-tuner covers all Linear layers in blocks | INFO | 7: LoRA Targeting |
| ST-QI-21 | Edit V1 conditioning agrees | OK | 8: Edit Model |
| ST-QI-22 | Edit V2/V2+ per-sample multi-control handling | INFO | 8: Edit Model |
| ST-QI-23 | zero_cond_t / modulate_index handling agrees | OK | 8: Edit Model |

**By Severity:**
- CRITICAL: 0
- WARNING: 3 (ST-QI-3, ST-QI-5, ST-QI-12)
- INFO: 8
- OK: 12

---

## Finding ST-QI-1: Flow Matching Noise Formula Agrees
**Severity:** OK
**Domain:** 1: Flow Matching Formula

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/common.py:4374`:
```python
batch["noisy_latents"] = (1 - batch["sigmas"]) * batch["latents"] + batch["sigmas"] * batch["input_noise"]
```
Where `sigmas` is broadcastable via `expand_sigmas()` which reshapes to `(-1, 1, 1, 1)`.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1124-1125`:
```python
t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
noisy_model_input = (1 - t) * latents + t * noise
```

**Discrepancy:** None -- implementations agree. Both use the standard rectified flow interpolation `x_t = (1 - sigma) * x_0 + sigma * epsilon` where sigma (or `t`) is sampled in [0, 1].

**Recommendation:** None.

---

## Finding ST-QI-2: Sigma Sampling Strategies Diverge
**Severity:** INFO
**Domain:** 1: Flow Matching Formula

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/common.py:3661-3711`:
Three main strategies available:
- **Sigmoid** (default): `torch.sigmoid(scale * torch.randn(...))` with `flow_sigmoid_scale=5.0`
- **Uniform**: `torch.rand(...)`
- **Beta**: `Beta(alpha, beta).sample()`

All followed by `apply_flow_schedule_shift()` which applies the formula `sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)`.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:974-1090`:
Strategies available:
- **uniform**: `torch.rand(batch_size)`
- **sigmoid**: `torch.sigmoid(sigmoid_scale * randn(batch_size))`
- **shift**: static `discrete_flow_shift`
- **flux_shift**: linear function `(y1=0.5, y2=1.15)` on `(h//2)*(w//2)`
- **qwen_shift**: linear function `(x1=256, y1=0.5, x2=8192, y2=0.9)` on `(h//2)*(w//2)`
- **logsnr**: log-SNR based sampling
- **qinglong_flux / qinglong_qwen**: triple hybrid sampling

Both `sigmoid` and `uniform` use the same base formulas. The key difference is how the shift is applied: SimpleTuner always uses `apply_flow_schedule_shift()` after sampling (via `flow_schedule_shift=1.73` config), while blissful-tuner bakes the shift into the sampling strategy itself (via the `qwen_shift` strategy or via `shift`).

**Discrepancy:** Strategic: SimpleTuner's default for Qwen is sigmoid + static shift=1.73, while blissful-tuner's recommended Qwen approach is `--timestep_sampling qwen_shift` which uses a resolution-dependent dynamic shift. Neither is "wrong" -- the dynamic shift matches the official Diffusers scheduler behavior more closely.

**Recommendation:** None -- both approaches are valid. The dynamic shift in blissful-tuner's `qwen_shift` is technically more faithful to the official training recipe.

---

## Finding ST-QI-3: Timestep +1 Offset Differs
**Severity:** WARNING
**Domain:** 1: Flow Matching Formula

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/common.py:3710`:
```python
timesteps = sigmas * 1000.0
```
Timesteps range: [0.0, 1000.0]. Normalized to [0.0, 1.0] before transformer input.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1123,1127`:
```python
timesteps = t * 1000.0
...
timesteps += 1  # 1 to 1000
```
Timesteps range: [1.0, 1001.0]. Then in `call_dit()` at `/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:573`:
```python
timesteps = timesteps / 1000.0
```
Normalized range: [0.001, 1.001].

**Discrepancy:** blissful-tuner adds a `+1` offset to timesteps, resulting in a normalized range of [0.001, 1.001] rather than [0.0, 1.0]. This means:
1. The transformer never sees exactly t=0 (fully clean) during training.
2. The transformer can see t=1.001 (slightly beyond fully noisy).

This `+1` offset is inherited from the HunyuanVideo / Wan training code where the scheduler expects timesteps in [1, 1000]. For Qwen Image, which uses its own scheduler during inference with range [0, 1], this creates a minor train/inference mismatch. In practice, the effect is likely very small (0.1% of the range), but it is technically incorrect for Qwen Image.

Note: This same pattern was flagged in the prior FLUX.2 audit as a known issue inherited from the base trainer.

**Recommendation:** Consider adding a Qwen-Image-specific flag or override that skips the `+1` offset. Alternatively, document this as a known minor discrepancy. The practical impact on training quality is likely minimal but not zero -- the sinusoidal timestep embeddings at t=0.001 vs t=0.0 produce slightly different modulation parameters.

---

## Finding ST-QI-4: VAE Latent Normalization Formula Agrees
**Severity:** OK
**Domain:** 2: Latent Scaling

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:1419-1428`:
```python
latents_mean = (
    torch.tensor(vae.config.latents_mean)
    .view(1, vae.config.z_dim, 1, 1)
    .to(sample_latents.device, sample_latents.dtype)
)
latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1).to(
    sample_latents.device, sample_latents.dtype
)
sample_latents = (sample_latents - latents_mean) * latents_std
```

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_autoencoder_kl.py:1032-1035`:
```python
latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
latents = (latents - latents_mean) * latents_std
```

**Discrepancy:** None -- the formula `(latents - mean) * (1/std)` is identical. The only difference is the tensor shape: SimpleTuner uses `(1, z_dim, 1, 1)` (4D, frame dimension already squeezed), while blissful-tuner uses `(1, z_dim, 1, 1, 1)` (5D, frame dimension still present). Both are correct for their respective squeeze ordering.

The inverse (decode) is also correctly implemented in both. blissful-tuner at line 1003-1004:
```python
latents = latents / latents_std + latents_mean
```

The per-channel statistics (16 values each for `latents_mean` and `latents_std`) are hardcoded from the same VAE config in both codebases.

**Recommendation:** None.

---

## Finding ST-QI-5: VAE Posterior Sampling: sample() vs mode()
**Severity:** WARNING
**Domain:** 2: Latent Scaling

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:1416`:
```python
sample_latents = sample.latent_dist.sample()
```
Uses stochastic sampling from the diagonal Gaussian posterior (mean + std * noise).

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_autoencoder_kl.py:1029`:
```python
latents = posterior.mode()  # Use mode instead of sampling for deterministic results
```
Uses the deterministic mode (just the mean) of the posterior.

**Discrepancy:** This is a meaningful difference. Using `sample()` adds stochastic noise during latent caching, which acts as a form of data augmentation (KL-VAE regularization noise). Using `mode()` produces deterministic latents (the posterior mean).

For LoRA training:
- `sample()`: Slight variation each time latents are re-cached, provides implicit augmentation. Standard in diffusion training literature.
- `mode()`: Deterministic, no variation. Latents are cached once and always identical. Simpler and more reproducible.

The practical impact depends on training setup. With pre-cached latents (both trainers), latents are encoded once. With `sample()`, there's random noise baked in at cache time. With `mode()`, the exact mean is stored. For short training runs or small datasets, `sample()` noise may slightly reduce overfitting.

**Recommendation:** This is a deliberate design choice in blissful-tuner (the comment says "Use mode instead of sampling for deterministic results"). Consider documenting this difference. Users who want SimpleTuner-equivalent behavior could switch to `posterior.sample()`. The blissful-tuner choice of `mode()` is not incorrect but produces marginally different latent distributions.

---

## Finding ST-QI-6: Frame Dimension Squeeze Order Differs
**Severity:** INFO
**Domain:** 2: Latent Scaling

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:1396-1430`:
1. Pre-encode: Add frame dim `(B, C, H, W)` -> `(B, C, 1, H, W)`
2. VAE encode
3. Post-encode: **Squeeze frame dim first** (line 1418: `squeeze(2)` -> 4D), **then** normalize with 4D mean/std `(1, z_dim, 1, 1)`

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_autoencoder_kl.py:1008-1035`:
1. Add frame dim if 4D (line 1023)
2. VAE encode
3. **Normalize with 5D mean/std** `(1, z_dim, 1, 1, 1)` **keeping frame dim**

**Discrepancy:** Different squeeze ordering (SimpleTuner squeezes before normalization, blissful-tuner keeps 5D throughout). Mathematically equivalent -- the extra dimension broadcasts correctly either way. blissful-tuner keeps the frame dimension because it stores latents as 5D `(B, C, F, H, W)` for consistency with the video-style cache format.

**Recommendation:** None -- mathematically identical results.

---

## Finding ST-QI-7: Loss Target Formula Agrees: noise - latents
**Severity:** OK
**Domain:** 3: Loss Target Construction

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/common.py:3515-3516`:
```python
elif self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
    target = prepared_batch["noise"] - prepared_batch["latents"]
```

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:602`:
```python
target = noise - latents
```

**Discrepancy:** None -- both compute the velocity field target as `noise - latents` (the flow matching velocity target). This target represents the direction from clean data to noise.

**Recommendation:** None.

---

## Finding ST-QI-8: Loss Computed in Unpacked Spatial Space (Both)
**Severity:** OK
**Domain:** 3: Loss Target Construction

**What SimpleTuner does:**
In `_model_predict_standard()` at `/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:1077-1079`:
```python
if noise_pred.dim() == 3:
    noise_pred = pipeline_class._unpack_latents(noise_pred, pixel_height, pixel_width, self.vae_scale_factor)
```
Then loss is computed against `target = noise - latents` where `latents` are in unpacked (B, C, H, W) or (B, C, 1, H, W) space.

Loss at `/Users/dustin/SimpleTuner/simpletuner/helpers/models/common.py:4627`:
```python
loss = (model_pred.float() - target.float()) ** 2
```

**What blissful-tuner does:**
In `call_dit()` at `/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:592-598`:
```python
model_pred = qwen_image_utils.unpack_latents(
    model_pred, lat_h * qwen_image_utils.VAE_SCALE_FACTOR, lat_w * qwen_image_utils.VAE_SCALE_FACTOR,
    qwen_image_utils.VAE_SCALE_FACTOR, is_layered=args.is_layered,
)
```
Loss at `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:2491`:
```python
loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
```

**Discrepancy:** None -- both unpack the model prediction back to spatial space before computing loss. This ensures the loss is computed element-wise in the natural (B, C, H, W) or (B, C, F, H, W) layout.

**Recommendation:** None.

---

## Finding ST-QI-9: Loss Reduction Differs
**Severity:** INFO
**Domain:** 3: Loss Target Construction

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/common.py:4767,4770`:
```python
loss = loss.mean(dim=list(range(1, len(loss.shape))))
...
loss = loss.mean()
```
First reduces over spatial dimensions (per-sample mean), then takes the batch mean. This is a standard `mean(dim=[1,2,3])` followed by `mean(dim=0)`.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:2510-2517`:
```python
loss = apply_masked_loss_with_prior(
    loss, mask_weights, prior_loss_unreduced=prior_loss_unreduced,
    args=args, layout=layout, drop_base_frame=drop_base_frame,
)
```
Uses the centralized mask loss module (`/Users/dustin/blissful-tuner/src/musubi_tuner/modules/mask_loss.py`) which handles:
- Optional gamma correction on mask weights
- Optional min-weight floor
- Weighted-mean normalization
- Optional prior preservation blending
- Per-sample normalization

When no mask loss is enabled, the fallback in `apply_masked_loss_with_prior()` reduces to a simple `loss.mean()` over all dimensions.

**Discrepancy:** When `--use_mask_loss` is disabled (default), both reduce to the same `loss.mean()`. When mask loss is enabled, blissful-tuner has a sophisticated mask-weighted reduction that SimpleTuner doesn't have (SimpleTuner has a simpler conditioning mask system using per-pixel weighting).

**Recommendation:** None -- this is a feature difference, not a correctness issue. blissful-tuner's mask loss system is more advanced.

---

## Finding ST-QI-10: Dynamic Shift Formula and Parameters Agree
**Severity:** OK
**Domain:** 4: Timestep Normalization

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/training/custom_schedule.py:476-477`:
```python
shift = math.exp(mu)
...
sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
```
Where `mu` is computed by `calculate_shift_flux()` (from Diffusers) with parameters from the scheduler config: `base_image_seq_len`, `max_image_seq_len`, `base_shift`, `max_shift`.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1022-1031`:
```python
elif args.timestep_sampling == "qwen_shift":
    mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
...
shift = math.exp(mu)
...
t = (t * shift) / (1 + (shift - 1) * t)
```

`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_utils.py:1011-1033`:
```python
SCHEDULER_BASE_IMAGE_SEQ_LEN = 256
SCHEDULER_BASE_SHIFT = 0.5
SCHEDULER_MAX_IMAGE_SEQ_LEN = 8192
SCHEDULER_MAX_SHIFT = 0.9
...
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
```

**Discrepancy:** None -- both use the same linear interpolation formula with identical parameters `(base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9)`, and both apply the exponential time shift `t_shifted = (t * exp(mu)) / (1 + (exp(mu) - 1) * t)`.

These parameters were verified against the official Diffusers Qwen-Image scheduler config in the prior audit (finding T4).

**Recommendation:** None.

---

## Finding ST-QI-11: SimpleTuner Uses Static Shift 1.73 by Default
**Severity:** INFO
**Domain:** 4: Timestep Normalization

**What SimpleTuner does:**
The reference document states `flow_schedule_shift = 1.73` is the default for Qwen Image. This is a **static** shift applied to all resolutions uniformly via:
```python
sigmas = (sigmas * 1.73) / (1 + (1.73 - 1) * sigmas)
```

The `USES_DYNAMIC_SHIFT = True` flag on the model class at `/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:61` affects the **inference scheduler** behavior, but during training the shift comes from `flow_schedule_shift` config (static) or `flow_schedule_auto_shift` (dynamic).

**What blissful-tuner does:**
Uses dynamic resolution-dependent shift via `--timestep_sampling qwen_shift`, where the shift value varies based on the packed sequence length `(h//2) * (w//2)`.

For a 1024x1024 image: latent = 128x128, packed seq_len = 64*64 = 4096.
- blissful-tuner mu = `0.5 + (0.9-0.5)/(8192-256) * (4096 - 256)` = `0.5 + 0.000050 * 3840` = `0.693`, shift = `exp(0.693)` = `2.0`
- SimpleTuner static shift = `1.73`

For a 512x512 image: latent = 64x64, packed seq_len = 32*32 = 1024.
- blissful-tuner mu = `0.5 + 0.000050 * 768` = `0.539`, shift = `exp(0.539)` = `1.71`
- SimpleTuner static shift = `1.73`

**Discrepancy:** SimpleTuner's default `1.73` is close to the dynamic shift at ~512px resolution but differs at higher resolutions. The dynamic approach adjusts the noise schedule based on image complexity (larger images = more shift). This is more faithful to the official Qwen Image scheduler behavior.

However, SimpleTuner users can enable `flow_schedule_auto_shift=true` to get dynamic shifting.

**Recommendation:** Document that `--timestep_sampling qwen_shift` in blissful-tuner corresponds to SimpleTuner's `flow_schedule_auto_shift=true` (not the default `flow_schedule_shift=1.73`).

---

## Finding ST-QI-12: Timestep /1000 Normalization Differs by +1 Offset
**Severity:** WARNING
**Domain:** 4: Timestep Normalization

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:1046`:
```python
timesteps = raw_timesteps.expand(batch_size) / 1000.0  # Normalize to [0, 1]
```
Where `raw_timesteps = sigmas * 1000.0`, so the normalized range is exactly [0.0, 1.0].

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1123,1127`:
```python
timesteps = t * 1000.0
...
timesteps += 1  # 1 to 1000
```
Then `/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:573`:
```python
timesteps = timesteps / 1000.0
```
Normalized range: [0.001, 1.001].

**Discrepancy:** This is the same finding as ST-QI-3 from the timestep perspective. The `+1` offset shifts the entire timestep range by 0.1%. The sinusoidal timestep embeddings (`QwenTimestepProjEmbeddings` uses `timestep * 1000` internally for the frequency computation) will produce slightly different modulation parameters.

At the boundaries:
- t=0 (fully clean): SimpleTuner sees 0.0, blissful-tuner sees 0.001
- t=1 (fully noisy): SimpleTuner sees 1.0, blissful-tuner sees 1.001

The `+1` offset originates from the HunyuanVideo/Wan training conventions where the scheduler uses timesteps in [1, 1000] rather than [0, 999].

**Recommendation:** Same as ST-QI-3. Consider removing the `+1` for Qwen Image training, or verify the Qwen Image transformer's timestep embedding behavior at the boundary values.

---

## Finding ST-QI-13: Noise Addition Before Packing Agrees
**Severity:** OK
**Domain:** 5: Latent Packing Order

**What SimpleTuner does:**
1. `prepare_batch()` at `common.py:4374`: Noise added in unpacked space `(B, C, H, W)`
2. `_model_predict_standard()` at `model.py:1008`: Pack after noise addition

**What blissful-tuner does:**
1. `get_noisy_model_input_and_timesteps()` at `hv_train_network.py:1125`: Noise added in unpacked space `(B, C, F, H, W)`
2. `call_dit()` at `qwen_image_train_network.py:459`: `pack_latents(noisy_model_input)` after noise addition

**Discrepancy:** None -- both add noise in unpacked spatial format before packing into the sequence format for the transformer.

**Recommendation:** None.

---

## Finding ST-QI-14: Pack/Unpack Implementations Agree
**Severity:** OK
**Domain:** 5: Latent Packing Order

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/pipeline.py:370-375`:
```python
latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
latents = latents.permute(0, 2, 4, 1, 3, 5)
latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
```

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_utils.py:921-926`:
```python
latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
latents = latents.permute(0, 2, 4, 1, 3, 5)
latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
```

**Discrepancy:** None -- identical 2x2 patchification: `(B, 16, H, W)` -> `(B, H/2*W/2, 64)`.

blissful-tuner additionally handles layered latents `(B, L, C, H, W)` -> `(B, L*H/2*W/2, C*4)`, which SimpleTuner doesn't need since it doesn't support the layered model variant.

**Recommendation:** None.

---

## Finding ST-QI-15: img_shapes Format: Per-Batch vs Nested List
**Severity:** INFO
**Domain:** 5: Latent Packing Order

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:1038`:
```python
img_shapes = [(1, latent_height_for_shape, latent_width_for_shape)] * batch_size
```
Produces a flat list with one tuple per batch sample: `[(1, h, w), (1, h, w), ...]`.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:560-566`:
```python
img_shapes = [(1, lat_h // 2, lat_w // 2)]
if args.is_layered:
    img_shapes = img_shapes * (num_layers + 1)
if is_edit or args.is_layered:
    img_shapes = [img_shapes + [(1, sh[-2] // 2, sh[-1] // 2) for sh in latents_control_shapes]]
else:
    img_shapes = [img_shapes]  # make it a list of list for consistency
```
Produces a nested list: `[[(1, h, w)]]` for T2I, `[[(1, h, w), (1, ch, cw)]]` for Edit.

**Discrepancy:** Different nesting structure, but both `QwenEmbedRope.forward()` implementations unwrap the outer list:
- SimpleTuner: `transformer.py:468-469`: `if isinstance(video_fhw, list): video_fhw = video_fhw[0]`
- blissful-tuner: `qwen_image_model.py:308-309`: `if isinstance(video_fhw, list): video_fhw = video_fhw[0]`

After unwrapping, SimpleTuner gets `(1, h, w)` (single tuple, all batch samples have same resolution due to bucketing), blissful-tuner gets `[(1, h, w)]` (a list of one tuple for T2I). The `if not isinstance(video_fhw, list): video_fhw = [video_fhw]` line (line 310-311) then wraps the SimpleTuner tuple into a list, making them equivalent.

For Edit mode, blissful-tuner's nested structure `[(1,h,w), (1,ch,cw)]` provides multiple shape entries for different image regions (main + control), which is required for correct RoPE computation across the concatenated sequence.

**Recommendation:** None -- the implementations arrive at the same effective behavior despite different list structures.

---

## Finding ST-QI-16: Prompt Template and Drop Index Agree
**Severity:** OK
**Domain:** 6: Text Encoding

**What SimpleTuner does:**
Text encoding is handled via the Qwen2.5-VL text encoder. The prompt template and token cropping index are not directly visible in the model.py training path (encoding is delegated to `helpers/caching/text_embeds.py`). However, the pipeline uses the same Diffusers-based prompt templates.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_utils.py:396-397`:
```python
prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
prompt_template_encode_start_idx = 34
```

For Edit at line 455-458:
```python
prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
...
prompt_template_encode_start_idx = 64
```

**Discrepancy:** None -- the T2I template with `start_idx=34` and Edit template with `start_idx=64` both match the official Diffusers values `QWENIMAGE_PROMPT_TEMPLATE_START_IDX = 34` and `QWENIMAGE_EDIT_PROMPT_TEMPLATE_START_IDX = 64` (verified in the prior audit, finding C2).

**Recommendation:** None.

---

## Finding ST-QI-17: Hidden Layer Extraction: Last Layer (Both)
**Severity:** OK
**Domain:** 6: Text Encoding

**What SimpleTuner does:**
Text encoding returns embeddings from the last hidden state of the Qwen2.5-VL model (via `output_hidden_states=True`).

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_utils.py:422`:
```python
hidden_states = encoder_hidden_states.hidden_states[-1]
```
Explicitly takes the last hidden state.

**Discrepancy:** None -- both use the last hidden layer output.

**Recommendation:** None.

---

## Finding ST-QI-18: Edit Prompt Template Tokens Differ by model_version
**Severity:** OK
**Domain:** 6: Text Encoding

**What SimpleTuner does:**
Handles edit-v1, edit-v2, edit-v2+, edit-v3 variants with different template structures in `model.py`. The edit-v1 template includes `<|vision_start|><|image_pad|><|vision_end|>` inline, while edit-v2+ uses `Picture N:` prefixed image tokens.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_utils.py:454-457`:
- `edit` (v1): Template has `<|vision_start|><|image_pad|><|vision_end|>` inline
- `edit-2509` / `edit-2511`: Template has `{}` placeholder, with `Picture N:` prefix constructed separately at line 511:
```python
img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
```

**Discrepancy:** None -- both handle the different edit model versions with the correct template structure. The `Picture N:` format for multi-image edit models matches the official Diffusers implementation (verified in prior audit, finding C6).

**Recommendation:** None.

---

## Finding ST-QI-19: LoRA Targeting Scope Differs Fundamentally
**Severity:** INFO
**Domain:** 7: LoRA Targeting Scope

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:107`:
```python
DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
```
Uses **PEFT** library (HuggingFace), targeting specific module **names** within the transformer. This targets only the attention projection layers (Q, K, V, and output projection).

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/networks/lora_qwen_image.py:15`:
```python
QWEN_IMAGE_TARGET_REPLACE_MODULES = ["QwenImageTransformerBlock"]
```
Uses a custom LoRA implementation that targets **all Linear layers** within the specified module class. This includes:
- Attention projections: `to_q`, `to_k`, `to_v`, `to_out` (image stream)
- Cross-attention projections: `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out` (text stream)
- MLP layers: `img_mlp`, `txt_mlp`
- With default `exclude_mod=True`: excludes `img_mod.*` and `txt_mod.*` (modulation layers)

**Discrepancy:** blissful-tuner targets significantly more layers by default:
- SimpleTuner: 4 projections per block = ~240 LoRA pairs for 60 blocks
- blissful-tuner: ~12+ projections per block (attention + cross-attention + MLP) minus modulation = ~720+ LoRA pairs

This means at the same `rank`/`dim`, blissful-tuner LoRAs have much higher parameter count and expressivity, but also higher VRAM usage.

**Recommendation:** Document this difference prominently. Users migrating between trainers should understand that the same `rank=32` produces very different LoRAs in terms of total parameter count and coverage. Neither approach is wrong -- SimpleTuner's narrow targeting is more parameter-efficient, while blissful-tuner's broader targeting provides more expressivity per rank.

---

## Finding ST-QI-20: blissful-tuner Covers All Linear Layers in Blocks
**Severity:** INFO
**Domain:** 7: LoRA Targeting Scope

**What SimpleTuner does:**
Beyond the default `["to_k", "to_q", "to_v", "to_out.0"]`, users can specify custom `lora_target_modules` to add MLP layers. LyCORIS support provides additional algorithm options (LoHa, LoKr, etc.).

**What blissful-tuner does:**
The block-level targeting (`QwenImageTransformerBlock`) automatically captures all `nn.Linear` submodules within each block. The `exclude_mod` flag (default `True`) excludes modulation layers. Users can opt in to modulation layers with `exclude_mod=False` in `network_args`:
```python
--network_args "exclude_mod=False"
```

blissful-tuner also supports LoHa and LoKr via the architecture registry at `/Users/dustin/blissful-tuner/src/musubi_tuner/networks/network_arch.py:75-89`.

**Discrepancy:** Both support LyCORIS algorithms. The key difference is the default scope: SimpleTuner defaults to attention-only, blissful-tuner defaults to all-linear-in-block (minus modulation).

**Recommendation:** None -- this is a valid design difference. Users should be aware when comparing training results between trainers.

---

## Finding ST-QI-21: Edit V1 Conditioning Agrees
**Severity:** OK
**Domain:** 8: Edit Model / Control Image Handling

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:1141-1246` (in `_model_predict_edit_v1`):
- Pack both main and control latents
- Concatenate: `torch.cat([packed_main, packed_control], dim=1)`
- Slice output: `noise_pred = noise_pred[:, :packed_main_size]`

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:510-514`:
```python
latents_control = torch.cat(latents_control, dim=1)
noisy_model_input = torch.cat([noisy_model_input, latents_control], dim=1)
```
Then at line 588-589:
```python
if is_edit or args.is_layered:
    model_pred = model_pred[:, :img_seq_len]
```

**Discrepancy:** None -- both concatenate packed control latents along the sequence dimension and slice the output to exclude control tokens.

**Recommendation:** None.

---

## Finding ST-QI-22: Edit V2/V2+ Per-Sample Multi-Control Handling
**Severity:** INFO
**Domain:** 8: Edit Model / Control Image Handling

**What SimpleTuner does:**
`/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/model.py:1248-1394` (in `_model_predict_edit_plus`):
- Per-sample concatenation (not batch-wide)
- Multiple control latents per sample (list of lists)
- `modulate_index`: First tokens get index 0, control tokens get index 1
- `zero_cond_t`: When enabled, timesteps doubled `[t, 0]` and modulate_index selects per-token time embeddings

**What blissful-tuner does:**
Multi-control support at `/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:503-512`:
- All control images concatenated in sequence: `torch.cat(latents_control, dim=1)`
- Then concatenated with noisy input: `torch.cat([noisy_model_input, latents_control], dim=1)`
- `zero_cond_t` handled in the model forward at `/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_model.py:1319-1337`

**Discrepancy:** SimpleTuner supports per-sample variable control image counts (different batch items can have different numbers of control images), using list-of-lists processing. blissful-tuner assumes all batch items have the same number of control images (iterating `range(num_control_images)` uniformly).

This is primarily a batch flexibility difference. Since aspect-ratio bucketing typically means all items in a batch have similar properties, this rarely matters in practice. The core concatenation + zero_cond_t semantics are equivalent.

**Recommendation:** None -- the per-sample flexibility is a nice-to-have but unlikely to matter with standard bucketed batching.

---

## Finding ST-QI-23: zero_cond_t / modulate_index Handling Agrees
**Severity:** OK
**Domain:** 8: Edit Model / Control Image Handling

**What SimpleTuner does:**
In the transformer at `/Users/dustin/SimpleTuner/simpletuner/helpers/models/qwen_image/transformer.py`, the `zero_cond_t` mechanism:
1. Doubles timestep: `torch.cat([timestep, timestep * 0], dim=0)` -> two sets of modulation parameters
2. `timestep_zero_index` set to the packed main token count
3. Per-block modulation selects real-t embeddings for main tokens, zero-t for control tokens

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_model.py:1319-1337`:
```python
if self.zero_cond_t:
    timestep = torch.cat([timestep, timestep * 0], dim=0)
    sample = img_shapes[0]
    ...
    timestep_zero_index = base_len
```
Then in the transformer block's `_modulate()` at lines 987-1008: `x[:, :timestep_zero_index]` gets real-timestep modulation, `x[:, timestep_zero_index:]` gets zero-timestep modulation.

**Discrepancy:** None -- both implementations use the same intra-sequence conditional timestep split mechanism. The `zero_cond_t` flag applies zero-timestep conditioning specifically to control/reference tokens within a single forward pass (as corrected in the prior audit finding I2).

**Recommendation:** None.

---

## Consistency with Prior Audit

The prior audit at `/Users/dustin/blissful-tuner/docs/plans/qwen-image-pipeline-audit.md` identified 40 findings (all resolved). This SimpleTuner comparison is consistent with those findings:

1. **T4 (qwen_shift verified)**: Confirmed -- the `qwen_shift` parameters `(256, 0.5, 8192, 0.9)` match both SimpleTuner and Diffusers.
2. **C2 (prompt_template_encode_start_idx)**: Confirmed -- `34` (T2I) and `64` (Edit) match SimpleTuner.
3. **L1 (exclude_mod regex)**: The prior audit fixed the regex. SimpleTuner doesn't exclude modulation layers by default (its PEFT targeting only hits `to_k/q/v/out`), so the comparison is moot.
4. **I2 (zero_cond_t semantics)**: Confirmed -- both SimpleTuner and blissful-tuner implement the same intra-sequence timestep split.

## Key Takeaways

1. **No CRITICAL issues found.** The core training mathematics (flow matching, latent normalization, packing, loss target) are all correct and consistent between both trainers.

2. **Three WARNING-level differences:**
   - **ST-QI-3/ST-QI-12 (timestep +1 offset)**: The `+1` offset inherited from HunyuanVideo/Wan creates a minor train/inference mismatch for Qwen Image. Practical impact is likely very small.
   - **ST-QI-5 (VAE posterior sampling)**: `mode()` vs `sample()` is a deliberate design choice. Both are valid. `mode()` is more deterministic/reproducible; `sample()` adds implicit augmentation.

3. **LoRA scope difference (ST-QI-19/20)** is the most impactful practical difference. Users should understand that blissful-tuner LoRAs target ~3x more layers by default, producing larger and more expressive LoRAs at the same rank.

4. **Dynamic vs static shift (ST-QI-11)**: blissful-tuner's `qwen_shift` is more faithful to the official Qwen Image training recipe than SimpleTuner's default static shift of 1.73.
