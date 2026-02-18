# SimpleTuner vs blissful-tuner: FLUX.2 LoRA Training Pipeline Comparison

**Date:** 2026-02-18
**Scope:** Cross-codebase comparison of FLUX.2 LoRA training pipelines
**SimpleTuner version:** 4.1.0 (codebase at `/Users/dustin/SimpleTuner/`)
**blissful-tuner:** main branch (codebase at `/Users/dustin/blissful-tuner/`)
**Method:** Source code reading across 8 comparison domains

---

## Summary Table

| Finding | Title | Severity | Domain |
|---------|-------|----------|--------|
| ST-1 | Flow matching formula agrees | OK | 1: Flow Matching |
| ST-2 | Timestep +1 offset in blissful-tuner | INFO | 1: Flow Matching |
| ST-3 | Sigma sampling strategies differ in availability | INFO | 1: Flow Matching |
| ST-4 | Sigmoid scale default differs (1.0 vs 5.0) | WARNING | 1: Flow Matching |
| ST-5 | Latent scaling via batch norm equivalent | OK | 2: Latent Scaling |
| ST-6 | VAE patchification + normalization at cache time equivalent | OK | 2: Latent Scaling |
| ST-7 | Loss target formula agrees (noise - latents) | OK | 3: Loss Target |
| ST-8 | Loss target computed in post-patchify 128-channel space in both | OK | 3: Loss Target |
| ST-9 | Loss reduction is equivalent | OK | 3: Loss Target |
| ST-10 | SimpleTuner normalizes timesteps via batch["timesteps"]/1000; blissful-tuner via call_dit | OK | 4: Timestep Normalization |
| ST-11 | Resolution-dependent shift sequence length differs between auto_shift and flux2_shift | WARNING | 4: Timestep Normalization |
| ST-12 | Noise occurs before packing in both codebases | OK | 5: Patchification |
| ST-13 | Channel count after patchification agrees (128) | OK | 5: Patchification |
| ST-14 | Packing implementations differ but are functionally equivalent | OK | 5: Patchification |
| ST-15 | SimpleTuner guidance configurable; blissful-tuner hardcodes 1.0 | INFO | 6: Guidance |
| ST-16 | Position ID construction agrees | OK | 7: Position IDs |
| ST-17 | Reference image time-offset scheme agrees | OK | 7: Position IDs |
| ST-18 | LoRA targeting: SimpleTuner PEFT-based vs blissful-tuner block-level | INFO | 8: LoRA Targeting |
| ST-19 | SimpleTuner has more targeting granularity modes | INFO | 8: LoRA Targeting |
| ST-20 | Both support LyCORIS but with different integration depths | INFO | 8: LoRA Targeting |

**Severity counts:** 0 CRITICAL, 2 WARNING, 8 INFO, 10 OK

---

## Finding ST-1: Flow Matching Formula Agrees
**Severity:** OK
**Domain:** 1: Flow Matching Formula

**What SimpleTuner does:**
File: `simpletuner/helpers/models/common.py:4374`
```python
batch["noisy_latents"] = (1 - batch["sigmas"]) * batch["latents"] + batch["sigmas"] * batch["input_noise"]
```
Where `batch["input_noise"]` = `batch["noise"]` (absent perturbation), and `batch["sigmas"]` is broadcastto latent shape via `expand_sigmas()` (`model.py:5144`: `batch["sigmas"].view(-1, 1, 1, 1)` for 4D).

**What blissful-tuner does:**
File: `src/musubi_tuner/hv_train_network.py:1124-1125`
```python
t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
noisy_model_input = (1 - t) * latents + t * noise
```

**Discrepancy:** None -- implementations agree. Both use the rectified flow formulation `noisy = (1-sigma)*clean + sigma*noise` with sigma (or `t`) in [0, 1].

**Recommendation:** None.

---

## Finding ST-2: Timestep +1 Offset in blissful-tuner
**Severity:** INFO
**Domain:** 1: Flow Matching Formula

**What SimpleTuner does:**
File: `simpletuner/helpers/models/common.py:3710-3711`
```python
timesteps = sigmas * 1000.0
return sigmas, timesteps
```
Sigmas are in [0, 1], so timesteps are in [0, 1000]. Then in `model_predict` (file: `simpletuner/helpers/models/flux2/model.py:820`):
```python
timesteps = timesteps.expand(batch_size) / SCHEDULER_CONFIG["num_train_timesteps"]
```
Where `num_train_timesteps = 1000`. Result: timesteps in [0, 1].

**What blissful-tuner does:**
File: `src/musubi_tuner/hv_train_network.py:1123,1127`
```python
timesteps = t * 1000.0
...
timesteps += 1  # 1 to 1000
```
So timesteps go from [1, 1001]. Then in `call_dit` (file: `src/musubi_tuner/flux_2_train_network.py:457`):
```python
timesteps = timesteps / 1000.0
```
Result: timesteps in [0.001, 1.001].

**Discrepancy:** blissful-tuner adds +1 to timesteps before dividing by 1000, producing a 0.001 offset at both ends compared to SimpleTuner. This `+1` is inherited from the base `NetworkTrainer` class and was designed for the FlowMatchDiscreteScheduler which uses 1-indexed timesteps [1, 1000]. For FLUX.2 where the transformer receives normalized timesteps in [0, 1], this means the transformer never sees exactly t=0.0 (fully clean) and slightly exceeds t=1.0 (fully noisy). The magnitude of the offset (0.1%) is negligible in practice and unlikely to affect training quality.

**Recommendation:** Low priority. If a purist correction is desired, the FLUX.2 trainer could override `get_noisy_model_input_and_timesteps` to skip the `+1` or use `timesteps / 1001.0`. However, this matches the behavior for all other architectures using the same base class and changing it may introduce subtle regressions.

---

## Finding ST-3: Sigma Sampling Strategies Differ in Availability
**Severity:** INFO
**Domain:** 1: Flow Matching Formula

**What SimpleTuner does:**
File: `simpletuner/helpers/models/common.py:3687-3711`
Four sigma sampling strategies:
1. **Sigmoid** (default): `torch.sigmoid(flow_sigmoid_scale * torch.randn(...))`
2. **Uniform**: `torch.rand(...)`
3. **Beta**: `Beta(alpha, beta).sample(...)`
4. **Fast** (discrete): `[1.0]*7 + [0.75, 0.5, 0.25]`

Plus `flow_custom_timesteps` (line 3668) for fixed-timestep training.

**What blissful-tuner does:**
File: `src/musubi_tuner/hv_train_network.py:974-1090`
Nine sigma sampling strategies:
1. **uniform**: `torch.rand(...)`
2. **sigmoid**: `torch.sigmoid(sigmoid_scale * randn(...))`
3. **shift**: Fixed `discrete_flow_shift` parameter
4. **flux_shift**: Resolution-dependent shift for FLUX.1 (`(h//2)*(w//2)`)
5. **flux2_shift**: Resolution-dependent shift for FLUX.2 (`h*w`)
6. **qwen_shift**: Resolution-dependent shift for Qwen-Image
7. **logsnr**: Log-SNR sampling from arXiv:2411.14793v3
8. **qinglong_flux**: Triple hybrid (80% mid_shift + 7.5% logsnr + 12.5% logsnr2)
9. **qinglong_qwen**: Same hybrid pattern for Qwen

Plus `weighting_scheme`-based sampling (logit_normal, mode, cosmap, sigma_sqrt, structure_bell) in the else branch (line 1128+).

**Discrepancy:** blissful-tuner has significantly more sigma sampling strategies. SimpleTuner has Beta and Fast (discrete) that blissful-tuner lacks. Neither codebase is missing something critical -- they simply offer different experimental options.

**Recommendation:** None needed. The core strategies (sigmoid, uniform, resolution-dependent shift) are present in both.

---

## Finding ST-4: Sigmoid Scale Default Differs (1.0 vs 5.0)
**Severity:** WARNING
**Domain:** 1: Flow Matching Formula

**What SimpleTuner does:**
File: `simpletuner/helpers/models/common.py:3693`
```python
sigmas = torch.sigmoid(self.config.flow_sigmoid_scale * torch.randn((bsz,), device=self.accelerator.device))
```
Default `flow_sigmoid_scale = 5.0` (from reference doc Section 8, confirmed in `simpletuner_sdk/interface.py:288`).

A scale of 5.0 produces a nearly uniform distribution of sigmas, because `sigmoid(5*N(0,1))` maps most of the normal distribution's mass to values near 0 or 1.

**What blissful-tuner does:**
File: `src/musubi_tuner/hv_train_network.py:1007,3077-3080`
```python
t = torch.sigmoid(args.sigmoid_scale * randn(batch_size, org_timesteps))
```
Default `sigmoid_scale = 1.0` (from argparse at line 3079).

A scale of 1.0 concentrates sigma sampling toward the middle (around sigma=0.5), with fewer samples at the extremes.

**Discrepancy:** The default sigmoid scale differs by 5x between the two codebases. With `sigmoid_scale=1.0`, blissful-tuner's sigmoid sampling is significantly more concentrated around mid-noise levels compared to SimpleTuner's nearly-uniform distribution with `flow_sigmoid_scale=5.0`. This is relevant because:

1. blissful-tuner's `sigmoid_scale=1.0` is the same default used by FLUX.1 (upstream musubi-tuner heritage). It may or may not be optimal for FLUX.2.
2. SimpleTuner's `flow_sigmoid_scale=5.0` was specifically chosen (or defaulted) for FLUX.2.
3. The `flux2_shift` and `flux_shift` modes in blissful-tuner also use `sigmoid_scale` (line 1029: `logits_norm * args.sigmoid_scale`) before applying the shift, so the scale affects the shifted distributions too.

Note that this is a *default* difference -- users can override `--sigmoid_scale` in blissful-tuner to match SimpleTuner's behavior.

**Recommendation:** Consider documenting that `--sigmoid_scale 5.0` may be more appropriate for FLUX.2 when using `sigmoid` or `flux2_shift` sampling. Alternatively, the `flux2_shift` mode could have its own default that differs from the global `sigmoid_scale`. No code change needed unless empirical testing shows the higher scale is beneficial.

---

## Finding ST-5: Latent Scaling via Batch Norm Equivalent
**Severity:** OK
**Domain:** 2: Latent Scaling

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/model.py:298-306`
```python
def _normalize_latents(self, latents: Tensor) -> Tensor:
    if self.vae is None or not hasattr(self.vae, "bn"):
        return latents
    bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
    return (latents - bn_mean) / bn_std
```
This is called from `post_vae_encode_transform_sample` (line 950) at VAE cache time. The VAE itself (`AutoencoderKLFlux2` in `autoencoder.py`) does NOT internally apply batch norm -- it returns raw DiagonalGaussianDistribution from its encoder. The batch norm statistics are stored in `self.vae.bn` (a `nn.BatchNorm2d` created at line 170-176).

The `AUTOENCODER_SCALING_FACTOR = 1.0` (no additional multiplicative scaling).

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2/flux2_models.py:367-388`
```python
def normalize(self, z):
    self.bn.eval()
    return self.bn(z)

def encode(self, x: Tensor) -> Tensor:
    moments = self.encoder(x)
    mean = torch.chunk(moments, 2, dim=1)[0]
    z = rearrange(mean, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=self.ps[0], pj=self.ps[1])
    z = self.normalize(z)
    return z
```
The batch norm is applied directly inside the VAE's `encode()` method. The BN module is created at lines 357-365 with the same configuration (eps=1e-4, momentum=0.1, affine=False).

**Discrepancy:** None functionally. Both apply the same batch norm normalization using the same running statistics loaded from the same checkpoint. The implementation differs in WHERE the normalization happens:
- SimpleTuner: VAE encode returns raw latents, then separate `_normalize_latents()` + `_patchify_latents()` called at cache time
- blissful-tuner: VAE encode internally does pixel shuffle (rearrange) + batch norm before returning

The end result is the same: 128-channel, BN-normalized latents stored in cache. No multiplicative scaling factor is applied in either codebase.

**Recommendation:** None.

---

## Finding ST-6: VAE Patchification + Normalization at Cache Time Equivalent
**Severity:** OK
**Domain:** 2: Latent Scaling

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/model.py:936-951`
```python
def post_vae_encode_transform_sample(self, sample):
    ...
    if isinstance(sample, Tensor) and sample.dim() == 4:
        if sample.shape[1] == 32:
            sample = self._patchify_latents(sample)
        sample = self._normalize_latents(sample)
    return sample
```
Called from `simpletuner/helpers/caching/vae.py:834`. The function first extracts the mean from the DiagonalGaussianDistribution (line 944: `.latent_dist.mode()`), then patchifies (32ch -> 128ch via pixel shuffle, line 949), then normalizes (batch norm, line 950).

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2_cache_latents.py:71`
```python
latents = ae.encode(contents.to(ae.device, dtype=ae.dtype))
```
Where `ae.encode()` (file: `flux2_models.py:377-388`) internally performs: encoder -> mean extraction -> pixel shuffle rearrange -> batch norm normalize. The cached result is already 128-channel, normalized.

**Discrepancy:** None. Both produce 128-channel, BN-normalized latents at cache time. Both use the mean of the distribution (not sampling).

**Recommendation:** None.

---

## Finding ST-7: Loss Target Formula Agrees (noise - latents)
**Severity:** OK
**Domain:** 3: Loss Target Construction

**What SimpleTuner does:**
File: `simpletuner/helpers/models/common.py:3516`
```python
target = prepared_batch["noise"] - prepared_batch["latents"]
```
This is the flow matching velocity target `v = z_1 - z_0 = noise - clean`.

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2_train_network.py:471-473`
```python
# flow matching loss: target = v = (z_1 - z_0) = (noise - latents)
target = noise - latents
```

**Discrepancy:** None -- implementations agree exactly.

**Recommendation:** None.

---

## Finding ST-8: Loss Target Computed in Post-Patchify 128-Channel Space in Both
**Severity:** OK
**Domain:** 3: Loss Target Construction

**What SimpleTuner does:**
In `prepare_batch` (`common.py:4342`): `noise = torch.randn_like(batch["latents"])` where `batch["latents"]` is 128-channel (post-patchify, post-normalize from cache). The target (`noise - latents`) is thus in 128-channel space.

In `model_predict` (`flux2/model.py:793-798`), patchification is applied to noisy_latents, latents, and noise IF they're still 32-channel, but the comments note "if they haven't been pixel-shuffled yet". Since they were processed by `post_vae_encode_transform_sample` at cache time, they're already 128-channel. So the guard at line 793 (`if shape[1] == 32`) is a safety fallback that normally doesn't trigger.

**What blissful-tuner does:**
In the training loop (`hv_train_network.py:2431`): `noise = torch.randn_like(latents)` where `latents` is 128-channel (from cache). In `call_dit` (`flux_2_train_network.py:473`): `target = noise - latents` in 128-channel space.

**Discrepancy:** None. Both compute loss target in 128-channel (patchified, BN-normalized) space.

**Recommendation:** None.

---

## Finding ST-9: Loss Reduction is Equivalent
**Severity:** OK
**Domain:** 3: Loss Target Construction

**What SimpleTuner does:**
File: `simpletuner/helpers/models/common.py:4627,4767-4770`
```python
loss = (model_pred.float() - target.float()) ** 2
...
loss = loss.mean(dim=list(range(1, len(loss.shape))))  # mean over spatial dims -> (B,)
loss = loss.mean()  # mean over batch -> scalar
```
This is equivalent to `loss.mean()` over all dimensions (two-step reduction = one-step mean).

**What blissful-tuner does:**
File: `src/musubi_tuner/hv_train_network.py:2491`
```python
loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
```
Then in `apply_masked_loss_with_prior` (`modules/mask_loss.py:313`):
```python
if mask_weights is None or not getattr(args, "use_mask_loss", False):
    return loss.float().mean()
```

**Discrepancy:** None for the non-masked case. Both compute unreduced MSE then take the global mean. For masked loss, blissful-tuner uses weighted-mean normalization which SimpleTuner does not support (SimpleTuner has its own conditioning mask system using `conditioning_pixel_values` with a different scaling approach at lines 4750-4765).

**Recommendation:** None.

---

## Finding ST-10: Timestep Normalization Approaches Agree
**Severity:** OK
**Domain:** 4: Timestep Normalization

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/model.py:820`
```python
timesteps = timesteps.expand(batch_size) / SCHEDULER_CONFIG["num_train_timesteps"]
```
Where `SCHEDULER_CONFIG["num_train_timesteps"] = 1000` (line 65). Timesteps are divided by 1000 to normalize to [0, 1] before passing to the transformer.

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2_train_network.py:457`
```python
timesteps = timesteps / 1000.0
```
Same normalization, but with the +1 offset noted in ST-2 producing [0.001, 1.001] instead of [0, 1].

**Discrepancy:** See ST-2 for the minor +1 offset. The normalization approach is the same.

**Recommendation:** None.

---

## Finding ST-11: Resolution-Dependent Shift Sequence Length Computation Differs
**Severity:** WARNING
**Domain:** 4: Timestep Normalization

**What SimpleTuner does:**
File: `simpletuner/helpers/training/custom_schedule.py:452-474`
```python
if noise.ndim == 5:
    num_frames, height, width = noise.shape[-3:]
else:
    num_frames = 1
    height, width = noise.shape[-2:]
patch_size = getattr(noise_scheduler.config, "patch_size", 2)
seq_len = num_frames * (height // patch_size) * (width // patch_size)
...
mu = calculate_shift_flux(seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift)
```
For FLUX.2 at 1024x1024: latents are 128ch / 64x64 (post-patchify). `seq_len = 1 * (64 // 2) * (64 // 2) = 1024`.
With `base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15`:
`mu = 0.5 + (1.15-0.5)/(4096-256) * (1024-256) = 0.5 + 0.13 = 0.630`
`shift = exp(0.630) = 1.877`

**What blissful-tuner does:**
File: `src/musubi_tuner/hv_train_network.py:1019-1021`
```python
elif args.timestep_sampling == "flux2_shift":
    # FLUX.2 uses h*w (not halved) due to different latent dimensions
    mu = train_utils.get_lin_function(y1=0.5, y2=1.15)(h * w)
```
For FLUX.2 at 1024x1024: latents are 128ch / 64x64 (post-patchify). `seq_len = 64 * 64 = 4096`.
`mu = 0.5 + (1.15-0.5)/(4096-256) * (4096-256) = 1.15`
`shift = exp(1.15) = 3.158`

**Discrepancy:** For the same resolution, blissful-tuner's `flux2_shift` computes a shift of 3.158 while SimpleTuner's `flow_schedule_auto_shift` computes 1.877. The root cause is that SimpleTuner additionally divides by `patch_size=2` when computing sequence length, while blissful-tuner uses the raw spatial dimensions of the (already patchified) latent tensor.

The question is: what sequence length does the FLUX.2 transformer actually see? After packing `(B, 128, 64, 64)` -> `(B, 4096, 128)`, the transformer processes 4096 tokens. This suggests blissful-tuner's interpretation (using `h*w = 4096` as the sequence length) is more aligned with the actual transformer's perspective.

SimpleTuner's implementation appears to apply a FLUX.1-era `patch_size=2` divisor that may not be appropriate for FLUX.2 latents that have already been pixel-shuffled. In FLUX.1, the VAE outputs 16-channel latents at `(H/8, W/8)`, and the pack operation applies 2x2 packing to create 64-channel `(H/16, W/16)` tokens. So `seq_len = (H/8 / 2) * (W/8 / 2)` = actual token count. For FLUX.2, the VAE already includes the 2x2 pixel shuffle to create 128-channel `(H/16, W/16)` latents, and the pack operation just flattens -- no additional 2x2 folding. So `seq_len = H/16 * W/16` = actual token count.

However, it must be noted that:
1. These are separate user-selected modes (`flux2_shift` in blissful-tuner vs `flow_schedule_auto_shift` in SimpleTuner).
2. The default in blissful-tuner is `--timestep_sampling sigmoid` (not `flux2_shift`), so this only affects users who explicitly opt in.
3. SimpleTuner's `flow_schedule_auto_shift` was designed for FLUX.1 and may have been ported to FLUX.2 without adjusting for the pre-patchified latents.

**Recommendation:** If users adopt `--timestep_sampling flux2_shift`, the current implementation (using `h*w` for sequence length) appears correct for FLUX.2's pre-patchified latent space. SimpleTuner's additional `//patch_size` division may be an artifact of FLUX.1 compatibility that creates under-shifted schedules for FLUX.2. Worth monitoring empirically but no code change recommended in blissful-tuner.

---

## Finding ST-12: Noise Occurs Before Packing in Both Codebases
**Severity:** OK
**Domain:** 5: Patchification / Packing Order

**What SimpleTuner does:**
1. Cached latents: 128-channel, BN-normalized, 4D `(B, 128, H, W)` (from `post_vae_encode_transform_sample`)
2. Noise sampled: `torch.randn_like(batch["latents"])` in 128ch 4D space (`common.py:4342`)
3. Noisy latents: `(1-sigma)*latents + sigma*noise` in 128ch 4D space (`common.py:4374`)
4. Packing: `pack_latents(noisy_latents)` from 4D to sequence format `(B, S, 128)` (`model.py:805`)
5. Transformer forward pass
6. Unpacking: back to 4D for loss computation

**What blissful-tuner does:**
1. Cached latents: 128-channel, BN-normalized, 4D `(B, 128, H, W)` (from `ae.encode()`)
2. Noise sampled: `torch.randn_like(latents)` in 128ch 4D space (`hv_train_network.py:2431`)
3. Noisy latents: `(1-t)*latents + t*noise` in 128ch 4D space (`hv_train_network.py:1125`)
4. Packing: `batched_prc_img(noisy_model_input)` from 4D to `(B, HW, C)` (`flux_2_train_network.py:405`)
5. Transformer forward pass
6. Unpacking: `rearrange` back to 4D (`flux_2_train_network.py:469`)

**Discrepancy:** None. Both add noise in the pre-packed 4D space and then pack for the transformer.

**Recommendation:** None.

---

## Finding ST-13: Channel Count After Patchification Agrees (128)
**Severity:** OK
**Domain:** 5: Patchification / Packing Order

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/model.py:92`
```python
LATENT_CHANNEL_COUNT = 128  # 32 VAE channels * 4 (2x2 pixel shuffle) = 128 transformer channels
```
Patchification (`_patchify_latents`, line 285-296): `(B, 32, H, W) -> (B, 128, H/2, W/2)`.

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2/flux2_models.py:30`
```python
@dataclass
class Flux2Params:
    in_channels: int = 128
```
The VAE `encode()` method (line 377-388) uses `rearrange("... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)` which is equivalent to pixel shuffle: 32ch -> 128ch with halved spatial.

**Discrepancy:** None. Both use 128 channels.

**Recommendation:** None.

---

## Finding ST-14: Packing Implementations Differ But Are Functionally Equivalent
**Severity:** OK
**Domain:** 5: Patchification / Packing Order

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/__init__.py:22-53`
```python
def pack_latents(latents: Tensor) -> Tuple[Tensor, Tensor]:
    ...
    for i in range(batch_size):
        x = latents[i]  # (C, H, W)
        coords = {"t": torch.arange(1), "h": torch.arange(h), "w": torch.arange(w), "l": torch.arange(1)}
        x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
        x_flat = rearrange(x, "c h w -> (h w) c")
        ...
```
Per-sample loop, creates position IDs with `cartesian_prod`.

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2/flux2_utils.py:392-415`
```python
def prc_img(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    ...
    x_coords = {"t": torch.arange(1) if t_coord is None else t_coord,
                 "h": torch.arange(h), "w": torch.arange(w), "l": torch.arange(1)}
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x = rearrange(x, "c h w -> (h w) c") if x.ndim == 3 else rearrange(x, "b c h w -> b (h w) c")
```
Then `batched_prc_img = batched_wrapper(prc_img)` (line 419) wraps it to process batch dimension.

**Discrepancy:** None functionally. Both flatten `(C, H, W)` -> `(HW, C)` and create 4D position tuples `(t, h, w, l)` using `cartesian_prod`. The implementation style differs (per-sample loop vs per-sample wrapper function) but produces identical outputs.

**Recommendation:** None.

---

## Finding ST-15: SimpleTuner Guidance Configurable; blissful-tuner Hardcodes 1.0
**Severity:** INFO
**Domain:** 6: Guidance Embedding During Training

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/model.py:822-839`
```python
if self.config.flux_guidance_mode == "constant":
    guidance_value = float(self.config.flux_guidance_value)
    guidance = torch.full((batch_size,), guidance_value, device=device, dtype=dtype)
elif self.config.flux_guidance_mode == "random-range":
    guidance = torch.tensor([random.uniform(self.config.flux_guidance_min, self.config.flux_guidance_max)
                             for _ in range(batch_size)], ...)
else:
    guidance = torch.ones(batch_size, device=device, dtype=dtype)
```
The default is set at line 976-977:
```python
self.config.flux_guidance_mode = "constant"
self.config.flux_guidance_value = 1.0
```
So the default is 1.0, but users can configure `flux_guidance_mode` to `"random-range"` for guidance dropout/variation during training.

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2_train_network.py:448-449`
```python
# use 1.0 as guidance scale for FLUX.2 training
guidance_vec = torch.full((bsize,), 1.0, device=accelerator.device, dtype=network_dtype)
```
Hardcoded to 1.0 with no configuration option.

**Discrepancy:** blissful-tuner always uses 1.0 for the guidance embedding during training, matching SimpleTuner's default. SimpleTuner additionally supports `random-range` guidance mode which can be useful for guidance-distilled model training (though the reference document states "always 1.0 during training" as the recommended default).

**Recommendation:** Low priority feature gap. Consider adding a `--training_guidance_scale` argument if users request guidance augmentation for FLUX.2 training. The current hardcoded 1.0 is correct for standard LoRA training.

---

## Finding ST-16: Position ID Construction Agrees
**Severity:** OK
**Domain:** 7: Position ID Construction

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/__init__.py:37-44` (images)
```python
coords = {
    "t": torch.arange(1, device=x.device),  # t=0
    "h": torch.arange(h, device=x.device),
    "w": torch.arange(w, device=x.device),
    "l": torch.arange(1, device=x.device),  # l=0
}
x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
```
Text IDs (lines 110-116): `t=0, h=0, w=0, l=0..seq_len-1`.

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2/flux2_utils.py:405-411` (images)
```python
x_coords = {
    "t": torch.arange(1) if t_coord is None else t_coord,
    "h": torch.arange(h),
    "w": torch.arange(w),
    "l": torch.arange(1),
}
x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
```
Text IDs (`prc_txt`, line 345-351): `t=0, h=0, w=0, l=0..seq_len-1`.

**Discrepancy:** None. Both use 4D tuples `(t, h, w, l)` with identical coordinate assignments.

**Recommendation:** None.

---

## Finding ST-17: Reference Image Time-Offset Scheme Agrees
**Severity:** OK
**Domain:** 7: Position ID Construction

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/__init__.py:196-197`
```python
t_coord = torch.tensor([scale + scale * ref_idx], device=device)  # scale=10
```
So reference image 0 gets t=10, reference 1 gets t=20, etc.

**What blissful-tuner does:**
File: `src/musubi_tuner/flux_2_train_network.py:412-414`
```python
scale = 10
t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
```
So reference image 0 gets t=10, reference 1 gets t=20, etc.

**Discrepancy:** None. Both use `t = scale + scale * index` with `scale=10`.

**Recommendation:** None.

---

## Finding ST-18: LoRA Targeting: SimpleTuner PEFT-Based vs blissful-tuner Block-Level
**Severity:** INFO
**Domain:** 8: LoRA Targeting Scope

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/model.py:97-116`
Uses PEFT (HuggingFace Parameter-Efficient Fine-Tuning) library:
```python
DEFAULT_LORA_TARGET = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.to_qkv_mlp_proj",
]
```
Applied via `LoraConfig(target_modules=[...])` from PEFT. This targets specific named Linear modules within the diffusers-style Flux2Transformer2DModel.

**What blissful-tuner does:**
File: `src/musubi_tuner/networks/lora_flux_2.py:16`
```python
FLUX_2_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]
```
Uses custom block-level LoRA implementation (from `networks/lora.py`). This targets ALL Linear layers within the specified block classes, excluding those matching `r".*(norm).*"`.

**Discrepancy:** Architectural difference in LoRA application:
- SimpleTuner: Targets specific named projection layers (diffusers naming). Default covers attention projections only.
- blissful-tuner: Targets all Linear layers within block classes (minus norms). This means blissful-tuner's default LoRA scope is WIDER than SimpleTuner's default -- it includes MLP/feedforward layers within the blocks, not just attention projections.

This means a blissful-tuner LoRA at the same rank will have more parameters (since it covers more layers) but each layer's adaptation may be thinner. The practical impact depends on the use case.

**Recommendation:** This is an intentional architectural difference. Document the coverage difference in training guides so users understand that blissful-tuner LoRAs target more layers by default. Users wanting to match SimpleTuner's narrower targeting can use `--exclude_patterns` to skip MLP layers, or use LoHa/LoKr which support per-module include/exclude patterns.

---

## Finding ST-19: SimpleTuner Has More Targeting Granularity Modes
**Severity:** INFO
**Domain:** 8: LoRA Targeting Scope

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/model.py:997-1036`
Provides named targeting modes:
- `"all"`: Attention + MLP + cross-attention (14 module types)
- `"attention"`: All attention projections including cross-attention (10 module types)
- `"mlp"`: Only feedforward layers (4 module types)
- `"tiny"`: Only q/k/v projections (3 module types)
- `"slider"`: Image-stream only attention (5 module types, excludes text cross-attention)

Plus `"standard"` (default) which maps to `DEFAULT_LORA_TARGET` (5 module types).

**What blissful-tuner does:**
File: `src/musubi_tuner/networks/lora_flux_2.py:16,39`
Single targeting mode: all Linear layers within `DoubleStreamBlock`/`SingleStreamBlock`, minus norms.

Per-layer targeting is possible through `--exclude_patterns` (regex patterns) and `--include_patterns`, but there are no named presets like SimpleTuner's "tiny"/"mlp"/"slider" modes.

**Discrepancy:** SimpleTuner provides more convenient named targeting modes. blissful-tuner's approach is more flexible (regex patterns can express any combination) but less discoverable.

**Recommendation:** Consider adding named targeting presets as syntactic sugar over exclude patterns, e.g., `--lora_target_mode tiny` that automatically sets appropriate exclude patterns. Low priority -- the current regex approach is functional.

---

## Finding ST-20: Both Support LyCORIS But With Different Integration Depths
**Severity:** INFO
**Domain:** 8: LoRA Targeting Scope

**What SimpleTuner does:**
File: `simpletuner/helpers/models/flux2/model.py:125`
```python
DEFAULT_LYCORIS_TARGET = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
```
LyCORIS support via `--lora_type=lycoris --lycoris_config=path/to/config.json`.
Supported algorithms: `lora, loha, lokr, full, ia3, dylora, diag-oft, boft, tlora`.

**What blissful-tuner does:**
File: `src/musubi_tuner/networks/network_arch.py:63-73`
```python
ARCHITECTURE_FLUX_2_DEV: {
    "target_modules": FLUX_2_TARGET_REPLACE_MODULES,  # ["DoubleStreamBlock", "SingleStreamBlock"]
    "exclude_patterns": [r".*(norm).*"],
},
```
LyCORIS support via `--prefer_lycoris` flag at inference, and via `networks/loha.py` and `networks/lokr.py` for training. The architecture registry (`network_arch.py`) provides defaults for LoHa/LoKr.

Additionally, blissful-tuner has native LoHa and LoKr implementations that use the same architecture registry but are not LyCORIS-based (they're custom implementations with `lokr_factor` persistence and LoKr v1/v2 modes).

**Discrepancy:** SimpleTuner delegates entirely to the LyCORIS library for advanced decomposition methods. blissful-tuner has custom LoHa/LoKr implementations alongside LyCORIS support for inference merging. blissful-tuner's native implementations may diverge from LyCORIS's behavior in edge cases.

**Recommendation:** The current approach is fine. The native LoHa/LoKr implementations provide more control and don't require the LyCORIS dependency for training.

---

## Appendix: Key File References

### SimpleTuner
| File | Purpose |
|------|---------|
| `simpletuner/helpers/models/flux2/model.py` | Flux2 class, model_predict, LoRA targets, VAE patchify/normalize |
| `simpletuner/helpers/models/flux2/__init__.py` | pack_latents, unpack_latents, build_conditioning_inputs |
| `simpletuner/helpers/models/flux2/autoencoder.py` | AutoencoderKLFlux2 (diffusers-style VAE) |
| `simpletuner/helpers/models/common.py` | prepare_batch (noise, sigmas), loss computation, sigma sampling |
| `simpletuner/helpers/training/custom_schedule.py` | apply_flow_schedule_shift (resolution-dependent) |
| `simpletuner/helpers/training/trainer.py` | Training loop, model_predict wrapper |

### blissful-tuner
| File | Purpose |
|------|---------|
| `src/musubi_tuner/flux_2_train_network.py` | Flux2NetworkTrainer, call_dit, scale_shift_latents |
| `src/musubi_tuner/hv_train_network.py` | NetworkTrainer base, get_noisy_model_input_and_timesteps, training loop |
| `src/musubi_tuner/flux_2/flux2_models.py` | Flux2 model, AutoEncoder (with BN), Flux2Params |
| `src/musubi_tuner/flux_2/flux2_utils.py` | prc_img/prc_txt, denoise, text encoders, compute_empirical_mu |
| `src/musubi_tuner/flux_2_cache_latents.py` | Latent caching pipeline |
| `src/musubi_tuner/networks/lora_flux_2.py` | LoRA target modules, create_arch_network |
| `src/musubi_tuner/networks/network_arch.py` | Architecture registry for LoHa/LoKr |
| `src/musubi_tuner/modules/mask_loss.py` | apply_masked_loss_with_prior |
