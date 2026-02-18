# OneTrainer vs blissful-tuner FLUX.2 LoRA Pipeline Comparison

**Date:** 2026-02-18
**Scope:** Side-by-side source code comparison of FLUX.2 LoRA training pipelines
**Method:** Line-by-line reading of OneTrainer (at `/Users/dustin/OneTrainer/`) and blissful-tuner (at `/Users/dustin/blissful-tuner/`) source files
**Prior context:** blissful-tuner internal audit findings from `docs/plans/2026-02-17-flux2-pipeline-audit-fixes.md` (37 findings, all addressed)

---

## Summary Table

| Finding | Title | Severity | Domain |
|---------|-------|----------|--------|
| OT-1 | Flow matching noise formula agrees | OK | 1: Flow Matching Formula |
| OT-2 | Sigma computation differs: 1-based vs 0-based indexing | WARNING | 1: Flow Matching Formula |
| OT-3 | Latent scaling pipeline is mathematically equivalent | OK | 2: Latent Scaling |
| OT-4 | Loss target formula agrees (noise - scaled_latent) | OK | 3: Loss Target Construction |
| OT-5 | Loss computed in different spatial domains (unpatchified vs patchified) | INFO | 3: Loss Target Construction |
| OT-6 | Mask-weighted loss operates at different spatial resolutions | WARNING | 3: Loss Target Construction |
| OT-7 | Timestep division by 1000 agrees | OK | 4: Timestep Normalization |
| OT-8 | Dynamic timestep shift formula differs significantly | WARNING | 4: Timestep Normalization |
| OT-9 | Patchification order is equivalent despite different staging | OK | 5: Patchification / Packing Order |
| OT-10 | Guidance embedding: blissful-tuner hardcodes 1.0, OneTrainer is configurable | WARNING | 6: Guidance Embedding |
| OT-11 | Position ID construction agrees | OK | 7: Position ID Construction |
| OT-12 | LoRA targeting scope agrees on transformer blocks, differs on granularity | INFO | 8: LoRA Targeting Scope |
| OT-13 | OneTrainer supports offset noise and perturbation noise; blissful-tuner does not | INFO | 1: Flow Matching Formula |
| OT-14 | OneTrainer supports multiple loss functions (MAE, Huber, log-cosh); blissful-tuner uses MSE only | INFO | 3: Loss Target Construction |
| OT-15 | OneTrainer has SIGMA loss weighting for flow matching; blissful-tuner has different weighting schemes | INFO | 3: Loss Target Construction |
| OT-16 | Blissful-tuner supports prior preservation loss; OneTrainer supports masked prior preservation | INFO | 3: Loss Target Construction |

---

## Finding OT-1: Flow Matching Noise Addition Formula Agrees

**Severity:** OK
**Domain:** 1: Flow Matching Formula

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py`, lines 36-37:
```python
scaled_noisy_latent_image = latent_noise.to(dtype=sigmas.dtype) * sigmas \
                            + scaled_latent_image.to(dtype=sigmas.dtype) * one_minus_sigmas
```
Formula: `noisy = noise * sigma + clean * (1 - sigma)`

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py`, line 1125:
```python
noisy_model_input = (1 - t) * latents + t * noise
```
Formula: `noisy = (1 - t) * clean + t * noise`

And for the `weighting_scheme` path, line 1147:
```python
noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
```
Formula: `noisy = sigma * noise + (1 - sigma) * clean`

**Discrepancy:** None. Both formulations are algebraically identical: `sigma * noise + (1 - sigma) * clean`. The two code paths in blissful-tuner (shift-based and weighting_scheme-based) both produce the same result.

**Recommendation:** None needed.

---

## Finding OT-2: Sigma Computation Differs in Indexing Convention

**Severity:** WARNING
**Domain:** 1: Flow Matching Formula

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py`, lines 22-24:
```python
all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, device=scaled_latent_image.device)
self.__sigma = all_timesteps / num_timesteps
```
This creates sigmas from `1/N` to `N/N = 1.0` (1-based). With `num_train_timesteps=1000`, sigma ranges from 0.001 to 1.0.

Timesteps are sampled as integers in `[0, num_train_timesteps)` (see `_get_timestep_discrete`, lines 141-142: `min_timestep = int(num_train_timesteps * config.min_noising_strength)`), then used as indices: `sigmas = self.__sigma[timestep]` (line 29). Since timesteps are 0-indexed, timestep 0 gives sigma = 1/1000 = 0.001, timestep 999 gives sigma = 1.0.

**What blissful-tuner does:**
For the `shift`/`flux2_shift`/`sigmoid` path (line 1123-1127):
```python
timesteps = t * 1000.0    # t is [0, 1]
timesteps += 1             # 1 to 1000
```
Then in `call_dit` (line 457): `timesteps = timesteps / 1000.0` -- giving values from 1/1000 to 1.0 passed to the transformer.

For the `weighting_scheme` path (line 1143-1146):
```python
timesteps = noise_scheduler.timesteps[indices].to(device=device)  # 1 to 1000
sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
```
Where `noise_scheduler.sigmas = torch.linspace(1, 0, num_train_timesteps + 1)` and `noise_scheduler.timesteps = (sigmas[:-1] * num_train_timesteps)` giving values from 1000 to 1.

**Discrepancy:** Both codebases use sigma in [0.001, 1.0] for FLUX.2, and both divide timesteps by 1000 before passing to the transformer. The indexing conventions differ (OneTrainer uses 0-based integer indexing into a precomputed sigma table; blissful-tuner uses continuous `t` values scaled to timesteps) but produce equivalent results.

However, there is a subtle difference: OneTrainer's minimum sigma is exactly `1/num_train_timesteps` while blissful-tuner's minimum depends on the sampling distribution and can in theory produce `t=0` which maps to timestep=0+1=1, then sigma=1/1000. These are equivalent for the default case. No correctness bug.

**Recommendation:** No action needed. The sigma ranges are equivalent.

---

## Finding OT-3: Latent Scaling Pipeline Is Mathematically Equivalent

**Severity:** OK
**Domain:** 2: Latent Scaling

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/model/Flux2Model.py`, lines 313-318:
```python
def scale_latents(self, latents: Tensor) -> Tensor:
    latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    return (latents - latents_bn_mean) / latents_bn_std
```
Called in `BaseFlux2Setup.predict()` (line 117): `scaled_latent_image = model.scale_latents(latent_image)` where `latent_image` is already patchified.

File: `/Users/dustin/OneTrainer/modules/model/Flux2Model.py`, lines 297-301 (patchify):
```python
def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents
```

Pipeline: raw VAE mean -> patchify (separate step) -> BN scale (separate step) -> noise addition.

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/flux_2/flux2_models.py`, lines 377-388:
```python
def encode(self, x: Tensor) -> Tensor:
    moments = self.encoder(x)
    mean = torch.chunk(moments, 2, dim=1)[0]
    z = rearrange(
        mean,
        "... c (i pi) (j pj)  -> ... (c pi pj) i j",
        pi=self.ps[0],
        pj=self.ps[1],
    )
    z = self.normalize(z)
    return z
```

Where `normalize` (lines 367-369):
```python
def normalize(self, z):
    self.bn.eval()
    return self.bn(z)
```

The `bn` is `BatchNorm2d` with `affine=False`, so `self.bn(z) = (z - running_mean) / sqrt(running_var + eps)` -- mathematically identical to OneTrainer's `scale_latents`.

The `rearrange("... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)` is mathematically identical to OneTrainer's `patchify_latents`.

Pipeline: encoder -> mean -> patchify+BN (fused in `encode()`) -> cached -> loaded -> noise addition.

`scale_shift_latents` is a no-op for FLUX.2 (line 371-372):
```python
def scale_shift_latents(latents):
    return latents
```

**Discrepancy:** None. Both apply the same patchification (2x2 pixel shuffle increasing channels 4x) and the same BN normalization (subtract running mean, divide by sqrt(running var + eps)). OneTrainer does them as separate steps at training time; blissful-tuner fuses them during caching. The mathematical result is identical.

**Recommendation:** None needed.

---

## Finding OT-4: Loss Target Formula Agrees

**Severity:** OK
**Domain:** 3: Loss Target Construction

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/BaseFlux2Setup.py`, line 166:
```python
flow = latent_noise - scaled_latent_image
```
Where `scaled_latent_image` is the patchified+BN-normalized latent, and `latent_noise` is the noise tensor (same shape as `scaled_latent_image`).

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/flux_2_train_network.py`, line 473:
```python
target = noise - latents
```
Where `latents` is the cached patchified+BN-normalized latent (passed through the no-op `scale_shift_latents`), and `noise` is the noise tensor.

**Discrepancy:** None. Both compute `target = noise - scaled_latent`. This is the standard rectified flow / flow matching velocity field target.

**Recommendation:** None needed.

---

## Finding OT-5: Loss Computed in Different Spatial Domains

**Severity:** INFO
**Domain:** 3: Loss Target Construction

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/BaseFlux2Setup.py`, lines 170-172:
```python
'predicted': model.unpatchify_latents(predicted_flow),
'target': model.unpatchify_latents(flow),
```
Then in `ModelSetupDiffusionLossMixin.__unmasked_losses()` (lines 150-155):
```python
losses += F.mse_loss(
    data['predicted'].to(dtype=torch.float32),
    data['target'].to(dtype=torch.float32),
    reduction='none'
).mean(mean_dim)
```
MSE loss is computed on **unpatchified** tensors: shape `(B, 32, H, W)` where H and W are the original latent spatial dims.

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/flux_2_train_network.py`, lines 469-473:
```python
model_pred = rearrange(model_pred, "b (h w) c -> b c h w", h=packed_latent_height, w=packed_latent_width)
target = noise - latents
```
Then in `hv_train_network.py` line 2491:
```python
loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
```
MSE loss is computed on **patchified** tensors: shape `(B, 128, H/2, W/2)`.

**Discrepancy:** OneTrainer unpatchifies before loss; blissful-tuner keeps patchified. Since unpatchification is a deterministic rearrangement of the same values (no scaling), the per-element MSE values are identical -- they're just arranged differently in the tensor. The mean of all elements is the same. This is NOT a correctness issue for unweighted loss.

However, this matters for **weighted loss** (see OT-6).

**Recommendation:** No action needed for unweighted training. See OT-6 for mask-weighted case.

---

## Finding OT-6: Mask-Weighted Loss Operates at Different Spatial Resolutions

**Severity:** WARNING
**Domain:** 3: Loss Target Construction

**What OneTrainer does:**
Masks are downscaled with factor 0.125 (= 1/8, matching VAE scale factor).
File: `/Users/dustin/OneTrainer/modules/dataLoader/Flux2BaseDataLoader.py`, line 40:
```python
downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.125)
```
This produces a mask at the unpatchified latent resolution: `(H/8, W/8)`. Loss is computed in unpatchified space `(B, 32, H/8, W/8)`, so the mask broadcasts correctly.

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/flux_2_cache_latents.py`, lines 96-98:
```python
lat_h, lat_w = target_latent.shape[-2:]  # target_latent is patchified
mask = F.interpolate(mask, size=(lat_h, lat_w), mode="area")  # (1, 1, lat_h, lat_w)
```
Since `target_latent` is already patchified (output of `ae.encode()`), `lat_h = H/16` and `lat_w = W/16`. The mask is downscaled to `(H/16, W/16)`, and loss is computed in patchified space `(B, 128, H/16, W/16)`, so the mask broadcasts correctly at the patchified resolution.

**Discrepancy:** blissful-tuner's masks operate at half the spatial resolution (H/16 x W/16) compared to OneTrainer (H/8 x W/8). For a 1024x1024 image:
- OneTrainer mask: 128x128 (16,384 spatial elements)
- blissful-tuner mask: 64x64 (4,096 spatial elements)

Each "pixel" in blissful-tuner's mask covers a 2x2 patch of latent pixels. Fine mask details smaller than 32x32 pixels in image space will be lost.

Both approaches are mathematically consistent within their respective loss domains (the total loss mean is equivalent). But blissful-tuner has lower spatial granularity for mask-weighted training.

**Recommendation:** This is a design choice rather than a bug. For most use cases (face LoRAs, subject isolation), the 64x64 mask resolution is sufficient. If pixel-precise mask weighting is important, consider:
1. Unpatchifying `model_pred` and `target` before loss computation (like OneTrainer), or
2. Keeping the current approach but documenting the 2x downsampling for users.

Priority: LOW. The current approach works correctly within its resolution.

---

## Finding OT-7: Timestep Division by 1000 Agrees

**Severity:** OK
**Domain:** 4: Timestep Normalization

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/BaseFlux2Setup.py`, line 151:
```python
timestep=timestep / 1000,
```

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/flux_2_train_network.py`, line 457:
```python
timesteps = timesteps / 1000.0
```

**Discrepancy:** None. Both normalize timesteps by dividing by 1000 before passing to the transformer.

**Recommendation:** None needed.

---

## Finding OT-8: Dynamic Timestep Shift Formula Differs Significantly

**Severity:** WARNING
**Domain:** 4: Timestep Normalization

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/model/Flux2Model.py`, lines 267-278:
```python
def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
    base_seq_len = self.noise_scheduler.config.base_image_seq_len
    max_seq_len = self.noise_scheduler.config.max_image_seq_len
    base_shift = self.noise_scheduler.config.base_shift
    max_shift = self.noise_scheduler.config.max_shift
    patch_size = 2

    image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return math.exp(mu)
```
Note: `latent_height` and `latent_width` are the **patchified** dims. The formula halves them again with `patch_size=2`, giving `image_seq_len = (latent_w/2) * (latent_h/2)`. The constants come from the noise scheduler config.

Then the shift is applied in `_get_timestep_discrete` (line 172):
```python
timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
```

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py`, lines 1019-1031:
```python
elif args.timestep_sampling == "flux2_shift":
    # FLUX.2 uses h*w (not halved) due to different latent dimensions
    mu = train_utils.get_lin_function(y1=0.5, y2=1.15)(h * w)
    ...
    shift = math.exp(mu)
    ...
    t = logits_norm.sigmoid()
    t = (t * shift) / (1 + (shift - 1) * t)
```
Where `get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15)` (default args) computes:
`mu = m * (h*w) + b` where `m = (1.15 - 0.5) / (4096 - 256)` and `b = 0.5 - m*256`.

Note: `h` and `w` here are the patchified dims (from `latents.shape[-2:]`), and it uses `h*w` directly (not halved).

**Discrepancy:** The formulas use different:
1. **Sequence length computation:** OneTrainer halves the patchified dims again (divides by patch_size=2), giving `seq_len = (h/2) * (w/2)`. Blissful-tuner uses the patchified dims directly: `seq_len = h * w`. For a 1024x1024 image with patchified latents at 64x64: OneTrainer gets `32*32 = 1024`; blissful-tuner gets `64*64 = 4096`.
2. **Linear interpolation anchors:** OneTrainer uses scheduler config values (`base_image_seq_len`, `max_image_seq_len`, `base_shift`, `max_shift`) -- these are inherited from FLUX.1 defaults and may not be optimal for FLUX.2. Blissful-tuner hardcodes `(256, 0.5)` to `(4096, 1.15)`.
3. **Shift application:** OneTrainer applies shift to the discrete timestep integer. Blissful-tuner applies shift to the continuous sigmoid-sampled `t` value. Both use the same `shift(t) = shift*t / ((shift-1)*t + 1)` formula but at different stages.

The comment in OneTrainer's code (line 264) is revealing: *"inference code uses empirical mu. But that code cannot be used for training because it depends on num of inference steps... the dynamic shifting parameters of the noise schedulers are probably just the default values (taken from Flux1) and not applicable - but the best values we have"*.

Blissful-tuner's separate `compute_empirical_mu` function (`flux2_utils.py:526-541`) is used only for inference, not training. The training shift uses the simpler linear function.

**Recommendation:** Both approaches are approximate. Neither is definitively "correct" -- FLUX.2's optimal training shift parameters aren't publicly documented. The blissful-tuner `flux2_shift` is likely more appropriate since it accounts for the actual patchified latent dimensions rather than halving them again. However, the OneTrainer approach of reading from the scheduler config allows automatic adaptation if BFL updates the config. Consider:
1. Adding a note in `docs/flux_2.md` that `--timestep_sampling flux2_shift` uses a custom shift formula.
2. Optionally supporting the scheduler-config-based shift as an alternative (e.g., `--timestep_sampling flux2_scheduler_shift`).

Priority: LOW (training quality difference is empirical and likely small).

---

## Finding OT-9: Patchification Order Is Equivalent Despite Different Staging

**Severity:** OK
**Domain:** 5: Patchification / Packing Order

**What OneTrainer does:**
```
raw_latent_mean (from VAE, cached)
  -> patchify (predict() line 114)
  -> BN scale (predict() line 117)
  -> add noise (predict() line 131)
  -> pack to sequence (predict() line 147)
  -> transformer (predict() line 149)
  -> unpack from sequence (predict() line 160)
  -> target = noise - scaled_latent (predict() line 166)
  -> unpatchify both predicted and target (predict() lines 171-172)
  -> MSE loss in unpatchified space
```

**What blissful-tuner does:**
```
raw image
  -> ae.encode() = encoder + mean + patchify + BN (cache time, flux2_models.py:377-388)
  -> save to cache
  -> load from cache = patchified+BN-scaled latent
  -> scale_shift_latents = no-op (flux_2_train_network.py:371)
  -> add noise (hv_train_network.py:1125 or 1147)
  -> pack to sequence via prc_img (flux_2_train_network.py:405)
  -> transformer (flux_2_train_network.py:458)
  -> unpack from sequence (flux_2_train_network.py:469)
  -> target = noise - latents (flux_2_train_network.py:473)
  -> MSE loss in patchified space (hv_train_network.py:2491)
```

**Discrepancy:** None in the mathematical pipeline. Both apply noise AFTER patchification and BN scaling. Both compute loss target as `noise - scaled_latent`. The only difference is that blissful-tuner keeps the final loss in patchified space while OneTrainer unpatchifies (see OT-5). The order of operations is equivalent.

**Recommendation:** None needed.

---

## Finding OT-10: Guidance Embedding Hardcoded to 1.0 in blissful-tuner Training

**Severity:** WARNING
**Domain:** 6: Guidance Embedding During Training

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/BaseFlux2Setup.py`, lines 139-143:
```python
if model.transformer.config.guidance_embeds:
    guidance = torch.tensor([config.transformer.guidance_scale], device=self.train_device, dtype=model.train_dtype.torch_dtype())
    guidance = guidance.expand(latent_input.shape[0])
else:
    guidance = None
```
The guidance value is **configurable** via `config.transformer.guidance_scale`. The default preset uses 4.0 for DEV. Only provided if the model actually has `guidance_embeds` (DEV has it, Klein does not).

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/flux_2_train_network.py`, line 449:
```python
# use 1.0 as guidance scale for FLUX.2 training
guidance_vec = torch.full((bsize,), 1.0, device=accelerator.device, dtype=network_dtype)
```
The guidance value is **hardcoded to 1.0** and is always provided regardless of the model variant.

The `Flux2.forward()` method (flux2_models.py:690) checks `self.use_guidance_embed` before using the guidance vector, so for Klein models the value is harmlessly ignored.

**Discrepancy:** For FLUX.2 DEV (the only variant with `use_guidance_embed=True`):
- OneTrainer passes a configurable guidance value (default 4.0 in their preset)
- blissful-tuner hardcodes 1.0

This means the model's guidance embedding layer sees `guidance=1.0` during LoRA training in blissful-tuner, regardless of what value is used during inference. If inference uses `guidance=4.0` (the default), there is a train/inference distribution mismatch.

For Klein models: no impact (guidance embedding is not used).

**Recommendation:** Make the training guidance value configurable via a CLI argument (e.g., `--training_guidance_scale`), defaulting to the model variant's default from `FLUX2_MODEL_INFO`. For DEV, this would default to 4.0 to match inference.

Alternatively, if 1.0 was intentionally chosen (e.g., to match guidance-free training convention), document this decision and its implications. Some LoRA training frameworks deliberately use 1.0 to make the LoRA "guidance-neutral".

Priority: MEDIUM. This affects DEV LoRA quality when inference guidance differs from 1.0.

---

## Finding OT-11: Position ID Construction Agrees

**Severity:** OK
**Domain:** 7: Position ID Construction

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/model/Flux2Model.py`, lines 240-251 (image IDs):
```python
@staticmethod
def prepare_latent_image_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape
    t = torch.arange(1, device=latents.device)
    h = torch.arange(height, device=latents.device)
    w = torch.arange(width, device=latents.device)
    l_ = torch.arange(1, device=latents.device)
    latent_ids = torch.cartesian_prod(t, h, w, l_)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids
```

Lines 281-294 (text IDs):
```python
@staticmethod
def prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    B, L, _ = x.shape
    out_ids = []
    for _ in range(B):
        t = torch.arange(1, device=x.device)
        h = torch.arange(1, device=x.device)
        w = torch.arange(1, device=x.device)
        l_ = torch.arange(L, device=x.device)
        coords = torch.cartesian_prod(t, h, w, l_)
        out_ids.append(coords)
    return torch.stack(out_ids)
```

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/flux_2/flux2_utils.py`, lines 392-415 (image IDs via `prc_img`):
```python
def prc_img(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    h = x.shape[-2]
    w = x.shape[-1]
    x_coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1),
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
```

Lines 333-354 (text IDs via `prc_txt`):
```python
def prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _l = x.shape[-2]
    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),
        "w": torch.arange(1),
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
```

**Discrepancy:** None. Both use the same 4D coordinate system `(t, h, w, l)`:
- Image IDs: `t=arange(1)=[0]`, `h=arange(H)`, `w=arange(W)`, `l=arange(1)=[0]`
- Text IDs: `t=arange(1)=[0]`, `h=arange(1)=[0]`, `w=arange(1)=[0]`, `l=arange(L)`

Both use `torch.cartesian_prod` to create the coordinate grid. Identical.

**Recommendation:** None needed.

---

## Finding OT-12: LoRA Targeting Scope Agrees on Block Types, Differs in Granularity

**Severity:** INFO
**Domain:** 8: LoRA Targeting Scope

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/Flux2LoRASetup.py`, lines 57-58:
```python
model.transformer_lora = LoRAModuleWrapper(
    model.transformer, "transformer", config, config.layer_filter.split(",")
)
```
With presets from `BaseFlux2Setup.LAYER_PRESETS`:
```python
LAYER_PRESETS = {
    "blocks": ["transformer_blocks"],  # Default
    "full": [],                         # All Linear/Conv2d layers
}
```
The filter `"transformer_blocks"` matches module names containing that substring. From `LoRAModuleWrapper.__create_modules()` (lines 648-653), it iterates ALL `Linear` and `Conv2d` layers and filters by name match.

OneTrainer targets ALL `Linear`/`Conv2d` layers inside `transformer_blocks` (double blocks) and `single_transformer_blocks` (single blocks), including attention, MLP, and projection layers. It does NOT exclude norm layers by name -- the filter is purely inclusion-based.

Modulation layers are NOT targeted because they live outside the blocks (on `Flux2Transformer2DModel` top level, not inside `transformer_blocks.*`).

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/networks/lora_flux_2.py`, lines 16, 39, 43-54:
```python
FLUX_2_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]
...
exclude_patterns.append(r".*(norm).*")
...
return lora.create_network(
    FLUX_2_TARGET_REPLACE_MODULES,
    "lora_unet",
    ...
)
```
Targets `DoubleStreamBlock` and `SingleStreamBlock` classes. Explicitly excludes `norm` layers via regex pattern. All `Linear` (and `Conv2d`) layers inside these module classes are targeted, except those matching `.*norm.*`.

**Discrepancy:**
1. **Norm exclusion:** blissful-tuner explicitly excludes norm layers (`QKNorm`, `LayerNorm`, `RMSNorm`). OneTrainer's `"blocks"` preset does NOT explicitly exclude norms -- they're included if they're `Linear` layers inside the target blocks. In practice, `QKNorm` contains `RMSNorm` which has a `scale` parameter but is NOT a `Linear`/`Conv2d` module, so it wouldn't be matched by OneTrainer either. `LayerNorm` is also not `Linear`. So this difference is cosmetic -- both end up targeting the same layers.

2. **No text encoder LoRA:** Both codebases only apply LoRA to the transformer. OneTrainer: only `model.transformer_lora` exists (no `text_encoder_lora`). blissful-tuner: `lora_flux_2.py` doesn't create text encoder modules. Consistent.

3. **Layer filter naming:** OneTrainer uses diffusers naming (`transformer_blocks`, `single_transformer_blocks`). blissful-tuner uses BFL naming (`DoubleStreamBlock`, `SingleStreamBlock`). Both cover the same architectural components.

**Recommendation:** None needed. The effective LoRA targeting scope is identical.

---

## Finding OT-13: OneTrainer Supports Offset Noise and Perturbation Noise

**Severity:** INFO
**Domain:** 1: Flow Matching Formula

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupNoiseMixin.py`, lines 92-117:
```python
if config.offset_noise_weight > 0:
    offset_noise = torch.randn(
        (source_tensor.shape[0], source_tensor.shape[1], *[1 for _ in range(source_tensor.ndim - 2)]),
        ...
    )
    noise = noise + (config.offset_noise_weight * offset_noise)

if config.perturbation_noise_weight > 0:
    perturbation_noise = torch.randn(source_tensor.shape, ...)
    noise = noise + (config.perturbation_noise_weight * perturbation_noise)
```
OneTrainer supports both offset noise (channel-wise constant noise for exposure/color shift) and perturbation noise (additional stochastic noise), plus a generalized offset noise variant from a paper.

**What blissful-tuner does:**
The noise is simply `torch.randn_like(latents)` (hv_train_network.py:2431). No offset noise or perturbation noise support.

**Discrepancy:** Feature gap -- blissful-tuner does not support offset noise or perturbation noise for FLUX.2 training.

**Recommendation:** Consider adding `--offset_noise_weight` support. Offset noise is well-studied and can improve contrast/color range in generated images. Priority: LOW (optional feature).

---

## Finding OT-14: OneTrainer Supports Multiple Loss Functions

**Severity:** INFO
**Domain:** 3: Loss Target Construction

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`, lines 139-191:
```python
# MSE/L2 Loss
if config.mse_strength != 0:
    losses += F.mse_loss(...) * config.mse_strength
# MAE/L1 Loss
if config.mae_strength != 0:
    losses += F.l1_loss(...) * config.mae_strength
# log-cosh Loss
if config.log_cosh_strength != 0:
    losses += self.__log_cosh_loss(...) * config.log_cosh_strength
# Huber Loss
if config.huber_strength != 0:
    losses += F.huber_loss(...) * config.huber_strength
```
Multiple loss functions with configurable weights, combinable.

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py`, line 2491:
```python
loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
```
MSE only.

**Discrepancy:** Feature gap. blissful-tuner only supports MSE loss. OneTrainer supports MSE, MAE, log-cosh, and Huber with configurable strengths.

**Recommendation:** MSE is the standard choice for flow matching. The other losses could be useful for specific use cases (Huber for outlier robustness, L1 for sharper results). Priority: LOW.

---

## Finding OT-15: Different Loss Weighting Schemes Available

**Severity:** INFO
**Domain:** 3: Loss Target Construction

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`, lines 334-341:
```python
match config.loss_weight_fn:
    case LossWeight.CONSTANT:
        pass
    case LossWeight.SIGMA:
        losses *= self.__sigma_loss_weight(data['timestep'], losses.device)
```
For flow matching models: CONSTANT (no weighting) or SIGMA (weight by sigma).

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py`, lines 350-374:
```python
def compute_loss_weighting_for_sd3(weighting_scheme, noise_scheduler, timesteps, device, dtype, n_dim=5):
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    elif weighting_scheme == "structure_bell":
        t = sigmas.to(torch.float32).clamp(0.0, 1.0)
        raw_weights = -2.4 * (t**2) + 3.0 * t + 0.5
        weighting = raw_weights * (5.0 / 6.0)
```
Three weighting schemes: `sigma_sqrt` (similar to OneTrainer's SIGMA but inverted), `cosmap`, and `structure_bell`.

**Discrepancy:** Different but complementary weighting options. OneTrainer's `SIGMA` weights by `sigma[t]` (higher weight for noisier timesteps). blissful-tuner's `sigma_sqrt` weights by `1/sigma^2` (higher weight for LESS noisy timesteps). These are inversely related.

**Recommendation:** Document which weighting scheme is most effective for FLUX.2 training. The `cosmap` and `structure_bell` options in blissful-tuner are more sophisticated than OneTrainer's options. Priority: LOW (documentation).

---

## Finding OT-16: Different Prior Preservation Approaches

**Severity:** INFO
**Domain:** 3: Loss Target Construction

**What OneTrainer does:**
File: `/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`, lines 47-62:
```python
losses += masked_losses_with_prior(
    losses=F.mse_loss(data['predicted'], data['target'], reduction='none'),
    prior_losses=F.mse_loss(data['predicted'], data['prior_target'], reduction='none') if 'prior_target' in data else None,
    mask=batch['latent_mask'],
    unmasked_weight=config.unmasked_weight,
    normalize_masked_area_loss=config.normalize_masked_area_loss,
    masked_prior_preservation_weight=config.masked_prior_preservation_weight,
)
```
Uses a separate `prior_target` tensor and `masked_losses_with_prior` function.

**What blissful-tuner does:**
File: `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py`, lines 2465-2517:
```python
if need_prior:
    with torch.no_grad():
        with self.prior_model_context(accelerator.unwrap_model(network)):
            prior_pred_raw, _ = self.call_dit(
                args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype,
            )
    prior_pred = prior_pred_raw.detach()
...
prior_loss_unreduced = torch.nn.functional.mse_loss(
    model_pred.to(network_dtype), prior_pred.to(network_dtype), reduction="none"
)
...
loss = apply_masked_loss_with_prior(
    loss, mask_weights, prior_loss_unreduced=prior_loss_unreduced, args=args, layout=layout,
)
```
blissful-tuner runs a full teacher forward pass with LoRA disabled to compute prior predictions, then uses `apply_masked_loss_with_prior` with gamma correction, min-weight floor, and per-sample normalization.

**Discrepancy:** Both support masked training with prior preservation, but the implementations differ in sophistication. blissful-tuner's mask loss module (`modules/mask_loss.py`) supports gamma correction, min-weight floor, threshold-based prior masking, and per-sample normalization. OneTrainer's approach is simpler (flat mask + unmasked_weight + prior weight).

**Recommendation:** No action needed. blissful-tuner's mask loss system is more feature-rich.

---

## Cross-Cutting Summary

### What OneTrainer Does Better
1. **Configurable guidance during training (OT-10):** OneTrainer allows setting the training guidance scale to match inference, avoiding train/inference distribution mismatch for DEV.
2. **Offset noise support (OT-13):** Useful for improving color/contrast range.
3. **Multiple loss functions (OT-14):** Huber, L1, log-cosh alongside MSE.

### What blissful-tuner Does Better
1. **Richer timestep sampling (OT-8):** More sampling strategies (sigmoid, logsnr, qinglong, etc.).
2. **More loss weighting options (OT-15):** cosmap, structure_bell, sigma_sqrt.
3. **Sophisticated mask loss (OT-16):** Gamma correction, min-weight floor, per-sample normalization, threshold-based prior.
4. **Extensive audit coverage:** The prior audit (v4.2) addressed 37 findings including several critical bugs.

### What Agrees
1. **Flow matching formula (OT-1):** Identical.
2. **Latent scaling (OT-3):** Mathematically equivalent despite different staging.
3. **Loss target (OT-4):** Both use `noise - scaled_latent`.
4. **Timestep normalization (OT-7):** Both divide by 1000.
5. **Patchification order (OT-9):** Equivalent.
6. **Position IDs (OT-11):** Identical.
7. **LoRA targeting (OT-12):** Same effective scope.

### Action Items (ordered by priority)

| Priority | Finding | Action |
|----------|---------|--------|
| MEDIUM | OT-10 | Make training guidance scale configurable (default to variant default, e.g., 4.0 for DEV) |
| LOW | OT-6 | Document that mask resolution is 2x lower than unpatchified due to patchified-space loss |
| LOW | OT-8 | Document `flux2_shift` formula differences vs scheduler-config-based shift |
| LOW | OT-13 | Consider adding offset noise support |
| LOW | OT-14 | Consider adding alternative loss functions |
