# SimpleTuner vs blissful-tuner: WAN (T2V/I2V) LoRA Training Pipeline Comparison

> **Date**: 2026-02-18
> **Scope**: WAN 2.1 T2V/I2V + WAN 2.2 T2V/I2V LoRA training pipeline
> **Method**: Source code comparison against SimpleTuner commit at `/Users/dustin/SimpleTuner/`

---

## Summary Table

| ID | Title | Severity | Domain |
|----|-------|----------|--------|
| ST-WAN-01 | Flow matching noise formula agrees | OK | 1. Flow Matching |
| ST-WAN-02 | Flow matching loss target agrees | OK | 1. Flow Matching |
| ST-WAN-03 | Timestep range differs: 0-1000 vs 1-1000 | WARNING | 1. Flow Matching |
| ST-WAN-04 | VAE per-channel normalization agrees | OK | 2. VAE Normalization |
| ST-WAN-05 | VAE patchify workaround only in SimpleTuner | INFO | 2. VAE Normalization |
| ST-WAN-06 | WAN 2.2 boundary values differ: 0.90 vs 0.875/0.900 | WARNING | 3. Multi-Stage |
| ST-WAN-07 | Multi-stage training architecture agrees conceptually | OK | 3. Multi-Stage |
| ST-WAN-08 | SimpleTuner trains one stage at a time; blissful-tuner trains both with rejection sampling | INFO | 3. Multi-Stage |
| ST-WAN-09 | Shift formula agrees for "shift" sampling mode | OK | 4. Timestep Sampling |
| ST-WAN-10 | Default sampling strategies differ: sigmoid vs sigma/shift | INFO | 4. Timestep Sampling |
| ST-WAN-11 | Frame count constraint: frames%8==1 vs T=4k+1 | WARNING | 4. Timestep Sampling |
| ST-WAN-12 | Loss computation approaches differ in reduction path | INFO | 5. Loss Computation |
| ST-WAN-13 | 5D sigma expansion agrees | OK | 5. Loss Computation |
| ST-WAN-14 | blissful-tuner has richer mask-weighted + prior loss | INFO | 5. Loss Computation |
| ST-WAN-15 | UMT5 text encoder agrees at 512 tokens max | OK | 6. Text Encoding |
| ST-WAN-16 | Variable-length text embedding handling agrees | OK | 6. Text Encoding |
| ST-WAN-17 | SimpleTuner applies prompt cleaning; blissful-tuner relies on tokenizer | INFO | 6. Text Encoding |
| ST-WAN-18 | LoRA target scope differs: attention projections vs entire WanAttentionBlock | WARNING | 7. LoRA Targeting |
| ST-WAN-19 | I2V conditioning approaches agree for both 2.1 and 2.2 styles | OK | 8. I2V Conditioning |
| ST-WAN-20 | blissful-tuner supports more I2V variants (FLF2V, Fun-Control, one-frame) | INFO | 8. I2V Conditioning |

---

## Domain 1: Flow Matching Formula

### Finding ST-WAN-01: Flow matching noise formula agrees
**Severity:** OK
**Domain:** 1. Flow Matching
**What SimpleTuner does:**
```python
# simpletuner/helpers/models/common.py:4374
batch["noisy_latents"] = (1 - batch["sigmas"]) * batch["latents"] + batch["sigmas"] * batch["input_noise"]
```
Sigmas are expanded to match latent dimensionality (5D for video) before this formula via `expand_sigmas()`.

**What blissful-tuner does:**
```python
# src/musubi_tuner/hv_train_network.py:1124-1125
t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
noisy_model_input = (1 - t) * latents + t * noise
```
The 5D reshaping of `t` is explicit in the code for both the primary "shift" path and the alternative "weighting_scheme" path (line 1147: `noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents`).

**Discrepancy:** None. Both implement `x_t = (1-sigma)*x_0 + sigma*noise` with correct 5D broadcasting.
**Recommendation:** None.

---

### Finding ST-WAN-02: Flow matching loss target agrees
**Severity:** OK
**Domain:** 1. Flow Matching
**What SimpleTuner does:**
```python
# simpletuner/helpers/models/common.py:3516
target = prepared_batch["noise"] - prepared_batch["latents"]
```

**What blissful-tuner does:**
```python
# src/musubi_tuner/wan_train_network.py:851
target = noise - latents
```

**Discrepancy:** None. Both use `v = noise - latents` as the velocity target for flow matching.
**Recommendation:** None.

---

### Finding ST-WAN-03: Timestep range differs: 0-1000 vs 1-1000
**Severity:** WARNING
**Domain:** 1. Flow Matching
**What SimpleTuner does:**
```python
# simpletuner/helpers/models/common.py:3710
timesteps = sigmas * 1000.0  # Range: [0, 1000]
```
Sigmas are in [0, 1] range. Timesteps are passed directly to the transformer.

**What blissful-tuner does:**
```python
# src/musubi_tuner/hv_train_network.py:1123,1127
timesteps = t * 1000.0
timesteps += 1  # 1 to 1000
```
For the primary "shift" sampling path, blissful-tuner adds 1 to shift the range from [0, 1000] to [1, 1000]. This avoids a timestep of exactly 0 (pure signal, no noise).

For the secondary "weighting_scheme" path (line 1143), timesteps come from the scheduler's `timesteps` array which is `sigmas[:-1] * num_train_timesteps` (line 81 of `scheduling_flow_match_discrete.py`), giving range approximately [1, 1000].

**Discrepancy:** blissful-tuner's primary sampling path shifts timesteps by +1 to avoid t=0. SimpleTuner does not, meaning t=0 is theoretically possible (though extremely rare due to sigmoid sampling). This difference is unlikely to cause training issues in practice since t=0 events are vanishingly rare, but the formulations diverge slightly in the boundary behavior.
**Recommendation:** Not a correctness bug. The +1 offset is a reasonable design choice to avoid degenerate zero-noise timesteps.

---

## Domain 2: VAE Latent Normalization

### Finding ST-WAN-04: VAE per-channel normalization agrees
**Severity:** OK
**Domain:** 2. VAE Normalization
**What SimpleTuner does:**
Uses diffusers' `AutoencoderKLWan` which stores `latents_mean` and `latents_std` in its config. The normalization is applied:
- During **I2V conditioning**: explicitly in `add_first_frame_conditioning()` (model.py:115-119):
  ```python
  latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
  latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
  latent_condition = (latent_condition - latents_mean) * latents_std
  ```
- During **VAE decode** (pipeline.py:725-728): inverse normalization `latents / latents_std + latents_mean`
- During **training latent caching**: delegated to the diffusers AutoencoderKLWan's internal encode pipeline.

**What blissful-tuner does:**
Uses a custom `WanVAE` class (src/musubi_tuner/wan/modules/vae.py:658-762) that applies normalization explicitly inside `encode()`:
```python
# vae.py:700-702
self.mean = torch.tensor(mean, dtype=dtype, device=device)  # 16 per-channel values
self.std = torch.tensor(std, dtype=dtype, device=device)
self.scale = [self.mean, 1.0 / self.std]

# vae.py:564-567 (inside WanVAE_.encode)
mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
```

The 16 per-channel mean/std values are hardcoded and match the official WAN VAE checkpoint values.

**Discrepancy:** None. Both apply identical per-channel `(latent - mean) * (1/std)` normalization, just via different mechanisms (diffusers built-in vs explicit). The 16 mean/std values are identical.
**Recommendation:** None.

---

### Finding ST-WAN-05: VAE patchify workaround only in SimpleTuner
**Severity:** INFO
**Domain:** 2. VAE Normalization
**What SimpleTuner does:**
Implements a patchify workaround in `_wan_prepare_vae_encode_inputs()` (model.py:625-692) to handle diffusers' `AutoencoderKLWan` having an internal `patch_size` parameter. Pre-patchifies the input and disables internal patchification to avoid double-patchification.

**What blissful-tuner does:**
Uses the original WAN VAE implementation directly (not diffusers), so no patchification workaround is needed. The custom `WanVAE_` class processes raw video inputs directly.

**Discrepancy:** Implementation detail only. blissful-tuner's approach is simpler because it uses the original WAN codebase's VAE rather than the diffusers port, eliminating the need for patchification workarounds.
**Recommendation:** None.

---

## Domain 3: WAN 2.2 Multi-Stage Architecture

### Finding ST-WAN-06: WAN 2.2 boundary values differ for T2V
**Severity:** WARNING
**Domain:** 3. Multi-Stage Architecture
**What SimpleTuner does:**
Uses `boundary_ratio: 0.90` for both I2V-2.2-high and I2V-2.2-low (model.py:261,270). SimpleTuner only supports WAN 2.2 I2V multi-stage (no T2V multi-stage flavour):
```python
# simpletuner/helpers/models/wan/model.py:254-291
WAN_STAGE_OVERRIDES = {
    "i2v-14b-2.2-high": {"boundary_ratio": 0.90, ...},
    "i2v-14b-2.2-low": {"boundary_ratio": 0.90, ...},
}
```

**What blissful-tuner does:**
Supports both T2V and I2V multi-stage with different boundary values:
```python
# src/musubi_tuner/wan/configs/wan_t2v_A14B.py:41
t2v_A14B.boundary = 0.875

# src/musubi_tuner/wan/configs/wan_i2v_A14B.py:41
i2v_A14B.boundary = 0.900
```
These match the official Alibaba WAN 2.2 paper recommendations: T2V uses 0.875 boundary, I2V uses 0.900 boundary.

**Discrepancy:** SimpleTuner only provides I2V multi-stage (0.90). blissful-tuner provides both T2V (0.875) and I2V (0.900) with task-specific defaults. The boundary values for I2V agree. SimpleTuner does not offer WAN 2.2 T2V multi-stage training.
**Recommendation:** No action needed for blissful-tuner. blissful-tuner is more complete here.

---

### Finding ST-WAN-07: Multi-stage training architecture agrees conceptually
**Severity:** OK
**Domain:** 3. Multi-Stage Architecture
**What SimpleTuner does:**
Trains one stage at a time via `model_flavour` selection (e.g., `"i2v-14b-2.2-high"` or `"i2v-14b-2.2-low"`). During validation, loads the complementary stage model for full-pipeline inference.

**What blissful-tuner does:**
Loads both models upfront. Uses rejection sampling per-step to select timesteps that fall within the active expert's range. Swaps the entire state dict when the expert changes.

**Discrepancy:** Different design philosophies but same end result: one expert is trained per gradient step, conditioned on the timestep boundary.
**Recommendation:** None.

---

### Finding ST-WAN-08: SimpleTuner trains one stage at a time; blissful-tuner trains both with rejection sampling
**Severity:** INFO
**Domain:** 3. Multi-Stage Architecture
**What SimpleTuner does:**
The user selects which stage to train via `model_flavour` (e.g., `"i2v-14b-2.2-high"`). Only one transformer is loaded for training. Timestep sampling is not constrained to the active expert's range during training.

**What blissful-tuner does:**
Loads both high and low noise models. Uses rejection sampling in `get_noisy_model_input_and_timesteps()` (wan_train_network.py:627-684) to ensure all batch elements land in the correct expert's timestep range. Swaps weights on every step if the expert needs to change.

```python
# wan_train_network.py:647-648
high_noise = sample_timesteps[0] / 1000.0 >= self.timestep_boundary
self.next_model_is_high_noise = high_noise

# wan_train_network.py:659-671  (rejection sampling loop)
for i in range(bsize):
    for _ in range(num_max_calls):
        ...
        if (high_noise and ts_i[0] / 1000.0 >= self.timestep_boundary) or (
            not high_noise and ts_i[0] / 1000.0 < self.timestep_boundary
        ):
            ...
            break
```

**Discrepancy:** blissful-tuner's approach is more complex but allows training both experts in a single run. SimpleTuner requires two separate training runs. The rejection sampling in blissful-tuner may be inefficient for the high-noise expert (~8 retries on average for a 0.875 boundary = 12.5% acceptance), but is correct.
**Recommendation:** Informational only. Both approaches are valid. blissful-tuner's single-run approach is more convenient but uses more VRAM (two models loaded).

---

## Domain 4: Timestep Sampling & Shift

### Finding ST-WAN-09: Shift formula agrees for "shift" sampling mode
**Severity:** OK
**Domain:** 4. Timestep Sampling
**What SimpleTuner does:**
```python
# simpletuner/helpers/training/custom_schedule.py:477
sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
```
Where `shift` comes from `flow_schedule_shift` config option.

**What blissful-tuner does:**
```python
# src/musubi_tuner/hv_train_network.py:1028-1031
logits_norm = randn(batch_size, org_timesteps)
logits_norm = logits_norm * args.sigmoid_scale
t = logits_norm.sigmoid()
t = (t * shift) / (1 + (shift - 1) * t)
```
Where `shift = args.discrete_flow_shift`.

**Discrepancy:** None. Both use the identical shift formula `sigma_shifted = (sigma * s) / (1 + (s-1) * sigma)`.
**Recommendation:** None.

---

### Finding ST-WAN-10: Default sampling strategies differ
**Severity:** INFO
**Domain:** 4. Timestep Sampling
**What SimpleTuner does:**
Default is sigmoid sampling with a flow_schedule_shift applied:
```python
# common.py:3693
sigmas = torch.sigmoid(self.config.flow_sigmoid_scale * torch.randn(...))
sigmas = apply_flow_schedule_shift(self.config, self.noise_schedule, sigmas, batch["noise"])
```
Typical WAN config uses `flow_schedule_shift: 3`.

**What blissful-tuner does:**
Default timestep_sampling is `"sigma"` (uniform from scheduler), but recommends `"shift"` for WAN 2.2:
```python
# wan_train_network.py:63-64
if args.task == "t2v-A14B":
    recommended_flow_shift = 12.0
elif args.task == "i2v-A14B":
    recommended_flow_shift = 5.0
```
These match the official WAN 2.2 recommended shift values.

**Discrepancy:** Different default strategies. SimpleTuner defaults to sigmoid + shift=3. blissful-tuner defaults to sigma/1.0 but strongly recommends shift=12.0 for T2V-A14B and shift=5.0 for I2V-A14B (with explicit warnings if using sigma sampling for WAN 2.2). The recommended shift values for WAN 2.2 are much larger than SimpleTuner's default of 3.
**Recommendation:** Informational. Users should follow the per-task recommendations. blissful-tuner's warnings for WAN 2.2 are a good safety feature.

---

### Finding ST-WAN-11: Frame count constraint differences
**Severity:** WARNING
**Domain:** 4. Timestep Sampling
**What SimpleTuner does:**
```python
# simpletuner/helpers/models/wan/model.py:307-313
@classmethod
def adjust_video_frames(cls, num_frames: int) -> int:
    """Adjust frame count to satisfy frames % 8 == 1 constraint."""
    if num_frames % 8 == 1:
        return num_frames
    adjusted = ((num_frames - 1) // 8) * 8 + 1
    return max(adjusted, 1)
```
Valid frame counts: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81...

**What blissful-tuner does:**
```python
# src/musubi_tuner/wan_cache_latents.py:63-78
# LC-01: Validate T=4k+1 frame count constraint for WAN VAE temporal compression.
num_frames = contents.shape[2]
if num_frames > 1 and (num_frames - 1) % 4 != 0:
    ...
    raise ValueError(f"Video frame count {num_frames} does not satisfy the T=4k+1 constraint...")
```
Valid frame counts: 1, 5, 9, 13, 17, 21, 25, ..., 77, 81, 85...

**Discrepancy:** SimpleTuner enforces `frames % 8 == 1` (stride 8), while blissful-tuner enforces `(frames - 1) % 4 == 0` (stride 4, i.e., `T=4k+1`). The WAN VAE has `temperal_downsample=[False, True, True]` meaning temporal compression factor is `2*2=4`, not 8. blissful-tuner's T=4k+1 constraint is MORE PERMISSIVE and CORRECT according to the actual VAE architecture. SimpleTuner's stricter constraint rejects valid frame counts like 5, 13, 21, 29, etc.

The VAE `vae_stride` is `(4, 8, 8)` in both codebases, confirming temporal stride of 4.

**Recommendation:** SimpleTuner's frame constraint is overly restrictive. Frame counts like 5, 13, 21, etc. should be valid. blissful-tuner's constraint is correct. However, the standard recommended frame count (81) satisfies both constraints, so this primarily affects users wanting shorter training clips.

---

## Domain 5: Loss Computation

### Finding ST-WAN-12: Loss computation approaches differ in reduction path
**Severity:** INFO
**Domain:** 5. Loss Computation
**What SimpleTuner does:**
```python
# common.py:4625-4627
# Flow matching always uses L2 loss
loss = (model_pred.float() - target.float()) ** 2
```
Then applies various optional weightings (ReflexFlow, scheduled sampling, etc.) before final reduction. The base flow matching path computes element-wise MSE then applies mean reduction.

**What blissful-tuner does:**
```python
# hv_train_network.py:2491
loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
```
Then applies optional `weighting` (sigma-based), followed by `apply_masked_loss_with_prior()` which either does a simple `.mean()` or the full mask-weighted normalization pipeline.

**Discrepancy:** Both compute MSE loss on the velocity target `(noise - latents)`. The reduction differs: SimpleTuner does `(pred - target)^2` then mean; blissful-tuner uses `F.mse_loss(..., reduction="none")` then applies mask weighting / prior loss before reducing. When no mask is used, blissful-tuner's path is equivalent to SimpleTuner's.

Note: blissful-tuner computes loss in `network_dtype` (typically bf16/fp16), while SimpleTuner explicitly upcasts to float32. This may cause minor numerical differences.
**Recommendation:** Consider upcasting loss computation to float32 in blissful-tuner for better numerical stability, similar to SimpleTuner's approach. However, the practical impact is likely negligible.

---

### Finding ST-WAN-13: 5D sigma expansion agrees
**Severity:** OK
**Domain:** 5. Loss Computation
**What SimpleTuner does:**
```python
# common.py:5260-5262 (VideoModelFoundation.expand_sigmas)
def expand_sigmas(self, batch):
    if len(batch["latents"].shape) == 5:  # Video: [B, C, T, H, W]
        batch["sigmas"] = batch["sigmas"].reshape(batch["latents"].shape[0], 1, 1, 1, 1)
```

**What blissful-tuner does:**
```python
# hv_train_network.py:1124
t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
```

**Discrepancy:** None. Both correctly expand sigmas to `(B, 1, 1, 1, 1)` for 5D video tensors.
**Recommendation:** None.

---

### Finding ST-WAN-14: blissful-tuner has richer mask-weighted + prior loss
**Severity:** INFO
**Domain:** 5. Loss Computation
**What SimpleTuner does:**
Has TREAD token routing for masked training (model.py:1202-1236), where force_keep_mask prevents dropping masked tokens. Loss masking is done via conditioning_pixel_values at the token level.

**What blissful-tuner does:**
Has a comprehensive mask loss system (`modules/mask_loss.py`) with:
- Per-channel gamma correction
- Minimum weight floor
- Weighted-mean normalization
- Prior preservation (teacher forward with LoRA disabled)
- Per-sample normalization
- Threshold mode for prior mask

The mask is applied in latent space during loss computation, not at the token/attention level.

**Discrepancy:** Different masking paradigms. SimpleTuner uses attention-level token routing (TREAD). blissful-tuner uses loss-level spatial masking. Both are valid approaches with different trade-offs (TREAD saves compute, loss masking is more fine-grained spatially).
**Recommendation:** Informational. These are complementary approaches. blissful-tuner's prior preservation system is a unique feature not present in SimpleTuner.

---

## Domain 6: Text Encoding -- UMT5

### Finding ST-WAN-15: UMT5 text encoder agrees at 512 tokens max
**Severity:** OK
**Domain:** 6. Text Encoding
**What SimpleTuner does:**
```python
# wan/pipeline.py:158-203
max_sequence_length=512
```
Uses diffusers' `UMT5EncoderModel` with `T5TokenizerFast`, max 512 tokens.

**What blissful-tuner does:**
```python
# wan/configs/shared_config.py:12
wan_shared_cfg.text_len = 512

# wan/modules/t5.py:520
self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, ...)
```
Uses custom T5 encoder implementation with the same UMT5-XXL architecture, max 512 tokens.

**Discrepancy:** None. Both use 512 tokens max for WAN text encoding.
**Recommendation:** None.

---

### Finding ST-WAN-16: Variable-length text embedding handling agrees
**Severity:** OK
**Domain:** 6. Text Encoding
**What SimpleTuner does:**
```python
# wan/pipeline.py:413-418
prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
# Re-pad to max_sequence_length for batching
prompt_embeds = torch.stack([
    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
    for u in prompt_embeds
], dim=0)
```

**What blissful-tuner does:**
```python
# wan/modules/t5.py:526-528
seq_lens = mask.gt(0).sum(dim=1).long()
context = self.model(ids, mask)
return [u[:v] for u, v in zip(context, seq_lens)]
```
Variable-length embeddings are stored per-prompt and zero-padded when batched in the transformer forward (model.py:1093):
```python
context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
```

**Discrepancy:** None. Both strip padding to actual sequence length, then re-pad when batching for transformer input.
**Recommendation:** None.

---

### Finding ST-WAN-17: SimpleTuner applies prompt cleaning; blissful-tuner relies on tokenizer
**Severity:** INFO
**Domain:** 6. Text Encoding
**What SimpleTuner does:**
```python
# wan/pipeline.py:399
prompt = [prompt_clean(p) for p in prompt]
```
Where `prompt_clean` uses `ftfy.fix_text()`, HTML unescaping, and whitespace normalization.

**What blissful-tuner does:**
```python
# wan/modules/t5.py:520
self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean="whitespace", ...)
```
Uses `clean="whitespace"` in the tokenizer, which applies basic whitespace normalization but not ftfy or HTML unescaping.

**Discrepancy:** SimpleTuner has more aggressive prompt cleaning (ftfy for unicode normalization, HTML entity decoding). blissful-tuner only does whitespace normalization. This could cause minor differences when prompts contain Unicode edge cases or HTML entities, but is unlikely to affect normal text prompts.
**Recommendation:** Consider adding ftfy-based prompt cleaning to blissful-tuner's text encoder pipeline for parity, especially for multilingual prompts. Low priority.

---

## Domain 7: LoRA Targeting

### Finding ST-WAN-18: LoRA target scope differs: attention projections vs entire WanAttentionBlock
**Severity:** WARNING
**Domain:** 7. LoRA Targeting
**What SimpleTuner does:**
```python
# wan/model.py:222-225
DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
```
Targets only the 4 attention projection layers (Q, K, V, output) within each WanTransformerBlock. This uses diffusers/PEFT naming conventions.

**What blissful-tuner does:**
```python
# networks/lora_wan.py:13
WAN_TARGET_REPLACE_MODULES = ["WanAttentionBlock"]
```
With exclude patterns:
```python
# networks/lora_wan.py:34
exclude_patterns.append(r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*")
```
This targets the entire `WanAttentionBlock` class, which includes the cross-attention projections AND the feed-forward network. The exclude pattern removes embedding, norm, and projection head layers.

**Discrepancy:** Significant scope difference. blissful-tuner applies LoRA to ALL Linear layers within `WanAttentionBlock` (which includes self-attention Q/K/V/O, cross-attention Q/K/V/O, AND the feed-forward network), minus excluded patterns. SimpleTuner only applies LoRA to `to_k`, `to_q`, `to_v`, `to_out.0` (self-attention projections only, possibly cross-attention as well depending on the diffusers model structure).

Looking at the WAN model architecture in blissful-tuner (model.py), `WanAttentionBlock` contains:
1. Self-attention: `self_attn` with Q/K/V/O projections
2. Cross-attention: `cross_attn` with Q/K/V/O projections (including `k_img`, `v_img` for I2V)
3. Feed-forward: `ffn` with gate, fc1, fc2 layers
4. AdaLN modulation: `modulation` (excluded by pattern)

So blissful-tuner applies LoRA to approximately 10-11 Linear layers per block vs SimpleTuner's 4. This gives blissful-tuner more expressiveness per rank but requires proportionally more VRAM.

**Recommendation:** This is a known design choice, not a bug. blissful-tuner's broader targeting may achieve better quality at the same rank but costs more VRAM. Users should be aware that rank-for-rank, the total parameter count differs significantly between the two trainers. A rank-32 LoRA in blissful-tuner is roughly equivalent to a rank-80+ LoRA in SimpleTuner in total parameter count.

---

## Domain 8: I2V Conditioning

### Finding ST-WAN-19: I2V conditioning approaches agree for both 2.1 and 2.2 styles
**Severity:** OK
**Domain:** 8. I2V Conditioning
**What SimpleTuner does:**
- **WAN 2.1 I2V** (model.py:73-146): Channel concatenation approach. Encodes first frame via VAE, creates temporal mask (first frame = 1, rest = 0), concatenates `[noisy_latents, mask, image_latent]` along channel dimension. Result: 33 channels (16 + 1*4 + 16 = 36 in practice, or 16 + 17).
- **WAN 2.2 I2V** (model.py:150-206): Latent overwrite approach. Overwrites first latent frame with encoded image, creates binary mask (0=conditioned, 1=free). Optional last frame conditioning for FLF2V.

**What blissful-tuner does:**
- **WAN 2.1 I2V**: During cache (wan_cache_latents.py:108-143): Creates 4-channel temporal mask + 16-channel image latent = 20-channel conditioning tensor (`y`). During training (wan_train_network.py:805-810): `image_latents = batch["latents_image"]` concatenated with noisy input.
- **WAN 2.2 I2V**: During inference (wan_train_network.py:370-421): Same latent overwrite + mask approach. No CLIP needed for 2.2 (handled correctly at line 203, 808).

**Discrepancy:** None functionally. Both implement the same two I2V conditioning styles. The mask format and concatenation order are compatible.
**Recommendation:** None.

---

### Finding ST-WAN-20: blissful-tuner supports more I2V variants
**Severity:** INFO
**Domain:** 8. I2V Conditioning
**What SimpleTuner does:**
Supports T2V, I2V (2.1 and 2.2), FLF2V, TI2V (text-image-to-video with expand timesteps), and S2V (speech-to-video). Uses diffusers pipeline classes.

**What blissful-tuner does:**
Supports T2V, I2V (2.1 and 2.2), FLF2V, Fun-Control models (T2V-FC, I2V-FC), and one-frame inference mode. Has explicit task configs for each variant:
```python
# wan/configs/__init__.py:43-56
WAN_CONFIGS = {
    "t2v-14B": t2v_14B,
    "t2v-1.3B": t2v_1_3B,
    "i2v-14B": i2v_14B,
    "t2i-14B": t2i_14B,
    "flf2v-14B": flf2v_14B,
    "t2v-1.3B-FC": t2v_1_3B_FC,
    "t2v-14B-FC": t2v_14B_FC,
    "i2v-14B-FC": i2v_14B_FC,
    "i2v-A14B": i2v_A14B,
    "t2v-A14B": t2v_A14B,
}
```

**Discrepancy:** Different variant coverage. SimpleTuner supports TI2V-5B and S2V which blissful-tuner does not. blissful-tuner supports Fun-Control models and one-frame mode which SimpleTuner does not. Both support the core T2V and I2V variants.
**Recommendation:** Informational. Different feature focus areas.

---

## Cross-Cutting Observations

### Numerical Precision
- SimpleTuner explicitly upcasts loss to float32 (`model_pred.float() - target.float()`).
- blissful-tuner computes loss in `network_dtype` (typically bf16): `torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")`.
- This is unlikely to cause significant training differences but could affect gradient precision in edge cases.

### Architecture Differences
- SimpleTuner uses diffusers' `WanTransformer3DModel` (rewritten to use diffusers conventions).
- blissful-tuner uses the original Alibaba WAN model code (`WanModel`), which takes list inputs and returns list outputs (`model_pred = torch.stack(model_pred, dim=0)` at line 848).
- Both implement equivalent 3D RoPE, AdaLN, and cross-attention mechanisms.

### Memory Optimization
- Both support block swapping (up to 39 of 40 blocks).
- Both support FP8 quantization of the transformer.
- Both support gradient checkpointing.
- blissful-tuner additionally supports `--offload_inactive_dit` for dual-expert WAN 2.2 training.
- SimpleTuner additionally supports feed-forward chunking and group offloading.

### Validation/Inference
- SimpleTuner uses a custom `WanPipeline` class for inference with Skip-Layer Guidance.
- blissful-tuner uses a manual denoising loop with `FlowUniPCMultistepScheduler`.
- Both support CFG during inference/validation.
