# OneTrainer vs blissful-tuner: Qwen-Image LoRA Pipeline Comparison

**Date**: 2026-02-18
**Status**: Complete
**Methodology**: Source-to-source comparison of actual code in both codebases

## Summary Table

| Finding | Severity | Domain | Title |
|---------|----------|--------|-------|
| OT-QI-1 | OK | 1. Flow Matching Formula | Noise addition formula matches |
| OT-QI-2 | OK | 2. Latent Scaling | Scaling produces identical results via different timing |
| OT-QI-3 | OK | 3. Loss Target | Both compute `noise - scaled_latent` in unpacked space |
| OT-QI-4 | OK | 4. Timestep Normalization | Division by 1000 and shift formulas match |
| OT-QI-5 | OK | 5. Latent Packing Order | Order matches: scale -> noise -> pack -> transformer |
| OT-QI-6 | WARNING | 6. Text Encoding | Template token crop is masked-extraction vs multiplication |
| OT-QI-7 | INFO | 6. Text Encoding | blissful-tuner allows 1024 user tokens vs OneTrainer's 512 |
| OT-QI-8 | WARNING | 6. Text Encoding | Sequence padding to multiple of 16 missing in blissful-tuner training |
| OT-QI-9 | INFO | 7. LoRA Targeting | blissful-tuner lacks text encoder LoRA; comparable transformer scope |
| OT-QI-10 | INFO | 7. LoRA Targeting | OneTrainer has layer presets; blissful-tuner has exclude_patterns |
| OT-QI-11 | OK | 8. Edit/Control | Prior audit findings T1/C1 consistent with OneTrainer's approach |
| OT-QI-12 | WARNING | 3. Loss Computation | OneTrainer supports MAE/Huber/Log-cosh; blissful-tuner MSE-only |
| OT-QI-13 | INFO | 1. Flow Matching | OneTrainer supports offset noise + perturbation noise; blissful-tuner does not |
| OT-QI-14 | WARNING | 4. Timestep Distribution | OneTrainer +1 offset on sigma index; blissful-tuner uses continuous [0,1] |

---

## Finding OT-QI-1: Noise Addition Formula Matches

**Severity:** OK
**Domain:** 1. Flow Matching Formula

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py:14-39`

```python
def _add_noise_discrete(self, scaled_latent_image, latent_noise, timestep, timesteps):
    if self.__sigma is None:
        num_timesteps = timesteps.shape[-1]
        all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, ...)
        self.__sigma = all_timesteps / num_timesteps
        self.__one_minus_sigma = 1.0 - self.__sigma

    sigmas = self.__sigma[timestep]
    one_minus_sigmas = self.__one_minus_sigma[timestep]
    # ...
    scaled_noisy_latent_image = latent_noise * sigmas + scaled_latent_image * one_minus_sigmas
```

Sigma is computed as integer index (1-based) divided by `num_train_timesteps` (1000). Formula: `noisy = noise * sigma + latent * (1 - sigma)`.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1123-1125`

```python
timesteps = t * 1000.0
t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
noisy_model_input = (1 - t) * latents + t * noise
```

Where `t` is sampled in [0, 1] and then shifted by the Qwen shift formula. The formula `(1-t)*latent + t*noise` is mathematically identical to `noise*sigma + latent*(1-sigma)` since `t = sigma`.

**Discrepancy:** None -- implementations agree. The only difference is parameterization: OneTrainer uses integer indices into a precomputed sigma table, blissful-tuner uses continuous `t` in [0,1]. Both produce `noisy = sigma * noise + (1 - sigma) * scaled_latent`.

**Recommendation:** None.

---

## Finding OT-QI-2: Latent Scaling Produces Identical Results via Different Timing

**Severity:** OK
**Domain:** 2. Latent Scaling

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/model/QwenModel.py:213-216`

```python
def scale_latents(self, latents):
    latents_mean = torch.tensor(self.vae.config.latents_mean, ...).view(1, self.vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, ...).view(1, self.vae.config.z_dim, 1, 1, 1)
    return (latents - latents_mean) * latents_std
```

Called during the training forward pass at `/Users/dustin/OneTrainer/modules/modelSetup/BaseQwenSetup.py:113`:
```python
scaled_latent_image = model.scale_latents(latent_image)
```

RAW latents from VAE are stored in cache (no scaling at data loading time). Scaling happens at training time before noise addition.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_autoencoder_kl.py:1032-1035`

```python
def encode_pixels_to_latents(self, pixels):
    # ... encode ...
    latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * latents_std
    return latents
```

Scaling is baked into latents at cache time. The trainer's `scale_shift_latents()` returns latents unchanged:

`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:417-418`
```python
def scale_shift_latents(self, latents):
    return latents
```

**Discrepancy:** Different timing but identical result. OneTrainer applies `(latent - mean) * (1/std)` at training time. blissful-tuner applies the same formula at caching time. Both use the same formula `(latents - latents_mean) * (1.0 / latents_std)` with the same VAE config values. The scaling is a no-op during training because it was already applied.

The inverse formula is also identical in both:
- OneTrainer: `latents / latents_std + latents_mean` (QwenModel.py:220-221)
- blissful-tuner: `latents / latents_std + latents_mean` (qwen_image_autoencoder_kl.py:1003-1004)

**Recommendation:** None. The cache-time approach is slightly more efficient (avoids redundant computation per training step).

---

## Finding OT-QI-3: Both Compute `noise - scaled_latent` as Loss Target in Unpacked Space

**Severity:** OK
**Domain:** 3. Loss Target Construction

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/BaseQwenSetup.py:156-162`

```python
predicted_flow = model.unpack_latents(packed_predicted_flow, height=..., width=...)
flow = latent_noise - scaled_latent_image
model_output_data = {
    'loss_type': 'target',
    'predicted': predicted_flow,
    'target': flow,
}
```

Loss target is `noise - scaled_latent_image`, both in unpacked (B, C, 1, H, W) space. Loss is then computed by `_flow_matching_losses` -> `__unmasked_losses` as MSE.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:592-602`

```python
model_pred = qwen_image_utils.unpack_latents(model_pred, ...)
latents = latents.to(device=accelerator.device, dtype=network_dtype)
target = noise - latents
return model_pred, target
```

Loss target is `noise - latents` where `latents` are already scaled (due to cache-time scaling). Both in unpacked space.

**Discrepancy:** None -- implementations agree. Both compute `noise - scaled_latent` and both unpack before loss computation.

**Recommendation:** None.

---

## Finding OT-QI-4: Timestep Normalization and Shift Formulas Match

**Severity:** OK
**Domain:** 4. Timestep Normalization

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/BaseQwenSetup.py:149`
```python
timestep=timestep / 1000,
```

Shift computation at `/Users/dustin/OneTrainer/modules/model/QwenModel.py:223-234`:
```python
def calculate_timestep_shift(self, latent_width, latent_height):
    base_seq_len = self.noise_scheduler.config.base_image_seq_len  # 256
    max_seq_len = self.noise_scheduler.config.max_image_seq_len    # 8192
    base_shift = self.noise_scheduler.config.base_shift             # 0.5
    max_shift = self.noise_scheduler.config.max_shift               # 0.9
    patch_size = 2
    image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return math.exp(mu)
```

Shift application at `/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupNoiseMixin.py:172`:
```python
timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
```

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:573`
```python
timesteps = timesteps / 1000.0
```

Shift computation at `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1022-1026`:
```python
elif args.timestep_sampling == "qwen_shift":
    mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
shift = math.exp(mu)
```

Shift application at line 1031:
```python
t = (t * shift) / (1 + (shift - 1) * t)
```

**Discrepancy:** None -- implementations agree. Both divide timestep by 1000 before passing to transformer. The shift formula is mathematically identical (verified: OneTrainer's `N*shift*t / ((shift-1)*t + N)` with t in [0,N] is equivalent to blissful-tuner's `t*shift / (1 + (shift-1)*t)` with t in [0,1] after rescaling). Both use `math.exp(mu)` as the shift value. Both derive `mu` from the same linear interpolation with parameters (base_seq_len=256, base_shift=0.5, max_seq_len=8192, max_shift=0.9).

**Recommendation:** None.

---

## Finding OT-QI-5: Latent Packing Order Matches

**Severity:** OK
**Domain:** 5. Latent Packing Order

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/BaseQwenSetup.py:112-134`

```python
scaled_latent_image = model.scale_latents(latent_image)           # 1. Scale
latent_noise = self._create_noise(scaled_latent_image, ...)       # 2. Create noise
# ... sample timestep ...
scaled_noisy_latent_image, sigma = self._add_noise_discrete(      # 3. Add noise
    scaled_latent_image, latent_noise, timestep, ...)
latent_input = scaled_noisy_latent_image
packed_latent_input = model.pack_latents(latent_input)            # 4. Pack
# ... pass to transformer ...
```

Order: **scale -> noise -> pack -> transformer**

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:2428-2436` (base trainer)
```python
latents = self.scale_shift_latents(latents)       # 1. Scale (no-op for Qwen, already scaled at cache time)
noise = torch.randn_like(latents)                 # 2. Create noise
noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(...)  # 3. Add noise
```

Then in `call_dit` at `/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image_train_network.py:459`:
```python
noisy_model_input = qwen_image_utils.pack_latents(noisy_model_input)  # 4. Pack
```

Order: **scale (at cache time) -> noise -> pack -> transformer**

**Discrepancy:** None -- both apply noise in unpacked space, then pack before transformer. The scaling happens at different times but the effective order is identical.

**Recommendation:** None.

---

## Finding OT-QI-6: Template Token Handling Differs in Approach (Not Result)

**Severity:** WARNING
**Domain:** 6. Text Encoding

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/model/QwenModel.py:159-166`

```python
text_encoder_output = text_encoder_output.hidden_states[-1]
tokens_mask = tokens_mask[:, DEFAULT_PROMPT_TEMPLATE_CROP_START:]  # Crop mask

# set masked state to 0
text_encoder_output = text_encoder_output[:, DEFAULT_PROMPT_TEMPLATE_CROP_START:,:] * tokens_mask.unsqueeze(-1)
```

Approach: Slice hidden states to remove first 34 tokens, multiply by mask (zeroing padded positions), then prune to max valid length and pad to multiple of 16.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_utils.py:426-431`

```python
split_hidden_states = extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
max_seq_len = max([e.size(0) for e in split_hidden_states])
prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
```

Approach: First extract only valid (non-padded) tokens using `extract_masked_hidden`, then drop first 34. Padding tokens never have non-zero hidden states because they were extracted before padding could be included.

**Discrepancy:** Both approaches produce equivalent results for the valid region. OneTrainer's mask multiplication (line 166) zeros padded positions explicitly; blissful-tuner avoids ever including them. The key functional difference is that OneTrainer passes the full hidden states through the text encoder first (padding included), then crops. Blissful-tuner does the same -- both pass padded input through the text encoder (at lines 415-416 of qwen_image_utils.py), then extract valid tokens.

However, there is a subtle difference in the hidden state computation: OneTrainer crops the template tokens AFTER the multiplication `text_encoder_output[:, CROP_START:,:] * tokens_mask.unsqueeze(-1)` which zeros padded positions. This means padded positions have exactly zero values. blissful-tuner's `extract_masked_hidden` extracts only positions where `attention_mask` is True, which also excludes padding. The result is functionally equivalent.

**Recommendation:** This is cosmetically different but functionally equivalent. No action needed. The blissful-tuner approach may be slightly more memory-efficient since it never materializes zeroed padding tokens.

---

## Finding OT-QI-7: blissful-tuner Allows 1024 User Tokens vs OneTrainer's 512

**Severity:** INFO
**Domain:** 6. Text Encoding

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/model/QwenModel.py:25`
```python
PROMPT_MAX_LENGTH = 512
```

Tokenizer max length: 512 + 34 (template) = 546 total tokens.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/qwen_image/qwen_image_utils.py:395`
```python
tokenizer_max_length = 1024
```

Tokenizer max length: 1024 + 34 (template) = 1058 total tokens.

**Discrepancy:** blissful-tuner supports up to 1024 user tokens, OneTrainer supports 512. This is a deliberate feature extension in blissful-tuner, not a bug. The official Qwen-Image model was trained with `max_length=512`, so longer prompts may not produce better results, but they will not cause errors.

**Recommendation:** Consider documenting this in `docs/qwen_image.md` as a known extension beyond the official tokenizer limit. Users should be aware that prompts longer than 512 tokens are beyond the model's training distribution.

---

## Finding OT-QI-8: Sequence Padding to Multiple of 16 Missing in blissful-tuner Training

**Severity:** WARNING
**Domain:** 6. Text Encoding

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/model/QwenModel.py:177-181`

```python
# pad to 16 because attention processors and/or torch.compile can have issues
# with uneven sequence lengths
if max_seq_length % 16 > 0 and (seq_lengths != max_seq_length).any():
    max_seq_length += (16 - max_seq_length % 16)

text_encoder_output = text_encoder_output[:, :max_seq_length, :]
bool_attention_mask = tokens_mask[:, :max_seq_length].bool()
```

This pads the sequence length to a multiple of 16 for compatibility with attention processors and torch.compile.

**What blissful-tuner does:**

During text caching (`qwen_image_cache_text_encoder_outputs.py:98-99`):
```python
txt_len = mask_i.to(dtype=torch.bool).sum().item()
embed_i = embed_i[:txt_len]
```

Embeddings are trimmed to exact valid length (no padding).

During training (`qwen_image_train_network.py:536-538`):
```python
max_len = max(txt_seq_lens)
vl_embed = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in vl_embed]
vl_embed = torch.stack(vl_embed, dim=0)
```

Pads to the max length in the batch, but NOT to a multiple of 16.

**Discrepancy:** blissful-tuner does not pad to a multiple of 16. With `torch.compile` enabled, this could trigger the PyTorch issue referenced in OneTrainer's comment (pytorch/pytorch#165506). Without torch.compile, this is harmless. With batch_size=1 (most common), the sequence length simply equals the prompt's valid token count.

**Recommendation:** Consider adding optional padding to a multiple of 16 in training when torch.compile is enabled. This would improve torch.compile compatibility and potentially avoid CUDAGraph recompilation with variable sequence lengths. Low urgency since batch_size=1 training is the norm.

---

## Finding OT-QI-9: blissful-tuner Lacks Text Encoder LoRA Support

**Severity:** INFO
**Domain:** 7. LoRA Targeting Scope

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/QwenLoRASetup.py:65-70`

```python
create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "text_encoder")
if model.text_encoder is not None:
    model.text_encoder_lora = LoRAModuleWrapper(
        model.text_encoder, "text_encoder", config
    ) if create_te else None
```

OneTrainer supports optional LoRA on the text encoder (Qwen2_5_VLForConditionalGeneration), in addition to the transformer.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/networks/lora_qwen_image.py:15`

```python
QWEN_IMAGE_TARGET_REPLACE_MODULES = ["QwenImageTransformerBlock"]
```

Only `QwenImageTransformerBlock` modules (within the transformer) are targeted. There is no text encoder LoRA support.

**Discrepancy:** OneTrainer can train LoRA on both text encoder and transformer. blissful-tuner only supports transformer LoRA. This is a feature gap, not a correctness issue. Text encoder LoRA is typically not needed for Qwen-Image (the OneTrainer preset defaults to `text_encoder.train: false`), but can be useful for specialized use cases.

**Recommendation:** Document as a known limitation. Text encoder LoRA is an advanced feature that most users do not need.

---

## Finding OT-QI-10: OneTrainer Has Layer Presets; blissful-tuner Has exclude_patterns

**Severity:** INFO
**Domain:** 7. LoRA Targeting Scope

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/BaseQwenSetup.py:38-43`

```python
LAYER_PRESETS = {
    "attn-mlp": ["attn", "img_mlp", "txt_mlp"],  # Recommended
    "attn-only": ["attn"],
    "blocks": ["transformer_block"],
    "full": [],  # All Linear/Conv2d layers
}
```

Layer filter applied via `config.layer_filter.split(",")`:
```python
model.transformer_lora = LoRAModuleWrapper(
    model.transformer, "transformer", config, config.layer_filter.split(",")
)
```

Default preset: `"attn,img_mlp,txt_mlp"` (attn-mlp). This targets attention and MLP layers within transformer blocks, but NOT modulation layers.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/networks/lora_qwen_image.py:42-46`

```python
exclude_mod = kwargs.get("exclude_mod", True)
if isinstance(exclude_mod, str):
    exclude_mod = ast.literal_eval(exclude_mod)
if exclude_mod:
    exclude_patterns.append(r".*\.(img_mod|txt_mod)\..*")
```

Default: All `Linear` layers within `QwenImageTransformerBlock` are targeted, except modulation layers (`img_mod`, `txt_mod`). This is controlled via `network_args` options (`exclude_mod`, `exclude_patterns`).

**Discrepancy:** Different configuration mechanisms but similar default behavior. OneTrainer's "attn-mlp" preset explicitly includes only attention and MLP layers. blissful-tuner includes everything within `QwenImageTransformerBlock` except modulation layers. The difference is that OneTrainer's preset-based system is more explicit about what's included, while blissful-tuner's exclusion-based system may inadvertently include layers not present in OneTrainer's defaults.

In practice, within a `QwenImageTransformerBlock`, the layers are: attention (qkv, proj), MLP (img_mlp, txt_mlp), and modulation (img_mod, txt_mod). With `exclude_mod=True`, blissful-tuner targets attention + MLP, which matches OneTrainer's "attn-mlp" preset.

**Recommendation:** None. The default behaviors are effectively equivalent.

---

## Finding OT-QI-11: Prior Audit Edit/Control Findings Consistent with OneTrainer

**Severity:** OK
**Domain:** 8. Edit/Control Image Handling

**What OneTrainer does:**

OneTrainer does not have a Qwen-Image Edit training flow in `BaseQwenSetup.predict`. The `predict` method only handles T2I (no control image concatenation). Edit model training in OneTrainer would require a separate setup class that is not present in the reviewed codebase.

**What blissful-tuner does:**

blissful-tuner's `call_dit` method (lines 462-501) supports Edit mode with control image concatenation:
```python
if is_edit:
    latents_control = [...]  # collect control latents
    noisy_model_input = torch.cat([noisy_model_input, latents_control], dim=1)
```

The prior audit findings T1 (silent fallback on missing control) and C1 (prompt/image misalignment) have been resolved:
- T1: Now raises `ValueError` by default when no control images found (line 496)
- C1: `raise ValueError()` instead of `continue` in cache script (line 47-49 of cache_text_encoder_outputs.py)

**Discrepancy:** OneTrainer's reviewed code only supports T2I training for Qwen-Image. blissful-tuner additionally supports Edit and Layered training modes. The prior audit's fixes (error-by-default for missing controls, no-skip in caching) are the correct approach and are consistent with how a well-implemented Edit training flow should work.

**Recommendation:** None. The prior audit findings were correctly resolved.

---

## Finding OT-QI-12: OneTrainer Supports MAE/Huber/Log-cosh Loss; blissful-tuner is MSE-only

**Severity:** WARNING
**Domain:** 3. Loss Computation

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py:45-119`

```python
# MSE/L2 Loss
if config.mse_strength != 0:
    losses += ... * config.mse_strength

# MAE/L1 Loss
if config.mae_strength != 0:
    losses += ... * config.mae_strength

# log-cosh Loss
if config.log_cosh_strength != 0:
    losses += ... * config.log_cosh_strength

# Huber Loss
if config.huber_strength != 0:
    losses += ... * config.huber_strength
```

Supports weighted combination of MSE, MAE, Log-cosh, and Huber losses with configurable strengths.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:2491`

```python
loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
```

Only MSE loss is supported. No option for MAE, Huber, or Log-cosh.

**Discrepancy:** OneTrainer provides more loss function options. MSE is the standard and most commonly used loss for flow matching, so this is not a correctness issue. However, Huber loss can be beneficial for robustness to outliers, and some users may prefer it.

**Recommendation:** This is a feature gap, not a bug. Consider adding `--loss_type` with options for MSE, MAE, Huber, and Log-cosh if there is user demand. Low priority since MSE is the standard for flow matching training.

---

## Finding OT-QI-13: OneTrainer Supports Offset Noise and Perturbation Noise

**Severity:** INFO
**Domain:** 1. Flow Matching

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupNoiseMixin.py:77-117`

```python
def _create_noise(self, source_tensor, config, generator, ...):
    noise = torch.randn(...)

    if config.offset_noise_weight > 0:
        offset_noise = torch.randn(
            (source_tensor.shape[0], source_tensor.shape[1], *[1 for _ in range(source_tensor.ndim - 2)]), ...)
        noise = noise + (config.offset_noise_weight * offset_noise)

    if config.perturbation_noise_weight > 0:
        perturbation_noise = torch.randn(source_tensor.shape, ...)
        noise = noise + (config.perturbation_noise_weight * perturbation_noise)

    return noise
```

Also supports generalized offset noise with time-dependent psi coefficients.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:2431`

```python
noise = torch.randn_like(latents)
```

Plain Gaussian noise only. No offset noise or perturbation noise support.

**Discrepancy:** OneTrainer supports offset noise (for improving dark/bright image generation) and perturbation noise. blissful-tuner does not. These are optional features; the default in OneTrainer is weight=0 (disabled).

**Recommendation:** Feature gap. Offset noise can be useful for some training scenarios but is not commonly needed for Qwen-Image LoRA training.

---

## Finding OT-QI-14: Sigma Index Offset Difference in Timestep Computation

**Severity:** WARNING
**Domain:** 4. Timestep Distribution

**What OneTrainer does:**

`/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py:22-24`

```python
all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, ...)
self.__sigma = all_timesteps / num_timesteps
```

Sigma table: `[1/1000, 2/1000, ..., 1000/1000]`. Index 0 maps to sigma=0.001, index 999 maps to sigma=1.0.

Then at `/Users/dustin/OneTrainer/modules/modelSetup/mixin/ModelSetupNoiseMixin.py:160`:
```python
timestep = logit_normal * num_timestep + min_timestep
```

After shift at line 172:
```python
timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
```

And the returned value at line 212:
```python
return timestep.int()
```

The `.int()` truncation means the shifted timestep is floored to an integer, which is then used as an index into the sigma table. The sigma table starts at 1/1000, not 0.

**What blissful-tuner does:**

`/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1123-1127`

```python
timesteps = t * 1000.0
# ...
timesteps += 1  # 1 to 1000
```

blissful-tuner computes `t` in [0,1], multiplies by 1000, then adds 1 to shift range from [0,1000] to [1,1001]. However, `t` values come from logit-normal sampling followed by the shift transformation, and are continuous. The `timesteps += 1` adjusts the range to match the scheduler's 1-indexed convention.

The actual noise addition uses `t` directly (not the integer timesteps):
```python
t = t.view(-1, 1, 1, 1, 1)
noisy_model_input = (1 - t) * latents + t * noise
```

So `t` is in [0,1] and `timesteps` is only used for the scheduler/loss weighting lookup.

**Discrepancy:** In OneTrainer, timestep indices are truncated to integers and used to look up sigma from a 1-indexed table. In blissful-tuner, continuous `t` values are used directly for noise mixing, and the integer timesteps (after +1) are only used for scheduler lookups.

The practical difference is minimal:
- OneTrainer: sigma = `floor(shifted_t) / 1000`, discrete
- blissful-tuner: sigma = `shifted_t` (continuous value)

For LOGIT_NORMAL sampling, the continuous approach in blissful-tuner is actually more precise since it avoids discretization artifacts. Both divide by 1000 before passing to the transformer.

**Recommendation:** The difference is minor and blissful-tuner's continuous approach may be slightly better (no discretization). No action needed.

---

## Cross-Reference with Prior Audit Findings

| Prior Finding | OneTrainer Approach | Consistent? |
|--------------|---------------------|-------------|
| T1: Edit model silent fallback | OneTrainer has no Edit training code | N/A (blissful-tuner goes beyond OneTrainer) |
| T4: qwen_shift correctness | OneTrainer uses same formula with scheduler config | Yes |
| C1: Missing control images | OneTrainer has no control image caching | N/A |
| C2: prompt_template_encode_start_idx 34/64 | OneTrainer uses 34 for T2I, same constant | Yes |
| L1: exclude_mod regex | OneTrainer uses layer filter presets, not regex | Different mechanism, same intent |
| I4: negative_prompt=None crash | OneTrainer sets empty string/space default | Consistent fix approach |

## Overall Assessment

The two implementations are **fundamentally equivalent** for the core training pipeline:

1. **Flow matching formula**: Identical
2. **Latent scaling**: Same formula, different timing (cache vs training time)
3. **Loss target**: Identical (`noise - scaled_latent`)
4. **Timestep shift**: Identical formula with same parameters
5. **Packing order**: Same (scale -> noise -> pack -> transformer)
6. **Text encoding**: Same hidden layer (-1), same template crop indices

**Key differences** are in feature scope, not correctness:
- OneTrainer has text encoder LoRA, multi-loss support, offset/perturbation noise, layer presets
- blissful-tuner has Edit/Layered training support, longer prompt support (1024 vs 512), mask-weighted loss with prior preservation, CFG normalization

**No critical correctness bugs found** that would affect training quality when comparing the core Qwen-Image LoRA pipeline between the two codebases.
