# OneTrainer vs blissful-tuner: Z-Image LoRA Pipeline Comparison

> **Date:** 2026-02-18
> **Methodology:** Line-by-line source code comparison across 8 critical domains
> **OneTrainer base:** `/Users/dustin/OneTrainer/`
> **blissful-tuner base:** `/Users/dustin/blissful-tuner/`

## Summary Table

| Finding | Domain | Severity | Synopsis |
|---------|--------|----------|----------|
| OT-ZI-1 | 1. Timestep Inversion | OK | Both invert: `(1000 - t) / 1000` |
| OT-ZI-2 | 2. Output Negation (Training) | **CRITICAL** | blissful-tuner does NOT negate transformer output during training |
| OT-ZI-3 | 2. Output Negation (Inference) | OK | Both negate during inference |
| OT-ZI-4 | 3. Flow Matching Target | **CRITICAL** | Target is inverted (`latents - noise` vs `noise - latents`) to compensate for missing negation, but this compensation is INCOMPLETE when combined with CFG sampling or weighting |
| OT-ZI-5 | 3. Flow Matching Noise Addition | OK | Both use `sigma * noise + (1 - sigma) * latents` |
| OT-ZI-6 | 4. Latent Scaling | OK | Both use `(latents - shift_factor) * scaling_factor` |
| OT-ZI-7 | 5. Input Format | WARNING | blissful-tuner passes batched tensor, not list of individual tensors |
| OT-ZI-8 | 6. Text Encoding | OK | Both use Qwen3 with `enable_thinking=True`, hidden state layer `-2` |
| OT-ZI-9 | 6. Text Encoding (attention mask type) | INFO | blissful-tuner passes `bool` mask to encoder; OneTrainer passes `float` mask |
| OT-ZI-10 | 7. LoRA Targeting | OK | Both exclude refiner layers by default; blissful-tuner allows `include_refiner=True` |
| OT-ZI-11 | 8. Loss Computation | WARNING | Loss is computed on non-negated predictions vs inverted target; mathematically equivalent only if both are consistent |
| OT-ZI-12 | 3. Dynamic Shift | INFO | Different dynamic shift formulas; both acknowledge parameters may be wrong for Z-Image |
| OT-ZI-13 | 5. Input Format (Training) | OK | blissful-tuner training uses batched tensor (NOT list); model `forward()` accepts batched tensor directly |

---

## Finding OT-ZI-1: Timestep Inversion (Training + Inference)
**Severity:** OK
**Domain:** 1. Timestep Inversion

**What OneTrainer does:**
```python
# modules/modelSetup/BaseZImageSetup.py:137
(1000 - timestep) / 1000
```
In both training (`predict()` line 137) and inference (`ZImageSampler.py` line 105).

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage_train_network.py:320
t_input = (1000.0 - timesteps) / 1000.0

# src/musubi_tuner/zimage_generate_image.py:743
timestep = (1000 - timestep) / 1000  # Reverse timestep for z-image

# src/musubi_tuner/zimage_train_network.py:191 (do_inference / sampling during training)
timestep = (1000 - timestep) / 1000  # Reverse for z-image
```

**Discrepancy:** None -- implementations agree. Both correctly invert the timestep for Z-Image in all code paths (training, inference, and sampling-during-training).

**Recommendation:** None.

---

## Finding OT-ZI-2: Output Negation in Training Forward Pass
**Severity:** CRITICAL
**Domain:** 2. Output Negation

**What OneTrainer does:**
```python
# modules/modelSetup/BaseZImageSetup.py:142
predicted_flow = - torch.stack(output_list, dim=0).squeeze(dim=2)
```
The transformer output is **negated** before computing loss. The target is:
```python
# modules/modelSetup/BaseZImageSetup.py:145
flow = latent_noise - scaled_latent_image  # noise - image
```
So: `loss = MSE(-model_output, noise - image)`

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage_train_network.py:335
model_pred = transformer(x=noisy_model_input, t=t_input, cap_feats=llm_embed, cap_mask=llm_mask)

# src/musubi_tuner/zimage_train_network.py:338
model_pred = model_pred.squeeze(2)  # [B, C, H, W]

# src/musubi_tuner/zimage_train_network.py:341
target = latents - noise  # image - noise (INVERTED target)
```
The transformer output is **NOT negated**. The target is `latents - noise` (opposite of OneTrainer). So: `loss = MSE(model_output, image - noise)`

**Discrepancy:** blissful-tuner compensates for the missing negation by inverting the target: instead of `MSE(-pred, noise - image)` it uses `MSE(pred, image - noise)`. Since `MSE(-a, b) == MSE(a, -b)`, these are **mathematically equivalent** for vanilla MSE loss. However, this equivalence breaks when:

1. **Loss weighting schemes** that depend on the sign/direction of the prediction are applied
2. **Prior preservation** computes `MSE(model_pred, prior_pred)` -- if `prior_pred` comes from the same non-negated model, this is consistent. But if any external tool or diagnostic expects the standard Z-Image convention (negated output), predictions will appear inverted.
3. **Debug/visualization** code would show inverted flow directions.

The compensation works for pure MSE training but represents a non-standard convention.

**Recommendation:** Consider aligning with the standard Z-Image convention for clarity and future-proofing:
- Negate the transformer output: `model_pred = -model_pred.squeeze(2)`
- Use the standard target: `target = noise - latents`

This is especially important if blissful-tuner ever adds loss types beyond MSE (e.g., Huber, L1, perceptual losses) where the direction matters, or if users want to compare training diagnostics with other frameworks.

---

## Finding OT-ZI-3: Output Negation in Inference
**Severity:** OK
**Domain:** 2. Output Negation

**What OneTrainer does:**
```python
# modules/modelSampler/ZImageSampler.py:110
noise_pred = - torch.stack(output_list, dim=0).squeeze(dim=2)
```

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage_generate_image.py:763
noise_pred = -noise_pred.squeeze(2)  # Remove frame dimension and invert sign

# src/musubi_tuner/zimage_train_network.py:206 (do_inference / sampling during training)
noise_pred = -noise_pred.squeeze(2)  # Remove frame dimension and invert sign
```

**Discrepancy:** None -- both negate the output during inference. The inference path is correct.

**Recommendation:** None.

---

## Finding OT-ZI-4: Flow Matching Target Formula
**Severity:** CRITICAL
**Domain:** 3. Flow Matching Formula

**What OneTrainer does:**
```python
# modules/modelSetup/BaseZImageSetup.py:145
flow = latent_noise - scaled_latent_image  # target = noise - image
```

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage_train_network.py:341
target = latents - noise  # target = image - noise (INVERTED)
```

The comment in the code reads: `# Target: Opposite of usual Flow matching`.

**Discrepancy:** As noted in OT-ZI-2, this inversion compensates for the missing negation. The pair `(no negation, inverted target)` is equivalent to `(negation, standard target)` for MSE loss. However, the non-standard convention creates a landmine:

1. **Loss weighting schemes** like `sigma_sqrt` (`sigma^{-2}`) multiply the *unreduced* loss tensor. Since unreduced MSE is non-negative and identical regardless of sign convention, sigma-based weighting is unaffected. This is safe.

2. **Prior preservation** in blissful-tuner uses: `MSE(model_pred, prior_pred)`. Since `prior_pred` also comes from the same non-negated model forward pass (see `hv_train_network.py:2474`), the signs are consistent. This is safe.

3. **The `do_inference` method in `ZImageNetworkTrainer`** (sampling during training) correctly negates the output at line 206, which is needed because the Euler step function expects the standard sign convention. This is correct because training uses inverted convention but inference must use standard convention.

**Net assessment:** While mathematically equivalent for current use, the dual-convention (non-negated training + negated inference) adds cognitive overhead and increases the risk of future bugs.

**Recommendation:** Align with standard convention for maintainability (see OT-ZI-2).

---

## Finding OT-ZI-5: Flow Matching Noise Addition
**Severity:** OK
**Domain:** 3. Flow Matching Formula

**What OneTrainer does:**
```python
# modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py:36-37
scaled_noisy_latent_image = latent_noise * sigmas + scaled_latent_image * one_minus_sigmas
# i.e., sigma * noise + (1 - sigma) * image
```

**What blissful-tuner does:**
```python
# src/musubi_tuner/hv_train_network.py:1125
noisy_model_input = (1 - t) * latents + t * noise
# i.e., (1 - sigma) * image + sigma * noise
```
This is the same formula, just written with terms in different order.

**Discrepancy:** None -- implementations agree. Both use the standard flow matching linear interpolation.

**Recommendation:** None.

---

## Finding OT-ZI-6: Latent Scaling
**Severity:** OK
**Domain:** 4. Latent Scaling

**What OneTrainer does:**
```python
# modules/model/ZImageModel.py:175-176
def scale_latents(self, latents):
    return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

def unscale_latents(self, latents):
    return latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
```
Where `shift_factor` and `scaling_factor` come from the diffusers AutoencoderKL config.

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage_train_network.py:268-273
def scale_shift_latents(self, latents):
    shift = zimage_config.ZIMAGE_VAE_SHIFT_FACTOR   # 0.1159
    scale = zimage_config.ZIMAGE_VAE_SCALING_FACTOR  # 0.3611
    latents = (latents - shift) * scale
    return latents

# src/musubi_tuner/zimage/zimage_utils.py:35-37
def shift_scale_latents_for_decode(latents):
    latents = (latents / zimage_config.ZIMAGE_VAE_SCALING_FACTOR) + zimage_config.ZIMAGE_VAE_SHIFT_FACTOR
    return latents
```

**Discrepancy:** None -- implementations agree. Both use `(latents - 0.1159) * 0.3611` for encoding and `latents / 0.3611 + 0.1159` for decoding.

**Recommendation:** None.

---

## Finding OT-ZI-7: Input Format - List vs Batched Tensor (Inference)
**Severity:** WARNING
**Domain:** 5. Input Format

**What OneTrainer does (inference):**
```python
# modules/modelSampler/ZImageSampler.py:98-108
latent_model_input = latent_image.unsqueeze(2).to(dtype=self.model.train_dtype.torch_dtype())
latent_model_input = torch.cat([latent_model_input] * batch_size)
latent_model_input_list = list(latent_model_input.unbind(dim=0))  # Split batch -> list
output_list = transformer(
    latent_model_input_list,       # List of individual 5D tensors
    (1000 - timestep_model_input) / 1000,
    prompt_embedding,
    return_dict=True
).sample
noise_pred = - torch.stack(output_list, dim=0).squeeze(dim=2)
```

OneTrainer's diffusers-based `ZImageTransformer2DModel` expects a **list** of individual tensors (one per batch element), returns a list, and uses `return_dict=True` to get `.sample`.

**What blissful-tuner does (inference):**
```python
# src/musubi_tuner/zimage_generate_image.py:745-750
latent_model_input = latents.to(model.dtype)
latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension
model_out = model(latent_model_input, timestep, embed, mask)
```

blissful-tuner's custom `ZImageTransformer2DModel` (`src/musubi_tuner/zimage/zimage_model.py:630-738`) has a `forward()` method that accepts a **batched tensor** `[B, C, F, H, W]` and returns a **batched tensor** `[B, C, F, H, W]`. It does not use `return_dict`.

**Discrepancy:** The models have different interfaces. OneTrainer uses diffusers' `ZImageTransformer2DModel` which takes a list; blissful-tuner has a custom re-implementation that takes a batched tensor. This is not a bug because blissful-tuner has its own model implementation. However, this means the two codebases are not weight-compatible at the forward-pass API level (though weights should be compatible after key remapping).

**Recommendation:** This is architectural and not a correctness issue. No action needed.

---

## Finding OT-ZI-8: Text Encoding - Core Parameters
**Severity:** OK
**Domain:** 6. Text Encoding

**What OneTrainer does:**
```python
# modules/model/ZImageModel.py:138-165
messages = format_input(prompt_item)  # [{"role": "user", "content": text}]
prompt_item = self.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
tokenizer_output = self.tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
text_encoder_output = text_encoder_output.hidden_states[-2]  # second-to-last layer
```

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage/zimage_utils.py:211-244
messages = [{"role": "user", "content": p}]
formatted_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
text_inputs = tokenizer(formatted_prompts, padding="max_length",
    max_length=zimage_config.DEFAULT_MAX_SEQUENCE_LENGTH,  # 512
    truncation=True, return_tensors="pt")
prompt_embeds = text_encoder(..., output_hidden_states=True).hidden_states[-2]
```

**Discrepancy:** None -- implementations agree on all key parameters:
- Chat template format: `[{"role": "user", "content": text}]`
- `enable_thinking=True`
- `add_generation_prompt=True`
- Max tokens: 512
- Hidden state layer: `-2` (second-to-last)
- Tokenizer: `Qwen2Tokenizer`
- Text encoder: `Qwen3ForCausalLM`

**Recommendation:** None.

---

## Finding OT-ZI-9: Text Encoding - Attention Mask Type
**Severity:** INFO
**Domain:** 6. Text Encoding

**What OneTrainer does:**
```python
# modules/model/ZImageModel.py:160
text_encoder_output = self.text_encoder(
    tokens, attention_mask=tokens_mask.float(), ...  # float mask
)
```

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage/zimage_utils.py:230
prompt_masks = text_inputs.attention_mask.to(text_encoder.device).bool()
# ...
prompt_embeds = text_encoder(input_ids=text_input_ids, attention_mask=prompt_masks, ...)  # bool mask
```

**Discrepancy:** OneTrainer passes a `float` attention mask to the text encoder; blissful-tuner passes a `bool` mask. In practice, `Qwen3ForCausalLM` internally converts the mask to the appropriate type for attention computation, so both should produce identical results. The HuggingFace transformers library handles both bool and float masks correctly.

**Recommendation:** No action needed. This is cosmetic.

---

## Finding OT-ZI-10: LoRA Targeting Scope
**Severity:** OK
**Domain:** 7. LoRA Targeting

**What OneTrainer does:**
```python
# modules/modelSetup/BaseZImageSetup.py:38-43
LAYER_PRESETS = {
    "full": [],
    "blocks": ["layers"],
    "attn-mlp": {'patterns': ["^(?=.*attention)(?!.*refiner).*", "^(?=.*feed_forward)(?!.*refiner).*"], 'regex': True},
    "attn-only": {'patterns': ["^(?=.*attention)(?!.*refiner).*"], 'regex': True},
}
```
Default presets explicitly exclude `refiner` layers via negative lookahead regex. The `full` preset trains everything.

**What blissful-tuner does:**
```python
# src/musubi_tuner/networks/lora_zimage.py:16
ZIMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]

# src/musubi_tuner/networks/lora_zimage.py:39-48
include_refiner = kwargs.pop("include_refiner", False)
if include_refiner:
    exclude_patterns.append(r".*_modulation.*")
else:
    exclude_patterns.append(r".*(_modulation|_refiner).*")
```

And in the architecture registry:
```python
# src/musubi_tuner/networks/network_arch.py:90-94
ARCHITECTURE_Z_IMAGE: {
    "target_modules": ZIMAGE_TARGET_REPLACE_MODULES,
    "exclude_patterns": [r".*(_modulation|_refiner).*"],
}
```

**Discrepancy:** Both exclude refiner layers by default. OneTrainer also always excludes `_modulation` layers (in all presets -- even `full` uses regex patterns that only match attention/feed_forward). blissful-tuner also excludes `_modulation` layers. The behavior is equivalent.

blissful-tuner additionally supports `include_refiner=True` via `network_args` to optionally include refiner layers, which OneTrainer does not offer in its presets. This is an enhancement, not a regression.

**Recommendation:** None. The implementations agree on defaults.

---

## Finding OT-ZI-11: Loss Computation Sign Consistency
**Severity:** WARNING
**Domain:** 8. Loss Computation

**What OneTrainer does:**
```python
# modules/modelSetup/BaseZImageSetup.py:142-150
predicted_flow = -torch.stack(output_list, dim=0).squeeze(dim=2)
flow = latent_noise - scaled_latent_image
# Loss: MSE(-model_out, noise - image)
```

Then in `calculate_loss()`:
```python
# modules/modelSetup/BaseZImageSetup.py:173-179
return self._flow_matching_losses(
    batch=batch, data=data, config=config,
    train_device=self.train_device, sigmas=model.noise_scheduler.sigmas
).mean()
```

The `_flow_matching_losses` function supports multiple loss types (MSE, MAE, Huber, Log-Cosh) and masked training.

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage_train_network.py:335-343
model_pred = transformer(...)
model_pred = model_pred.squeeze(2)  # NOT negated
target = latents - noise  # inverted target

# src/musubi_tuner/hv_train_network.py:2491
loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
```

**Discrepancy:** As analyzed in OT-ZI-2 and OT-ZI-4, the loss computation is mathematically equivalent for MSE:
- OneTrainer: `MSE(-pred, noise - image)` = `MSE(pred, image - noise)`
- blissful-tuner: `MSE(pred, image - noise)`

Both produce identical gradients. OneTrainer additionally supports MAE, Huber, and Log-Cosh losses where sign matters for MAE and Log-Cosh (these are odd-function-based losses). blissful-tuner currently only uses MSE for Z-Image training, so this is not an active issue.

**Recommendation:** If additional loss functions are ever added to blissful-tuner's Z-Image training, the sign convention must be reconsidered. For MSE-only training, this is not a problem.

---

## Finding OT-ZI-12: Dynamic Timestep Shift Formula
**Severity:** INFO
**Domain:** 3. Flow Matching Formula (Shift)

**What OneTrainer does:**
```python
# modules/model/ZImageModel.py:181-193
# Uses FLUX defaults from FlowMatchEulerDiscreteScheduler config:
# base_image_seq_len, max_image_seq_len, base_shift, max_shift
m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
b = base_shift - m * base_seq_len
mu = image_seq_len * m + b
return math.exp(mu)  # <-- exponential of mu
```
OneTrainer's developers note this is "likely wrong" for Z-Image (UI tooltip warning).

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage/zimage_utils.py:282-291
def compute_dynamic_shift(image_seq_len: int) -> float:
    mu = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN) \
         * (image_seq_len - BASE_IMAGE_SEQ_LEN) + BASE_SHIFT
    return max(BASE_SHIFT, min(MAX_SHIFT, mu))  # <-- clamped linear mu (NO exp)
```
Where `BASE_SHIFT=0.5`, `MAX_SHIFT=1.15`, `BASE_IMAGE_SEQ_LEN=256`, `MAX_IMAGE_SEQ_LEN=4096`.

**Discrepancy:** OneTrainer returns `exp(mu)` while blissful-tuner returns `mu` directly (clamped). These give very different shift values. For example, at image_seq_len=1024:
- OneTrainer mu = linear_interp = ~0.70, shift = exp(0.70) = ~2.01
- blissful-tuner mu = linear_interp = ~0.70, shift = 0.70

Both codebases acknowledge the parameters may be wrong for Z-Image. blissful-tuner defaults to `--discrete_flow_shift 3.0` (static), and OneTrainer defaults to not using dynamic shifting.

**Recommendation:** Since dynamic shifting for Z-Image has no known-correct parameters, and both codebases default to static shifting, this is not a practical issue. However, users who enable dynamic shifting will get very different behavior between the two frameworks. Consider documenting this difference.

---

## Finding OT-ZI-13: Input Format During Training
**Severity:** OK
**Domain:** 5. Input Format

**What OneTrainer does:**
```python
# modules/modelSetup/BaseZImageSetup.py:132-140
latent_input = scaled_noisy_latent_image.unsqueeze(2)  # Add frame dim
latent_input = latent_input.to(dtype=model.train_dtype.torch_dtype())
latent_input_list = list(latent_input.unbind(dim=0))  # Split batch -> list
output_list = model.transformer(
    latent_input_list,           # List of individual 5D tensors
    (1000 - timestep) / 1000,
    text_encoder_output,         # List of variable-length embeddings
    return_dict=True
).sample
```
Uses diffusers' `ZImageTransformer2DModel` which takes lists.

**What blissful-tuner does:**
```python
# src/musubi_tuner/zimage_train_network.py:296-335
noisy_model_input = noisy_model_input.unsqueeze(2)  # [B, C, 1, H, W]
# ... padding and stacking embeddings to [B, L, D] ...
model_pred = transformer(x=noisy_model_input, t=t_input, cap_feats=llm_embed, cap_mask=llm_mask)
```
Uses custom `ZImageTransformer2DModel` which takes batched tensor `[B, C, F, H, W]` directly.

**Discrepancy:** Different APIs due to different model implementations (diffusers vs custom). blissful-tuner's custom model handles batching internally (patchify, RoPE, attention mask creation), which is functionally equivalent. The custom implementation also handles the text embedding padding to a multiple of `SEQ_MULTI_OF` for attention efficiency.

**Recommendation:** None. This is an architectural choice, not a bug.

---

## Overall Assessment

### Critical Issues

1. **OT-ZI-2 / OT-ZI-4: Non-Standard Sign Convention in Training.** blissful-tuner uses an inverted target (`latents - noise`) instead of negating the transformer output. While mathematically equivalent for MSE loss, this creates a non-standard convention that:
   - Diverges from the official Z-Image convention
   - Diverges from OneTrainer's convention
   - Requires careful attention if additional loss functions are added
   - May confuse users comparing training diagnostics across frameworks

   **Suggested fix:** Negate the model output in `call_dit()` and use the standard target `noise - latents`. This is a two-line change.

### Warnings

2. **OT-ZI-7: Model Interface Difference.** blissful-tuner has a custom model that takes batched tensors while OneTrainer uses diffusers' list-based interface. This is by design and works correctly, but means the forward pass implementations are not directly comparable.

3. **OT-ZI-11: Loss Type Limitation.** Current MSE-only loss is fine with the inverted convention, but adding Huber/MAE/Log-Cosh losses would require revisiting the sign convention.

### Info Items

4. **OT-ZI-9:** Attention mask type (bool vs float) -- no practical impact.
5. **OT-ZI-12:** Dynamic shift formula differs (exp vs linear) -- neither has validated Z-Image parameters.

### Things blissful-tuner Gets Right

- Timestep inversion: correct everywhere (training, inference, sampling-during-training)
- Latent scaling: correct formulas with matching constants
- Text encoding: all parameters match (chat template, thinking mode, layer index, max tokens)
- LoRA targeting: correct exclusion of refiner and modulation layers
- Inference negation: correct in all inference paths
- Flow matching noise addition: correct formula
