# SimpleTuner vs blissful-tuner: Z-Image LoRA Training Pipeline Comparison

> **Date**: 2026-02-18
> **Scope**: Z-Image (Base & Turbo) LoRA training pipeline
> **SimpleTuner commit base**: Current local checkout at `/Users/dustin/SimpleTuner/`
> **blissful-tuner commit base**: main branch at `/Users/dustin/blissful-tuner/`

---

## Summary Table

| Finding | Title | Severity | Domain |
|---------|-------|----------|--------|
| ST-ZI-1 | Timestep inversion agrees | OK | 1. Timestep Inversion |
| ST-ZI-2 | Output negation: training path differs (target inversion vs output inversion) | WARNING | 2. Output Negation |
| ST-ZI-3 | Flow matching noise formula agrees | OK | 3. Flow Matching Formula |
| ST-ZI-4 | Loss target: `noise - latents` vs `latents - noise` (equivalent due to negation) | INFO | 3. Flow Matching Formula |
| ST-ZI-5 | Latent scaling formula agrees | OK | 4. Latent Scaling |
| ST-ZI-6 | Input format: batched tensors vs list of tensors | WARNING | 5. Input Format |
| ST-ZI-7 | Text encoding: agree on core pipeline | OK | 6. Text Encoding |
| ST-ZI-8 | Text encoding: `enable_thinking=True` in both | OK | 6. Text Encoding |
| ST-ZI-9 | LoRA targeting: broader scope in blissful-tuner, narrower in SimpleTuner | INFO | 7. LoRA Targeting |
| ST-ZI-10 | No turbo/assistant LoRA support in blissful-tuner | INFO | 8. Turbo Handling |
| ST-ZI-11 | RoPE `axes_lens` differs: `[1536, 512, 512]` vs `[1024, 512, 512]` default | WARNING | 5. Input Format |
| ST-ZI-12 | Inference CFG: two forward passes vs joint batch | INFO | 2. Output Negation |
| ST-ZI-13 | CFG Zero\* not implemented in blissful-tuner inference | INFO | 8. Turbo Handling |
| ST-ZI-14 | Default flow_shift agrees at 3.0 | OK | 8. Turbo Handling |

---

## Finding ST-ZI-1: Timestep Inversion Agrees

**Severity:** OK
**Domain:** 1. Timestep Inversion

**What SimpleTuner does:**
`model.py:363` (inside `model_predict()`):
```python
normalized_t = (1000.0 - timesteps) / 1000.0
```
Timesteps come from the scheduler in range [0, 1000]. The inversion maps `t=100 -> 0.9`, `t=500 -> 0.5`, `t=1000 -> 0.0`.

**What blissful-tuner does:**
`/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_train_network.py:320` (inside `call_dit()`):
```python
t_input = (1000.0 - timesteps) / 1000.0
```
And in inference at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_generate_image.py:743`:
```python
timestep = (1000 - timestep) / 1000  # Reverse timestep for z-image
```
Also in `do_inference()` at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_train_network.py:191`:
```python
timestep = (1000 - timestep) / 1000  # Reverse for z-image
```

**Discrepancy:** None -- implementations agree on timestep inversion formula in both training and inference.

**Recommendation:** None.

---

## Finding ST-ZI-2: Output Negation Strategy Differs Between Training Paths

**Severity:** WARNING
**Domain:** 2. Output Negation

**What SimpleTuner does:**
In `model_predict()` at `model.py:381`:
```python
noise_pred = -noise_pred  # CRITICAL: negate for flow matching
```
The model output is negated AFTER the transformer forward pass. The loss target is:
At `common.py:3516`:
```python
target = prepared_batch["noise"] - prepared_batch["latents"]  # noise - latents
```
So SimpleTuner negates the prediction and uses `noise - latents` as the target.

**What blissful-tuner does:**
In `call_dit()` at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_train_network.py:338-342`:
```python
model_pred = model_pred.squeeze(2)  # [B, C, H, W]

# Target: Opposite of usual Flow matching
target = latents - noise
```
blissful-tuner does NOT negate `model_pred` during training. Instead, it inverts the target from `noise - latents` to `latents - noise`.

In inference, blissful-tuner DOES negate:
At `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_generate_image.py:763`:
```python
noise_pred = -noise_pred.squeeze(2)  # Remove frame dimension and invert sign
```
And in `do_inference()` at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_train_network.py:206`:
```python
noise_pred = -noise_pred.squeeze(2)  # Remove frame dimension and invert sign
```

**Discrepancy:** The mathematical end result is equivalent. If `model_output = M(x)`, then:
- SimpleTuner: `loss = MSE(-M(x), noise - latents)` = `MSE(M(x), latents - noise)`
- blissful-tuner: `loss = MSE(M(x), latents - noise)`

These are identical. The inference path also agrees: both negate the output before the scheduler step.

However, this means the trained LoRA weights are conceptually learning the same thing but via different code paths. Any future refactor that touches only one side of this equation could introduce a sign bug. The asymmetry between training (no negation) and inference (negation) in blissful-tuner is slightly fragile.

**Recommendation:** This is mathematically correct but consider adding a code comment in `call_dit()` explaining that the target inversion (`latents - noise` instead of `noise - latents`) is equivalent to negating the model output, to prevent future confusion.

---

## Finding ST-ZI-3: Flow Matching Noise Addition Formula Agrees

**Severity:** OK
**Domain:** 3. Flow Matching Formula

**What SimpleTuner does:**
At `common.py:4374`:
```python
batch["noisy_latents"] = (1 - batch["sigmas"]) * batch["latents"] + batch["sigmas"] * batch["input_noise"]
```

**What blissful-tuner does:**
At `/Users/dustin/blissful-tuner/src/musubi_tuner/hv_train_network.py:1147`:
```python
noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
```
And for the shift-based path at line 1125:
```python
noisy_model_input = (1 - t) * latents + t * noise
```

**Discrepancy:** None -- both use the standard flow matching interpolation formula: `noisy = (1 - sigma) * latents + sigma * noise`. The operand order differs but the result is identical.

**Recommendation:** None.

---

## Finding ST-ZI-4: Loss Target Sign Convention Equivalent

**Severity:** INFO
**Domain:** 3. Flow Matching Formula

**What SimpleTuner does:**
At `common.py:3516`:
```python
target = prepared_batch["noise"] - prepared_batch["latents"]
```
Then negates the model output (`model_predict()` line 381: `noise_pred = -noise_pred`).

**What blissful-tuner does:**
At `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_train_network.py:342`:
```python
target = latents - noise
```
Does NOT negate the model output during training.

**Discrepancy:** As analyzed in ST-ZI-2, these are mathematically equivalent: `MSE(-pred, noise - latents) = MSE(pred, latents - noise)`. Both produce identical gradients.

**Recommendation:** None required; this is a valid alternative.

---

## Finding ST-ZI-5: Latent Scaling Formula Agrees

**Severity:** OK
**Domain:** 4. Latent Scaling

**What SimpleTuner does:**
Encoding (implicit via `AutoencoderKL`):
`scale: (latents - shift_factor) * scaling_factor`
Decoding at `pipeline.py:1064`:
```python
latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
```

**What blissful-tuner does:**
Encoding scale at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_train_network.py:271-273`:
```python
shift = zimage_config.ZIMAGE_VAE_SHIFT_FACTOR  # 0.1159
scale = zimage_config.ZIMAGE_VAE_SCALING_FACTOR  # 0.3611
latents = (latents - shift) * scale
```
Decoding unscale at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage/zimage_utils.py:37`:
```python
latents = (latents / zimage_config.ZIMAGE_VAE_SCALING_FACTOR) + zimage_config.ZIMAGE_VAE_SHIFT_FACTOR
```

**Discrepancy:** None -- the encode formula is `(latents - shift) * scale` and decode formula is `latents / scale + shift`. Constants match: `scaling_factor = 0.3611`, `shift_factor = 0.1159` (from `zimage_config.py:58-59`).

**Recommendation:** None.

---

## Finding ST-ZI-6: Input Format Differs (Batched Tensor vs List of Tensors)

**Severity:** WARNING
**Domain:** 5. Input Format

**What SimpleTuner does:**
The `ZImageTransformer2DModel.forward()` at `transformer.py:636-648` takes:
```python
def forward(
    self,
    x: List[torch.Tensor],  # List of [C, F, H, W] tensors
    t,
    cap_feats: List[torch.Tensor],  # List of [seq_i, 2560] tensors
    ...
```
In `model_predict()` at `model.py:356`:
```python
latent_list = [sample.to(...) for sample in latents]
```
Each sample in the batch is passed as a separate tensor in a list. The transformer internally handles per-sample padding and variable-length sequences via `patchify_and_embed()` which processes each item individually (transformer.py:537-634).

**What blissful-tuner does:**
The `ZImageTransformer2DModel.forward()` at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage/zimage_model.py:630-638` takes:
```python
def forward(
    self,
    x: torch.Tensor,  # [B, C, F, H, W] batched tensor
    t: torch.Tensor,
    cap_feats: torch.Tensor,  # [B, cap_seq_len, cap_feat_dim] batched tensor
    cap_mask: torch.Tensor,  # [B, cap_seq_len]
    ...
```
blissful-tuner uses standard batched tensors with an explicit attention mask (`cap_mask`) to handle variable-length captions.

**Discrepancy:** SimpleTuner's transformer accepts a list of individual tensors (enabling natural variable-length caption handling via per-sample padding), while blissful-tuner pre-pads all captions to the same length and passes a batched tensor with an attention mask.

This is an architectural difference that affects how padding tokens are handled:
- SimpleTuner: The `x_pad_token` / `cap_pad_token` are applied per-sample in `patchify_and_embed()` where each sample is independently padded to `SEQ_MULTI_OF` alignment.
- blissful-tuner: `cap_mask` controls padding, and `cap_pad_token` is applied via `masked_fill` at model.py:693-694.

For training with batch_size=1 (the common case for LoRA), this difference is negligible. For batch_size>1, the padding handling differs but both should produce functionally equivalent results IF the attention masking is correct.

**Recommendation:** Verify that blissful-tuner's attention masking in the main transformer layers correctly excludes caption padding tokens when `batch_size > 1`. The SimpleTuner approach naturally handles this via `pad_sequence` + `unified_attn_mask`, while blissful-tuner relies on `AttentionParams.create_attention_params_from_mask()`.

---

## Finding ST-ZI-7: Text Encoding Core Pipeline Agrees

**Severity:** OK
**Domain:** 6. Text Encoding

**What SimpleTuner does:**
At `model.py:251-291`:
1. Apply Qwen3 chat template with `enable_thinking=True`
2. Tokenize with `padding="max_length"`, `max_length=512`, `truncation=True`
3. Extract `hidden_states[-2]` (second-to-last layer)
4. Convert to variable-length via attention mask

**What blissful-tuner does:**
At `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage/zimage_utils.py:190-245`:
1. Apply Qwen3 chat template with `enable_thinking=True`
2. Tokenize with `padding="max_length"`, `max_length=512` (`DEFAULT_MAX_SEQUENCE_LENGTH`), `truncation=True`
3. Extract `hidden_states[-2]` (second-to-last layer)
4. Returns full-padded embeddings + mask; trimming done elsewhere

**Discrepancy:** None -- the core encoding pipeline is identical. Both use the same chat template, `enable_thinking=True`, max sequence length of 512, and second-to-last hidden state.

**Recommendation:** None.

---

## Finding ST-ZI-8: `enable_thinking=True` Consistent

**Severity:** OK
**Domain:** 6. Text Encoding

**What SimpleTuner does:** `model.py:267`: `enable_thinking=True`

**What blissful-tuner does:** `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage/zimage_utils.py:217`: `enable_thinking=True`

**Discrepancy:** None.

**Recommendation:** None.

---

## Finding ST-ZI-9: LoRA Targeting Scope Differs

**Severity:** INFO
**Domain:** 7. LoRA Targeting

**What SimpleTuner does:**
Default LoRA targets at `common.py` (inherited `DEFAULT_LORA_TARGET`):
```python
DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
```
These are module-name substring matches applied to the transformer's attention layers. This targets 4 specific linear layers per `ZImageTransformerBlock`.

SimpleTuner does NOT have a concept of explicitly excluding refiner layers -- all `ZImageTransformerBlock` instances (including noise_refiner and context_refiner blocks) would match the LoRA targets.

**What blissful-tuner does:**
At `/Users/dustin/blissful-tuner/src/musubi_tuner/networks/lora_zimage.py:16`:
```python
ZIMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]
```
This targets the entire `ZImageTransformerBlock` module class, meaning ALL `nn.Linear` layers within each block are targeted (attention Q/K/V/out, FFN w1/w2/w3, and potentially adaLN modulation). However, default exclude patterns filter out modulation and refiner layers:
At `lora_zimage.py:47`:
```python
exclude_patterns.append(r".*(_modulation|_refiner).*")
```

So the effective targeting is:
- **Main 30 transformer blocks**: All linear layers in attention AND FFN (broader than SimpleTuner)
- **noise_refiner / context_refiner**: Excluded by default (narrower than SimpleTuner)
- **adaLN_modulation**: Excluded (both trainers agree this causes instability)

**Discrepancy:** blissful-tuner targets FFN layers (w1, w2, w3) in the main transformer blocks by default, while SimpleTuner only targets attention Q/K/V/out projections. blissful-tuner excludes refiner layers by default, while SimpleTuner would target them. This means:
- blissful-tuner: ~7 linear layers per block x 30 blocks = ~210 LoRA modules
- SimpleTuner: ~4 linear layers per block x 34 blocks (30 main + 2 noise_refiner + 2 context_refiner) = ~136 LoRA modules

**Recommendation:** This is a design choice rather than a bug. The broader FFN targeting in blissful-tuner may provide more expressiveness but at higher memory cost. Consider documenting this difference for users migrating LoRA weights between trainers. The `include_refiner=True` network_arg is already available in blissful-tuner for users who want refiner targeting.

---

## Finding ST-ZI-10: No Turbo / Assistant LoRA Support in blissful-tuner

**Severity:** INFO
**Domain:** 8. Turbo Handling

**What SimpleTuner does:**
Z-Image supports multiple model flavours (`model.py:57-63`): `base`, `turbo`, `turbo-ostris-v2`, `ostris-de-turbo`.
For turbo variants during LoRA training, an assistant LoRA adapter is loaded (`model.py:226-249`):
```python
ASSISTANT_LORA_FLAVOURS = ["turbo", "turbo-ostris-v2"]
ASSISTANT_LORA_PATH = "ostris/zimage_turbo_training_adapter"
```
This is a pre-trained adapter that helps stabilize turbo model LoRA training. If not provided, SimpleTuner raises a `ValueError` (`model.py:319-336`).

**What blissful-tuner does:**
blissful-tuner has no concept of model flavours, assistant LoRAs, or turbo-specific handling. Users load whatever checkpoint they want via `--dit`. The trainer treats all Z-Image checkpoints identically.

**Discrepancy:** blissful-tuner users training LoRAs on Z-Image-Turbo will not have the assistant LoRA that SimpleTuner considers mandatory. This may lead to training instability or suboptimal results for turbo variants, though it is not a correctness bug -- users can still train without an assistant adapter.

**Recommendation:** Consider documenting that Z-Image-Turbo LoRA training may benefit from using the `ostris/zimage_turbo_training_adapter` (available on HuggingFace). This could be implemented as a pre-merge LoRA that gets applied before training begins, or as documentation guidance.

---

## Finding ST-ZI-11: RoPE `axes_lens` Default Differs

**Severity:** WARNING
**Domain:** 5. Input Format (Architecture Constants)

**What SimpleTuner does:**
The `ZImageTransformer2DModel.__init__()` at `transformer.py:414-415`:
```python
axes_dims=[32, 48, 48],
axes_lens=[1024, 512, 512],
```
However, SimpleTuner uses `ConfigMixin` and `register_to_config`, so these are merely Python defaults. The actual values are loaded from the pretrained model's `config.json` via `from_pretrained()`. The HuggingFace model config determines the runtime values.

**What blissful-tuner does:**
At `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage/zimage_config.py:41`:
```python
ROPE_AXES_LENS = [1536, 512, 512]
```
This is hardcoded and used directly in model creation at `zimage_model.py:765`:
```python
axes_lens=zimage_config.ROPE_AXES_LENS,
```
The model is always created with `axes_lens=[1536, 512, 512]` regardless of what the original checkpoint used.

The comment at `zimage_model.py:352` confirms this was intentional:
```python
# [torch.Size([1536, 16]), torch.Size([512, 24]), torch.Size([512, 24])]
```

**Discrepancy:** The first axis length differs: blissful-tuner uses `1536` while SimpleTuner's Python default is `1024`. The first axis corresponds to the sequence/frame dimension of the RoPE embedding.

This is NOT necessarily a bug. The `axes_lens` values define the MAXIMUM positions for the precomputed RoPE frequency tables. A larger value (1536) simply allocates a bigger lookup table, supporting positions up to index 1535 on the first axis. As long as no actual position index exceeds the maximum, the RoPE values for indices within both ranges will be identical. Since Z-Image operates on images (not video), and the first axis encodes `cap_seq_len + 1 + F_tokens` where `F_tokens=1`, the actual position indices are bounded by `max_sequence_length + 1 + 1 = 514`, which is well within both `1024` and `1536`.

The blissful-tuner value of `1536` likely comes from the original Z-Image Team source code, while SimpleTuner's `1024` is just a more conservative default that gets overridden by the model config.

**Recommendation:** This is functionally harmless for the current Z-Image model because position indices stay below 1024. However, it would be prudent to verify the value from the official Z-Image `config.json` on HuggingFace and use the canonical value to avoid any potential issue if future model variants use positions up to 1535.

---

## Finding ST-ZI-12: Inference CFG Implementation Differs

**Severity:** INFO
**Domain:** 2. Output Negation (Inference)

**What SimpleTuner does:**
At `pipeline.py:935-963`, positive and negative latents are concatenated and processed in a single forward pass:
```python
latent_model_input = torch.cat([latents.to(dtype)] * 2)
prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
timestep_model_input = torch.cat([timestep] * 2)
model_out_list = self.transformer(latent_model_input_list, timestep_model_input, prompt_embeds_model_input, ...)[0]
```
Then the outputs are split for CFG computation. The output negation happens at line 1003:
```python
noise_pred = -noise_pred
```

**What blissful-tuner does:**
At `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_generate_image.py:749-756`, positive and negative prompts are processed in separate forward passes:
```python
model_out = model(latent_model_input, timestep, embed, mask)
neg_model_out = model(latent_model_input, timestep, negative_embed, negative_mask)
noise_pred = model_out + effective_guidance_scale * (model_out - neg_model_out)
```
Then negation at line 763:
```python
noise_pred = -noise_pred.squeeze(2)
```

**Discrepancy:** SimpleTuner batches both passes together (more efficient on GPU), while blissful-tuner uses two separate forward passes. The mathematical result is the same. The CFG formula also differs slightly:
- SimpleTuner (CFG Zero*): `guided = neg * alpha + scale * (pos - neg * alpha)`
- blissful-tuner (standard CFG): `noise_pred = pos + scale * (pos - neg)`

The standard CFG formula in blissful-tuner is `pos + scale * (pos - neg)`, which is equivalent to `pos * (1 + scale) - neg * scale`. SimpleTuner's non-zero-star fallback is `guided = pos + scale * (pos - neg)`, which is the same.

**Recommendation:** The two-pass approach is functionally correct but roughly 2x slower for CFG inference. Consider batching both passes if performance is a concern.

---

## Finding ST-ZI-13: CFG Zero\* Not Implemented in blissful-tuner Inference

**Severity:** INFO
**Domain:** 8. Turbo Handling

**What SimpleTuner does:**
At `pipeline.py:973-980`, CFG Zero\* is the default guidance method:
```python
if use_cfg_zero_star:
    pos_flat = pos_out.view(actual_batch_size, -1)
    neg_flat = neg_out.view(actual_batch_size, -1)
    alpha = optimized_scale(pos_flat, neg_flat).view(...)
    guided = neg_out * alpha + current_guidance_scale * (pos_out - neg_out * alpha)
```

**What blissful-tuner does:**
Standard CFG at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_generate_image.py:756`:
```python
noise_pred = model_out + effective_guidance_scale * (model_out - neg_model_out)
```
No CFG Zero\* implementation. blissful-tuner does have CFG normalization as an option (`--cfg_normalization`).

**Discrepancy:** blissful-tuner lacks CFG Zero\* for Z-Image inference. Note that blissful-tuner does implement CFG Zero\* for other architectures (e.g., in `src/blissful_tuner/guidance.py`), so this could be added.

**Recommendation:** Consider wiring in the existing `guidance.py` CFG Zero\* implementation for Z-Image generation. This would improve inference quality when using CFG guidance, especially at higher guidance scales.

---

## Finding ST-ZI-14: Default Flow Schedule Shift Agrees

**Severity:** OK
**Domain:** 8. Turbo Handling

**What SimpleTuner does:**
Default `flow_schedule_shift = 3.0` (documented in reference, loaded from scheduler config).

**What blissful-tuner does:**
At `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_train_network.py:48`:
```python
self.default_discrete_flow_shift = 3.0
```
And in inference at `/Users/dustin/blissful-tuner/src/musubi_tuner/zimage_generate_image.py:166`:
```python
parser.add_argument("--flow_shift", type=float, default=3.0, ...)
```

**Discrepancy:** None -- both default to 3.0 for Z-Image flow matching shift.

**Recommendation:** None.

---

## Overall Assessment

The blissful-tuner Z-Image training pipeline is **correct** in its core mathematical operations. The timestep inversion, noise formula, latent scaling, and loss computation are all functionally equivalent to SimpleTuner's implementation. The key differences are:

1. **Correct but worth noting**: The output negation is handled via target inversion (`latents - noise`) in training rather than output negation (`-model_pred`). This is mathematically equivalent but creates a code-level asymmetry between training and inference.

2. **Architecture difference**: blissful-tuner uses batched tensors with attention masks while SimpleTuner uses per-sample lists. Both are valid approaches.

3. **Feature gaps**: blissful-tuner lacks turbo assistant LoRA support and CFG Zero\* for Z-Image inference. These are quality-of-life improvements rather than correctness issues.

4. **LoRA scope**: blissful-tuner targets more layers (including FFN) by default while excluding refiner layers. SimpleTuner targets fewer layers (attention only) but includes refiners. Neither is "wrong" but users should be aware when comparing results or transferring LoRA weights.

5. **RoPE `axes_lens`**: The first axis maximum differs (`1536` vs `1024`) but this is harmless for current Z-Image models since actual position indices stay well below 1024.

No critical bugs or regressions were found.
