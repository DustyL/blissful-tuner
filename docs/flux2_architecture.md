# FLUX.2 Architecture Reference

This document provides comprehensive architectural details for FLUX.2-DEV and FLUX.2-klein-9B models to assist in implementing LoRA training pipelines and debugging.

## Quick Reference

| Parameter | FLUX.2-DEV | FLUX.2-klein-9B | FLUX.2-klein-base-9B |
|-----------|------------|-----------------|----------------------|
| **Transformer** |
| Double-stream blocks | 8 | 8 | 8 |
| Single-stream blocks | 48 | 24 | 24 |
| Attention heads | 48 | 32 | 32 |
| Head dimension | 128 | 128 | 128 |
| Hidden dimension | 6144 | 4096 | 4096 |
| Joint attention dim | 15360 | 12288 | 12288 |
| MLP ratio | 3.0 | 3.0 | 3.0 |
| In channels | 128 | 128 | 128 |
| Patch size | 1 | 1 | 1 |
| RoPE axes | [32,32,32,32] | [32,32,32,32] | [32,32,32,32] |
| RoPE theta | 2000 | 2000 | 2000 |
| Guidance embeds | true | false | false |
| **Text Encoder** |
| Architecture | Mistral3/Pixtral | Qwen3 | Qwen3 |
| Hidden size | 5120 | 4096 | 4096 |
| Layers | 40 | 36 | 36 |
| Attention heads | 32 | 32 | 32 |
| KV heads (GQA) | 8 | 8 | 8 |
| Head dim | 128 | 128 | 128 |
| Intermediate size | 32768 | 12288 | 12288 |
| Vocab size | 131072 | 151936 | 151936 |
| Vision encoder | Pixtral (1024 hidden) | None | None |
| **VAE** |
| Latent channels | 32 | 32 | 32 |
| Patch size | [2, 2] | [2, 2] | [2, 2] |
| Spatial compression | 16× (8× encoder + 2×2 patchify) | 16× (8× encoder + 2×2 patchify) | 16× (8× encoder + 2×2 patchify) |
| Block channels | [128,256,512,512] | [128,256,512,512] | [128,256,512,512] |
| **Training** |
| Distillation | Guidance distillation | 4-step distilled | None (base teacher) |
| Typical steps | 50 | 4 | 50 |
| Guidance scale | 4.0 (default) | 1.0 (fixed) | 4.0 (default) |

> **Guidance vs. CFG note:** In FLUX.2, “guidance” is not always “negative-prompt classifier-free guidance (CFG)”.
> - **DEV** is guidance-distilled and uses an explicit guidance scalar embedding (`guidance_in`) rather than CFG-style batch doubling.
> - **klein (distilled)** uses fixed `guidance_scale=1.0` (no guidance embedding, no CFG).
> - **klein-base (non-distilled)** is the variant in this repo that uses true CFG with a negative prompt.

---

## Architecture Overview

### High-Level Pipeline

```
Text Prompt
    │
    ▼
┌─────────────────┐
│  Text Encoder   │  Mistral3/Pixtral (DEV) or Qwen3 (klein)
│  (VLM-based)    │  Extract embeddings from intermediate layers
└────────┬────────┘
         │
         │  prompt_embeds: [B, seq_len, joint_attention_dim]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Flux2Transformer2DModel                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         Double-Stream Blocks (×8)                    │    │
│  │  ┌─────────────┐      ┌─────────────┐              │    │
│  │  │ Image Stream │◄────►│ Text Stream  │  Joint attn │    │
│  │  │   (hidden)   │      │  (context)   │             │    │
│  │  └─────────────┘      └─────────────┘              │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │        Single-Stream Blocks (×48 DEV / ×24 klein)   │    │
│  │  ┌─────────────────────────────────────────────┐   │    │
│  │  │   Concatenated image + text hidden states    │   │    │
│  │  │   Fused QKV + MLP projection                 │   │    │
│  │  └─────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │  noise prediction
                             ▼
                    ┌─────────────────┐
                    │       VAE       │  AutoencoderKLFlux2
                    │    Decoder      │  32 latent channels
                    └────────┬────────┘
                             │
                             ▼
                        Output Image
```

### Model Variants

#### FLUX.2-DEV (32B rectified-flow transformer)
- HF model card describes the **rectified-flow transformer** as **32B parameters**
- Uses **Mistral3/Pixtral** as the text encoder (Mistral Small 3.1 24B + Pixtral vision tower)
- **Guidance-distilled**: takes a `guidance_scale` scalar (embedded via `guidance_in`), not negative-prompt CFG
- Optimized for quality with ~50 sampling steps

#### FLUX.2-klein-9B (~9B parameters)
- Step-distilled variant trained for 4-step generation
- Uses Qwen3 as text encoder (text-only LLM)
- No guidance embedding in the DiT (and no negative-prompt CFG in the distilled variant)
- Significantly faster inference at slightly reduced quality

#### FLUX.2-klein-base-9B (~9B parameters)
- Non-distilled teacher model for klein-9B
- Same architecture as klein-9B
- Uses 50 sampling steps like DEV
- Base for distillation training

---

## Component Details

### 1. Text Encoder

#### FLUX.2-DEV: Mistral3 with Pixtral Vision

```json
{
  "architectures": ["Mistral3ForConditionalGeneration"],
  "model_type": "mistral3",
  "text_config": {
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 32768,
    "head_dim": 128,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000000.0,
    "vocab_size": 131072
  },
  "vision_config": {
    "model_type": "pixtral",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "patch_size": 14
  }
}
```

**Key characteristics:**
- Grouped Query Attention (GQA): 32 query heads, 8 KV heads (4:1 ratio)
- Published config sets `sliding_window: null` (no sliding-window attention)
- Vision encoder (Pixtral) not used for text-to-image, but model has multimodal capabilities
- RoPE with θ=1,000,000,000

#### FLUX.2-klein: Qwen3

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 4096,
  "num_hidden_layers": 36,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 12288,
  "head_dim": 128,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "vocab_size": 151936,
  "max_position_embeddings": 40960
}
```

**Key characteristics:**
- Text-only LLM (no vision encoder)
- Same GQA ratio as Mistral3 (32:8)
- All 36 layers use full attention (no sliding window)

#### Text Embedding Extraction

Unlike CLIP-based models, FLUX.2 extracts embeddings from **multiple intermediate layers**:

```python
# From diffusers pipeline_flux2.py
hidden_states_layers: List[int] = (10, 20, 30)  # Extract from 3 intermediate layers

# During encoding:
output = text_encoder(input_ids, output_hidden_states=True)
out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
# Shape: [batch, 3, seq_len, hidden_size]

prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
# Final shape for DEV: [batch, seq_len, 15360]  (3 × 5120)
# Final shape for klein: [batch, seq_len, 12288]  (3 × 4096)
```

This multi-layer extraction concatenates representations from different depths of the text encoder, providing richer semantic information.

---

### 2. Transformer (Flux2Transformer2DModel)

The transformer uses a **Multimodal Diffusion Transformer (MMDiT)** architecture with two distinct block types:

#### Double-Stream Blocks

Process image and text separately, then run **joint attention** over the concatenated token sequence (this is *not* classic cross-attention):

```
Input: img (image tokens), txt (text tokens)
       ↓
┌──────────────────────────────────────────────────────────┐
│                    AdaLN Modulation                       │
│  double_stream_modulation_img.lin (6 × hidden_size)      │
│  double_stream_modulation_txt.lin (6 × hidden_size)      │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│                   Joint Attention                         │
│  Image stream:                                            │
│    - img_attn.qkv  (fused QKV)                            │
│    - img_attn.proj (out proj)                             │
│  Text stream:                                             │
│    - txt_attn.qkv  (fused QKV)                            │
│    - txt_attn.proj (out proj)                             │
│  Q/K/V are concatenated across (txt_len + img_len), one   │
│  attention is computed, then the output is split back.    │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│                   Feed-Forward Networks                   │
│  Image: img_mlp.0 → SiLU gate (SwiGLU-style) → img_mlp.2  │
│  Text:  txt_mlp.0 → SiLU gate (SwiGLU-style) → txt_mlp.2  │
└──────────────────────────────────────────────────────────┘
       ↓
Output: updated img, txt
```

**Weight dimensions (BFL-native implementation in this repo):**

Let:
- `D = hidden_size` (DEV: 6144, klein: 4096)
- `r = mlp_ratio` (3.0)
- `mlp_hidden = r * D`

Double-stream attention + MLP (per block, per stream):
- `*.qkv.weight`: `[3D, D]`  (fused QKV projection)
- `*.proj.weight`: `[D, D]`  (attention output projection)
- `*.mlp.0.weight`: `[2*mlp_hidden, D]`  (SwiGLU gate needs 2× inner dim)
- `*.mlp.2.weight`: `[D, mlp_hidden]`

Shared modulation (outside blocks, reused for all blocks):
- `double_stream_modulation_img.lin.weight`: `[6D, D]`  (shift/scale/gate ×2)
- `double_stream_modulation_txt.lin.weight`: `[6D, D]`

Concrete numbers (DEV, `D=6144`, `mlp_hidden=18432`):
- `double_blocks.{N}.img_attn.qkv.weight`: `[18432, 6144]`
- `double_blocks.{N}.img_mlp.0.weight`: `[36864, 6144]`
- `double_stream_modulation_img.lin.weight`: `[36864, 6144]`

#### Single-Stream Blocks

Process concatenated image+text with fused attention:

```
Input: concatenated [txt, img] tokens (all in hidden_size space)
       ↓
┌──────────────────────────────────────────────────────────┐
│                    AdaLN Modulation                       │
│  single_stream_modulation.lin (3 × hidden_size)          │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│         Fused QKV + MLP-in (linear1) + QK Norm            │
│  linear1: [hidden] -> [QKV (3D) | MLP-gate (2*r*D)]       │
│  norm.query_norm / norm.key_norm: RMSNorm scales (D/head) │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│      Fused attn out + MLP-out projection (linear2)        │
│  linear2: concat([attn_out (D), mlp_out (r*D)]) -> D      │
└──────────────────────────────────────────────────────────┘
       ↓
Output: updated concatenated tokens (later split back to txt/img)
```

**Weight dimensions (DEV, `D=6144`, `mlp_hidden=18432`):**
- `single_blocks.{N}.linear1.weight`: `[3D + 2*mlp_hidden, D]` = `[55296, 6144]`
  - split as: `QKV = 3D = 18432`, `MLP-gate = 2*mlp_hidden = 36864`
- `single_blocks.{N}.linear2.weight`: `[D, D + mlp_hidden]` = `[6144, 24576]`
- `single_blocks.{N}.norm.query_norm.scale`: `[head_dim]` = `[128]`
- `single_blocks.{N}.norm.key_norm.scale`: `[head_dim]` = `[128]`
- `single_stream_modulation.lin.weight`: `[3D, D]` = `[18432, 6144]`

#### RoPE (Rotary Position Embedding)

FLUX.2 uses 4D rotary position embeddings:

```python
"axes_dims_rope": [32, 32, 32, 32],  # 4 axes × 32 dims each = 128 total
"rope_theta": 2000
```

The 4 axes likely correspond to:
1. Spatial X position
2. Spatial Y position
3. Temporal/sequence position
4. Additional positional dimension

Total RoPE dimension: 32 × 4 = 128 = head_dim

---

### 3. VAE (AutoencoderKLFlux2)

```json
{
  "_class_name": "AutoencoderKLFlux2",
  "latent_channels": 32,
  "patch_size": [2, 2],
  "in_channels": 3,
  "out_channels": 3,
  "block_out_channels": [128, 256, 512, 512],
  "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D",
                       "DownEncoderBlock2D", "DownEncoderBlock2D"],
  "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D",
                     "UpDecoderBlock2D", "UpDecoderBlock2D"],
  "layers_per_block": 2,
  "mid_block_add_attention": true,
  "use_quant_conv": true,
  "use_post_quant_conv": true,
  "sample_size": 1024
}
```

**Key characteristics:**
- **32 latent channels** (vs 4 for SD, 16 for FLUX.1)
- **16× total spatial compression**: 3 downsampling stages (8×) + 2×2 post-encoder patchification (2×)
- Uses quant_conv layers (KL regularization)

**Latent dimensions:**
```
Input image:      [B,   3,   H,     W    ]
After encoder:    [B,  64,   H/8,   W/8  ]  (3 stride-2 downsamples = 8×, 64 = 2×z_channels for moments)
After mean split: [B,  32,   H/8,   W/8  ]  (take first half of channels)
After 2×2 patchify (rearrange): [B, 128, H/16, W/16]  (fold 2×2 spatial patches into channels)
After BatchNorm:  [B, 128,   H/16,  W/16 ]  (final latent space)

# Example: 1024×1024 image → 64×64 latent (each latent pixel covers 16×16 pixels)

# Transformer expects:
in_channels: 128 = 32 latent_channels × 4 (from 2×2 patchification)
```

> **Note:** The 2×2 patchification happens *after* the encoder via `rearrange("... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)`.
> This is equivalent to `nn.PixelUnshuffle(2)` — spatial information is preserved in the channel dimension.
> Diffusers reports `vae_scale_factor = 8` (encoder only) but internally multiplies by `patch_size` for image alignment, yielding **16× total**.

---

### 4. Scheduler (FlowMatchEulerDiscreteScheduler)

```json
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "max_image_seq_len": 4096,
  "max_shift": 1.15,
  "num_train_timesteps": 1000,
  "shift": 3.0,
  "use_dynamic_shifting": true,
  "time_shift_type": "exponential"
}
```

**Flow Matching formulation:**
- Learns velocity field v(x_t, t) instead of noise ε
- Linear interpolation: x_t = (1-t)x_0 + t·ε
- Loss: MSE between predicted and target velocity

**Dynamic shifting:**
- Adjusts noise schedule based on image resolution
- `base_shift=0.5`, `max_shift=1.15`
- Exponential time shift for better high-resolution handling

---

## Weight Tensor Naming for LoRA

> **Important:** This repo’s FLUX.2 implementation is **BFL-native** (`src/musubi_tuner/flux_2/flux2_models.py`).
> Diffusers uses different module/weight names (e.g. `transformer_blocks.*`, `attn.to_q`, `attn.add_q_proj`), so
> diffusers-style weight paths do **not** match blissful-tuner’s checkpoints or LoRA keys.

### BFL-native base model keys (what actually exists in this repo)

#### Double-Stream Blocks (`double_blocks.{0-7}`)

```text
double_blocks.{N}.img_attn.qkv.weight
double_blocks.{N}.img_attn.proj.weight
double_blocks.{N}.img_attn.norm.query_norm.scale
double_blocks.{N}.img_attn.norm.key_norm.scale

double_blocks.{N}.txt_attn.qkv.weight
double_blocks.{N}.txt_attn.proj.weight
double_blocks.{N}.txt_attn.norm.query_norm.scale
double_blocks.{N}.txt_attn.norm.key_norm.scale

double_blocks.{N}.img_mlp.0.weight
double_blocks.{N}.img_mlp.2.weight
double_blocks.{N}.txt_mlp.0.weight
double_blocks.{N}.txt_mlp.2.weight

# Note: img_norm{1,2} / txt_norm{1,2} are LayerNorm(elementwise_affine=False) → no weight/bias entries.
```

#### Single-Stream Blocks (`single_blocks.{0-47}` for DEV, `{0-23}` for klein)

```text
single_blocks.{N}.linear1.weight
single_blocks.{N}.linear2.weight
single_blocks.{N}.norm.query_norm.scale
single_blocks.{N}.norm.key_norm.scale

# Note: pre_norm is LayerNorm(elementwise_affine=False) → no weight/bias entries.
```

#### Shared / top-level weights

```text
img_in.weight
txt_in.weight

time_in.in_layer.weight
time_in.out_layer.weight

guidance_in.in_layer.weight        # DEV only (use_guidance_embed=True)
guidance_in.out_layer.weight        # DEV only

double_stream_modulation_img.lin.weight
double_stream_modulation_txt.lin.weight
single_stream_modulation.lin.weight

final_layer.linear.weight
final_layer.adaLN_modulation.1.weight
```

### How blissful-tuner names LoRA tensors (safetensors keys)

Musubi/Blissful LoRA safetensors keys are derived from the **module path** by:
1. prefixing with `lora_unet_`
2. replacing `.` with `_`

Example:

```text
Base module: double_blocks.0.img_attn.qkv
LoRA prefix: lora_unet_double_blocks_0_img_attn_qkv
Tensors:     lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight
             lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight
             lora_unet_double_blocks_0_img_attn_qkv.alpha
```

### Recommended LoRA Targets

In blissful-tuner, FLUX.2 LoRA targeting is implemented in `src/musubi_tuner/networks/lora_flux_2.py`:

- Targets by **class name**: `["DoubleStreamBlock", "SingleStreamBlock"]`
- Creates LoRA modules for **Linear layers inside those blocks** (e.g. `*.qkv`, `*.proj`, `*.mlp.*`, `linear1`, `linear2`)
- Does **not** target top-level embeddings/projections (`img_in`, `txt_in`, `time_in`, `guidance_in`, `final_layer`) unless you change the targeting rules
- Excludes modulation and norms by default:
  - `exclude_patterns` includes `.*(img_mod\\.lin|txt_mod\\.lin|modulation\\.lin).*`
  - and any module name containing `norm`

This is why LoRA works without hard-coding exact weight paths, and also why diffusers-style target lists like `to_q/to_k/...` are misleading for this repo.

---

## Training Pipeline Considerations

### Latent Caching

```python
# VAE encoding (returns post-patchification latents)
latents = vae.encode(images)
# Shape: [B, 128, H//16, W//16]
# 128 channels = 32 z_channels × 4 (from 2×2 patchification)
# 16× spatial compression = 8× encoder + 2× patchify
#
# Note: In this repo’s BFL AutoEncoder (`flux2_models.AutoEncoder`), `encode()` already returns normalized latents
# (BatchNorm) and there is no SD-style `vae.config.scaling_factor` step.

# For transformer input, latents are reshaped:
# [B, 128, H//16, W//16] → [B, (H//16)*(W//16), 128]
```

### Text Encoder Caching

```python
# Multi-layer extraction
hidden_states_layers = (10, 20, 30)  # For 40-layer Mistral3
# Adjust for Qwen3 (36 layers): possibly (9, 18, 27)

output = text_encoder(input_ids, output_hidden_states=True)
embeddings = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
prompt_embeds = embeddings.permute(0, 2, 1, 3).reshape(B, seq_len, joint_attention_dim)

# Cache: prompt_embeds of shape [B, seq_len, 15360] (DEV) or [B, seq_len, 12288] (klein)
```

### Flow Matching Loss

```python
# Sample timesteps
t = torch.rand(batch_size, device=device)

# Apply dynamic shift if configured
if use_dynamic_shifting:
    # Shift based on image sequence length
    shift = calculate_shift(image_seq_len, base_shift, max_shift)
    t = shift_timestep(t, shift)

# Linear interpolation
noise = torch.randn_like(latents)
noisy_latents = (1 - t) * latents + t * noise

# Target velocity
target = noise - latents

# Model prediction
pred = transformer(noisy_latents, t, prompt_embeds)

# MSE loss
loss = F.mse_loss(pred, target)
```

### Timestep Sampling Strategies

- **uniform**: Standard uniform sampling t ~ U(0, 1)
- **logit_normal**: Sample from logit-normal distribution (default for many diffusion trainers)
- **shift**: Apply discrete flow shift for better coverage

For FLUX.2, the scheduler uses `shift=3.0` with `time_shift_type="exponential"`.

### Gradient Checkpointing

Enable for memory efficiency:

```python
transformer.enable_gradient_checkpointing()
```

### Mixed Precision

FLUX.2 models are trained in bf16:

```python
# Text encoder
text_encoder.to(dtype=torch.bfloat16)

# Transformer
transformer.to(dtype=torch.bfloat16)

# VAE typically kept in fp32 for encoding quality
vae.to(dtype=torch.float32)
```

### Block Swapping for Memory

For very large models, swap transformer blocks to CPU:

```python
# `--blocks_to_swap` is a weighted heuristic (double-stream blocks are treated as “heavier”).
# DEV practical max is 29 (implementation keeps at least 2 blocks of each type resident).
--blocks_to_swap 20
```

---

## Model Variant Comparison

### FLUX.2-DEV vs FLUX.2-klein

| Aspect | FLUX.2-DEV | FLUX.2-klein-9B |
|--------|------------|-----------------|
| **Use case** | Maximum quality | Fast inference |
| **Parameters** | 32B transformer (+24B text encoder) | 9B transformer (+8B text embedder) |
| **Steps** | 50 | 4 |
| **Negative prompt CFG** | No (guidance-distilled scalar) | No |
| **Text encoder** | Mistral3/Pixtral | Qwen3 |
| **Vision support** | Yes (via Pixtral) | No |
| **Training** | Standard flow matching | Distillation |

### When to Use Each

- **FLUX.2-DEV**: When quality is paramount, computational resources available
- **FLUX.2-klein-9B**: Real-time applications, limited VRAM, batch processing
- **FLUX.2-klein-base-9B**: When you need to fine-tune a non-distilled klein model

---

## Prompting Guidelines

### FLUX.2-DEV Prompting

FLUX.2-DEV supports structured JSON prompting for precise control:

```json
{
  "subject": "A 35-year-old woman",
  "setting": "a sunlit Parisian café terrace",
  "action": "reading a vintage book",
  "style": "cinematic photography, shallow depth of field",
  "lighting": "golden hour, warm sunlight through leaves",
  "colors": "#F5DEB3 warm wheat tones, #8B4513 rich brown accents",
  "camera": "85mm lens, f/1.8, eye-level shot"
}
```

**Key techniques:**
- Use hex colors for precise color specification
- Include typography details when text is needed
- Specify camera parameters for photographic styles

### FLUX.2-klein Prompting

Klein models respond better to **narrative, prose-style prompts**:

```
A weathered lighthouse keeper stands at the edge of a storm-battered cliff,
his silver beard catching the last golden rays of sunset. Salt-worn hands
grip the rusty railing as waves crash below, sending up spray that catches
the dying light like scattered diamonds. The scene is captured in the style
of a master oil painter, with rich amber shadows and luminous highlights.
```

**Key differences from DEV:**
- More descriptive, flowing language
- Emphasis on emotional context and atmosphere
- Lighting descriptions are particularly impactful
- Avoid structured/JSON formats

---

## Implementation Checklist

### Latent Caching Script

- [ ] Load VAE (AutoencoderKLFlux2)
- [ ] Process images through VAE encoder
- [ ] Handle 128 latent channels for the transformer (32 z_channels × 4 patchify), not 4 like SD
- [ ] Account for 2×2 patchification
- [ ] Save with proper dtype (bf16 recommended)
- [ ] Include image dimensions in cache filename

### Text Encoder Caching Script

- [ ] Load appropriate text encoder (Mistral3 or Qwen3)
- [ ] Configure tokenizer with correct special tokens
- [ ] Extract from multiple intermediate layers (10, 20, 30 for Mistral3)
- [ ] Concatenate to joint_attention_dim (15360 for DEV, 12288 for klein)
- [ ] Cache with prompt hash or filename

### Training Script

- [ ] Load Flux2Transformer2DModel
- [ ] Configure LoRA targets (attention + optionally FFN)
- [ ] Handle different block types (double-stream vs single-stream)
- [ ] Implement flow matching loss
- [ ] Support dynamic timestep shifting
- [ ] Handle guidance embeddings (DEV only)
- [ ] Memory optimization (gradient checkpointing, block swapping)
- [ ] Mixed precision (bf16 transformer, fp32 VAE encoding)

### Inference Script

- [ ] Load all components (text encoder, transformer, VAE, scheduler)
- [ ] Configure scheduler for correct number of steps
- [ ] Handle LoRA weight loading/merging
- [ ] Support guidance-distilled `guidance_scale` (DEV + distilled klein)
- [ ] Support true negative-prompt CFG for *base* klein variants (non-distilled)
- [ ] VAE decoding with proper scaling

---

## Differences from FLUX.1

| Aspect | FLUX.1 | FLUX.2 |
|--------|--------|--------|
| VAE latent channels | 16 | 32 (128 after patchify) |
| VAE spatial compression | 16× (4 downsamples) | 16× (3 downsamples + 2×2 patchify) |
| Text encoder | T5 + CLIP | VLM (Mistral3/Qwen3) |
| Text embedding | Dual encoder concat | Multi-layer extraction |
| Typical resolution | 1024×1024 | 1024×1024 (same) |
| Model variants | dev, schnell | dev, klein, klein-base |

---

## Resources

- **Diffusers Pipeline**: `diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline`
- **Transformer Model**: `diffusers.models.transformers.transformer_flux2.Flux2Transformer2DModel`
- **VAE Model**: `diffusers.models.autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2`
- **Scheduler**: `diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler`
- **This repo (BFL-native)**: `src/musubi_tuner/flux_2/flux2_models.py`
- **This repo (loading/packing/denoise)**: `src/musubi_tuner/flux_2/flux2_utils.py`
- **This repo (LoRA targeting)**: `src/musubi_tuner/networks/lora_flux_2.py`
