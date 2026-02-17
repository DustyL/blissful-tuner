# FLUX.2 Architecture Reference

This document provides comprehensive architectural details for the full FLUX.2 model family (DEV, klein-4B, klein-9B, and their base variants) to assist in implementing LoRA training pipelines and debugging.

## Quick Reference

| Parameter | FLUX.2-DEV | FLUX.2-klein-9B | FLUX.2-klein-base-9B | FLUX.2-klein-4B | FLUX.2-klein-base-4B |
|-----------|------------|-----------------|----------------------|-----------------|----------------------|
| **Transformer** |
| Double-stream blocks | 8 | 8 | 8 | **5** | **5** |
| Single-stream blocks | 48 | 24 | 24 | **20** | **20** |
| Total layers | 56 | 32 | 32 | **25** | **25** |
| Attention heads | 48 | 32 | 32 | **24** | **24** |
| Head dimension | 128 | 128 | 128 | 128 | 128 |
| Hidden dimension | 6144 | 4096 | 4096 | **3072** | **3072** |
| Joint attention dim | 15360 | 12288 | 12288 | **7680** | **7680** |
| MLP ratio | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| In channels | 128 | 128 | 128 | 128 | 128 |
| Patch size | 1 | 1 | 1 | 1 | 1 |
| RoPE axes | [32,32,32,32] | [32,32,32,32] | [32,32,32,32] | [32,32,32,32] | [32,32,32,32] |
| RoPE theta | 2000 | 2000 | 2000 | 2000 | 2000 |
| Guidance embeds | true | false | false | false | false |
| Timestep guidance channels | 256 | 256 | 256 | 256 | 256 |
| **Text Encoder** |
| Architecture | Mistral3/Pixtral | Qwen3-8B | Qwen3-8B | **Qwen3-4B** | **Qwen3-4B** |
| Hidden size | 5120 | 4096 | 4096 | **2560** | **2560** |
| Layers | 40 | 36 | 36 | **36** | **36** |
| Attention heads | 32 | 32 | 32 | **20** | **20** |
| KV heads (GQA) | 8 | 8 | 8 | **4** | **4** |
| Head dim | 128 | 128 | 128 | 128 | 128 |
| Intermediate size | 32768 | 12288 | 12288 | **6912** | **6912** |
| Vocab size | 131072 | 151936 | 151936 | 151936 | 151936 |
| Extraction layers | [10, 20, 30] | [9, 18, 27] | [9, 18, 27] | [9, 18, 27] | [9, 18, 27] |
| Vision encoder | Pixtral (1024 hidden) | None | None | None | None |
| Max token length | 512 | 512 | 512 | 512 | 512 |
| **VAE** (shared across all variants) |
| Latent channels | 32 | 32 | 32 | 32 | 32 |
| Patch size | [2, 2] | [2, 2] | [2, 2] | [2, 2] | [2, 2] |
| Spatial compression | 16× | 16× | 16× | 16× | 16× |
| Block channels | [128,256,512,512] | [128,256,512,512] | [128,256,512,512] | [128,256,512,512] | [128,256,512,512] |
| **Training / Inference Defaults** |
| Distillation | Guidance-distilled | 4-step distilled | None (base teacher) | 4-step distilled | None (base teacher) |
| Default steps | 50 | 4 (fixed) | 50 | 4 (fixed) | 50 |
| Default guidance | 4.0 | 1.0 (fixed) | 4.0 | 1.0 (fixed) | 4.0 |
| Architecture code | `f2d` | `f2k9b` | `f2k9b` | `f2k4b` | `f2k4b` |
| Diffusers pipeline | Flux2Pipeline | Flux2KleinPipeline | Flux2KleinPipeline | Flux2KleinPipeline | Flux2KleinPipeline |

> **Guidance vs. CFG note:** In FLUX.2, "guidance" is not always "negative-prompt classifier-free guidance (CFG)".
> - **DEV** is guidance-distilled and uses an explicit guidance scalar embedding (`guidance_in`) rather than CFG-style batch doubling. During training, guidance is always set to 1.0.
> - **klein (distilled)** uses fixed `guidance_scale=1.0` (no guidance embedding, no CFG). Guidance and steps cannot be changed.
> - **klein-base (non-distilled)** is the variant in this repo that uses true CFG with a negative prompt via `denoise_cfg()`.
>
> **Code reference:** `FLUX2_MODEL_INFO` in `src/musubi_tuner/flux_2/flux2_utils.py` is the canonical source for defaults and fixed params per variant.

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
│  │     Double-Stream Blocks (×8 DEV/9B, ×5 klein-4B)  │    │
│  │  ┌─────────────┐      ┌─────────────┐              │    │
│  │  │ Image Stream │◄────►│ Text Stream  │  Joint attn │    │
│  │  │   (hidden)   │      │  (context)   │             │    │
│  │  └─────────────┘      └─────────────┘              │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │    Single-Stream (×48 DEV / ×24 klein-9B / ×20 4B) │    │
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
- Uses true CFG with negative prompt (via `denoise_cfg()`)
- Ideal for LoRA fine-tuning (full training signal preserved)

#### FLUX.2-klein-4B (~4B parameters)
- Step-distilled variant trained for 4-step generation
- Uses **Qwen3-4B** as text encoder (smaller than 9B's Qwen3-8B)
- **Architecturally distinct**: only 5 double-stream blocks (vs 8 for all other variants) and 20 single-stream blocks
- Hidden dimension of 3072 with 24 attention heads
- Fixed guidance=1.0, steps=4

#### FLUX.2-klein-base-4B (~4B parameters)
- Non-distilled teacher model for klein-4B
- Same architecture as klein-4B (5+20 blocks, 3072 hidden dim)
- Uses 50 sampling steps with true CFG
- Base for distillation training

---

## Component Details

### 1. Text Encoder

#### FLUX.2-DEV: Mistral3 with Pixtral Vision

```json
{
  "architectures": ["Mistral3ForConditionalGeneration"],
  "model_type": "mistral3",
  "dtype": "bfloat16",
  "image_token_index": 10,
  "spatial_merge_size": 2,
  "projector_hidden_act": "gelu",
  "multimodal_projector_bias": false,
  "vision_feature_layer": -1,
  "text_config": {
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 32768,
    "head_dim": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000000.0,
    "vocab_size": 131072,
    "max_position_embeddings": 131072,
    "sliding_window": null,
    "attention_dropout": 0.0
  },
  "vision_config": {
    "model_type": "pixtral",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "head_dim": 64,
    "intermediate_size": 4096,
    "patch_size": 14,
    "image_size": 1540,
    "hidden_act": "silu",
    "rope_theta": 10000.0
  }
}
```

**Key characteristics:**
- Grouped Query Attention (GQA): 32 query heads, 8 KV heads (4:1 ratio)
- Published config sets `sliding_window: null` (no sliding-window attention)
- Max context: 131072 tokens
- RoPE with θ=1,000,000,000 (text) / θ=10,000 (vision)
- **Pixtral vision encoder**: 1540×1540 input, 14×14 patches (~110 patches per side, ~12k patch tokens)
- `spatial_merge_size: 2` reduces vision tokens by 4× (2×2 merge → ~3k tokens after projection)
- `image_token_index: 10` maps to the `[IMG]` special token
- Tokenizer: `PixtralProcessor` with `LlamaTokenizerFast` (131072 vocab)

#### FLUX.2-klein-9B: Qwen3-8B

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "dtype": "bfloat16",
  "hidden_size": 4096,
  "num_hidden_layers": 36,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 12288,
  "head_dim": 128,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "attention_bias": false,
  "sliding_window": null,
  "tie_word_embeddings": false
}
```

**Key characteristics:**
- Text-only LLM (no vision encoder)
- GQA ratio: 32 query heads, 8 KV heads (4:1)
- All 36 layers use `full_attention` (no sliding window)
- RoPE θ=1,000,000 (vs 1 billion for Mistral3)
- Tokenizer: `Qwen2TokenizerFast` (151936 vocab, `<|im_start|>`/`<|im_end|>` markers)
- Supports FP8 inference via `--fp8_text_encoder`

#### FLUX.2-klein-4B: Qwen3-4B

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "dtype": "bfloat16",
  "hidden_size": 2560,
  "num_hidden_layers": 36,
  "num_attention_heads": 20,
  "num_key_value_heads": 4,
  "intermediate_size": 6912,
  "head_dim": 128,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "attention_bias": false,
  "sliding_window": null,
  "tie_word_embeddings": true
}
```

**Key characteristics:**
- Same depth as 8B (36 layers) but narrower (2560 vs 4096 hidden dim)
- GQA ratio: 20 query heads, 4 KV heads (5:1 — more aggressive than 8B)
- Shared tokenizer with 8B variant (`Qwen2TokenizerFast`, 151936 vocab)
- `tie_word_embeddings: true` (unlike 8B which uses separate input/output embeddings)
- joint_attention_dim = 3 × 2560 = 7680 (from 3-layer extraction)

#### Text Embedding Extraction

Unlike CLIP-based models, FLUX.2 extracts embeddings from **multiple intermediate layers** and concatenates them:

```python
# From src/musubi_tuner/flux_2/flux2_utils.py
OUTPUT_LAYERS_MISTRAL = [10, 20, 30]  # Mistral3 (40-layer): layers 10, 20, 30
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]    # Qwen3 (36-layer): layers 9, 18, 27
MAX_LENGTH = 512                      # Maximum token length for all variants

# During encoding:
output = text_encoder(input_ids, output_hidden_states=True)
out = torch.stack([output.hidden_states[k] for k in extraction_layers], dim=1)
# Shape: [batch, 3, seq_len, hidden_size]

prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, 3 * hidden_dim)
# Final shapes:
#   DEV:      [batch, seq_len, 15360]  (3 × 5120)
#   klein-9B: [batch, seq_len, 12288]  (3 × 4096)
#   klein-4B: [batch, seq_len,  7680]  (3 × 2560)
```

The multi-layer extraction concatenates representations from different depths of the text encoder, providing richer semantic information. The extraction layer indices are evenly spaced at roughly layers 25%, 50%, 75% through the encoder.

#### Text Encoding Pipeline

**HuggingFace Model IDs (from BFL reference):**

| Variant | Model ID | Processor/Tokenizer ID |
|---------|----------|----------------------|
| DEV (Mistral3) | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| Klein 9B (Qwen3-8B) | `Qwen/Qwen3-8B` (or FP8 variant) | Same as model |
| Klein 4B (Qwen3-4B) | `Qwen/Qwen3-4B` (or FP8 variant) | Same as model |

> **Note:** The DEV model and processor use *different* Mistral3 versions (3.2 for model, 3.1 for processor). In blissful-tuner, the tokenizer ID is hardcoded as `M3_TOKENIZER_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"`.

**System Message (Mistral3 only):**

All Mistral3 text encoding prepends a system message in chat format:

```
"You are an AI that reasons about image descriptions. You give structured responses
focusing on object relationships, object attribution and actions without speculation."
```

This shapes how the text encoder interprets prompts — it's not an optional prompt-engineering detail but a hard-coded part of the encoding pipeline. Both BFL reference and blissful-tuner use this same system message.

**Chat Template Formatting:**

```python
# Mistral3 (DEV) — system message + user prompt, max 512 tokens
messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]
inputs = processor.apply_chat_template(messages, tokenize=True, max_length=512, ...)

# Qwen3 (Klein) — user prompt only, no system message, thinking disabled
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,  # Prevents <think> blocks during embedding extraction
)
```

> **Why `enable_thinking=False`?** Qwen3 models have an extended thinking mode that generates internal reasoning tokens before responding. When using the model as a *text encoder* (extracting hidden states, not generating text), these thinking tokens would corrupt the embedding extraction. Disabling it ensures clean, deterministic hidden state extraction.

**NSFW Filtering (BFL reference only):**

The BFL reference implementation includes a two-tier safety system in the Mistral3 encoder:
1. **Image classification**: `Falconsai/nsfw_image_detection` model with threshold 0.85
2. **Content integrity**: Uses the Mistral3 model itself to detect copyright characters/public figures (constrained yes/no generation)

Blissful-tuner does **not** include these safety filters — they are BFL API-specific and not relevant to local training/inference.

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
- `D = hidden_size` (DEV: 6144, klein-9B: 4096, klein-4B: 3072)
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

The 4 axes correspond to the 4D position ID system (confirmed from BFL reference):
1. **Time** (t) — temporal/reference offset
2. **Height** (h) — spatial row position
3. **Width** (w) — spatial column position
4. **Language/sequence** (l) — token sequence index

Total RoPE dimension: 32 × 4 = 128 = head_dim

#### Position ID System

FLUX.2 uses a 4D position ID scheme to distinguish different token types within the same attention context:

```python
# Position ID format: (t, h, w, l) — 4 dimensions per token

# Text tokens: only sequence index varies
text_ids[i] = [t=0, h=0, w=0, l=i]        # i = 0..seq_len-1

# Image latent tokens: spatial coordinates vary
img_ids[h, w] = [t=0, h=h, w=w, l=0]      # h = 0..H-1, w = 0..W-1

# Reference image tokens: time offset distinguishes from target image
ref_ids[h, w] = [t=offset, h=h, w=w, l=0]
# offset = scale + scale * ref_index (scale=10)
# First reference: t=10, second: t=20, third: t=30, etc.
```

This allows the model to attend across text, image, and reference tokens simultaneously while maintaining spatial and temporal awareness through RoPE.

#### Reference Image / Control Latent Handling

Reference images (for editing/I2I) are encoded through the VAE and concatenated along the **sequence dimension**:

```python
# In denoise() — from BFL sampling.py:
img_input = torch.cat([img, img_cond_seq], dim=1)       # Concat along seq dim
img_input_ids = torch.cat([img_ids, img_cond_seq_ids], dim=1)

# Model sees: [noisy_image_tokens, reference_image_tokens] as one sequence
pred = model(x=img_input, x_ids=img_input_ids, ...)

# Only the noisy image portion of the output is used:
pred = pred[:, :img.shape[1], ...]  # Slice to original image length
```

**Pixel limits:**
- **BFL reference**: Single reference up to **2024² pixels** (~4M), multiple references up to **1024² pixels** each
- **Diffusers**: Enforces **1024² pixels** (1M) max area for *all* reference images (auto-resizes if exceeded)
- **Blissful-tuner**: Follows BFL limits by default; `--no_resize_control` skips resize entirely
- All images center-cropped to multiples of 16 (VAE alignment) in BFL/blissful-tuner, multiples of 32 (`vae_scale_factor * 2`) in diffusers

**Image validation constraints (from diffusers):**
- Minimum side length: **64px**
- Maximum aspect ratio: **8:1**
- Height/width must be divisible by 16 (or 32 in diffusers)

> **Batched inference limitation (diffusers):** Cannot support batched inference with different image resolutions or text prompt lengths in the same batch. All batch items must have the same spatial dimensions and sequence lengths. This is noted explicitly in the diffusers transformer code.

---

### 3. VAE (AutoencoderKLFlux2)

The VAE is **shared across all FLUX.2 variants** (all Klein configs reference `black-forest-labs/FLUX.2-dev` as the source).

```json
{
  "_class_name": "AutoencoderKLFlux2",
  "act_fn": "silu",
  "batch_norm_eps": 0.0001,
  "batch_norm_momentum": 0.1,
  "block_out_channels": [128, 256, 512, 512],
  "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D",
                       "DownEncoderBlock2D", "DownEncoderBlock2D"],
  "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D",
                     "UpDecoderBlock2D", "UpDecoderBlock2D"],
  "force_upcast": true,
  "in_channels": 3,
  "out_channels": 3,
  "latent_channels": 32,
  "layers_per_block": 2,
  "mid_block_add_attention": true,
  "norm_num_groups": 32,
  "patch_size": [2, 2],
  "sample_size": 1024,
  "use_quant_conv": true,
  "use_post_quant_conv": true
}
```

**Key characteristics:**
- **32 latent channels** (vs 4 for SD, 16 for FLUX.1)
- **16× total spatial compression**: 3 downsampling stages (8×) + 2×2 post-encoder patchification (2×)
- Uses quant_conv layers (KL regularization)
- `force_upcast: true` — VAE operations run at higher precision for numerical stability (always loaded as fp32)
- `batch_norm_eps: 0.0001`, `batch_norm_momentum: 0.1` — running stats tracking in encoder
- `norm_num_groups: 32` — GroupNorm used throughout
- Activation: SiLU (Swish)

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

> **Implementation difference — Patchification and BatchNorm location:**
> - **BFL reference / blissful-tuner**: Patchification and BatchNorm happen *inside* the VAE `encode()` method. The VAE returns fully patchified, normalized latents ready for the transformer.
> - **Diffusers**: The VAE `encode()` returns raw 32-channel latents (no patchification). The pipeline applies patchification via `_patchify_latents()`, then manually denormalizes using `vae.bn.running_mean` and `vae.bn.running_var`. Decoding reverses this: renormalize → `_unpatchify_latents()` → `vae.decode()`.
>
> This means **BFL-format cached latents are not directly compatible with diffusers** — the patchification and BatchNorm are baked in at different stages. Blissful-tuner uses BFL-format latents (patchified + normalized).
>
> The VAE's `patch_size` config parameter exists in diffusers but is **not used** by the VAE forward pass — it's only used by the pipeline's external patchification logic. The `bn` layer's running statistics are preserved from BFL pretraining and accessed by the pipeline.

---

### 4. Scheduler (FlowMatchEulerDiscreteScheduler)

The scheduler config is **identical across all FLUX.2 variants**.

```json
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "invert_sigmas": false,
  "max_image_seq_len": 4096,
  "max_shift": 1.15,
  "num_train_timesteps": 1000,
  "shift": 3.0,
  "shift_terminal": null,
  "stochastic_sampling": false,
  "time_shift_type": "exponential",
  "use_beta_sigmas": false,
  "use_dynamic_shifting": true,
  "use_exponential_sigmas": false,
  "use_karras_sigmas": false
}
```

**Flow Matching formulation:**
- Learns velocity field v(x_t, t) instead of noise ε
- Linear interpolation: x_t = (1-t)x_0 + t·ε
- Loss: MSE between predicted and target velocity

**Dynamic shifting:**
- Adjusts noise schedule based on image resolution (sequence length)
- `base_shift=0.5`, `max_shift=1.15`
- Exponential time shift for better high-resolution handling
- `base_image_seq_len=256` (reference: 16×16 latent grid ≈ 256×256 image at 16× compression)
- `max_image_seq_len=4096` (maximum: 64×64 latent grid)

**Empirical schedule in this repo** (`get_schedule()` in `flux2_utils.py`):
- When `--flow_shift <value>` is provided: `t' = (t × shift) / (1 + (shift - 1) × t)` — compresses timestep distribution toward high noise
- Without `--flow_shift`: uses an empirical mu calculation based on image sequence length and number of steps, with different linear interpolation coefficients for 10-step vs 200-step regimes

---

## Weight Tensor Naming for LoRA

> **Important:** This repo’s FLUX.2 implementation is **BFL-native** (`src/musubi_tuner/flux_2/flux2_models.py`).
> Diffusers uses different module/weight names (e.g. `transformer_blocks.*`, `attn.to_q`, `attn.add_q_proj`), so
> diffusers-style weight paths do **not** match blissful-tuner’s checkpoints or LoRA keys.

### BFL-native base model keys (what actually exists in this repo)

#### Double-Stream Blocks (`double_blocks.{0-7}` for DEV/9B, `{0-4}` for 4B)

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

#### Single-Stream Blocks (`single_blocks.{0-47}` for DEV, `{0-23}` for 9B, `{0-19}` for 4B)

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

### Weight Format Conversion (BFL ↔ Diffusers)

FLUX.2 has three weight naming conventions. When converting LoRAs between ecosystems, the key mappings and transformations below apply.

#### Top-Level Key Renames

| BFL / Blissful-Tuner | Diffusers |
|---------------------|-----------|
| `img_in` | `x_embedder` |
| `txt_in` | `context_embedder` |
| `time_in.in_layer` | `time_guidance_embed.timestep_embedder.linear_1` |
| `time_in.out_layer` | `time_guidance_embed.timestep_embedder.linear_2` |
| `guidance_in.in_layer` | `time_guidance_embed.guidance_embedder.linear_1` |
| `guidance_in.out_layer` | `time_guidance_embed.guidance_embedder.linear_2` |
| `double_stream_modulation_img.lin` | `double_stream_modulation_img.linear` |
| `double_stream_modulation_txt.lin` | `double_stream_modulation_txt.linear` |
| `single_stream_modulation.lin` | `single_stream_modulation.linear` |
| `final_layer.linear` | `proj_out` |
| `final_layer.adaLN_modulation.1` | `norm_out.linear` |

#### Double-Stream Block Key Renames

| BFL / Blissful-Tuner | Diffusers |
|---------------------|-----------|
| `double_blocks.{N}.img_attn.qkv` | **Split** → `transformer_blocks.{N}.attn.to_q/to_k/to_v` |
| `double_blocks.{N}.txt_attn.qkv` | **Split** → `transformer_blocks.{N}.attn.add_q_proj/add_k_proj/add_v_proj` |
| `double_blocks.{N}.img_attn.proj` | `transformer_blocks.{N}.attn.to_out.0` |
| `double_blocks.{N}.txt_attn.proj` | `transformer_blocks.{N}.attn.to_add_out` |
| `double_blocks.{N}.img_attn.norm.query_norm` | `transformer_blocks.{N}.attn.norm_q` |
| `double_blocks.{N}.img_attn.norm.key_norm` | `transformer_blocks.{N}.attn.norm_k` |
| `double_blocks.{N}.txt_attn.norm.query_norm` | `transformer_blocks.{N}.attn.norm_added_q` |
| `double_blocks.{N}.txt_attn.norm.key_norm` | `transformer_blocks.{N}.attn.norm_added_k` |
| `double_blocks.{N}.img_mlp.0` | `transformer_blocks.{N}.ff.linear_in` |
| `double_blocks.{N}.img_mlp.2` | `transformer_blocks.{N}.ff.linear_out` |
| `double_blocks.{N}.txt_mlp.0` | `transformer_blocks.{N}.ff_context.linear_in` |
| `double_blocks.{N}.txt_mlp.2` | `transformer_blocks.{N}.ff_context.linear_out` |

#### Single-Stream Block Key Renames

| BFL / Blissful-Tuner | Diffusers |
|---------------------|-----------|
| `single_blocks.{N}.linear1` | `single_transformer_blocks.{N}.attn.to_qkv_mlp_proj` |
| `single_blocks.{N}.linear2` | `single_transformer_blocks.{N}.attn.to_out` |
| `single_blocks.{N}.norm.query_norm` | `single_transformer_blocks.{N}.attn.norm_q` |
| `single_blocks.{N}.norm.key_norm` | `single_transformer_blocks.{N}.attn.norm_k` |

> **RMSNorm naming:** BFL uses `.scale`, diffusers uses `.weight` for RMSNorm parameters.

#### Critical Conversion Transformations

**1. AdaLN Scale/Shift Swap:**

BFL stores modulation parameters as `[shift, scale]`, diffusers stores them as `[scale, shift]`. The `final_layer.adaLN_modulation.1.weight` must be swapped:
```python
shift, scale = weight.chunk(2, dim=0)
new_weight = torch.cat([scale, shift], dim=0)  # BFL → Diffusers
```

**2. QKV Split (for fused QKV weights):**

BFL uses fused QKV projections (`img_attn.qkv`). Diffusers splits these into separate `to_q`, `to_k`, `to_v`:
```python
q, k, v = torch.chunk(fused_qkv_weight, 3, dim=0)
```

**3. LoRA QKV Conversion (critical for LoRA portability):**

When converting BFL-format LoRA to diffusers:
- **`lora_A` (lora_down)**: NOT split — the fused weight is **replicated** to all three Q/K/V projections
- **`lora_B` (lora_up)**: IS split via `torch.chunk(3, dim=0)` into separate Q, K, V weights

This asymmetry is because `lora_A` projects from the same input space (shared), while `lora_B` projects to different output spaces (Q, K, V dimensions).

**4. LoRA Key Naming:**
- BFL: `diffusion_model.{key}.lora_down.weight` / `.lora_up.weight`
- Diffusers: `transformer.{key}.lora_A.weight` / `.lora_B.weight`
- Blissful-tuner: `lora_unet_{key_with_dots_as_underscores}.lora_down.weight` / `.lora_up.weight`

> **Note:** Blissful-tuner's `convert_lora.py` handles BFL ↔ musubi ↔ diffusers conversions. The `--format` flag selects the target format.

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
# Multi-layer extraction (confirmed from flux2_utils.py)
OUTPUT_LAYERS_MISTRAL = [10, 20, 30]  # Mistral3 (DEV): layers 10, 20, 30 of 40
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]    # Qwen3 (klein): layers 9, 18, 27 of 36
MAX_LENGTH = 512                      # Max tokens for all variants

output = text_encoder(input_ids, output_hidden_states=True)
embeddings = torch.stack([output.hidden_states[k] for k in extraction_layers], dim=1)
prompt_embeds = embeddings.permute(0, 2, 1, 3).reshape(B, seq_len, joint_attention_dim)

# Cache shapes:
#   DEV:      [B, seq_len, 15360]  (3 × 5120, Mistral3)
#   klein-9B: [B, seq_len, 12288]  (3 × 4096, Qwen3-8B)
#   klein-4B: [B, seq_len,  7680]  (3 × 2560, Qwen3-4B)
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
# `--blocks_to_swap N` uses a ratio-based distribution algorithm:
#   swap_ratio = num_single_blocks / num_double_blocks
#   double_blocks_to_swap = round(N / (1.0 + swap_ratio / 2.0))
#   single_blocks_to_swap = round(double_blocks_to_swap * swap_ratio)
#
# Implementation keeps at least 2 blocks of each type on GPU.
#
# DEV (8 double + 48 single):
#   --blocks_to_swap 20 → ~3 double, ~18 single swapped
#   Maximum safe: 6 double + 42 single (leaves 2 resident per type)
#
# Klein-9B (8 double + 24 single):
#   --blocks_to_swap 16 → ~4 double, ~12 single swapped
#
# Klein-4B (5 double + 20 single):
#   --blocks_to_swap 12 → ~3 double, ~9 single swapped
--blocks_to_swap 20
```

Block swapping uses `ModelOffloader` instances (one for double blocks, one for single blocks) with optional pinned memory for faster PCIe transfers.

---

## Model Variant Comparison

### Full Variant Comparison

| Aspect | FLUX.2-DEV | klein-9B | klein-base-9B | klein-4B | klein-base-4B |
|--------|------------|----------|---------------|----------|---------------|
| **Use case** | Max quality | Fast 9B | Fine-tune 9B | Fast 4B | Fine-tune 4B |
| **Transformer params** | ~32B | ~9B | ~9B | ~4B | ~4B |
| **Text encoder** | Mistral3/Pixtral | Qwen3-8B | Qwen3-8B | Qwen3-4B | Qwen3-4B |
| **Total layers** | 56 (8+48) | 32 (8+24) | 32 (8+24) | 25 (5+20) | 25 (5+20) |
| **Default steps** | 50 | 4 (fixed) | 50 | 4 (fixed) | 50 |
| **Default guidance** | 4.0 | 1.0 (fixed) | 4.0 | 1.0 (fixed) | 4.0 |
| **Guidance distilled** | Yes (embedded) | Yes (step) | No | Yes (step) | No |
| **Negative prompt CFG** | No | No | Yes | No | Yes |
| **Vision support** | Yes (Pixtral) | No | No | No | No |
| **Denoise function** | `denoise()` | `denoise()` | `denoise_cfg()` | `denoise()` | `denoise_cfg()` |

### When to Use Each

- **FLUX.2-DEV**: Maximum quality, computational resources available, need vision/editing capabilities
- **FLUX.2-klein-9B**: Real-time applications, limited VRAM, batch processing
- **FLUX.2-klein-base-9B**: LoRA fine-tuning on the 9B architecture (full training signal, no distillation artifacts)
- **FLUX.2-klein-4B**: Ultra-fast inference, very limited VRAM, edge deployment
- **FLUX.2-klein-base-4B**: LoRA fine-tuning on the 4B architecture (smallest trainable model)

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

- [ ] Load appropriate text encoder (Mistral3 for DEV, Qwen3-8B for 9B, Qwen3-4B for 4B)
- [ ] Configure tokenizer (PixtralProcessor for DEV, Qwen2TokenizerFast for klein)
- [ ] Extract from correct intermediate layers: [10,20,30] for Mistral3, [9,18,27] for Qwen3
- [ ] Concatenate to joint_attention_dim (15360 for DEV, 12288 for 9B, 7680 for 4B)
- [ ] Respect MAX_LENGTH=512 token limit
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
| Text encoder | T5 + CLIP | VLM (Mistral3 or Qwen3, variant-dependent) |
| Text embedding | Dual encoder concat | Multi-layer extraction (3 layers) |
| Typical resolution | 1024×1024 | 1024×1024 (same) |
| Model variants | dev, schnell | dev, klein-4b, klein-9b, klein-base-4b, klein-base-9b |
| Variant architectures | Single arch | 3 distinct (DEV, 9B, 4B differ in depth/width) |

---

## Architecture Codes and Cache Filenames

This repo uses short architecture codes in latent/text cache filenames to avoid cross-contamination between variants:

| Variant | Short Code | Full Name | Cache Filename Example |
|---------|-----------|-----------|------------------------|
| DEV | `f2d` | `flux_2_dev` | `image_1280x832_f2d.safetensors` |
| klein-4B (& base) | `f2k4b` | `flux_2_klein_4b` | `image_1280x832_f2k4b.safetensors` |
| klein-9B (& base) | `f2k9b` | `flux_2_klein_9b` | `image_1280x832_f2k9b.safetensors` |

> **Note:** Distilled and base variants of the same size share the same architecture code (e.g., both `klein-9b` and `klein-base-9b` use `f2k9b`) since their architectures are identical — only the weights differ.

Defined in `src/musubi_tuner/networks/network_arch.py` as `ARCHITECTURE_FLUX_2_DEV`, `ARCHITECTURE_FLUX_2_KLEIN_4B`, `ARCHITECTURE_FLUX_2_KLEIN_9B`.

---

## Attention Modes

FLUX.2 supports the following attention implementations:

| Mode | Device | Notes |
|------|--------|-------|
| `torch` | Any | Standard PyTorch SDPA (default) |
| `flash` | CUDA only | Flash Attention (fastest) |
| `xformers` | CUDA only | Memory-efficient attention |
| `sageattn` | CUDA only | SageAttention |

> **Not supported:** `flash3`, `cute` — these will raise an explicit error for FLUX.2.

---

## Implementation Comparison (BFL / Diffusers / Blissful-Tuner)

This section documents notable differences between the three FLUX.2 implementations. Understanding these is important for debugging training/inference discrepancies and ensuring weight/LoRA portability.

### Numerical Precision

| Component | BFL Reference | Diffusers | Blissful-Tuner |
|-----------|---------------|-----------|----------------|
| **RoPE frequency scale** | `pos.dtype` (typically bf16) | `torch.float64` (float32 on MPS/NPU) | `torch.float64` |
| **RoPE output** | `.float()` (fp32) | `.float()` (fp32) | `.float()` (fp32) |
| **apply_rope()** | `.float()` intermediate, `.type_as()` return | `.float()` intermediate, `.to(x.dtype)` return | `.float()` intermediate |
| **RMSNorm** | `.float()` → compute → `.to(dtype=x_dtype)` | `.to(torch.float32).pow(2)` → compute → conditional dtype conversion | Same as BFL |
| **QKNorm** | `.to(v)` (cast q,k to match v dtype) | Per-head RMSNorm (learnable scale) | Same as BFL |

> **Key difference:** BFL uses `pos.dtype` for RoPE frequency scale, while both diffusers and blissful-tuner use `torch.float64` for higher precision in the `1.0 / (theta ** scale)` computation.

### Attention Backend

| | BFL Reference | Diffusers | Blissful-Tuner |
|-|---------------|-----------|----------------|
| **Implementation** | Direct `F.scaled_dot_product_attention` | `dispatch_attention_fn()` with backend registry | `unified_attention()` wrapper |
| **Backends** | PyTorch SDPA only | sdpa, xformers, flash_attention (via registry) | torch (SDPA), flash, xformers, sageattn |
| **Selection** | Hardcoded | Backend registry + context managers | Runtime CLI flags |

### VAE and Latent Pipeline

| Aspect | BFL Reference | Diffusers | Blissful-Tuner |
|--------|---------------|-----------|----------------|
| **Patchification location** | Inside VAE `encode()` | Pipeline `_patchify_latents()` | Inside VAE `encode()` |
| **BatchNorm location** | Inside VAE (post-patchify) | Pipeline (manual `vae.bn.running_mean/var`) | Inside VAE (post-patchify) |
| **VAE output** | Patchified + normalized latents | Raw 32-channel latents | Patchified + normalized latents |
| **Scaling factor** | None (raw latents) | None (raw latents) | None (raw latents) |
| **Latent format compatibility** | BFL native | Different (no patchification in latents) | BFL-compatible |

> **Important:** BFL/blissful-tuner cached latents are **not directly compatible** with diffusers pipelines because patchification and BatchNorm are applied at different stages. The latent values themselves are equivalent after accounting for the transformation ordering.

### FP16 Safety

| | BFL Reference | Diffusers | Blissful-Tuner |
|-|---------------|-----------|----------------|
| **Overflow protection** | None | `hidden_states.clip(-65504, 65504)` in both block types | None |

Diffusers adds explicit FP16 overflow clamping to prevent NaN propagation when running in float16 precision. BFL reference and blissful-tuner do not include this safeguard (both expect bf16 or fp32).

### Reference Image Handling

| | BFL Reference | Diffusers | Blissful-Tuner |
|-|---------------|-----------|----------------|
| **Max area (single ref)** | 2024² pixels (~4M) | 1024² pixels (1M) | 2024² (BFL default) |
| **Max area (multi ref)** | 1024² each | 1024² each | 1024² each |
| **Alignment** | Multiples of 16 | Multiples of 32 (`vae_scale_factor * 2`) | Multiples of 16 |
| **Resize behavior** | Center crop | Auto-resize to target area (LANCZOS) | `--no_resize_control` to skip |

### Guidance / CFG Approach

| Variant | BFL Reference | Diffusers | Blissful-Tuner |
|---------|---------------|-----------|----------------|
| **DEV** | `denoise()` with embedded guidance | Single forward pass, `guidance` param to transformer | `denoise()` with embedded guidance |
| **Klein distilled** | `denoise()` with guidance=1.0 (fixed) | Single forward pass, `guidance=None`, `is_distilled=True` | `denoise()` with guidance=1.0 |
| **Klein base** | `denoise_cfg()` with batch doubling | Two forward passes with `cache_context("cond"/"uncond")` | `denoise_cfg()` with batch doubling |

> **Diffusers-specific:** Klein pipeline uses KV cache context management (`transformer.cache_context("cond"/"uncond")`) for efficient CFG, avoiding redundant computation in the two-pass loop. BFL and blissful-tuner use simple batch doubling.

### Denoise Functions

All three implementations provide two denoise modes:

```python
# denoise() — for distilled models (DEV, klein distilled)
# Uses embedded guidance (guidance_vec passed to model.guidance_in)
# No batch doubling — single forward pass per step

# denoise_cfg() — for base/teacher models (klein-base-4B, klein-base-9B)
# Uses classifier-free guidance with batch doubling
# text input is pre-concatenated as [txt_empty, txt_prompt]
# Formula: pred = pred_uncond + guidance * (pred_cond - pred_uncond)
```

### Scheduling

All three use the same empirical schedule computation:

```python
def get_schedule(num_steps, image_seq_len):
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    return generalized_time_snr_shift(timesteps, mu, sigma=1.0)

# compute_empirical_mu() uses piecewise linear interpolation:
#   a1=8.73809524e-05, b1=1.89833333 (short schedules)
#   a2=0.00016927,     b2=0.45666666 (long schedules)
```

### Features Unique to Blissful-Tuner

These capabilities are **not present** in the BFL reference or diffusers implementations:

| Feature | Description | Relevant Code |
|---------|-------------|---------------|
| **Block swapping** | CPU offload of transformer blocks with pinned memory | `flux2_models.py: enable_block_swap()` |
| **FP8 quantization** | Dynamic FP8 at load time (excluding norms, embedders, modulation) | `flux2_models.py: FP8_OPTIMIZATION_TARGET_KEYS` |
| **LoRA training** | Module-level LoRA injection targeting `DoubleStreamBlock`/`SingleStreamBlock` | `networks/lora_flux_2.py` |
| **LoRA weight merging** | Merge LoRA weights at load time (no runtime hooks) | `flux2_utils.py: load_safetensors_with_lora_and_fp8()` |
| **Mask-weighted loss** | Spatial mask-weighted training loss with prior preservation | `modules/mask_loss.py` |
| **Latent preview** | Real-time visualization during generation | `blissful_tuner/latent_preview.py` |
| **Advanced CFG scheduling** | CFGZero*, NAG, perpendicular CFG | `blissful_tuner/guidance.py` |

### Features Unique to Diffusers

| Feature | Description |
|---------|-------------|
| **Context parallelism** | Distributed training support with `_cp_plan` for splitting hidden states across devices |
| **KV cache context** | `cache_context("cond"/"uncond")` for efficient CFG in Klein pipeline |
| **FP16 overflow clipping** | Safety clamping to float16 range in transformer blocks |
| **QKV fusion/unfusion** | Runtime `fuse_qkv_projections()` / `unfuse_qkv_projections()` methods |
| **PEFT adapter support** | Built-in `PeftAdapterMixin` for LoRA via HuggingFace PEFT library |
| **Modular pipeline system** | Decomposed pipeline components (separate encoder, denoiser, decoder steps) |
| **Prompt upsampling** | `upsample_prompt()` method using Mistral3 for prompt enhancement (DEV only) |

### Features Shared Between Diffusers and Blissful-Tuner (not in BFL)

| Feature | Description |
|---------|-------------|
| **Gradient checkpointing** | Per-block checkpointing for memory savings (both support activation CPU offloading) |
| **Multiple attention backends** | Both support SDPA, Flash Attention, xFormers (blissful-tuner adds SageAttention) |

### Prompt Upsampling (BFL Reference / Diffusers Only)

The BFL reference and diffusers include a prompt upsampling system using the Mistral3 text encoder to enhance prompts. This is **not** implemented in blissful-tuner but the system messages provide useful context for prompt engineering:

**T2I upsampling system message:**
> *"You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent. [...] Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality (signs, labels, screens, etc.) — without it, the model generates gibberish."*

**I2I upsampling system message:**
> *"You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert editing requests into one concise instruction (50-80 words, ~30 for brief requests). [...] Specify what changes AND what stays the same (face, lighting, composition). Turn negatives into positives ('don't change X' → 'keep X'). Make abstractions concrete ('futuristic' → 'glowing cyan neon, metallic panels')."*

> **Recommended upsampling temperature:** 0.15 (from official BFL implementation).

---

## Resources

- **Diffusers Pipeline (DEV)**: `diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline`
- **Diffusers Pipeline (Klein)**: `diffusers.pipelines.flux2.pipeline_flux2_klein.Flux2KleinPipeline`
- **Transformer Model**: `diffusers.models.transformers.transformer_flux2.Flux2Transformer2DModel`
- **VAE Model**: `diffusers.models.autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2`
- **Scheduler**: `diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler`
- **This repo (BFL-native models)**: `src/musubi_tuner/flux_2/flux2_models.py` — `Flux2Params`, `Klein9BParams`, `Klein4BParams`
- **This repo (model info/loading/denoise)**: `src/musubi_tuner/flux_2/flux2_utils.py` — `FLUX2_MODEL_INFO`, embedders, `denoise()`/`denoise_cfg()`
- **This repo (LoRA targeting)**: `src/musubi_tuner/networks/lora_flux_2.py`
- **This repo (architecture registry)**: `src/musubi_tuner/networks/network_arch.py`
