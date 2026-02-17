# Qwen-Image Architecture Reference

This document provides a comprehensive reference for the Qwen-Image model family architecture, covering all variants (Original, Edit, Edit-2509, Edit-2511, 2512, and Layered). Configuration details are extracted directly from the official HuggingFace model configuration files. This information is intended to aid debugging, optimization, and development work on the Qwen-Image training pipeline.

**Primary Source**: [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)
**Edit Source**: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
**Diffusers Version**: 0.36.0.dev0
**Base Pipeline Class**: `QwenImagePipeline` (T2I) / `QwenImageEditPlusPipeline` (Edit) / `QwenImageLayeredPipeline` (Layered)

---

## Component Summary

| Component | Class | Parameters | Key Characteristics |
|-----------|-------|------------|---------------------|
| Text Encoder | `Qwen2_5_VLForConditionalGeneration` | ~8.3B | VL model with 28-layer LLM + 32-layer ViT |
| Transformer (DiT) | `QwenImageTransformer2DModel` | ~20.4B | 60-layer MMDiT with joint attention |
| VAE | `AutoencoderKLQwenImage` | ~83M est. | 16-channel latent, 8x spatial compression |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | - | Flow matching with dynamic shift |
| Tokenizer | `Qwen2Tokenizer` | 152k vocab | BPE tokenizer |

**Total Model Size**: ~38GB (bf16), ~40.8GB raw weights

---

## Model Variant Comparison

The Qwen-Image family consists of several model variants built on the same core MMDiT architecture. All share the same `QwenImageTransformer2DModel` class and weight structure — the differences lie in config flags, pipeline classes, and specialized training.

### Variant Overview

| Model | Release | Purpose | Pipeline Class |
|-------|---------|---------|----------------|
| Qwen-Image | Aug 2025 | Text-to-image generation | `QwenImagePipeline` |
| Qwen-Image-Edit | Aug 2025 | Single-image editing | `QwenImageEditPipeline` |
| Qwen-Image-Edit-2509 | Sep 2025 | Multi-image editing + ControlNet | `QwenImageEditPlusPipeline` |
| Qwen-Image-Edit-2511 | Nov 2025 | Improved multi-image editing | `QwenImageEditPlusPipeline` |
| Qwen-Image-2512 | Dec 2025 | Improved T2I (updated weights) | `QwenImagePipeline` |
| Qwen-Image-Layered | Dec 2025 | RGBA layer decomposition | `QwenImageLayeredPipeline` |

### Transformer Config Flags by Variant

All variants use `in_channels=64`, `out_channels=16`, 60 layers, 24 heads, and `axes_dims_rope=[16,56,56]`. The differences are:

| Config Flag | Original/2512 | Edit/Edit-2509 | Edit-2511 | Layered |
|-------------|:-------------:|:--------------:|:---------:|:-------:|
| `guidance_embeds` | false | false | false | false |
| `zero_cond_t` | -- | -- | **true** | false |
| `use_additional_t_cond` | -- | -- | -- | **true** |
| `use_layer3d_rope` | -- | -- | -- | **true** |

### VAE and Pipeline Differences

| Aspect | Original/2512 | Edit/Edit-2509/Edit-2511 | Layered |
|--------|:-------------:|:------------------------:|:-------:|
| VAE input channels | 3 (RGB) | 3 (RGB) | **4 (RGBA)** |
| VAE class | AutoencoderKLQwenImage | AutoencoderKLQwenImage | AutoencoderKLQwenImage |
| VAE z_dim | 16 | 16 | 16 |
| Processor component | No | **Yes** (Qwen2VLProcessor) | **Yes** (Qwen2VLProcessor) |
| Control images | None | 1+ reference images | Input image for decomposition |
| Output | Single RGB image | Single RGB image | **Multiple RGBA layers** |

### Model Weight Sizes

| Component | Size (raw safetensors) | Shards |
|-----------|----------------------|--------|
| Transformer (2512) | ~40.86 GB | 9 files |
| Transformer (Edit-2511) | ~40.86 GB | 5 files |
| Text Encoder (all variants) | ~16.58 GB | 4 files |
| VAE (all variants) | ~1 GB | 1 file |

### Key Insight: Structural Compatibility

Because all variants share the same transformer architecture (`QwenImageTransformer2DModel` with identical layer structure), a LoRA trained on one variant is **structurally loadable** on any other variant — though semantic behavior will differ due to different base weights and config flags.

---

## Data Flow Overview

```
Input Image (H × W × 3)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  VAE Encoder                                                     │
│  - Spatial: 8× downsampling                                      │
│  - Output: (H/8) × (W/8) × 16 latent channels                   │
│  - Normalized with per-channel mean/std                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Latent: (H/8) × (W/8) × 16
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────────────┐
│  Patchify           │              │  Text Encoder (Qwen2.5-VL)  │
│  patch_size=2       │              │  - Tokenize prompt          │
│  in_channels=64     │              │  - 28-layer transformer     │
│                     │              │  - Output: (seq_len × 3584) │
└─────────────────────┘              └─────────────────────────────┘
         │                                      │
         │      ┌───────────────────────────────┘
         │      │
         ▼      ▼
┌─────────────────────────────────────────────────────────────────┐
│  MMDiT Transformer (60 layers)                                   │
│  - Joint attention between image and text tokens                 │
│  - MSRoPE positional encoding: axes [16, 56, 56]                │
│  - Timestep embedding via sinusoidal + MLP                       │
│  - Separate img_mlp and txt_mlp per block                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Predicted Noise/Velocity
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  VAE Decoder                                                     │
│  - Spatial: 8× upsampling                                        │
│  - Output: (H × W × 3) RGB image                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Text Encoder: Qwen2.5-VL

The text encoder is a full Vision-Language model based on Qwen2.5-VL architecture, capable of processing both text and images as input context.

### Language Model Configuration

```json
{
  "architectures": ["Qwen2_5_VLForConditionalGeneration"],
  "model_type": "qwen2_5_vl",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "hidden_act": "silu",
  "max_position_embeddings": 128000,
  "rms_norm_eps": 1e-06,
  "vocab_size": 152064,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "max_window_layers": 28,
  "tie_word_embeddings": false
}
```

### Key Architectural Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden Dimension | 3584 | Matches MMDiT joint_attention_dim |
| Attention Heads | 28 | Full attention |
| KV Heads | 4 | Grouped Query Attention (GQA) 7:1 ratio |
| FFN Intermediate | 18944 | ~5.3× expansion ratio |
| Layers | 28 | All use full attention (no sliding window) |
| Max Context | 128k tokens | Extremely long context support |
| Activation | SiLU | Smooth activation function |
| Normalization | RMSNorm (eps=1e-6) | Pre-norm architecture |

### MROPE Configuration (Text Encoder)

The text encoder uses Multi-dimensional RoPE with three sections:

```json
{
  "rope_scaling": {
    "mrope_section": [16, 24, 24],
    "rope_type": "default"
  }
}
```

- Total RoPE dimensions: 16 + 24 + 24 = 64
- Per-head dimension: 3584 / 28 = 128
- RoPE covers half of head dim (64/128)

### Vision Encoder (Embedded in Text Encoder)

The text encoder includes an integrated vision transformer for processing image inputs:

```json
{
  "vision_config": {
    "depth": 32,
    "hidden_size": 1280,
    "out_hidden_size": 3584,
    "num_heads": 16,
    "patch_size": 14,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "window_size": 112,
    "intermediate_size": 3420,
    "fullatt_block_indexes": [7, 15, 23, 31],
    "hidden_act": "silu"
  }
}
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Vision Layers | 32 | Deep vision encoder |
| Vision Hidden | 1280 | Internal ViT dimension |
| Output Projection | 3584 | Projects to LLM hidden size |
| Patch Size | 14×14 | Spatial tokenization |
| Spatial Merge | 2×2 | Pools 4 patches → 1 token |
| Full Attention Layers | [7, 15, 23, 31] | Every 8th layer uses global attention |

### Special Token IDs

```json
{
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "image_token_id": 151655,
  "video_token_id": 151656,
  "vision_start_token_id": 151652,
  "vision_end_token_id": 151653,
  "vision_token_id": 151654
}
```

### System Prompts for Feature Extraction

The text encoder uses distinct system prompts depending on the task, which directly affect the embeddings produced:

**Text-to-Image (T2I) System Prompt:**
> "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background"

**Text-Image-to-Image (TI2I / Edit) System Prompt:**
> "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."

The last layer hidden state from the Qwen2.5-VL language model backbone (`h = ϕ(S)`) is extracted as the conditioning representation for the MMDiT transformer. This is crucial for training: the system prompt affects what information the text encoder emphasizes in its embeddings.

### Weight Tensor Patterns (Text Encoder)

```
model.embed_tokens.weight                    # Embedding layer
model.layers.{0-27}.input_layernorm.weight   # Pre-attention norm
model.layers.{0-27}.self_attn.q_proj.*       # Query projection (+ bias)
model.layers.{0-27}.self_attn.k_proj.*       # Key projection (+ bias)
model.layers.{0-27}.self_attn.v_proj.*       # Value projection (+ bias)
model.layers.{0-27}.self_attn.o_proj.weight  # Output projection
model.layers.{0-27}.post_attention_layernorm.weight
model.layers.{0-27}.mlp.gate_proj.weight     # FFN gating
model.layers.{0-27}.mlp.up_proj.weight       # FFN up projection
model.layers.{0-27}.mlp.down_proj.weight     # FFN down projection
model.norm.weight                            # Final layer norm
lm_head.weight                               # Output projection

# Vision encoder weights (visual.*)
visual.patch_embed.proj.weight
visual.blocks.{0-31}.attn.qkv.*              # Fused QKV
visual.blocks.{0-31}.attn.proj.*             # Attention output
visual.blocks.{0-31}.mlp.gate_proj.*
visual.blocks.{0-31}.mlp.up_proj.*
visual.blocks.{0-31}.mlp.down_proj.*
visual.blocks.{0-31}.norm1.weight
visual.blocks.{0-31}.norm2.weight
visual.merger.ln_q.weight                    # Merger layer norm
visual.merger.mlp.{0,2}.*                    # Merger MLP
```

---

## Transformer (MMDiT): QwenImageTransformer2DModel

The core diffusion transformer uses a Multi-Modal DiT architecture with joint image-text attention.

### Configuration

```json
{
  "_class_name": "QwenImageTransformer2DModel",
  "num_layers": 60,
  "num_attention_heads": 24,
  "attention_head_dim": 128,
  "joint_attention_dim": 3584,
  "in_channels": 64,
  "out_channels": 16,
  "patch_size": 2,
  "axes_dims_rope": [16, 56, 56],
  "guidance_embeds": false
}
```

### Key Architectural Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Layers | 60 | Deep transformer |
| Attention Heads | 24 | For image processing |
| Head Dimension | 128 | Per-head dimension |
| Hidden Dimension | 24 × 128 = 3072 | Image stream hidden |
| Joint Attention Dim | 3584 | Text conditioning dimension |
| Input Channels | 64 | After patchify (16 latent × 2×2 patch) |
| Output Channels | 16 | Matches VAE latent channels |
| Patch Size | 2×2 | Patchifies latent space |
| Guidance Embeds | false | No CFG embedding in model (unused; may be removed in future diffusers) |

### MSRoPE Configuration (Transformer)

The transformer uses Multi-Scale RoPE with axes optimized for image generation:

```python
axes_dims_rope = [16, 56, 56]  # [temporal/batch, height, width]
```

- **Axis 0** (16 dims): Temporal/batch position (for video extension)
- **Axis 1** (56 dims): Height position encoding
- **Axis 2** (56 dims): Width position encoding
- **Total**: 128 dims = attention_head_dim

This design allows the model to encode 2D spatial positions separately, critical for maintaining spatial coherence in generated images.

#### Symmetric Frequency Indices (`scale_rope=True`)

Qwen-Image uses `scale_rope=True` (hardcoded), which creates **symmetric (negative + positive) frequency indices** centered at zero rather than starting from 0. This means positions are encoded relative to the image center:

```python
# With scale_rope=True:
freqs_height = torch.cat([
    freqs_neg[1][-(height - height // 2):],  # Negative frequencies (top half)
    freqs_pos[1][:height // 2]                # Positive frequencies (bottom half)
], dim=0)
freqs_width = torch.cat([
    freqs_neg[2][-(width - width // 2):],     # Negative frequencies (left half)
    freqs_pos[2][:width // 2]                  # Positive frequencies (right half)
], dim=0)
```

**Implications**:
- Spatial positions are relative to image center (negative = top-left, positive = bottom-right)
- Text tokens are positioned at offset `max(height // 2, width // 2)` from center along the diagonal
- The symmetric design improves resolution generalization by keeping center-relative semantics consistent
- Frequency computation uses LRU caching (maxsize=128) for efficiency across inference steps

### Transformer Block Structure

Each of the 60 transformer blocks contains:

```
TransformerBlock
├── img_mod          # Image modulation (timestep-conditioned)
├── txt_mod          # Text modulation (timestep-conditioned)
├── attn             # Joint attention module
│   ├── to_q, to_k, to_v           # Image projections
│   ├── add_q_proj, add_k_proj, add_v_proj  # Text projections
│   ├── norm_q, norm_k              # Image QK normalization
│   ├── norm_added_q, norm_added_k  # Text QK normalization
│   ├── to_out                      # Image output projection
│   └── to_add_out                  # Text output projection
├── img_mlp          # Image FFN (activation: gelu-approximate → GEGLU structure)
│   ├── net.0.proj   # Gated projection (splits into gate + value, applies GELU)
│   └── net.2        # Output projection
└── txt_mlp          # Text FFN (activation: gelu-approximate → GEGLU structure)
    ├── net.0.proj   # Gated projection
    └── net.2        # Output projection
```

### Weight Tensor Patterns (Transformer)

```
img_in.{weight,bias}                         # Latent to hidden projection
time_text_embed.timestep_embedder.linear_{1,2}.*  # Timestep embedding MLP

transformer_blocks.{0-59}.img_mod.1.*        # Image modulation
transformer_blocks.{0-59}.txt_mod.1.*        # Text modulation

# Joint attention weights
transformer_blocks.{0-59}.attn.to_q.*        # Image query
transformer_blocks.{0-59}.attn.to_k.*        # Image key
transformer_blocks.{0-59}.attn.to_v.*        # Image value
transformer_blocks.{0-59}.attn.to_out.0.*    # Image output
transformer_blocks.{0-59}.attn.add_q_proj.*  # Text query
transformer_blocks.{0-59}.attn.add_k_proj.*  # Text key
transformer_blocks.{0-59}.attn.add_v_proj.*  # Text value
transformer_blocks.{0-59}.attn.to_add_out.*  # Text output
transformer_blocks.{0-59}.attn.norm_q.*      # Image Q normalization
transformer_blocks.{0-59}.attn.norm_k.*      # Image K normalization
transformer_blocks.{0-59}.attn.norm_added_q.* # Text Q normalization
transformer_blocks.{0-59}.attn.norm_added_k.* # Text K normalization

# FFN weights
transformer_blocks.{0-59}.img_mlp.net.0.proj.*  # Image FFN gate+up
transformer_blocks.{0-59}.img_mlp.net.2.*       # Image FFN down
transformer_blocks.{0-59}.txt_mlp.net.0.proj.*  # Text FFN gate+up
transformer_blocks.{0-59}.txt_mlp.net.2.*       # Text FFN down

norm_out.linear.*                            # Final normalization
proj_out.*                                   # Output projection to latent
```

### Memory Optimization Notes

With 60 layers, the transformer is a prime target for memory optimization:

- **Block Swapping**: Up to 59 blocks can be swapped to CPU
- **Gradient Checkpointing**: Reduces activation memory ~3-4×
- **Attention Precision**: bf16 recommended, fp8 possible with quality tradeoff

---

## Control Image Architecture (Edit Models)

Qwen-Image-Edit models handle reference/control images through a **dual-path conditioning mechanism** — not through extra input channels. This is a critical architectural distinction.

### Dual-Path Overview

```
Reference Image (H × W × 3)
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼ (Semantic Path)                 ▼ (Appearance Path)
┌─────────────────────────┐    ┌─────────────────────────────┐
│  Qwen2.5-VL Encoder     │    │  VAE Encoder                │
│  Resolution: ~384×384    │    │  Resolution: ~1024×1024     │
│  Processes image + text  │    │  Encodes to 16-ch latent    │
│  jointly as VL input     │    │  Patchified to tokens       │
│  Output: text+vision     │    │  Output: latent token seq   │
│  embeddings (3584-dim)   │    │                             │
└─────────────────────────┘    └─────────────────────────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────────────┐    ┌─────────────────────────────┐
│  Cross-Attention Path    │    │  Sequence Concatenation      │
│  → txt stream of MMDiT   │    │  [noise_tokens, ref_tokens]  │
│  (joint_attention_dim)   │    │  along sequence dimension    │
└─────────────────────────┘    └─────────────────────────────┘
         │                                 │
         └──────────┬──────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  MMDiT Transformer   │
         │  (60 layers)         │
         │  Attends to both     │
         │  streams jointly     │
         └─────────────────────┘
                    │
                    ▼
         model_pred[:, :img_seq_len]
         (only target portion used for loss)
```

### Key Design Decisions

1. **Sequence concatenation, NOT channel concatenation**: Reference image latents are appended along the token sequence dimension, not stacked as extra channels. This keeps `in_channels=64` constant across all variants.

2. **Two resolutions for the reference image**:
   - **~384×384 pixels** for the semantic path through Qwen2.5-VL (controlled by `CONDITION_IMAGE_SIZE`)
   - **~1024×1024 pixels** for the appearance path through the VAE (controlled by `VAE_IMAGE_SIZE`)

3. **Loss computation**: During training, only the target portion of the output sequence is used for loss calculation: `model_pred[:, :img_seq_len]`. The reference token predictions are discarded.

4. **Multi-image support** (Edit-2509, Edit-2511): Multiple reference images are each VAE-encoded and their latent tokens are concatenated sequentially: `[target_seq, ref1_seq, ref2_seq, ...]`

### Comparison with Other Architectures

| Architecture | Reference Image Method | Input Channels Change? |
|-------------|----------------------|----------------------|
| **Qwen-Image-Edit** | Sequence concatenation + VL encoder | No (always 64) |
| FLUX.1 Kontext | Sequence concatenation | No |
| WAN 2.2 I2V | Channel concatenation | Yes (+extra channels) |
| SD Inpainting | Channel concatenation (mask+image) | Yes (+5 channels) |

### Processor Component (Qwen2VLProcessor)

Edit and Layered models include a `Qwen2VLProcessor` component for preparing image+text inputs for the Qwen2.5-VL text encoder:

```json
{
  "image_processor_type": "Qwen2VLImageProcessorFast",
  "image_mean": [0.48145466, 0.4578275, 0.40821073],
  "image_std": [0.26862954, 0.26130258, 0.27577711],
  "patch_size": 14,
  "merge_size": 2,
  "max_pixels": 12845056,
  "min_pixels": 3136,
  "resample": "bicubic"
}
```

The processor handles:
- Image normalization with ImageNet-like statistics
- Patch extraction at 14×14 resolution
- Spatial merging (2×2 patches → 1 token)
- Multi-image batching for Edit-2509/2511

### Prompt Templates by Version

The text encoder uses different prompt templates depending on the model version:

| Version | Prompt Template |
|---------|----------------|
| `edit` | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>{prompt}` |
| `edit-2509` / `edit-2511` | `Picture 1: <\|vision_start\|><\|image_pad\|><\|vision_end\|>\n{prompt}` |
| `layered` | Same as original, but can auto-caption via Qwen2.5-VL |

### Prompt Template Token Indices

Each pipeline has a `prompt_template_encode_start_idx` that determines how many system/template tokens to skip when extracting the user's actual prompt embeddings:

| Pipeline | `prompt_template_encode_start_idx` | Full Template |
|----------|:----------------------------------:|---------------|
| T2I | **34** | `<\|im_start\|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<\|im_end\|>\n<\|im_start\|>user\n{}<\|im_end\|>\n<\|im_start\|>assistant\n` |
| Edit | **64** | Longer template including `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` tokens |

The tokenizer max length is **1024** tokens, but the default `max_sequence_length` in the T2I pipeline's `__call__` is **512**. These indices are important for understanding how text encoder outputs map to the transformer's `encoder_hidden_states_mask`.

---

## VAE: AutoencoderKLQwenImage

The VAE handles encoding images to latent space and decoding back to pixel space. It uses a **single-encoder, dual-decoder** architecture designed for both image and video, built on the Wan-2.1-VAE foundation.

### Training Origin

The VAE encoder is **frozen** (inherited directly from Wan-2.1-VAE). Only the **decoder** was fine-tuned, specifically on text-rich images:

| Aspect | Detail |
|--------|--------|
| Encoder | **Frozen** from Wan-2.1-VAE (19M params for images) |
| Decoder | **Fine-tuned** on text-rich images (25M params for images) |
| Fine-tuning data | In-house corpus: PDFs, PowerPoint slides, posters |
| Languages | Alphabetic (English) + logographic (Chinese) |
| Loss | Reconstruction + perceptual loss (dynamically adjusted ratio) |

The decoder fine-tuning on text-rich images is what enables Qwen-Image's exceptional text rendering quality. Adversarial loss was found ineffective as reconstruction quality increased.

### Configuration

```json
{
  "_class_name": "AutoencoderKLQwenImage",
  "z_dim": 16,
  "base_dim": 96,
  "dim_mult": [1, 2, 4, 4],
  "num_res_blocks": 2,
  "dropout": 0.0,
  "temperal_downsample": [false, true, true],
  "attn_scales": []
}
```

### Architecture Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Latent Channels | 16 | High-capacity latent space |
| Base Dimension | 96 | Initial conv channels |
| Dimension Multipliers | [1, 2, 4, 4] | Channel scaling per stage |
| Stages | 4 | Encoder/decoder depth |
| Spatial Compression | 8× | 3 downsample ops across 4 stages (2³) |
| Res Blocks per Stage | 2 | Standard depth |
| Temporal Downsampling | [F, T, T] | For video (stages 1, 2 only) |

### Channel Progression

```
Stage 0: 96 × 1  = 96  channels
Stage 1: 96 × 2  = 192 channels
Stage 2: 96 × 4  = 384 channels
Stage 3: 96 × 4  = 384 channels
```

### Latent Normalization Statistics

The VAE uses per-channel normalization for stable training:

```python
latents_mean = [
    -0.7571, -0.7089, -0.9113,  0.1075,
    -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632,
    -0.1922, -0.9497,  0.2503, -0.2921
]

latents_std = [
    2.8184, 1.4541, 2.3275, 2.6558,
    1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916
]
```

**Usage**: Normalize latents with `(latent - mean) / std` before training, denormalize after inference.

### Latent Space Properties

For an input image of size H × W:

- **Latent Size**: (H/8) × (W/8) × 16
- **Example** (1664×928 image): 208 × 116 × 16 = 386,048 latent values

### Layered VAE Variant

The Qwen-Image-Layered model uses a **modified VAE** with 4-channel (RGBA) input instead of the standard 3-channel (RGB) VAE. The alpha channel encodes layer transparency for compositing.

| Parameter | Standard VAE | Layered VAE |
|-----------|:-----------:|:-----------:|
| Input Channels | 3 (RGB) | **4 (RGBA)** |
| z_dim | 16 | 16 |
| Architecture | Identical | Identical |
| Latent Stats | Identical | Identical |
| Weight File | `qwen_image_vae.safetensors` | `qwen_image_layered_vae.safetensors` |

**Important**: Edit models (all versions) use the **standard 3-channel RGB VAE** — NOT a 4-channel VAE. Only the Layered variant requires the RGBA VAE.

---

## Variant-Specific Architecture Details

### Edit-2511: `zero_cond_t`

Qwen-Image-Edit-2511 introduces the `zero_cond_t = true` config flag, unique among all variants. This changes how timestep conditioning is applied to the image stream during editing:

Despite the name, this is **not** "CFG unconditional" behavior. It is an **intra-sequence timestep split** used during editing, where the image stream contains concatenated tokens (target/noise tokens followed by control/appearance tokens).

- **With `zero_cond_t = true`**: the model builds two timestep conditionings (`t` and `0`) and applies them to different segments of the **image-token sequence**:
  - **Target/noise tokens** get modulation at the real diffusion timestep (`t`)
  - **Control/appearance tokens** get modulation at a **zero timestep** (`t=0`)
- **Text stream**: still uses the real timestep (`t`) — only the image modulation is split
- **Purpose / Result**: treat control tokens as a "clean reference" while denoising the target tokens, reducing edit drift and improving structural stability

#### Edit-2511 Improvements over Edit-2509

| Improvement | Description |
|-------------|-------------|
| Mitigated image drift | Better preservation of original image content during editing |
| Improved character consistency | Better identity preservation for both single and multi-person edits |
| Integrated LoRA capabilities | Popular community LoRAs merged into base weights |
| Enhanced industrial design | Specialized for product design, material replacement |
| Strengthened geometric reasoning | Better spatial understanding, construction line generation |

### Layered Model: Layer-Conditional Generation

Qwen-Image-Layered extends the base architecture with three additional capabilities:

#### `use_additional_t_cond = true`

Adds a **layer-index conditioning signal** to the timestep embedding. This allows the transformer to know which layer it is currently generating (e.g., foreground, midground, background). The conditioning is a binary tensor indicating whether each output frame is an RGB image.

#### `use_layer3d_rope = true`

Extends the standard 2D spatial RoPE to **3D positional encoding** by adding a layer dimension:

```
Standard RoPE: [temporal=16, height=56, width=56]  → 2D spatial + batch
Layered RoPE:  [layer_index, height, width]         → 3D with layer awareness
```

This enables the transformer to differentiate between different output layers while maintaining spatial coherence within each layer.

#### RGBA Output

The Layered model generates **multiple RGBA images** (configurable number of layers). Each layer includes an alpha channel for compositing:

- **Layer 0**: Original/composite image (can be excluded with `--remove_first_image_from_target`)
- **Layer 1+**: Individual decomposed layers (foreground, objects, background, etc.)

The number of output layers is controlled by `--output_layers` at inference time.

### Layered Pipeline Inference Differences

The Layered pipeline differs from T2I and Edit in several significant ways at inference time:

#### Different Shift Formula

The Layered pipeline does **not** use the standard `calculate_shift()` linear interpolation. Instead, it uses a square-root formula that produces gentler shift scaling at higher resolutions:

```python
# T2I and Edit pipelines: linear interpolation
mu = calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                     base_shift=0.5, max_shift=1.15)

# Layered pipeline: square-root scaling
base_seqlen = 256 * 256 / 16 / 16  # = 256
mu = (image_latents.shape[1] / base_seqlen) ** 0.5
```

For a 1328×1328 image (seq_len ≈ 6889): T2I mu ≈ 0.93, Layered mu ≈ 5.19.

#### CFG Normalization is Optional

| Pipeline | CFG Normalization | Default |
|----------|:-----------------:|:-------:|
| T2I | **Always applied** | — |
| Edit | **Always applied** | — |
| Layered | Optional via `cfg_normalize` | **False** |

The normalization formula rescales the combined CFG prediction to match the magnitude of the conditional prediction:
```python
comb_pred = neg_pred + true_cfg_scale * (noise_pred - neg_pred)
if cfg_normalize:
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    noise_pred = comb_pred * (cond_norm / noise_norm)
else:
    noise_pred = comb_pred  # Layered default: no rescaling
```

#### Edit Pipeline Batch Size Restriction

The Edit pipeline (`QwenImageEditPlusPipeline`) enforces `batch_size=1` and raises `ValueError` if a larger batch is requested. This is relevant for training-time sample generation configuration.

### Model Loading in Training Pipeline

The training pipeline activates variant-specific features based on `--model_version`:

```python
model = load_qwen_image_model(
    ...
    zero_cond_t       = (model_version == "edit-2511"),      # Only Edit-2511
    use_additional_t_cond = (model_version == "layered"),     # Only Layered
    use_layer3d_rope  = (model_version == "layered"),         # Only Layered
)
```

The VAE input channels are similarly conditioned:

```python
input_channels = 4 if is_layered else 3  # RGBA only for Layered
```

---

## Scheduler: FlowMatchEulerDiscreteScheduler

The scheduler implements flow matching with dynamic timestep shifting.

### Configuration

```json
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "num_train_timesteps": 1000,
  "base_shift": 0.5,
  "max_shift": 0.9,
  "shift_terminal": 0.02,
  "time_shift_type": "exponential",
  "use_dynamic_shifting": true,
  "base_image_seq_len": 256,
  "max_image_seq_len": 8192,
  "stochastic_sampling": false,
  "invert_sigmas": false,
  "use_karras_sigmas": false,
  "use_exponential_sigmas": false,
  "use_beta_sigmas": false
}
```

### Key Parameters

| Parameter | Scheduler Config | Pipeline Default | Notes |
|-----------|:----------------:|:----------------:|-------|
| Training Timesteps | 1000 | — | Discrete timestep range |
| Base Shift | 0.5 | 0.5 | Shift for base sequence length |
| Max Shift | **0.9** | **1.15** | Shift for max sequence length |
| Shift Terminal | 0.02 | — | Terminal shift value |
| Time Shift Type | exponential | — | Exponential shift curve |
| Dynamic Shifting | true | — | Adapts to image resolution |
| Base Seq Length | 256 | 256 | Reference sequence length |
| Max Seq Length | **8192** | **4096** | Maximum supported sequence |

**Important**: The scheduler config and pipeline `calculate_shift()` defaults differ for `max_shift` and `max_seq_len`. At inference time, the pipeline reads from the scheduler config with its own defaults as fallbacks:
```python
mu = calculate_shift(
    image_seq_len,
    self.scheduler.config.get("base_image_seq_len", 256),
    self.scheduler.config.get("max_image_seq_len", 4096),   # fallback if missing
    self.scheduler.config.get("base_shift", 0.5),
    self.scheduler.config.get("max_shift", 1.15),            # fallback if missing
)
```

Since the official scheduler config **does** include these values, inference uses the scheduler config values (0.9 / 8192). The pipeline defaults (1.15 / 4096) only apply if the scheduler config lacks these keys.

### Dynamic Shift Formula

The scheduler adjusts the noise schedule based on image resolution via **linear interpolation** between `(base_seq_len, base_shift)` and `(max_seq_len, max_shift)`:

```python
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
```

Where `image_seq_len = (height // 8 // patch_size) * (width // 8 // patch_size)`. The returned `mu` is passed to the scheduler's `set_timesteps()` to shift the noise schedule.

**Layered Pipeline Uses a Different Formula**: The Layered pipeline does NOT use `calculate_shift()`. Instead it computes mu via a square-root formula (see [Layered Pipeline Inference Differences](#layered-pipeline-inference-differences) below).

---

## Flow Matching Training Details

Qwen-Image uses a **Rectified Flow** formulation for flow matching training. Understanding these details aids in configuring timestep sampling and loss computation.

### Core Formulation

Given a training image `x₀` encoded by the VAE as `z = E(x₀)`, and random noise `x₁ ~ N(0, I)`:

```
Intermediate latent:  x_t = t·x₀ + (1-t)·x₁
Target velocity:      v_t = dx_t/dt = x₀ - x₁
Training loss:        L = E[‖v_θ(x_t, t, h) - v_t‖²]
```

Where `h = ϕ(S)` is the conditioning from the Qwen2.5-VL text encoder, and `t ∈ [0, 1]`.

### Timestep Sampling

Timesteps are sampled from a **logit-normal distribution** (not uniform). This concentrates more training signal in the middle timesteps where the model learns the most:

```
t ~ LogitNormal(μ, σ²)    where t ∈ [0, 1]
```

In the blissful-tuner training pipeline, this corresponds to `--timestep_sampling shift` with `--discrete_flow_shift` controlling the shift parameter.

### Normalization Layers

The transformer uses:
- **RMSNorm** for QK-Norm (query-key normalization in attention)
- **LayerNorm** for all other normalization layers
- **Scale & Shift modulation** conditioned on timestep throughout all blocks

### Official Inference Parameters

The official Qwen-Image examples use these defaults:

| Parameter | T2I (Original/2512) | Edit (Original) | Edit-2509/2511 |
|-----------|:-------------------:|:---------------:|:--------------:|
| `num_inference_steps` | 50 | 50 | **40** |
| `true_cfg_scale` | 4.0 | 4.0 | 4.0 |
| `guidance_scale` | N/A | N/A | **1.0** |
| `negative_prompt` | " " or detailed | " " | " " |

**Important**: Edit-2509/2511 uses `guidance_scale=1.0` (disabling standard CFG) combined with `true_cfg_scale=4.0` (enabling true CFG with a separate unconditional forward pass). This dual-parameter approach is different from T2I, which only uses `true_cfg_scale`.

---

## Progressive Curriculum Learning

Qwen-Image was trained using a 7-stage progressive curriculum. Understanding these stages provides context for training strategy decisions:

### Training Stages

| Stage | Resolution | Focus | Key Filters |
|-------|-----------|-------|-------------|
| 1 | 256×256 | Initial pre-training | Broken files, dedup, NSFW, min resolution |
| 2 | 256×256 | Image quality | Rotation, clarity, luma, saturation, entropy, texture |
| 3 | 256×256 | Image-text alignment | CLIP/SigLIP scoring, caption quality, recaptioning |
| 4 | 256×256 | Text rendering | Synthetic text data (3 strategies), intensive text filter |
| 5 | 640×640 | High-resolution | Quality, resolution, aesthetic, watermark filters |
| 6 | 640×640 | Category balance | Portrait augmentation, domain resampling |
| 7 | 640→1328 | Multi-scale | Hierarchical taxonomy, balanced multi-resolution |

### Data Distribution

The training data is distributed across four primary domains:
- **Nature** (~55%): Objects, landscapes, cityscapes, plants, animals, indoor, food
- **Design** (~27%): Posters, UIs, slides, paintings, sculptures, digital arts
- **People** (~13%): Portraits, sports, activities
- **Synthetic** (~5%): Controlled text rendering (not AI-generated images)

### Text Rendering Data Synthesis

Three synthesis strategies address the long-tail distribution of textual content:

1. **Pure Rendering**: Text paragraphs rendered on clean backgrounds with dynamic layout
2. **Compositional Rendering**: Synthetic text composited into realistic scenes (paper, boards, etc.)
3. **Complex Rendering**: Pre-defined templates (PowerPoint slides, UI mockups) with placeholder text substitution

---

## Supported Resolutions

The model is trained on specific aspect ratios for optimal quality:

| Aspect Ratio | Resolution | Latent Size | Sequence Length |
|--------------|------------|-------------|-----------------|
| 1:1 | 1328 × 1328 | 166 × 166 | 6889 |
| 16:9 | 1664 × 928 | 208 × 116 | 6032 |
| 9:16 | 928 × 1664 | 116 × 208 | 6032 |
| 4:3 | 1472 × 1104 | 184 × 138 | 6348 |
| 3:4 | 1104 × 1472 | 138 × 184 | 6348 |
| 3:2 | 1584 × 1056 | 198 × 132 | 6534 |
| 2:3 | 1056 × 1584 | 132 × 198 | 6534 |

**Note**: Sequence length = (latent_h / patch_size) × (latent_w / patch_size) where patch_size=2

**Resolution Ambiguity**: The official repo is inconsistent for the 4:3 and 3:4 aspect ratios. The Qwen-Image-2512 README uses 1472×1104 / 1104×1472, while the original Qwen-Image demo scripts use 1472×1140 / 1140×1472. The values above match the 2512 README.

---

## LoRA Target Modules

For fine-tuning with LoRA, the following module patterns are typically targeted:

### Transformer Targets (High Impact)

```python
# Joint attention projections
"transformer_blocks.*.attn.to_q"
"transformer_blocks.*.attn.to_k"
"transformer_blocks.*.attn.to_v"
"transformer_blocks.*.attn.to_out.0"
"transformer_blocks.*.attn.add_q_proj"
"transformer_blocks.*.attn.add_k_proj"
"transformer_blocks.*.attn.add_v_proj"
"transformer_blocks.*.attn.to_add_out"

# FFN projections
"transformer_blocks.*.img_mlp.net.0.proj"
"transformer_blocks.*.img_mlp.net.2"
"transformer_blocks.*.txt_mlp.net.0.proj"
"transformer_blocks.*.txt_mlp.net.2"
```

### Diffusers Default LoRA Targets (Comparison)

The official diffusers DreamBooth training script uses a more conservative default — **attention-only**, without FFN layers:

```python
# Official diffusers default (train_dreambooth_lora_qwen_image.py):
["to_q", "to_k", "to_v", "to_out.0"]
# Default rank=4, alpha=4, dropout=0.0
```

The blissful-tuner pipeline targets **both attention and FFN** modules (as listed above), which allows for greater expressiveness but requires more VRAM. Use the diffusers defaults as a conservative starting point, and the full target list for maximum quality.

**Text Encoder LoRA**: Not supported in diffusers (`supports_text_encoder_loras = False` in `QwenImageLoraLoaderMixin`). The blissful-tuner pipeline supports text encoder LoRA through its own implementation.

### Text Encoder Targets (Optional, Blissful-Tuner Only)

```python
# Self-attention
"model.layers.*.self_attn.q_proj"
"model.layers.*.self_attn.k_proj"
"model.layers.*.self_attn.v_proj"
"model.layers.*.self_attn.o_proj"

# FFN
"model.layers.*.mlp.gate_proj"
"model.layers.*.mlp.up_proj"
"model.layers.*.mlp.down_proj"
```

---

## Prompt Enhancement System

Qwen-Image uses a sophisticated prompt rewriting system to improve generation quality. There are **two versions** of the system:

### Version 1: Original (`prompt_utils.py`)

Used with the original Qwen-Image model. Uses the **Qwen-Plus** API (DashScope) for rewriting.

#### Language Detection

```python
# Detects CJK characters to determine language
if any('\u4e00' <= char <= '\u9fff' for char in prompt):
    return 'zh'  # Chinese
return 'en'  # English
```

#### Rewriting Categories

The prompt enhancer classifies prompts into three categories:

1. **Portrait Images**: Detailed human descriptions (ethnicity, age, clothing, pose)
2. **Text-Containing Images**: Images with visible text (signs, labels, UI)
3. **General Images**: Landscapes, objects, abstract compositions

#### Magic Suffix (Original Only)

- **Chinese**: Appends "超清，4K，电影级构图"
- **English**: Appends "Ultra HD, 4K, cinematic composition"

### Version 2: Updated for 2512 (`prompt_utils_2512.py`)

Dramatically more sophisticated (~6000 word system prompt). Key differences from v1:

- **No magic suffix** — the "Ultra HD, 4K" suffix is **not appended** for Qwen-Image-2512
- **Portrait subtask** now demands: ethnicity, gender, specific age, face shape, eye shape/color, nose type, skin tone/texture, detailed makeup (eyeshadow, eyeliner, eyelashes, eyebrow shape, lipstick, blush, highlight), clothing with materials, hairstyle details, pose breakdown (~200 words target)
- **Text-containing subtask** requires: exact text transcription with punctuation, font style, color, layout direction, and for infographics, explicit text content (rejects vague descriptions like "a list")
- **General subtask** covers: spatial layering (foreground/midground/background), surface textures, dynamic interactions, time/weather, emotional tone

### Edit Prompt Enhancement

For editing tasks, a separate system uses **Qwen-VL-Max** (vision-language model) to rewrite prompts with image context. This is fundamentally different — it sends both the prompt AND the reference image to the API.

Six task-type-specific handling rules:
1. **Add/Delete/Replace**: Supplements vague instructions with specific details (color, size, position)
2. **Text Editing**: All text in double quotes, preserves original language/capitalization
3. **Human (ID) Editing**: Maintains visual consistency (ethnicity, gender, age, hairstyle, expression)
4. **Style Conversion**: Describes style with key visual features; fixed colorization template
5. **Content Filling**: Fixed inpainting/outpainting templates
6. **Multi-Image**: Clearly identifies which image's element is modified

**Warning**: The official documentation states that editing results may become **unstable without prompt rewriting**. Prompt rewriting is strongly recommended for all Edit tasks.

### Recommended Prompting

- **Text in images**: Wrap displayed text in quotation marks ("text here")
- **Negative prompt (2512 full version)**: "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
- **Negative prompt (English equivalent)**: "Low resolution, low quality, deformed limbs, deformed fingers, oversaturated, wax-like, faceless detail, overly smooth, AI-generated look. Chaotic composition. Blurry, distorted text."
- **Edit models**: Negative prompt should be a single space `" "` (hardcoded in official demos)

---

## Training Recommendations

### Memory Requirements

| Configuration | VRAM (bf16) | Notes |
|--------------|-------------|-------|
| Full model | ~80GB | A100 80GB minimum |
| Gradient checkpointing | ~50GB | Recommended baseline |
| + Block swap (30) | ~35GB | Good for 40GB GPUs |
| + FP8 quantization | ~25GB | Quality tradeoff |

### Timestep Sampling

For Qwen-Image training, use flow matching with shift:

```bash
--timestep_sampling shift
--discrete_flow_shift 2.2  # Low relative to other architectures; ~2.2 for 1328x1328
```

### Batch Accumulation

With the deep 60-layer transformer, gradient accumulation is often necessary:

```bash
--gradient_accumulation_steps 4
--gradient_checkpointing
```

---

## Technical Report Insights

The following insights are extracted from the official Qwen-Image Technical Report and provide deeper understanding of the architectural decisions and training methodology.

### MSRoPE Design Rationale

The Multi-Scale Rotary Position Embedding design serves a critical purpose beyond standard 2D position encoding:

**Text Token Positioning**: Text tokens from the Qwen2.5-VL text encoder are positioned along the diagonal of the image grid rather than prepended or appended. This design:

1. Preserves the gradual positional encoding from the Qwen2.5-VL encoder
2. Allows smooth interpolation between text and image positions
3. Enables better handling of long text sequences for text rendering tasks
4. The axes `[16, 56, 56]` allocate 16 dims for temporal/text position and 56 dims each for spatial height/width

### VAE Architecture & Performance

The Qwen-Image VAE uses a **single-encoder, dual-decoder** architecture designed for both image and video:

| Metric | Qwen-Image-VAE | FLUX-VAE | SD-3.5-VAE | Wan2.1-VAE |
|--------|----------------|----------|------------|------------|
| **Image PSNR↑** | **33.42** | 29.41 | 31.22 | 31.04 |
| **Image SSIM↑** | **0.9159** | 0.8596 | 0.8839 | 0.8916 |
| **Text-Rich PSNR↑** | **36.63** | 28.02 | 29.93 | 31.17 |
| **Text-Rich SSIM↑** | **0.9839** | 0.9374 | 0.9658 | 0.9702 |

**Effective Parameters** (image processing mode):
- Encoder: 19M (vs. 34M for SD-3.5-VAE)
- Decoder: 25M (vs. 50M for SD-3.5-VAE)

The decoder was specifically fine-tuned on text-rich images to improve reconstruction of small text, which is critical for the model's text rendering capabilities.

### Training Strategy

Qwen-Image employs a sophisticated **progressive curriculum learning** approach:

#### Resolution Progression
Training proceeds from low to high resolution, allowing the model to learn coarse structure before fine details:
```
Phase 1: 256×256 → Phase 2: 512×512 → Phase 3: 1024×1024 → Phase 4: 2512×2512
```

#### Text Complexity Progression
The model learns to generate images before learning text rendering:
```
Phase 1: Non-text images → Phase 2: Simple text → Phase 3: Complex text layouts
```

#### Data Pipeline (7 Stages)
1. Aesthetic filtering
2. Resolution filtering
3. Deduplication
4. NSFW filtering
5. Quality scoring
6. Caption quality assessment
7. Text rendering quality assessment (for text-rich data)

### Multi-Task Training

The model supports unified training across multiple tasks:

| Task | Conditioning | Description |
|------|-------------|-------------|
| **T2I** | Text only | Standard text-to-image generation |
| **TI2I** | Text + Image | Image editing with text instructions |
| **I2I** | Image only | Image reconstruction/manipulation |

For image editing (TI2I), the model uses a **frame dimension extension** mechanism where the input image is treated as an additional frame, enabling the model to learn editing operations through paired data.

### Post-Training Methods

After base training, Qwen-Image undergoes three additional fine-tuning stages:

#### 1. SFT (Supervised Fine-Tuning)
Fine-tuning on hierarchically organized, human-annotated high-quality datasets. Selection criteria: clear, rich detail, bright, photorealistic images. Guides the model toward greater realism and finer details.

#### 2. DPO (Direct Preference Optimization)

Adapted for flow matching (not the standard LLM DPO formulation). Large-scale offline preference learning:

**Data preparation**: Multiple images generated per prompt with different seeds. Human annotators select best/worst. For prompts with reference images, the worst is rejected if it deviates significantly.

**Flow-matching DPO loss**:
```
Diff_policy = ‖v_θ(x_win_t, h, t) - v_win_t‖² - ‖v_θ(x_lose_t, h, t) - v_lose_t‖²
Diff_ref    = ‖v_ref(x_win_t, h, t) - v_win_t‖² - ‖v_ref(x_lose_t, h, t) - v_lose_t‖²
L_DPO       = -E[log σ(-β(Diff_policy - Diff_ref))]
```

Where `v_θ` is the policy model's velocity prediction, `v_ref` is the reference model's, and β is a scaling parameter. This operates on velocity predictions rather than token probabilities.

#### 3. GRPO (Group Relative Policy Optimization)

Small-scale fine-grained RL refinement after DPO. Uses Flow-GRPO framework:

**Algorithm**: Generate group G of images `{x_i_0}` with trajectories. Compute group-normalized advantages:
```
A_i = [R(x_i_0, h) - mean(R)] / std(R)
```

**Flow-GRPO loss**:
```
L_GRPO = E[1/G Σ_i 1/T Σ_t min(r_i_t(θ)·A_i, clip(r_i_t(θ), 1-ε, 1+ε)·A_i) - β·D_KL(π_θ ‖ π_ref)]
```

Where `r_i_t(θ)` is the importance ratio and the KL-divergence has a closed form:
```
D_KL = Δt/2 · [σ_t(1-t)/2t + 1/σ_t² · ‖v_θ - v_ref‖²]
```

**Impact**: RL fine-tuning improves GenEval benchmark score from **0.87 → 0.91**, making Qwen-Image the first foundation model to exceed 0.9 on this benchmark.

### Benchmark Performance Summary

#### Text-to-Image Generation
| Benchmark | Qwen-Image Score | Ranking |
|-----------|-----------------|---------|
| DPG | 88.32 | #1 |
| GenEval | 0.91 (RL) | #1 |
| OneIG-EN | 0.539 | #1 |
| OneIG-ZH | 0.548 | #1 |

#### Chinese Text Rendering (ChineseWord Benchmark)
| Difficulty | Qwen-Image | GPT Image 1 | Seedream 3.0 |
|-----------|------------|-------------|--------------|
| Level-1 (3500 chars) | **97.29%** | 68.37% | 53.48% |
| Level-2 (3000 chars) | **40.53%** | 15.97% | 26.23% |
| Level-3 (1605 chars) | **6.48%** | 3.55% | 1.25% |

#### Image Editing (TI2I)
| Benchmark | Qwen-Image | Ranking |
|-----------|-----------|---------|
| GEdit-Bench-EN | 7.56 | #1 |
| GEdit-Bench-CN | 7.52 | #1 |
| ImgEdit | 4.27 | #1 |

### Extended Capabilities

The multi-task training enables Qwen-Image to perform tasks beyond standard T2I:

1. **Novel View Synthesis**: Competitive with specialized 3D models (PSNR 15.11 on GSO dataset)
2. **Depth Estimation**: Performs on par with state-of-the-art depth models (δ1=0.951 on KITTI)
3. **Chained Editing**: Sequential edits while maintaining consistency
4. **Pose Manipulation**: Modify subject poses while preserving identity
5. **Material Editing**: Change object materials/textures realistically

### Producer-Consumer Training Framework

Qwen-Image uses a distributed training architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  Producer Workers (Data Pipeline)                           │
│  - Image loading and preprocessing                          │
│  - Caption processing and tokenization                      │
│  - VAE encoding (cached or on-the-fly)                     │
│  - Data augmentation                                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Consumer Workers (Model Training)                          │
│  - DiT forward/backward passes                             │
│  - Gradient synchronization (FSDP/DeepSpeed)               │
│  - Optimizer steps                                          │
└─────────────────────────────────────────────────────────────┘
```

This separation allows efficient utilization of heterogeneous hardware and prevents data loading from becoming a bottleneck.

---

## Official Prompting Guidelines

Based on Alibaba's official T2I prompt guide, follow this structure for optimal results:

### Basic Formula
```
Subject + Scene + Style
```

### Advanced Formula
```
Subject + Scene + Style + Camera Language + Atmosphere + Detail
```

### Prompt Components

| Component | Description | Examples |
|-----------|-------------|----------|
| **Subject** | Main focus of image | "a young woman", "a mountain landscape" |
| **Scene** | Environment/setting | "in a coffee shop", "at sunset" |
| **Style** | Visual aesthetic | "oil painting", "photorealistic", "anime" |
| **Camera** | Shot type & angle | "close-up", "wide angle", "bird's eye view" |
| **Atmosphere** | Mood/lighting | "warm golden hour", "moody and dramatic" |
| **Detail** | Specific attributes | "wearing a red dress", "with intricate patterns" |

### Recommended Shot Types
- Extreme close-up (ECU)
- Close-up (CU)
- Medium shot (MS)
- Full shot (FS)
- Wide shot (WS)
- Extreme wide shot (EWS)

### Recommended Styles
- Photorealistic / Photography
- Oil painting / Watercolor
- Digital art / Concept art
- Anime / Manga
- 3D render / CGI
- Ink wash / Chinese painting

---

## Version History

| Version | Date | Pipeline Class | Key Changes |
|---------|------|----------------|-------------|
| Qwen-Image | 2025-08-04 | `QwenImagePipeline` | Initial T2I release, 60-layer MMDiT, Qwen2.5-VL text encoder |
| Qwen-Image-Edit | 2025-08-04 | `QwenImageEditPipeline` | Single-image editing with dual-path conditioning |
| Qwen-Image-Edit-2509 | 2025-09 | `QwenImageEditPlusPipeline` | Multi-image input, ControlNet support, improved consistency |
| Qwen-Image-Edit-2511 | 2025-11 | `QwenImageEditPlusPipeline` | `zero_cond_t`, reduced drift, integrated LoRAs, geometric reasoning |
| Qwen-Image-2512 | 2025-12-31 | `QwenImagePipeline` | Improved realism, textures, text rendering, fine detail |
| Qwen-Image-Layered | 2025-12 | `QwenImageLayeredPipeline` | RGBA layer decomposition, 3D RoPE, layer-index conditioning |

---

## Additional Pipeline Variants (Diffusers)

Beyond the three core pipelines documented above, the HuggingFace diffusers library implements several additional Qwen-Image pipeline variants. These are listed here for reference even though the blissful-tuner training pipeline does not directly use them:

| Pipeline Class | Purpose | Notes |
|---------------|---------|-------|
| `QwenImagePipeline` | Text-to-image generation | Core T2I pipeline |
| `QwenImageEditPipeline` | Single-image editing | Original edit variant |
| `QwenImageEditPlusPipeline` | Multi-image editing (2509/2511) | Enhanced edit with `zero_cond_t` |
| `QwenImageLayeredPipeline` | RGBA layer decomposition | 3D RoPE, layer-index conditioning |
| `QwenImageImg2ImgPipeline` | Image-to-image | Init from existing image |
| `QwenImageInpaintPipeline` | Inpainting | Mask-guided generation |
| `QwenImageControlNetPipeline` | ControlNet-guided generation | Uses `QwenImageControlNetModel` |
| `QwenImageControlNetInpaintPipeline` | ControlNet + inpainting | Combined control |
| `QwenImageEditInpaintPipeline` | Edit + inpainting | Combined edit + mask |
| `ReduxImageEncoder` | Image encoding helper | Used by some pipelines |

Additionally, diffusers provides **modular pipeline** variants under `modular_pipelines/qwenimage/` that decompose the pipeline into reusable blocks (encoders, decoders, denoising, prompt handling).

---

## References

- [Qwen-Image-2512 HuggingFace](https://huggingface.co/Qwen/Qwen-Image-2512)
- [Qwen-Image-Edit-2511 HuggingFace](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Qwen-Image-Layered HuggingFace](https://huggingface.co/Qwen/Qwen-Image-Layered)
- [Qwen-Image Technical Report (arXiv 2508.02324)](https://arxiv.org/abs/2508.02324)
- [Qwen-Image-Layered Technical Report (arXiv 2512.15603)](https://arxiv.org/abs/2512.15603)
- [Qwen-Image Blog](https://qwenlm.github.io/blog/qwen-image/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Comfy-Org Qwen-Image Weights](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
- [Comfy-Org Qwen-Image-Edit Weights](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI)
- [Comfy-Org Qwen-Image-Layered Weights](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI)
