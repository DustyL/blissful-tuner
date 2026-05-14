# WAN 2.2 Architecture Reference

A comprehensive architecture reference for the WAN 2.2 video generation model family, optimized for coding agents, training pipeline development, and debugging.

---

## Quick Reference

| Component | Value | Notes |
|-----------|-------|-------|
| **Model Family** | WAN 2.2 | Alibaba Wan Team |
| **Architecture** | DiT (Diffusion Transformer) | Flow Matching + Cross-Attention |
| **A14B Checkpoints** | 2× ~14.3B (high-noise + low-noise) | Switches between two *separate* DiT checkpoints via `boundary` (not in-model MoE routing) |
| **DiT Dimension** | 5120 | Hidden size (`attention_head_dim × num_attention_heads`) |
| **DiT Layers** | 40 | Transformer blocks |
| **Attention Heads** | 40 | 128 dim per head |
| **FFN Dimension** | 13824 | ~2.7x hidden dim |
| **QK Norm** | RMSNorm across heads | Weight shape `[5120]` (full dim, not per-head) |
| **Text Encoder** | `google/umt5-xxl` | ~4.7B encoder-only params (umT5-XXL) |
| **Text Length** | 512 tokens | Max sequence |
| **Text Output Dim** | 4096 → 5120 | Projected via `text_embedding` MLP |
| **VAE Compression** | 4×8×8 | Temporal × Height × Width (A14B uses Wan2.1-VAE) |
| **Latent Channels** | 16 | z_dim (both in and out) |
| **Patch Size** | (1, 2, 2) | Additional 2x spatial compression |
| **RoPE Max Seq Len** | 1024 | Maximum position index for RoPE frequencies |
| **Freq Dim** | 256 | Sinusoidal time embedding dimension (θ=10000) |
| **Precision** | bfloat16 | `param_dtype` for training and inference |
| **Weight Init** | Xavier uniform (Linear), Normal(σ=0.02) (embed), zeros (head) | See Weight Initialization section |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WAN 2.2 Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   Input      │     │   Text Prompt    │     │   Timestep t     │    │
│  │   Video      │     │                  │     │                  │    │
│  │  [F,H,W,3]   │     │                  │     │                  │    │
│  └──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘    │
│         │                      │                        │              │
│         ▼                      ▼                        ▼              │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   Wan-VAE    │     │   umT5-XXL       │     │   Sinusoidal     │    │
│  │   Encoder    │     │   Encoder        │     │   + MLP          │    │
│  │  (4×8×8)     │     │   (~4.7B enc)    │     │                  │    │
│  └──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘    │
│         │                      │                        │              │
│         ▼                      │                        │              │
│  ┌──────────────┐              │                        │              │
│  │ Patch Embed  │              │                        │              │
│  │ Conv3d(1,2,2)│              │                        │              │
│  └──────┬───────┘              │                        │              │
│         │                      │                        │              │
│         ▼                      ▼                        ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │     High/Low-Noise Checkpoint Selection (boundary-based)        │   │
│  │  ┌───────────────────┐    ┌───────────────────┐                 │   │
│  │  │ High-noise model   │    │ Low-noise model    │                 │   │
│  │  │ (t >= boundary)    │    │ (t < boundary)     │                 │   │
│  │  │ Layout/Structure   │    │ Details/Refine     │                 │   │
│  │  └───────────────────┘    └───────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              WanAttentionBlock × 40                             │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │ Self-Attention (Full Spatio-Temporal + 3D RoPE)         │   │   │
│  │  │ ├── LayerNorm + Modulation (scale/shift)               │   │   │
│  │  │ ├── Q, K, V projection (dim → dim)                     │   │   │
│  │  │ ├── RMSNorm on Q, K (across-heads, weight=[5120])     │   │   │
│  │  │ ├── 3D RoPE: rope_apply(q/k, grid_sizes, freqs)       │   │   │
│  │  │ └── FlashAttention                                     │   │   │
│  │  ├─────────────────────────────────────────────────────────┤   │   │
│  │  │ Cross-Attention (Text + optional Image Conditioning)   │   │   │
│  │  │ ├── Q from visual, K/V from text embeddings           │   │   │
│  │  │ ├── I2V: +add_k_proj/add_v_proj for image tokens     │   │   │
│  │  │ └── No positional encoding on text                     │   │   │
│  │  ├─────────────────────────────────────────────────────────┤   │   │
│  │  │ Feed-Forward Network                                   │   │   │
│  │  │ ├── LayerNorm + Modulation                            │   │   │
│  │  │ ├── Linear(dim → ffn_dim) + GELU(tanh approx)         │   │   │
│  │  │ └── Linear(ffn_dim → dim)                             │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────┐                                                      │
│  │    Head      │  LayerNorm + Modulation + Linear                     │
│  │  (Unpatchify)│  Output: [C_out, F, H/8, W/8]                       │
│  └──────┬───────┘                                                      │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────┐                                                      │
│  │   Wan-VAE    │                                                      │
│  │   Decoder    │                                                      │
│  └──────────────┘                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model Variants

### WAN 2.2 A14B (Dual-checkpoint “MoE-like” setup)

WAN 2.2 A14B is commonly described as a 2-expert “MoE”, but in this repo it’s implemented as **two separate DiT checkpoints**
(`low_noise_model` and `high_noise_model`) and the pipeline **switches which checkpoint is used** based on a fixed `boundary`.

| Checkpoint | Activation | Focus | Parameters | Weight Size (fp32) |
|--------|------------|-------|------------|-------------------|
| **High-noise model** | t >= boundary | Overall layout, composition | ~14.29B | 57.15 GB (T2V), 57.16 GB (I2V) |
| **Low-noise model** | t < boundary | Fine details, textures | ~14.29B | Same architecture |
| **Total** | - | - | ~28.6B (only one "active" at a given timestep; memory may include both if loaded) | - |

> **Note:** The I2V experts are ~1.6 MB larger than T2V due to the expanded `patch_embedding` (36 input channels vs 16).

#### Boundary (`boundary` / `timestep_boundary`)

Official materials often motivate the switch using SNR, but the code path here uses a **fixed threshold** from config.
Internally, timesteps are typically integers in `[0..1000]` and the code compares `t / 1000.0` against `boundary`.

| Task | Boundary | Interpretation (this repo) |
|------|----------|----------------|
| **T2V** | 0.875 | Use **high-noise** checkpoint for `t >= 0.875`, otherwise low-noise |
| **I2V** | 0.900 | Use **high-noise** checkpoint for `t >= 0.900`, otherwise low-noise |

```python
# Checkpoint switching logic used in this repo
# (t is typically in [0..1000] during training/inference codepaths)
if (t / 1000.0) >= boundary:
    model = high_noise_model  # high-noise region (early steps if timesteps decrease during sampling)
else:
    model = low_noise_model   # lower-noise region (later refinement)
```

### Model Configurations

#### T2V-A14B (Text-to-Video)

```python
# From wan/configs/wan_t2v_A14B.py
t2v_A14B = {
    # Text Encoder
    "t5_checkpoint": "models_t5_umt5-xxl-enc-bf16.pth",
    "t5_tokenizer": "google/umt5-xxl",
    "t5_dtype": torch.bfloat16,
    "text_len": 512,

    # VAE
    "vae_checkpoint": "Wan2.1_VAE.pth",
    "vae_stride": (4, 8, 8),  # Temporal, Height, Width compression

    # Transformer (DiT)
    "patch_size": (1, 2, 2),
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "num_heads": 40,
    "num_layers": 40,
    "window_size": (-1, -1),  # Global attention
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,

    # MoE Checkpoints
    "low_noise_checkpoint": "low_noise_model",
    "high_noise_checkpoint": "high_noise_model",

    # Inference
    "sample_shift": 12.0,
    "sample_steps": 40,
    "boundary": 0.875,
    "sample_guide_scale": (3.0, 4.0),  # (low_noise, high_noise)

    # Generation
    "num_train_timesteps": 1000,
    "sample_fps": 16,
    "frame_num": 81,
    "param_dtype": torch.bfloat16,
}
```

#### I2V-A14B (Image-to-Video)

```python
# From wan/configs/wan_i2v_A14B.py
i2v_A14B = {
    # Same as T2V except:
    "in_dim": 36,             # IMPORTANT: I2V changes patch_embedding input channels (16 -> 36)
    "sample_shift": 5.0,      # Lower shift for I2V
    "boundary": 0.900,        # Later switch to low-noise expert
    "sample_guide_scale": (3.5, 3.5),  # Equal guidance for both
}
```

> **I2V conditioning note (WAN 2.2):** The image conditioning is done via **extra latent channels** (passed as `y` and concatenated
> to the noisy latents), *not* via CLIP tokens in the cross-attention context (that CLIP path exists for WAN 2.1 I2V).
> The Diffusers `model_index.json` confirms this: `image_encoder: [null, null]` and `image_processor: [null, null]`.
> The 36 input channels break down as: 16 (noisy latent) + 4 (temporal mask) + 16 (image latent from VAE-encoded reference frame).

#### TI2V-5B (Dense Model with High-Compression VAE)

```python
# From wan/configs/wan_ti2v_5B.py
ti2v_5B = {
    # DiT Architecture
    "dim": 3072,
    "ffn_dim": 14336,           # ~4.67x hidden dim (vs ~2.7x for A14B)
    "num_heads": 24,
    "num_layers": 30,
    "patch_size": (1, 2, 2),

    # VAE (uses Wan2.2_VAE, NOT Wan2.1_VAE)
    "vae_checkpoint": "Wan2.2_VAE.pth",
    "vae_stride": (4, 16, 16),  # 2x more spatial compression than A14B
    "vae_z_dim": 48,            # 3x more latent channels

    # Generation
    "frame_num": 121,           # 5s @ 24fps (vs 81 frames @ 16fps for A14B)
    "sample_fps": 24,
    "sample_shift": 5.0,
    "sample_steps": 40,
    "sample_guide_scale": (2.5, 2.5),

    # Text
    "text_len": 512,
    "param_dtype": torch.bfloat16,
}
```

| Parameter | TI2V-5B | A14B (T2V/I2V) |
|-----------|---------|-----------------|
| **Parameters** | ~5B (dense, no MoE) | ~14.3B × 2 experts |
| **Dim** | 3072 | 5120 |
| **FFN Dim** | 14336 (~4.67×) | 13824 (~2.7×) |
| **Heads** | 24 (128/head) | 40 (128/head) |
| **Layers** | 30 | 40 |
| **VAE** | Wan2.2 (z=48, 4×16×16) | Wan2.1 (z=16, 4×8×8) |
| **Total Compression** | 4×32×32 with patchification | 4×16×16 with patchification |
| **FPS** | 24 | 16 |
| **Frames** | 121 (5s) | 81 (5s) |
| **Guidance** | (2.5, 2.5) | T2V: (3.0, 4.0), I2V: (3.5, 3.5) |
| **Consumer GPU** | Yes (single 4090) | 80GB+ (or heavy offloading for 24GB) |

> **Integration note:** Upstream WAN 2.2 includes TI2V-5B configs/sizes, but blissful-tuner currently does not expose a `--task ti2v-5B`
> entry in `WAN_CONFIGS` (even though some TI2V sizes appear in `SUPPORTED_SIZES`). Treat this section as upstream/paper reference unless you
> add the missing config wiring.

#### S2V-A14B (Subject-to-Video / Audio-Driven)

```python
# From wan/configs/wan_s2v_14B.py — reference only
s2v_A14B = {
    "audio_dim": 1024,              # wav2vec audio features
    "motion_encoder_dim": 512,      # Motion conditioning
    "sample_shift": 3,
    "sample_guide_scale": 4.5,      # Single value (not tuple)
    "sample_steps": 20,
    "sample_fps": 30,               # Higher than T2V/I2V
    "frame_num": 77,
    # Uses CLIP encoder + face_encoder for subject identity
    # Audio injection via audio_proj layers at specific blocks
}
```

#### Animate-A14B (Animation / Character Motion)

```python
# From wan/configs/wan_animate_14B.py — reference only
animate_A14B = {
    "motion_encoder_dim": 512,
    "sample_steps": 20,
    "sample_fps": 30,
    "frame_num": 77,
    # Uses CLIP encoder + face_encoder (same as S2V)
    # Motion-conditioned animation pipeline
}
```

> **Note:** S2V (Subject-to-Video) and Animate are upstream tasks that blissful-tuner does not currently support. They are included here for completeness and potential future implementation reference.

---

## Component Details

### 1. Wan-VAE (Spatio-Temporal VAE)

#### Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model Size** | 127M | Compact design |
| **Compression Ratio** | 4×8×8 | T×H×W |
| **Latent Channels** | 16 | z_dim |
| **Input Format** | T×H×W×3 | First frame processed separately, remaining frames in chunks of 4 |
| **Output Format** | (1 + (T-1)/4)×H/8×W/8×16 | Training typically uses `T = 4k + 1` (e.g. 81 → 21 latent frames) |
| **Normalization** | RMSNorm | Replaces GroupNorm for causality |

#### Causal 3D Convolution

```python
class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution - future frames don't influence past"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Asymmetric temporal padding: (2*pad, 0) instead of (pad, pad)
        self._padding = (
            self.padding[2], self.padding[2],  # Width: symmetric
            self.padding[1], self.padding[1],  # Height: symmetric
            2 * self.padding[0], 0             # Temporal: causal (left only)
        )
        self.padding = (0, 0, 0)
```

#### Feature Cache Mechanism (Infinite Video Support)

```python
# Chunk-wise encoding/decoding for arbitrarily long videos
# - Process video in (1 + T/4) chunks
# - Each chunk handles 4 frames maximum
# - Cache features from preceding chunks for temporal continuity
# - Zero padding for initial chunk
CACHE_T = 2  # Cache last 2 frames for temporal convolutions
```

#### Latent Normalization Values

The pipeline normalizes latents using per-channel mean/std statistics before training and reverses them before decoding:

```python
# WAN 2.1 VAE (z_dim=16, used by A14B models)
latents_mean = [-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
                 0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921]
latents_std  = [ 2.8184,  1.4541,  2.3275,  2.6558,  1.2196,  1.7708,  2.6052,  2.0743,
                 3.2687,  2.1526,  2.8652,  1.5579,  1.6382,  1.1253,  2.8251,  1.9160]

# Encode: latents_normalized = (latents - mean) * (1/std)
# Decode: latents_original = latents_normalized / (1/std) + mean
```

#### Wan2.2-VAE (High-Compression, TI2V-5B only)

The TI2V-5B model uses a newer VAE with dramatically higher compression:

| Parameter | Wan2.1-VAE (A14B) | Wan2.2-VAE (TI2V-5B) |
|-----------|-------------------|----------------------|
| **z_dim** | 16 | 48 |
| **Stride** | 4×8×8 | 4×16×16 |
| **Spatial Compression** | 8× | 16× (2× more) |
| **Latent Spatial** | H/8 × W/8 | H/16 × W/16 |
| **Architecture** | Shared enc/dec dims | Separate enc/dec dims |
| **Downsampling** | Strided Conv3d | `AvgDown3D` (average pooling) |
| **Upsampling** | Nearest + Conv3d | `DupUp3D` (duplication) |
| **Patchification** | None | 2×2 `patchify`/`unpatchify` in latent space |
| **Checkpoint** | `Wan2.1_VAE.pth` | `Wan2.2_VAE.pth` |

```python
# Wan2.2-VAE key architectural differences:
# - Separate encoder/decoder channel configs (enc_dim ≠ dec_dim)
# - Patchify: rearranges latent [B,C,T,H,W] → [B,C*4,T,H/2,W/2] before output
# - Unpatchify: reverses before decoding
# - AvgDown3D for spatial downsampling (smoother gradients vs strided conv)
# - DupUp3D for spatial upsampling (duplication vs nearest-neighbor)
```

> **Important:** A14B models (T2V, I2V) use `Wan2.1_VAE.pth`. Only TI2V-5B uses `Wan2.2_VAE.pth`. Using the wrong VAE will produce garbage output.

#### VAE Performance Benchmarks

| Metric | Wan-VAE | HunyuanVideo | CogVideoX | Open Sora Plan |
|--------|---------|--------------|-----------|----------------|
| **PSNR (720p)** | ~32.5 | ~31.8 | ~30.2 | ~29.5 |
| **Speed** | 2.5x faster | 1x baseline | 0.8x | 1.2x |
| **Parameters** | 127M | ~200M | ~150M | ~180M |

### 2. Text Encoder (umT5-XXL)

WAN 2.2 uses only the **encoder** portion of umT5-XXL (no decoder). The Diffusers class is `UMT5EncoderModel`.

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | umT5-XXL (Google) | `UMT5EncoderModel` in Diffusers |
| **Parameters** | ~4.7B (encoder-only) | Full T5 enc+dec is ~13B; only encoder is used |
| **Hidden Dim (`d_model`)** | 4096 | Output dimension before DiT projection |
| **FFN Dim (`d_ff`)** | 10240 | ~2.5× hidden dim |
| **FFN Type** | Gated-GELU | `feed_forward_proj: "gated-gelu"`, `dense_act_fn: "gelu_new"` |
| **Attention Heads** | 64 | `d_kv: 64` per head → key/value dim = 64 |
| **Encoder Layers** | 24 | |
| **Attention** | Bidirectional | Non-causal (unlike LLM decoders) |
| **Position Encoding** | Relative bias | `relative_attention_num_buckets: 32`, `max_distance: 128` |
| **Languages** | Multilingual | Chinese, English, and 100+ languages |
| **Max Length** | 512 tokens | |
| **Output Dim** | 4096 → 5120 | Projected via `text_embedding` MLP in DiT |
| **Vocab Size** | 256,384 | SentencePiece tokenizer |
| **Precision** | bfloat16 | |
| **Scalable Attention** | true | Enables optimized attention kernels |

#### T5 Attention: No Scaling Factor

A unique architectural detail: the T5 attention implementation used by WAN **does not** apply the standard `1/√d_k` scaling factor:

```python
# Standard attention:  scores = Q @ K^T / sqrt(d_k)
# T5 attention:        scores = Q @ K^T              ← no scaling!
```

This is a deliberate choice from the [original T5 paper](https://arxiv.org/abs/1910.10683) — the scaling is absorbed into the learned parameters via relative attention bias. The native WAN implementation (`wan/modules/t5.py`) preserves this behavior. Be aware of this if you ever need to debug text encoder outputs or implement custom attention for the text conditioning path.

#### Why umT5 over LLMs?

From ablation studies in the technical report:
1. **Bidirectional attention** better suited for diffusion models than causal LLMs
2. **Superior convergence** at same parameter scale
3. **Strong multilingual** support for Chinese and English visual text generation
4. **Better compositional understanding** compared to Qwen/GLM

### 3. Diffusion Transformer (DiT)

#### WAN 2.2 vs 2.1 Key Differences

| Feature | WAN 2.1 | WAN 2.2 |
|---------|---------|---------|
| **Architecture** | Single model (dense) | Dual-expert (timestep-switched) |
| **Time Embedding** | Per-batch (scalar timestep) | Per-batch (T2V) or per-token (I2V with `expand_timesteps`) |
| **Time Projection** | Projects to `[B, 6, dim]` | `[B, 6, dim]` (T2V) or `[B, L, 6, dim]` (I2V) |
| **I2V Conditioning** | CLIP image encoder + cross-attention | Latent concatenation (36 input channels, no CLIP) |
| **I2V Image Attention** | Via CLIP cross-attention tokens | Via `add_k_proj`/`add_v_proj` in cross-attn blocks |
| **VAE** | Wan2.1-VAE (shared) | A14B uses Wan2.1-VAE; TI2V-5B uses Wan2.2-VAE (4×16×16) |
| **Text Encoder** | umT5-XXL (same) | umT5-XXL (same) |
| **Guidance Scale** | Single value | Tuple `(low_noise, high_noise)` per expert |
| **Tasks** | T2V, I2V (1.3B, 14B) | T2V, I2V, TI2V, S2V, Animate (A14B + 5B) |
| **Flow Shift** | Variable | T2V: 12.0, I2V: 5.0, TI2V: 5.0, S2V: 3.0 |

> **Per-token time embedding:** In the native WAN implementation, the timestep is *always* expanded to `[B, seq_len]` before sinusoidal embedding, even for T2V. The modulation tensor is therefore always `[B, L, 6, dim]`. However, for T2V all positions receive the **same** timestep value (uniform broadcast), so this is semantically equivalent to per-batch. True per-token variation only occurs with `expand_timesteps=True` (I2V/TI2V-5B), where the pipeline generates per-token timesteps from a spatial mask: `temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()`. This gives each spatial-temporal position its own effective timestep (0 for the conditioned first frame, `t` for frames being denoised). The Diffusers implementation uses `[B, 6, dim]` for T2V and only expands to `[B, L, 6, dim]` when `expand_timesteps=True`.

#### Precision Handling (FP32 Preservation)

All normalization layers in the DiT use `FP32LayerNorm`, which forces computation in float32 regardless of input dtype:

```python
class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs):
        return F.layer_norm(inputs.float(), ...).to(origin_dtype)
```

The Diffusers implementation explicitly keeps these modules in FP32 during mixed-precision:

```python
_keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
```

This prevents numerical instability in the modulation pathway (scale/shift/gate) and normalization layers during bf16/fp16 training.

#### WanAttentionBlock Structure

```python
class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, ...):
        # Normalization layers
        self.norm1 = WanLayerNorm(dim, eps)      # Pre-self-attn
        self.norm2 = WanLayerNorm(dim, eps)      # Pre-FFN
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)  # Pre-cross-attn

        # Attention layers (qk_norm="rms_norm_across_heads" normalizes across full dim [5120])
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        # FFN (Diffusers uses FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate"))
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),       # ffn.0 / ffn.net.0.proj
            nn.GELU(approximate='tanh'),   # F.gelu(x, approximate="tanh")
            nn.Linear(ffn_dim, dim)        # ffn.2 / ffn.net.2
        )

        # Modulation (shared AdaLN - reduces parameters by ~25%)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
```

#### Modulation (AdaLN-single)

```python
# 6 modulation slots per block:
# [shift1, scale1, gate1, shift2, scale2, gate2]
# Applied to: self-attn (1-3), FFN (4-6)

def forward(self, x, e, ...):
    # e: time embedding projected to [B, L, 6, dim]
    e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

    # Self-attention with modulation
    y = self.self_attn(
        self.norm1(x) * (1 + e[1]) + e[0],  # scale + shift
        seq_lens, grid_sizes, freqs
    )
    x = x + y * e[2]  # gate

    # Cross-attention (no modulation)
    x = x + self.cross_attn(self.norm3(x), context, context_lens)

    # FFN with modulation
    y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
    x = x + y * e[5]
```

#### 3D RoPE (Rotary Position Embedding)

```python
# RoPE frequency dimensions split for 3D:
# Real-valued per-head dimension is `d = dim // num_heads` (A14B: 5120/40 = 128).
# The model constructs freqs using the real-dim split:
# - Temporal: d - 4*(d//6) dims = 128 - 4*(128//6) = 128 - 84 = 44 dims (~34.4%)
# - Height:   2*(d//6) dims     = 2*(128//6)       = 42 dims        (~32.8%)
# - Width:    2*(d//6) dims     = 2*(128//6)        = 42 dims        (~32.8%)
#
# Note: RoPE freqs are stored as complex pairs, so the freqs tensor's last dim is `c = d//2`,
# and split sizes use the complex-pair dimensions.
# rope_max_seq_len = 1024 (from config.json)

def rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2) / dim)
    )
    return torch.polar(torch.ones_like(freqs), freqs)

# Application to Q, K (matches `rope_apply_inplace_cached` / `rope_apply` in this repo)
def rope_apply(x, grid_sizes, freqs):
    # x: [B, L, num_heads, head_dim]
    # freqs: [max_seq_len, head_dim//2] complex
    d = x.size(3)        # head_dim
    c = d // 2           # complex-pair dim

    # freqs split (complex-pair dims): [temporal, height, width]
    # Equivalent real-dim split: [d - 4*(d//6), 2*(d//6), 2*(d//6)]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # For each sample, expand freqs to match grid
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1)
        x_i = x_i * freqs_i  # Complex multiplication
```

> **Native vs Diffusers RoPE implementation:**
> - **Native WAN** uses complex arithmetic: `torch.polar(ones, freqs)` creates unit-magnitude complex numbers, then `x * freqs` performs complex multiplication (which implements rotation).
> - **Diffusers** converts to real-valued cos/sin: `freqs_cos = freqs.real`, `freqs_sin = freqs.imag`, then applies the standard rotation formula: `x_rotated = x * cos + rotate_half(x) * sin`.
> - Both produce **identical results** — the native approach is more elegant but requires complex tensor support; the Diffusers approach is more widely compatible.
> - **Debugging tip:** If comparing native vs Diffusers RoPE outputs, note that native operates on complex-valued tensors while Diffusers operates on real-valued tensors with twice the last dimension. Use `torch.view_as_real()` / `torch.view_as_complex()` to convert between them.

#### Weight Initialization

The native WAN DiT uses specific initialization patterns (from `wan/modules/model.py:init_weights`):

```python
def init_weights(self):
    # Linear layers: Xavier uniform
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Embedding layers: Normal(mean=0, std=0.02)
    nn.init.normal_(self.patch_embedding.weight, std=0.02)
    nn.init.normal_(self.text_embedding[0].weight, std=0.02)

    # Output head: Zero-initialized (important for stable training start)
    nn.init.zeros_(self.head.head.weight)
    nn.init.zeros_(self.head.head.bias)
```

> **Training relevance:** The zero-initialized output head means the model starts by predicting zero velocity (no denoising). This is standard for flow matching models and ensures stable initial training. If you ever need to extend the model (e.g., adding output channels), zero-initialize the new weights.

#### Sinusoidal Time Embedding

```python
# From wan/modules/model.py:sinusoidal_embedding_1d
# Uses theta=10000, cos-then-sin ordering (NOT interleaved)
def sinusoidal_embedding_1d(dim, position):
    # position: [B] or [B, L] tensor of timesteps
    half = dim // 2
    sinusoid = torch.outer(
        position,
        1.0 / torch.pow(10000, torch.arange(half).float() / half)
    )
    return torch.cat([sinusoid.cos(), sinusoid.sin()], dim=-1)  # [B, dim] or [B, L, dim]
    # → fed into time_embedding MLP: Linear(freq_dim, dim) → SiLU → Linear(dim, dim)
    # → then time_projection: SiLU → Linear(dim, 6*dim) → reshape to [B, (L,) 6, dim]
```

> **Native time embedding shape:** The native WAN code always expands the timestep to `[B, seq_len]` before embedding, even for T2V where all positions share the same value. This means the modulation tensor is always `[B, L, 6, dim]` in native code. However, for T2V this is just a broadcast (all positions identical) — true per-token variation only occurs with `expand_timesteps` (I2V/TI2V) where the spatial mask creates different timesteps per position.

#### I2V Image Attention (Cross-Attention Extension)

For I2V models (`added_kv_proj_dim != None`), each cross-attention block has additional projections for image tokens:

```python
# In WanAttention (I2V only):
self.add_k_proj = nn.Linear(added_kv_proj_dim, inner_dim, bias=True)
self.add_v_proj = nn.Linear(added_kv_proj_dim, inner_dim, bias=True)
self.norm_added_k = nn.RMSNorm(inner_dim, eps=eps)  # No elementwise_affine

# In WanAttnProcessor.__call__:
# 1. Split encoder_hidden_states at hardcoded 512-token boundary
image_context_length = encoder_hidden_states.shape[1] - 512
encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
encoder_hidden_states_text = encoder_hidden_states[:, image_context_length:]

# 2. Compute main text cross-attention
hidden_states = attention(query, key_text, value_text)

# 3. Compute separate image attention via add_k_proj/add_v_proj
key_img = norm_added_k(add_k_proj(encoder_hidden_states_img))
value_img = add_v_proj(encoder_hidden_states_img)
hidden_states_img = attention(query, key_img, value_img)

# 4. Sum image and text attention outputs
hidden_states = hidden_states + hidden_states_img
```

> **Note:** The 512-token boundary is hardcoded — image embeddings are prepended to text embeddings, and the split uses `shape[1] - 512` to separate them. The image attention uses RMSNorm on keys only (no elementwise_affine), unlike the main attention which uses elementwise_affine=True.

---

## Weight Tensor Naming Patterns

### DiT Weights (per block)

```
blocks.{N}.self_attn.q.weight        # [5120, 5120]
blocks.{N}.self_attn.q.bias          # [5120]
blocks.{N}.self_attn.k.weight        # [5120, 5120]
blocks.{N}.self_attn.k.bias          # [5120]
blocks.{N}.self_attn.v.weight        # [5120, 5120]
blocks.{N}.self_attn.v.bias          # [5120]
blocks.{N}.self_attn.o.weight        # [5120, 5120]
blocks.{N}.self_attn.o.bias          # [5120]
blocks.{N}.self_attn.norm_q.weight   # [5120] (RMSNorm)
blocks.{N}.self_attn.norm_k.weight   # [5120] (RMSNorm)

blocks.{N}.cross_attn.q.weight       # [5120, 5120]
blocks.{N}.cross_attn.q.bias         # [5120]
blocks.{N}.cross_attn.k.weight       # [5120, 5120]
blocks.{N}.cross_attn.k.bias         # [5120]
blocks.{N}.cross_attn.v.weight       # [5120, 5120]
blocks.{N}.cross_attn.v.bias         # [5120]
blocks.{N}.cross_attn.o.weight       # [5120, 5120]
blocks.{N}.cross_attn.o.bias         # [5120]
blocks.{N}.cross_attn.norm_q.weight  # [5120]
blocks.{N}.cross_attn.norm_k.weight  # [5120]

blocks.{N}.ffn.0.weight              # [13824, 5120] (Linear + GELU)
blocks.{N}.ffn.0.bias                # [13824]
blocks.{N}.ffn.2.weight              # [5120, 13824]
blocks.{N}.ffn.2.bias                # [5120]

blocks.{N}.norm3.weight              # [5120] (cross_attn_norm)
blocks.{N}.norm3.bias                # [5120]

blocks.{N}.modulation                # [1, 6, 5120]
```

### Embeddings

```
patch_embedding.weight               # [5120, 16, 1, 2, 2] (Conv3d)
patch_embedding.bias                 # [5120]

text_embedding.0.weight              # [5120, 4096]
text_embedding.0.bias                # [5120]
text_embedding.2.weight              # [5120, 5120]
text_embedding.2.bias                # [5120]

time_embedding.0.weight              # [5120, 256]
time_embedding.0.bias                # [5120]
time_embedding.2.weight              # [5120, 5120]
time_embedding.2.bias                # [5120]

time_projection.1.weight             # [30720, 5120] (6 * dim)
time_projection.1.bias               # [30720]
```

> **Variant caveat:** for **I2V-A14B**, `in_dim=36`, so `patch_embedding.weight` is `[5120, 36, 1, 2, 2]`.

> **Checkpoint key caveats:** Some WAN checkpoints (notably 1.3B) use a `model.diffusion_model.` prefix (the loader strips it).
> If you train with `--compile` and swap high/low weights, keys inside `blocks.*` may be nested under `blocks.{N}._orig_mod.*`.

### Head

```
head.norm.weight                     # (none - no affine)
head.head.weight                     # [64, 5120] (out_dim * prod(patch_size))
head.head.bias                       # [64]
head.modulation                      # [1, 2, 5120]
```

### Diffusers Weight Naming (HuggingFace Format)

When loading from HuggingFace Diffusers checkpoints (`WanTransformer3DModel`), weight names differ from the native format. The mapping below shows **Diffusers → Native** for each module:

```
# Self-Attention
blocks.{N}.attn1.to_q.weight          → blocks.{N}.self_attn.q.weight
blocks.{N}.attn1.to_q.bias            → blocks.{N}.self_attn.q.bias
blocks.{N}.attn1.to_k.weight          → blocks.{N}.self_attn.k.weight
blocks.{N}.attn1.to_k.bias            → blocks.{N}.self_attn.k.bias
blocks.{N}.attn1.to_v.weight          → blocks.{N}.self_attn.v.weight
blocks.{N}.attn1.to_v.bias            → blocks.{N}.self_attn.v.bias
blocks.{N}.attn1.to_out.0.weight      → blocks.{N}.self_attn.o.weight
blocks.{N}.attn1.to_out.0.bias        → blocks.{N}.self_attn.o.bias
blocks.{N}.attn1.norm_q.weight        → blocks.{N}.self_attn.norm_q.weight
blocks.{N}.attn1.norm_k.weight        → blocks.{N}.self_attn.norm_k.weight

# Cross-Attention
blocks.{N}.attn2.to_q.weight          → blocks.{N}.cross_attn.q.weight
blocks.{N}.attn2.to_k.weight          → blocks.{N}.cross_attn.k.weight
blocks.{N}.attn2.to_v.weight          → blocks.{N}.cross_attn.v.weight
blocks.{N}.attn2.to_out.0.weight      → blocks.{N}.cross_attn.o.weight
blocks.{N}.attn2.norm_q.weight        → blocks.{N}.cross_attn.norm_q.weight
blocks.{N}.attn2.norm_k.weight        → blocks.{N}.cross_attn.norm_k.weight

# I2V Image Attention (only present when added_kv_proj_dim != None)
blocks.{N}.attn2.add_k_proj.weight    → blocks.{N}.cross_attn.k_img.weight
blocks.{N}.attn2.add_v_proj.weight    → blocks.{N}.cross_attn.v_img.weight
blocks.{N}.attn2.norm_added_k.weight  → blocks.{N}.cross_attn.norm_k_img.weight

# FFN
blocks.{N}.ffn.net.0.proj.weight      → blocks.{N}.ffn.0.weight
blocks.{N}.ffn.net.0.proj.bias        → blocks.{N}.ffn.0.bias
blocks.{N}.ffn.net.2.weight           → blocks.{N}.ffn.2.weight
blocks.{N}.ffn.net.2.bias             → blocks.{N}.ffn.2.bias

# Cross-Attention Norm
blocks.{N}.norm2.weight               → blocks.{N}.norm3.weight
blocks.{N}.norm2.bias                 → blocks.{N}.norm3.bias

# Modulation (scale_shift_table)
blocks.{N}.scale_shift_table          → blocks.{N}.modulation

# Embeddings
patch_embed.proj.weight               → patch_embedding.weight
patch_embed.proj.bias                 → patch_embedding.bias
condition_embedder.text_embedder.linear_1.{weight,bias}  → text_embedding.0.{weight,bias}
condition_embedder.text_embedder.linear_2.{weight,bias}  → text_embedding.2.{weight,bias}
condition_embedder.time_embedder.linear_1.{weight,bias}  → time_embedding.0.{weight,bias}
condition_embedder.time_embedder.linear_2.{weight,bias}  → time_embedding.2.{weight,bias}
condition_embedder.time_proj.weight                → time_projection.1.weight
condition_embedder.time_proj.bias                  → time_projection.1.bias

# I2V Image Embedder (only present for I2V models)
condition_embedder.image_embedder.norm1.*  → img_emb.proj.0.*
condition_embedder.image_embedder.ff.net.0.proj.*  → img_emb.proj.1.*
condition_embedder.image_embedder.ff.net.2.*       → img_emb.proj.3.*
condition_embedder.image_embedder.norm2.*  → img_emb.proj.4.*

# Head
norm_out.weight                       → head.norm.weight
proj_out.weight                       → head.head.weight
proj_out.bias                         → head.head.bias
scale_shift_table                     → head.modulation
```

> **Key differences:** Diffusers uses `attn1`/`attn2` for self/cross-attention (standard Diffusers convention), `ffn.net.{0,2}` for FFN layers, and `scale_shift_table` for modulation parameters.
>
> **Norm naming swap (critical!):** The conversion script performs a deliberate 3-way swap:
> - Native `norm1` (pre-self-attn) → Diffusers `norm1` (unchanged)
> - Native `norm3` (pre-cross-attn, with `elementwise_affine=True`) → Diffusers `norm2`
> - Native `norm2` (pre-FFN) → Diffusers `norm3`
>
> This means Diffusers `norm2` has `elementwise_affine=True` (the cross-attn norm) while `norm1`/`norm3` have `elementwise_affine=False`. The conversion uses a `norm__placeholder` intermediate to avoid collisions.

#### Variable-Length Flash Attention

The native WAN implementation (`wan/modules/attention.py`) supports **variable-length sequences** within a batch using FlashAttention's `varlen` mode:

```python
# When sequences in a batch have different lengths (e.g., different resolutions in bucket training):
# - Sequences are concatenated into a single 1D tensor
# - cu_seqlens (cumulative sequence lengths) tracks boundaries
# - FA's varlen_func handles the ragged batch efficiently
#
# Supports both FlashAttention-2 and FlashAttention-3 with automatic fallback:
#   FA3 → FA2 → Error
```

> **Training relevance:** This is how the native WAN code handles mixed-resolution batches during training. Blissful-tuner's attention module (`src/musubi_tuner/modules/attention.py`) implements its own variable-length handling — be aware of potential behavioral differences if comparing outputs.

### LoRA Target Modules (Blissful Tuner)

```python
# src/musubi_tuner/networks/lora_wan.py
# LoRA targeting is class-based, not a hard-coded module list.
WAN_TARGET_REPLACE_MODULES = ["WanAttentionBlock"]

# Default exclude patterns (anything matching these paths is skipped):
exclude_patterns.append(r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*")
```

### LoRA Format Conversion (Diffusers ↔ Native/Musubi)

Diffusers supports loading LoRA weights from three formats:

| Format | Key Pattern | Converter |
|--------|------------|-----------|
| **Diffusers** | `transformer.blocks.{i}.attn1.to_q.lora_A.weight` | Native (no conversion) |
| **Native WAN** | `blocks.{i}.self_attn.q.lora_down.weight` | `_convert_non_diffusers_wan_lora_to_diffusers()` |
| **Musubi Tuner** | `lora_unet_blocks_{i}_self_attn_q.lora_down.weight` | `_convert_musubi_wan_lora_to_diffusers()` |

Key conversion patterns (Native → Diffusers):
```
self_attn.{q,k,v,o}.lora_{down,up}  → attn1.{to_q,to_k,to_v,to_out.0}.lora_{A,B}
cross_attn.{q,k,v,o}.lora_{down,up} → attn2.{to_q,to_k,to_v,to_out.0}.lora_{A,B}
cross_attn.{k_img,v_img}.lora_{down,up} → attn2.{add_k_proj,add_v_proj}.lora_{A,B}
ffn.{0,2}.lora_{down,up}            → ffn.{net.0.proj,net.2}.lora_{A,B}
```

Alpha scaling is applied during conversion: `scale = alpha / rank`.

> **T2V → I2V expansion:** Diffusers can auto-expand T2V LoRA for I2V by zero-initializing `add_k_proj`/`add_v_proj` LoRA layers via `_maybe_expand_t2v_lora_for_i2v()`.

---

## Training Pipeline

### Flow Matching Objective

```python
# Rectified Flow formulation
x_t = t * x_1 + (1 - t) * x_0  # Linear interpolation
v_t = x_1 - x_0                 # Target velocity

# Loss
loss = MSE(model(x_t, t, context), v_t)
```

### Timestep Sampling

| Parameter | T2V Value | I2V Value |
|-----------|-----------|-----------|
| **Sampling** | `shift` | `shift` |
| **Discrete Flow Shift** | 12.0 | 5.0 |
| **Train Timesteps** | 1000 | 1000 |
| **Sample Steps** | 40 | 40 |

### Scheduler Configuration (Diffusers)

The Diffusers pipeline uses `UniPCMultistepScheduler` with the following config:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Scheduler** | `UniPCMultistepScheduler` | Unified Predictor-Corrector |
| **Prediction Type** | `flow_prediction` | Flow matching velocity prediction |
| **Flow Shift** | 3.0 | `time_shift_type: "exponential"` (different from training `--discrete_flow_shift 12.0`) |
| **Num Train Timesteps** | 1000 | |
| **Solver Order** | 2 | Second-order ODE solver |
| **Solver Type** | `bh2` | |
| **Use Flow Sigmas** | true | |

> **Flow shift note:** The Diffusers scheduler config uses `flow_shift: 3.0` with `exponential` shift type, while native WAN training uses `sample_shift: 12.0` (T2V) or `5.0` (I2V). The difference is due to different parameterizations — Diffusers applies the shift within the scheduler, while native applies it during timestep sampling.

### Solver Implementations (Native WAN)

The native WAN repo includes two ODE solvers for flow matching, both in `wan/utils/`:

| Solver | File | Key Feature |
|--------|------|-------------|
| **FlowDPMSolverMultistepScheduler** | `fm_solvers.py` | DPM++ adapted for flow matching (default) |
| **FlowUniPCMultistepScheduler** | `fm_solvers_unipc.py` | UniPC adapted for flow matching (Diffusers default) |

Both solvers adapt their standard diffusion counterparts for the flow matching formulation:
- **Prediction type:** `flow_prediction` (velocity prediction, not noise or sample)
- **Sigma schedule:** Linear schedule from the flow matching formulation, modified by `shift` parameter
- **Shift application:** `sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)` — this maps the uniform [0,1] schedule to concentrate more steps in the high-noise region

```python
# Native solver shift application (from fm_solvers.py)
# The shift parameter controls how timesteps are distributed:
#   shift=1.0  → uniform spacing
#   shift=12.0 → heavily front-loaded (more steps at high noise, T2V default)
#   shift=5.0  → moderately front-loaded (I2V default)
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

### Guidance Scale Mapping

The native WAN config uses a tuple `sample_guide_scale: (low_noise, high_noise)`:

| Task | Native Format | Diffusers Equivalent |
|------|---------------|---------------------|
| **T2V** | `(3.0, 4.0)` = (low_noise=3.0, high_noise=4.0) | `guidance_scale=4.0` (high-noise), `guidance_scale_2=3.0` (low-noise) |
| **I2V** | `(3.5, 3.5)` = (low_noise=3.5, high_noise=3.5) | `guidance_scale=3.5`, `guidance_scale_2=3.5` |

> **Mapping convention:** In Diffusers, `transformer` is the primary (high-noise expert) and `transformer_2` is the secondary (low-noise expert). `guidance_scale` controls the primary, `guidance_scale_2` controls the secondary.

### Official Inference Parameters (per Task)

| Parameter | T2V-A14B | I2V-A14B | TI2V-5B | S2V-A14B |
|-----------|----------|----------|---------|----------|
| **Steps** | 40 | 40 | 40 | 20 |
| **Shift** | 12.0 | 5.0 | 5.0 | 3.0 |
| **Guidance (high/low)** | (4.0, 3.0) | (3.5, 3.5) | (2.5, 2.5) | 4.5 (single) |
| **Boundary** | 0.875 | 0.900 | N/A (dense) | 0.875 |
| **FPS** | 16 | 16 | 24 | 30 |
| **Frames** | 81 | 81 | 121 | 77 |
| **Solver** | DPM++ / UniPC | DPM++ / UniPC | DPM++ / UniPC | DPM++ |
| **VAE** | Wan2.1 | Wan2.1 | Wan2.2 | Wan2.1 |

> **Guidance convention:** Native WAN uses tuple `(low_noise, high_noise)` ordering. Diffusers reverses this: `guidance_scale` = high-noise expert, `guidance_scale_2` = low-noise expert. For TI2V-5B (dense, no MoE), only a single guidance value is needed.

### Training Stages (from Technical Report)

| Stage | Resolution | Content | Duration |
|-------|------------|---------|----------|
| **1. Image Pre-training** | 256px | Text-to-Image | - |
| **2. Joint Training 1** | 256px images + 192px video | 5s clips @ 16fps | - |
| **3. Joint Training 2** | 480px | Images + 5s videos | - |
| **4. Joint Training 3** | 720px | Images + 5s videos | - |
| **5. Post-training** | 480px + 720px | Curated high-quality data | - |

### Negative Prompt (Default)

```
色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，
最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，
画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，
杂乱的背景，三条腿，背景人很多，倒着走
```

Translation: Vivid colors, overexposure, static, blurry details, subtitles, style, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, still frame, cluttered background, three legs, many people in background, walking backwards.

---

## Memory & Performance

### GPU Requirements

#### Official VRAM Benchmarks (from WAN 2.2 README)

| Model | Task | Resolution | Frames | VRAM (Single GPU) | Notes |
|-------|------|-----------|--------|-------------------|-------|
| **A14B** | T2V | 720P | 81 | ~44 GB | With `--offload_model` |
| **A14B** | T2V | 480P | 81 | ~30 GB | With `--offload_model` |
| **A14B** | I2V | 720P | 81 | ~44 GB | With `--offload_model` |
| **TI2V-5B** | I2V | 480P | 121 | ~12 GB | Consumer GPU friendly |
| **1.3B** | T2V | 480P | 81 | ~8.2 GB | Consumer-grade |

| Config | VRAM | Notes |
|--------|------|-------|
| **14B Multi-GPU** | 8× 80GB | FSDP + Ulysses (no offloading needed) |
| **14B + offload** | 24 GB | Heavy offloading (`--offload_model True --t5_cpu`), significantly slower |
| **14B + FP8** | ~22 GB | `--convert_model_dtype` reduces precision |
| **5B (TI2V)** | 12-24 GB | Single 4090 supported |
| **1.3B** | ~8.2 GB | Consumer-grade, no offloading needed |

### Optimization Flags

```bash
# Memory optimization
--offload_model True           # Offload model to CPU
--convert_model_dtype          # Convert to param_dtype (bf16)
--t5_cpu                       # Keep T5 on CPU

# Multi-GPU (FSDP + Ulysses)
--dit_fsdp                     # Enable FSDP for DiT
--t5_fsdp                      # Enable FSDP for T5
--ulysses_size 8               # Sequence parallel size
```

### Inference Acceleration

| Technique | Speedup | Notes |
|-----------|---------|-------|
| **Diffusion Cache** | 1.62× | Attention + CFG caching |
| **FP8 GEMM** | 1.13× | DiT linear layers |
| **8-bit FlashAttention** | 1.27× | INT8 QK, FP8 PV |
| **Multi-GPU Scaling** | ~Linear | Up to 8 GPUs |

---

## Supported Resolutions

### SIZE_CONFIGS (from `wan/configs/__init__.py`)

All supported resolution presets:

| Config Key | Width × Height | Aspect Ratio |
|------------|---------------|--------------|
| `720*1280` | 1280 × 720 | 16:9 landscape |
| `1280*720` | 720 × 1280 | 9:16 portrait |
| `960*960` | 960 × 960 | 1:1 square |
| `480*832` | 832 × 480 | ~16:9 landscape |
| `832*480` | 480 × 832 | ~9:16 portrait |
| `624*624` | 624 × 624 | 1:1 square |
| `480*848` | 848 × 480 | ~16:9 landscape (480P) |
| `848*480` | 480 × 848 | ~9:16 portrait (480P) |

### SUPPORTED_SIZES per Task

| Task | Supported Sizes | Default Frames |
|------|----------------|----------------|
| **t2v-A14B** | 720×1280, 1280×720, 960×960, 480×832, 832×480, 624×624 | 81 (5s @ 16fps) |
| **i2v-A14B** | 720×1280, 1280×720, 960×960, 480×832, 832×480, 624×624 | 81 |
| **ti2v-5B** | 480×848, 848×480, 624×624 | 121 (5s @ 24fps) |
| **s2v-A14B** | 480×832, 832×480, 624×624 | 77 (~2.6s @ 30fps) |
| **animate-A14B** | 480×832, 832×480, 624×624 | 77 |

> **Note:** A14B tasks support 720P resolutions while 5B/S2V/Animate tasks are limited to 480P. TI2V uses slightly different resolution presets (`480*848`/`848*480`) than the others (`480*832`/`832*480`).

### Frame Calculation

```python
# VAE temporal compression = 4x
# First-frame special: T frames -> (1 + (T-1)/4) latent frames (for T = 4k+1)
# Example: 81 frames -> 1 + 80/4 = 21 latent frames

# Sequence length calculation (matches training code)
latent_frames = 1 + (frames - 1) // 4  # e.g., 81 -> 21
lat_h = height // 8                   # VAE latent grid
lat_w = width // 8
seq_len = latent_frames * lat_h * lat_w // 4  # patch_size=(1,2,2) reduces spatial tokens by 2x2
```

---

## Official Prompting Guidelines

### Basic Formula (Beginners)

```
Prompt = Subject + Scene + Motion
```

**Example**: "A cat playing with a ball in a garden"

### Standard Formula

```
Prompt = Subject Description + Scene Description + Motion Description
         + Aesthetic Control + Stylization
```

**Aesthetic Control** includes:
- Light source and lighting environment
- Shot size (framing): close-up, medium shot, wide shot
- Camera angle: eye-level, high angle, low angle
- Lens type: wide-angle, telephoto, macro
- Camera movement: dolly, pan, tilt, tracking

**Example**: "A young woman with long flowing hair walks through a neon-lit cyberpunk city street at night. She turns to look at the camera with a mysterious smile. Medium shot, tracking camera following from behind, warm orange neon lighting contrasting with cool blue shadows, cinematic film grain, Blade Runner aesthetic"

### Image-to-Video Formula

```
Prompt = Motion Description + Camera Movement
```

Since the source image establishes subject, scene, and style, focus on:
- What should move and how
- Camera movement (or "static shot" / "fixed shot" for stationary)

**Example**: "The woman turns her head slowly to the right while her hair flows in the wind. Subtle dolly in."

### Multi-Shot Formula

```
Prompt = Overall Description + [Shot 1: Timestamp + Subject Behavior]
         + [Shot 2: Timestamp + Subject Behavior] + ...
```

**Example**:
```
A dramatic scene of a warrior facing a dragon.
[0:00-0:03] Wide shot establishing the battlefield, warrior stands ready
[0:03-0:06] Close-up on warrior's determined face
[0:06-0:10] The dragon breathes fire, warrior raises shield
```

### Cinematic System Prompts (Official)

The native WAN repo includes 8 system prompts (`wan/utils/system_prompt.py`) used for automatic prompt enhancement via LLM rewriting. These prompts instruct an LLM to transform simple user prompts into detailed cinematic descriptions:

| Prompt | Task | Key Instructions |
|--------|------|-----------------|
| `t2v_sys_prompt` | T2V | "Rewrite user input to 150-word cinematic scene description with subject, motion, environment, atmosphere, lighting" |
| `t2v_sys_prompt_zh` | T2V (Chinese) | Same as above, in Chinese |
| `i2v_sys_prompt` | I2V | "Describe motion and camera movement for the given image" |
| `i2v_sys_prompt_zh` | I2V (Chinese) | Same as above, in Chinese |
| `ti2v_sys_prompt` | TI2V | "For image + text, describe cinematic transition" |
| `ti2v_sys_prompt_zh` | TI2V (Chinese) | Same as above, in Chinese |
| `s2v_sys_prompt` | S2V | "Describe subject animation and lip sync" |
| `s2v_sys_prompt_zh` | S2V (Chinese) | Same as above, in Chinese |

> **Training relevance:** The official models were fine-tuned on LLM-enhanced prompts following these templates. For best results during inference, prompts should include cinematic detail (lighting, camera angles, motion descriptions). During training, matching this prompt style in captions may improve alignment with the model's learned distribution.

### Camera Movement Terms

| Term | Description |
|------|-------------|
| **Dolly in/out** | Camera moves toward/away from subject |
| **Pan left/right** | Camera rotates horizontally |
| **Tilt up/down** | Camera rotates vertically |
| **Tracking shot** | Camera follows moving subject |
| **Aerial shot** | Camera from above, often moving |
| **Static shot** | Camera remains stationary |
| **Handheld** | Slight camera shake for realism |

### Shot Types

| Term | Description |
|------|-------------|
| **Extreme close-up (ECU)** | Very tight on face/detail |
| **Close-up (CU)** | Face fills frame |
| **Medium close-up (MCU)** | Head and shoulders |
| **Medium shot (MS)** | Waist up |
| **Full shot (FS)** | Entire body |
| **Wide shot (WS)** | Subject + environment |
| **Extreme wide shot (EWS)** | Vast landscape, subject small |

---

## Benchmark Performance

### Wan-Bench Results

| Metric | Wan 14B | Wan 1.3B | Sora | HunyuanVideo |
|--------|---------|----------|------|--------------|
| **Large Motion** | 0.415 | 0.468 | 0.482 | 0.413 |
| **Physical Plausibility** | 0.939 | 0.912 | 0.933 | 0.898 |
| **Smoothness** | 0.910 | 0.790 | 0.930 | 0.890 |
| **Image Quality** | 0.640 | 0.596 | 0.665 | 0.605 |
| **ID Consistency** | 0.946 | 0.938 | 0.925 | 0.935 |
| **Weighted Score** | **0.724** | 0.689 | 0.700 | 0.673 |

### VBench Results

| Model | Quality Score | Semantic Score | Total |
|-------|--------------|----------------|-------|
| **Wan 14B** | **86.67%** | **84.44%** | **86.22%** |
| Wan 1.3B | 84.92% | 80.10% | 83.96% |
| Sora | 85.51% | 79.35% | 84.28% |
| HunyuanVideo | 85.09% | 75.82% | 83.24% |

---

## Native Denoising Loop

The official WAN denoising loop (from `wan/text2video.py` and `wan/image2video.py`) illustrates the boundary-switching logic:

```python
# Simplified native denoising loop (T2V)
for i, t in enumerate(timesteps):  # timesteps: high → low
    t_norm = t / num_train_timesteps  # Normalize to [0, 1]

    # Select expert based on boundary
    if t_norm >= boundary:
        model = high_noise_model
        guide_scale = sample_guide_scale[1]  # high-noise guidance
    else:
        model = low_noise_model
        guide_scale = sample_guide_scale[0]  # low-noise guidance

    # Classifier-free guidance
    noise_pred_uncond = model(x_t, t, context=null_context)
    noise_pred_cond = model(x_t, t, context=text_context)
    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

    # Solver step
    x_t = solver.step(noise_pred, t, x_t)
```

```python
# I2V conditioning (A14B, WAN 2.2)
# Reference implementation: src/musubi_tuner/wan_cache_latents.py (encode_and_save_batch)
#
# Goal: build the extra 20 channels passed as `y`:
#   4-channel temporal mask + 16-channel image latent  → 20ch conditioning
# Then at forward time the model sees:
#   [noisy_latent(16ch) | conditioning(20ch)] = 36 input channels.

# 1. Construct temporal mask directly in latent space:
#    first latent frame = 1 (known image), rest = 0 (to generate).
#    The mask has 4 channels because WAN uses a 4× temporal VAE scale factor.
msk = torch.zeros(B, 4, lat_f, lat_h, lat_w, dtype=vae_dtype, device=device)
msk[:, :, 0] = 1

# 2. Pad the reference image with zeros to full pixel-time length, then VAE-encode as a "video".
images_padded = torch.cat(
    [reference_image, torch.zeros(B, 3, num_pixel_frames - 1, H, W, dtype=vae_dtype, device=device)],
    dim=2,
)
image_latent = vae.encode(images_padded)  # [B, 16, lat_f, H/8, W/8]

# 3. Concatenate mask + image latent → 20-channel I2V conditioning tensor.
image_cond = torch.cat([msk, image_latent], dim=1)  # [B, 20, lat_f, H/8, W/8]

# 4. At forward time: [noisy_latent(16ch) | image_cond(20ch)] = 36 channels.
model_input = torch.cat([noisy_latent, image_cond], dim=1)
```

---

## Blissful Tuner Integration

### Training Command

```bash
# Cache latents
python wan_cache_latents.py --dataset_config config.toml \
    --vae /path/to/Wan2.1_VAE.pth --vae_chunk_size 32 --vae_tiling

# Cache text encoder outputs
python wan_cache_text_encoder_outputs.py --dataset_config config.toml \
    --t5 /path/to/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16

# Train LoRA (WAN 2.2)
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    wan_train_network.py \
    --task t2v-A14B \
    --dit /path/to/low_noise_model.safetensors \
    --dit_high_noise /path/to/high_noise_model.safetensors \
    --dataset_config config.toml \
    --network_module networks.lora_wan \
    --network_dim 32 \
    --timestep_sampling shift \
    --discrete_flow_shift 12.0
```

### Key Training Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| **network_dim** | 32-64 | LoRA rank |
| **timestep_sampling** | shift | Flow matching |
| **discrete_flow_shift** | 12.0 (T2V), 5.0 (I2V) | Task-specific |
| **blocks_to_swap** | 0-39 | Memory optimization |
| **gradient_checkpointing** | True | Recommended |

### Memory Optimizations

Two memory optimizations specific to the Blissful Tuner WAN 2.2 codepaths, layered from least to most aggressive:

| Optimization | Flag (to disable / opt out) | Default | Numerical effect |
|---|---|---|---|
| **Compact time embedding** | `--no_compact_time_embedding` | enabled | None (exact-equal under broadcast, `atol=0`) |
| **Wan2.1-style modulation** | `--force_v2_1_time_embedding` (alias of `--simple_modulation`) | disabled | Different modulation shape — comparable in practice but not bit-identical |

#### Compact Time Embedding

When the timestep is uniform across all tokens (`t.dim() == 1`, the T2V training case), `WanModel.get_time_embedding()` keeps the time embedding at `[B, 1, dim]` and the modulation projection at `[B, 1, 6, dim]` instead of expanding to `[B, seq_len, dim]` / `[B, seq_len, 6, dim]`. Broadcasting in `WanAttentionBlock.get_modulation()` and `Head.forward()` produces numerically identical outputs — the two paths differ only in allocated tensor size.

For WAN 2.2 14B at fp32, the savings are multi-GiB per forward pass (scaling with `seq_len`, which scales with resolution × frames). Default enabled. The per-token timestep path used by I2V/TI2V via `expand_timesteps` produces `t.dim() == 2` and bypasses the compact branch entirely, so the gating is transparent across task families.

`WanModel.compact_time_embedding` is exposed as an instance attribute initialized to `True` in `__init__`. Inference scripts run with compact mode unconditionally — no CLI opt-out is wired into `wan_generate_video.py` since the two modes are numerically equivalent and the smaller allocations are pure upside.

Locked down by `tests/test_wan_compact_time_embedding.py`:
- `[B, 1, dim]` / `[B, 1, 6, dim]` shape contract when `t.dim() == 1`
- Compact vs full path equivalence under broadcast (`rtol=0, atol=0`)
- I2V gate: `t.dim() == 2` bypasses compact even when the attribute is `True`

#### RoPE Frequency Cache (FIFO eviction)

`WanModel.freqs_fhw` caches the computed RoPE frequency tensor for each unique `(F, H', W')` grid-size tuple seen during patch embedding. Without a cap this dict grows unboundedly across long multi-bucket training runs (one entry per resolution × frame-count combination, plus any `f_indices` variation expanding the keyspace).

The cache is capped at `WanModel._FREQS_CACHE_MAX_SIZE = 512` with oldest-first FIFO eviction. A one-shot `logger.warning` fires on first eviction (sets `self._freqs_eviction_warned = True`) — useful as a diagnostic for "is my bucket diversity unusually high or has `f_indices` exploded the keyspace?". Subsequent evictions stay silent to avoid log spam.

Locked down by `tests/test_wan_compact_time_embedding.py::TestWanRoPEFreqsCacheEviction`: inserting `MAX_SIZE + 1` distinct grid-size keys (with `_FREQS_CACHE_MAX_SIZE` monkey-patched down to 3 for test speed) caps the dict, evicts the first-inserted key, and sets `_freqs_eviction_warned`.

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **Wan 2.1** | Mar 2025 | Initial release, T2V + I2V (1.3B + 14B dense), CLIP I2V conditioning |
| **Wan 2.2** | Jul 28, 2025 | Dual-expert architecture (A14B), TI2V-5B with Wan2.2-VAE (4×16×16), per-token time embedding (I2V), latent I2V conditioning (no CLIP), S2V/Animate tasks, cinematic prompt enhancement |

---

## Diffusers Pipeline Structure

The HuggingFace Diffusers checkpoints organize the model into subdirectories:

### T2V-A14B (`WanPipeline`)

```
Wan2.2-T2V-A14B-Diffusers/
├── model_index.json           # Pipeline class, boundary_ratio: 0.875
├── transformer/               # High-noise expert (WanTransformer3DModel)
│   ├── config.json            # in_channels: 16, 40 layers, 40 heads
│   └── diffusion_pytorch_model.safetensors.index.json  # 12 shards, ~57.15 GB
├── transformer_2/             # Low-noise expert (same architecture)
│   ├── config.json            # Identical to transformer/
│   └── diffusion_pytorch_model.safetensors.index.json
├── text_encoder/              # UMT5EncoderModel
│   ├── config.json            # d_model: 4096, 24 layers, 64 heads
│   └── model.safetensors.index.json
├── tokenizer/                 # T5TokenizerFast (SentencePiece)
├── scheduler/                 # UniPCMultistepScheduler
│   └── scheduler_config.json  # flow_shift: 3.0, prediction_type: flow_prediction
└── vae/                       # AutoencoderKLWan
```

### I2V-A14B (`WanImageToVideoPipeline`)

```
Wan2.2-I2V-A14B-Diffusers/
├── model_index.json           # boundary_ratio: 0.9, image_encoder: [null, null]
├── transformer/               # in_channels: 36 (16 latent + 4 mask + 16 image)
├── transformer_2/             # in_channels: 36 (same expanded input)
├── text_encoder/              # Same UMT5EncoderModel
├── tokenizer/                 # Same T5TokenizerFast
├── scheduler/                 # Same UniPCMultistepScheduler
└── vae/                       # Same AutoencoderKLWan
```

> **Key structural notes:**
> - `transformer` = high-noise expert (primary), `transformer_2` = low-noise expert (secondary)
> - Both T2V and I2V use `WanTransformer3DModel` — the only config difference is `in_channels` (16 vs 36)
> - No `image_encoder` or `image_processor` in I2V — image conditioning is purely through latent concatenation
> - Both experts share identical architecture (layers, heads, FFN dim) — they differ only in trained weights

### Diffusers Pipeline Variants

| Pipeline Class | Task | Key Feature |
|---------------|------|-------------|
| `WanPipeline` | T2V | Dual-expert boundary switching, `expand_timesteps` support |
| `WanImageToVideoPipeline` | I2V | 36-channel input with latent conditioning, image attention |
| `WanVideoToVideoPipeline` | V2V | Strength-based noise injection on input video, timestep trimming |

The V2V pipeline (`pipeline_wan_video2video.py`) adds a `strength` parameter (0.0–1.0) that controls how much the input video is noised before denoising. Higher strength = more noise = more creative freedom but less fidelity to input.

---

## References

- [Wan 2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Wan 2.2 HuggingFace](https://huggingface.co/Wan-AI/)
- [Technical Report: arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- [Wan-Bench Evaluation](https://wan.video)
