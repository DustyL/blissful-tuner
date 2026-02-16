# Z-Image Integration Reference

**Document Version**: 1.0
**Date**: 2026-01-27
**Sources**: Official Z-Image GitHub, Technical Report, HuggingFace Repository, sdbds/musubi-tuner fork

This document provides comprehensive reference material for integrating and improving Z-Image support in Blissful Tuner.

---

## Table of Contents

1. [Model Variants Overview](#1-model-variants-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Configuration Reference](#3-configuration-reference)
4. [Training Methodology](#4-training-methodology)
5. [LoRA Fine-Tuning Guidance](#5-lora-fine-tuning-guidance)
6. [OmniBase Integration (Image Editing)](#6-omnibase-integration-image-editing)
7. [Implementation Comparison](#7-implementation-comparison)
8. [Integration Recommendations](#8-integration-recommendations)
9. [Performance Benchmarks](#9-performance-benchmarks)

---

## 1. Model Variants Overview

Z-Image is a family of models with four variants:

| Variant | Parameters | NFEs | CFG | Use Case |
|---------|------------|------|-----|----------|
| **Z-Image** | 6.15B | 50-100 | Yes (3.0-5.0) | High-quality generation |
| **Z-Image-Turbo** | 6.15B | 8 | No (0.0) | Fast inference, sub-second |
| **Z-Image-Omni-Base** | 6.15B | Variable | Yes | Generation + editing |
| **Z-Image-Edit** | 6.15B | Variable | Yes | Dedicated editing |

### Key Characteristics

- **Single-stream MM-DiT** (Multi-Modal Diffusion Transformer)
- **Bilingual**: Native Chinese + English support via Qwen3-4B encoder
- **Arbitrary resolution**: 512×512 to 2048×2048
- **Flow matching**: Velocity prediction with Euler discrete scheduler
- **Ranked #1** open-source on Artificial Analysis T2I leaderboard

---

## 2. Architecture Deep Dive

### 2.1 Core Architecture: S3-DiT (Scalable Single-Stream DiT)

```
Input: [Text Tokens | Visual Semantic Tokens | VAE Image Tokens]
                            ↓
                  Unified Sequence Input
                            ↓
              ┌─────────────────────────────┐
              │    Noise Refiner (2 layers) │  ← Modulated by timestep
              └─────────────────────────────┘
                            ↓
              ┌─────────────────────────────┐
              │  Context Refiner (2 layers) │  ← No modulation (clean)
              └─────────────────────────────┘
                            ↓
              ┌─────────────────────────────┐
              │  Main Transformer (30 layers)│  ← Full self-attention
              └─────────────────────────────┘
                            ↓
              ┌─────────────────────────────┐
              │       Final Layer           │  ← Modulated projection
              └─────────────────────────────┘
                            ↓
                    Output Latents
```

### 2.2 Transformer Block Structure

Each `ZImageTransformerBlock` contains:

```python
class ZImageTransformerBlock:
    # Pre-attention normalization
    attention_norm1: RMSNorm
    attention_norm2: RMSNorm  # Sandwich norm

    # Attention with QK normalization
    attention: ZImageAttention
        - to_q, to_k, to_v: Linear(dim, dim, bias=False)
        - norm_q, norm_k: RMSNorm  # QK-Norm for stability
        - to_out: Linear(dim, dim, bias=False)

    # Pre-FFN normalization
    ffn_norm1: RMSNorm
    ffn_norm2: RMSNorm  # Sandwich norm

    # SwiGLU FFN
    feed_forward: FeedForward
        - w1: Linear(dim, hidden, bias=False)  # Gate
        - w2: Linear(hidden, dim, bias=False)  # Down
        - w3: Linear(dim, hidden, bias=False)  # Up

    # AdaLN modulation (scale + gate)
    adaLN_modulation: Linear(adaln_dim, 6 * dim)
```

### 2.3 Key Architectural Features

#### RMSNorm Everywhere
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

#### QK-Norm (Attention Stability)
```python
# Applied BEFORE RoPE
q = self.norm_q(self.to_q(x))
k = self.norm_k(self.to_k(x))
# Then apply RoPE
q = apply_rotary_emb(q, freqs)
k = apply_rotary_emb(k, freqs)
```

#### 3D Unified RoPE
```python
# Dimensions per axis
ROPE_AXES_DIMS = [32, 48, 48]  # [time, height, width]
ROPE_AXES_LENS = [1536, 512, 512]  # Max positions
ROPE_THETA = 256.0

# Text tokens: increment along temporal dimension only
# Image tokens: expand across spatial (H, W) dimensions
# Reference vs target: same spatial RoPE, different temporal offset
```

#### AdaLN Modulation
```python
# Timestep → embedding → 6 modulation values per block
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6)

# Applied as:
x = norm(x) * (1 + scale) + shift  # Before attention/FFN
x = x * gate  # After attention/FFN (residual gating)
```

### 2.4 Patch Embedding

```python
# Configuration
all_patch_size = [2]      # 2×2 spatial patches
all_f_patch_size = [1]    # 1 temporal patch
in_channels = 16          # VAE latent channels

# Embedding
x_embedder = nn.Linear(in_channels * patch_h * patch_w, dim)
# For 2×2 patches: Linear(16 * 2 * 2, 3840) = Linear(64, 3840)
```

---

## 3. Configuration Reference

### 3.1 Transformer Configuration

```python
# From HuggingFace config.json
DEFAULT_TRANSFORMER_CONFIG = {
    "dim": 3840,                    # Hidden dimension
    "in_channels": 16,              # VAE latent channels
    "n_layers": 30,                 # Main transformer layers
    "n_heads": 30,                  # Attention heads
    "n_kv_heads": 30,               # KV heads (no GQA in transformer)
    "n_refiner_layers": 2,          # Refiner layers (noise + context)
    "norm_eps": 1e-5,               # RMSNorm epsilon
    "qk_norm": True,                # Enable QK normalization
    "cap_feat_dim": 2560,           # Text embedding dimension (from Qwen3)
    "siglip_feat_dim": None,        # SigLIP dimension (OmniBase only: 1152)
    "t_scale": 1000.0,              # Timestep scaling
    "rope_theta": 256.0,            # RoPE base frequency
    "axes_dims": [32, 48, 48],      # RoPE dimensions [T, H, W]
    "axes_lens": [1536, 512, 512],  # Max RoPE positions
    "all_patch_size": [2],          # Spatial patch size
    "all_f_patch_size": [1],        # Temporal patch size
}
```

### 3.2 Text Encoder Configuration (Qwen3-4B)

```python
DEFAULT_TEXT_ENCODER_CONFIG = {
    "hidden_size": 2560,            # Output embedding dimension
    "num_hidden_layers": 36,        # Transformer layers
    "num_attention_heads": 32,      # Query heads
    "num_key_value_heads": 8,       # KV heads (GQA: 4× compression)
    "head_dim": 128,                # Per-head dimension
    "intermediate_size": 9728,      # FFN hidden dimension
    "hidden_act": "silu",           # Activation function
    "vocab_size": 151936,           # Vocabulary size
    "max_position_embeddings": 40960,
    "rope_theta": 1000000,          # Different from transformer!
    "rms_norm_eps": 1e-6,
}
```

### 3.3 VAE Configuration (Flux VAE)

```python
DEFAULT_VAE_CONFIG = {
    "in_channels": 3,               # RGB input
    "out_channels": 3,              # RGB output
    "latent_channels": 16,          # Latent space channels
    "scaling_factor": 0.3611,       # Latent scaling
    "shift_factor": 0.1159,         # Latent shift
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "act_fn": "silu",
}
```

### 3.4 Scheduler Configuration

```python
DEFAULT_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "shift": 6.0,                   # Flow matching shift (Z-Image standard)
    # For Z-Image-Turbo: different shift optimized during distillation
    "use_dynamic_shifting": False,
}
```

### 3.5 Inference Parameters

| Model | Steps | CFG Scale | Shift | Resolution |
|-------|-------|-----------|-------|------------|
| Z-Image | 28-50 | 3.0-5.0 | 6.0 | 512-2048 |
| Z-Image-Turbo | 8 | 0.0 | optimized | 512-2048 |

---

## 4. Training Methodology

### 4.1 Flow Matching Objective

```python
# Velocity prediction loss
def compute_loss(model, x0, x1, t, condition):
    # Linear interpolation
    x_t = t * x1 + (1 - t) * x0

    # Target velocity
    v_target = x1 - x0

    # Model prediction
    v_pred = model(x_t, t, condition)

    # MSE loss
    loss = F.mse_loss(v_pred, v_target)
    return loss
```

### 4.2 Timestep Sampling

```python
# Logit-normal sampling (concentrates on intermediate timesteps)
def sample_timestep(batch_size):
    u = torch.randn(batch_size)
    t = torch.sigmoid(u)  # Maps to (0, 1)
    return t

# Dynamic time shifting for resolution
def compute_shift(image_seq_len):
    BASE_SEQ_LEN = 256
    MAX_SEQ_LEN = 4096
    BASE_SHIFT = 0.5
    MAX_SHIFT = 1.15

    mu = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN) * image_seq_len + BASE_SHIFT
    return mu
```

### 4.3 Three-Phase Training Curriculum

**Phase 1: Low-Resolution Pre-Training**
- Resolution: 256×256 (fixed)
- Task: Text-to-image only
- Budget: >50% of total compute
- Objective: Establish cross-modal alignment

**Phase 2: Omni-Pre-Training**
- Arbitrary resolution training
- Joint T2I + I2I tasks
- Multi-level bilingual captions
- Budget: ~45% of compute

**Phase 3: Supervised Fine-Tuning**
- High-quality curated data
- Concept balancing via BM25
- Model merging for robustness
- Budget: ~5% of compute

### 4.4 Few-Step Distillation (Turbo)

**Decoupled DMD** separates:
1. **CFG-Augmentation (CA)**: Builds few-step capability
2. **Distribution Matching (DM)**: Ensures stability

**DMDR** adds RL with DM as regularizer:
- Prevents reward hacking
- Enables preference alignment
- Results: 8 NFEs matching 100-step teacher

---

## 5. LoRA Fine-Tuning Guidance

### 5.1 Recommended Target Modules

```python
# Primary targets (highest impact)
ZIMAGE_TARGET_REPLACE_MODULES = [
    "ZImageTransformerBlock",  # All 30 main blocks
]

# Specifically target within blocks:
target_patterns = [
    r"layers\.\d+\.attention\.to_q",
    r"layers\.\d+\.attention\.to_k",
    r"layers\.\d+\.attention\.to_v",
    r"layers\.\d+\.attention\.to_out\.0",
    r"layers\.\d+\.feed_forward\.w1",
    r"layers\.\d+\.feed_forward\.w2",
    r"layers\.\d+\.feed_forward\.w3",
]

# Exclude from LoRA (keep frozen)
exclude_patterns = [
    r".*_modulation.*",      # AdaLN modulation
    r".*_refiner.*",         # Refiner layers
    r".*norm.*",             # All normalization layers
    r".*embedder.*",         # Input embedders
]
```

### 5.2 Hyperparameter Recommendations

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| Rank (r) | 16 | 32 | 64 |
| Alpha (α) | 16 | 32-64 | 64-128 |
| Learning Rate | 1e-5 | 5e-5 | 1e-4 |
| Batch Size | 1-2 | 2-4 | 4-8 |
| Epochs | 10-20 | 5-10 | 3-5 |

### 5.3 Training Configuration Template

```toml
# Dataset configuration
[[datasets]]
resolution = [1024, 1024]
batch_size = 2
enable_bucket = true
bucket_no_upscale = true

[[datasets.subsets]]
image_directory = "/path/to/images"
caption_extension = ".txt"

# Training arguments
network_module = "networks.lora_zimage"
network_dim = 32
network_alpha = 32

learning_rate = 5e-5
lr_scheduler = "cosine"
lr_warmup_steps = 100

max_train_epochs = 10
save_every_n_epochs = 2

# Z-Image specific
timestep_sampling = "shift"  # or "logit_normal"
discrete_flow_shift = 6.0
```

### 5.4 Caption Recommendations (From Technical Report)

For best results, use multi-level captions:

```
# Long description (detailed)
A serene mountain landscape at sunset with golden light reflecting off
a pristine alpine lake, surrounded by pine forests and snow-capped peaks...

# Medium description
Mountain lake at sunset with golden reflections and pine forests.

# Short description
Sunset mountain lake scene.

# Tags
landscape, mountain, lake, sunset, nature, alpine, golden hour, reflection
```

---

## 6. OmniBase Integration (Image Editing)

### 6.1 Architecture Extensions

OmniBase adds these components to the base Z-Image:

```python
class ZImageTransformer2DModel:
    # NEW: SigLIP feature processing
    siglip_feat_dim: int = 1152  # SigLIP2 hidden size
    siglip_embedder: nn.Sequential  # RMSNorm + Linear(1152, 3840)
    siglip_refiner: nn.ModuleList   # 2 transformer blocks
    siglip_pad_token: nn.Parameter  # Learnable padding

    # NEW: Per-token noise selection
    def select_per_token(noisy, clean, noise_mask, seq_len):
        """Select between noisy and clean embeddings per token."""
        mask = noise_mask.unsqueeze(-1).expand(-1, seq_len, -1)
        return torch.where(mask == 1, noisy, clean)
```

### 6.2 Training Data Construction

**Dual-Branch Processing:**
- Reference images: `noise_mask = 0` (clean modulation)
- Target images: `noise_mask = 1` (noisy modulation)

**Cache Format:**
```python
# Additional keys for OmniBase
cache_keys = {
    "latents_{F}x{H}x{W}_{dtype}": target_latents,
    "latents_control_{i}_{F}x{H}x{W}_{dtype}": reference_latents,
    "siglip_{i}_{dtype}": siglip_features,  # [H_sig, W_sig, 1152]
}
```

### 6.3 SigLIP2 Integration

```python
# Loading SigLIP2 (from transformers)
try:
    from transformers import Siglip2VisionModel, Siglip2Processor
    SIGLIP2_AVAILABLE = True
except ImportError:
    SIGLIP2_AVAILABLE = False

# SigLIP2 configuration
SIGLIP2_CONFIG = {
    "hidden_size": 1152,
    "intermediate_size": 4608,
    "num_hidden_layers": 27,
    "num_attention_heads": 18,
    "image_size": 256,
    "patch_size": 16,
}

# Converting SigLIP tokens to spatial grid
def siglip_last_hidden_to_grid(last_hidden_state):
    """Convert [num_tokens, C] to [H, W, C] spatial grid."""
    num_tokens = last_hidden_state.shape[0]
    # Handle CLS token if present
    grid_size = int(math.sqrt(num_tokens))
    return last_hidden_state.reshape(grid_size, grid_size, -1)
```

### 6.4 Forward Pass (OmniBase)

```python
def forward_omni(self, x_list, t, cap_feats, siglip_feats, noise_masks):
    """
    Args:
        x_list: List of image tensors (reference + target per sample)
        t: Timesteps
        cap_feats: Caption features
        siglip_feats: List of SigLIP features per reference image
        noise_masks: List of masks (0=clean/reference, 1=noisy/target)
    """
    # 1. Embed all images
    all_embeddings = []
    for i, (imgs, sigs, masks) in enumerate(zip(x_list, siglip_feats, noise_masks)):
        img_emb = self.patchify_and_embed_omni(imgs, sigs)
        all_embeddings.append(img_emb)

    # 2. Process through noise refiner (with modulation)
    t_emb = self.t_embedder(t)
    for layer in self.noise_refiner:
        x = layer(x, t_emb, noise_mask=masks)

    # 3. Process through context refiner (no modulation)
    for layer in self.context_refiner:
        x = layer(x)

    # 4. Main transformer with dual modulation
    for layer in self.layers:
        adaln_noisy = layer.adaLN_modulation(t_emb_noisy)
        adaln_clean = layer.adaLN_modulation(t_emb_clean)
        x = layer(x, noise_mask=masks, adaln_noisy=adaln_noisy, adaln_clean=adaln_clean)

    # 5. Final layer
    return self.final_layer(x, t_emb)
```

---

## 7. Implementation Comparison

### 7.1 Current Blissful-Tuner vs Official Implementation

| Aspect | Blissful-Tuner | Official Z-Image |
|--------|----------------|------------------|
| Transformer layers | 30 | 30 |
| Refiner layers | 2 (noise + context) | 2 (noise + context) |
| Hidden dim | 3840 | 3840 |
| QK-Norm | ✅ Yes | ✅ Yes |
| RoPE 3D | ✅ Yes | ✅ Yes |
| SigLIP support | ❌ No | ✅ OmniBase only |
| Attention backends | PyTorch SDPA | FA2/FA3/SDPA |
| Text encoder | Qwen3 | Qwen3 |
| VAE | Flux VAE | Flux VAE |

### 7.2 sdbds Fork Additions

The sdbds fork adds OmniBase support:

| Component | Lines Changed | Key Additions |
|-----------|---------------|---------------|
| zimage_model.py | +552 | siglip_embedder, siglip_refiner, select_per_token, omni forward |
| zimage_utils.py | +108 | SigLIP2 loading, image encoder utils |
| zimage_cache_latents.py | +120 | I2V caching, --i2v flag, siglip grid conversion |
| zimage_train_network.py | +75 | omni-mode training path, siglip loading |
| zimage_config.py | +1 | SIGLIP_FEAT_DIM = 1152 |

### 7.3 Gaps to Address

1. **SigLIP2 Integration**: Not in current blissful-tuner
2. **OmniBase Forward Path**: Requires model architecture changes
3. **I2V Caching**: Cache script needs control image support
4. **Inference**: OmniBase generation not implemented anywhere
5. **Attention Backends**: Could add FA3 support for performance

---

## 8. Integration Recommendations

### 8.1 Priority Order

1. **Phase 1 (Low Risk)**: Update configuration constants
   - Add `SIGLIP_FEAT_DIM = 1152` to zimage_config.py
   - Verify all default values match official config

2. **Phase 2 (Medium Risk)**: Add SigLIP2 utilities
   - Implement `load_image_encoders()` in zimage_utils.py
   - Add graceful fallback when SigLIP2 unavailable

3. **Phase 3 (High Risk)**: Merge model architecture
   - Add siglip_embedder, siglip_refiner, siglip_pad_token
   - Implement select_per_token() function
   - Add patchify_and_embed_omni() method
   - Modify forward() with omni branch
   - **CRITICAL**: Ensure backward compatibility with existing LoRAs

4. **Phase 4 (Medium Risk)**: Update caching
   - Add --i2v flag to zimage_cache_latents.py
   - Implement siglip feature caching
   - Add control image latent caching

5. **Phase 5 (Medium Risk)**: Update training
   - Load siglip features in training loop
   - Construct multi-image batches
   - Pass noise_masks to model

### 8.2 Backward Compatibility Strategy

```python
class ZImageTransformer2DModel:
    def __init__(
        self,
        siglip_feat_dim: Optional[int] = None,  # None = no OmniBase
        use_default_siglip_feat_dim: bool = False,  # Explicit flag
        **kwargs
    ):
        # Only create SigLIP modules if explicitly enabled
        if siglip_feat_dim is not None or use_default_siglip_feat_dim:
            self.siglip_feat_dim = siglip_feat_dim or 1152
            self.siglip_embedder = ...
            self.siglip_refiner = ...
        else:
            self.siglip_feat_dim = None
            # Modules not created, saves memory
```

### 8.3 BlissfulLogger Integration

When adding new code, use the existing logging pattern:

```python
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")

# Usage
logger.info("Loading SigLIP2 encoder...")
logger.warning("SigLIP2 not available, falling back to T2I only")
```

---

## 9. Performance Benchmarks

### 9.1 Quality Benchmarks

**GenEval (Compositional Generation):**
- Z-Image: 0.84 average (Rank 2)
- Best: Qwen-Image 0.87

**DPG-Bench (Detailed Prompt):**
- Z-Image: 88.14 (Rank 3)
- Z-Image-Turbo: 84.86 (Rank 7)

**PRISM-Bench English:**
- Z-Image-Turbo: 77.4 (Rank 3)

**PRISM-Bench Chinese:**
- Z-Image: 75.3 (Rank 2)
- Text Rendering: 83.4 (exceptional)

### 9.2 Efficiency Benchmarks

| Model | Parameters | Steps | Inference Cost |
|-------|------------|-------|----------------|
| Z-Image-Turbo | 6.15B | 8 | $5.0/1K images |
| Qwen-Image | 20B | ~50 | Higher |
| FLUX.2 | 32B | ~28 | Higher |
| Hunyuan-3.0 | 80B | ~50 | Much higher |

### 9.3 Training Costs (Reference)

| Phase | GPU Hours | Cost (est.) |
|-------|-----------|-------------|
| Low-res pre-training | 147.5K H800 | $295K |
| Omni pre-training | 142.5K H800 | $285K |
| Post-training | 24K H800 | $48K |
| **Total** | **314K H800** | **$628K** |

---

## Appendix A: File Structure Reference

```
Official Z-Image GitHub:
├── src/
│   ├── config/
│   │   ├── model.py           # Architecture constants
│   │   └── inference.py       # Inference defaults
│   ├── zimage/
│   │   ├── transformer.py     # ZImageTransformer2DModel
│   │   ├── pipeline.py        # Generation pipeline
│   │   ├── scheduler.py       # FlowMatchEulerDiscreteScheduler
│   │   └── autoencoder.py     # VAE
│   └── utils/
│       ├── attention.py       # Attention backend dispatch
│       ├── loader.py          # Component loading
│       └── helpers.py         # Utilities
├── inference.py               # Single image generation
└── batch_inference.py         # Batch generation

HuggingFace Model:
├── model_index.json           # Pipeline structure
├── transformer/
│   ├── config.json            # Transformer config
│   └── diffusion_pytorch_model.safetensors.index.json
├── text_encoder/
│   ├── config.json            # Qwen3 config
│   └── model.safetensors.index.json
├── tokenizer/
│   └── tokenizer_config.json  # Qwen2Tokenizer
├── scheduler/
│   └── scheduler_config.json  # FlowMatchEuler
└── vae/
    └── config.json            # Flux VAE

sdbds Fork (OmniBase additions):
├── src/musubi_tuner/zimage/
│   ├── zimage_model.py        # +552 lines (OmniBase)
│   ├── zimage_utils.py        # +108 lines (SigLIP2)
│   └── zimage_config.py       # +1 line
├── src/musubi_tuner/
│   ├── zimage_cache_latents.py    # +120 lines (I2V)
│   └── zimage_train_network.py    # +75 lines (omni training)
```

---

## Appendix B: Key Code Patterns

### B.1 Attention with QK-Norm

```python
def forward(self, x, freqs):
    B, N, C = x.shape

    # Project and normalize
    q = self.norm_q(self.to_q(x))
    k = self.norm_k(self.to_k(x))
    v = self.to_v(x)

    # Reshape for multi-head
    q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

    # Apply RoPE
    q = apply_rotary_emb(q, freqs)
    k = apply_rotary_emb(k, freqs)

    # Attention
    out = F.scaled_dot_product_attention(q, k, v)

    # Output projection
    out = out.transpose(1, 2).reshape(B, N, C)
    return self.to_out(out)
```

### B.2 AdaLN Modulation

```python
def forward(self, x, t_emb, noise_mask=None):
    # Get modulation parameters
    mod = self.adaLN_modulation(t_emb)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

    # Attention block
    x_norm = self.attention_norm1(x)
    x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    x_norm = self.attention_norm2(x_norm)
    attn_out = self.attention(x_norm)
    x = x + gate_msa.unsqueeze(1) * attn_out

    # FFN block
    x_norm = self.ffn_norm1(x)
    x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    x_norm = self.ffn_norm2(x_norm)
    ffn_out = self.feed_forward(x_norm)
    x = x + gate_mlp.unsqueeze(1) * ffn_out

    return x
```

### B.3 OmniBase Per-Token Selection

```python
def select_per_token(noisy_emb, clean_emb, noise_mask, seq_len):
    """
    Select between noisy and clean embeddings per token.

    Args:
        noisy_emb: [B, 6*D] modulation for noisy tokens
        clean_emb: [B, 6*D] modulation for clean tokens
        noise_mask: [B, N] where 1=noisy, 0=clean
        seq_len: Sequence length N

    Returns:
        [B, N, 6*D] selected modulation per token
    """
    mask = noise_mask.unsqueeze(-1)  # [B, N, 1]
    noisy_expanded = noisy_emb.unsqueeze(1).expand(-1, seq_len, -1)
    clean_expanded = clean_emb.unsqueeze(1).expand(-1, seq_len, -1)
    return torch.where(mask == 1, noisy_expanded, clean_expanded)
```

---

*Document generated from analysis of official Z-Image sources for Blissful Tuner integration planning.*
