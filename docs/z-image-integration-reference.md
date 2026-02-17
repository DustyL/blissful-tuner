# Z-Image Integration Reference

**Document Version**: 2.3
**Date**: 2026-02-17
**Sources**: Z-Image Technical Report (arXiv), Official Z-Image GitHub (commit `26f23ed`, 2026-02-09), HuggingFace Repository (`Tongyi-MAI/Z-Image`, `Tongyi-MAI/Z-Image-Turbo`), HuggingFace Diffusers (latest main, 51 Z-Image files), sdbds/musubi-tuner fork, Blissful Tuner implementation audit

This document provides comprehensive reference material for integrating and improving Z-Image support in Blissful Tuner. Version 2.3 incorporates findings from the HuggingFace diffusers repository (transformer, 6 pipelines, ControlNet, LoRA conversion, modular pipelines, scheduler, and tests), building on previous rounds of analysis from the Z-Image technical report, HuggingFace model configs, the official Z-Image GitHub source code, and the Blissful Tuner implementation.

---

## Table of Contents

1. [Model Variants Overview](#1-model-variants-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Configuration Reference](#3-configuration-reference)
4. [Training Methodology](#4-training-methodology)
   - 4.1-4.3: Flow Matching, Timestep Sampling, Three-Phase Curriculum
   - 4.4: Few-Step Distillation (Turbo)
   - 4.5: Post-Training RLHF (DPO + GRPO)
   - 4.6: Prompt Enhancer (PE) System
   - 4.7: Data Pipeline (AIGC Filtering, Dedup, OCR-CoT Captioning)
   - 4.8: Editing Data Construction
   - 4.9: Original Training Infrastructure
5. [LoRA Fine-Tuning Guidance](#5-lora-fine-tuning-guidance)
6. [OmniBase Integration (Image Editing)](#6-omnibase-integration-image-editing)
7. [Diffusers Pipeline Ecosystem](#7-diffusers-pipeline-ecosystem-v23) *(v2.3)*
   - 7.1: Pipeline Variants (6 pipelines)
   - 7.2: ControlNet Architecture
   - 7.3: Modular Pipeline System
   - 7.4: Flow Matching Convention (Diffusers)
   - 7.5: torch.compile Considerations
8. [Implementation Comparison (Three-Way)](#8-implementation-comparison-three-way)
9. [Integration Recommendations](#9-integration-recommendations)
10. [Performance Benchmarks](#10-performance-benchmarks)
    - 10.1: Generation Quality (GenEval, DPG-Bench, TIIF, PRISM-Bench)
    - 10.2: Image Editing (ImgEdit, GEdit-Bench)
    - 10.3: Efficiency & Competitive Landscape
    - 10.4: Training Costs
11. [Weight Storage & ComfyUI Compatibility](#11-weight-storage--comfyui-compatibility)

---

## 1. Model Variants Overview

Z-Image is a family of models with four variants:

| Variant | DiT Params | Steps (NFEs) | CFG | Shift | Use Case | Status |
|---------|------------|--------------|-----|-------|----------|--------|
| **Z-Image** | ~6.15B | 28-50 (~56-100 NFEs w/ CFG) | Yes (3.0-5.0) | 6.0 | High-quality generation, fine-tuning | **Released** |
| **Z-Image-Turbo** | ~6.15B | 8 (8 NFEs, no CFG) | No (0.0) | 3.0 | Fast inference, sub-second on H800 | **Released** |
| **Z-Image-Omni-Base** | ~6.15B | ~50 | Yes | Variable | Generation + editing (SigLIP2) | To be released |
| **Z-Image-Edit** | ~6.15B | ~50 | Yes | Variable | Dedicated editing (Rank 3 on ImgEdit) | Benchmarked, unreleased |

> **Z-Image base model released** (Feb 2026): The undistilled foundation model is now publicly available at `Tongyi-MAI/Z-Image`. Key advantages over Turbo: full CFG support, negative prompts, higher output diversity, better for fine-tuning/LoRA training. Recommended: `cfg_normalization=False` for stylistic output, `True` for photorealism.

> **Note on NFE counting**: The technical report states the SFT model requires ~100 NFEs for high-quality generation (50 denoising steps × 2 forwards for CFG). The diffusers pipeline for Z-Image-Turbo takes `num_inference_steps=9`, yielding **8 DiT forward passes** (NFEs). Blissful Tuner's `DEFAULT_INFERENCE_STEPS = 8` counts direct forwards. Sub-second inference (Turbo) requires **FlashAttention-3 + torch.compile**.

### Key Characteristics

- **Single-stream S3-DiT** (Scalable Single-Stream Diffusion Transformer)
- **Bilingual**: Native Chinese + English support via Qwen3-4B encoder (Qwen3-8B also supported)
- **Emergent multilingual**: Beyond training languages, Z-Image shows initial ability to handle Portuguese, Russian, Korean, Spanish, German, French, and other languages with culturally appropriate output
- **Arbitrary resolution**: 512×512 to 2048×2048 (must be divisible by 16)
- **Reversed flow matching**: Velocity prediction with reversed timestep convention (see [Section 4.1](#41-flow-matching-objective))
- **Euler discrete scheduler**: `FlowMatchEulerDiscreteScheduler` from diffusers
- **Ranked #1** open-source on Artificial Analysis T2I leaderboard

### Weight Storage on Disk

| Variant | DiT Weights | Text Encoder | Total (approx) | DiT dtype |
|---------|-------------|--------------|-----------------|-----------|
| **Z-Image** | 12.31 GB (2 files) | 8.04 GB (3 files) | ~20.4 GB | bfloat16 |
| **Z-Image-Turbo** | 24.62 GB (3 files) | 8.04 GB (3 files) | ~32.7 GB | float32 |

> **Important**: Z-Image-Turbo stores transformer weights in float32 (2x the size of Z-Image's bfloat16). The architecture and parameter count are identical — only the storage precision differs.

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
    # Pre-attention normalization (sandwich norm)
    attention_norm1: RMSNorm
    attention_norm2: RMSNorm

    # Attention with QK normalization
    attention: ZImageAttention
        - to_q, to_k, to_v: Linear(dim, dim, bias=False)
        - norm_q, norm_k: RMSNorm       # QK-Norm for stability
        - to_out: ModuleList[Linear(dim, dim, bias=False)]

    # Pre-FFN normalization (sandwich norm)
    ffn_norm1: RMSNorm
    ffn_norm2: RMSNorm

    # SwiGLU FFN (8/3 expansion ratio)
    feed_forward: FeedForward
        - w1: Linear(dim, hidden, bias=False)  # Gate  (3840 → 10240)
        - w2: Linear(hidden, dim, bias=False)  # Down  (10240 → 3840)
        - w3: Linear(dim, hidden, bias=False)  # Up    (3840 → 10240)
        # hidden_dim = int(dim / 3 * 8) = 10240

    # AdaLN modulation (4 outputs: scale + gate only, NO shift)
    # Low-rank decomposition: shared down-projection + per-layer up-projection
    # Down: t_embedder (shared across ALL layers): timestep → 256-dim
    # Up: adaLN_modulation (per-layer): 256-dim → 4 * dim
    adaLN_modulation: Linear(256, 4 * dim)  # 256 → 15360 (per-layer)
    # ⚠️ Block adaLN has NO activation (raw linear output, then chunk)
    # Outputs: scale_msa, gate_msa, scale_mlp, gate_mlp (tanh-gated)

# FinalLayer structure (DIFFERENT from main blocks):
class FinalLayer:
    norm_final: nn.LayerNorm(dim)           # ⚠️ LayerNorm, NOT RMSNorm!
    linear: Linear(dim, patch_h * patch_w * out_channels)
    adaLN_modulation: Sequential(           # ⚠️ HAS SiLU activation (unlike blocks)
        SiLU(),
        Linear(256, dim)                    # Only 1 output (scale), not 4
    )
    # FinalLayer: norm → scale → linear projection to pixel space
```

> **Key difference from standard AdaLN-Zero**: Z-Image uses only **4 modulation values** (scale + gate for attention and FFN), not the 6 values (shift + scale + gate) used by models like DiT/SD3. There are no shift terms. The gating uses `tanh` activation. The technical report describes this as a **low-rank decomposition**: a **shared, layer-agnostic down-projection** (`t_embedder`: timestep → 256-dim, shared across all 30 layers) followed by **layer-specific up-projections** (`adaLN_modulation`: 256 → 4×dim, unique per layer). This reduces parameter overhead while maintaining per-layer specialization.
>
> **Block vs FinalLayer adaLN (v2.3)**: The block-level `adaLN_modulation` is a bare `Linear(256, 4*dim)` with **no activation** — the timestep embedding passes through a linear projection and is immediately chunked. The `FinalLayer.adaLN_modulation` is a `Sequential(SiLU(), Linear(256, dim))` — it **does** have an activation. Additionally, FinalLayer uses `nn.LayerNorm` (not RMSNorm) and produces only **1 modulation value** (scale), not 4. This asymmetry is confirmed by the diffusers implementation.
>
> **TimestepEmbedder detail (v2.3)**: The shared `t_embedder` that produces the 256-dim bottleneck uses `mid_size=1024` internally: `Linear(256, 1024) → SiLU → Linear(1024, 256)`. The 256-dim output is then passed to every layer's per-layer up-projection.

### 2.3 Key Architectural Features

#### RMSNorm Everywhere + Sandwich-Norm
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

The technical report names the dual-normalization pattern as **"Sandwich-Norm"**: RMSNorm is applied at both the input and output of each attention/FFN sub-block to constrain signal amplitudes. Specifically:
- `norm1` is applied to the **input** (before modulation/attention)
- `norm2` is applied to the **output** of the attention/FFN computation (after the sub-block, before the gated residual add)

> **Diffusers correction (v2.3)**: The v2.2 pseudocode incorrectly showed `norm2` applied before the attention/FFN computation. The diffusers implementation (`transformer_z_image.py`) confirms `norm2` wraps the *output*: `x = x + gate * norm2(attn(norm1(x) * scale))`. This means the sandwich constrains both input and output amplitudes.

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
# Editing (OmniBase): reference and target image tokens share aligned spatial
# RoPE coordinates but are separated by a UNIT INTERVAL OFFSET in the temporal
# dimension. This encodes the reference→target relationship positionally.
```

#### AdaLN Modulation (Scale + Gate, No Shift)
```python
# Low-rank decomposition:
# Step 1: Shared down-projection (layer-agnostic, computed ONCE per timestep)
t_emb = self.t_embedder(t)  # [B] → [B, 256]  (shared across all 30 layers)
# Note: t_embedder uses mid_size=1024 internally (TimestepEmbedder: Linear(256,1024) → SiLU → Linear(1024,256))
# Step 2: Per-layer up-projection → 4 modulation values (NOT 6)
# Block adaLN has NO activation before chunking
scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(4)

# Applied with tanh gating (no shift term):
# CORRECTED ordering (v2.3): norm2 wraps the OUTPUT, not the input
x_norm = attention_norm1(x) * (1 + scale_msa.unsqueeze(1))  # Scale only (pre-attention)
attn_out = attention(x_norm, freqs)
attn_out = attention_norm2(attn_out)                          # Sandwich norm (post-attention)
x = x + tanh(gate_msa).unsqueeze(1) * attn_out               # Tanh-gated residual

# Same pattern for FFN:
x_norm = ffn_norm1(x) * (1 + scale_mlp.unsqueeze(1))
ffn_out = feed_forward(x_norm)
ffn_out = ffn_norm2(ffn_out)                                  # Sandwich norm (post-FFN)
x = x + tanh(gate_mlp).unsqueeze(1) * ffn_out
```

> **Context refiner blocks** (`context_refiner`) use `modulation=False` — they skip AdaLN entirely, processing caption tokens without timestep conditioning.

### 2.4 Patch Embedding

```python
# Configuration
all_patch_size = [2]      # 2×2 spatial patches
all_f_patch_size = [1]    # 1 temporal patch (image-only)
in_channels = 16          # VAE latent channels

# Embedding (stored in ModuleDict for multi-patch support)
all_x_embedder["2-1"] = nn.Linear(in_channels * patch_h * patch_w, dim)
# For 2×2 patches: Linear(16 * 2 * 2, 3840) = Linear(64, 3840)

# Learnable padding tokens
x_pad_token = nn.Parameter(torch.zeros(1, dim))    # Image padding
cap_pad_token = nn.Parameter(torch.zeros(1, dim))   # Caption padding
```

### 2.5 FP16 Safety Mechanisms

The Blissful Tuner implementation includes numerical stability guards for FP16 training/inference:

```python
# FeedForward: divide output by 32 in fp16 to prevent overflow
if x.dtype == torch.float16:
    x = self.w2(F.silu(self.w1(x)) * (self.w3(x) / 32.0))
else:
    x = self.w2(F.silu(self.w1(x)) * self.w3(x))

# Attention: divide output by 4 in fp16
if out.dtype == torch.float16:
    return self.to_out[0](out / 4.0)
else:
    return self.to_out[0](out)
```

> **Recommendation**: Use bfloat16 (`--mixed_precision bf16`) for training to avoid these fp16 numerical workarounds.

---

## 3. Configuration Reference

### 3.1 Transformer Configuration

```python
# From HuggingFace config.json (identical for both Z-Image and Z-Image-Turbo)
DEFAULT_TRANSFORMER_CONFIG = {
    "_class_name": "ZImageTransformer2DModel",
    "dim": 3840,                    # Hidden dimension
    "in_channels": 16,              # VAE latent channels
    "n_layers": 30,                 # Main transformer layers
    "n_heads": 30,                  # Attention heads
    "n_kv_heads": 30,               # KV heads (no GQA; ⚠️ unused in diffusers impl)
    "n_refiner_layers": 2,          # Refiner layers (noise + context)
    "norm_eps": 1e-5,               # RMSNorm epsilon
    "qk_norm": True,                # Enable QK normalization
    "cap_feat_dim": 2560,           # Text embedding dimension (from Qwen3-4B)
    "siglip_feat_dim": None,        # SigLIP dimension (OmniBase only: 1152)
    "t_scale": 1000.0,              # Timestep scaling (0-1 mapped to 0-1000)
    "rope_theta": 256.0,            # RoPE base frequency
    "axes_dims": [32, 48, 48],      # RoPE dimensions [T, H, W] (sum=128=head_dim)
    "axes_lens": [1536, 512, 512],  # Max RoPE positions (⚠️ see discrepancy note below)
    "all_patch_size": [2],          # Spatial patch size (2×2)
    "all_f_patch_size": [1],        # Temporal patch size (1 = image-only)
}

# ⚠️ PAPER DISCREPANCY: The technical report Table 2 states 32 attention heads,
# but the HuggingFace config.json specifies 30. With dim=3840:
#   32 heads → head_dim = 120 (DOES NOT match RoPE axes sum 32+48+48=128)
#   30 heads → head_dim = 128 (MATCHES RoPE axes sum perfectly)
# The HF config (30 heads) is correct; the paper's "32" is likely a typo.

# ⚠️ AXES_LENS DISCREPANCY (v2.3): The HF config.json specifies [1536, 512, 512],
# but the diffusers transformer code defaults to [1024, 512, 512] for the temporal axis.
# The HF config value (1536) takes precedence when loading from pretrained, but the
# code default (1024) may affect newly constructed models or tests.

# ⚠️ N_KV_HEADS (v2.3): The `n_kv_heads` parameter exists in the config but is
# UNUSED in the diffusers implementation — all attention projections use `n_heads` (30).
# The transformer does not implement GQA (unlike the text encoder which uses GQA).

# Derived constants (computed in Blissful Tuner implementation):
DERIVED_CONSTANTS = {
    "head_dim": 128,                # dim // n_heads = 3840 // 30
    "ffn_hidden_dim": 10240,        # int(dim / 3 * 8) = int(3840 / 3 * 8)
    "adaln_embed_dim": 256,         # min(dim, 256) — timestep bottleneck
    "seq_multi_of": 32,             # Sequence length padding multiple
    "rope_axes_sum": 128,           # 32 + 48 + 48 = head_dim (must match)
}
```

### 3.2 Text Encoder Configuration (Qwen3-4B / Qwen3-8B)

The default text encoder is **Qwen3-4B**. Blissful Tuner also supports Qwen3-8B via the `is_8b` flag.

> **Text encoder class (v2.3)**: The HF `text_encoder/config.json` specifies `Qwen3ForCausalLM`, but the `model_index.json` pipeline definition uses `Qwen3Model` (the base model without the LM head). The diffusers pipeline loads `Qwen3Model` directly — only the hidden states are used for conditioning, not logits. Blissful Tuner loads via `Qwen3ForCausalLM` and extracts hidden states, which produces identical conditioning since only `hidden_states[-2]` is used.

```python
# Qwen3-4B (default, from HuggingFace text_encoder/config.json)
QWEN3_4B_CONFIG = {
    "_class_name": "Qwen3ForCausalLM",  # ⚠️ Diffusers uses Qwen3Model (see note above)
    "model_type": "qwen3",
    "hidden_size": 2560,            # Output embedding dimension → cap_feat_dim
    "num_hidden_layers": 36,        # Transformer layers
    "num_attention_heads": 32,      # Query heads
    "num_key_value_heads": 8,       # KV heads (GQA: 4× compression)
    "head_dim": 128,                # Per-head dimension (2560 / 32 = 80? No: explicit 128)
    "intermediate_size": 9728,      # FFN hidden dimension
    "hidden_act": "silu",           # Activation function
    "vocab_size": 151936,           # Vocabulary size
    "max_position_embeddings": 40960,
    "rope_theta": 1000000,          # Different from transformer's 256.0!
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
}

# Qwen3-8B (alternative, larger encoder)
QWEN3_8B_CONFIG = {
    "hidden_size": 4096,            # Larger embedding dimension
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "tie_word_embeddings": False,   # Different from 4B!
}
```

**Text Encoding Pipeline** (important implementation details):

```python
# Tokenizer: Qwen2Tokenizer (not Qwen3Tokenizer!)
# Loaded from: Tongyi-MAI/Z-Image repo, "tokenizer" subfolder
# model_max_length: 131072, but truncated to 512 for diffusion

# Encoding steps (from zimage_utils.py get_text_embeds):
# 1. Apply chat template with enable_thinking=True
# 2. Tokenize: padding="max_length", max_length=512, truncation=True
# 3. Forward through Qwen3 model
# 4. Extract hidden_states[-2]  ← PENULTIMATE layer (not last!)
# 5. Apply attention_mask to zero out padding

# Generation config (text_encoder/generation_config.json):
# temperature=0.6, top_k=20, top_p=0.95, do_sample=True
# eos_token_id: [151645, 151643] (dual EOS tokens)
```

> **Critical**: The text encoder uses `hidden_states[-2]` (penultimate layer), NOT the final hidden state. This is a common pattern in diffusion models where intermediate representations provide better conditioning signals. The chat template with `enable_thinking=True` is also applied before tokenization.

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
    "use_quant_conv": False,        # No quantization conv
    "use_post_quant_conv": False,
    "mid_block_add_attention": True, # Self-attention in mid block
    "force_upcast": True,           # Always runs in float32
}
```

> **VAE Config Mismatch Warning**: The official Z-Image GitHub repo `config/model.py` contains **incorrect default values**: `DEFAULT_VAE_LATENT_CHANNELS = 4` and `DEFAULT_VAE_SCALING_FACTOR = 0.18215` (these are SD1.x/SDXL defaults, not the actual Z-Image VAE values). The **correct** values (latent_channels=16, scaling_factor=0.3611, shift_factor=0.1159) come from the HuggingFace `vae/config.json`. Blissful Tuner correctly overrides these in `zimage_config.py`.
>
> **Decode-only upstream**: The official Z-Image VAE implementation only has a `decode()` method (inference-only). Blissful Tuner's `zimage_autoencoder.py` implements both `encode()` and `decode()` for training support.

### 3.4 Scheduler Configuration

Both variants use `FlowMatchEulerDiscreteScheduler` but with different shift values:

```python
# Z-Image (from Tongyi-MAI/Z-Image scheduler_config.json)
ZIMAGE_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.37.0.dev0",
    "num_train_timesteps": 1000,
    "shift": 6.0,                   # Higher shift for multi-step sampling
    "use_dynamic_shifting": False,
}

# Z-Image-Turbo (from Tongyi-MAI/Z-Image-Turbo scheduler_config.json)
ZIMAGE_TURBO_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.36.0.dev0",
    "num_train_timesteps": 1000,
    "shift": 3.0,                   # Lower shift optimized during distillation
    "use_dynamic_shifting": False,
}
```

> **Note**: Blissful Tuner's `DEFAULT_SCHEDULER_SHIFT = 3.0` matches the Turbo default. The diffusers versions also differ between variants (0.37.0.dev0 vs 0.36.0.dev0).
>
> **Diffusers scheduler features (v2.3)**: The diffusers `FlowMatchEulerDiscreteScheduler` supports two additional features not in Blissful Tuner's standalone scheduler:
> - **`stochastic_sampling`**: Adds noise injection during Euler steps for diversity (off by default)
> - **`per_token_timesteps`**: Per-token timestep scheduling used by OmniBase (reference tokens get different timesteps than target tokens)
>
> The diffusers scheduler also forces `sigma_min=0.0` for Z-Image pipelines (overriding any configured value).

### 3.5 Dynamic Shift Computation

For resolution-dependent shifting (used during training and optionally during inference):

```python
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15

def compute_shift(image_seq_len):
    mu = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN) * image_seq_len + BASE_SHIFT
    return mu
```

### 3.6 Inference Parameters (Verified from HuggingFace Configs)

| Model | Denoising Steps | NFEs (with CFG) | CFG Scale | Shift | Resolution | Pipeline Steps Arg |
|-------|-----------------|-----------------|-----------|-------|------------|-------------------|
| Z-Image | 28-50 | ~56-100 (2× for CFG) | 3.0-5.0 | 6.0 | 512-2048 | `num_inference_steps=50` |
| Z-Image-Turbo | 8 | 8 (no CFG) | 0.0 | 3.0 | 512-2048 | `num_inference_steps=9` (→ 8 forwards) |

> **Blissful Tuner CLI defaults**: `--infer_steps 25`, `--guidance_scale 0.0`, `--flow_shift 3.0`. For Z-Image base, override: `--infer_steps 50 --guidance_scale 4.0 --flow_shift 6.0`.

### 3.7 CFG Truncation & Normalization (Official Pipeline Features)

The official Z-Image pipeline supports two advanced CFG features:

```python
# CFG Truncation: disables CFG at high noise levels
# When t_normalized > cfg_truncation, guidance_scale is forced to 0.0
# Default: cfg_truncation=1.0 (CFG applies at all timesteps when guidance > 1.0)
# Lower values (e.g., 0.8) disable CFG for the noisiest 20% of timesteps
generate(..., cfg_truncation=0.8)

# CFG Normalization: rescales prediction to match positive prediction norm
# Prevents CFG from amplifying overall image magnitude
# Recommended: False for stylistic output, True for photorealism
generate(..., cfg_normalization=True)
```

> **Blissful Tuner status**: CFG truncation and normalization are available through the shared guidance/scheduling modules but are not exposed as Z-Image-specific CLI flags.

### 3.8 Attention Backend Configuration

The official Z-Image repo supports 8 attention backends, configurable via environment variable:

```bash
# Environment variable (official repo):
export ZIMAGE_ATTENTION="_native_flash"  # default

# Available backends:
# CUDA backends:
#   "flash"           - Flash Attention 2
#   "flash_varlen"    - Flash Attention 2 (variable-length)
#   "_flash_3"        - Flash Attention 3
#   "_flash_varlen_3" - Flash Attention 3 (variable-length)
# Apple Silicon (NEW as of 2026-01-30):
#   "mps_flash"       - MPS Flash Attention (requires: pip install mps-flash-attn)
# PyTorch native:
#   "native"          - SDPA auto-select kernel
#   "_native_flash"   - SDPA forced flash kernel (default)
#   "_native_math"    - SDPA forced math kernel

# Blissful Tuner equivalent (via --attn_mode CLI flag):
#   "torch"     → PyTorch SDPA (auto-select)
#   "flash"     → Flash Attention 2 (with varlen)
#   "sageattn"  → SageAttention (not in upstream)
#   "xformers"  → xformers (not in upstream)
#   "sdpa"      → alias for "torch"
```

> **Note**: The upstream `dispatch_attention()` uses explicit if/elif chains instead of dict lookups to avoid `torch.compile` dynamo guard issues. Blissful Tuner uses a unified `attention()` function with `AttentionParams` dataclass.

---

## 4. Training Methodology

### 4.1 Flow Matching Objective (Reversed Convention)

Z-Image uses a **reversed flow matching convention** compared to standard formulations. The timestep, target, and inference noise prediction are all inverted:

```python
# Blissful Tuner training convention (from zimage_train_network.py):
def compute_loss(model, latents, noise, timesteps, condition):
    # Timestep reversal: scheduler gives t ∈ [0, 1000], convert to reversed [0, 1]
    t_input = (1000.0 - timesteps) / 1000.0

    # Noisy latents via standard interpolation
    x_t = (1 - t) * noise + t * latents  # (standard flow matching interpolation)

    # Target velocity (REVERSED: latents - noise, not noise - latents)
    v_target = latents - noise

    # Model prediction with frame dimension
    latents_5d = latents.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W]
    v_pred = model(latents_5d, t_input, condition)
    v_pred = v_pred.squeeze(2)  # [B, C, 1, H, W] → [B, C, H, W]

    # MSE loss (unreduced for mask loss support)
    loss = F.mse_loss(v_pred, v_target, reduction="none")
    return loss

# During inference, noise prediction is NEGATED:
noise_pred = -model_output.squeeze(2)  # Reverse the convention
prev_sample = sample + dt * noise_pred  # Euler step
```

> **Why reversed?**: The Z-Image architecture internally treats `t=0` as clean data and `t=1` as pure noise, opposite to the diffusers convention where `t=1000` is noise. The timestep is reversed at the boundary (`(1000 - t) / 1000`), the target is inverted (`latents - noise`), and the inference output is negated (`-model_output`). These three inversions cancel out to produce correct results.

### 4.2 Timestep Sampling & Scheduling

```python
# The original Z-Image pre-training used logit-normal sampling (following SD3)
# combined with Flux-style dynamic time shifting. For LoRA fine-tuning,
# logit_normal may better match the base model's training distribution.
#
# Blissful Tuner supports multiple timestep sampling strategies:
# --timestep_sampling logit_normal (matches original training distribution)
# --timestep_sampling shift        (Blissful Tuner default)
# --discrete_flow_shift 3.0        (Turbo default; use 6.0 for Z-Image base)

# Sigma schedule generation (from zimage_utils.py):
def get_timesteps_sigmas(num_inference_steps, shift):
    timesteps = torch.linspace(1000, 1, num_inference_steps)
    sigmas = timesteps / 1000.0
    # Apply flux-style shift
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1)])  # Append final sigma=0
    return timesteps, sigmas  # Both float32

# Euler step:
def step(model_output, sample, sigmas, step_index):
    dt = sigmas[step_index + 1] - sigmas[step_index]
    prev_sample = sample + dt * model_output  # All in float32
    return prev_sample
```

> **Training defaults**: `--discrete_flow_shift 3.0` (matches Turbo). For Z-Image base LoRA training, use `--discrete_flow_shift 6.0`.
>
> **Original training distribution**: The technical report confirms pre-training used **logit-normal sampling** (following SD3) combined with **Flux-style dynamic time shifting**. Logit-normal concentrates training on intermediate timesteps while dynamic shifting adjusts the noise schedule per resolution. For LoRA fine-tuning, `--timestep_sampling logit_normal` may better match the base model's training distribution than the default `shift`.

### 4.3 Three-Phase Training Curriculum

**Phase 1: Low-Resolution Pre-Training**
- Resolution: 256×256 (fixed)
- Task: Text-to-image only
- Budget: >50% of total compute (~147.5K H800 GPU hours)
- Objective: Establish cross-modal alignment
- **Key insight**: The majority of foundational visual knowledge (including Chinese text rendering) is acquired at this low resolution

**Phase 2: Omni-Pre-Training**
- Arbitrary resolution training
- Joint T2I + I2I tasks (enables Z-Image-Edit derivation via this "omni-pretraining paradigm")
- Multi-level bilingual captions (5 types — see [Section 5.5](#55-caption-recommendations-from-technical-report))
- Budget: ~45% of compute (~142.5K H800 GPU hours)

**Phase 3: Supervised Fine-Tuning (PE-aware)**
- High-quality curated data with **Prompt Enhancer (PE)**-augmented captions (see [Section 4.6](#46-prompt-enhancer-pe-system))
- Concept balancing via BM25-based dynamic resampling: rarity scores computed on-the-fly, under-represented concepts up-weighted in mini-batch construction
- **Model merging**: Multiple SFT variants trained on different capability dimensions (e.g., aesthetics, text rendering, composition), then combined via **linear weight interpolation**: `θ_final = Σᵢ αᵢθᵢ`
- Budget: ~5% of compute (~24K H800 GPU hours)

### 4.4 Few-Step Distillation (Turbo)

**Decoupled DMD** separates:
1. **CFG-Augmentation (CA)**: Builds few-step capability
2. **Distribution Matching (DM)**: Ensures stability

**DMDR** adds RL with DM as regularizer:
- Prevents reward hacking
- Enables preference alignment
- Results: 8 NFEs matching 100-step teacher

### 4.5 Post-Training: RLHF Pipeline (DPO + GRPO)

The post-training pipeline uses a two-stage human preference alignment process:

**Stage 1: Offline DPO (Objective Dimensions)**
- Targets measurable quality dimensions: text rendering accuracy, object counting, spatial relationships
- Preference pairs generated by VLM evaluation with human verification
- Offline training on curated comparison dataset

**Stage 2: Online GRPO (Subjective Dimensions)**
- Targets subjective quality: realism, aesthetics, instruction following
- Uses a multi-dimensional reward model trained on human preferences
- Online reinforcement learning with Distribution Matching as regularizer (prevents reward hacking)

> **"Reward post-training"**: The technical report names this as a distinct optimization phase alongside PE-aware SFT and few-step distillation.

### 4.6 Prompt Enhancer (PE) System

Z-Image uses a **Prompt Enhancer (PE)** to compensate for the 6B model's limited world knowledge and reasoning capacity. The PE is a **frozen pre-trained VLM** (not fine-tuned) that processes user prompts through a **4-stage reasoning chain**:

1. **Core Subject Analysis**: Identify the main subject and intent of the prompt
2. **Problem Solving / World Knowledge**: Inject specific historical, cultural, or factual details (e.g., solving math word problems, retrieving architectural details for landmarks)
3. **Aesthetic Enhancement**: Add artistic and photographic quality descriptors
4. **Comprehensive Description**: Synthesize into a detailed generation prompt

```
User prompt: "After Passing the Imperial Examination"
    ↓ PE Stage 1: Identifies subject (Chinese imperial exam celebration)
    ↓ PE Stage 2: Retrieves historical details (Song dynasty, jinshi degree, horse parade)
    ↓ PE Stage 3: Adds aesthetic cues (dramatic lighting, traditional Chinese painting style)
    ↓ PE Stage 4: Full generation prompt with all enrichments
```

**PE-aware SFT**: During supervised fine-tuning (Phase 3), the training data uses PE-enhanced captions, so the base model learns to work with PE output format. The PE's VLM remains frozen — only the diffusion model is aligned to the PE's output distribution.

**Language locking**: The PE determines output language from input language and maintains consistency throughout the reasoning chain.

> **Relevance to Blissful Tuner**: The PE is an inference-time system component. For LoRA training, the PE is not directly involved, but awareness of PE-enhanced caption format may inform caption preparation strategy.

### 4.7 Data Pipeline

**AIGC Content Filtering**: Following Imagen 3's findings, a dedicated classifier was trained to **detect and filter out AI-generated content** from the training data. This prevents quality degradation and maintains physical realism in outputs.

**Semantic Deduplication**: GPU-accelerated k-NN community detection pipeline processes ~1 billion items in ~8 hours on 8×H800 GPUs (index construction + 100-NN querying). Enhanced from SD3's deduplication method, reformulated as a scalable graph-based community detection task.

**OCR-Augmented Captioning (Chain-of-Thought)**: A two-step captioning process that explains Z-Image's exceptional text rendering:
1. **OCR pass**: Explicitly recognize all optical characters in the image, preserving original languages without translation
2. **Caption generation**: Generate the image caption incorporating the OCR results as grounding context

This CoT-style approach ensures text content is accurately represented in training captions.

### 4.8 Editing Data Construction

For image editing training (OmniBase/Z-Image-Edit):

- **Pair generation**: From N edited versions of one input image, generate all `2 × C(N+1, 2)` pair combinations — both forward edits, inverse edits, and mixed-editing pairs (combining two edits into one instruction)
- **Resolution progression**: Start at **512²** for a few thousand steps (quick adaptation), then increase to **1024²** for quality
- **Data ratio**: **4:1 T2I:I2I** data during editing training to prevent catastrophic forgetting

### 4.9 Original Training Infrastructure

For reference, the original Z-Image pre-training used:
- **FSDP2** (Fully Sharded Data Parallelism v2) for the DiT model — not plain DDP
- **Data Parallelism only** for VAE and text encoder (kept frozen)
- **Gradient checkpointing** across all DiT layers
- **torch.compile** for DiT blocks
- **Sequence length-aware batch construction**: Dynamic batch sizing (smaller batches for long sequences, larger for short) to maximize GPU utilization

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

# Default exclude from LoRA (from network_arch.py registry)
exclude_patterns = [
    r".*(_modulation|_refiner).*",  # AdaLN modulation + refiner layers
]
# Note: norm layers, embedders are implicitly excluded because they are
# not inside ZImageTransformerBlock (the target module class).
# LoRA prefix: "lora_unet"
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

### 5.4 Diffusers LoRA Differences (v2.3)

The diffusers DreamBooth LoRA training script (`train_dreambooth_lora_z_image.py`) uses significantly different defaults from Blissful Tuner. Understanding these differences is important for LoRA interoperability.

| Aspect | Blissful Tuner | Diffusers DreamBooth |
|--------|----------------|---------------------|
| **Default rank** | 32 | 4 |
| **Target layers** | Attention + FFN (all `ZImageTransformerBlock` children) | Attention only (`to_q`, `to_k`, `to_v`, `to_out.0`) |
| **noise_refiner LoRA** | Excluded (`_refiner` in exclude pattern) | Included (LoRA applied to noise_refiner layers) |
| **Alpha handling** | Separate `lora_alpha` parameter stored in metadata | Baked into weights during conversion (`weight *= alpha/rank`) |
| **Alpha default** | `network_alpha` (explicit) | Same as rank (alpha=rank → scale=1.0) |

**Key implications:**

1. **LoRA portability**: LoRAs trained in diffusers (rank 4, attention-only, with noise_refiner) are NOT directly compatible with Blissful Tuner LoRAs (rank 32, attention+FFN, without noise_refiner). The alpha baking difference means weight scales differ at load time.

2. **noise_refiner inclusion**: Diffusers applies LoRA to the 2 noise refiner layers, which Blissful Tuner excludes. This means diffusers LoRAs may learn noise-level-specific adaptations that Blissful Tuner LoRAs cannot. Whether this is beneficial depends on the training objective.

3. **Timestep weighting schemes**: The diffusers training script supports 5 weighting options:
   - `none` — uniform weighting
   - `sigma_sqrt` — weight by sqrt(sigma)
   - `logit_normal` — logit-normal distribution
   - `mode` — mode-seeking
   - `cosmap` — cosine mapping

4. **LoRA key conversion**: Diffusers' `lora_conversion_utils.py` uses a protected n-gram algorithm to convert between diffusers and non-diffusers key formats. The conversion bakes alpha into weights: `weight = weight * (alpha / rank)`.

> **Recommendation**: When training Z-Image LoRAs intended for cross-ecosystem use, document the rank, alpha, and target layer choices clearly in the LoRA metadata.

### 5.5 Caption Recommendations (From Technical Report)

The technical report specifies **five caption types** used during training (not four):

```
# 1. Long description (detailed)
A serene mountain landscape at sunset with golden light reflecting off
a pristine alpine lake, surrounded by pine forests and snow-capped peaks...

# 2. Medium description
Mountain lake at sunset with golden reflections and pine forests.

# 3. Short description
Sunset mountain lake scene.

# 4. Tags
landscape, mountain, lake, sunset, nature, alpine, golden hour, reflection

# 5. Simulated user prompts (NEW — mimics real user behavior)
# Intentionally incomplete, focusing on specific aspects of interest.
# These teach the model to handle partial/casual prompts gracefully.
mountain sunset with lake
```

> **OCR-augmented captions**: Training captions were generated using a CoT process — OCR first (preserving original languages), then caption generation incorporating the OCR results. This is a key factor in Z-Image's text rendering quality. See [Section 4.7](#47-data-pipeline) for details.

---

## 6. OmniBase Integration (Image Editing)

### 6.1 Architecture Extensions

OmniBase adds these components to the base Z-Image:

```python
class ZImageTransformer2DModel:
    # NEW: SigLIP feature processing (DEFAULT_TRANSFORMER_SIGLIP_FEAT_DIM = 1152)
    siglip_feat_dim: int = 1152  # SigLIP2 hidden size
    siglip_embedder: nn.Sequential  # RMSNorm + Linear(1152, 3840)
    siglip_refiner: nn.ModuleList   # 2 transformer blocks
    siglip_pad_token: nn.Parameter  # Learnable padding

    # NEW: Per-token noise selection (4*D modulation, not 6*D)
    def select_per_token(noisy, clean, noise_mask, seq_len):
        """Select between noisy and clean embeddings per token.
        Modulation is 4*dim (scale+gate for attn and FFN)."""
        mask = noise_mask.unsqueeze(-1).expand(-1, seq_len, -1)
        return torch.where(mask == 1, noisy, clean)

# OmniBase detection (from zimage_utils.py):
def should_enable_omnibase(state_dict):
    """Auto-detect OmniBase from checkpoint keys."""
    return any(k.startswith("siglip_embedder.") for k in state_dict.keys())

def infer_siglip_feat_dim(state_dict):
    """Infer SigLIP feature dim from weights (fallback: 1152)."""
    key = "siglip_embedder.1.weight"
    return state_dict[key].shape[1] if key in state_dict else 1152

# OMNIBASE_T_CLEAN = 1.0  # Reference-image time-conditioning value
```

> **Implementation status**: Blissful Tuner has the utility functions (`load_siglip2_encoder`, `siglip_last_hidden_to_grid`, OmniBase detection/inference) and caching support, but the transformer model's `forward()` does not yet include the OmniBase branch.

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

## 7. Diffusers Pipeline Ecosystem (v2.3)

This section documents the Z-Image pipeline variants, ControlNet architecture, and modular pipeline system as implemented in HuggingFace diffusers. These findings are based on analysis of 51 Z-Image-related files in the diffusers repository.

### 7.1 Pipeline Variants

Diffusers implements **6 pipeline variants** for Z-Image:

| Pipeline | Class | Use Case | Key Differences |
|----------|-------|----------|-----------------|
| `ZImagePipeline` | Base T2I | Text-to-image generation | Standard flow matching |
| `ZImageImg2ImgPipeline` | Img2Img | Image-to-image transformation | Adds initial latent encoding + noise injection |
| `ZImageInpaintPipeline` | Inpaint | Masked region filling | Latent-space blending (not model-conditioned) |
| `ZImageControlNetPipeline` | ControlNet | Conditioned generation | Adds ControlNet residuals to transformer |
| `ZImageControlNetInpaintPipeline` | ControlNet+Inpaint | Conditioned inpainting | 33-channel control input |
| `ZImageOmniPipeline` | OmniBase | Generation + editing | Radically different text encoding |

**All pipelines share:**
- List-based transformer I/O (not batched tensors)
- `sigma_min` forced to `0.0`
- Same VAE encode/decode path (always fp32)
- Support for `cfg_truncation` and `cfg_normalization`

#### Inpainting Details

Z-Image inpainting uses **latent-space blending**, not model-conditioned inpainting:

```python
# During each denoising step:
# 1. Noise the original latents to current timestep
init_latents_noised = scheduler.scale_noise(init_latents, timestep, noise)
# 2. Blend: keep original in non-mask regions, denoised in mask regions
latents = init_latents_noised * (1 - mask) + latents * mask
```

> **Implication for Blissful Tuner**: This is simpler than model-conditioned inpainting (no extra channels) but may produce visible seams at mask boundaries. The mask loss training approach in Blissful Tuner is complementary — it trains the model to focus on specific regions rather than performing inference-time blending.

#### ControlNet Inpainting (33 Channels)

The ControlNet inpaint pipeline concatenates three inputs into the control signal:

```python
# control_image → encode → control_latents (16ch)
# mask → downsample → mask_latents (1ch)
# init_image → encode → init_image_latents (16ch)
control_input = torch.cat([control_latents, mask_latents, init_image_latents], dim=1)
# Total: 16 + 1 + 16 = 33 channels
```

#### Omni Pipeline Text Encoding

The Omni pipeline uses a **completely different text encoding approach** from all other Z-Image pipelines:

```python
# Standard pipelines: uses apply_chat_template + model forward
# Omni pipeline: manual special token construction
tokens = [BOS] + thinking_tokens + [END_TURN] + text_tokens + [END_TURN] + [PAD...]
# No apply_chat_template() call — manually constructs the token sequence
# This affects any future OmniBase integration in Blissful Tuner
```

### 7.2 ControlNet Architecture

The Z-Image ControlNet (`ZImageControlTransformerBlock`) is entirely undocumented in previous versions. Key architecture details:

```python
class ZImageControlNetModel:
    # Inherits transformer structure but adds control-specific components:

    # Weight sharing: initialized from a pretrained transformer
    @classmethod
    def from_transformer(cls, transformer, ...):
        """Copy transformer weights into ControlNet structure."""
        # Shares: layers, noise_refiner, embedders, t_embedder
        # Adds: per-block before_proj + after_proj (zero-initialized)

    # Per-block control injection (zero-init for stable training start):
    class ZImageControlTransformerBlock:
        before_proj: Linear(dim, dim)    # Zero-init, applied before block
        after_proj: Linear(dim, dim)     # Zero-init, applied after block
        # These projections start at zero → no effect at init → gradual learning

    # Control signal injection (dict-indexed, additive):
    # The ControlNet produces a dict of per-block residuals:
    # block_samples = {"block_0": residual_0, "block_1": residual_1, ...}
    # Main transformer adds: x = x + controlnet_block_samples[f"block_{i}"]

    # Three modes for noise refinement:
    # 1. Use transformer's noise_refiner on control input
    # 2. Skip noise refinement for control
    # 3. Use separate control-specific noise_refiner

    # ControlNet Union design: single ControlNet, NOT multi-stacked
    # (no support for combining multiple ControlNets)
```

> **Blissful Tuner status**: No ControlNet support currently exists. The architecture is well-suited for extension — the `from_transformer()` pattern means a trained Z-Image model can spawn a ControlNet with shared weights + zero-init projections for fine-tuning.

### 7.3 Modular Pipeline System

Diffusers implements a **modular pipeline system** for Z-Image with composable blocks:

```python
# Modular pipeline auto-detection:
# Based on loaded components, auto-selects the right pipeline:
# - Has ControlNet? → ControlNet pipeline blocks
# - Has init_image? → Img2Img blocks
# - Has mask? → Inpaint blocks

# Composable blocks (each handles one stage):
# - TextEncoderStep: Tokenize + encode text
# - PrepareLatentsStep: Create/encode initial latents
# - DenoiseStep: Run denoising loop
# - DecodeStep: VAE decode latents to pixels

# Guider abstraction:
# - ClassifierFreeGuidanceGuider: Standard CFG
# - Handles cfg_truncation and negative prompts
# - ⚠️ Missing: cfg_normalization (not in modular pipeline yet)
```

> **Note**: The modular pipeline system is a newer diffusers feature. It lacks some features present in the standard pipelines (notably `cfg_normalization`). For Blissful Tuner development, the standard pipeline implementations are the more reliable reference.

### 7.4 Flow Matching Convention (Diffusers)

The diffusers implementation uses a slightly different flow matching sign convention:

```python
# Diffusers scheduler target:
target = noise - model_input    # (opposite of Blissful Tuner's latents - noise)

# To compensate, diffusers NEGATES the model output before the Euler step:
noise_pred = -model_output
prev_sample = sample + dt * noise_pred

# Net effect is identical to Blissful Tuner's convention — the two negations cancel.
# Both produce correct denoising. This is a convention difference, not a bug.
```

### 7.5 torch.compile Considerations

From the diffusers tests and implementation:

- **`@maybe_allow_in_graph`** decorator on `ZImageTransformerBlock` for torch.compile compatibility
- **`_skip_layerwise_casting_patterns`** list to exempt certain modules from layerwise dtype casting
- **`torch.compile(fullgraph=True)` is BROKEN** for Z-Image in diffusers tests — falls back to non-fullgraph mode
- **RoPE cache state pollution**: Tests discovered that RoPE frequency caches can leak between forward passes with different resolutions, causing subtle numerical errors
- **`complex64` RoPE requires non-deterministic mode**: `torch.use_deterministic_algorithms(False)` needed for RoPE with complex64 dtype

---

## 8. Implementation Comparison (Three-Way)

### 8.1 Blissful-Tuner vs Official vs Diffusers

| Aspect | Blissful-Tuner | Official Z-Image | Diffusers |
|--------|----------------|------------------|-----------|
| **Architecture** | | | |
| Transformer layers | 30 | 30 | 30 |
| Refiner layers | 2 (noise + context) | 2 (noise + context) | 2 (noise + context) |
| Hidden dim | 3840 | 3840 | 3840 |
| FFN hidden dim | 10240 (8/3 ratio) | 10240 (8/3 ratio) | 10240 (8/3 ratio) |
| AdaLN block modulation | 4 outputs (scale+gate, tanh) | 4 outputs (scale+gate, tanh) | 4 outputs, **no activation** |
| AdaLN FinalLayer | SiLU + 1 output (scale only) | SiLU + 1 output (scale only) | SiLU + 1 output (scale only) |
| FinalLayer norm | **nn.LayerNorm** (correct) | RMSNorm | **nn.LayerNorm** (not RMSNorm!) |
| AdaLN bottleneck | 256-dim | 256-dim | 256-dim (mid_size=1024) |
| QK-Norm | ✅ Yes (RMSNorm) | ✅ Yes (RMSNorm) | ✅ Yes (RMSNorm) |
| RMSNorm fp32 cast | ✅ Yes (internal) | ❌ No (operates in input dtype) | ❌ No |
| Sandwich-Norm ordering | norm2 **post-output** (correct) | norm2 pre-attention | norm2 **post-output** (correct) |
| RoPE 3D | ✅ Yes (3-axis) | ✅ Yes (3-axis) | ✅ Yes (3-axis) |
| n_kv_heads | Used (matches n_heads) | Used | **Unused** (no GQA) |
| axes_lens temporal | 1536 (from config) | 1536 | Default **1024** (overridden by config) |
| **Forward signature** | Batched `[B, C, F, H, W]` tensor | `List[Tensor]` per-sample | `List[Tensor]` per-sample |
| **Return type** | Single batched tensor | `Transformer2DModelOutput` | `Transformer2DModelOutput` |
| **Text encoder** | | | |
| Text encoder class | Qwen3ForCausalLM | Qwen3ForCausalLM | **Qwen3Model** (base) |
| Text layer used | hidden_states[-2] | hidden_states[-2] | hidden_states[-2] |
| Chat template | enable_thinking=True | enable_thinking=True | enable_thinking=True |
| Max seq length | 512 | 512 | 512 |
| **VAE** | | | |
| VAE | Flux VAE (always fp32) | Flux VAE (always fp32, decode-only) | AutoencoderKL (fp32) |
| VAE encode() | ✅ Yes (training support) | ❌ No (inference-only) | ✅ Yes |
| VAE config values | ✅ Correct (16ch, 0.3611) | ⚠️ Config has SD1.x defaults | ✅ Correct |
| **Attention** | | | |
| Attention backends | torch/flash/sageattn/xformers/sdpa | FA2/FA3/MPS Flash/SDPA (8) | PyTorch SDPA (via diffusers) |
| MPS Flash Attention | ❌ | ✅ (new, via mps-flash-attn) | ❌ |
| Flash Attention 3 | ❌ (CLI accepts but unhandled) | ✅ (flash_3, flash_varlen_3) | ❌ |
| SageAttention | ✅ | ❌ | ❌ |
| xformers | ✅ | ❌ | ✅ (via diffusers) |
| split_attn mode | ✅ | ❌ | ❌ |
| torch.compile compat | Not specifically optimized | ✅ (LRU cache, no graph breaks) | ⚠️ fullgraph=True **broken** |
| **Pipelines** | | | |
| T2I | ✅ | ✅ | ✅ |
| Img2Img | ❌ | ❌ | ✅ |
| Inpaint | ❌ | ❌ | ✅ (latent-space blending) |
| ControlNet | ❌ | ❌ | ✅ (+ ControlNet inpaint) |
| Omni/Editing | ❌ | ❌ | ✅ (different text encoding) |
| Modular pipelines | ❌ | ❌ | ✅ (composable blocks) |
| **OmniBase/Editing** | | | |
| SigLIP2 utilities | ✅ Partial (loading, caching) | ❌ (unreleased) | ✅ (in transformer model) |
| SigLIP2 forward pass | ❌ Not in transformer | ❌ | ✅ (OmniBase components in model) |
| **LoRA** | | | |
| LoRA targets | Attention + FFN | N/A | Attention only (default) |
| LoRA default rank | 32 | N/A | 4 |
| LoRA on noise_refiner | ❌ Excluded | N/A | ✅ Included |
| LoRA alpha handling | Separate parameter | N/A | Baked into weights |
| **Training features** | | | |
| FP16 safety | ✅ Div guards (÷32 FFN, ÷4 attn) | ❌ (bf16/fp32 only) | ❌ |
| ComfyUI compat | ✅ Key conversion + QKV split | ❌ | ❌ |
| FP8 optimization | ✅ (layers, refiners) | ❌ | ❌ |
| Gradient checkpointing | ✅ Per-module | ❌ | ✅ (via accelerate) |
| Block swap/CPU offload | ✅ (up to 28 blocks) | ❌ | ❌ |
| CFG truncation | Via shared modules | ✅ Pipeline-level | ✅ Pipeline-level |
| CFG normalization | Via shared modules | ✅ Pipeline-level | ⚠️ Standard only (not modular) |
| Stochastic sampling | ❌ | ❌ | ✅ (scheduler option) |
| Per-token timesteps | ❌ | ❌ | ✅ (for OmniBase) |

### 8.2 sdbds Fork Additions

The sdbds fork adds OmniBase support:

| Component | Lines Changed | Key Additions |
|-----------|---------------|---------------|
| zimage_model.py | +552 | siglip_embedder, siglip_refiner, select_per_token, omni forward |
| zimage_utils.py | +108 | SigLIP2 loading, image encoder utils |
| zimage_cache_latents.py | +120 | I2V caching, --i2v flag, siglip grid conversion |
| zimage_train_network.py | +75 | omni-mode training path, siglip loading |
| zimage_config.py | +1 | SIGLIP_FEAT_DIM = 1152 |

### 8.3 Gaps to Address (Updated Feb 2026)

**Completed** (since v1.0):
- ~~SigLIP2 utility functions~~ → `load_siglip2_encoder()`, `siglip_last_hidden_to_grid()` in `zimage_utils.py`
- ~~I2V caching~~ → OmniBase caching with `--image_encoder` support in `zimage_cache_latents.py`
- ~~SIGLIP_FEAT_DIM constant~~ → `DEFAULT_TRANSFORMER_SIGLIP_FEAT_DIM = 1152` in `zimage_config.py`
- ~~OmniBase detection~~ → `should_enable_omnibase()`, `infer_siglip_feat_dim()` in `zimage_utils.py`

**Remaining**:
1. **OmniBase Forward Path**: Transformer model does not include `siglip_embedder`/`siglip_refiner` in its `forward()`. The utility functions exist but the model architecture hasn't been extended. Note: the official Z-Image repo also does NOT contain OmniBase code — it's unreleased. However, diffusers already has OmniBase/SigLIP components in the transformer model.
2. **OmniBase Inference**: Generation script lacks OmniBase/editing inference path.
3. **OmniBase Training**: Training script lacks multi-image batch construction and `noise_mask` passing.
4. **FA3 Attention Backend**: The upstream repo now has full FA3 support (flash_3, flash_varlen_3). Blissful Tuner's training CLI accepts `--flash3` but the shared `attention()` function does not handle it (latent bug — would raise `ValueError`).
5. **MPS Flash Attention**: The upstream repo added Apple Silicon support (2026-01-30 via mps-flash-attn). Not in Blissful Tuner.
6. **torch.compile optimization**: The upstream uses LRU-cached varlen preparation and explicit dispatch (no dict lookups) to avoid dynamo graph breaks. Blissful Tuner's attention module is not specifically optimized for torch.compile.
7. **ControlNet support** *(v2.3)*: Diffusers has full ControlNet architecture (`ZImageControlNetModel`). Not in Blissful Tuner or official repo.
8. **Img2Img / Inpainting pipelines** *(v2.3)*: Diffusers implements both. Blissful Tuner only has T2I inference.
9. **LoRA interoperability** *(v2.3)*: Diffusers LoRAs include noise_refiner layers and bake alpha into weights. Blissful Tuner LoRAs exclude noise_refiner and keep alpha separate. Cross-ecosystem LoRA loading requires conversion.

---

## 9. Integration Recommendations

### 9.1 Priority Order (Updated Feb 2026)

1. ~~**Phase 1 (Low Risk)**: Update configuration constants~~ **DONE**
   - ✅ `DEFAULT_TRANSFORMER_SIGLIP_FEAT_DIM = 1152` in zimage_config.py
   - ✅ All default values verified against official HF configs

2. ~~**Phase 2 (Medium Risk)**: Add SigLIP2 utilities~~ **DONE**
   - ✅ `load_siglip2_encoder()` in zimage_utils.py
   - ✅ `siglip_last_hidden_to_grid()` conversion
   - ✅ `should_enable_omnibase()` / `infer_siglip_feat_dim()` auto-detection

3. **Phase 3 (High Risk)**: Merge model architecture — **REMAINING**
   - Add siglip_embedder, siglip_refiner, siglip_pad_token to `ZImageTransformer2DModel`
   - Implement select_per_token() function
   - Add patchify_and_embed_omni() method
   - Modify forward() with omni branch
   - **CRITICAL**: Ensure backward compatibility with existing LoRAs

4. ~~**Phase 4 (Medium Risk)**: Update caching~~ **PARTIALLY DONE**
   - ✅ OmniBase caching with `--image_encoder` and `--image_encoder_dtype` flags
   - ✅ SigLIP feature caching (keyed as `siglip_{i}_{dtype}`)
   - ✅ Control image latent caching (keyed as `latents_control_{i}_{F}x{H}x{W}_{dtype}`)

5. **Phase 5 (Medium Risk)**: Update training — **REMAINING**
   - Load siglip features in training loop
   - Construct multi-image batches
   - Pass noise_masks to model

### 9.2 Backward Compatibility Strategy

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

### 9.3 BlissfulLogger Integration

When adding new code, use the existing logging pattern:

```python
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")

# Usage
logger.info("Loading SigLIP2 encoder...")
logger.warning("SigLIP2 not available, falling back to T2I only")
```

---

## 10. Performance Benchmarks

### 10.1 Generation Quality Benchmarks

**GenEval (Compositional Generation):**

| Model | Single Obj | Two Obj | Counting | Colors | Position | Color Attrib | Overall | Rank |
|-------|-----------|---------|----------|--------|----------|-------------|---------|------|
| Qwen-Image | 1.00 | 0.93 | 0.82 | 0.92 | 0.71 | 0.73 | **0.87** | 1 |
| Z-Image | 1.00 | 0.91 | 0.74 | 0.93 | 0.62 | 0.83 | **0.84** | 2 (tied) |
| Z-Image-Turbo | 1.00 | 0.85 | 0.76 | 0.89 | 0.65 | 0.78 | **0.82** | 6 |

> Z-Image achieves perfect 1.00 on Single Object and leads on Colors (0.93). Weakest subcategory: Position (0.62).

**DPG-Bench (Detailed Prompt):**

| Model | Global | Entity | Attribute | Relation | Other | Overall | Rank |
|-------|--------|--------|-----------|----------|-------|---------|------|
| Z-Image | 93.39 | 91.22 | 93.16 | 92.22 | 91.52 | **88.14** | 3 |
| Z-Image-Turbo | 91.29 | 89.59 | 90.14 | 92.16 | 88.68 | **84.86** | 7 |

**TIIF Bench (Text-to-Image Instruction Following):**

| Model | Overall | Rank |
|-------|---------|------|
| GPT Image 1 [High] | 89.15 | 1 |
| Qwen-Image | 86.14 | 2 |
| Seedream 3.0 | 86.02 | 3 |
| **Z-Image** | **80.20** | **4** |
| **Z-Image-Turbo** | **77.73** | **5** |

> Notable gap between Z-Image (80.20) and top models (86-89) on instruction following — an area for potential improvement.

**PRISM-Bench English:**
- Z-Image: 75.6 (Rank 5)
- **Z-Image-Turbo: 77.4 (Rank 3)** — Turbo *outperforms* the base model on this English benchmark
- Z-Image-Turbo Composition: 86.2 (highest subcategory)

> **Unusual finding**: The distilled Turbo model exceeds the teacher on PRISM-Bench English (77.4 vs 75.6). This may be relevant for fine-tuning decisions — Turbo may be preferred for certain quality-sensitive use cases.

**PRISM-Bench Chinese:**
- Z-Image: 75.3 (Rank 2)
- Z-Image-Turbo: 75.1 (Rank 3)
- Z-Image Text Rendering: 83.4 (exceptional)
- Z-Image-Turbo Text Rendering: 79.6
- Z-Image Composition: 88.6

> PRISM-Bench evaluated by Qwen2.5-VL-72B as the VLM judge model.

### 10.2 Image Editing Benchmarks

**ImgEdit Benchmark** (instruction-based editing):

| Model | Add | Adjust | Extract | Replace | Remove | Background | Style | Hybrid | Action | Overall | Rank |
|-------|-----|--------|---------|---------|--------|------------|-------|--------|--------|---------|------|
| UniWorld-V2 | 4.62 | 4.42 | 4.78 | 4.63 | 4.40 | 4.58 | 4.58 | 3.53 | 4.80 | **4.49** | 1 |
| Qwen-Image-Edit [2509] | 4.55 | 4.19 | 4.57 | 4.42 | 4.17 | 4.48 | 4.50 | 3.45 | 4.68 | **4.35** | 2 |
| **Z-Image-Edit** | 4.40 | 4.14 | 4.30 | 4.57 | 4.13 | 4.57 | 4.85 | 3.63 | 4.50 | **4.30** | **3** |
| GPT-Image-1 [High] | - | - | - | - | - | - | - | - | - | 4.20 | 5 |
| FLUX.1 Kontext [Pro] | - | - | - | - | - | - | - | - | - | 4.00 | 6 |

> Z-Image-Edit's **strongest**: Style (4.85), Background (4.57), Replace (4.57). **Weakest**: Hybrid (3.63). Outperforms both GPT-Image-1 and FLUX.1 Kontext Pro.

**GEdit-Bench** (bilingual editing):

| Model | EN G_SC | EN G_PQ | EN G_O | CN G_SC | CN G_PQ | CN G_O | Rank |
|-------|---------|---------|--------|---------|---------|--------|------|
| UniWorld-V2 | - | - | 7.83 | - | - | - | 1 |
| Qwen-Image-Edit [2509] | - | - | 7.54 | - | - | - | 2 |
| **Z-Image-Edit** | 8.11 | 7.72 | **7.57** | 8.03 | 7.80 | **7.54** | **3** |

> Z-Image-Edit demonstrates strong bilingual editing: Chinese performance (7.54) nearly matches English (7.57), consistent with the bilingual text encoder design.

**Z-Image-Edit Capabilities** (from qualitative evaluation):
- Composite multi-instruction editing (simultaneous background swap + object insertion + removal)
- Bounding-box-based location constraints for targeted text modification
- Identity preservation during scene transformation
- The editing model is derived from the omni-pretraining paradigm (Phase 2)

### 10.3 Efficiency & Competitive Landscape

| Model | Parameters | Steps | Inference Cost | Notes |
|-------|------------|-------|----------------|-------|
| Z-Image-Turbo | 6.15B | 8 | $5.0/1K images | Sub-second on H800 (requires FA3 + torch.compile) |
| Z-Image | 6.15B | 28-50 | ~$25-30/1K images | Full CFG, higher quality |
| Lumina-Image 2.0 | ~2B | ~28 | Lower | Lightweight competitor |
| Qwen-Image | 20B | ~50 | Higher | Leads GenEval (0.87) |
| FLUX.2 | 32B | ~28 | Higher | |
| Seedream 3.0 | - | - | - | Tied for Rank 2 GenEval |
| Hunyuan-3.0 | 80B | ~50 | Much higher | |
| Nano Banana Pro | - | - | - | Google Gemini 3 Pro; rivals Z-Image text rendering |
| Imagen 4 Ultra | - | - | - | Closed-source, qualitative competitor |

### 10.4 Training Costs (Reference)

| Phase | GPU Hours | Cost (est.) |
|-------|-----------|-------------|
| Low-res pre-training | 147.5K H800 | $295K |
| Omni pre-training | 142.5K H800 | $285K |
| Post-training | 24K H800 | $48K |
| **Total** | **314K H800** | **$628K** |

---

## Appendix A: File Structure Reference

```
Official Z-Image GitHub (zimage-native v0.1.0, Python >=3.10, PyTorch >=2.5.0):
├── pyproject.toml                 # Package: zimage-native, deps: torch, transformers, safetensors, loguru
├── inference.py                   # Single image generation (defaults to Z-Image-Turbo)
├── batch_inference.py             # Batch from prompts file (env: PROMPTS_FILE, ZIMAGE_ATTENTION)
├── src/
│   ├── __init__.py                # Exports: ZImageTransformer2DModel, generate, load_from_local_dir
│   ├── config/
│   │   ├── __init__.py            # Re-exports all 47 constants
│   │   ├── model.py               # Architecture constants (⚠️ VAE defaults are SD1.x, not Z-Image)
│   │   ├── inference.py           # Inference defaults (8 steps, 0.0 guidance, 512 max_seq)
│   │   └── manifests/
│   │       ├── README.md          # Manifest format docs
│   │       └── z-image-turbo.txt  # MD5 checksums for Turbo weights
│   ├── tools/
│   │   └── generate_manifest.py   # CLI tool: generate weight manifests
│   ├── zimage/
│   │   ├── transformer.py         # ZImageTransformer2DModel (S3-DiT)
│   │   ├── pipeline.py            # generate() with CFG truncation + normalization
│   │   ├── scheduler.py           # FlowMatchEulerDiscreteScheduler (standalone)
│   │   └── autoencoder.py         # AutoencoderKL (decode-only, from FLUX)
│   └── utils/
│       ├── attention.py           # 8-backend dispatch (FA2/FA3/MPS Flash/SDPA)
│       ├── import_utils.py        # Feature detection (flash_attn, torch version)
│       ├── loader.py              # Full model loading pipeline (transformer, VAE, Qwen3, tokenizer)
│       └── helpers.py             # Model download/verify, memory stats

HuggingFace Model (verified from Tongyi-MAI/Z-Image & Z-Image-Turbo):
├── model_index.json                         # ZImagePipeline structure
├── README.md                                # Model card
├── transformer/
│   ├── config.json                          # ZImageTransformer2DModel config
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── diffusion_pytorch_model-*.safetensors  # 2 files (Z-Image) / 3 files (Turbo)
├── text_encoder/
│   ├── config.json                          # Qwen3ForCausalLM config
│   ├── generation_config.json               # Sampling defaults (temp=0.6, top_k=20)
│   ├── model.safetensors.index.json
│   └── model-*.safetensors                  # 3 files (~8 GB)
├── tokenizer/
│   ├── tokenizer_config.json                # Qwen2Tokenizer (model_max_length=131072)
│   ├── tokenizer.json                       # Full tokenizer definition
│   ├── vocab.json                           # 151936 vocab entries
│   └── merges.txt                           # BPE merge rules
├── scheduler/
│   └── scheduler_config.json                # FlowMatchEuler (shift=6.0 or 3.0)
├── vae/
│   └── config.json                          # AutoencoderKL (Flux-derived)
└── assets/                                  # Model card images

Blissful Tuner Z-Image Implementation:
├── src/musubi_tuner/zimage/
│   ├── zimage_config.py           # All architecture constants + OmniBase config
│   ├── zimage_model.py            # ZImageTransformer2DModel, loading, ComfyUI compat
│   ├── zimage_autoencoder.py      # AutoencoderKL (Flux-based, always fp32)
│   └── zimage_utils.py            # Text encoding, scheduling, SigLIP2 loading
├── src/musubi_tuner/
│   ├── zimage_cache_latents.py        # Latent + mask + OmniBase caching
│   ├── zimage_cache_text_encoder_outputs.py  # Qwen3 text cache
│   ├── zimage_train_network.py        # ZImageNetworkTrainer (LoRA)
│   ├── zimage_train.py               # ZImageTrainer (full fine-tune)
│   └── zimage_generate_image.py       # Inference (single/batch/interactive)
├── src/musubi_tuner/networks/
│   └── lora_zimage.py                # LoRA targets: ZImageTransformerBlock
│                                      # Excludes: _modulation, _refiner

sdbds Fork (OmniBase additions, for reference):
├── src/musubi_tuner/zimage/
│   ├── zimage_model.py        # +552 lines (OmniBase forward path)
│   ├── zimage_utils.py        # +108 lines (SigLIP2 utilities)
│   └── zimage_config.py       # +1 line (SIGLIP_FEAT_DIM)
├── src/musubi_tuner/
│   ├── zimage_cache_latents.py    # +120 lines (I2V caching)
│   └── zimage_train_network.py    # +75 lines (omni training path)

HuggingFace Diffusers Z-Image Files (51 files, v2.3):
├── src/diffusers/models/transformers/
│   └── transformer_z_image.py              # Core transformer (FinalLayer=LayerNorm, not RMSNorm)
├── src/diffusers/models/controlnets/
│   └── controlnet_z_image.py               # ControlNet (zero-init, from_transformer, dict injection)
├── src/diffusers/pipelines/z_image/
│   ├── pipeline_z_image.py                 # Base T2I
│   ├── pipeline_z_image_img2img.py         # Img2Img
│   ├── pipeline_z_image_inpaint.py         # Inpaint (latent-space blending)
│   ├── pipeline_z_image_controlnet.py      # ControlNet
│   ├── pipeline_z_image_controlnet_inpaint.py  # ControlNet + Inpaint (33ch)
│   └── pipeline_z_image_omni.py            # OmniBase (different text encoding)
├── src/diffusers/modular_pipelines/z_image/  # Composable block system
├── src/diffusers/loaders/
│   └── lora_conversion_utils.py            # LoRA key conversion (alpha baking)
├── examples/dreambooth/
│   └── train_dreambooth_lora_z_image.py    # DreamBooth training (rank=4, attn-only)
├── src/diffusers/schedulers/
│   └── scheduling_flow_match_euler_discrete.py  # Shared scheduler (stochastic, per_token)
└── tests/                                  # torch.compile broken, RoPE cache issues
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

### B.2 AdaLN Modulation (Actual Implementation — Corrected v2.3)

```python
def forward(self, x, t_emb, freqs, attn_mask=None):
    # Low-rank decomposition: t_emb is already the shared 256-dim output
    # from the layer-agnostic t_embedder (computed once, passed to all layers).
    # adaLN_modulation is the per-layer up-projection: Linear(256, 4 * 3840)
    # ⚠️ NO activation before chunking (unlike FinalLayer which has SiLU)
    mod = self.adaLN_modulation(t_emb)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)

    # Attention block (scale only, no shift; sandwich norm; tanh gate)
    # CORRECTED (v2.3): norm2 wraps the OUTPUT of attention, not the input
    x_norm = self.attention_norm1(x)
    x_norm = x_norm * (1 + scale_msa.unsqueeze(1))  # Scale only (pre-attention)
    attn_out = self.attention(x_norm, freqs, attn_mask)
    attn_out = self.attention_norm2(attn_out)         # Sandwich norm (post-attention)
    x = x + torch.tanh(gate_msa).unsqueeze(1) * attn_out

    # FFN block (same corrected pattern)
    x_norm = self.ffn_norm1(x)
    x_norm = x_norm * (1 + scale_mlp.unsqueeze(1))
    ffn_out = self.feed_forward(x_norm)
    ffn_out = self.ffn_norm2(ffn_out)                 # Sandwich norm (post-FFN)
    x = x + torch.tanh(gate_mlp).unsqueeze(1) * ffn_out

    return x

# FinalLayer forward (different from blocks):
def final_forward(self, x, t_emb):
    # FinalLayer uses nn.LayerNorm (NOT RMSNorm)
    # FinalLayer adaLN HAS SiLU activation
    scale = self.adaLN_modulation(t_emb)  # Sequential(SiLU, Linear) → [B, dim]
    x = self.norm_final(x) * (1 + scale.unsqueeze(1))
    x = self.linear(x)                    # Project to pixel space
    return x
```

> **Unmodulated variant** (context_refiner blocks): When `modulation=False`, the block skips all adaLN logic — just norm → attention → residual, norm → FFN → residual.

### B.3 OmniBase Per-Token Selection

```python
def select_per_token(noisy_emb, clean_emb, noise_mask, seq_len):
    """
    Select between noisy and clean embeddings per token.

    Args:
        noisy_emb: [B, 4*D] modulation for noisy tokens (scale+gate for attn+FFN)
        clean_emb: [B, 4*D] modulation for clean tokens
        noise_mask: [B, N] where 1=noisy, 0=clean
        seq_len: Sequence length N

    Returns:
        [B, N, 4*D] selected modulation per token
    """
    mask = noise_mask.unsqueeze(-1)  # [B, N, 1]
    noisy_expanded = noisy_emb.unsqueeze(1).expand(-1, seq_len, -1)
    clean_expanded = clean_emb.unsqueeze(1).expand(-1, seq_len, -1)
    return torch.where(mask == 1, noisy_expanded, clean_expanded)
```

### B.4 Forward Pass Flow (Blissful Tuner)

```python
# Simplified forward from ZImageTransformer2DModel.forward():
def forward(self, x, t, cap_feats, cap_mask, freqs_cis=None):
    # 1. Scale timestep: 0-1 → 0-1000
    t = t * self.t_scale  # t_scale = 1000.0

    # 2. Timestep embedding (256-dim bottleneck)
    t_emb = self.t_embedder(t)  # [B] → [B, 256]

    # 3. Patchify: [B, 16, 1, H, W] → [B, N_img, 64] → embed → [B, N_img, 3840]
    x = self.all_x_embedder["2-1"](patchified_x)

    # 4. Compute 3D RoPE frequencies
    freqs = self.rope_embedder(position_ids)  # [N_total, head_dim]

    # 5. Noise refiner (2 blocks, modulated, image tokens only)
    for layer in self.noise_refiner:
        x_img = layer(x_img, t_emb, freqs_img)

    # 6. Caption embedding + context refiner (2 blocks, unmodulated)
    cap = self.cap_embedder(cap_feats)  # [B, N_cap, 2560] → [B, N_cap, 3840]
    for layer in self.context_refiner:
        cap = layer(cap, None, freqs_cap)  # No t_emb

    # 7. Concatenate [image_tokens, caption_tokens]
    x = torch.cat([x_img, cap], dim=1)

    # 8. Main transformer (30 blocks, modulated)
    for layer in self.layers:
        x = layer(x, t_emb, freqs_all, attn_mask)

    # 9. Final layer (modulated projection)
    x = self.all_final_layer["2-1"](x[:, :N_img], t_emb)

    # 10. Unpatchify back to [B, 16, 1, H, W]
    return unpatchified_x
```

### B.5 Latent Normalization

```python
# Training: encode latents (scale_shift_latents in zimage_train_network.py)
normalized = (raw_latents - SHIFT_FACTOR) * SCALING_FACTOR
#          = (raw_latents - 0.1159) * 0.3611

# Inference: decode latents (shift_scale_latents_for_decode in zimage_utils.py)
denormalized = (normalized / SCALING_FACTOR) + SHIFT_FACTOR
#            = (normalized / 0.3611) + 0.1159
```

---

## 11. Weight Storage & ComfyUI Compatibility

### 11.1 HuggingFace Weight Layout

Both Z-Image variants use identical architecture but different weight storage dtypes:

```
Tongyi-MAI/Z-Image/
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00002.safetensors
│   └── diffusion_pytorch_model-00002-of-00002.safetensors
│       └── Total: 12.31 GB (~6.15B params × 2 bytes/bfloat16)
├── text_encoder/
│   ├── config.json
│   ├── generation_config.json
│   ├── model-00001-of-00003.safetensors
│   ├── model-00002-of-00003.safetensors
│   └── model-00003-of-00003.safetensors
│       └── Total: 8.04 GB (~4B params)
├── tokenizer/
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── vocab.json
│   └── merges.txt
├── scheduler/
│   └── scheduler_config.json (shift=6.0)
└── vae/
    └── config.json

Tongyi-MAI/Z-Image-Turbo/
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00003.safetensors
│   ├── diffusion_pytorch_model-00002-of-00003.safetensors
│   └── diffusion_pytorch_model-00003-of-00003.safetensors
│       └── Total: 24.62 GB (~6.15B params × 4 bytes/float32)
├── text_encoder/  (identical to Z-Image)
├── tokenizer/     (identical to Z-Image)
├── scheduler/
│   └── scheduler_config.json (shift=3.0)
└── vae/           (identical to Z-Image)
```

### 11.2 Pipeline Component Classes

From `model_index.json` (identical for both variants except diffusers version):

| Component | Class | Library |
|-----------|-------|---------|
| Pipeline | `ZImagePipeline` | diffusers |
| Transformer | `ZImageTransformer2DModel` | diffusers |
| Text Encoder | `Qwen3Model` | transformers |
| Tokenizer | `Qwen2Tokenizer` | transformers |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | diffusers |
| VAE | `AutoencoderKL` | diffusers |

> **Diffusers version**: Z-Image uses `0.37.0.dev0`, Z-Image-Turbo uses `0.36.0.dev0`.

### 11.3 ComfyUI Weight Key Conversion

Blissful Tuner's `load_zimage_model()` handles ComfyUI-format checkpoints with automatic key conversion:

```python
# Key renames (ComfyUI → Blissful Tuner internal):
COMFYUI_KEY_MAPPINGS = {
    "final_layer.linear":           "all_final_layer.2-1.linear",
    "final_layer.adaLN_modulation": "all_final_layer.2-1.adaLN_modulation",
    "x_embedder.weight":            "all_x_embedder.2-1.weight",
    "x_embedder.bias":              "all_x_embedder.2-1.bias",
    ".attention.out.weight":        ".attention.to_out.0.weight",
    ".attention.out.bias":          ".attention.to_out.0.bias",
    ".attention.k_norm.weight":     ".attention.norm_k.weight",
    ".attention.q_norm.weight":     ".attention.norm_q.weight",
}

# Fused QKV splitting (ComfyUI stores fused, Blissful Tuner stores separate):
# "layers.N.attention.qkv.weight" → split into:
#   "layers.N.attention.to_q.weight"
#   "layers.N.attention.to_k.weight"
#   "layers.N.attention.to_v.weight"
# Split dimensions: [n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim]
```

### 11.4 FP8 Optimization Targets

```python
# Keys targeted for FP8 quantization:
FP8_OPTIMIZATION_TARGET_KEYS = ["layers.", "noise_refiner.", "context_refiner."]

# Keys excluded from FP8 (must remain full precision):
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["_modulation", ".norm_", "_norm"]
```

### 11.5 VAE ComfyUI Compatibility

The VAE loading also handles ComfyUI key conversion:
```python
# ComfyUI → Blissful Tuner VAE key renames:
"decoder.norm_out."  → "decoder.conv_norm_out."
# Block numbering remapped
# Conv2d(1×1) converted to Linear for attention weights
```

---

*Document generated from analysis of the Z-Image Technical Report (arXiv), official Z-Image GitHub source code (commit 26f23ed), HuggingFace configs (`Tongyi-MAI/Z-Image`, `Tongyi-MAI/Z-Image-Turbo`), HuggingFace diffusers repository (51 Z-Image files: transformer, 6 pipelines, ControlNet, LoRA conversion, modular pipelines, scheduler, tests), and Blissful Tuner implementation audit. Updated 2026-02-17 (v2.3).*
