# Z-Image OmniBase Integration Plan

**Version**: 1.0
**Created**: 2026-01-27
**Status**: Draft - Awaiting Review
**Estimated Total Effort**: 12-18 hours

---

## Executive Summary

This plan outlines the integration of Z-Image OmniBase (image editing) capabilities into Blissful Tuner, based on analysis of:
- Official Z-Image GitHub repository
- Z-Image Technical Report (3 parts)
- HuggingFace model configuration
- sdbds/musubi-tuner fork implementation

**Goal**: Enable training LoRAs for Z-Image-Omni-Base with image editing capabilities while maintaining full backward compatibility with existing Z-Image-Turbo LoRA training.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Phase 1: Configuration Alignment](#2-phase-1-configuration-alignment)
3. [Phase 2: SigLIP2 Infrastructure](#3-phase-2-siglip2-infrastructure)
4. [Phase 3: Model Architecture Extensions](#4-phase-3-model-architecture-extensions)
5. [Phase 4: Caching Pipeline Updates](#5-phase-4-caching-pipeline-updates)
6. [Phase 5: Training Loop Integration](#6-phase-5-training-loop-integration)
7. [Phase 6: Testing & Validation](#7-phase-6-testing-validation)
8. [Optional Enhancements](#8-optional-enhancements)
9. [Risk Assessment](#9-risk-assessment)
10. [Success Criteria](#10-success-criteria)
11. [Rollback Plan](#11-rollback-plan)

---

## 1. Prerequisites

### 1.1 Dependencies to Verify/Install

```bash
# Required for SigLIP2 support
pip install transformers>=4.51.0

# Verify PyTorch version (2.5+ for SDPA improvements)
python -c "import torch; print(torch.__version__)"
```

### 1.2 Files to Back Up Before Starting

```
src/musubi_tuner/zimage/
├── zimage_config.py      # Will be modified
├── zimage_model.py       # Major modifications
└── zimage_utils.py       # Will be extended

src/musubi_tuner/
├── zimage_cache_latents.py    # Will be extended
└── zimage_train_network.py    # Will be extended
```

### 1.3 Reference Materials

| Resource | Location | Purpose |
|----------|----------|---------|
| Integration Reference | `docs/z-image-integration-reference.md` | Architecture details |
| Official Z-Image | `~/Downloads/Z-Image/` | Reference implementation |
| Technical Report | `~/Downloads/Z-Image-Technical-Report/` | Training methodology |
| HuggingFace Config | `~/Downloads/Z-IMAGE-HUGGINGFACE/` | Configuration values |
| sdbds Fork | `~/musubi-tuner-forks/sdbds/` | OmniBase implementation |

---

## 2. Phase 1: Configuration Alignment

**Effort**: 30-45 minutes
**Risk**: Low
**Files Modified**: 1

### 2.1 Objectives

- [ ] Add OmniBase-specific configuration constants
- [ ] Verify existing constants match official values
- [ ] Ensure backward compatibility flag exists

### 2.2 Changes to `zimage_config.py`

```python
# ADD: OmniBase/SigLIP2 configuration
DEFAULT_TRANSFORMER_SIGLIP_FEAT_DIM = 1152  # SigLIP2 hidden size

# VERIFY: These should match official config
DEFAULT_TRANSFORMER_DIM = 3840
DEFAULT_TRANSFORMER_N_LAYERS = 30
DEFAULT_TRANSFORMER_N_REFINER_LAYERS = 2
DEFAULT_TRANSFORMER_N_HEADS = 30
DEFAULT_TRANSFORMER_N_KV_HEADS = 30
DEFAULT_TRANSFORMER_NORM_EPS = 1e-5
DEFAULT_TRANSFORMER_QK_NORM = True
DEFAULT_TRANSFORMER_CAP_FEAT_DIM = 2560
DEFAULT_TRANSFORMER_T_SCALE = 1000.0

# VERIFY: RoPE configuration
ROPE_THETA = 256.0
ROPE_AXES_DIMS = [32, 48, 48]  # [time, height, width]
ROPE_AXES_LENS = [1536, 512, 512]

# VERIFY: Scheduler
DEFAULT_SCHEDULER_SHIFT = 6.0
DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS = 1000

# ADD: Sequence padding multiple
SEQ_MULTI_OF = 32
```

### 2.3 Verification Steps

1. Compare each constant against `~/Downloads/Z-IMAGE-HUGGINGFACE/transformer/config.json`
2. Compare scheduler config against `~/Downloads/Z-IMAGE-HUGGINGFACE/scheduler/scheduler_config.json`
3. Run existing Z-Image training to ensure no regression

### 2.4 Acceptance Criteria

- [ ] All constants match official HuggingFace config
- [ ] Existing Z-Image LoRA training still works
- [ ] No import errors

---

## 3. Phase 2: SigLIP2 Infrastructure

**Effort**: 45-60 minutes
**Risk**: Low
**Files Modified**: 1

### 3.1 Objectives

- [ ] Add SigLIP2 model loading utilities
- [ ] Implement graceful fallback when unavailable
- [ ] Add spatial grid conversion function

### 3.2 Changes to `zimage_utils.py`

```python
# ADD: Imports with graceful fallback
import math
from typing import Optional, Tuple

try:
    from transformers import (
        Siglip2VisionModel,
        Siglip2Processor,
        Siglip2VisionConfig,
    )
    SIGLIP2_AVAILABLE = True
except ImportError:
    Siglip2VisionModel = None
    Siglip2Processor = None
    Siglip2VisionConfig = None
    SIGLIP2_AVAILABLE = False

# ADD: SigLIP2 configuration
SIGLIP2_CONFIG = {
    "hidden_size": 1152,
    "intermediate_size": 4608,
    "num_hidden_layers": 27,
    "num_attention_heads": 18,
    "image_size": 256,
    "patch_size": 16,
}

# ADD: Loading function
def load_image_encoders(
    image_encoder_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load SigLIP2 vision encoder and processor.

    Returns:
        Tuple of (vision_model, processor) or (None, None) if unavailable
    """
    if not SIGLIP2_AVAILABLE:
        logger.warning("SigLIP2 not available. Install transformers>=4.51.0")
        return None, None

    try:
        processor = Siglip2Processor.from_pretrained(image_encoder_path)
        vision_model = Siglip2VisionModel.from_pretrained(
            image_encoder_path,
            torch_dtype=dtype,
        ).to(device)
        vision_model.eval()
        logger.info(f"Loaded SigLIP2 encoder from {image_encoder_path}")
        return vision_model, processor
    except Exception as e:
        logger.error(f"Failed to load SigLIP2: {e}")
        return None, None

# ADD: Spatial grid conversion
def siglip_last_hidden_to_grid(
    last_hidden_state: torch.Tensor,
) -> torch.Tensor:
    """
    Convert SigLIP2 last_hidden_state [num_tokens, C] to spatial grid [H, W, C].

    Handles both perfect square token counts and CLS-token-removed counts.
    """
    num_tokens, channels = last_hidden_state.shape

    # Try perfect square first
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size == num_tokens:
        return last_hidden_state.reshape(grid_size, grid_size, channels)

    # Try with CLS token removed (num_tokens = grid_size^2 - 1)
    grid_size = int(math.sqrt(num_tokens + 1))
    if grid_size * grid_size - 1 == num_tokens:
        # Pad with zeros for missing CLS position (or handle differently)
        logger.debug(f"SigLIP tokens suggest CLS removed: {num_tokens} -> {grid_size}x{grid_size}")
        return last_hidden_state.reshape(grid_size, grid_size - 1, channels)

    raise ValueError(f"Cannot reshape {num_tokens} tokens to square grid")
```

### 3.3 Verification Steps

1. Test import with and without transformers>=4.51.0
2. Test loading a SigLIP2 checkpoint (if available)
3. Test grid conversion with mock data

### 3.4 Acceptance Criteria

- [ ] No import errors when SigLIP2 unavailable
- [ ] `SIGLIP2_AVAILABLE` flag correctly set
- [ ] `load_image_encoders()` returns (None, None) gracefully
- [ ] Grid conversion produces correct shapes

---

## 4. Phase 3: Model Architecture Extensions

**Effort**: 3-4 hours
**Risk**: High (core model changes)
**Files Modified**: 1

### 4.1 Objectives

- [ ] Add SigLIP feature processing modules (optional)
- [ ] Implement per-token noise selection
- [ ] Add omni-mode forward path
- [ ] **Maintain full backward compatibility**

### 4.2 Changes to `zimage_model.py`

#### 4.2.1 New Helper Function

```python
def select_per_token(
    noisy_emb: torch.Tensor,
    clean_emb: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    Select between noisy and clean embeddings per token based on noise_mask.

    Args:
        noisy_emb: [B, D] modulation for noisy tokens
        clean_emb: [B, D] modulation for clean tokens
        noise_mask: [B, N] where 1=noisy (target), 0=clean (reference)
        seq_len: Sequence length N

    Returns:
        [B, N, D] per-token selected modulation
    """
    mask = noise_mask.unsqueeze(-1)  # [B, N, 1]
    noisy_expanded = noisy_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, N, D]
    clean_expanded = clean_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, N, D]
    return torch.where(mask == 1, noisy_expanded, clean_expanded)
```

#### 4.2.2 ZImageTransformer2DModel Extensions

```python
class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        # ... existing parameters ...
        siglip_feat_dim: Optional[int] = None,  # NEW: None = no OmniBase
    ):
        super().__init__()
        # ... existing initialization ...

        # NEW: OmniBase components (only if siglip_feat_dim provided)
        self.siglip_feat_dim = siglip_feat_dim
        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=norm_eps),
                nn.Linear(siglip_feat_dim, dim, bias=False),
            )
            self.siglip_refiner = nn.ModuleList([
                ZImageTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                )
                for _ in range(2)  # 2 refiner layers
            ])
            self.siglip_pad_token = nn.Parameter(torch.zeros(1, dim))
            logger.info(f"OmniBase enabled with SigLIP dim={siglip_feat_dim}")
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

    def patchify_and_embed_omni(
        self,
        images: List[torch.Tensor],
        siglip_feats: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process multiple images with corresponding SigLIP features for OmniBase.

        Args:
            images: List of image latents [B, C, H, W] per image
            siglip_feats: List of SigLIP features [H_sig, W_sig, D_sig] or None

        Returns:
            Tuple of (embeddings, position_ids, attention_mask)
        """
        # Implementation following sdbds pattern
        # ... (detailed implementation)
        pass

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: Optional[torch.Tensor] = None,
        # NEW: OmniBase parameters
        siglip_feats: Optional[List[torch.Tensor]] = None,
        image_noise_mask: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass supporting both standard and OmniBase modes.

        Standard mode (backward compatible):
            x: [B, C, F, H, W] single image tensor
            siglip_feats: None
            image_noise_mask: None

        OmniBase mode:
            x: List of image tensors per batch item
            siglip_feats: List of SigLIP features per batch item
            image_noise_mask: List of noise masks per batch item
        """
        # Detect mode
        omni_mode = isinstance(x, list) and siglip_feats is not None

        if omni_mode:
            return self._forward_omni(x, t, cap_feats, cap_mask, siglip_feats, image_noise_mask)
        else:
            return self._forward_standard(x, t, cap_feats, cap_mask)

    def _forward_standard(self, x, t, cap_feats, cap_mask):
        """Original forward path - NO CHANGES to preserve compatibility."""
        # ... existing implementation unchanged ...
        pass

    def _forward_omni(self, x_list, t, cap_feats, cap_mask, siglip_feats, noise_masks):
        """NEW: OmniBase forward path with dual-branch modulation."""
        if self.siglip_feat_dim is None:
            raise RuntimeError("OmniBase forward called but siglip_feat_dim is None")

        # Implementation following sdbds pattern with improvements
        # ... (detailed implementation)
        pass
```

#### 4.2.3 ZImageTransformerBlock Extensions

```python
class ZImageTransformerBlock(nn.Module):
    def _forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        # NEW: OmniBase parameters
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with optional per-token modulation selection.

        Standard mode: adaln_input provided, noise_mask is None
        OmniBase mode: adaln_noisy/clean provided, noise_mask selects per-token
        """
        if noise_mask is not None and adaln_noisy is not None:
            # OmniBase: select modulation per token
            seq_len = x.shape[1]

            # Get modulation for noisy and clean branches
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)

            # Chunk into components
            (shift_msa_n, scale_msa_n, gate_msa_n,
             shift_mlp_n, scale_mlp_n, gate_mlp_n) = mod_noisy.chunk(6, dim=-1)
            (shift_msa_c, scale_msa_c, gate_msa_c,
             shift_mlp_c, scale_mlp_c, gate_mlp_c) = mod_clean.chunk(6, dim=-1)

            # Select per token
            scale_msa = select_per_token(scale_msa_n, scale_msa_c, noise_mask, seq_len)
            shift_msa = select_per_token(shift_msa_n, shift_msa_c, noise_mask, seq_len)
            gate_msa = select_per_token(gate_msa_n, gate_msa_c, noise_mask, seq_len)
            scale_mlp = select_per_token(scale_mlp_n, scale_mlp_c, noise_mask, seq_len)
            shift_mlp = select_per_token(shift_mlp_n, shift_mlp_c, noise_mask, seq_len)
            gate_mlp = select_per_token(gate_mlp_n, gate_mlp_c, noise_mask, seq_len)
        else:
            # Standard mode: single modulation for all tokens
            mod = self.adaLN_modulation(adaln_input)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
            # Expand for broadcasting
            scale_msa = scale_msa.unsqueeze(1)
            shift_msa = shift_msa.unsqueeze(1)
            # ... etc

        # Rest of forward unchanged
        # ... attention and FFN with modulation ...
```

### 4.3 Verification Steps

1. **Backward compatibility test**: Load existing Z-Image checkpoint, run inference
2. **OmniBase initialization test**: Create model with `siglip_feat_dim=1152`
3. **Shape verification**: Test forward pass shapes with mock data
4. **Gradient flow test**: Verify gradients flow through new modules

### 4.4 Acceptance Criteria

- [ ] Existing Z-Image LoRAs load and work unchanged
- [ ] `siglip_feat_dim=None` produces identical behavior to before
- [ ] OmniBase modules only created when `siglip_feat_dim` provided
- [ ] Forward pass produces correct output shapes in both modes
- [ ] No memory increase when OmniBase disabled

---

## 5. Phase 4: Caching Pipeline Updates

**Effort**: 1.5-2 hours
**Risk**: Medium
**Files Modified**: 1

### 5.1 Objectives

- [ ] Add `--i2v` flag for image-to-video/editing mode
- [ ] Add `--image_encoder` argument for SigLIP2 path
- [ ] Implement control image latent caching
- [ ] Implement SigLIP feature caching
- [ ] Auto-detect I2V mode from dataset

### 5.2 Changes to `zimage_cache_latents.py`

```python
# ADD: New arguments
def setup_parser():
    parser = argparse.ArgumentParser()
    # ... existing arguments ...

    # NEW: OmniBase/I2V arguments
    parser.add_argument(
        "--i2v",
        action="store_true",
        help="Enable I2V/editing mode: cache control images and SigLIP features",
    )
    parser.add_argument(
        "--image_encoder",
        type=str,
        default=None,
        help="Path to SigLIP2 encoder for I2V mode",
    )
    parser.add_argument(
        "--auto_i2v",
        action="store_true",
        default=True,
        help="Auto-detect I2V mode if control images found in dataset",
    )
    return parser

# ADD: SigLIP feature extraction
def extract_siglip_features(
    vision_model,
    processor,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Extract SigLIP2 features from a control image."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vision_model(**inputs)
        last_hidden = outputs.last_hidden_state[0]  # Remove batch dim

    # Convert to spatial grid
    grid = siglip_last_hidden_to_grid(last_hidden)
    return grid

# MODIFY: encode_and_save_batch
def encode_and_save_batch(
    # ... existing parameters ...
    i2v_mode: bool = False,
    vision_model = None,
    processor = None,
):
    """
    Encode images/videos and save to cache.

    In I2V mode, also caches:
    - latents_control_{i}_{F}x{H}x{W}_{dtype}: Control image latents
    - siglip_{i}_{dtype}: SigLIP features for each control image
    """
    # ... existing VAE encoding ...

    if i2v_mode and control_images is not None:
        for i, ctrl_img in enumerate(control_images):
            # Encode control image with VAE
            ctrl_latent = encode_image(vae, ctrl_img)
            cache_dict[f"latents_control_{i}_{shape_str}"] = ctrl_latent

            # Extract SigLIP features if encoder available
            if vision_model is not None:
                siglip_feat = extract_siglip_features(
                    vision_model, processor, ctrl_img, device
                )
                cache_dict[f"siglip_{i}_{dtype_str}"] = siglip_feat

    # Save cache
    save_safetensors(cache_dict, cache_path)
```

### 5.3 Cache File Format

```
# Standard mode (existing)
{name}_{W}x{H}_zimage.safetensors:
├── latents_{F}x{H}x{W}_{dtype}     # Target image latents

# I2V/OmniBase mode (new)
{name}_{W}x{H}_zimage.safetensors:
├── latents_{F}x{H}x{W}_{dtype}           # Target image latents
├── latents_control_0_{F}x{H}x{W}_{dtype} # Control image 0 latents
├── latents_control_1_{F}x{H}x{W}_{dtype} # Control image 1 latents (if multiple)
├── siglip_0_{dtype}                       # SigLIP features for control 0
└── siglip_1_{dtype}                       # SigLIP features for control 1
```

### 5.4 Dataset Configuration

```toml
# I2V dataset configuration example
[[datasets]]
resolution = [1024, 1024]
batch_size = 1

[[datasets.subsets]]
image_directory = "/path/to/target_images"
control_directory = "/path/to/control_images"  # NEW: matched by filename
caption_extension = ".txt"
```

### 5.5 Verification Steps

1. Run caching without `--i2v` flag, verify identical output
2. Run caching with `--i2v` flag, verify new keys in cache
3. Verify SigLIP features have correct shape [H, W, 1152]
4. Test auto-detection of I2V mode

### 5.6 Acceptance Criteria

- [ ] Standard caching produces identical files
- [ ] I2V caching includes control latents and SigLIP features
- [ ] Auto-detection correctly identifies I2V datasets
- [ ] Error handling for missing SigLIP2 encoder

---

## 6. Phase 5: Training Loop Integration

**Effort**: 2-3 hours
**Risk**: Medium
**Files Modified**: 1

### 6.1 Objectives

- [ ] Load SigLIP features from cache
- [ ] Construct multi-image batches for OmniBase
- [ ] Build noise masks (0=reference, 1=target)
- [ ] Pass OmniBase parameters to model

### 6.2 Changes to `zimage_train_network.py`

```python
# ADD: OmniBase batch construction
def prepare_omni_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List, List, List]:
    """
    Prepare OmniBase batch from cached data.

    Returns:
        x_list: List of image tensor lists per batch item
        siglip_list: List of SigLIP feature lists per batch item
        noise_mask_list: List of noise masks per batch item
    """
    batch_size = batch["latents"].shape[0]
    x_list = []
    siglip_list = []
    noise_mask_list = []

    for b in range(batch_size):
        images = []
        siglips = []
        masks = []

        # Add control images (noise_mask = 0, clean)
        i = 0
        while f"latents_control_{i}" in batch:
            ctrl_latent = batch[f"latents_control_{i}"][b].to(device, dtype)
            images.append(ctrl_latent)
            masks.append(0)  # Clean

            if f"siglip_{i}" in batch:
                sig = batch[f"siglip_{i}"][b].to(device, dtype)
                siglips.append(sig)
            else:
                siglips.append(None)
            i += 1

        # Add target image (noise_mask = 1, noisy)
        target_latent = batch["latents"][b].to(device, dtype)
        images.append(target_latent)
        masks.append(1)  # Noisy
        siglips.append(None)  # No SigLIP for target

        x_list.append(images)
        siglip_list.append(siglips)
        noise_mask_list.append(torch.tensor(masks, device=device))

    return x_list, siglip_list, noise_mask_list

# MODIFY: Training step
def train_step(
    batch,
    transformer,
    optimizer,
    # ... other params ...
    omni_mode: bool = False,
):
    """Training step with optional OmniBase mode."""

    if omni_mode:
        # OmniBase training
        x_list, siglip_list, noise_masks = prepare_omni_batch(batch, device, dtype)

        # Add noise only to target images (mask=1)
        noisy_x_list = []
        for x_items, masks in zip(x_list, noise_masks):
            noisy_items = []
            for x, m in zip(x_items, masks):
                if m == 1:  # Target: add noise
                    noise = torch.randn_like(x)
                    noisy_x = t * x + (1 - t) * noise
                    noisy_items.append(noisy_x)
                else:  # Reference: keep clean
                    noisy_items.append(x)
            noisy_x_list.append(noisy_items)

        # Forward pass
        model_pred = transformer(
            x=noisy_x_list,
            t=timesteps,
            cap_feats=text_embeddings,
            siglip_feats=siglip_list,
            image_noise_mask=noise_masks,
        )

        # Loss only on target predictions
        # ... loss computation ...
    else:
        # Standard training (unchanged)
        # ... existing code ...
```

### 6.3 Training Arguments

```python
# ADD: New training arguments
parser.add_argument(
    "--omni_mode",
    action="store_true",
    help="Enable OmniBase training with control images",
)
parser.add_argument(
    "--siglip_feat_dim",
    type=int,
    default=None,
    help="SigLIP feature dimension (1152 for SigLIP2). Enables OmniBase architecture.",
)
```

### 6.4 Verification Steps

1. Run standard training, verify identical behavior
2. Run OmniBase training with cached I2V data
3. Verify loss decreases appropriately
4. Check gradient flow through all new modules

### 6.5 Acceptance Criteria

- [ ] Standard training unchanged
- [ ] OmniBase training runs without errors
- [ ] Loss computation correct (only on target)
- [ ] Noise applied only to target images
- [ ] SigLIP features correctly loaded and used

---

## 7. Phase 6: Testing & Validation

**Effort**: 4-6 hours
**Risk**: N/A (validation phase)

### 7.1 Test Matrix

| Test | Standard Mode | OmniBase Mode |
|------|--------------|---------------|
| Config loading | ✓ | ✓ |
| Model initialization | ✓ | ✓ |
| Forward pass shapes | ✓ | ✓ |
| Backward pass | ✓ | ✓ |
| LoRA application | ✓ | ✓ |
| Caching pipeline | ✓ | ✓ |
| Training step | ✓ | ✓ |
| Checkpoint save/load | ✓ | ✓ |

### 7.2 Regression Tests

```bash
# Test 1: Standard Z-Image training (must work exactly as before)
python zimage_cache_latents.py --dataset_config test_config.toml --vae /path/to/vae
python zimage_train_network.py --dit /path/to/dit --dataset_config test_config.toml

# Test 2: OmniBase caching
python zimage_cache_latents.py --dataset_config test_i2v_config.toml --vae /path/to/vae \
    --i2v --image_encoder /path/to/siglip2

# Test 3: OmniBase training
python zimage_train_network.py --dit /path/to/dit --dataset_config test_i2v_config.toml \
    --omni_mode --siglip_feat_dim 1152
```

### 7.3 Validation Checkpoints

- [ ] Phase 1 complete: Config verified, no regressions
- [ ] Phase 2 complete: SigLIP2 utils work, graceful fallback
- [ ] Phase 3 complete: Model backward compatible, OmniBase mode works
- [ ] Phase 4 complete: Caching produces correct files
- [ ] Phase 5 complete: Training runs end-to-end
- [ ] Phase 6 complete: All tests pass

---

## 8. Optional Enhancements

### 8.1 Attention Backend Abstraction (Future)

Add support for Flash Attention 2/3 with backend selection:

```python
# Backend registry pattern from official Z-Image
ATTENTION_BACKENDS = {
    "native": native_attention,
    "flash": flash_attention_2,
    "flash_3": flash_attention_3,
}

def set_attention_backend(backend: str):
    global _ATTENTION_BACKEND
    _ATTENTION_BACKEND = ATTENTION_BACKENDS[backend]
```

**Benefit**: ~2x inference speedup with FA3

### 8.2 Logit-Normal Timestep Sampling

Implement official training methodology:

```python
def sample_timestep_logit_normal(batch_size: int) -> torch.Tensor:
    """Sample timesteps using logit-normal distribution."""
    u = torch.randn(batch_size)
    t = torch.sigmoid(u)
    return t
```

**Benefit**: Better training efficiency, matches official methodology

### 8.3 OmniBase Inference

Implement generation with control images:

```python
def generate_with_reference(
    transformer,
    vae,
    text_encoder,
    prompt: str,
    reference_images: List[Image.Image],
    **kwargs,
) -> Image.Image:
    """Generate image guided by reference images."""
    # Implementation
    pass
```

**Benefit**: Complete OmniBase support including inference

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing LoRAs | Low | High | Extensive backward compat testing |
| Memory increase | Medium | Medium | Lazy module initialization |
| SigLIP2 unavailable | Medium | Low | Graceful fallback implemented |
| Training instability | Low | Medium | Match official hyperparameters |
| Performance regression | Low | Medium | Benchmark before/after |

### 9.1 High-Risk Areas

1. **Model architecture changes** (Phase 3): Core model modifications
   - Mitigation: Separate `_forward_standard` and `_forward_omni` paths

2. **Cache format changes** (Phase 4): Data pipeline changes
   - Mitigation: New keys only, existing keys unchanged

### 9.2 Dependencies

- `transformers>=4.51.0` for SigLIP2 (optional)
- PyTorch 2.5+ recommended for SDPA improvements

---

## 10. Success Criteria

### 10.1 Must Have

- [ ] Existing Z-Image LoRA training works identically
- [ ] Existing Z-Image LoRAs load and work
- [ ] No memory increase when OmniBase disabled
- [ ] Clear error messages for missing dependencies

### 10.2 Should Have

- [ ] OmniBase training produces functional LoRAs
- [ ] I2V caching works with SigLIP2
- [ ] Auto-detection of I2V mode

### 10.3 Nice to Have

- [ ] Attention backend abstraction
- [ ] Logit-normal timestep sampling
- [ ] OmniBase inference support

---

## 11. Rollback Plan

If integration causes issues:

### 11.1 Quick Rollback

```bash
# Revert all changes
git checkout HEAD~N -- src/musubi_tuner/zimage/
git checkout HEAD~N -- src/musubi_tuner/zimage_cache_latents.py
git checkout HEAD~N -- src/musubi_tuner/zimage_train_network.py
```

### 11.2 Partial Rollback

Each phase can be reverted independently:
- Phase 1: Revert config constants (minimal impact)
- Phase 2: Remove SigLIP utils (no other code depends on it)
- Phase 3: Revert model changes (highest risk, revert first if issues)
- Phase 4: Revert caching changes (existing caches still work)
- Phase 5: Revert training changes (standard training unaffected)

### 11.3 Feature Flag

If needed, add a global flag to disable OmniBase:

```python
# In zimage_config.py
OMNIBASE_ENABLED = False  # Set to True when stable
```

---

## Appendix: File Change Summary

| File | Lines Added | Lines Modified | Lines Removed |
|------|-------------|----------------|---------------|
| `zimage_config.py` | ~5 | ~2 | 0 |
| `zimage_utils.py` | ~80 | ~5 | 0 |
| `zimage_model.py` | ~300 | ~50 | 0 |
| `zimage_cache_latents.py` | ~100 | ~20 | 0 |
| `zimage_train_network.py` | ~80 | ~30 | 0 |
| **Total** | **~565** | **~107** | **0** |

---

*Plan created for Blissful Tuner Z-Image OmniBase integration. Review and adjust as needed before implementation.*
