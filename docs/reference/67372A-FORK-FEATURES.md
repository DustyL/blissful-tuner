# 67372a Fork Features Reference

**Fork URL:** https://github.com/67372a/blissful-tuner
**Local Clone:** `/Users/dustin/musubi-tuner-forks/67372a`
**Divergence:** 52 commits ahead of Sarania/blissful-tuner:main
**Last Reviewed:** 2026-01-30
**Status:** Feature-rich fork with production-ready components

This document catalogs the significant features and changes in the 67372a fork of blissful-tuner that may be candidates for integration or serve as reference implementations.

---

## Quick Comparison Summary

| Feature | blissful-tuner Has? | 67372a Adds Value? | Recommendation |
|---------|---------------------|-------------------|----------------|
| **WandB/TensorBoard** | ✅ Yes (identical) | ❌ No | No action needed |
| **Schedule-Free Optimizers** | ✅ Yes | ⚠️ Adds dependencies only | Optional deps |
| **RamTorch Memory Opt** | ❌ No | ✅ Yes | Monitor maturity |
| **Alternative Loss Functions** | ❌ No (MSE only) | ✅ Yes (7 types) | Evaluate after validation |
| **WAN Trainer GUI** | ⚠️ Has Gradio GUI | ✅ PySide6 desktop GUI | Different approach |
| **CUDA 12.9 Support** | ❌ No | ✅ Yes | Consider adding |
| **Triton Dependencies** | ❌ No | ✅ Yes | Consider adding |
| **Mask-Weighted Loss** | ✅ Yes (461 lines) | ❌ No | blissful-tuner is superior |
| **Muon Optimizer** | ✅ Yes | ❌ No | blissful-tuner has this |

---

## Table of Contents

1. [RamTorch Memory Optimization](#1-ramtorch-memory-optimization)
2. [Alternative Loss Functions](#2-alternative-loss-functions)
3. [WAN 2.2 Trainer GUI](#3-wan-22-trainer-gui)
4. [Extended Optimizer Ecosystem](#4-extended-optimizer-ecosystem)
5. [Dependency & CUDA Updates](#5-dependency--cuda-updates)
6. [Integration Priority Assessment](#6-integration-priority-assessment)

---

## 1. RamTorch Memory Optimization

### Overview

RamTorch is a memory optimization library that enables hybrid CPU-GPU training by keeping model weights on CPU and transferring them to GPU on-demand during forward passes. This allows training large models (14B+ parameters) on consumer GPUs with limited VRAM.

### Problem Solved

- GPU memory constraints when training large video/image diffusion models
- Enables training on 8-16GB VRAM systems that would otherwise OOM
- Complements existing `--blocks_to_swap` optimization

### Command-Line Arguments

```bash
--use_ramtorch          # Apply RamTorch to DiT/transformer model
--use_ramtorch_network  # Apply RamTorch to LoRA network weights
```

### Implementation

```python
from ramtorch.helpers import replace_linear_with_ramtorch

# Model loading to CPU when RamTorch enabled
loading_device = "cpu" if blocks_to_swap > 0 or args.use_ramtorch else accelerator.device

# Replace Linear layers with CPU-bouncing equivalents
if args.use_ramtorch:
    transformer = replace_linear_with_ramtorch(transformer, accelerator.device, weight_dtype)

# Required synchronization after backward pass
if args.use_ramtorch or args.use_ramtorch_network:
    torch.cuda.synchronize()
```

### Supported Architectures

| Architecture | Model Support | Network/LoRA Support |
|-------------|---------------|---------------------|
| HunyuanVideo | ✅ | ✅ |
| WAN 2.1/2.2 | ✅ (incl. high-noise model) | ✅ |
| Qwen-Image | ✅ | ✅ |
| FLUX.1 Kontext | ✅ | ✅ |
| FramePack | ✅ | ✅ |

### Trade-offs

| Aspect | Impact |
|--------|--------|
| VRAM Reduction | 40-60% depending on model |
| Speed Overhead | 10-20% slower due to CPU-GPU transfers |
| Best For | Consumer GPUs (8-16GB) with ample system RAM |
| Not Recommended | A100/H100 where VRAM is abundant |

### Dependencies

```toml
RamTorch @ git+https://github.com/67372a/RamTorch
```

Uses bleeding-edge GitHub fork rather than stable PyPI release.

### Maturity: EXPERIMENTAL

- Well-integrated but still iterating (recent dtype fixes)
- No formal documentation or benchmarks
- Requires explicit developer installation

---

## 2. Alternative Loss Functions

### Overview

Implements 7 alternative loss functions beyond standard MSE, with significant numerical stability improvements through float64 upcasting. Inspired by "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697).

### Supported Loss Types

| Loss Type | Formula | Use Case |
|-----------|---------|----------|
| `l2` (default) | `(pred - target)²` | Standard flow-matching loss |
| `l1` | `\|pred - target\|` | Robust to outliers |
| `huber` | Quadratic ≤δ, linear >δ | Hybrid robustness |
| `pseudo_huber` | `δ² * (√(1 + (e/δ)²) - 1)` | Smooth Huber approximation |
| `smooth_l1` | Piecewise smooth | Differentiable L1 alternative |
| `scaled_quadratic` | `(e/δ)²` | Scaled MSE for adaptive weighting |
| `smooth_l2` | `((e²) / (\|e\| + β))²` | Normalized MSE for stability |

### Command-Line Arguments

```bash
--loss_type l2|l1|huber|pseudo_huber|smooth_l1|scaled_quadratic|smooth_l2
--loss_delta_beta 1.0  # Hyperparameter for loss types that use δ or β
```

### Numerical Precision Improvements

**Three-layer float64 strategy:**

1. **Early Input Casting** (before subtraction):
   ```python
   # Instead of: target = noise - latents (loses precision)
   target = noise.to(torch.float64) - latents.to(torch.float64)
   ```

2. **Loss Function Internal Casting**:
   ```python
   differences = predictions.to(torch.float64) - targets.to(torch.float64)
   ```

3. **Epsilon Removal**: Float64 provides sufficient precision without artificial `eps` addition

### File Location

```
src/musubi_tuner/utils/loss_utils.py  # 105 lines, 7 implementations + dispatcher
```

### Integration Status

- Fully implemented in `hv_train.py` (HunyuanVideo)
- **Not** yet integrated into WAN, FLUX.2, or other architectures

### Maturity: BETA

- Clean implementation with proper dispatcher pattern
- Research-backed numerical improvements
- Limited to single architecture currently

---

## 3. WAN 2.2 Trainer GUI

### Overview

A comprehensive PySide6-based desktop application for configuring and executing WAN 2.2 training workflows. Provides visual interface for the complex training pipeline while maintaining full parameter control.

### Technology Stack

- **Framework:** PySide6 (Qt for Python) >=6.7.0
- **Entry Point:** `wan-trainer-gui` command
- **Code Size:** ~1,054 lines

### Launch Command

```bash
wan-trainer-gui
# Or: python -m blissful_tuner.gui.wan_trainer_gui
```

### Feature Overview

#### Three-Tab Layout

1. **Train Tab** - Main training configuration
2. **Cache Tab** - VAE latent and text encoder caching
3. **Datasets (TOML) Tab** - Built-in dataset editor with validation

#### Configuration Capabilities

**Path Configuration:**
- WAN T2V HIGH/LOW model paths
- WAN I2V HIGH/LOW model paths
- T5, VAE paths
- Dataset TOML, Log dir, Output dir

**Task Configuration:**
- Task selection: `t2v-A14B`, `i2v-A14B`
- Noise band: `high`, `low` (auto-populates timestep ranges)
- Min/max timestep override

**Runtime Configuration:**
- Attention backend (SDPA, FlashAttention, xFormers)
- Blocks to swap, LoRA rank/alpha, network dropout
- DataLoader workers, discrete flow shift
- Custom environment variables and extra args

**Optimizer Presets (8 built-in):**
1. AdamW
2. AdamW8bit
3. CAME
4. **SAEM** (default, recommended)
5. FFTD
6. SingState
7. TALON
8. SGD

**Dual-Phase Training:**
- Sequential HIGH→LOW noise band training
- Per-phase epoch/dataset/args overrides
- Automatic phase progression

**Profile System:**
- JSON-based configuration persistence
- Auto-save before training
- Load/Save dialogs with versioning

### Generated Command Example

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  -m musubi_tuner.wan_train_network \
  --task i2v-A14B \
  --blocks_to_swap 36 \
  --t5 /path/to/t5.pth \
  --vae /path/to/vae.pth \
  --dit /path/to/i2v_high.safetensors \
  --min_timestep 900 --max_timestep 1000 \
  --network_module networks.lora_wan \
  --network_dim 16 --network_alpha 16 \
  --gradient_checkpointing \
  --fp8_base --fp8_scaled --mixed_precision_transformer \
  --optimizer_type customized_optimizers.simplifiedademamix.SimplifiedAdEMAMixExM \
  --learning_rate 2e-4 \
  ...
```

### Cross-Platform Support

- **POSIX:** `os.setsid()` for process group isolation
- **Windows:** `CREATE_NEW_PROCESS_GROUP` flag
- Graceful termination with 5s timeout before SIGKILL

### File Locations

```
wan_trainer_gui.py                           # Root-level wrapper
src/blissful_tuner/gui/wan_trainer_gui.py    # Main implementation
~/.wan_profiles/*.json                       # Profile storage
```

### Maturity: PRODUCTION-READY

- Comprehensive parameter coverage (30+ fields)
- Robust process management
- Clean architecture with good error handling
- MIT licensed

---

## 4. Extended Optimizer Ecosystem

### Overview

Significantly expands optimizer options from 3 (AdamW, AdamW8bit, Adafactor) to 8+ advanced optimizers, plus support for schedule-free training.

### New Optimizer Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `prodigy-plus-schedule-free` | ~=2.0.1 | Prodigy + schedule-free combination |
| `pytorch_optimizer` | ~=3.8.2 | Meta-learning optimizers |
| `dadaptation` | ~=3.2 | Automatic learning rate selection |
| `came-pytorch` | ~=0.1.3 | CAME optimizer |
| `schedulefree` | ~=1.4.1 | Schedule-free training wrapper |
| `torch-optimi` | ~=0.3.2 | Additional optimizers |
| `lion-pytorch` | ~=0.2.3 | Memory-efficient Lion optimizer |

### Custom Optimizer Fork

```toml
customized-optimizers @ git+https://github.com/67372a/customized-optimizers
```

Contains custom implementations like `SimplifiedAdEMAMixExM` (SAEM).

### Usage Examples

```bash
# Standard PyTorch
--optimizer_type AdamW

# Bitsandbytes variants
--optimizer_type bitsandbytes.optim.AdEMAMix8bit
--optimizer_type bitsandbytes.optim.PagedAdEMAMix8bit

# New optimizers
--optimizer_type pytorch_optimizer.Prodigy
--optimizer_type pytorch_optimizer.DAdapt
--optimizer_type schedulefree.AdamWScheduleFree
--optimizer_type lion_pytorch.Lion
--optimizer_type came_pytorch.CAME

# Custom prodigy variant
--optimizer_type ProdigyPlusScheduleFree
```

### Optimizer Selection Guide

| Use Case | Recommended | Arguments |
|----------|-------------|-----------|
| Standard training | AdamW | (default) |
| Memory-constrained | Lion | `--optimizer_type lion_pytorch.Lion` |
| Automatic LR tuning | DAdapt | `--optimizer_type pytorch_optimizer.DAdapt` |
| Stability + speed | Prodigy | `--optimizer_type pytorch_optimizer.Prodigy` |
| No learning schedule | ScheduleFree | `--optimizer_type schedulefree.AdamWScheduleFree` |

### Special Handling

Schedule-free optimizers are detected and return dummy scheduler:
```python
def is_schedulefree_optimizer(self, optimizer, args):
    return args.optimizer_type.lower().endswith("schedulefree".lower())
```

---

## 5. Dependency & CUDA Updates

### New CUDA Versions

**CUDA 13.0 (cu130):**
```toml
cu130 = [
    "torch~=2.9.1",
    "torchvision~=0.24.1",
    "triton~=3.5.1; platform_system == 'Linux'",
]
```

**CUDA 12.9 (cu129):**
```toml
cu129 = [
    "torch~=2.8.0",
    "torchvision~=0.23.0",
    "triton~=3.4.0; platform_system == 'Linux'",
]
```

### Monitoring & Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| `wandb` | ~=0.21.3 | Experiment tracking |
| `tensorboard` | ~=2.20.0 | Training visualization |
| `matplotlib` | ~=3.10.6 | Plotting |

### Build Tools

| Package | Purpose |
|---------|---------|
| `ninja` | Fast builds for torch.compile |
| `triton` / `triton-windows` | GPU-accelerated operations |

### LyCORIS Fork

```toml
lycoris-lora @ git+https://github.com/67372a/LyCORIS
```

Custom fork (not pinned to specific commit) - may contain experimental features.

### Version Pinning Change

Changed from exact (`==`) to compatible (`~=`) specifier for flexibility:
- `accelerate`: `==1.6.0` → `~=1.6.0`
- `safetensors`: `==0.4.5` → `~=0.6.2`
- `transformers`: `==4.56.1` → `~=4.56.2`

### Installation Commands

```bash
# CUDA 13.0 (latest)
uv sync --extra cu130

# CUDA 12.9
uv sync --extra cu129

# With dev tools
pip install -e . --group dev --extra cu130
```

---

## 6. Integration Priority Assessment

### High Priority (Ready for Integration)

| Feature | Rationale |
|---------|-----------|
| **WAN Trainer GUI** | Production-ready, complete feature set |
| **Extended Optimizers** | Low risk, additive change |
| **WandB/TensorBoard** | Standard tooling, easy integration |

### Medium Priority (Needs Validation)

| Feature | Rationale |
|---------|-----------|
| **Alternative Loss Functions** | Good implementation, needs training validation |
| **CUDA 12.9/13.0** | Requires hardware testing |
| **LyCORIS Fork** | Unclear why forked, needs investigation |

### Low Priority (Experimental)

| Feature | Rationale |
|---------|-----------|
| **RamTorch** | Still iterating, recent fixes, speed trade-off |
| **Custom Optimizer Forks** | Maintenance burden, unclear advantages |

---

## Future Actions

### Monitoring

- [ ] Track GUI updates for new features
- [ ] Watch for RamTorch stabilization
- [ ] Monitor loss function integration into other architectures

### Validation Before Integration

For **GUI**:
- [ ] Test on macOS and Windows
- [ ] Validate all optimizer presets work

For **Loss Functions**:
- [ ] Run training comparison: L2 vs Huber vs Pseudo-Huber
- [ ] Validate float64 overhead is acceptable

For **RamTorch**:
- [ ] Benchmark memory savings vs speed loss
- [ ] Test with WAN 2.2 14B on 16GB GPU

### Commands to Sync Fork

```bash
cd /Users/dustin/musubi-tuner-forks/67372a
git fetch origin
git log --oneline origin/main -10
git diff upstream/main..origin/main --stat
```

---

---

## 7. Detailed Comparison Analysis

### WandB/TensorBoard: NO ACTION NEEDED

Both codebases have **identical logging implementations**:

| Feature | blissful-tuner | 67372a |
|---------|---------------|--------|
| `--logging_dir` | ✅ | ✅ |
| `--log_prefix` | ✅ | ✅ |
| `--log_with tensorboard\|wandb\|all` | ✅ | ✅ |
| `--log_tracker_name` | ✅ | ✅ |
| `--wandb_api_key` | ✅ | ✅ |

**blissful-tuner advantages:**
- Better LR logging with parameter group names
- Muon optimizer momentum tracking
- `structure_bell` weighting scheme (experimental)
- Comprehensive documentation in `advanced_config.md`

**67372a differences:**
- WandB/TensorBoard as main deps (not optional)
- Slightly different case handling

**Verdict:** blissful-tuner's implementation is equal or better. No adoption needed.

---

### Optimizer Ecosystem: OPTIONAL ADDITIONS

**blissful-tuner already supports:**
- Dynamic import via `--optimizer_type package_name.ClassName`
- Schedule-Free optimizers (documented in `advanced_config.md`)
- Muon optimizer (custom implementation)
- AdamW8bit, Adafactor

**67372a adds explicit dependencies for:**

| Package | Purpose | Value |
|---------|---------|-------|
| `dadaptation~=3.2` | D-Adaptation family | HIGH - automatic LR |
| `prodigy-plus-schedule-free~=2.0.1` | Prodigy + Schedule-Free | HIGH - combines benefits |
| `pytorch_optimizer~=3.8.2` | 20+ optimizers | MEDIUM - fallback collection |
| `came-pytorch~=0.1.3` | CAME optimizer | MEDIUM - specialized |
| `lion-pytorch~=0.2.3` | Lion optimizer | MEDIUM - memory efficient |

**Recommendation:** Add high-value optimizers as optional dependencies. No code changes needed - just `pyproject.toml` additions.

---

### RamTorch: MONITOR FOR MATURITY

**blissful-tuner:** Does NOT have RamTorch support.

**67372a implementation:**
- `--use_ramtorch` for model weights
- `--use_ramtorch_network` for LoRA weights
- 40-60% VRAM reduction, 10-20% speed penalty
- Uses bleeding-edge GitHub fork

**Concerns:**
- Still experimental (recent dtype fixes)
- No formal benchmarks
- Uses custom fork, not stable release

**Recommendation:** Wait for stabilization. Useful for consumer GPU users.

---

### Alternative Loss Functions: EVALUATE AFTER VALIDATION

**blissful-tuner:** MSE only, but has **superior mask-weighted loss** (461 lines):
- Spatial mask weighting
- Prior preservation
- Gamma correction
- Per-sample normalization
- Supports HV, WAN, FLUX.2, Qwen-Image

**67372a adds:**

| Loss Type | Formula | Use Case |
|-----------|---------|----------|
| `l2` (MSE) | `(pred - target)²` | Default |
| `l1` | `\|pred - target\|` | Robust to outliers |
| `huber` | Quadratic ≤δ, linear >δ | Hybrid robustness |
| `pseudo_huber` | Smooth Huber | Differentiable |
| `smooth_l1` | Piecewise smooth | L1 alternative |
| `scaled_quadratic` | `(e/δ)²` | Adaptive weighting |
| `smooth_l2` | Normalized MSE | Stability |

**Plus:** Float64 numerical precision improvements

**Concerns:**
- Only integrated into HunyuanVideo (not WAN, FLUX.2)
- No training validation data
- **67372a removed mask loss support!**

**Recommendation:** These are complementary. Could integrate loss_utils.py while keeping mask loss. Needs validation first.

---

### GUI: DIFFERENT APPROACHES

**blissful-tuner:** Has Gradio-based web GUI
- Location: `src/musubi_tuner/gui/gui.py` (1,137 lines)
- Framework: Gradio (web-based)
- Supports: Qwen-Image, Z-Image-Turbo

**67372a:** Has PySide6 desktop GUI
- Location: `src/blissful_tuner/gui/wan_trainer_gui.py` (1,054 lines)
- Framework: PySide6 (native desktop)
- Supports: WAN 2.2 (T2V/I2V)
- Features: Dual-phase training, 8 optimizer presets

**These target different architectures and use different frameworks.**

**Recommendation:** Consider if WAN 2.2 GUI is valuable. Different UX paradigm.

---

### CUDA Support: CONSIDER ADDITIONS

| CUDA Version | blissful-tuner | 67372a |
|-------------|----------------|--------|
| cu124 | ✅ (unpinned) | ✅ torch ~=2.5.1 |
| cu128 | ✅ (unpinned) | ✅ torch ~=2.7.1 + Triton |
| **cu129** | ❌ | ✅ torch ~=2.8.0 + Triton |
| cu130 | ✅ torch >=2.9.1 | ✅ torch ~=2.9.1 + Triton |

**67372a adds:**
- CUDA 12.9 support
- Explicit Triton dependencies (Linux + Windows)
- Stricter version pinning (`~=` vs unpinned)

**Recommendation:** Add cu129 support and Triton dependencies for better compatibility.

---

## 8. Recommended Actions

### High Priority (Low Effort, High Value)

1. **Add optimizer dependencies** to `pyproject.toml`:
   ```toml
   # Optional optimizer extras
   optimizers = [
       "dadaptation~=3.2",
       "prodigy-plus-schedule-free~=2.0.1",
       "pytorch_optimizer~=3.8.2",
   ]
   ```

2. **Add CUDA 12.9 support**:
   ```toml
   cu129 = [
       "torch~=2.8.0",
       "torchvision~=0.23.0",
       "triton~=3.4.0; platform_system == 'Linux'",
       "triton-windows~=3.4.0.post21; platform_system == 'Windows'",
   ]
   ```

### Medium Priority (Needs Validation)

3. **Evaluate alternative loss functions** after training comparison
4. **Monitor RamTorch** for stability improvements

### Low Priority (Different Direction)

5. **WAN 2.2 GUI** - only if desktop GUI preferred over web GUI
6. **Custom optimizer forks** - maintenance burden, unclear benefits

---

## Changelog

| Date | Event |
|------|-------|
| 2026-01-30 | Initial documentation created from multi-agent investigation |
| 2026-01-30 | Added detailed comparison analysis after codebase comparison |
