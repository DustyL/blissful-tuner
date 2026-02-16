# Unified Attention Refactor Plan for FLUX.2

## Overview

Port upstream's unified attention system to blissful-tuner's FLUX.2 implementation, enabling support for multiple attention backends (`torch`, `flash`, `xformers`, `sageattn`).

**Current state:** FLUX.2 uses a local `attention()` function that only supports `torch` (PyTorch SDPA).
**Target state:** Use `modules/attention.py` unified attention with `AttentionParams` dataclass.

**Important note:** `modules/attention.py` does not support `attn_mode="sdpa"` as a string. For user-facing consistency, keep `sdpa -> torch` normalization in
FLUX.2 CLI scripts (same pattern as `hv_generate_video.py`).

---

## Architecture Comparison

### Current Blissful Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│ flux2_models.py                                                 │
│                                                                 │
│  Flux2.forward()                                                │
│    └── for block in double_blocks:                              │
│          block(img, txt, pe_x, pe_ctx, mod_img, mod_txt)        │
│                │                                                │
│                ▼                                                │
│  DoubleStreamBlock._forward(... attn_mode, split_attn)          │
│    └── attention(q, k, v, pe, attn_mode, split_attn)            │
│          │                                                      │
│          ▼                                                      │
│  Local attention() function                                     │
│    - apply_rope(q, k, pe)                                       │
│    - torch.nn.functional.scaled_dot_product_attention()         │
│    - assert attn_mode == "torch"  ◄── LIMITATION                │
└─────────────────────────────────────────────────────────────────┘
```

### Target Upstream Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│ flux2_models.py                                                 │
│                                                                 │
│  from modules.attention import AttentionParams                  │
│  from modules.attention import attention as unified_attention   │
│                                                                 │
│  Flux2.forward()                                                │
│    └── attn_params = AttentionParams.create_attention_params()  │
│        for block in double_blocks:                              │
│          block(img, txt, pe_x, pe_ctx, mod_img, mod_txt,        │
│                attn_params)  ◄── NEW PARAMETER                  │
│                │                                                │
│                ▼                                                │
│  DoubleStreamBlock._forward(..., attn_params: AttentionParams)  │
│    └── attention(qkv_list, pe, attn_params)                     │
│          │                                                      │
│          ▼                                                      │
│  Local attention() wrapper                                      │
│    - apply_rope(q, k, pe)                                       │
│    - transpose B,H,L,D → B,L,H,D                                │
│    - unified_attention(qkv_list, attn_params=attn_params)       │
│          │                                                      │
│          ▼                                                      │
│  modules/attention.py::attention()                              │
│    - Supports: torch, flash, xformers, sageattn                 │
│    - Handles split_attn, attention_mask, varlen                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/musubi_tuner/flux_2/flux2_models.py` | Major: imports, attention wrapper, block signatures, forward method |
| `src/musubi_tuner/flux_2_train_network.py` | Minor: keep `sdpa -> torch` alias + add backend availability checks |
| `src/musubi_tuner/flux_2_generate_image.py` | Minor: keep `sdpa -> torch` alias + add backend availability checks |

---

## Phase 1: Verify `modules/attention.py` Compatibility

### 1.1 Check blissful's `modules/attention.py` vs upstream

```bash
diff /Users/dustin/musubi-tuner/src/musubi_tuner/modules/attention.py \
     /Users/dustin/blissful-tuner/src/musubi_tuner/modules/attention.py
```

**Expected:** Files should be identical or blissful should have the `AttentionParams` dataclass and `attention()` function.

### 1.2 Verify imports work

```python
from musubi_tuner.modules.attention import AttentionParams
from musubi_tuner.modules.attention import attention as unified_attention
```

**Gate:** If `modules/attention.py` is missing or incompatible, sync it from upstream first.

---

## Phase 2: Update `flux2_models.py` Imports

### 2.1 Add new imports

```python
# At top of file, add:
from musubi_tuner.modules.attention import AttentionParams
from musubi_tuner.modules.attention import attention as unified_attention
```

### 2.2 Remove `Optional` from typing if not already imported

The `AttentionParams` type is used in block signatures.

---

## Phase 3: Create Wrapper `attention()` Function

Replace the current local `attention()` function with an upstream-compatible wrapper.

### Current (lines ~1079-1098):

```python
def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pe: Tensor,
    attn_mask: Optional[Tensor] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    control_lengths: Optional[list[int]] = None,
) -> Tensor:
    assert attn_mask is None, "attn_mask is not supported in flux attention"
    assert attn_mode == "torch", f"{attn_mode} not implemented"
    assert split_attn is False, "split_attn not implemented"

    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x
```

### Target:

```python
def attention(qkv_list: list[Tensor], pe: Tensor, attn_params: AttentionParams) -> Tensor:
    """FLUX.2 attention wrapper that applies RoPE and delegates to unified attention.

    Args:
        qkv_list: List of [q, k, v] tensors, each (B, H, L, D)
        pe: Positional encoding tensor for RoPE
        attn_params: Attention configuration (mode, split_attn, masks)

    Returns:
        Attention output (B, L, H*D)
    """
    q, k, v = qkv_list
    del qkv_list

    # Apply rotary position embeddings
    q, k = apply_rope(q, k, pe)

    # Transpose from (B, H, L, D) to (B, L, H, D) for unified attention
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    qkv_list = [q, k, v]
    del q, k, v

    x = unified_attention(qkv_list, attn_params=attn_params)
    return x
```

---

## Phase 4: Update Block Classes

### 4.1 `SingleStreamBlock`

#### Remove attention config parameters from the constructor

The blocks should become stateless with respect to attention configuration (they should not store `attn_mode`/`split_attn` at all).

```python
# FROM (example):
def __init__(..., attn_mode: str = "torch", split_attn: bool = False):
    ...

# TO:
def __init__(..., ...):
    ...
```

#### Remove instance variables (in `__init__`)

```python
# REMOVE these lines:
self.attn_mode = attn_mode
self.split_attn = split_attn
```

#### Update `_forward` signature:

```python
# FROM:
def _forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor]) -> Tensor:
    ...
    # Old local attention call used self.attn_mode/self.split_attn
    ...

# TO:
def _forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor], attn_params: AttentionParams) -> Tensor:
    ...
    qkv_list = [q, k, v]
    del q, k, v
    attn = attention(qkv_list, pe, attn_params)
    del qkv_list, pe
    ...
```

#### Update `forward` signature:

```python
# FROM:
def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor]) -> Tensor:
    if self.training and self.gradient_checkpointing:
        ...
        return checkpoint(forward_fn, x, pe, mod, use_reentrant=False)
    else:
        return self._forward(x, pe, mod)

# TO:
def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor], attn_params: AttentionParams) -> Tensor:
    if self.training and self.gradient_checkpointing:
        ...
        return checkpoint(forward_fn, x, pe, mod, attn_params, use_reentrant=False)
    else:
        return self._forward(x, pe, mod, attn_params)
```

### 4.2 `DoubleStreamBlock`

Same pattern as `SingleStreamBlock`:

1. Remove `attn_mode`/`split_attn` parameters from `__init__` signature (not just instance vars)
2. Update `_forward` signature to add `attn_params: AttentionParams`
3. Update attention call to use `attention(qkv_list, pe, attn_params)`
4. Update `forward` signature and checkpoint call

### 4.3 `Flux2.__init__`

#### Remove passing block-level attn_mode/split_attn

```python
# Current:
self.double_blocks = nn.ModuleList([
    DoubleStreamBlock(
        self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio,
    )
    for _ in range(params.depth)
])

# Target:
self.double_blocks = nn.ModuleList([
    DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
    for _ in range(params.depth)
])
```

Same for `single_blocks`.

---

## Phase 5: Update `Flux2.forward()`

### 5.1 Create `AttentionParams` once

```python
def forward(self, x: Tensor, x_ids: Tensor, timesteps: Tensor, ctx: Tensor, ctx_ids: Tensor, guidance: Tensor | None) -> Tensor:
    num_txt_tokens = ctx.shape[1]

    # ... time embedding code ...

    # Create attention params once for all blocks
    attn_params = AttentionParams.create_attention_params(self.attn_mode, self.split_attn)

    # ... rest of forward ...
```

### 5.2 Pass `attn_params` to blocks

```python
# Double blocks
for block_idx, block in enumerate(self.double_blocks):
    if self.blocks_to_swap:
        self.offloader_double.wait_for_block(block_idx)

    img, txt = block(img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt, attn_params)

    if self.blocks_to_swap:
        self.offloader_double.submit_move_blocks_forward(self.double_blocks, block_idx)

# Single blocks
for block_idx, block in enumerate(self.single_blocks):
    if self.blocks_to_swap:
        self.offloader_single.wait_for_block(block_idx)

    img = block(img, pe, single_block_mod, attn_params)

    if self.blocks_to_swap:
        self.offloader_single.submit_move_blocks_forward(self.single_blocks, block_idx)
```

---

## Phase 6: Optional Memory Optimizations

Upstream adds aggressive `del` statements for memory optimization. These are optional but recommended for large models.

### Pattern:

```python
# After using a tensor that won't be needed again:
del tensor_name

# Example in _forward:
img_qkv = self.img_attn.qkv(img_modulated)
del img_modulated  # Not needed after this
```

**Recommendation:** Add `del` statements in a separate commit after the refactor is verified working.

---

## Phase 7: Update CLI Scripts

### 7.1 `flux_2_train_network.py`

Replace torch-only fail-fast validation with:

1. Keep `sdpa -> torch` normalization (UX consistency)
2. Fail fast on missing optional dependencies with clear install instructions (avoid deep runtime failures)

Recommended pattern (pseudo-code; exact placement depends on where `attn_mode` is resolved from flags):

```python
from musubi_tuner.modules import attention as attention_module

if attn_mode == "sdpa":
    attn_mode = "torch"

if attn_mode == "flash" and attention_module.flash_attn is None:
    raise ValueError("--attn_mode flash requires flash-attn. Install with: pip install flash-attn")
if attn_mode == "xformers" and attention_module.xops is None:
    raise ValueError("--attn_mode xformers requires xformers. Install with: pip install xformers")
if attn_mode == "sageattn" and attention_module.sageattn is None:
    raise ValueError("--attn_mode sageattn requires sageattention. Install with: pip install sageattention")
```

```python
# REMOVE torch-only restrictions (these were a temporary workaround pre-refactor)
```

### 7.2 `flux_2_generate_image.py`

Replace torch-only fail-fast validation with:

1. Keep `sdpa -> torch` normalization
2. Fail fast on missing optional dependencies (flash/xformers/sageattention)
3. Optionally fail fast if user selects a CUDA-only backend on CPU

Recommended pattern:

```python
from musubi_tuner.modules import attention as attention_module

if args.attn_mode == "sdpa":
    args.attn_mode = "torch"

if args.attn_mode == "flash" and attention_module.flash_attn is None:
    raise ValueError("--attn_mode flash requires flash-attn. Install with: pip install flash-attn")
if args.attn_mode == "xformers" and attention_module.xops is None:
    raise ValueError("--attn_mode xformers requires xformers. Install with: pip install xformers")
if args.attn_mode == "sageattn" and attention_module.sageattn is None:
    raise ValueError("--attn_mode sageattn requires sageattention. Install with: pip install sageattention")
```

```python
# REMOVE torch-only restrictions (these were a temporary workaround pre-refactor)
```

---

## Phase 8: Testing

### 8.1 Syntax/Import Check

```bash
python -m compileall -q src/musubi_tuner/flux_2/flux2_models.py
python -c "from musubi_tuner.flux_2 import flux2_models; print('Import OK')"
```

### 8.2 Unit Test: FLUX.2 Attention Wrapper (RoPE + transpose + delegate)

This test exercises the *new* FLUX.2 wrapper (`flux2_models.attention(...)`), not `modules/attention.py` directly.

```python
from musubi_tuner.flux_2.flux2_models import attention
from musubi_tuner.modules.attention import AttentionParams
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

B, H, L, D = 2, 8, 16, 64
q = torch.randn(B, H, L, D, dtype=dtype, device=device)
k = torch.randn(B, H, L, D, dtype=dtype, device=device)
v = torch.randn(B, H, L, D, dtype=dtype, device=device)

# RoPE freqs_cis shape must be broadcastable to (B, H, L, D//2, 1, 2) after apply_rope reshape.
# This shape matches EmbedND(...).forward(...).unsqueeze(1) output: (B, 1, L, D//2, 2, 2)
pe = torch.randn(B, 1, L, D // 2, 2, 2, device=device)

attn_params = AttentionParams.create_attention_params("torch", False)
out = attention([q, k, v], pe, attn_params)
assert out.shape == (B, L, H * D), f"Expected {(B, L, H * D)}, got {out.shape}"
```

### 8.3 Unit Test: Attention Backends (Optional)

This test validates that `modules/attention.py` can execute on the selected backend. It will skip modes that are unavailable or unsupported in the current
environment.

```bash
# Test each attention mode works
for mode in torch flash xformers sageattn; do
    python -c "
from musubi_tuner.modules import attention as attention_module
from musubi_tuner.modules.attention import AttentionParams, attention
import torch

# Create test tensors
B, L, H, D = 2, 16, 8, 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

# Skip CUDA-only backends on CPU
if '$mode' in ('flash', 'xformers', 'sageattn') and device.type != 'cuda':
    print(f'$mode: SKIP - requires CUDA')
    raise SystemExit(0)

# Skip if dependency is missing
if '$mode' == 'flash' and attention_module.flash_attn is None:
    print('$mode: SKIP - flash-attn not installed')
    raise SystemExit(0)
if '$mode' == 'xformers' and attention_module.xops is None:
    print('$mode: SKIP - xformers not installed')
    raise SystemExit(0)
if '$mode' == 'sageattn' and attention_module.sageattn is None:
    print('$mode: SKIP - sageattention not installed')
    raise SystemExit(0)

q = torch.randn(B, L, H, D, device=device, dtype=dtype)
k = torch.randn(B, L, H, D, device=device, dtype=dtype)
v = torch.randn(B, L, H, D, device=device, dtype=dtype)

params = AttentionParams.create_attention_params('$mode', False)
try:
    out = attention([q, k, v], attn_params=params)
    print(f'$mode: OK - output shape {out.shape}')
except Exception as e:
    print(f'$mode: SKIP - {e}')
"
done
```

### 8.4 Integration Test: Training

```bash
accelerate launch --mixed_precision bf16 flux_2_train_network.py \
    --model_version dev \
    --dit /path/to/flux2-dev.safetensors \
    --text_encoder /path/to/mistral3.safetensors \
    --vae /path/to/ae.sft \
    --dataset_config test.toml \
    --network_module networks.lora_flux_2 \
    --network_dim 32 \
    --max_train_steps 5 \
    --sdpa  # Should work now
```

### 8.5 Integration Test: Inference with Different Modes

```bash
for mode in torch flash; do
    python flux_2_generate_image.py \
        --model_version dev \
        --dit /path/to/flux2-dev.safetensors \
        --vae /path/to/ae.sft \
        --text_encoder /path/to/mistral3.safetensors \
        --prompt "test prompt" \
        --attn_mode $mode \
        --save_path /tmp/test_${mode}.png
done
```

---

## Notes / Clarifications

### `torch.compile` Compatibility

`--attn_mode torch` is the most likely to work reliably with `torch.compile`. Alternate backends (flash/xformers/sageattn) may require disabling compile
(e.g., `--compile_dit=None`) to avoid graph breaks or backend-specific errors.

### `--split_attn` Semantics for FLUX.2

FLUX.2 does not currently construct attention masks or per-sample sequence lengths, so `--split_attn` primarily affects execution strategy (split per-sample
attention vs a single batched attention call). This differs from architectures where split attention is used for true variable-length/masked attention.

---

## Commit Strategy

1. **Commit 1: Port unified attention wrapper**
   - Add imports
   - Replace local `attention()` with wrapper
   - Update block signatures
   - Update `Flux2.forward()`

2. **Commit 2: Update CLI validation**
   - Keep `sdpa -> torch` normalization
   - Add backend availability checks (flash/xformers/sageattn)

3. **Commit 3 (optional): Add memory optimizations**
   - Add `del` statements throughout

---

## Rollback Strategy

```bash
# If issues arise, revert to pre-refactor state:
git revert <commit-hash>

# Or restore individual files:
git checkout HEAD~1 -- src/musubi_tuner/flux_2/flux2_models.py
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Weight loading incompatibility | Low | Block structure unchanged, only forward signatures |
| Performance regression | Low | Unified attention is well-tested in other architectures |
| Gradient checkpointing breaks | Medium | Test with `--gradient_checkpointing` explicitly |
| Block swap breaks | Medium | Test with `--blocks_to_swap N` |

---

## Done Criteria

- [ ] `python -m compileall -q src/musubi_tuner/flux_2/` passes
- [ ] `ruff check` passes on modified files
- [ ] Import check passes
- [ ] Training smoke test completes 5 steps with `--sdpa`
- [ ] Training smoke test completes 5 steps with `--flash_attn` (if available)
- [ ] Inference produces valid output with `--attn_mode torch`
- [ ] Inference produces valid output with `--attn_mode flash` (if available)
- [ ] Gradient checkpointing still works
- [ ] Block swap still works
