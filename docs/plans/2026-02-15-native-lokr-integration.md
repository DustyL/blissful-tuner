# Native LoKr Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate upstream musubi-tuner LoHa/LoKr support (commit `0b2d692`) into Blissful Tuner, preserving all Blissful extensions (DoRA, RS-LoRA, BlissfulLogger, MUON, mask loss, Conv2d LoHa).

**Architecture:** LoKr reuses `LoRANetwork` via `module_class`/`module_kwargs` injection (upstream pattern). Existing Blissful LoHa keeps its standalone `LoHaNetwork` (convergence deferred to Slice 8). A shared architecture registry provides target modules and default exclude/include patterns for both LoHa and LoKr. Factor persistence uses dual state-dict buffer + safetensors metadata. Backend routing is per-weight-file with `--prefer_lycoris` (native formats always merge natively).

**Tech Stack:** PyTorch, safetensors, accelerate. Tests use `unittest.TestCase` style (pytest-discovered).

**Branch:** `feat/native-lokr-integration`

---

## Design Decisions (Finalized)

1. **LoKr v1 is Linear-only.** Conv2d candidates are skipped with a counted warning. `conv_dim`/`conv_alpha` kwargs are warned and popped.
2. **Kandinsky5 is supported** for LoKr when `architecture=kandinsky5` is explicitly selected. Auto-detect does not promise Kandinsky.
3. **Factor persistence is end-to-end:**
   - Persisted as a `lokr_factor` buffer key in state dict (works for `.pt` + `.safetensors`).
   - Mirrored to safetensors metadata as `ss_lokr_factor` for human/debug/tooling visibility.
   - Load precedence: explicit CLI/network_args > state_dict buffer > default (`-1`). Warn on mismatch.
   - Metadata-only fallback handled by caller (caller reads `ss_lokr_factor` from file metadata and passes as `factor=` kwarg).
   - Factor must survive conversion workflows (default↔diffusers↔default). Converters must preserve `lokr_factor` tensor key and `ss_lokr_factor` metadata.
4. **`--lycoris` becomes a deprecated alias** for `--prefer_lycoris` with a one-time warning. Backend routing is per-weight-file. `--force_lycoris` is dropped from initial integration.
   - **CLI migration:** Replace existing `--lycoris` arg definitions in-place with `"--prefer_lycoris", "--lycoris"` and `dest="prefer_lycoris"`. Do NOT add a second parser argument (would cause argparse conflict).
   - **Loading strategy:** A pre-scan of all weight files determines per-file `(detected_type, backend)` and per-list `needs_lycoris_low`/`needs_lycoris_high` flags. Downstream branches use the computed flags, NOT `args.prefer_lycoris` directly.
   - **Inference LoRA format:** `.safetensors` only. Non-safetensors files (`.pt`, `.bin`) get an early actionable error directing users to `convert_lora.py`.
   - **Deprecation warning:** One-time per process. Use a module-level `_lycoris_warned = False` flag.
5. **LoHa/LoKr asymmetry is acknowledged.** LoHa keeps `LoHaNetwork`; LoKr reuses `LoRANetwork`. TODO comments added; convergence tracked as Slice 8 (future).
6. **Reserved key retention.** Non-dotted keys (`lokr_factor`, `use_rslora_flag`, `use_dora_flag`) are preserved through `filter_lora_state_dict` and `key.split(".", 1)` call sites.
7. **Logger style:** New files (`lokr.py`, `network_arch.py`) use standard `logging` (not `BlissfulLogger`), matching `loha.py`'s convention. Comment: `# Note: logging.basicConfig removed to avoid conflicts with BlissfulLogger - configure at entry points`.
8. **Static merge calls use network-level `merge_to(text_encoders, unet, weights_sd, ...)`**, NOT module-level `merge_to(sd, dtype, device)`. These have different signatures. Implementation must always call the network-level method in static merge helpers.
9. **Network type detection is per-key-family, not per-file.** A state dict can contain mixed key types (e.g., after QKV conversion: `lokr_*` for non-QKV layers + `lora_*` for QKV layers). The merge dispatch in `lora_utils.py` attempts merges by key family in deterministic order (LoHA → LoKr → LoRA), not by a single file-level type. The `detect_network_type()` function returns `"hybrid"` when multiple types are present, and the file-level routing in Slice 6 uses this for summary logging only (not for dispatch decisions).

---

## File Reference: Architecture Constants

| Architecture | Constant | Target Modules | Default Exclude Patterns |
|---|---|---|---|
| WAN | `ARCHITECTURE_WAN = "wan"` | `["WanAttentionBlock"]` | `[r".*(patch_embedding\|text_embedding\|time_embedding\|time_projection\|norm\|head).*"]` |
| HunyuanVideo | `ARCHITECTURE_HUNYUAN_VIDEO = "hv"` | `["MMDoubleStreamBlock", "MMSingleStreamBlock"]` | `[r".*(img_mod\|txt_mod\|modulation).*"]` |
| HunyuanVideo 1.5 | `ARCHITECTURE_HUNYUAN_VIDEO_1_5 = "hv15"` | `["MMDoubleStreamBlock"]` | `[r".*(_in).*"]` |
| FramePack | `ARCHITECTURE_FRAMEPACK = "fp"` | `["HunyuanVideoTransformerBlock", "HunyuanVideoSingleTransformerBlock"]` | `[r".*(norm).*"]` |
| FLUX Kontext | `ARCHITECTURE_FLUX_KONTEXT = "fk"` | `["DoubleStreamBlock", "SingleStreamBlock"]` | `[r".*(img_mod\.lin\|txt_mod\.lin\|modulation\.lin).*", r".*(norm).*"]` |
| FLUX 2 (Dev) | `ARCHITECTURE_FLUX_2_DEV = "f2d"` | `["DoubleStreamBlock", "SingleStreamBlock"]` | `[r".*(img_mod\.lin\|txt_mod\.lin\|modulation\.lin).*", r".*(norm).*"]` |
| FLUX 2 (Klein 4B) | `ARCHITECTURE_FLUX_2_KLEIN_4B = "f2k4b"` | `["DoubleStreamBlock", "SingleStreamBlock"]` | `[r".*(img_mod\.lin\|txt_mod\.lin\|modulation\.lin).*", r".*(norm).*"]` |
| FLUX 2 (Klein 9B) | `ARCHITECTURE_FLUX_2_KLEIN_9B = "f2k9b"` | `["DoubleStreamBlock", "SingleStreamBlock"]` | `[r".*(img_mod\.lin\|txt_mod\.lin\|modulation\.lin).*", r".*(norm).*"]` |
| Qwen-Image | `ARCHITECTURE_QWEN_IMAGE = "qi"` | `["QwenImageTransformerBlock"]` | `[r".*(_mod_).*"]` (when `exclude_mod=True`, which is default) |
| Qwen-Image Edit | `ARCHITECTURE_QWEN_IMAGE_EDIT = "qie"` | `["QwenImageTransformerBlock"]` | `[r".*(_mod_).*"]` (same) |
| Qwen-Image Layered | `ARCHITECTURE_QWEN_IMAGE_LAYERED = "qil"` | `["QwenImageTransformerBlock"]` | `[r".*(_mod_).*"]` (same) |
| Z-Image | `ARCHITECTURE_Z_IMAGE = "zi"` | `["ZImageTransformerBlock"]` | `[r".*(_modulation\|_refiner).*"]` |
| Kandinsky5 | `ARCHITECTURE_KANDINSKY5 = "k5"` | `["TransformerEncoderBlock", "TransformerDecoderBlock"]` | `[r".*modulation.*"]` + default include_patterns (see `lora_kandinsky.py:19-32`) |

**Note:** Current Blissful LoHa (`loha.py:34-60`) is **missing FLUX_2 variants**. The registry must add them.

---

## Slice 0: Baseline + Branch Hygiene

**Goal:** Establish a clean branch and baseline smoke checks.

**Step 1: Create branch**

```bash
git checkout -b feat/native-lokr-integration
```

**Step 2: Record baseline commands**

These will be rerun after every slice:

```bash
# Syntax check
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src

# Lint
/Users/dustin/blissful-tuner/venv/bin/python -m ruff check src/musubi_tuner/networks/lora.py src/musubi_tuner/networks/loha.py src/musubi_tuner/utils/lora_utils.py

# Tests
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
```

**Step 3: Commit**

```bash
git commit --allow-empty -m "chore: start native-lokr-integration branch"
```

---

## Slice 1: `lora.py` Plumbing + Safety Guards

**Why first:** LoKr reuse via `LoRANetwork` requires `module_kwargs` plumbing and safer regex matching. Max-norm and LoRA+ need defensive behavior for non-LoRA module classes.

**Files:**
- Modify: `src/musubi_tuner/networks/lora.py`

### Task 1.1: Regex fix — `.match` → `.fullmatch`

**What:** `re.match()` only anchors at the start of the string. `re.fullmatch()` requires the entire string to match. This prevents partial matches on module names.

**Where:** `src/musubi_tuner/networks/lora.py:805` and `src/musubi_tuner/networks/lora.py:810`

**Current code (lines 803-815):**
```python
                            # exclude/include filter
                            excluded = False
                            for pattern in exclude_re_patterns:
                                if pattern.match(original_name):
                                    excluded = True
                                    break
                            included = False
                            for pattern in include_re_patterns:
                                if pattern.match(original_name):
                                    included = True
                                    break
```

**Change to:**
```python
                            # exclude/include filter
                            excluded = False
                            for pattern in exclude_re_patterns:
                                if pattern.fullmatch(original_name):
                                    excluded = True
                                    break
                            included = False
                            for pattern in include_re_patterns:
                                if pattern.fullmatch(original_name):
                                    included = True
                                    break
```

**Migration note:** All existing default patterns use `r".*something.*"` which match identically under both `match()` and `fullmatch()`. User-provided patterns like bare `"norm"` would need to become `".*norm.*"`. This matches upstream behavior.

### Task 1.2: Add `module_kwargs` support

**What:** Allow `LoRANetwork` to forward extra kwargs (like `factor` for LoKr) to the module class constructor.

**1.2a — `LoRANetwork.__init__` signature** (`src/musubi_tuner/networks/lora.py:696`)

Add `module_kwargs: Optional[Dict[str, Any]] = None` parameter after `module_class`. Store it as `self.module_kwargs = module_kwargs or {}`.

**Current signature (lines 696-718):**
```python
    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None:
```

**Change to:**
```python
    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        module_kwargs: Optional[Dict[str, Any]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        use_rslora: bool = False,
        use_dora: bool = False,
        enable_conv2d: bool = True,
    ) -> None:
```

Also add `Any` to the imports at the top of the file (line 5):
```python
from typing import Any, Dict, List, Optional, Type, Union
```

**1.2b — Store on self** (after existing `self.prefix = prefix` assignment, around line 740):

```python
        self.module_kwargs = module_kwargs or {}
        self.enable_conv2d = enable_conv2d
```

**1.2c — Unpack into module creation call** (`src/musubi_tuner/networks/lora.py:845`)

**Current code (lines 845-856):**
```python
                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                use_rslora=self.use_rslora,
                                use_dora=self.use_dora,
                            )
```

**Change to:**
```python
                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                use_rslora=self.use_rslora,
                                use_dora=self.use_dora,
                                **self.module_kwargs,
                            )
```

**Note:** LoKr's `LoKrModule.__init__` will accept `**kwargs` to absorb `use_rslora`/`use_dora` it doesn't use. LoHa's existing `LoHaModule.__init__` does not — but LoHa doesn't go through this path (it uses its own `LoHaNetwork`). If LoHa convergence happens in Slice 8, add `**kwargs` to `LoHaModule.__init__` then.

**1.2d — Plumb through `create_network()` factory** (`src/musubi_tuner/networks/lora.py:597`)

**Current code (lines 597-608):**
```python
def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
```

Add extraction from kwargs inside the function body (after existing kwargs extractions, around line 650):

```python
    module_class = kwargs.pop("module_class", LoRAModule)
    module_kwargs = kwargs.pop("module_kwargs", None)
    enable_conv2d = kwargs.pop("enable_conv2d", True)
```

And pass them to `LoRANetwork(...)` constructor (around line 670):

```python
    network = LoRANetwork(
        target_replace_modules,
        prefix,
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
        use_rslora=use_rslora,
        use_dora=use_dora,
        enable_conv2d=enable_conv2d,
    )
```

**1.2e — Plumb through `create_network_from_weights()`** (`src/musubi_tuner/networks/lora.py:1288`)

**Current code (lines 1288-1295):**
```python
def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
```

After existing `module_class = LoRAInfModule if for_inference else LoRAModule` (line 1339), add:

```python
    # Allow caller to override module_class (for LoHa/LoKr)
    module_class = kwargs.pop("module_class", module_class)
    module_kwargs = kwargs.pop("module_kwargs", None)
    enable_conv2d = kwargs.pop("enable_conv2d", True)
```

And pass to `LoRANetwork(...)` (around line 1341):

```python
    network = LoRANetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
        use_rslora=use_rslora,
        use_dora=use_dora,
        enable_conv2d=enable_conv2d,
    )
```

**Note:** Without `enable_conv2d` here, LoKr inference/resume via `create_network_from_weights()` would default `enable_conv2d=True`, allowing Conv2d modules through the scan to `LoKrInfModule.__init__()` which raises on Conv2d — turning a graceful skip into a hard crash.

### Task 1.3: LoRA+ warning for non-LoRA networks

**What:** Warn when `loraplus_lr_ratio` is set but the network has no `lora_up` parameters (true for LoHa/LoKr module classes).

**Where:** `src/musubi_tuner/networks/lora.py` inside `prepare_optimizer_params`, after the `assemble_params` call (around line 1136).

**Current code (lines 1136-1140):**
```python
    if self.unet_loras:
        params, descriptions = assemble_params(self.unet_loras, unet_lr, self.loraplus_lr_ratio)
        all_params.extend(params)
        lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

    return all_params, lr_descriptions
```

**Add after the `assemble_params` call:**
```python
    if self.unet_loras:
        params, descriptions = assemble_params(self.unet_loras, unet_lr, self.loraplus_lr_ratio)
        all_params.extend(params)
        lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        if self.loraplus_lr_ratio is not None and "plus" not in descriptions:
            logger.warning("LoRA+ is not effective for this network type (no 'lora_up' parameters found)")

    return all_params, lr_descriptions
```

**Note:** `descriptions` contains `"plus"` only if `assemble_params` found lora_up params and the plus group was non-empty. If all params went into the `"lora"` group, descriptions will be `[""]` and the warning fires.

### Task 1.4: Max-norm guard for non-LoRA networks

**What:** `apply_max_norm_regularization` scans for `lora_down`/`lora_up` keys. LoHa/LoKr state dicts don't have these keys — guard against empty results.

**Where:** `src/musubi_tuner/networks/lora.py:1222` (start of `apply_max_norm_regularization`)

**Add early return after the state_dict scan loop** (after line ~1240, after the for loop that populates `downkeys`):

```python
        # Guard: only supported for LoRA (lora_down/lora_up parameterization)
        if not downkeys:
            logger.warning("max_norm_regularization is only supported for LoRA (no lora_down keys found)")
            return 0, 0.0, 0.0
```

Also guard the final return against empty `norms` (around line 1271):

**Current code (line 1271):**
```python
    return keys_scaled, sum(norms) / len(norms), max(norms)
```

**Change to:**
```python
    if not norms:
        return keys_scaled, 0.0, 0.0
    return keys_scaled, sum(norms) / len(norms), max(norms)
```

### Task 1.5: `enable_conv2d` enforcement in module scan

**What:** When `enable_conv2d=False`, skip Conv2d modules during the network scan with a counted warning. This is the clean UX path for LoKr's Linear-only contract.

**Where:** `src/musubi_tuner/networks/lora.py` inside the module scan loop in `LoRANetwork.__init__`. The scan determines `is_linear` and `is_conv2d` (around lines 826-835).

**Current code (around lines 826-838):**
```python
                        is_linear = child_module.__class__.__name__ in ["Linear", "LoRACompatibleLinear"]
                        is_conv2d = child_module.__class__.__name__ in ["Conv2d", "LoRACompatibleConv2d"]
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear:
                            dim = self.lora_dim
                            alpha = self.alpha
                        elif is_conv2d:
                            ...
```

**Add skip logic before the dim/alpha assignment:**
```python
                        is_linear = child_module.__class__.__name__ in ["Linear", "LoRACompatibleLinear"]
                        is_conv2d = child_module.__class__.__name__ in ["Conv2d", "LoRACompatibleConv2d"]
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        # Skip Conv2d when disabled (e.g. LoKr Linear-only)
                        if is_conv2d and not self.enable_conv2d:
                            conv2d_skipped_count = getattr(self, "_conv2d_skipped_count", 0) + 1
                            self._conv2d_skipped_count = conv2d_skipped_count
                            continue

                        if is_linear:
                            dim = self.lora_dim
                            ...
```

**Add logging after the scan completes** (after the module scan loop ends, before the "create module" summary):

```python
        if hasattr(self, "_conv2d_skipped_count") and self._conv2d_skipped_count > 0:
            logger.warning(f"Skipped {self._conv2d_skipped_count} Conv2d modules (enable_conv2d=False)")
```

### Smoke checks

```bash
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src
/Users/dustin/blissful-tuner/venv/bin/python -m ruff check src/musubi_tuner/networks/lora.py
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
```

### Commit

```bash
git add src/musubi_tuner/networks/lora.py
git commit -m "feat(lora): add module_kwargs plumbing, fullmatch, max-norm/LoRA+ guards, enable_conv2d"
```

### Review checkpoint

- Diff of `src/musubi_tuner/networks/lora.py` only.
- Verify: existing LoRA training/inference behavior unchanged when `module_kwargs` is not provided.
- Verify: `apply_max_norm_regularization` never throws on empty key sets.

---

## Slice 2: Architecture Registry + LoHa Fixes

**Why now:** Removes drift before adding LoKr; prevents future drift between LoRA/LoHa/LoKr defaults.

**Files:**
- Create: `src/musubi_tuner/networks/network_arch.py`
- Modify: `src/musubi_tuner/networks/loha.py`

### Task 2.1: Create registry module (`network_arch.py`)

**Structure:** Dict-of-dicts keyed by architecture constants. Imports authoritative `*_TARGET_REPLACE_MODULES` from existing `lora_*.py` files (those remain the source of truth).

```python
"""Architecture detection and configuration for network modules (LoHa, LoKr, etc.)."""

import logging

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_FLUX_2,
    ARCHITECTURE_FLUX_2_DEV,
    ARCHITECTURE_FLUX_2_KLEIN_4B,
    ARCHITECTURE_FLUX_2_KLEIN_9B,
    ARCHITECTURE_FLUX_KONTEXT,
    ARCHITECTURE_FRAMEPACK,
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_KANDINSKY5,
    ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_QWEN_IMAGE_EDIT,
    ARCHITECTURE_QWEN_IMAGE_LAYERED,
    ARCHITECTURE_WAN,
    ARCHITECTURE_Z_IMAGE,
)
from musubi_tuner.networks.lora import HUNYUAN_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_flux import FLUX_KONTEXT_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_flux_2 import FLUX_2_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_framepack import FRAMEPACK_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_hv_1_5 import HV_1_5_IMAGE_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_kandinsky import KANDINSKY5_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_qwen_image import QWEN_IMAGE_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_wan import WAN_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_zimage import ZIMAGE_TARGET_REPLACE_MODULES

logger = logging.getLogger(__name__)

# Architecture registry: single source of truth for LoHa/LoKr defaults.
# target_modules are imported from lora_*.py (authoritative source).
# exclude_patterns / include_patterns match what each lora_*.py applies by default.
# exclude_mod_patterns is Qwen-specific: only appended when exclude_mod=True (default).
ARCH_CONFIGS = {
    ARCHITECTURE_WAN: {
        "target_modules": WAN_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*"],
    },
    ARCHITECTURE_HUNYUAN_VIDEO: {
        "target_modules": HUNYUAN_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod|txt_mod|modulation).*"],
    },
    ARCHITECTURE_HUNYUAN_VIDEO_1_5: {
        "target_modules": HV_1_5_IMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(_in).*"],
    },
    ARCHITECTURE_FRAMEPACK: {
        "target_modules": FRAMEPACK_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(norm).*"],
    },
    ARCHITECTURE_FLUX_KONTEXT: {
        "target_modules": FLUX_KONTEXT_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*"],
    },
    ARCHITECTURE_FLUX_2_DEV: {
        "target_modules": FLUX_2_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*"],
    },
    ARCHITECTURE_FLUX_2_KLEIN_4B: {
        "target_modules": FLUX_2_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*"],
    },
    ARCHITECTURE_FLUX_2_KLEIN_9B: {
        "target_modules": FLUX_2_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*"],
    },
    ARCHITECTURE_QWEN_IMAGE: {
        "target_modules": QWEN_IMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [],
        "exclude_mod_patterns": [r".*(_mod_).*"],  # appended when exclude_mod=True (default)
    },
    ARCHITECTURE_QWEN_IMAGE_EDIT: {
        "target_modules": QWEN_IMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [],
        "exclude_mod_patterns": [r".*(_mod_).*"],
    },
    ARCHITECTURE_QWEN_IMAGE_LAYERED: {
        "target_modules": QWEN_IMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [],
        "exclude_mod_patterns": [r".*(_mod_).*"],
    },
    ARCHITECTURE_Z_IMAGE: {
        "target_modules": ZIMAGE_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*(_modulation|_refiner).*"],
    },
    ARCHITECTURE_KANDINSKY5: {
        "target_modules": KANDINSKY5_TARGET_REPLACE_MODULES,
        "exclude_patterns": [r".*modulation.*"],
        "include_patterns": [
            r".*self_attention\.to_query.*",
            r".*self_attention\.to_key.*",
            r".*self_attention\.to_value.*",
            r".*self_attention\.out_layer.*",
            r".*cross_attention\.to_query.*",
            r".*cross_attention\.to_key.*",
            r".*cross_attention\.to_value.*",
            r".*cross_attention\.out_layer.*",
            r".*feed_forward\.in_layer.*",
            r".*feed_forward\.out_layer.*",
        ],
    },
}

SUPPORTED_ARCHITECTURES = list(ARCH_CONFIGS.keys())


def get_arch_config(architecture: str) -> dict:
    """Return config dict for given architecture. Raises ValueError if unsupported."""
    if architecture not in ARCH_CONFIGS:
        supported_list = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        raise ValueError(f"Architecture '{architecture}' is not supported by LoHa/LoKr. Supported: {supported_list}")
    return ARCH_CONFIGS[architecture]
```

### Task 2.2: Update LoHa to consume registry

**What:** Replace the in-file `TARGET_REPLACE_MODULES` and `DEFAULT_EXCLUDE_PATTERNS` dicts with registry lookups. Fix `.match` → `.fullmatch`. Add `merge_weights_to_tensor()`. Add FLUX_2 support (missing today).

**Key changes to `src/musubi_tuner/networks/loha.py`:**

1. **Remove lines 1-62** (the old `TARGET_REPLACE_MODULES`, `DEFAULT_EXCLUDE_PATTERNS`, `SUPPORTED_ARCHITECTURES` dicts and all the per-arch lora_* imports used only for those dicts).

2. **Add import** at the top:
   ```python
   from musubi_tuner.networks.network_arch import get_arch_config, SUPPORTED_ARCHITECTURES
   ```

3. **Update `create_arch_network()`** (lines 249-288) to use registry:
   ```python
   def create_arch_network(..., **kwargs):
       architecture = kwargs.get("architecture", ARCHITECTURE_HUNYUAN_VIDEO)
       config = get_arch_config(architecture)

       exclude_patterns = kwargs.get("exclude_patterns", None)
       if exclude_patterns is None:
           exclude_patterns = []
       else:
           exclude_patterns = ast.literal_eval(exclude_patterns)

       exclude_patterns.extend(config["exclude_patterns"])

       # Qwen exclude_mod support (parity with lora_qwen_image.py)
       if "exclude_mod_patterns" in config:
           exclude_mod = kwargs.get("exclude_mod", True)
           if isinstance(exclude_mod, str):
               exclude_mod = ast.literal_eval(exclude_mod)
           if exclude_mod:
               exclude_patterns.extend(config["exclude_mod_patterns"])

       kwargs["exclude_patterns"] = exclude_patterns

       # Kandinsky include_patterns support
       if "include_patterns" in config and "include_patterns" not in kwargs:
           kwargs["include_patterns"] = config["include_patterns"]

       return create_network(
           config["target_modules"],
           "lora_unet",
           ...
       )
   ```

4. **Update `create_arch_network_from_weights()`** (lines 745-762) similarly.

5. **Fix `.match` → `.fullmatch`** at lines 465 and 470 (same pattern as Slice 1).

6. **Add `merge_weights_to_tensor()` function** for on-the-fly merging (required by Slice 4):

   ```python
   def merge_weights_to_tensor(
       model_weight: torch.Tensor,
       lora_name: str,
       lora_sd: Dict[str, torch.Tensor],
       lora_weight_keys: set,
       multiplier: float,
       calc_device: torch.device,
   ) -> torch.Tensor:
       """Merge LoHa weights directly into a model weight tensor.
       Supports Linear and Conv2d. Consumed keys are removed from lora_weight_keys.
       Returns model_weight unchanged if no matching LoHa keys found."""

       w1a_key = lora_name + ".hada_w1_a"
       w1b_key = lora_name + ".hada_w1_b"
       w2a_key = lora_name + ".hada_w2_a"
       w2b_key = lora_name + ".hada_w2_b"
       alpha_key = lora_name + ".alpha"

       if w1a_key not in lora_weight_keys:
           return model_weight

       w1a = lora_sd[w1a_key].to(calc_device)
       w1b = lora_sd[w1b_key].to(calc_device)
       w2a = lora_sd[w2a_key].to(calc_device)
       w2b = lora_sd[w2b_key].to(calc_device)

       dim = w1b.shape[0]
       alpha = lora_sd.get(alpha_key, torch.tensor(dim))
       if isinstance(alpha, torch.Tensor):
           alpha = alpha.item()
       scale = alpha / dim

       original_dtype = model_weight.dtype
       if original_dtype.itemsize == 1:  # fp8
           model_weight = model_weight.to(torch.float16)
           w1a, w1b = w1a.to(torch.float16), w1b.to(torch.float16)
           w2a, w2b = w2a.to(torch.float16), w2b.to(torch.float16)

       # ΔW = ((w1a @ w1b) * (w2a @ w2b)) * scale
       diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale

       # Reshape for Conv2d if needed (diff is always 2D from matmul)
       if model_weight.dim() == 4 and diff_weight.dim() == 2:
           diff_weight = diff_weight.view(model_weight.shape)

       model_weight = model_weight + multiplier * diff_weight

       if original_dtype.itemsize == 1:
           model_weight = model_weight.to(original_dtype)

       # Remove consumed keys
       for key in [w1a_key, w1b_key, w2a_key, w2b_key, alpha_key]:
           lora_weight_keys.discard(key)

       return model_weight
   ```

7. **Add asymmetry comment** at top of file:
   ```python
   # NOTE: LoHa uses its own LoHaNetwork class (not LoRANetwork reuse).
   # LoKr (lokr.py) reuses LoRANetwork via module_class/module_kwargs injection.
   # This asymmetry exists because LoHa was implemented first with Conv2d support
   # and a standalone network class. Convergence to LoRANetwork reuse is tracked
   # as a future follow-up (Slice 8).
   ```

### Smoke checks

```bash
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src
/Users/dustin/blissful-tuner/venv/bin/python -m ruff check src/musubi_tuner/networks/network_arch.py src/musubi_tuner/networks/loha.py
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
```

### Commit

```bash
git add src/musubi_tuner/networks/network_arch.py src/musubi_tuner/networks/loha.py
git commit -m "feat(networks): add shared arch registry, update LoHa to use it, add merge_weights_to_tensor"
```

### Review checkpoint

- Diff of `src/musubi_tuner/networks/network_arch.py` and `src/musubi_tuner/networks/loha.py`
- Verify: LoHa defaults match per-arch LoRA defaults (including new FLUX_2 variants).
- Verify: LoHa `merge_weights_to_tensor` handles Linear and Conv2d, fp8 temporary cast.

---

## Slice 3: Add native `lokr.py` + Factor Persistence

**Why here:** Once `lora.py` supports `module_kwargs` and `enable_conv2d`, we can reuse `LoRANetwork` safely for LoKr.

**Files:**
- Create: `src/musubi_tuner/networks/lokr.py`
- Modify: `src/musubi_tuner/hv_train_network.py` (metadata mirror + `net_kwargs` plumbing)

### Task 3.1: Implement `lokr.py`

Based on upstream `lokr.py` (440 lines), adapted to Blissful conventions:

- Use standard `logging` (per Design Decision 7, matching `loha.py`'s convention).
- Use the shared registry (`get_arch_config`) for target modules and default patterns.
- Enforce Linear-only: pass `enable_conv2d=False` to `lora.create_network(...)`. Also raise `ValueError` in `LoKrModule.__init__` if `org_module` is Conv2d (defense-in-depth).
- If `conv_dim`/`conv_alpha` in kwargs, warn and pop them.
- Support `exclude_mod` kwarg for Qwen architectures (same as LoHa in Slice 2).
- Support Kandinsky `include_patterns` (same as LoHa in Slice 2).

**Key components:**

1. `factorization(dimension, factor)` — dimension factorization helper (from upstream)
2. `make_kron(w1, w2, scale)` — Kronecker product helper (from upstream)
3. `LoKrModule(nn.Module)` — training module, Linear-only, Conv2d raises ValueError
4. `LoKrInfModule(LoKrModule)` — inference module with `merge_to()` and `get_weight()`
5. `create_arch_network(...)` — uses registry, passes `module_class=LoKrModule`, `module_kwargs={"factor": factor}`, `enable_conv2d=False`
6. `create_arch_network_from_weights(...)` — factor recovery with precedence
7. `merge_weights_to_tensor(...)` — for on-the-fly merging in Slice 4

**Factor persistence (in `create_arch_network`):**
```python
    factor = int(kwargs.pop("factor", -1))

    network = lora.create_network(
        config["target_modules"],
        "lora_unet",
        ...,
        module_class=LoKrModule,
        module_kwargs={"factor": factor},
        enable_conv2d=False,
        **kwargs,
    )

    # Persist factor as network-level buffer for save/load round-trip
    network.register_buffer("lokr_factor", torch.tensor(factor, dtype=torch.int64))
    return network
```

**Factor resolution extracted as a testable helper** (`_resolve_factor`):
```python
def _resolve_factor(weights_sd: Dict[str, torch.Tensor], explicit_factor: Optional[int] = None) -> tuple:
    """Resolve LoKr factor with precedence: explicit > persisted buffer > default(-1).
    Returns (factor: int, had_mismatch_warning: bool)."""
    persisted_factor = None
    if "lokr_factor" in weights_sd:
        persisted_factor = int(weights_sd["lokr_factor"].item())

    if explicit_factor is not None:
        factor = int(explicit_factor)
        if persisted_factor is not None and factor != persisted_factor:
            logger.warning(
                f"Explicit factor={factor} differs from persisted factor={persisted_factor}. Using explicit."
            )
            return factor, True
        return factor, False
    elif persisted_factor is not None:
        return persisted_factor, False
    else:
        return -1, False
```

**Factor recovery (in `create_arch_network_from_weights`) calls the helper:**
```python
    explicit_factor = kwargs.pop("factor", None)
    factor, _ = _resolve_factor(weights_sd, explicit_factor)

    module_class = LoKrInfModule if for_inference else LoKrModule
    module_kwargs = {"factor": factor}

    network = lora.create_network_from_weights(
        config["target_modules"],
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        module_class=module_class,
        module_kwargs=module_kwargs,
        **kwargs,
    )

    # Re-register factor buffer on reconstructed network
    network.register_buffer("lokr_factor", torch.tensor(factor, dtype=torch.int64))
    return network
```

**Asymmetry comment at top of file:**
```python
# NOTE: LoKr reuses LoRANetwork from lora.py via module_class/module_kwargs injection.
# LoHa (loha.py) uses its own LoHaNetwork class for historical reasons (Conv2d support).
# This asymmetry is tracked for future convergence (Slice 8).
```

### Task 3.2: Mirror factor to safetensors metadata + plumb `net_kwargs`

**Where:** `src/musubi_tuner/hv_train_network.py`

**3.2a — Metadata mirror** (around line 2241, after `metadata[SS_METADATA_KEY_NETWORK_ARGS]`):

```python
        # Mirror LoKr factor to metadata for human/tooling visibility
        unwrapped_nw = accelerator.unwrap_model(network)
        if hasattr(unwrapped_nw, "lokr_factor"):
            metadata["ss_lokr_factor"] = str(int(unwrapped_nw.lokr_factor.item()))
```

**3.2b — Pass `net_kwargs` to base_weights merge** (line 1976):

`net_kwargs` is parsed at line 1984, but the base_weights merge loop is at line 1964 — BEFORE `net_kwargs` is parsed. Two options:

Option A (recommended): Move `net_kwargs` parsing before the base_weights block. Currently:
```
line 1964: if args.base_weights is not None:  (merge loop)
line 1982: (end of merge block)
line 1984: net_kwargs = {}  (parsing)
```

Move the `net_kwargs` parsing block (lines 1984-1988) to before line 1964.

Then update the base_weights call (line 1976):
```python
                module = network_module.create_arch_network_from_weights(
                    multiplier, weights_sd, unet=transformer, for_inference=True,
                    architecture=self.architecture, **net_kwargs
                )
```

**3.2c — Pass `net_kwargs` to dim_from_weights** (line 1996):
```python
            network = network_module.create_arch_network_from_weights(
                1, weights_sd, unet=transformer, architecture=self.architecture, **net_kwargs
            )
```

### Smoke checks

```bash
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src
/Users/dustin/blissful-tuner/venv/bin/python -m ruff check src/musubi_tuner/networks/lokr.py src/musubi_tuner/hv_train_network.py
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
# Quick import sanity:
/Users/dustin/blissful-tuner/venv/bin/python -c "from musubi_tuner.networks import lokr; print(lokr)"
```

### Commit

```bash
git add src/musubi_tuner/networks/lokr.py src/musubi_tuner/hv_train_network.py
git commit -m "feat(networks): add native LoKr with factor persistence, plumb net_kwargs to base_weights/dim_from_weights"
```

### Review checkpoint

- Diff of `src/musubi_tuner/networks/lokr.py` and `src/musubi_tuner/hv_train_network.py`
- Verify: LoKr can be instantiated with `--network_module networks.lokr`
- Verify: Factor round-trips through save/load

---

## Slice 4: `lora_utils.py` On-The-Fly Merge Support

**Why here:** Makes native LoHa/LoKr usable in WAN's fp8 load path and any "merge while loading" code.

**Files:**
- Modify: `src/musubi_tuner/utils/lora_utils.py`

### Task 4.1: Add `detect_network_type()`

**Returns `"hybrid"` when multiple key families coexist** (e.g., after QKV conversion: `lokr_*` for non-QKV + `lora_*` for QKV). The file-level routing in Slice 6 uses this for summary logging only — actual merge dispatch is per-key-family, not per-file-type.

**IMPORTANT:** Must also recognize Diffusers-format LoRA keys (`.lora_A.` / `.lora_B.` and `_lora_A_` / `_lora_B_`) as `"lora"`, because the Backend Resolution Prepass in Slice 6 runs on raw unconverted keys BEFORE `convert_from_diffusers()` is called. Without this, Diffusers-format LoRAs would be misclassified as `"unknown"` and error unless `--prefer_lycoris` is set — a regression from current behavior.

Also accepts `Union[Dict, Iterable[str]]` for keys-only scanning (avoids constructing thousands of empty tensors in the prepass).

```python
def detect_network_type(lora_sd_or_keys: Union[Dict[str, torch.Tensor], Iterable[str]]) -> str:
    """Detect network type from state dict keys.
    Returns 'lora', 'loha', 'lokr', 'hybrid', or 'unknown'.
    'hybrid' means multiple key families coexist (e.g. after QKV conversion).
    Accepts a state dict or an iterable of key strings."""
    keys = lora_sd_or_keys.keys() if isinstance(lora_sd_or_keys, dict) else lora_sd_or_keys
    found_types = set()
    for key in keys:
        # Standard LoRA keys (lora_down/lora_up) AND Diffusers-format keys (lora_A/lora_B)
        if "lora_down" in key or "lora_up" in key or "lora_A" in key or "lora_B" in key:
            found_types.add("lora")
        elif "hada_w1_a" in key or "hada_w2_a" in key:
            found_types.add("loha")
        elif "lokr_w1" in key or "lokr_w2" in key or "lokr_w2_a" in key:
            found_types.add("lokr")
    if len(found_types) > 1:
        return "hybrid"
    if len(found_types) == 1:
        return found_types.pop()
    return "unknown"
```

### Task 4.2: Extend weight hook dispatch (per-key-family)

**Where:** `src/musubi_tuner/utils/lora_utils.py:115` (inside `load_safetensors_with_lora_and_fp8`)

**Before the hook definition** (around line 112), add network type detection for summary logging only:
```python
        # Detect network types for summary logging (actual dispatch is per-key-family)
        lora_network_types = [detect_network_type(lora_sd) for lora_sd in lora_weights_list]
        logger.info(f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}, types: {lora_network_types}")
```

**Inside `weight_hook_func`**, use **per-key-family dispatch** — try each merge function in deterministic order (LoHa → LoKr → LoRA) instead of branching on file-level type. This handles hybrid state dicts (e.g., `lokr_*` + `lora_*` after QKV conversion) correctly:

```python
        def weight_hook_func(model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            original_dtype = model_weight.dtype
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)

            for lora_weight_keys, lora_sd, multiplier in zip(
                list_of_lora_weight_keys, lora_weights_list, lora_multipliers
            ):
                lora_name = model_weight_key.rsplit(".", 1)[0]
                lora_name = "lora_unet_" + lora_name.replace(".", "_")

                # Per-key-family dispatch: try each family in deterministic order.
                # Each merge function is a no-op if no matching keys found.
                # This handles hybrid dicts (lokr_* + lora_* after QKV conversion).
                from musubi_tuner.networks.loha import merge_weights_to_tensor as loha_merge
                model_weight = loha_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)

                from musubi_tuner.networks.lokr import merge_weights_to_tensor as lokr_merge
                model_weight = lokr_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)

                # Standard LoRA path (existing code, unchanged)
                down_key = lora_name + ".lora_down.weight"
                up_key = lora_name + ".lora_up.weight"
                alpha_key = lora_name + ".alpha"
                if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
                    continue
                # ... (keep all existing LoRA merge math unchanged) ...

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device, original_dtype)

            return model_weight
```

**Note:** The `from ... import` statements inside the hook are deferred imports to avoid circular dependencies. They are cached by Python after the first call, so there is no performance penalty on subsequent calls.

### Task 4.3: Reserved key retention in `filter_lora_state_dict`

**Where:** `src/musubi_tuner/utils/lora_utils.py:20`

**What:** Non-dotted keys (`lokr_factor`, `use_rslora_flag`, `use_dora_flag`) are network-level metadata, not module parameters. They must survive include/exclude filtering.

**Current code (line 29):**
```python
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
```

**Change to:**
```python
        weights_sd = {k: v for k, v in weights_sd.items() if "." not in k or regex_include.search(k)}
```

**Same for exclude (line 35):**
```python
        weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
```

**Change to:**
```python
        weights_sd = {k: v for k, v in weights_sd.items() if "." not in k or not regex_exclude.search(k)}
```

### Task 4.4: Suppress false "unused key" warnings for reserved keys

**Where:** After the hook runs, remaining keys in `lora_weight_keys` trigger warnings. Non-dotted keys should be excluded.

Find the warning that logs remaining keys (search for `"Remaining"` or `"remaining"` or leftover key logic) and exclude non-dotted keys from the count.

### Smoke checks

```bash
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src
/Users/dustin/blissful-tuner/venv/bin/python -m ruff check src/musubi_tuner/utils/lora_utils.py
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
```

### Commit

```bash
git add src/musubi_tuner/utils/lora_utils.py
git commit -m "feat(lora_utils): add LoHa/LoKr on-the-fly merge dispatch, reserved key retention"
```

### Review checkpoint

- Diff of `src/musubi_tuner/utils/lora_utils.py`
- Verify: `quantization_mode` argument unchanged (WAN's call at `src/musubi_tuner/wan/modules/model.py:1278` still works)
- Verify: `filter_lora_state_dict` with include_pattern preserves `lokr_factor`

---

## Slice 5: Converters (Diffusers + ComfyUI Z-Image QKV)

**Why now:** Converters are interop-critical and should be updated before inference routing changes.

**Files:**
- Modify: `src/musubi_tuner/convert_lora.py`
- Modify: `src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py`

### Task 5.1: `convert_lora.py` — LoHa/LoKr support

**5.1a — `convert_from_diffusers()`**: Support `_hada_` → `.hada_` and `_lokr_` → `.lokr_` key mappings. Preserve BlissfulLogger. Handle keys without `.` gracefully (e.g., `lokr_factor`).

Port upstream logic:
```python
        if "_lora_" in new_key:  # LoRA
            new_key = new_key.replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
            new_key = new_key.replace("_lora_down_", ".lora_down.").replace("_lora_up_", ".lora_up.")
        else:  # LoHa or LoKr
            new_key = new_key.replace("_hada_", ".hada_").replace("_lokr_", ".lokr_")
```

**5.1b — `convert_to_diffusers()`**: Pass through `.hada_*` and `.lokr_*` weights. Copy alpha for LoHa/LoKr (don't apply LoRA-style sqrt scaling). Log estimated type.

Port upstream logic for the key handling and alpha scaling changes.

**5.1c — Update parser description:**
```python
    parser = argparse.ArgumentParser(description="Convert LoRA/LoHa/LoKr weights between default and other formats")
```

### Task 5.1d: Conversion Persistence Rules

**What:** Factor must survive conversion workflows (default ↔ diffusers ↔ default). Two persistence channels:

1. **Tensor key `lokr_factor`**: Treat as a pass-through key in both `convert_from_diffusers` and `convert_to_diffusers`. If the key exists in the source dict, copy it unchanged to the output dict. Do NOT attempt key renaming on non-dotted keys.

2. **Safetensors metadata `ss_lokr_factor`**: When writing the output file, copy `ss_lokr_factor` from source metadata to output metadata if present.

**Implementation in `convert_from_diffusers`:**
```python
    # Pass through non-dotted metadata keys (lokr_factor, use_rslora_flag, etc.)
    for key, value in state_dict.items():
        if "." not in key:
            new_state_dict[key] = value
```

**Implementation in `convert_to_diffusers`:**
```python
    # Same: pass through non-dotted metadata keys
    for key, value in state_dict.items():
        if "." not in key:
            new_state_dict[key] = value
```

**Metadata passthrough** (in the safetensors save call, both directions):

**IMPORTANT:** Preserve ALL source metadata by default, then set/override specific keys as needed. Do NOT construct an empty dict and selectively copy — that would silently drop hashes, training provenance, and other metadata the user may depend on.

```python
    # Read source metadata (preserve all existing metadata)
    with safe_open(args.input, framework="pt") as f:
        output_metadata = dict(f.metadata() or {})

    # Ensure LoKr factor is preserved (already in source_metadata if present)
    # If conversion changed tensors, recompute hashes as needed (existing behavior)

    save_file(new_state_dict, args.output, metadata=output_metadata)
```

### Task 5.2: `convert_z_image_lora_to_comfy.py` — LoHa/LoKr QKV merge

**5.2a — LoHa QKV merge** (lossless via `block_diag`/`cat`): Port upstream's block-diagonal merge for `hada_w1_a`/`hada_w2_a` and concatenation for `hada_w1_b`/`hada_w2_b`. Alpha scaled by 3x (rank triples).

**5.2b — LoKr QKV merge** (lossy via SVD → LoRA): Port upstream's "materialize deltas + concat + SVD → LoRA for QKV only" with `--lokr_rank` cap.

**5.2c — Add `--lokr_rank` argument** and update parser description.

**5.2d — Add `key not in state_dict` guard** to all key iteration loops (upstream fix for keys already consumed by earlier passes).

### Smoke checks

```bash
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src
/Users/dustin/blissful-tuner/venv/bin/python -m ruff check src/musubi_tuner/convert_lora.py src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
```

### Commit

```bash
git add src/musubi_tuner/convert_lora.py src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py
git commit -m "feat(converters): add LoHa/LoKr support to diffusers and ComfyUI Z-Image converters"
```

### Review checkpoint

- Diff of both converter files
- Verify: LoHa/LoKr survive default → diffusers → default without losing keys
- Verify: Z-Image QKV conversion works for LoRA + LoHa + LoKr

---

## Slice 6a: Per-File Backend Routing Helper + WAN Generation Path

**Why split:** WAN path is most reused; keep blast radius manageable.

**Files:**
- Modify: `src/musubi_tuner/wan_generate_video.py`

### Task 6a.1: CLI flags (in-place replacement)

**CRITICAL:** Do NOT add a second parser argument. Replace the existing `--lycoris` argument definition in-place. This avoids argparse conflicts.

**Current code** (find `add_argument("--lycoris"` in `wan_generate_video.py`):
```python
    parser.add_argument("--lycoris", action="store_true", ...)
```

**Replace with:**
```python
    parser.add_argument("--prefer_lycoris", "--lycoris", dest="prefer_lycoris", action="store_true",
        help="Enable LyCORIS backend for non-native weight formats. Native formats always merge natively. "
             "(--lycoris is a deprecated alias for --prefer_lycoris)")
```

Add `import sys` at the top of the file (generation scripts generally do NOT import it — verify before adding).

Add deprecation warning near the top of `main()`:
```python
    if "--lycoris" in sys.argv:
        logger.warning("--lycoris is deprecated. Use --prefer_lycoris instead. Behavior is unchanged.")
```

**Note:** All downstream code now reads `args.prefer_lycoris`, NOT `args.lycoris`. The `import sys` must also be added to every generation script in Slice 6b/6c that gets this deprecation check.

### Task 6a.2: Backend Resolution Prepass

**What:** Before any loading or merging, pre-scan ALL weight files to determine per-file `(detected_type, backend)` and a run-level `needs_lycoris_backend_any` flag. This replaces all downstream `args.lycoris` / `args.prefer_lycoris` branches.

**IMPORTANT:** The prepass runs on raw unconverted keys. `detect_network_type()` recognizes both standard (`lora_down`/`lora_up`) and Diffusers (`lora_A`/`lora_B`) naming as `"lora"`. The actual Diffusers→default key conversion (`convert_from_diffusers()`) still happens later during the loading phase, unchanged from current behavior.

**Where:** In `main()`, after argument parsing and before model loading begins.

```python
    # --- Backend Resolution Prepass ---
    from musubi_tuner.utils.lora_utils import detect_network_type
    from safetensors import safe_open

    lora_file_routing = []  # list of (path, detected_type, backend)
    needs_lycoris_backend_any = False

    def _resolve_lora_routing(lora_paths, label=""):
        """Resolve per-file routing for a list of LoRA paths. Returns list of (path, detected, backend)."""
        nonlocal needs_lycoris_backend_any
        routing = []
        if lora_paths is None:
            return routing
        for lora_path in lora_paths:
            # Inference LoRA weights MUST be .safetensors (load_file() is safetensors-only).
            # .pt files are supported for training/resume but not inference loading.
            if not lora_path.endswith(".safetensors"):
                raise ValueError(
                    f"Inference LoRA weights must be .safetensors format: {lora_path}. "
                    f"Convert with: python convert_lora.py --input {lora_path} --output output.safetensors"
                )

            with safe_open(lora_path, framework="pt") as f:
                detected = detect_network_type(f.keys())

            if detected in ("lora", "loha", "lokr", "hybrid"):
                backend = "native"
                if args.prefer_lycoris:
                    logger.info(f"  {label}{lora_path}: detected={detected}, using native merge (--prefer_lycoris ignored for native types)")
            elif detected == "unknown" and args.prefer_lycoris:
                backend = "lycoris"
                needs_lycoris_backend_any = True
            elif detected == "unknown":
                raise ValueError(
                    f"Cannot detect network type for {lora_path}. Use --prefer_lycoris to route through LyCORIS backend."
                )
            else:
                backend = "native"

            routing.append((lora_path, detected, backend))
            logger.info(f"  LoRA routing: {label}{lora_path} detected={detected} backend={backend}")
        return routing

    lora_file_routing = _resolve_lora_routing(args.lora_weight)

    # WAN 2.2 high-noise model has separate LoRA weights
    lora_file_routing_high_noise = []
    if hasattr(args, "lora_weight_high_noise"):
        lora_file_routing_high_noise = _resolve_lora_routing(
            args.lora_weight_high_noise, label="[high-noise] "
        )

    # Compute per-list lycoris flags (one unknown high-noise file shouldn't
    # force the low-noise model into the CPU static path)
    needs_lycoris_low = any(b == "lycoris" for _, _, b in lora_file_routing)
    needs_lycoris_high = any(b == "lycoris" for _, _, b in lora_file_routing_high_noise)
    # --- End Backend Resolution Prepass ---
```

### Task 6a.2b: Thread routing through WAN's 3 consumption points

WAN has two fundamentally different loading strategies:
- **Non-lycoris path** (lines ~649-701): on-the-fly merge via weight hooks during model load
- **LyCORIS path** (lines ~706-729): loads model to CPU first, then merges post-load via `merge_lora_weights()`

**All 3 places that consume LoRA weights must use the prepass routing, NOT raw args:**

**Point 1 — Initial low-noise model load** (`load_dit_model(args, args.dit, args.lora_weight, ...)` around line 594):
Replace the current call to thread `lora_file_routing` and `needs_lycoris_low` instead of `args.lora_weight` and `args.lycoris`.

**Point 2 — Initial high-noise model load** (`load_dit_model(args, args.dit_high_noise, args.lora_weight_high_noise, ...)` around line 602):
Same treatment with `lora_file_routing_high_noise` and `needs_lycoris_high`.

**Point 3 — Lazy-loading path** (most likely to regress — around line 1615-1622):
Current code:
```python
    lora_weight = args.lora_weight_high_noise if is_high_noise else args.lora_weight
    lora_multiplier = args.lora_multiplier_high_noise if is_high_noise else args.lora_multiplier
    model = load_dit_model(args, dit_path, lora_weight, lora_multiplier, ...)
```

Must become:
```python
    routing = lora_file_routing_high_noise if is_high_noise else lora_file_routing
    lora_multiplier = args.lora_multiplier_high_noise if is_high_noise else args.lora_multiplier
    needs_lycoris = needs_lycoris_high if is_high_noise else needs_lycoris_low
    model = load_dit_model(args, dit_path, routing, lora_multiplier, ..., needs_lycoris=needs_lycoris)
```

**Note:** `load_dit_model()` signature needs updating. Currently takes `lora_weights: List[str]`. Change to accept routing tuples `List[Tuple[str, str, str]]` or alternatively keep paths but pass routing dict separately. The simpler approach: build a `path_to_routing: Dict[str, Tuple[str, str]]` dict from all routing lists and pass it alongside the existing args, so the function signature change is minimal (add `path_routing: Optional[Dict] = None`).

**`needs_lycoris_backend_any` is now per-list** (`needs_lycoris_low` / `needs_lycoris_high`), so one unknown high-noise file doesn't force the low-noise model into the CPU static path.

### Task 6a.3: Factor metadata injection at load time

**What:** For the on-the-fly merge path (weight hook in `lora_utils.py`), the hook only receives the state dict — it has no access to file paths or safetensors metadata. To support old LoKr weights missing the `lokr_factor` buffer key, inject the factor into the state dict at load time (before the dict is passed to the hook).

**Where:** In every load path, after `load_file()` and before passing the state dict to the weight hook or merge helper:

```python
    lora_sd = load_file(lora_path)

    # Inject lokr_factor from metadata if buffer key is missing (old weights compat)
    if any("lokr_w1" in k or "lokr_w2" in k for k in lora_sd):
        if "lokr_factor" not in lora_sd:
            with safe_open(lora_path, framework="pt") as f:
                file_metadata = f.metadata() or {}
            if "ss_lokr_factor" in file_metadata:
                lora_sd["lokr_factor"] = torch.tensor(int(file_metadata["ss_lokr_factor"]), dtype=torch.int64)
                logger.info(f"  Injected lokr_factor={int(file_metadata['ss_lokr_factor'])} from metadata into state dict")
```

This injection happens once at load time and covers BOTH:
- The on-the-fly merge path (weight hook has the factor in the state dict)
- The static merge path (native_merge_file receives a complete state dict)

**Note:** Factor stored as `torch.int64` (not float32) to make the intent explicit.

### Task 6a.4: Fix `filter_lora_state_dict` call with per-file pattern indexing

**Pre-existing bug:** The on-the-fly load path calls `filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)` (line 666), passing `List[str]` where `Optional[str]` is expected. This would crash with `re.compile(list)` if the user supplies patterns. The `merge_lora_weights()` function (line 832) correctly indexes per-file, but the on-the-fly path doesn't.

**Fix in every generation script** (WAN, HV, FramePack, Qwen, Z-Image, FLUX, HV1.5):
```python
    for i, lora_weight in enumerate(lora_weights):
        lora_sd = load_file(lora_weight)
        # ... conversion logic ...

        # Per-file pattern indexing (args.include_patterns is a list, one per file)
        include_pat = args.include_patterns[i] if args.include_patterns and len(args.include_patterns) > i else None
        exclude_pat = args.exclude_patterns[i] if args.exclude_patterns and len(args.exclude_patterns) > i else None
        lora_sd = filter_lora_state_dict(lora_sd, include_pat, exclude_pat)
```

**This replaces all current `filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)` calls** in the on-the-fly path. The existing per-file indexing in `merge_lora_weights()` (LyCORIS path) is already correct.

### Task 6a.5: Key parsing robustness

All `prefix, key_body = key.split(".", 1)` loops must handle keys without `.`:
```python
if "." not in key:
    continue  # skip network-level metadata keys (lokr_factor, use_rslora_flag, etc.)
```

### Smoke checks

```bash
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
```

### Commit

```bash
git add src/musubi_tuner/wan_generate_video.py
git commit -m "feat(wan): add per-file backend routing with --prefer_lycoris, deprecate --lycoris"
```

---

## Slice 6b: HV Generation Path

**Files:**
- Modify: `src/musubi_tuner/hv_generate_video.py`

### Task 6b.1: CLI flags + Backend Resolution Prepass

Same pattern as Slice 6a:
1. Add `import sys` at the top of the file (if not already present).
2. Replace existing `--lycoris` arg in-place with `"--prefer_lycoris", "--lycoris"`, `dest="prefer_lycoris"`.
3. Add deprecation warning on `"--lycoris" in sys.argv`.
4. Add Backend Resolution Prepass (identical logic to Slice 6a, minus the high-noise subsection which is WAN-only).
5. Replace all `args.lycoris` branches with `needs_lycoris_backend_any`.

### Task 6b.2: Per-file native merge dispatch (hybrid-aware)

For each weight file with `backend="native"`, use **per-key-family dispatch** (consistent with Design Decision 8 and Slice 4's on-the-fly hook). Do NOT select a single network type per file.

For static merge paths (CPU merge via `create_arch_network_from_weights(...).merge_to(...)`), the correct approach is:

```python
    # Per-key-family native merge for a single weight file
    def native_merge_file(sd, model, multiplier, architecture, device, lora_path=None, **extra_kwargs):
        """Merge a single weight file natively, handling hybrid dicts.

        Uses the network-level merge_to() signature:
            network.merge_to(text_encoders, unet, weights_sd, dtype=None, device=device)
        where text_encoders=None for unet-only merges.
        """
        merged_any = False

        # 1. LoHa keys
        loha_keys = {k for k in sd if "hada_w1_a" in k or "hada_w2_a" in k or "hada_w1_b" in k or "hada_w2_b" in k}
        if loha_keys:
            # Also include associated alpha keys and non-dotted metadata keys
            loha_modules = {k.rsplit(".", 1)[0] for k in loha_keys}
            loha_sd = {k: v for k, v in sd.items() if any(k.startswith(m) for m in loha_modules) or "." not in k}
            net = loha.create_arch_network_from_weights(multiplier, loha_sd, unet=model, for_inference=True, architecture=architecture)
            net.merge_to(None, model, loha_sd, device=device)
            merged_any = True

        # 2. LoKr keys (with factor fallback readback)
        lokr_keys = {k for k in sd if "lokr_w1" in k or "lokr_w2" in k or "lokr_w2_a" in k}
        if lokr_keys:
            lokr_modules = {k.rsplit(".", 1)[0] for k in lokr_keys}
            lokr_sd = {k: v for k, v in sd.items() if any(k.startswith(m) for m in lokr_modules) or "." not in k}
            net = lokr.create_arch_network_from_weights(multiplier, lokr_sd, unet=model, for_inference=True,
                                                         architecture=architecture, **extra_kwargs)
            net.merge_to(None, model, lokr_sd, device=device)
            merged_any = True

        # 3. LoRA keys (standard + Diffusers-converted)
        lora_keys = {k for k in sd if "lora_down" in k or "lora_up" in k}
        if lora_keys:
            lora_modules = {k.rsplit(".", 1)[0] for k in lora_keys}
            lora_sd = {k: v for k, v in sd.items() if any(k.startswith(m) for m in lora_modules) or "." not in k}
            net = lora_module.create_arch_network_from_weights(multiplier, lora_sd, unet=model, for_inference=True,
                                                                architecture=architecture)
            net.merge_to(None, model, lora_sd, device=device)
            merged_any = True

        if not merged_any:
            logger.warning(f"No mergeable keys found in weight file")
```

**Consider centralizing this helper** (e.g., in `lora_utils.py` or a new `merge_helpers.py`) so hybrid dispatch logic isn't duplicated across 8+ generation scripts. The on-the-fly hook path (Slice 4) handles hybrid natively; this helper covers the static merge path.

For `backend="lycoris"`: use LyCORIS backend (unchanged).

### Task 6b.3: Key parsing robustness

Same `"." not in key: continue` guard as Slice 6a.

### Smoke checks and commit

```bash
git add src/musubi_tuner/hv_generate_video.py
git commit -m "feat(hv): add per-file backend routing with --prefer_lycoris"
```

---

## Slice 6c: Sweep Other Entrypoints

**Files:** All files with `key.split(".", 1)` that could encounter non-dotted keys:
- `src/musubi_tuner/qwen_image_generate_image.py`
- `src/musubi_tuner/fpack_generate_video.py`
- `src/musubi_tuner/flux_kontext_generate_image.py`
- `src/musubi_tuner/flux_2_generate_image.py`
- `src/musubi_tuner/zimage_generate_image.py`
- `src/musubi_tuner/hv_1_5_generate_video.py`
- `src/musubi_tuner/convert_lora.py`
- `src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py`
- `src/musubi_tuner/networks/convert_hunyuan_video_1_5_lora_to_comfy.py` (lines 69, 105)

### Task 6c.1: Defensive `key.split(".", 1)` guards

**Two distinct rules** (do NOT apply blanket "skip all non-dotted"):

**Rule A — Parse/dispatch loops** (generation scripts, merge_lora_weights, key iteration for routing):
```python
if "." not in key:
    continue  # skip network-level metadata keys (lokr_factor, use_rslora_flag, etc.)
prefix, key_body = key.split(".", 1)
```

**Rule B — Converter loops** (convert_lora.py, convert_z_image_lora_to_comfy.py):
Non-dotted keys are **preserved/copied through**, NOT skipped. Per Slice 5 Task 5.1d:
```python
if "." not in key:
    new_state_dict[key] = value  # pass through metadata keys (lokr_factor, etc.)
    continue
# ... normal key conversion logic ...
```

**Why the distinction:** Converters must preserve `lokr_factor` and other non-dotted metadata for round-trip fidelity (Design Decision 3). Parsing loops that do `prefix, body = key.split(".", 1)` would crash on these keys and have no use for them.

### Task 6c.2: Add `--prefer_lycoris` / `--lycoris` alias (in-place replacement)

For each generation script that currently has `--lycoris`:
1. Add `import sys` at the top of the file (if not already present).
2. Replace the existing arg definition in-place (same pattern as Slice 6a.1 — do NOT add a second arg):
```python
    parser.add_argument("--prefer_lycoris", "--lycoris", dest="prefer_lycoris", action="store_true",
        help="Enable LyCORIS backend for non-native weight formats. (--lycoris is deprecated)")
```
3. Add deprecation warning: `if "--lycoris" in sys.argv: logger.warning(...)`.

And add the Backend Resolution Prepass where applicable. Scripts that don't have complex loading strategies (simpler ones that just pass-through to a single merge path) may only need the CLI rename + `args.prefer_lycoris` branch substitution.

### Task 6c.3: Fix `filter_lora_state_dict` per-file pattern indexing

Apply the same fix from Slice 6a Task 6a.4 to all generation scripts that call `filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)`. This is a pre-existing bug (passes `List[str]` where `Optional[str]` is expected).

Scripts affected: `fpack_generate_video.py:477`, `hv_1_5_generate_video.py:355`, `qwen_image_generate_image.py:368`, `zimage_generate_image.py:267`.

### Task 6c.4: Ensure reserved-key retention in manual key-filtering comprehensions

In generation scripts that have manual dict comprehensions for filtering (e.g., regex-based key filtering in `merge_lora_weights()` line 846), ensure non-dotted keys are preserved:
```python
    remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys() if "." in k]))
```

### Smoke checks and commit

```bash
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
git add \
    src/musubi_tuner/qwen_image_generate_image.py \
    src/musubi_tuner/fpack_generate_video.py \
    src/musubi_tuner/flux_kontext_generate_image.py \
    src/musubi_tuner/flux_2_generate_image.py \
    src/musubi_tuner/zimage_generate_image.py \
    src/musubi_tuner/hv_1_5_generate_video.py \
    src/musubi_tuner/convert_lora.py \
    src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py \
    src/musubi_tuner/networks/convert_hunyuan_video_1_5_lora_to_comfy.py
git commit -m "fix: add key.split guard for non-dotted keys, add --prefer_lycoris across all generation scripts"
```

---

## Slice 7: Tests + Documentation

**Files:**
- Create: `tests/test_detect_network_type.py`
- Create: `tests/test_merge_weights_loha.py`
- Create: `tests/test_merge_weights_lokr.py`
- Create: `tests/test_lokr_factor_roundtrip.py`
- Create: `tests/test_arch_registry_defaults.py`
- Create/modify: `docs/loha_lokr.md`
- Modify: `CLAUDE.md`

### Task 7.1: `test_detect_network_type.py`

```python
import unittest
import torch
from musubi_tuner.utils.lora_utils import detect_network_type

class TestDetectNetworkType(unittest.TestCase):
    def test_lora_detection(self):
        sd = {"module.lora_down.weight": torch.zeros(4, 8), "module.lora_up.weight": torch.zeros(8, 4)}
        self.assertEqual(detect_network_type(sd), "lora")

    def test_loha_detection(self):
        sd = {"module.hada_w1_a": torch.zeros(8, 4), "module.hada_w1_b": torch.zeros(4, 8)}
        self.assertEqual(detect_network_type(sd), "loha")

    def test_lokr_detection(self):
        sd = {"module.lokr_w1": torch.zeros(4, 4), "module.lokr_w2_a": torch.zeros(8, 4)}
        self.assertEqual(detect_network_type(sd), "lokr")

    def test_hybrid_detection(self):
        """After QKV conversion: lokr_* keys + lora_* keys coexist."""
        sd = {
            "module_a.lokr_w1": torch.zeros(4, 4),
            "module_a.lokr_w2_a": torch.zeros(8, 4),
            "module_b.lora_down.weight": torch.zeros(4, 8),
            "module_b.lora_up.weight": torch.zeros(8, 4),
        }
        self.assertEqual(detect_network_type(sd), "hybrid")

    def test_unknown_detection(self):
        sd = {"some_random_key": torch.zeros(4)}
        self.assertEqual(detect_network_type(sd), "unknown")

    def test_empty_dict(self):
        self.assertEqual(detect_network_type({}), "unknown")

    def test_non_dotted_keys_ignored(self):
        """Network-level metadata keys (lokr_factor) don't affect type detection."""
        sd = {"lokr_factor": torch.tensor(4.0)}
        self.assertEqual(detect_network_type(sd), "unknown")

    def test_diffusers_lora_detection(self):
        """Diffusers-format LoRA keys (lora_A/lora_B) detected as 'lora'."""
        sd = {
            "diffusion_model.blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 8),
            "diffusion_model.blocks.0.attn.to_q.lora_B.weight": torch.zeros(8, 4),
        }
        self.assertEqual(detect_network_type(sd), "lora")

    def test_accepts_keys_iterable(self):
        """Can accept plain key strings (not just dict) for prepass efficiency."""
        keys = ["module.lora_down.weight", "module.lora_up.weight"]
        self.assertEqual(detect_network_type(keys), "lora")
```

### Task 7.2: `test_merge_weights_loha.py`

Test `merge_weights_to_tensor` from `loha.py`:
- Linear weights (2D model_weight)
- Conv2d weights (4D model_weight)
- FP8 temporary cast — **guarded with skip**: `@unittest.skipUnless(getattr(torch, "float8_e4m3fn", None) is not None, "float8 not available")`
- Key consumption (verify consumed keys are removed from set)
- No-op when no matching keys

### Task 7.3: `test_merge_weights_lokr.py`

Test `merge_weights_to_tensor` from `lokr.py`:
- Low-rank mode (w2_a + w2_b)
- Full matrix mode (w2)
- FP8 temporary cast — **guarded with skip**: `@unittest.skipUnless(getattr(torch, "float8_e4m3fn", None) is not None, "float8 not available")`
- Key consumption
- No-op when no matching keys

### Task 7.4: `test_lokr_factor_roundtrip.py`

```python
import unittest
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open
import tempfile, os

class TestLoKrFactorRoundtrip(unittest.TestCase):
    def _make_lokr_sd(self, factor=4):
        return {
            "lokr_factor": torch.tensor(factor, dtype=torch.int64),
            "lora_unet_block.lokr_w1": torch.randn(2, 2),
            "lora_unet_block.lokr_w2_a": torch.randn(4, 2),
            "lora_unet_block.lokr_w2_b": torch.randn(2, 4),
            "lora_unet_block.alpha": torch.tensor(2.0),
        }

    def test_factor_persists_in_safetensors(self):
        """Factor saved as buffer survives save/load via safetensors."""
        state_dict = self._make_lokr_sd(factor=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.safetensors")
            save_file(state_dict, path)
            loaded = load_file(path)
            self.assertEqual(int(loaded["lokr_factor"].item()), 4)

    def test_factor_persists_in_pt(self):
        """Factor saved as buffer survives save/load via torch.save (.pt)."""
        state_dict = self._make_lokr_sd(factor=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            torch.save(state_dict, path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(int(loaded["lokr_factor"].item()), 8)

    def test_metadata_mirror_in_safetensors(self):
        """ss_lokr_factor in safetensors metadata is readable."""
        state_dict = self._make_lokr_sd(factor=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.safetensors")
            save_file(state_dict, path, metadata={"ss_lokr_factor": "4"})
            with safe_open(path, framework="pt") as f:
                meta = f.metadata()
                self.assertEqual(meta["ss_lokr_factor"], "4")

    def test_factor_precedence_explicit_over_persisted(self):
        """Explicit factor kwarg takes precedence over persisted buffer value.
        Calls real _resolve_factor() helper (extracted from create_arch_network_from_weights)."""
        from musubi_tuner.networks.lokr import _resolve_factor
        sd = self._make_lokr_sd(factor=4)
        # Explicit factor=8 should override persisted factor=4
        factor, warning = _resolve_factor(sd, explicit_factor=8)
        self.assertEqual(factor, 8)
        self.assertTrue(warning)  # mismatch warning should be flagged

    def test_factor_precedence_persisted_when_no_explicit(self):
        """Persisted factor used when no explicit kwarg provided."""
        from musubi_tuner.networks.lokr import _resolve_factor
        sd = self._make_lokr_sd(factor=4)
        factor, warning = _resolve_factor(sd, explicit_factor=None)
        self.assertEqual(factor, 4)
        self.assertFalse(warning)

    def test_factor_default_when_nothing_persisted(self):
        """Default factor=-1 when no explicit kwarg and no persisted value."""
        from musubi_tuner.networks.lokr import _resolve_factor
        sd = self._make_lokr_sd(factor=4)
        del sd["lokr_factor"]
        factor, warning = _resolve_factor(sd, explicit_factor=None)
        self.assertEqual(factor, -1)
        self.assertFalse(warning)

    def test_factor_fallback_from_metadata(self):
        """When lokr_factor buffer is missing, factor recovered from ss_lokr_factor metadata."""
        state_dict = self._make_lokr_sd(factor=4)
        del state_dict["lokr_factor"]  # simulate missing buffer
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.safetensors")
            save_file(state_dict, path, metadata={"ss_lokr_factor": "4"})
            # Load and check buffer is missing
            loaded = load_file(path)
            self.assertNotIn("lokr_factor", loaded)
            # But metadata is present
            with safe_open(path, framework="pt") as f:
                meta = f.metadata()
                self.assertEqual(meta["ss_lokr_factor"], "4")

    def test_filter_preserves_lokr_factor(self):
        """filter_lora_state_dict with include pattern preserves non-dotted keys."""
        from musubi_tuner.utils.lora_utils import filter_lora_state_dict
        sd = self._make_lokr_sd(factor=4)
        # Include only a specific module — note: param is include_pattern, NOT include_filter
        filtered = filter_lora_state_dict(sd, include_pattern="lora_unet_block")
        self.assertIn("lokr_factor", filtered)
```

### Task 7.5: `test_arch_registry_defaults.py`

Validates registry outputs match the defaults in per-arch `lora_*.py` files. This prevents drift.

```python
import unittest
from musubi_tuner.networks.network_arch import ARCH_CONFIGS
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_WAN, ARCHITECTURE_FRAMEPACK, ARCHITECTURE_FLUX_KONTEXT,
    ARCHITECTURE_FLUX_2_DEV, ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5, ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_Z_IMAGE, ARCHITECTURE_KANDINSKY5,
)
from musubi_tuner.networks.lora_wan import WAN_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_framepack import FRAMEPACK_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_flux import FLUX_KONTEXT_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_flux_2 import FLUX_2_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora import HUNYUAN_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_hv_1_5 import HV_1_5_IMAGE_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_qwen_image import QWEN_IMAGE_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_zimage import ZIMAGE_TARGET_REPLACE_MODULES
from musubi_tuner.networks.lora_kandinsky import KANDINSKY5_TARGET_REPLACE_MODULES

class TestArchRegistryDefaults(unittest.TestCase):
    def test_target_modules_match_authoritative_source(self):
        """Registry target_modules must match the constants in lora_*.py files."""
        cases = [
            (ARCHITECTURE_WAN, WAN_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_FRAMEPACK, FRAMEPACK_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_FLUX_KONTEXT, FLUX_KONTEXT_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_FLUX_2_DEV, FLUX_2_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_HUNYUAN_VIDEO, HUNYUAN_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_HUNYUAN_VIDEO_1_5, HV_1_5_IMAGE_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_QWEN_IMAGE, QWEN_IMAGE_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_Z_IMAGE, ZIMAGE_TARGET_REPLACE_MODULES),
            (ARCHITECTURE_KANDINSKY5, KANDINSKY5_TARGET_REPLACE_MODULES),
        ]
        for arch, expected in cases:
            with self.subTest(arch=arch):
                self.assertEqual(ARCH_CONFIGS[arch]["target_modules"], expected)

    def test_all_architectures_have_exclude_patterns(self):
        for arch, config in ARCH_CONFIGS.items():
            with self.subTest(arch=arch):
                self.assertIn("exclude_patterns", config)

    def test_qwen_has_exclude_mod_patterns(self):
        for arch in [ARCHITECTURE_QWEN_IMAGE]:
            self.assertIn("exclude_mod_patterns", ARCH_CONFIGS[arch])

    def test_kandinsky_has_include_patterns(self):
        self.assertIn("include_patterns", ARCH_CONFIGS[ARCHITECTURE_KANDINSKY5])
        self.assertTrue(len(ARCH_CONFIGS[ARCHITECTURE_KANDINSKY5]["include_patterns"]) > 0)
```

### Minimum Test Acceptance Matrix

All 31 tests MUST pass before Slice 7 commit. No `pass` placeholders allowed.

| Test File | Test Case | Asserts |
|---|---|---|
| `test_detect_network_type.py` | `test_lora_detection` | `== "lora"` |
| | `test_loha_detection` | `== "loha"` |
| | `test_lokr_detection` | `== "lokr"` |
| | `test_hybrid_detection` | `== "hybrid"` |
| | `test_unknown_detection` | `== "unknown"` |
| | `test_empty_dict` | `== "unknown"` |
| | `test_non_dotted_keys_ignored` | `== "unknown"` |
| | `test_diffusers_lora_detection` | `== "lora"` |
| | `test_accepts_keys_iterable` | `== "lora"` |
| `test_merge_weights_loha.py` | `test_linear_merge` | ΔW ≠ 0, correct shape |
| | `test_conv2d_merge` | 4D reshape works |
| | `test_fp8_cast` | dtype restored (skip if float8 unavailable) |
| | `test_key_consumption` | consumed keys removed |
| | `test_noop_no_keys` | returns unchanged |
| `test_merge_weights_lokr.py` | `test_lowrank_merge` | w2_a+w2_b path |
| | `test_fullmatrix_merge` | w2-only path |
| | `test_fp8_cast` | dtype restored (skip if float8 unavailable) |
| | `test_key_consumption` | consumed keys removed |
| | `test_noop_no_keys` | returns unchanged |
| `test_lokr_factor_roundtrip.py` | `test_factor_persists_in_safetensors` | factor == 4 |
| | `test_factor_persists_in_pt` | factor == 8 |
| | `test_metadata_mirror_in_safetensors` | `ss_lokr_factor == "4"` |
| | `test_factor_precedence_explicit_over_persisted` | calls real `_resolve_factor`, explicit wins |
| | `test_factor_precedence_persisted_when_no_explicit` | persisted used, no warning |
| | `test_factor_default_when_nothing_persisted` | returns -1 |
| | `test_factor_fallback_from_metadata` | metadata readable when buffer missing |
| | `test_filter_preserves_lokr_factor` | non-dotted key survives filter |
| `test_arch_registry_defaults.py` | `test_target_modules_match_authoritative_source` | all arches match |
| | `test_all_architectures_have_exclude_patterns` | key present |
| | `test_qwen_has_exclude_mod_patterns` | key present |
| | `test_kandinsky_has_include_patterns` | non-empty list |

### Task 7.6: Documentation

- Port/adapt upstream `docs/loha_lokr.md` — remove Japanese sections, add Blissful-specific notes (Conv2d LoHa, factor persistence, `--prefer_lycoris`).
- Update `CLAUDE.md`:
  - Add LoKr to the LoRA Target Modules table
  - Add `networks.loha` and `networks.lokr` to Network Module column
  - Document `--prefer_lycoris` flag
  - Document factor persistence

### Smoke checks

```bash
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q
```

### Commit

```bash
git add tests/ docs/
git commit -m "test: add LoHa/LoKr merge, factor roundtrip, and registry drift tests; docs: add loha_lokr.md"
```

---

## Slice 8 (Future): LoHa Convergence to LoRANetwork Reuse

**Tracked but not implemented in this plan.**

- Migrate `LoHaNetwork` to reuse `LoRANetwork` via `module_class=LoHaModule`.
- Add `**kwargs` to `LoHaModule.__init__` to absorb `use_rslora`/`use_dora`.
- Remove the standalone `LoHaNetwork` class.
- Update `create_arch_network` and `create_arch_network_from_weights` to match LoKr's pattern.
- Remove the asymmetry comments added in Slices 2-3.

---

## Quick Reference: Smoke Check Commands

```bash
# Syntax check
/Users/dustin/blissful-tuner/venv/bin/python -m compileall -q src

# Lint specific files
/Users/dustin/blissful-tuner/venv/bin/python -m ruff check <files>

# Format check
/Users/dustin/blissful-tuner/venv/bin/python -m ruff format --check <files>

# Tests
/Users/dustin/blissful-tuner/venv/bin/python -m pytest tests/ -x -q

# Import sanity (after Slice 3)
/Users/dustin/blissful-tuner/venv/bin/python -c "from musubi_tuner.networks import lokr; print(lokr)"
```

---

## Handoff Protocol

For each slice:
1. Implement all tasks in the slice.
2. Run smoke checks.
3. Send diff(s) + smoke check output.
4. Wait for review before proceeding to next slice.
