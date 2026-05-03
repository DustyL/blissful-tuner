# `blissful_core.py` — API Contract

**Path:** `src/blissful_tuner/blissful_core.py`

**Why this matters:** `blissful_core` is the single cleanest surface where blissful-specific code touches the upstream Musubi Tuner. Unlike the modifications inside `hv_train_network.py` or `image_video_dataset.py`, this module **does not monkey-patch anything**. It exposes purely additive functions that callers explicitly invoke. As a result, it survives upstream refactors essentially unchanged — but only as long as the contract documented here is preserved.

If a future upstream change (or a future blissful change) breaks this contract, the breakage will cascade through every architecture's `*_train_network.py` and `*_generate_*.py` script.

## 1 — Module-level state (computed at import time)

| Symbol | Source | Used by |
|--------|--------|---------|
| `BLISSFUL_VERSION` | hardcoded string `"0.12.67"` | `blissful_prefunc()` banner; `get_current_version()` |
| `ROOT_SCRIPT` | `os.path.basename(sys.argv[0]).lower()` | Determines `DIFFUSION_MODEL` |
| `DIFFUSION_MODEL` | One of `{"hunyuan", "wan", "framepack", "flux", "qwen", "k5", None}`, derived from the script filename prefix | `add_blissful_args()` switches on this; consumers call `get_current_model_type()` |
| `MODE` | `"generate"` if `"generate"` in script name, else `"train"` if `"train"` in name, else `None` | `blissful_prefunc()` gates `args.optimized` behavior |
| `logger` | `BlissfulLogger(__name__, "#8e00ed")` | All blissful_core logging |

**⚠️ Invariant:** `argv[0]` MUST be set to a meaningful filename before importing `blissful_core`. The thin wrapper scripts at the repo root (`wan_train_network.py`, `hv_generate_video.py`, etc.) provide this naturally. If a downstream caller imports blissful_core from a context where `argv[0]` is something generic (`python`, `pytest`), `DIFFUSION_MODEL` will be `None` and `add_blissful_*_args()` will silently skip the model-specific arg additions.

## 2 — Public API surface

### Read-only accessors

```python
get_current_model_type() -> str | None
    # Returns DIFFUSION_MODEL. Stable across the process lifetime.

get_current_version() -> str
    # Returns BLISSFUL_VERSION (e.g., "0.12.67").
```

### Parser builders (additive — must be called by each script)

```python
add_blissful_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser
    # Branches on DIFFUSION_MODEL and adds args for that model family:
    #   wan      → NAG (4 args), optimized_compile, simple_modulation, lower_precision_attention,
    #              i2i_path, v2/i2 noise, denoise_strength, noise_mode, v2v_pad_mode,
    #              prompt_weighting, rope_func, mixed_precision_transformer
    #   hunyuan  → from_latent, scheduler, disable_embedded_for_cfg, hidden_state_skip_layer,
    #              apply_final_norm, reproduce, fp8_scaled, prompt_2, te_multiplier
    #   framepack → preview_latent_every, te_multiplier
    #   (wan|hunyuan) → riflex_index, cfgzerostar_*, preview_latent_every, cfg_schedule, perp_neg
    #   (all)    → prompt_wildcards, preview_vae, keep_pngs, codec, container,
    #              fp16_accumulation, optimized
    # Side effect: install_rich_tracebacks() — globally configures Rich error formatting.

add_blissful_flux_args(parser) -> parser
    # Flux-specific arg set (not auto-dispatched; flux scripts call it explicitly).
    # Args: offload_transformer_for_decode, scheduler, fp32_cpu_te, fp32_working_dtype,
    #       preview_vae, preview_latent_every, cfgzerostar_*, cfg_schedule,
    #       fp16_accumulation, guidance_scale, negative_prompt, optimized, prompt_wildcards

add_blissful_qwen_args(parser) -> parser
    # Qwen-specific. Defensive: checks _option_string_actions before adding to avoid duplicates.
    # Args: compile, compile_args, prompt_wildcards

add_blissful_k5_args(parser) -> parser
    # Kandinsky-5-specific arg set. Largest of the per-arch sets.
```

**⚠️ Invariant:** The functions are *additive* — they do not modify existing args. Calling them on a parser that already has these args will raise `argparse.ArgumentError` (except `add_blissful_qwen_args`, which guards via `_option_string_actions`).

**⚠️ Invariant:** No function in this module ever calls `parser.parse_args()`. The script is responsible for that.

### Post-parse processor (must be called after `parser.parse_args()`)

```python
parse_blissful_args(args: argparse.Namespace) -> argparse.Namespace
    # 1. Calls blissful_prefunc(args) — banner, GPU info, optimized-mode application
    # 2. Validates conflicting flag combinations:
    #    - cfgzerostar_scaling + perp_neg (wan|hunyuan) → ValueError
    #    - perp_neg + slg_mode='original' (wan) → ValueError
    #    - compile + optimized_compile (wan) → ValueError
    #    - riflex_index != 0 + rope_func != 'comfy' (wan) → ValueError
    #    - preview_latent_every + sde scheduler → ValueError
    # 3. Resolves args.seed via power_seed() (allows string seeds)
    # 4. Processes prompt wildcards on args.prompt, args.negative_prompt, args.prompt_2
    # 5. Issues warnings (NAG alpha > 1)
    # Returns the same args object (mutated in place).

blissful_prefunc(args: argparse.Namespace)
    # Side effect: sets torch.backends.cuda.matmul.allow_fp16_accumulation if requested.
    # Side effect: applies args.optimized → bulk-overrides several args (fp16_accumulation,
    #              attn_mode='sageattn', compile, fp8_scaled, plus per-model extras).
    # Logs banner with PyTorch/CUDA/VRAM info.
```

**⚠️ Invariant:** `parse_blissful_args` MUST be called after `parser.parse_args()` and before any model loading. Skipping it leaves wildcards unresolved, optimized-mode unapplied, and validation un-enforced.

## 3 — Integration pattern (verified usage in current codebase)

Every Blissful-aware script follows this pattern:

```python
# In e.g. wan_generate_video.py:
from blissful_tuner.blissful_core import add_blissful_args, parse_blissful_args

def setup_parser():
    parser = base_parser_from_upstream()  # e.g. setup_parser_common()
    parser = add_blissful_args(parser)    # additive
    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()
    args = parse_blissful_args(args)      # post-process
    # ... rest of script uses args ...
```

For Flux/Qwen/K5, substitute `add_blissful_flux_args` / `add_blissful_qwen_args` / `add_blissful_k5_args`.

## 4 — Why this contract survives the upstream refactor

The upstream PR #930 trainer-split refactor distributes `setup_parser_common()` into 15 `_add_*_args` helper functions, each responsible for one CLI category. **`add_blissful_args` is independent of this internal restructure** because it appends to an already-built parser — it does not care which helpers built that parser, only that the resulting `ArgumentParser` is well-formed.

Specifically:
- ✅ `add_blissful_args(parser)` works against pre- and post-refactor parsers identically.
- ✅ `parse_blissful_args(args)` works against the resulting `argparse.Namespace` regardless of which helpers populated it.
- ✅ `DIFFUSION_MODEL` derivation depends only on `argv[0]`, which is unchanged.
- ⚠️ The one risk: if upstream renames any of the args that `parse_blissful_args` reads (e.g., `args.slg_mode`, `args.scheduler`, `args.sample_solver`, `args.compile`), the validators will break. Guard via `hasattr(args, ...)` checks where blissful is the only owner.

## 5 — Maintenance checklist (when modifying blissful_core.py)

- [ ] Did you add a new arg in `add_blissful_*_args`? Update §2 above.
- [ ] Did you add a new validator in `parse_blissful_args`? Update §2.
- [ ] Did you read a new attribute from `args` that upstream owns? Add a `hasattr` guard and document it in §4 "one risk".
- [ ] Did you change `DIFFUSION_MODEL` detection? Verify all 6 script families (`hv_`, `wan_`, `fpack_`, `flux_`, `qwen_`, `kandinsky_`) still resolve correctly.
- [ ] Did you bump `BLISSFUL_VERSION`? It appears in the startup banner and in `pyproject.toml`; keep them in sync.

## 6 — Anti-patterns to avoid

- ❌ **Do not** import any module from `musubi_tuner.training` at module load time. That would re-establish the coupling to upstream internals that this contract is designed to avoid.
- ❌ **Do not** monkey-patch `argparse.ArgumentParser` or `Accelerator` from this module. Add explicit hooks instead.
- ❌ **Do not** call `parse_blissful_args()` from inside `add_blissful_args()`. They are deliberately separated so callers can do their own post-processing in between.
- ❌ **Do not** make `DIFFUSION_MODEL` mutable after import. Tests that need different values must subprocess-launch with the appropriate `argv[0]`.
