# PEFT Integration Reference

This file provides guidance to Claude Code (claude.ai/code) for investigating
HuggingFace PEFT (`~/peft`, version `0.19.2.dev0`, editable install) as a source
of techniques to integrate into the blissful-tuner LoRA training pipeline.

> **Scope.** This is *not* a CLAUDE.md for blissful-tuner — that already exists
> at `/home/dustin/blissful-tuner/CLAUDE.md` and is authoritative for the
> project itself. This file is a focused codebase guide for *the PEFT
> repository at `/home/dustin/peft/`*, written from the perspective of someone
> in blissful-tuner asking "what's worth lifting?"

> **Last reviewed: 2026-05-02.** Re-ranked after auditing the current state of
> `src/musubi_tuner/networks/lora.py`. DoRA (`use_dora`) and rsLoRA
> (`use_rslora`) are **already shipped** with regression tests
> (`tests/test_lora_dora_init_device.py`); the previous #1 recommendation
> ("integrate DoRA") is therefore retired.
>
> **Tier 0 added 2026-05-02 (CPU-probe-confirmed bug):** the fast state-dict
> merge helper at `src/musubi_tuner/utils/lora_utils.py:316` does plain
> `alpha / rank` scaling and ignores `use_rslora_flag`, `use_dora_flag`, and
> `dora_layer.weight`. Every modern load-time merge path (WAN, Qwen,
> FLUX.2, Z-Image, HV1.5, FramePack, FLUX Kontext) and the standalone
> `merge_lora.py` CLI silently mis-merges any rsLoRA or DoRA adapter
> produced by this fork. **Fix this before any new feature work.**

## Where PEFT lives in this stack

| Question | Answer |
|----------|--------|
| Source tree | `/home/dustin/peft/` |
| Editable install in `venv314` | `peft 0.19.2.dev0` (constrained by `~/.pip_editable_constraints.txt`) |
| Currently used by blissful-tuner training? | **No.** Blissful-tuner rolls its own `networks.lora_<arch>` + `networks.loha` + `networks.lokr` |
| Used elsewhere in venv? | Yes — pulled in transitively by `transformers`, `diffusers`, `trl`, `accelerate`. The version installed editable is what those libs see when they call `from peft import ...`. |
| Recent commits worth knowing | `dc2e5b2` AdaLoRA Conv2d support, `7d927c3` BEFT method added, `629036a` PVeRA RNG seeding, `2bf7bc2` transformers weight-conversion regression fix |

## What's in the box (PEFT method taxonomy)

Run from `/home/dustin/peft/`. All public classes are exported from
`src/peft/__init__.py`. Tuner implementations live under `src/peft/tuners/`,
one subdirectory per method.

### LoRA-family (rank-r matrix factorizations)

| Method | Layer types | Highlight |
|--------|-------------|-----------|
| **LoRA** (`tuners/lora/`) | Linear, Embedding, Conv1d, Conv2d | Reference impl. ~2,500 lines in `layer.py` alone — far more layer types and quant integrations than blissful-tuner's `networks/lora_*` |
| **DoRA** (toggle `use_dora=True` in `LoraConfig`) | Linear, Conv1d, Conv2d, Embedding | Magnitude-direction decomposition; impl. in `tuners/lora/dora.py` + dispatch via `tuners/lora/variants.py:resolve_lora_variant()` |
| **AdaLoRA** (`tuners/adalora/`) | Linear, Conv2d, Conv1d, Embedding (Conv2d added in `dc2e5b2`) | Per-layer rank learned via SVD-style `lora_A`/`lora_E`/`lora_B` triple + a training-loop callback that prunes singular values |
| **LoraGA** (`tuners/lora/loraga.py`) | Linear (primary) | Gradient-aligned init |
| **PiSSA / OLoRA / EVA / LoftQ / orthogonal / corda** (`init_lora_weights="..."` in `LoraConfig`) | Linear (mostly) | Drop-in initialization variants of LoRA. EVA + CorDA need a calibration pass; PiSSA/OLoRA/orthogonal are weight-only |
| **VeRA** (`tuners/vera/`) | Linear (no Conv2d) | Shared global low-rank A/B + per-layer scalars; ~1% of LoRA params |
| **RandLora** (`tuners/randlora/`) | Linear | Random-basis variant of VeRA |
| **TinyLora / Lily / SHIRA / GraLoRA / Peanut / Miss / RoaD / WaveFT / FourierFT / Adamss** | Linear (mostly) | Newer compact-parameter variants. Each has its own ~500-line `layer.py` + `model.py` |
| **C3A / DeloRA / PsoFT / PVeRA / Beft / BdLoRA** | Linear | Various recent additions. BdLoRA is interesting for inference (block-diagonal layout for tensor-parallel serving) |

### Orthogonal / structured-update family (NO multiplicative LoRA)

| Method | Layer types | Highlight |
|--------|-------------|-----------|
| **OFT** (`tuners/oft/`) | Linear, **Conv2d** | Block-diagonal orthogonal `(I + R)·W` updates with Cayley–Neumann approximation |
| **BOFT** (`tuners/boft/`) | Linear, **Conv2d** | Butterfly-factored orthogonal updates with custom CUDA kernel (`tuners/boft/fbd/fbd_cuda*`); multiple butterfly factors stack |
| **HRA** (`tuners/hra/`) | Linear, **Conv2d** | Householder-reflection adapters; ~2r params/layer |
| **OSF** (`tuners/osf/`) | Linear | Orthogonal subspace fine-tuning (newer) |

### Other (mostly NLP-only — low priority for diffusion)

`adaption_prompt`, `prefix_tuning`, `prompt_tuning`, `p_tuning`,
`multitask_prompt_tuning`, `cpt`, `cartridge`, `xlora`, `poly`,
`trainable_tokens`, `ia3`, `ln_tuning`, `vblora`, `mixed`. Of these,
**X-LoRA** (`tuners/xlora/`) — runtime classifier that dynamically mixes
multiple LoRA adapters per token — is the most cross-domain, but it's
designed around causal-LM token streams; adapting to per-timestep diffusion
mixing would be net-new work.

### Conv2d support (the diffusion-relevant filter)

LoHa ✅ · LoKr ✅ · LoRA ✅ · DoRA ✅ · AdaLoRA ✅ (new) · OFT ✅ · BOFT ✅ ·
HRA ✅ · VeRA ❌ · IA3 ✅ · BOFT-CUDA-kernel ✅. Most of the parameter-frugal
NLP methods (Tiny/Lily/SHIRA/etc.) are Linear-only.

## Core API surface (what an integrator touches)

Three entry points cover ~95% of integrations:

```python
from peft import LoraConfig, get_peft_model, PeftModel, inject_adapter_in_model

# (1) Wrap a model -> returns PeftModel (high-level)
config = LoraConfig(r=16, lora_alpha=32, target_modules=["to_q", "to_v"])
model = get_peft_model(base_model, config)            # in-place + wraps

# (2) Inject without wrapping (low-level — better for custom training loops)
inject_adapter_in_model(config, base_model, adapter_name="default")
# ^ blissful-tuner's create_network() pattern is morally equivalent

# (3) Save/load
model.save_pretrained("out/")                          # writes adapter_model.safetensors
loaded = PeftModel.from_pretrained(base_model, "out/") # restores
```

Key files for understanding the wrapper:
- `src/peft/mapping_func.py:get_peft_model` (128 lines) — entry, dispatches by `peft_type`
- `src/peft/peft_model.py:PeftModel` (3,441 lines) — the wrapper, its
  state-dict APIs, `merge_and_unload()`, `set_adapter()`, `disable_adapter()`,
  `add_adapter()`, the multi-adapter machinery, save/load orchestration
- `src/peft/tuners/tuners_utils.py:BaseTuner` (2,228 lines) — shared base for
  every `*Model` class. The injection algorithm lives in
  `BaseTuner.inject_adapter`; module-name matching in
  `BaseTuner._check_target_module_exists`; the offload-aware merge context
  manager `onload_layer()` is around line 75–140.

## How layer injection works (the part most worth understanding)

Pattern is uniform across every tuner. Read `tuners/lora/layer.py:Linear`
(starts ~line 200) plus `tuners/lora/model.py:_create_and_replace` to see
all of it:

1. `BaseTuner.inject_adapter` walks `model.named_modules()`.
2. For each module, `_check_target_module_exists()` matches against
   `config.target_modules` (string-list-suffix-or-exact, regex, or
   `"all-linear"` magic).
3. On match, `_create_new_module()` instantiates the wrapper class
   (`lora.Linear`, `lora.Conv2d`, `lora.Embedding`, etc.). The original
   module becomes `wrapper.base_layer` (so its weights are *not copied* —
   just referenced).
4. The wrapper holds adapter parameters as **`ModuleDict`/`ParameterDict`
   keyed by adapter name** — e.g. `self.lora_A[adapter_name]`,
   `self.lora_B[adapter_name]`, `self.scaling[adapter_name]`. This is what
   makes multi-adapter coexistence cheap: adding an adapter just appends to
   the dicts; switching is `model.set_adapter(name)`.
5. Forward = `base_layer(x) + scaling * lora_B(lora_A(x))` (LoRA-flavored
   variants override the second term).
6. `merge()` folds the delta into `base_layer.weight` in place (with optional
   `safe_merge` to detect NaN). `unmerge()` reverses it. `merge_and_unload()`
   merges *and* swaps the wrapper out for the underlying nn.Linear so the
   PEFT machinery is gone from the graph.
7. `onload_layer()` context manager (in `tuners_utils.py`) handles
   accelerate-style CPU/disk-offloaded modules during merge — moves to GPU,
   merges, returns to original device.

Compare to blissful-tuner's `src/musubi_tuner/networks/lora.py` (and its
`lora_wan.py`, `lora_flux_2.py` etc. variants): blissful-tuner builds a
parallel `LoRANetwork` object that *holds* the adapter modules and patches
forward via hooks, rather than swapping the target modules in place.
Different abstraction, similar end state.

## Save/load and state-dict shape

- `src/peft/utils/save_and_load.py:get_peft_model_state_dict` (line 77) —
  extracts adapter-only weights, strips adapter-name infix
  (`q_proj.lora_A.default.weight` → `q_proj.lora_A.weight`), handles
  TP/sharded gathers.
- `src/peft/utils/save_and_load.py:set_peft_model_state_dict` (line 602) —
  inverse, with prefix validation.
- **PEFT's on-disk format is `lora_A.weight` / `lora_B.weight`.** This is
  *not* the kohya/diffusers/comfy format that blissful-tuner's
  `convert_lora.py` round-trips. Diffusers-side conversion lives in
  `diffusers.loaders` (e.g. `_convert_kohya_lora_to_diffusers`); PEFT itself
  does not ship a kohya converter.
- `src/peft/utils/transformers_weight_conversion.py` — handles the
  `transformers`-side LoRA layers (added in transformers 5.x) ↔ PEFT key
  remapping. Recently regressed/fixed in commit `2bf7bc2`. Not relevant to
  diffusion architectures.

## Hotswap (potentially valuable for inference)

`src/peft/utils/hotswap.py` — runtime adapter swap without recompiling a
torch.compile'd graph:

- `prepare_model_for_compiled_hotswap(model, target_rank=...)` — pads all
  adapters to the same `target_rank` and converts scalar `scaling` to
  buffers, so the compiled graph sees a fixed shape.
- `hotswap_adapter(model, path, adapter_name)` — loads the new adapter and
  swaps weights *in place*. Constraint: configs must match (rank, alpha,
  dropout, use_dora, target modules).
- `hotswap_adapter_from_state_dict(...)` — same thing but takes a state dict
  directly.

This is genuinely missing from blissful-tuner's inference path — the current
generate scripts (`*_generate_video.py`) load LoRAs by re-merging into the
model, which doesn't compose with `--compile`.

## What blissful-tuner already covers (don't reinvent)

| Feature | Blissful-tuner location | PEFT equivalent |
|---------|-------------------------|------------------|
| LoRA on Linear/Conv2d | `networks/lora.py`, `networks/lora_<arch>.py` | `tuners/lora/layer.py` |
| LoHa (Hadamard product) | `networks/loha.py` | `tuners/loha/` |
| LoKr (Kronecker, w/ factor) | `networks/lokr.py` | `tuners/lokr/` |
| **DoRA** (Linear-only, dropout-conflict guard, device-mismatch-safe) | `networks/lora.py:213–219, :236–242, :248–340` | `tuners/lora/dora.py` + `variants.py` |
| **rsLoRA** with persistent format flags in safetensors | `networks/lora.py:128, 187–193, 749–752, 967–985` | `LoraConfig(use_rslora=True)` |
| Architecture-keyed target modules | `networks/network_arch.py` (13 variants) | `LoraConfig.target_modules` (per-call) |
| Per-arch LoRA factory + create_network | `networks/lora_<arch>:create_network()` | `get_peft_model(config)` |
| LyCORIS bridge | `networks/lycoris.py` | (no LyCORIS bridge in PEFT) |
| Format conversion (kohya/diffusers/comfy) | `convert_lora.py`, `networks/convert_*` | (none in PEFT) |
| `torch.compile` wiring in generation (per-block) | `wan_generate_video.py:756–769, 918–930`, `model_utils.compile_transformer` | (PEFT relies on `prepare_model_for_compiled_hotswap`) |
| Mask-weighted loss + prior preservation | `modules/mask_loss.py`, `modules/lora_ema_teacher.py` | (no diffusion mask-loss in PEFT) |

If the goal were "replace blissful-tuner's networks/ with PEFT," LyCORIS
support, kohya-format I/O, and the mask-loss/prior-preservation pipeline are
blockers. The more realistic integration is selective lifting (below).

### Where blissful-tuner *deliberately diverged from* PEFT's design

These are not "missing features" — they are intentional patches on top of the
PEFT design that future PRs proposing "let's just use PEFT's version" should
not silently revert:

| Divergence | Blissful-tuner | PEFT | Reason to keep blissful-tuner's version |
|------------|----------------|------|------------------------------------------|
| Format-validation guards | `use_rslora_flag` / `use_dora_flag` registered as **buffers in the safetensors** (`lora.py:751–752`) and re-checked at load (`:967–985`) | Encoded only in sidecar `adapter_config.json` | Diffusion users ship `.safetensors` standalone (Civitai / Discord). A missing sidecar → silently wrong scaling. Per-tensor flags fail loudly with an actionable mismatch message. |
| DoRA at `multiplier == 0` | Short-circuits to `zeros_like(base)` (`lora.py:263–264`) | No short-circuit; DoRA still rescales the base output | Blissful-tuner's CFG/scheduling stack drives multiplier dynamically. Without the short-circuit, "disable LoRA" still perturbs the base output. |
| DoRA dropout interaction | DoRA is **disabled** if `dropout > 0` or `rank_dropout > 0`, with a logged reason (`lora.py:212–226`) | Dropout + DoRA silently coexist | PEFT issue history shows DoRA + dropout produces unstable magnitude updates. Explicit guard is safer for the diffusion user base. |
| DoRA weight-norm computation | `get_weight_norm_efficient` avoids materializing `B@A` (`lora.py:265–269`) | `dora.py` materializes when small enough | Diffusion DiT layers are large; the materialized path is a real VRAM hit on FLUX.2 / WAN A14B. |
| DoRA init device safety | Device-cast guard pinned by `tests/test_lora_dora_init_device.py` for the FLUX.2 trainer flow | Patched only after upstream issue reports | The blissful-tuner trainer order (`move_to_device_except_swap_blocks` → `apply_to`) hits this every run. |

## Recommended integration targets (highest value first)

These are the items worth lifting into blissful-tuner, sorted by ROI as of
the 2026-05-02 audit. Items marked **shipped** were on previous versions of
this list and are kept here as anchor points for grep searches.

### Tier 0 — correctness bug, fix before anything else

0. **Fast state-dict merge helper does not honor `use_rslora_flag` or
   `dora_layer.weight`.**
   _Status: confirmed via CPU probe — with stored `alpha=4`, `rank=4`,
   `use_rslora_flag=True`, the helper returns the regular-LoRA delta (scale
   `4/4=1`, magnitude 4.0); the correct rsLoRA delta is scale `4/sqrt(4)=2`,
   magnitude 8.0._
   - **Buggy site**: `src/musubi_tuner/utils/lora_utils.py:316`
     (`lora_merge_weights_to_tensor`). Computes `scale = alpha / dim` (line
     343) unconditionally and never reads the rsLoRA / DoRA metadata keys
     that `networks/lora.py` writes to the safetensors.
   - **Reference correct math**: `src/musubi_tuner/networks/lora.py:406–444`
     (`merge_to`) — the runtime merge path on the live `LoRAModule`. It
     correctly branches on `self.use_rslora` (via `self.scale` set at
     `lora.py:194`) and `self.use_dora` (DoRA materialized merge with
     `weight_norm` and `dora_factor` at `lora.py:431–437`).
   - **Blast radius**: every consumer of `lora_merge_weights_to_tensor` or
     its wrapper `wan_generate_video.merge_lora_weights` —
     `wan_generate_video.py:703`, `qwen_image_generate_image.py:491`,
     `zimage_generate_image.py:406`, `flux_2_generate_image.py:771, 1108`,
     `flux_kontext_generate_image.py:612, 1042`, `fpack_generate_video.py:517`,
     `hv_1_5_generate_video.py:388`, the `merge_lora.py` CLI, and
     `load_safetensors_with_lora_and_fp8` (called by `zimage_model.py:34`
     and `hunyuan_video_1_5_models.py:15`).
   - **Required fix shape**:
     1. Detect `lora_name + ".use_rslora_flag"` / `".use_dora_flag"` /
        `".dora_layer.weight"` in `lora_sd`. Treat both `flag` keys as
        boolean tensor checks (mirroring the load-time guard at
        `lora.py:967–985`).
     2. Compute `scale = alpha / sqrt(dim)` when `use_rslora_flag=True`.
     3. When `dora_layer.weight` is present, replicate the merge math from
        `networks/lora.py:431–437` — materialize the delta, compute
        `weight_norm = self.dora_layer.get_weight_norm_materialized(...)`,
        then `weight = (mag / norm).view(-1,1) * (weight + delta)`. Skip
        DoRA on Conv2d (the live path skips it too — Linear-only).
     4. Discard the metadata keys from `lora_weight_keys` after consumption
        so the leftover-key warning behaves correctly.
     5. Optional but recommended: `safe_merge` mode that computes on a
        clone, runs `torch.isfinite(...).all()`, and only commits on pass.
        Default to off for the fast load path; default to on for the
        offline `merge_lora.py` CLI.

### Tier 1 — high value, low cost, no architectural conflict

1. **Compile-friendly LoRA hotswap** for `*_generate_video.py --compile`.
   _Status: not implemented._
   - **Symptom this fixes**: `wan_generate_video.py:756–769, 918–930`
     compiles each transformer block with `torch.compile`. LoRAs are merged
     into the base via `merge_lora.py` flow before compile. Switching LoRAs
     across a sweep recompiles the entire graph — currently the loop trades
     compile overhead for LoRA flexibility.
   - **PEFT references**:
     `src/peft/utils/hotswap.py:_convert_scalings_to_tensor` (line 56),
     `_pad_lora_weights` (line 230), `prepare_model_for_compiled_hotswap`
     (line 268).
   - **Two changes needed in `networks/lora.py:LoRAModule`**:
     (a) Convert `self.scale` from a Python float to a registered buffer so
     the compiled graph sees a tensor of fixed dtype, not a recompile-
     triggering scalar. (b) Pad `lora_down`/`lora_up` to a configurable
     `target_rank` so a smaller-rank adapter can hotswap into a graph
     compiled for a larger one.
   - **One new entry point**:
     `LoRANetwork.hotswap_from_state_dict(new_sd)` that copies weights into
     the existing tensors in place (no `setattr`, no buffer reassignment —
     either of those triggers recompile).

2. **Orthogonal LoRA initialization** (single new option).
   _Status: only kaiming_uniform_ is wired (`networks/lora.py:160, 172`)._
   - **PEFT reference**: `/home/dustin/peft/src/peft/tuners/lora/layer.py:498`
     (the `"orthogonal"` branch of the init dispatch). One `torch.linalg.qr`
     call on `lora_down`, leaves `lora_up` zeros — zero base-weight mutation.
   - **Why this and not PiSSA**: PiSSA / OLoRA mutate the base DiT weights
     at init (PiSSA at `peft/.../layer.py:360`, OLoRA at `:315`). Doing that
     safely requires a base-weight hash in the safetensors metadata plus a
     hash check in `merge_lora.py`. Ship orthogonal first; defer PiSSA to
     its own design pass. (See Tier 2 #2a for the deferred PiSSA roadmap.)
   - **Cost**: ~50 LOC plus an `--init_lora_weights {kaiming,orthogonal}`
     CLI knob and a CPU deterministic test pinning the QR result.

3. **Safe-merge finite checks** as opt-in (offline CLIs) and default
   (interactive merges).
   _Status: no merge path validates `torch.isfinite` before committing._
   - **PEFT reference**:
     `/home/dustin/peft/src/peft/tuners/lora/layer.py:817` — the
     `safe_merge=True` path clones, merges, runs `torch.isfinite(...).all()`,
     then commits only on pass. Otherwise raises with the offending key.
   - **Wire-in points**:
     `lora_utils.py:lora_merge_weights_to_tensor` (gets a `safe_merge` kwarg
     in the Tier 0 fix), `merge_lora.py` CLI (default-on), `merge_to` in
     `networks/lora.py`, `loha.py`, `lokr.py`. Same shape across all four.
   - **Why this matters more than it sounds**: bf16 / fp16 base + a
     poorly-trained adapter can produce NaN deltas that silently corrupt the
     merged checkpoint. Currently the user only finds out at generation time
     when the model emits black frames. Catching it at merge time saves a
     debugging session.

4. **Static `rank_pattern` / `alpha_pattern`** for per-layer rank/alpha
   overrides (regex → int dicts).
   _Status: blissful-tuner has include/exclude regex filters but no
   per-layer rank or alpha map._
   - **PEFT reference**: `LoraConfig(rank_pattern={"to_q": 32, "to_v": 16})`
     in `src/peft/tuners/lora/config.py`. Applied in `lora/model.py`
     `_create_and_replace` per-module.
   - **Why this is a stepping stone before AdaLoRA**: gives users a manual
     equivalent of "spend rank where it matters" without the trainer-step
     integration AdaLoRA needs. Diffusion users frequently want larger rank
     on attention than on MLP — currently impossible without two separate
     networks.
   - **Cost**: ~80 LOC in `LoRANetwork.create_modules` to consume the regex
     dict before instantiating each `LoRAModule`. Persist as a `dict` field
     in the network metadata; loader reads it back.

### Tier 2 — higher cost, real upside, needs a design pass

5. **Adapter merge algebra** (TIES, DARE, SVD, magnitude pruning) for
   `lora_post_hoc_ema.py` family.
   _Status: v1 shipped; v1.5 #1 / #2 / #3 shipped (prune_threshold,
   output_use_rslora, fold_into); v1.5 #4 (`--base_dit` for DoRA input
   materialization) implemented and real-weights validated locally,
   pending commit split. EMA averaging still exists separately._
   - **PEFT reference**: `/home/dustin/peft/src/peft/utils/merge_utils.py:185`
     (`ties`/`dare`/`task_arithmetic`) — pure tensor algebra, **no PEFT
     runtime dependency**, transcribed into a standalone offline CLI.
   - **Locked v1 shape**:
     1. Operate on materialized deltas in fp32 CPU, one module at a time.
     2. Support `linear`, `ties`, `dare_linear`, and `dare_ties`.
     3. Recompress output to standard LoRA safetensors via SVD at explicit
        `--output_rank`; no `--method svd` peer method.
     4. Accept rsLoRA inputs by baking their scale into the materialized
        delta; output standard LoRA (`alpha / rank`) without
        `use_rslora_flag`.
     5. Reject DoRA, LoHa, LoKr, hybrid, split-dims, and unknown inputs
        during preflight with actionable errors. DoRA support deferred
        until v1.5 #4.
   - **v1.5 output modes**:
     1. `--prune_threshold` skips near-zero materialized deltas before output.
     2. `--output_use_rslora` writes rsLoRA-shaped LoRA output when requested.
     3. `--fold_into <base.safetensors>` switches from LoRA-output to
        checkpoint-output mode: materialized deltas are added into matching
        base tensors at full rank, the result is written atomically as a full
        base-shaped safetensors file, and LoRA-only recompression metadata is
        omitted.
     4. `--base_dit <base.safetensors>` enables DoRA input materialization
        with standard-LoRA output (v1.5 #4). Calls the production runtime
        DoRA helper `dora_weight_norm_materialized` to compute the
        direction × magnitude decomposition correctly, returns the resulting
        delta `merged_weight - base_weight` to the merge algebra, and
        SVD-recompresses back to a standard LoRA. Lossy (DoRA magnitude
        vector discarded on output) but interoperable with every loader.
        Mutually exclusive with `--fold_into`. Compatible with
        `--output_use_rslora`. Linear-only; Conv2d DoRA mirrors the
        production helper's `NotImplementedError` as a clean preflight
        rejection. Full DoRA output (with re-derived magnitudes) deferred
        to a hypothetical v1.5 #5.
   - **Implemented CLI shape**: `tools/merge_loras_algebra.py` takes repeated
     `--input PATH WEIGHT` pairs plus `--method`, method-specific args, and
     `--output_rank`, then emits a normal blissful-tuner/Kohya-format
     `.safetensors` with provenance metadata.

6. **PiSSA / OLoRA initialization (deferred from Tier 1).**
   _Status: not implemented; gated on metadata + safety-check infrastructure
   that doesn't exist yet._
   - **PEFT references**: `/home/dustin/peft/src/peft/tuners/lora/layer.py:360`
     (PiSSA), `:315` (OLoRA). Both are SVD-on-base then split-out
     initialization that *mutates the base DiT weights at init time*.
   - **Prerequisite work** before this is safe to ship:
     1. Persist a base-weight hash in the LoRA safetensors metadata
        (e.g. `ss_base_sha256` for the DiT file) at save time.
     2. `merge_lora.py` reads the hash and refuses to merge if the user-
        supplied base doesn't match. Same check in the Tier 0 fast helper.
     3. Explicit metadata flag like `ss_lora_init=pissa` so downstream
        tooling can render warnings or special-case behavior.
   - **Why deferred**: an init that mutates base weights is a foot-gun
     without the hash check. A user training PiSSA on `flux2-base.safetensors`
     and then merging into `flux2-base-finetuned.safetensors` gets silently
     wrong results.

7. **AdaLoRA rank allocator** for adaptive `--network_dim`.
   _Status: blissful-tuner's `--network_dim` is a fixed scalar; users
   routinely overprovision. (See Tier 1 #4 for a static-pattern stepping
   stone before tackling this.)_
   - **PEFT reference**: `src/peft/tuners/adalora/model.py:RankAllocator`
     plus the per-step prune/grow step.
   - **Cost**: rewrites the LoRA layer to the SVD-style triple
     (`lora_A` / `lora_E` / `lora_B`); a hook is needed in
     `NetworkTrainer.train()` between optimizer step and zero-grad
     (`hv_train_network.py:train()` mainline).
   - **Mutual-exclusion check**: verify whether AdaLoRA + DoRA is supported
     in PEFT (the SVD triple may not commute with magnitude decomposition).
     If incompatible, this becomes a config-validation concern for the
     existing DoRA users.

8. **OFT / BOFT / HRA — orthogonal-update adapters for diffusion
   experimentation.**
   _Status: blissful-tuner has none of these._
   - **PEFT references**: `src/peft/tuners/oft/layer.py` (Conv2d-supporting,
     Cayley–Neumann); `src/peft/tuners/boft/layer.py` (butterfly, has CUDA
     kernel `boft/fbd/fbd_cuda.cu`); `src/peft/tuners/hra/layer.py`
     (Householder reflections, ~2r params/layer).
   - **Why interesting for diffusion**: orthogonal updates *can* preserve
     smoothness better than additive LoRA — empirically tied to the "LoRA
     collapses style after long training" failure mode common in video
     finetuning.
   - **Gate on a real eval** before integration. The untracked
     `tools/lora_eval/inventory.py` looks like the right harness.
   - **BOFT cost note**: `boft/fbd/fbd_cuda.cu` is pre-built for sm_8x. An
     sm_120 rebuild is required (similar shape to the existing
     `flash-attention-sm120` work).

### Tier 3 — research / observability, not training-path changes

9. **Intruder-dimension diagnostics**.
   _Status: no LoRA-quality metric in `tools/lora_eval/`._
   - **PEFT reference**: `/home/dustin/peft/src/peft/tuners/lora/intruders.py:20`
     — identifies LoRA singular vectors that don't align with the base
     model's weight subspace (proxy for "this adapter is overfitting away
     from the base"). Could feed `tools/lora_eval/` as a pruning score.
   - **Integration**: feed into `tools/lora_eval/inventory.py` as a per-
     adapter score. No trainer changes required.

10. **Adapter status reporting pattern** (PEFT's `get_layer_status()` /
    `get_model_status()`).
    - **PEFT reference**: `src/peft/peft_model.py:1130, 1160`.
    - Lift the *pattern* — a structured per-layer dataclass with
      `enabled`, `active_adapters`, `merged`, `trainable`, `device`,
      `dtype`, `rank`, `alpha`, `use_dora`/`use_rslora` flags. Useful for
      multi-architecture LoRA debugging and as part of `tools/lora_eval/`
      manifests. Dovetails with `BlissfulLogger`.

11. **`target_parameters`** for MoE-style `nn.Parameter` experts.
    - Not relevant today — none of WAN, FLUX.2, Qwen-Image, Z-Image,
      Kandinsky 5, HV1.5 use parameter-as-expert layouts. Worth knowing
      about if a future Llama-4-style MoE-DiT hybrid lands.

### Out-of-band audit (not a feature, but worth one investigation)

A. **`merge_lora.py` + `--base_weights` + block-swap interaction** in
   `src/musubi_tuner/hv_train_network.py:2034`.
   - **Concern**: standalone `src/musubi_tuner/merge_lora.py:109` loads on
     CPU and merges before saving — that path is fine. Generation paths
     also typically merge before block swap. But `--base_weights` in
     training may set up block swap *before* merge runs, which would put
     some target modules on CPU at merge time.
   - **Action**: read the train-time path end-to-end; if confirmed,
     either reorder so merge runs before block-swap setup, or wrap the
     merge call in a "force-onload" context. PEFT's `onload_layer` is
     designed for the Accelerate `_hf_hook` offload pattern, which is
     not what blissful-tuner uses — so don't lift it directly; do the
     equivalent for blissful-tuner's `custom_offloading_utils.py`
     pattern instead.
   - **Why this is "out-of-band"**: it's a defensive read with a
     potential narrow fix, not a feature. Doesn't deserve a tier slot.

### Already shipped (kept for grep anchors)

- **DoRA** — see `networks/lora.py:213–219, :236–242, :248–340`. Tested by
  `tests/test_lora_dora_init_device.py`. Diverges from PEFT in dropout
  handling, multiplier-zero short-circuit, and weight-norm computation
  (see "Where blissful-tuner deliberately diverged" above).
- **rsLoRA** — see `networks/lora.py:128, 187–193, 749–752, 967–985`.
  Format-validation flag stored as a state-dict buffer, not just sidecar
  config.
- **Tier 0 fast merge correctness for rsLoRA + DoRA** — commit
  `7b522b8` (2026-05-02). `lora_utils.py:lora_merge_weights_to_tensor`
  now reads `use_rslora_flag` and `use_dora_flag` from saved state
  dicts and mirrors `networks/lora.py:merge_to` math. Tested by
  `tests/test_lora_merge_weights_rslora_dora.py` (16 tests). See
  `docs/plans/2026-05-02-peft-tier0-merge-fix.md`.
- **Tier 1 #1 compile-friendly LoRA hotswap (WAN slice)** — commits
  `3a541ba` / `762f8da` / `499d9c7` (2026-05-03). Opt-in
  `--prepare_for_hotswap` flag enables `param.data.copy_()`-based LoRA
  swap without recompiling per-block torch.compile graphs. Phase 1:
  WAN-only, standard LoRA only, rejects `--prefer_lycoris` /
  `--fp8_scaled` / `--save_merged_model` at parse time. Tested by
  `tests/test_lora_hotswap.py` (36 tests, including CUDA-required
  Dynamo recompile probe). **Real-weights bit-identical A/B verified
  2026-05-03** — see `docs/plans/2026-05-02-peft-tier1-hotswap.md`
  "Validation summary." Other architectures (FLUX.2, Z-Image, etc.)
  deferred to Phase 2.

## What is *not* worth integrating

- **The `PeftModel` wrapper itself.** It's transformers-shaped (TaskType,
  `prepare_inputs_for_generation`, generation-config plumbing) and would
  conflict with blissful-tuner's training wrapper that already speaks DiT.
  If anything, use `inject_adapter_in_model()` (the lower-level entry) —
  it's about 20 lines and just calls `BaseTuner.inject_adapter`.
- **NLP-only methods** (prefix/prompt tuning, p-tuning, multitask, CPT,
  cartridge, IA3, ln_tuning, trainable_tokens, adaption_prompt). They tune
  token embeddings or KV cache prefixes — wrong shape for image/video
  diffusion.
- **VeRA / RandLoRA**. Linear-only — would *silently skip* Conv2d
  projections present in FLUX.2 attention `proj_out` and similar paths.
  The ~1% param-savings claim is dishonest in a setting where the skipped
  Conv2d is where most adaptation happens.
- **X-LoRA** without first solving "what does dynamic adapter mixing mean
  for a single-prompt diffusion sample?" Token-level mixing doesn't have an
  obvious diffusion analog. Per-timestep mixing is plausible but net-new
  research, not lifting.
- **Tiny / Lily / SHIRA / GraLoRA / Peanut / Miss / RoaD / WaveFT /
  FourierFT / Adamss / C3A / DeloRA / PsoFT / PVeRA / BdLoRA / Beft.**
  Recent compact-parameter variants, all Linear-only, all tuned for NLP
  parameter-budget benchmarks. None demonstrated on diffusion at the time
  of this review. Re-audit if a paper shows one outperforming LoRA on a
  diffusion benchmark.

## Common commands (in `/home/dustin/peft/`)

```bash
# Lint (PEFT uses ruff, line-length 119)
make quality                                # check
make style                                  # autoformat
ruff check src tests examples docs scripts docker
ruff format src tests examples docs scripts docker

# Test (uses pytest-xdist with -n 3)
make test
python -m pytest tests/                     # all
python -m pytest tests/test_initialization.py            # one file
python -m pytest tests/test_lora.py -k "test_dora"       # filter

# Single-GPU bnb / common-GPU subsets
make tests_examples_single_gpu
make tests_common_gpu

# Coverage already enabled via pyproject (--cov=src/peft)
```

PEFT setup.py declares `python_requires=">=3.10.0"` and `torch>=1.13.0`,
so it imports fine in `venv314` against the editable torch (2.13.0a0).

## Where to look first (file:line cheat sheet)

| Question | File |
|----------|------|
| All public exports | `src/peft/__init__.py` |
| Top-level entry | `src/peft/mapping_func.py:get_peft_model` |
| Wrapper model | `src/peft/peft_model.py:PeftModel` |
| Injection algorithm | `src/peft/tuners/tuners_utils.py:BaseTuner.inject_adapter` |
| Offload-aware merge ctx | `src/peft/tuners/tuners_utils.py:onload_layer` (~line 75–140) |
| LoRA layer subclasses | `src/peft/tuners/lora/layer.py` (Linear, Conv2d, Embedding) |
| LoRA model orchestration | `src/peft/tuners/lora/model.py` |
| LoRA config (every knob) | `src/peft/tuners/lora/config.py` |
| DoRA implementation | `src/peft/tuners/lora/dora.py` + `variants.py` |
| Init variants (PiSSA/OLoRA/EVA/LoftQ/orthogonal) | `src/peft/tuners/lora/layer.py` (init dispatch) + `src/peft/utils/loftq_utils.py` + `src/peft/tuners/lora/eva.py` |
| AdaLoRA rank-allocator | `src/peft/tuners/adalora/model.py:RankAllocator` |
| OFT layer | `src/peft/tuners/oft/layer.py` |
| BOFT layer + CUDA kernel | `src/peft/tuners/boft/layer.py`, `src/peft/tuners/boft/fbd/fbd_cuda.cpp` |
| Save / load | `src/peft/utils/save_and_load.py` |
| Hotswap / compile-friendly swap | `src/peft/utils/hotswap.py` (`:56` scaling-as-buffer, `:230` rank pad, `:268` prepare entry) |
| Adapter merge algebra (TIES/DARE/SVD) | `src/peft/utils/merge_utils.py` |
| Intruder-dimension diagnostics | `src/peft/tuners/lora/intruders.py` |
| Layer/model status reporting | `src/peft/peft_model.py:1130, 1160` |
| Diffusers integration examples | `examples/lora_dreambooth/`, `oft_dreambooth/`, `boft_dreambooth/`, `hra_dreambooth/`, `boft_controlnet/` |

## How to use this file from blissful-tuner

When reasoning about whether to wire something from PEFT into blissful-tuner:

1. Locate the algorithm in PEFT using the cheat sheet above.
2. Cross-reference with the "Blissful-tuner already covers" table to confirm
   it's not duplicate work.
3. Check Conv2d support — diffusion DiT/UNet blocks use Conv2d for some
   projections (FLUX vae, UNet attention proj_out, etc.). VeRA-family
   methods that are Linear-only will silently skip those modules.
4. Trace state-dict keys — anything saved with PEFT keys
   (`lora_A.weight` / `lora_B.weight`) will not load cleanly into
   blissful-tuner's `merge_lora.py` or `convert_lora.py` flows without a
   key remapper. Plan that conversion *before* committing to PEFT-trained
   adapter outputs.
5. If considering a deeper integration than "lift one algorithm":
   `inject_adapter_in_model()` is the seam to use, not `get_peft_model()` —
   the latter wraps the model with transformers-shaped methods that don't
   fit blissful-tuner's `NetworkTrainer` flow.
