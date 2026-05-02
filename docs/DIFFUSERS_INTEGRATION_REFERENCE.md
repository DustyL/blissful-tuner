# Diffusers Integration Reference

This file provides guidance to Claude Code (claude.ai/code) for investigating
HuggingFace Diffusers (`~/diffusers`, version `0.38.0.dev0`, editable install)
as a source of techniques to integrate into the blissful-tuner LoRA training
pipeline.

> **Scope.** This is *not* a CLAUDE.md for blissful-tuner — that already exists
> at `/home/dustin/blissful-tuner/CLAUDE.md` and is authoritative for the
> project itself. This file is a focused codebase guide for *the diffusers
> repository at `/home/dustin/diffusers/`*, written from the perspective of
> someone in blissful-tuner asking "what's worth lifting?"
>
> **Sister documents:**
> - `docs/PEFT_INTEGRATION_REFERENCE.md` — same shape, focused on `~/peft`.
> - `docs/planning/DIFFUSERS_INTEGRATION_OPPORTUNITIES.md` — older (Jan 2026,
>   diffusers 0.37.0.dev0) but much deeper analysis of hook architecture,
>   group offloading, layerwise casting, and FasterCache with full code
>   examples. **Read that file when you need implementation detail; this file
>   is the navigation map.**

> **Last reviewed: 2026-05-02.** Diffusers HEAD `c8eba433a`
> (`[agents docs] update models.md with class attributes and attention mask`).
> Blissful-tuner HEAD `6e74139`. Highest-ROI gaps right now are LoRA-format
> converters (broader than blissful-tuner's `convert_lora.py`), inference
> caches (`first_block_cache`, `taylorseer_cache`, `pyramid_attention_broadcast`),
> NF4 / torchao base-model quantization for LoRA-on-quantized workflows, and
> diffing the vendored 0.29.2 schedulers against current upstream.

## Where diffusers lives in this stack

| Question | Answer |
|----------|--------|
| Source tree | `/home/dustin/diffusers/` |
| Editable install in `venv314` | `diffusers 0.38.0.dev0` (constrained by `~/.pip_editable_constraints.txt`) |
| Currently used by blissful-tuner training? | **Partially.** Blissful-tuner imports diffusers as a base-class library + a few utilities; its training loop, LoRA networks, scheduler set, attention dispatch, mask loss, EMA teacher, and per-arch caches are all first-party. |
| Used elsewhere in venv? | Yes — pulled in transitively by `transformers`, `accelerate`, anything that does `from diffusers import ...` for inference. The version installed editable is what those libs see. |
| Recent activity worth knowing | `c8eba433a` agents docs update; `ffd5da5f7` CI workflow permissions; `42a46e48c` VAE latents_bn_std dtype cast fix; `1a8a17b71` ACE-Step text-to-music pipeline merged; `303c1d8b0` Ernie-Image LoRA support. |

## What blissful-tuner imports from diffusers today

This is the **actual** dependency surface as of `6e74139`.
`grep -rn "from diffusers" /home/dustin/blissful-tuner/src/` returns ~50 hits
across ~19 files. Categorized:

### Building-block classes (inherited or composed)

```python
# Used in vendored VAE / scheduler / pipeline classes
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin, KarrasDiffusionSchedulers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor

# NN building blocks
from diffusers.models.attention_processor import SpatialNorm, Attention
from diffusers.models.normalization import RMSNorm, AdaGroupNorm
from diffusers.models.activations import get_activation
```

Sites:
- `src/musubi_tuner/hunyuan_model/autoencoder_kl_causal_3d.py:25–43`
- `src/musubi_tuner/hunyuan_model/pipeline_hunyuan_video.py:27–43`
- `src/musubi_tuner/modules/unet_causal_3d_blocks.py:27–31`
- `src/musubi_tuner/modules/scheduling_flow_match_discrete.py:24–26`
- `src/musubi_tuner/wan/utils/fm_solvers_unipc.py:10–14`

### Utilities

```python
from diffusers.utils.torch_utils import randn_tensor    # 7 callsites
from diffusers.utils import BaseOutput, is_torch_version, deprecate, is_scipy_available
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.optimization import (                     # LR schedulers
    get_constant_schedule, get_cosine_schedule_with_warmup, ...
)  # imported by hv_train_network.py:37 and hv_train.py:32
```

### Vendored "Modified from diffusers==0.29.2" forks

Diffusers code physically copied into blissful-tuner at version 0.29.2 with
attribution. Drift risk is the practical concern: bugs fixed upstream after
0.29.2 are not reflected.

| Blissful-tuner file | Forked from | Drift risk |
|---------------------|-------------|------------|
| `hunyuan_model/autoencoder_kl_causal_3d.py:16` | `diffusers==0.29.2` AutoencoderKL | **High** — VAE attention/norm internals have evolved |
| `hunyuan_model/pipeline_hunyuan_video.py:16` | `diffusers==0.29.2` pipeline base | **Medium** — pipeline API drift, callback shapes |
| `modules/unet_causal_3d_blocks.py:16` | `diffusers==0.29.2` UNet 3D blocks | **Medium** |
| `modules/scheduling_flow_match_discrete.py:16` | `diffusers==0.29.2` flow-match (discrete) | **High** — current diffusers has `use_dynamic_shifting` and `shift_terminal` |
| `qwen_image/qwen_image_utils.py:1056, 1118, 1242` | `FlowMatchDiscreteScheduler`, `retrieve_timesteps`, DPM-Solver helpers | Medium |
| `wan/utils/fm_solvers_unipc.py:148–757` | Multiple diffusers schedulers (FlowMatchEulerDiscrete, DDPM, DPMSolverMultistep) | **High** — many `# Copied from` markers; refresh-from-upstream is mechanical |

Workflow when fixing a bug in any of these: `git -C ~/diffusers log -p src/diffusers/<original_path>` to see what's changed since 0.29.2.

### Notably absent imports

These diffusers modules are **not** imported by blissful-tuner anywhere —
every one is a candidate area for the "Recommended integration targets"
section:

- `diffusers.loaders.*` — no LoRA loading or format conversion
- `diffusers.training_utils` — no `compute_snr`, `compute_density_for_timestep_sampling`, `EMAModel`, `cast_training_params`
- `diffusers.hooks.*` — none of `group_offloading`, `layerwise_casting`, `faster_cache`, `first_block_cache`, `pyramid_attention_broadcast`, `mag_cache`, `taylorseer_cache`, `text_kv_cache`, `layer_skip`
- `diffusers.quantizers.*` — no bnb / torchao / quanto / gguf / modelopt integration
- `diffusers.guiders.*` — blissful-tuner has its own CFGZero\* / NAG / perpendicular CFG

## What's in the box (diffusers component taxonomy)

All public exports come from `src/diffusers/__init__.py`. By directory:

### Models (`src/diffusers/models/`)

| Subdir | Highlight |
|--------|-----------|
| `transformers/` | One file per architecture. **All** blissful-tuner targets are present: `transformer_flux.py`, `transformer_flux2.py`, `transformer_qwenimage.py`, `transformer_z_image.py`, `transformer_kandinsky.py`, `transformer_hunyuan_video.py`, `transformer_hunyuan_video15.py`, `transformer_hunyuan_video_framepack.py`, `transformer_wan.py`, `transformer_wan_animate.py`, `transformer_wan_vace.py`, plus newer `ace_step_transformer.py` (text-to-music) and `transformer_hunyuanimage.py` |
| `autoencoders/` | VAE family — diffusers' `AutoencoderKL` is the upstream of blissful-tuner's vendored 0.29.2 copy in `hunyuan_model/autoencoder_kl_causal_3d.py` |
| `attention_processor.py` | Per-architecture attention processors (`FluxAttnProcessor`, `HunyuanAttnProcessor2_0`, `XFormersAttnProcessor`, `XLAFlashAttnProcessor2_0`, `PAG*` variants, ...). Different model from blissful-tuner's enum-string switch in `modules/attention.py:124–260` |
| `attention_dispatch.py` | Newer dispatch helpers — skim if proposing changes to attention dispatch |
| `normalization.py` | `RMSNorm`, `AdaGroupNorm`, `AdaLayerNorm`, `LayerNorm` etc. blissful-tuner imports `RMSNorm` and `AdaGroupNorm` here |
| `lora.py` | LoRA layer base classes; consumed by `loaders/` |
| `controlnets/` | Reference for new conditioning pipelines; not used by blissful-tuner |

### Schedulers (`src/diffusers/schedulers/`)

50+ samplers. Only the flow-match family is relevant for blissful-tuner's
targets:

| File | Highlight |
|------|-----------|
| `scheduling_flow_match_euler_discrete.py` | Current Flow-Match Euler. Has `use_dynamic_shifting` (constructor `:95`), `shift_terminal` (`:101`), `_stretch_and_shift_timesteps()` (`:262`), applied in `set_timesteps` (`:347–353`). **None of this exists in blissful-tuner's vendored 0.29.2 copy.** |
| `scheduling_flow_match_heun_discrete.py` | Heun variant of flow-match |
| `scheduling_flow_match_lcm.py` | LCM flavor |

The other ~50 schedulers (DDIM, DPM-Solver, Euler ancestral, EDM, consistency
models, …) are not in blissful-tuner's training path. `wan/utils/fm_solvers_unipc.py`
has multiple `# Copied from` markers referencing them, useful as a refresh-
from-upstream playbook.

### Loaders (`src/diffusers/loaders/`)

| File | Highlight |
|------|-----------|
| `lora_pipeline.py` (~260 KB) | Main LoRA load/save mixin. Delegates managed adapters to PEFT |
| `lora_conversion_utils.py` (~136 KB) | **Single highest-value file in the repo for blissful-tuner.** 18+ format converters covering kohya, civitai, BFL, FAL, xlabs, lumina2, hunyuan-video, hidream, ltxv, ltx2, qwen, flux2, z_image, wan-musubi |
| `lora_base.py` | Base LoRA class + alpha scaling |
| `peft.py` | PEFT adapter wrapping (incompatible with blissful-tuner's `networks.lora_*` design) |
| `transformer_flux.py`, `transformer_sd3.py` | Per-architecture LoRA loaders |
| `single_file_utils.py` | Safetensors / checkpoint loading |
| `ip_adapter.py` | IP-Adapter integration |
| `textual_inversion.py` | Embedding loaders |

### Hooks (`src/diffusers/hooks/`)

Pluggable model hooks — none currently used by blissful-tuner. **High-value**
target area; see `docs/planning/DIFFUSERS_INTEGRATION_OPPORTUNITIES.md` for
deep implementation patterns.

| File | Purpose |
|------|---------|
| `hooks.py` | Base class + state manager registry |
| `group_offloading.py` | Multi-block group CPU offload with prefetching |
| `layerwise_casting.py` | Per-layer dtype casting (generalizes blissful-tuner's `--fp8_base` / `--fp8_t5`) |
| `faster_cache.py` | KV-cache optimization |
| `first_block_cache.py` | Cache first transformer block only — strong default speedup |
| `pyramid_attention_broadcast.py` | PAB inference acceleration |
| `mag_cache.py` | Magnitude-aware step skipping |
| `taylorseer_cache.py` | TaylorSeer prediction-based cache (newer SOTA) |
| `text_kv_cache.py` | Text-side KV caching |
| `layer_skip.py` | Skip-layer-guidance training-time hook |
| `context_parallel.py` | Context parallelism |
| `smoothed_energy_guidance_utils.py` | SEG support |

### Quantizers (`src/diffusers/quantizers/`)

| Subdir | Highlight |
|--------|-----------|
| `bitsandbytes/` | NF4, fp4, 8-bit |
| `torchao/` | int8, fp8 via TorchAO |
| `quanto/` | int8, int4 |
| `gguf/` | GGUF format support |
| `modelopt/` | NVIDIA ModelOpt |
| `pipe_quant_config.py` | Unified pipeline-level quantization config |
| `auto.py`, `base.py`, `quantization_config.py` | Plumbing |

Blissful-tuner has only fp8 (`src/blissful_tuner/fp8_optimization.py`) +
bnb 8bit AdamW (`hv_train_network.py:572`). NF4-quantized base-model with
LoRA-on-top is a popular workflow that's currently impossible.

### Training utilities (`src/diffusers/training_utils.py`)

Single file, ~700 lines. Every helper relevant to blissful-tuner:

| Symbol | Line | Notes |
|--------|------|-------|
| `set_seed` | 56 | Multi-source seed |
| `compute_snr` | 76 | Min-SNR weighting (~30 LOC, self-contained) |
| `compute_confidence_aware_loss` | 113 | |
| `unet_lora_state_dict` | 297 | Extract LoRA-only weights from a UNet |
| `cast_training_params` | 316 | Cast model params to target dtype |
| `_set_state_dict_into_text_encoder` | 333 | Load LoRA into text encoder |
| `_collate_lora_metadata` | 352 | |
| `compute_density_for_timestep_sampling` | 360 | SD3-style timestep density (logit-normal / mode / uniform / cosmap) |
| `compute_loss_weighting_for_sd3` | 387 | sigma_sqrt / cosmap / uniform loss weights |
| `free_memory` | 405 | GC + cache clear |
| `offload_models` | 422 | |
| `parse_buckets_string`, `find_nearest_bucket` | 453, 481 | CLI-string bucket spec (different abstraction from blissful-tuner's TOML) |
| `get_fsdp_kwargs_from_accelerator`, `wrap_with_fsdp` | 497, 520 | FSDP support |
| `EMAModel` | 571 | EMA wrapper — **not graph-safe**; allocates new tensors per step |

### Guiders (`src/diffusers/guiders/`)

| File | Method |
|------|--------|
| `classifier_free_guidance.py` | Standard CFG |
| `classifier_free_zero_star_guidance.py` | CFGZero\* |
| `perturbed_attention_guidance.py` | PAG |
| `skip_layer_guidance.py` | SLG |
| `frequency_decoupled_guidance.py` | FDG |
| `magnitude_aware_guidance.py` | MAG |
| `smoothed_energy_guidance.py` | SEG |
| `tangential_classifier_free_guidance.py` | Tangential CFG |
| `adaptive_projected_guidance.py`, `_mix.py` | APG, APG-Mix |
| `auto_guidance.py` | Auto-select |

Mostly redundant with `src/blissful_tuner/guidance.py` (which already has
CFGZero\*, NAG, perpendicular CFG). PAG/SLG/FDG/MAG/SEG are inference-time
toys — not training-relevant.

### Other top-level dirs

| Dir | Purpose | Notes |
|-----|---------|-------|
| `pipelines/` | 100+ inference pipelines | blissful-tuner doesn't use diffusers pipelines for sampling — read for reference impl only |
| `modular_pipelines/` | Experimental component-based pipeline architecture | Not yet stable |
| `image_processor/`, `video_processor/` | Asset preprocessing | Reference for video frame layout / normalization |
| `callbacks/` | Pipeline callbacks | Imported by `hunyuan_model/pipeline_hunyuan_video.py:27` |
| `utils/` | torch_utils (`randn_tensor`), accelerate_utils, BaseOutput, deprecate | blissful-tuner imports `randn_tensor` 7 times |
| `examples/dreambooth/` | Per-architecture LoRA training scripts | Single-image dreambooth-style; doesn't fit kohya/musubi cache-then-train |
| `tests/` | `tests/lora/`, `tests/models/`, `tests/schedulers/` | Useful as parity oracles |

## Per-architecture cross-reference

For each architecture blissful-tuner trains:

| Architecture | blissful-tuner module | diffusers transformer | diffusers training script | LoRA converter |
|--------------|------------------------|------------------------|---------------------------|----------------|
| WAN 2.1 / 2.2 | `src/musubi_tuner/wan/` | `transformer_wan{,_animate,_vace}.py` | none | `loaders/lora_conversion_utils.py:1812, 2065` (incl. `_convert_musubi_wan_lora_to_diffusers` — inverse of blissful-tuner's emitter; useful as roundtrip oracle) |
| HunyuanVideo | `src/musubi_tuner/hunyuan_model/` | `transformer_hunyuan_video.py` | none | `:1566` |
| HunyuanVideo 1.5 | `src/musubi_tuner/hunyuan_video_1_5/` | `transformer_hunyuan_video15.py` | none | parity check via `lora_pipeline.py` |
| FramePack | `src/musubi_tuner/frame_pack/` | `transformer_hunyuan_video_framepack.py` | none | none in diffusers |
| Flux / Flux Kontext | `src/musubi_tuner/flux/` | `transformer_flux.py` | `train_dreambooth_lora_flux.py`, `train_dreambooth_lora_flux_kontext.py` | `:360, 918, 1043, 1344` |
| Flux.2 (incl. Klein 4B / 9B) | `src/musubi_tuner/flux_2/` | `transformer_flux2.py` | `train_dreambooth_lora_flux2{,_klein}{,_img2img}.py` | `:2320, 2462` |
| Z-Image (Turbo) | `src/musubi_tuner/zimage/` | `transformer_z_image.py` | `train_dreambooth_lora_z_image.py` | `:2647` |
| Qwen-Image (incl. Edit / Layered) | `src/musubi_tuner/qwen_image/` | `transformer_qwenimage.py` | `train_dreambooth_lora_qwen_image.py` | `:2193` |
| Kandinsky 5 | `src/musubi_tuner/kandinsky5/` | `transformer_kandinsky.py` | none (only `examples/kandinsky2_2/` inference) | none |
| ACE-Step (text-to-music) | not trained | `ace_step_transformer.py` | none | none |
| Lumina2 | not trained | (in models/transformers) | `train_dreambooth_lora_lumina2.py` | `:1741` |
| HiDream | not trained | (in models/transformers) | `train_dreambooth_lora_hidream.py` | `:2126` |
| SD3 | not trained | `transformer_sd3.py` | `train_dreambooth_lora_sd3.py` | `:152, 260, 316` (generic non-diffusers/PEFT) |

**Pattern:** for every architecture blissful-tuner trains, diffusers has the
model code but typically **no LoRA trainer for video models**. The dreambooth
scripts are image-only and don't fit the kohya/musubi cache-then-train flow.

## What blissful-tuner already covers (don't reinvent)

| Feature | Blissful-tuner location | Diffusers equivalent |
|---------|-------------------------|----------------------|
| LoRA / LoHa / LoKr networks | `networks/lora_*.py`, `networks/loha.py`, `networks/lokr.py` | `loaders/{lora_pipeline,peft}.py`, `models/lora.py` (delegates to PEFT) |
| LoRA format conversion (kohya / diffusers / comfy) | `convert_lora.py`, `networks/convert_*.py` | `loaders/lora_conversion_utils.py` (broader format graph — see Tier 1) |
| Training loop | `hv_train_network.py:NetworkTrainer` | each `examples/train_*.py` reimplements per-arch |
| Mask-weighted loss + prior preservation | `modules/mask_loss.py` | (none — diffusers doesn't have this) |
| EMA teacher (graph-safe, in-place swap) | `modules/lora_ema_teacher.py:LoRAEmaTeacher` | `training_utils.py:571` `EMAModel` (allocates per step — not torch.compile-safe) |
| Muon optimizer w/ Newton-Schulz | `optimizers/muon.py`, `muon_util.py` | (none) |
| Adafactor-fused (stochastic rounding for bf16) | `modules/adafactor_fused.py` | (uses `transformers.optimization.Adafactor`) |
| RexLR scheduler | `modules/lr_schedulers.py` | already imports `optimization.py` LR helpers |
| Timestep sampling (`uniform`/`sigmoid`/`shift`/`flux_shift`/`flux2_shift`/`qwen_shift`/`logsnr`/`qinglong_*`) | `hv_train_network.py:320, 997–1100` | `training_utils.py:360` `compute_density_for_timestep_sampling` (**blissful-tuner's docstring literally cites diffusers PR #8528 as the source**) |
| Attention dispatch (sdpa / FA2 / SageAttention / CuTe / xformers) | `modules/attention.py:124–260` (enum-string `attn_mode`) | `models/attention_processor.py` (per-arch processor classes — different abstraction) |
| FP8 quantization | `src/blissful_tuner/fp8_optimization.py` | `quantizers/{bitsandbytes,torchao,quanto,gguf,modelopt}/` |
| Custom CPU offloading (per-block) | `modules/custom_offloading_utils.py` | `hooks/group_offloading.py` (group-level) |
| 8-bit AdamW | `hv_train_network.py:572` (via `bitsandbytes.optim`) | (none built-in) |
| Sample-during-training | `hv_train_network.py:1280 sample_images()` (multi-GPU dispatch via `PartialState`) | each example script rolls its own |
| Latent / text-encoder cache (per-arch) | `cache_latents.py`, `cache_text_encoder_outputs.py`, per-arch variants | (none — diffusers doesn't pre-cache) |
| LyCORIS bridge | `networks/lycoris.py` | (no LyCORIS bridge in diffusers) |
| CFGZero\*, NAG, perpendicular CFG | `src/blissful_tuner/guidance.py` | `guiders/classifier_free_zero_star_guidance.py` (only CFGZero\* overlap) |

If the goal were "replace blissful-tuner's training stack with diffusers,"
mask-loss/prior-preservation, the per-arch cache pipeline, and LoRA-format
roundtrips with kohya/comfy are blockers. Realistic integration is selective
lifting (below).

### Where blissful-tuner deliberately diverged from diffusers' design

These are not "missing features" — intentional patches future PRs proposing
"let's just use diffusers' version" should not silently revert.

| Divergence | Blissful-tuner | Diffusers | Reason to keep blissful-tuner's version |
|------------|----------------|-----------|------------------------------------------|
| EMA strategy | `LoRAEmaTeacher` does **in-place parameter swap** with no new tensor allocation (`modules/lora_ema_teacher.py`) | `EMAModel` clones state at each `.step()` (`training_utils.py:571`) | The in-place swap is `torch.compile`-graph-safe. Allocating per-step would invalidate the compiled graph at every EMA update on the WAN A14B / FLUX.2 trainers. |
| Attention backend dispatch | Single `attn_mode` enum string: `sdpa` / `flash` / `cute` / `sage` / `xformers`; runtime probe of CuTe (`probe_cute_runtime` at `modules/attention.py:50–95`) | Per-arch processor *class* (e.g. `FluxAttnProcessor`, `HunyuanAttnProcessor2_0`) | Diffusers' model is more flexible per-arch but requires a new processor class for every backend × arch pair. The enum-string switch carries one CuTe code path that handles all DiTs, important for sm_120 where CuTe is the only competitive kernel. |
| Vendored 0.29.2 schedulers | `wan/utils/fm_solvers_unipc.py`, `modules/scheduling_flow_match_discrete.py` | Current `scheduling_flow_match_euler_discrete.py` with `use_dynamic_shifting` | Vendored copies were taken to avoid pinning the entire training stack to whatever diffusers version was current at fork. Refresh is mechanical (`# Copied from` markers); replacement is not. |
| Sample-during-training is a trainer responsibility | Single `sample_images()` in `hv_train_network.py:1280` with multi-GPU dispatch baked in | Each example script reimplements (no shared utility) | Centralization is what makes "sample on rank 0 only" / `PartialState` dispatch / fp8/blocks-to-swap toggling correct without per-arch boilerplate. |

## Recommended integration targets (highest value first)

### Tier 1 — high value, low cost, no architectural conflict

1. **Selective LoRA format converters** from `loaders/lora_conversion_utils.py`.
   _Status: blissful-tuner has its own per-arch converters in `networks/convert_*.py` and `convert_lora.py` covering musubi ↔ diffusers ↔ comfy for some archs._
   - **Diffusers has more formats**: kohya (`:360`), xlabs (`:918`), BFL control (`:1043`), FAL Kontext (`:1344`), HunyuanVideo (`:1566`), Lumina2 (`:1741`), WAN non-diffusers (`:1812`), WAN-musubi (`:2065` — *the inverse of what blissful-tuner emits*; perfect roundtrip oracle), HiDream (`:2126`), LTXV (`:2134`), LTX2 (`:2142`), Qwen (`:2193`), Flux2 non-diffusers (`:2320`), Kohya Flux2 (`:2462`), Z-Image (`:2647`).
   - **Lift order**: start with `_convert_musubi_wan_lora_to_diffusers` as a roundtrip test to validate blissful-tuner's WAN emitter; then pick whichever new formats the user community requests (BFL/FAL for Kontext are the most-asked).
   - **Effort: low.** Each `_convert_*` function is pure-tensor key remapping with no diffusers-runtime dependency; copy into `src/blissful_tuner/lora_format_utils.py` essentially verbatim.

2. **`use_dynamic_shifting` + `_stretch_and_shift_timesteps` from current `scheduling_flow_match_euler_discrete.py`**.
   _Status: blissful-tuner's vendored 0.29.2 copy at `modules/scheduling_flow_match_discrete.py` lacks both. WAN-specific UniPC at `wan/utils/fm_solvers_unipc.py` has multiple `# Copied from` markers tied to 0.29.2._
   - **Diffusers references**: `scheduling_flow_match_euler_discrete.py:95` (constructor arg `use_dynamic_shifting`), `:101` (`shift_terminal`), `:262` (`_stretch_and_shift_timesteps()`), applied in `set_timesteps` at `:347–353`.
   - **Why now**: Flux.2 and FLUX-family training at variable resolutions benefits materially from dynamic shift. Without it, `--discrete_flow_shift 12.0` becomes an arch-specific magic number rather than a function of resolution.
   - **Effort: low.** Method is ~30 LOC of pure-tensor math.

3. **`compute_snr` for min-SNR loss weighting**.
   _Status: blissful-tuner doesn't expose min-SNR; loss weighting is currently MSE / Huber + mask only._
   - **Diffusers reference**: `training_utils.py:76` (~30 LOC, self-contained).
   - **Integration site**: `modules/loss_utils.py:compute_unreduced_target_loss` already gates on Huber vs MSE; add a `min_snr_gamma` arg threading through to a new `_apply_min_snr_weighting()`.
   - **Effort: low.**

4. **`hooks/layerwise_casting.py` as a generalization of `--fp8_base` / `--fp8_t5`**.
   _Status: blissful-tuner's `src/blissful_tuner/fp8_optimization.py` works but is binary (cast or don't). Mixed fp4/fp8/bf16 across the model requires per-arch code today._
   - **Diffusers reference**: `hooks/layerwise_casting.py` — registers a forward hook that casts dtype per-layer based on a config dict. **Detailed walkthrough in `docs/planning/DIFFUSERS_INTEGRATION_OPPORTUNITIES.md` §3** (lines ~426–588).
   - **Effort: medium** — needs `NetworkTrainer` integration for the cast to apply during the training forward, not just inference.

### Tier 2 — higher cost, real upside, needs a design pass

5. **NF4 / torchao base-model quantization with LoRA-on-top**.
   _Status: blissful-tuner has fp8 + bnb 8bit AdamW only._
   - **Diffusers references**: `quantizers/bitsandbytes/` (NF4 + fp4), `quantizers/torchao/` (int8/fp8), `quantizers/pipe_quant_config.py` (unified config pattern).
   - **Why interesting**: NF4 base + bf16 LoRA training is the dominant memory-saving recipe in the community right now (LoRAs trained against quantized base run on consumer GPUs).
   - **Cost**: needs careful interaction with `--blocks_to_swap`, `--fp8_base`, and the LoRA forward (the LoRA delta must be applied at the *dequantized* output of the base layer, not the quantized weight). Plan to gate behind a flag and validate on FLUX.2 before adding to other archs.
   - **Mutual-exclusion check**: NF4 + DoRA — diffusers / PEFT may have already solved this; cross-reference `~/peft/tuners/lora/dora.py` before designing.

6. **`hooks/group_offloading.py` for very-large-model training**.
   _Status: blissful-tuner's `--blocks_to_swap N` works but is per-block CPU offload via `modules/custom_offloading_utils.py`._
   - **Diffusers reference**: `hooks/group_offloading.py` does multi-block group offload with prefetching. **Walkthrough in `docs/planning/DIFFUSERS_INTEGRATION_OPPORTUNITIES.md` §2** (lines ~236–426).
   - **Why interesting**: WAN A14B + Flux.2 9B at high context length push the per-block swap threshold; group offloading consolidates I/O.
   - **Cost**: behavioral parity with blissful-tuner's existing per-block path needs an A/B; not a drop-in.

7. **Inference cache hooks for `*_generate_*.py` scripts**:
   `hooks/first_block_cache.py`, `hooks/taylorseer_cache.py`,
   `hooks/faster_cache.py`, `hooks/pyramid_attention_broadcast.py`.
   _Status: none exist in blissful-tuner. Sample-during-training and standalone inference both miss this entire optimization class._
   - **Why interesting**: real wall-clock improvements (typically 1.5–2× on long video sampling). `first_block_cache` is the safest default; `taylorseer_cache` is the newer SOTA.
   - **Effort: medium**. Each cache is a hook on the transformer's forward. Integration site is the per-arch `*_generate_*.py` after the model is loaded, before sampling. **FasterCache walkthrough in `docs/planning/DIFFUSERS_INTEGRATION_OPPORTUNITIES.md` §4** (lines ~589+).

### Tier 3 — research / observability, not training-path changes

8. **`compute_density_for_timestep_sampling` parity audit** vs.
   `hv_train_network.py:320`. The blissful-tuner copy explicitly cites
   diffusers PR #8528 as its origin and has diverged since. Diff against
   current diffusers to pick up upstream fixes for `sigma_sqrt`, `cosmap`,
   `mode` weighting math. **Effort: low** (one-shot audit). **Value: low**
   in absolute terms — most users don't change `--weighting_scheme`.

9. **`wrap_with_fsdp` + `get_fsdp_kwargs_from_accelerator`** for multi-GPU
   full fine-tune. _Status: blissful-tuner uses Accelerate but doesn't
   expose FSDP knobs cleanly._ Worth reading if a user actually requests
   multi-GPU full fine-tune of HunyuanVideo / FLUX.2 9B; otherwise dormant.

10. **`hooks/layer_skip.py`** for SLG-style training-time experiments.
    No production case yet; useful research scaffold if SLG papers land.

### Already shipped (kept for grep anchors)

- **LoRA / LoHa / LoKr networks** with per-arch target modules
  (`networks/lora_*.py`, `networks/loha.py`, `networks/lokr.py`,
  `networks/network_arch.py` — 13 arch variants). Diffusers' equivalent is
  PEFT-mediated.
- **Mask-weighted loss + prior preservation** (`modules/mask_loss.py`,
  `modules/prior_scheduling.py`, `modules/lora_ema_teacher.py`). No
  diffusers equivalent.
- **Per-arch cache scripts** (`*_cache_latents.py`,
  `*_cache_text_encoder_outputs.py`). No diffusers equivalent.
- **Custom offloading** (`modules/custom_offloading_utils.py`). Per-block,
  superseded by Tier 2 #6 if/when group_offloading is integrated.

## What is *not* worth integrating

- **The `examples/dreambooth/train_*.py` scripts as a whole.** Single-image,
  accelerate-launched, no pre-cache step. Doesn't fit the kohya/musubi
  cache-then-train architecture. Cherry-pick patterns at most.
- **`loaders/peft.py` and PEFT integration in diffusers.** Blissful-tuner's
  `networks.lora_*` with mask-loss-aware merging, LyCORIS bridge, and
  rsLoRA / DoRA flag-as-buffer is incompatible. Replacing it would break
  LoRA / LoHa / LoKr factor persistence.
- **`training_utils.py:571` `EMAModel`.** `LoRAEmaTeacher` is graph-safe
  (in-place parameter swap, `torch.compile`-friendly). Diffusers' `EMAModel`
  allocates per-step. See "deliberately diverged" above.
- **`guiders/*` (most).** Blissful-tuner already has CFGZero\*, NAG,
  perpendicular CFG. Diffusers' PAG / SLG / FDG / MAG / SEG are inference-
  time toys, not training-relevant.
- **`pipelines/*` for sampling.** Blissful-tuner's per-arch
  `*_generate_*.py` scripts are already custom and integrate latent preview,
  CFG schedule, and CuTe attention. Diffusers pipelines wrap a different
  lifecycle (callbacks, IP-Adapter plumbing). Read for reference impl only.
- **`modular_pipelines/`.** Experimental upstream; don't bet on it yet.
- **`# Copied from` markers within diffusers.** These are diffusers' internal
  copy-paste discipline. Don't propagate them into blissful-tuner.

## Common commands (in `/home/dustin/diffusers/`)

```bash
# Lint (diffusers uses ruff)
make quality                                # check
make style                                  # autoformat
ruff check src tests examples scripts utils
ruff format src tests examples scripts utils

# Test
pytest tests/                                            # all (slow — ~hours)
pytest tests/lora/                                       # LoRA-only
pytest tests/schedulers/test_scheduler_flow_match_euler_discrete.py
pytest tests/models/test_models_transformer_flux.py
pytest tests/lora/test_lora_layers_flux.py -k "test_simple_inference"

# Quick existence checks (cheaper than running tests)
python -c "from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler; print('ok')"

# Build & install for a different venv (do NOT do this in venv314 —
# diffusers is already editable from this tree)
pip install -e .[quality,test]
```

Diffusers `setup.py` declares `python_requires=">=3.10.0"` and supports
torch >= 2.1, so it imports fine in `venv314` against the editable torch
(2.13.0a0).

## Where to look first (file:line cheat sheet)

| Question | File |
|----------|------|
| All public exports | `src/diffusers/__init__.py` |
| Top-level pipeline factory | `src/diffusers/pipelines/auto_pipeline.py` |
| Generic pipeline base | `src/diffusers/pipelines/pipeline_utils.py:DiffusionPipeline` |
| LoRA load/save mixin | `src/diffusers/loaders/lora_pipeline.py` |
| **LoRA format converters** (the highest-value file for blissful-tuner) | `src/diffusers/loaders/lora_conversion_utils.py:152, 260, 316, 360, 918, 1043, 1344, 1566, 1741, 1812, 2065, 2126, 2134, 2142, 2193, 2320, 2462, 2647` |
| PEFT bridge | `src/diffusers/loaders/peft.py` |
| Training helpers | `src/diffusers/training_utils.py` (`compute_snr:76`, `compute_density_for_timestep_sampling:360`, `compute_loss_weighting_for_sd3:387`, `EMAModel:571`) |
| LR schedulers (already imported by blissful-tuner) | `src/diffusers/optimization.py` |
| Flow-match Euler scheduler (current) | `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:95, 101, 262, 347–353` |
| Hook base + state manager | `src/diffusers/hooks/hooks.py:27–167` |
| Group offloading | `src/diffusers/hooks/group_offloading.py` |
| Layerwise casting | `src/diffusers/hooks/layerwise_casting.py` |
| Inference caches | `src/diffusers/hooks/{first_block_cache,taylorseer_cache,mag_cache,faster_cache,pyramid_attention_broadcast,text_kv_cache}.py` |
| BnB / TorchAO / Quanto / GGUF / ModelOpt quantizers | `src/diffusers/quantizers/{bitsandbytes,torchao,quanto,gguf,modelopt}/` |
| Pipeline-level quant config | `src/diffusers/quantizers/pipe_quant_config.py` |
| Flux transformer | `src/diffusers/models/transformers/transformer_flux.py` |
| Flux.2 (incl. Klein) transformer | `src/diffusers/models/transformers/transformer_flux2.py` |
| Qwen-Image transformer | `src/diffusers/models/transformers/transformer_qwenimage.py` |
| Z-Image transformer | `src/diffusers/models/transformers/transformer_z_image.py` |
| WAN transformer (+ Animate, VACE) | `src/diffusers/models/transformers/transformer_wan{,_animate,_vace}.py` |
| HunyuanVideo / HV1.5 / FramePack transformers | `src/diffusers/models/transformers/transformer_hunyuan_video{,15,_framepack}.py` |
| Kandinsky 5 transformer | `src/diffusers/models/transformers/transformer_kandinsky.py` |
| ACE-Step (text-to-music, new) | `src/diffusers/models/transformers/ace_step_transformer.py` |
| Per-arch attention processors | `src/diffusers/models/attention_processor.py` |
| Building blocks blissful-tuner imports | `src/diffusers/models/normalization.py`, `models/activations.py`, `utils/torch_utils.py` (`randn_tensor`) |
| Example LoRA trainers (image-only) | `examples/dreambooth/train_dreambooth_lora_{flux,flux2,flux2_klein,flux_kontext,qwen_image,z_image,sd3,lumina2,hidream,sana}.py` |

## How to use this file from blissful-tuner

When reasoning about whether to wire something from diffusers into blissful-tuner:

1. **Check the import surface first.** `grep -rn "from diffusers" /home/dustin/blissful-tuner/src/`. If the symbol you're considering is already imported, the integration question is "extend the existing seam" rather than "add a new dependency edge."
2. **Cross-reference the "blissful-tuner already covers" table** to confirm
   it's not duplicate work. Several seemingly attractive diffusers utilities
   (timestep sampling helpers, EMA model, LR schedulers) are already
   covered, sometimes with deliberate divergence.
3. **For implementation depth, defer to `docs/planning/DIFFUSERS_INTEGRATION_OPPORTUNITIES.md`** —
   it has full code examples for hook architecture, group offloading,
   layerwise casting, and FasterCache. This file is the navigation map; that
   file is the construction guide.
4. **Watch the vendored-fork drift.** Anything in
   `hunyuan_model/`, `wan/utils/fm_solvers_unipc.py`,
   `modules/scheduling_flow_match_discrete.py`,
   `modules/unet_causal_3d_blocks.py`, or
   `qwen_image/qwen_image_utils.py` was forked from diffusers 0.29.2.
   Before "fixing a bug" in those files, check current diffusers — the fix
   is often already upstream.
5. **Trace state-dict keys for LoRA work.** Anything saved by diffusers
   (PEFT-style `lora_A.weight` / `lora_B.weight` keys, or pipe-emitted
   diffusers-format) will not load cleanly into blissful-tuner's
   `merge_lora.py` or `convert_lora.py` flows without a key remapper. The
   `loaders/lora_conversion_utils.py:_convert_*` functions are the seam to
   use, not raw `safetensors.load_file`.
6. **Don't try to swap `NetworkTrainer` for an `accelerate launch
   examples/train_*.py` flow.** The dreambooth scripts assume per-step
   forward of un-cached latents and text embeddings; blissful-tuner's
   architecture is fundamentally different (cache → train → optionally
   sample). Lift algorithms, not training loops.
