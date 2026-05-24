# Agent B — Blissful-Tuner MUST-KEEP Training Feature Inventory

Target: fold blissful's training features onto upstream `musubi-tuner` v0.3.0 (`upstream/main`, HEAD `78acafb`).
Merge-base: `8ec9577`. All paths under `/home/dustin/blissful-tuner/src/`. Read-only analysis; no files/git state modified.

**Headline structural fact:** Upstream v0.3.0 *refactored* the training stack. The monolithic `NetworkTrainer`
was extracted into a new `src/musubi_tuner/training/` package — `trainer_base.py` (2154 LOC) now hosts
`NetworkTrainer` (class at line 100, `train()` at line 1295), with `accelerator_setup.py`, `parser_common.py`,
`sampling_prompts.py`, `timesteps.py`. Upstream commit `0e6c554` is titled *"introduce extension seams on
NetworkTrainer (for Self-Flow + future extensions)"*. Blissful's `NetworkTrainer` is still inline at
`hv_train_network.py:496` inside a 3865-line file. **This extraction is the "new seam" every blissful training
feature must be re-expressed against.**

---

## 1. Training Features Table

| Feature | File(s) | LOC | Tests (fns) | Present upstream? |
|---|---|---|---|---|
| **Masked/weighted loss + prior preservation** | `modules/mask_loss.py` | 877 | 71 (across 14 files: `test_mask_loss.py`, `test_wan_mask_loss_integration.py`, `test_wan_mask_spatial_validation.py`, `test_mask_loss_disabled_warning.py`, `test_mask_loss_dtype_cast_regression.py`, `test_zimage_mask_weights_cache.py`, `test_mask_cache_metadata.py`, `test_cache_mask_preprocessing.py`, `test_cache_mask_wiring.py`, `test_mask_downsample_ringing.py`, `test_mask_enforcement.py`, `test_mask_weights_cache_dtype.py`, `test_loss_weighting_ndim.py`, `test_video_mask_duplicate_warning.py`) | **N** (`use_mask_loss`, `mask_gamma`, `prior_preservation_weight`, `mask_area_scale_beta`, `prior_decay_schedule`, `normalize_per_sample`, `prior_teacher_mode` → 0 upstream files; file absent) |
| **Unreduced MSE/Huber loss** | `modules/loss_utils.py` | 48 | 4 (`test_loss_utils.py`) | **N** (file absent) |
| **Timestep-adaptive prior scheduling** | `modules/prior_scheduling.py` | 61 | 3 (`test_prior_scheduling.py`) | **N** (file absent) |
| **EMA teacher (graph-safe)** | `modules/lora_ema_teacher.py` | 95 | 3 (`test_lora_ema_teacher.py`) | **N** (file absent) |
| **Muon optimizer + per-arch layer registry** | `optimizers/muon.py` (143), `optimizers/muon_util.py` (236) | 379 | 31 (`test_muon_optimizer.py`) | **N** (`optimizers/` dir absent upstream; 0 `muon` hits) |
| **WAN 2.2 compact time embedding + RoPE FIFO eviction** | inside `wan/modules/model.py` (`compact_time_embedding`, `_FREQS_CACHE_MAX_SIZE=512`, `freqs_fhw` FIFO) | model.py diff = +428/−171 vs upstream | 4 (`test_wan_compact_time_embedding.py`) | **N** for compact-embed (`compact_time` → 0 upstream hits). NOTE: distinct knob `--force_v2_1_time_embedding` IS upstream (`wan/modules/model.py:646/701`, `wan_train_network.py:484`) — do not conflate. |
| **Block-swap × no_grad restore** | `hv_train_network.py:623` (`restore_block_swap_after_no_grad_forward`), called `:2793`; sample-switch pair | within hv_train_network divergence | 7 (`test_block_swap_prior_restore.py`) + 1 manual (`tests/manual/test_flux2_block_swap_smoke.py`) | **N** on `upstream/main` (helper grep empty). Upstream has an *in-flight* branch `upstream/no-grad-on-block-swap` — NOT merged to main. |
| **PiSSA / orthogonal init + base-hash metadata** | `networks/lora.py` (1745 LOC; diff +874/−58), `utils/lora_utils.py` (974 LOC; diff +777/−99) | divergence | 86 across PiSSA/init files (`test_lora_pissa_init.py`, `test_pissa_training_validation.py`, `test_merge_lora_pissa_preflight.py`, `test_lora_orthogonal_init.py`) | **N** (`pissa`/`base_hash` → 0 upstream hits in `networks/lora.py`) |
| **LoRA invariants (target coverage, DoRA delta, Conv2d, split-dims/rank-dropout)** | `networks/lora.py` + flux2 variants | (in lora.py divergence) | 136 across 10 files (`test_lora_target_coverage.py`, `test_flux2_lora_target_coverage.py`, `test_lora_dora_delta_path.py`, `test_lora_conv2d_forward.py`, `test_lora_split_dims_rank_dropout.py`, `test_lora_dora_init_device.py`, `test_lora_rank_alpha_pattern.py`, `test_lora_hotswap.py`, `test_lora_eval_inventory.py`, `test_lora_merge_weights_rslora_dora.py`) | **N** (these are blissful hardening atop the diverged lora.py) |
| **xformers graceful degradation / multi-backend attention** | `modules/attention.py` (diff +209/−10 vs upstream) | divergence | 3 (`test_attention_backend_status.py`) | **N** on main (blissful version diverged). Upstream has in-flight `feat-multi-backend-attention` branch — not on main. |
| **Dynamo cache-entry recompile logging** | `hv_train_network.py:2990` (`count_dynamo_cache_entries`) | within hv divergence | — (no dedicated test file) | **N** |
| **Args injection + global behavior** | `blissful_tuner/blissful_core.py` (580) | 580 | — | **N** (`src/blissful_tuner/` package absent upstream entirely) |
| **FP8 quantization (DiT/T5)** | `blissful_tuner/fp8_optimization.py` (488) + `modules/fp8_optimization_utils.py` (diff +28/−16) | 488 + diff | — (utils diverged) | **N** for blissful fp8_optimization.py; `modules/fp8_optimization_utils.py` exists upstream but blissful diverged |
| **Advanced RoPE / RifleX pos-emb** | `blissful_tuner/advanced_rope.py` (107), `hvw_posemb_layers.py` | 107+ | — | **N** |
| **Rich logger** (imported by *every* train script) | `blissful_tuner/blissful_logger.py` | — | — | **N** (`from blissful_tuner.blissful_logger import BlissfulLogger` at `hv_train_network.py:69` and 40+ src files) |

### Already upstream — ZERO port cost (do not list as differentiators)
| Item | File | Status |
|---|---|---|
| **RexLR scheduler** | `modules/lr_schedulers.py` (67 LOC) | **BYTE-IDENTICAL** blissful HEAD ↔ upstream/main (diff empty). Comes free with any merge. |
| **Fused Adafactor** | `modules/adafactor_fused.py` (141 LOC) | **BYTE-IDENTICAL** (diff empty). Free. |
| **`force_v2_1_time_embedding`** | `wan/modules/model.py`, `wan_train_network.py`, `wan_generate_video.py` | Present upstream natively. (Compact time embedding is the separate, still-blissful-only knob.) |

### `src/blissful_tuner/` package (24 .py files) — split by relevance
- **Training-touching (must port / re-wire):** `blissful_core.py` (Args injection, 580), `blissful_logger.py` (imported by all train scripts), `fp8_optimization.py` (488), `advanced_rope.py` (107), `hvw_posemb_layers.py`, `common_extensions.py` (V2V/I2I noise prep — also used at train), `profiling.py` (VRAM tracking), `model_utility.py`.
- **Inference / post-processing (NOT required for a training-feature port):** `latent_preview.py`, `guidance.py`, `scheduling.py`, `prompt_management.py`, `taehv.py`, `taesd.py`, `facefix.py`, `upscaler.py`, `GIMMVFI.py`, `yolo_blur.py`, `video_to_png.py`, `metaview.py`, `video_processing_common.py`, `extract_lora.py`, `utils.py`.

---

## 2. Architecture Parity Table (upstream v0.3.0 vs blissful)

| Architecture | Model dir upstream? | `*_train_network.py` upstream? | blissful divergence (insertions/deletions) | Equivalent or ahead? |
|---|---|---|---|---|
| WAN 2.1/2.2 | ✅ `wan/` | ✅ | wan_train_network +239/−34; `wan/modules/model.py` +428/−171 | Blissful AHEAD (compact time embed, RoPE FIFO, mask loss wiring) |
| HunyuanVideo | ✅ `hunyuan_model/` | ✅ (`hv_train_network.py`) | +3467/−104 (hosts entire `NetworkTrainer` + all differentiators) | Blissful FAR AHEAD; also `hv_train.py` full-FT +56/−6 |
| HunyuanVideo 1.5 | ✅ `hunyuan_video_1_5/` | ✅ | hv_1_5_train_network +2/−4 (≈ parity) | Near-parity; inherits base differentiators |
| FramePack | ✅ `frame_pack/` | ✅ | fpack_train_network +18/−19 | Near-parity (no mask support either side) |
| FLUX.1 Kontext | ✅ `flux/` | ✅ | flux_kontext_train_network +12/−14 | Near-parity |
| FLUX.2 | ✅ `flux_2/` | ✅ | flux_2_train_network +263/−84 | Blissful AHEAD (block-swap×DoRA fixes, LoRA target coverage, mask loss) |
| Qwen-Image | ✅ `qwen_image/` | ✅ | qwen_image_train_network +104/−29; qwen_image_train (full-FT) +117/−11 | Blissful AHEAD (mask loss, full-FT) |
| Z-Image | ✅ `zimage/` | ✅ | zimage_train_network +25/−13; zimage_train (full-FT) +64/−17 | Blissful AHEAD (mask loss cache) |
| Kandinsky 5 | ✅ `kandinsky5/` | ✅ | kandinsky5_train_network +130/−112 | Near-parity / blissful slight-ahead |

**All 9 architecture families exist NATIVELY in upstream v0.3.0** — both the model dir and the `*_train_network.py`
script. None are stubs; upstream independently carries the full model zoo. Migration inherits architecture
scaffolding for free. Every train script nonetheless carries blissful divergence (the hooks that invoke mask loss /
prior teacher / EMA / logger), so the integration surface spans all 9 scripts, not just `hv_train_network.py`.

---

## 3. Bottom Line (4 sentences)

This is a **feature-port, not a model-port**: upstream v0.3.0 already ships all nine architecture families
(WAN, HunyuanVideo, HV1.5, FramePack, FLUX.1 Kontext, FLUX.2, Qwen-Image, Z-Image, Kandinsky 5) with full model
dirs and train scripts, plus it has *independently* absorbed RexLR and fused Adafactor byte-for-byte and added
`force_v2_1_time_embedding`, so those come along free. The irreducible "must re-express against new seams" surface
is dominated by upstream's `NetworkTrainer` refactor — blissful keeps a 3865-line monolithic `hv_train_network.py`
(+3467 lines vs upstream) while upstream extracted `NetworkTrainer` into a 2154-line `training/trainer_base.py`
with explicit extension seams, so the centralized mask-loss/prior/EMA/block-swap-restore logic must be re-hung off
that new base rather than dropped in as files. Around that core sit ~4400 LOC of standalone droppable feature
files (mask_loss 877, lora.py +874, lora_utils +777, muon 379, prior_scheduling/loss_utils/ema ≈200, attention
+209) that need re-wiring but not redesign, the entirely-absent 24-file `src/blissful_tuner/` package (only ~8 of
which are training-touching — `blissful_core` Args injection and `blissful_logger`, imported by all 9 train scripts,
are mandatory), and a per-architecture re-wiring tax across all 9 train scripts plus 3 full-FT trainers. Net: the
hard, non-mechanical work is re-expressing one diverged base trainer + 9 thin integration hooks against upstream's
new `training/` seam; the feature math itself (≈300+ tests guarding it) ports as discrete files.
