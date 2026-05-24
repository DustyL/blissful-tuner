# Blissful-Tuner Dead-Weight Analysis (READ-ONLY)

Scope: determine what is **safe to delete** given the user keeps only the TRAINING surface
(`*_train_network.py`, `hv_train.py`/`qwen_image_train.py`/`zimage_train.py`, all `*_cache_*` scripts,
and `src/musubi_tuner/{modules,dataset,networks,optimizers,utils}/`).

Method: built an AST import graph seeded from the training entry points (`/tmp/import_graph.py`),
cross-checked with `git grep` for every candidate. The training surface transitively reaches exactly
**14 of 177** `blissful_tuner` files.

---

## THE CENTRAL GOTCHA (read this first)

`wan_train_network.py`, `hv_train_network.py`, `fpack_train_network.py`, and `kandinsky5_train_network.py`
import small utility symbols **from three generate scripts**:

| Training file | Imports from generate script |
|---|---|
| `hv_train_network.py:67` | `from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid, resize_image_to_bucket, encode_to_latents` |
| `wan_train_network.py:12` | `from musubi_tuner.hv_generate_video import resize_image_to_bucket` |
| `wan_train_network.py:22` | `from musubi_tuner.wan_generate_video import parse_one_frame_inference_args` |
| `fpack_train_network.py:12` | `from musubi_tuner.fpack_generate_video import decode_latent` |
| `kandinsky5_train_network.py:408` | `from musubi_tuner.hv_generate_video import save_videos_grid` (function-local) |

Because Python executes the whole module on import, these 4-5 narrow imports drag the **entire inference
cluster** into the training graph: `hv_generate_video` → `wan_generate_video` → and through them
`latent_preview`, `guidance`, `scheduling`, `prompt_management`, `common_extensions`, `fp8_optimization`,
`blissful_core`, `taehv`, `taesd`, `video_processing_common`.

**Consequence:** a naive `git rm` of generate scripts or the inference cluster breaks training import.
The inference cluster is NOT used by training *logic* — it is only reachable because training borrows a
handful of helpers (image resizing, grid saving, one-frame arg parsing, latent decode) that happen to live
in generate scripts. To truly delete them, those ~5 helpers must first be relocated into a training-side
util module. Until then they must be treated as KEEP.

Dependency direction among the 3 training-needed generate scripts is clean:
`hv_generate_video` (leaf, imports no other generate script) ← `wan_generate_video` (imports only hv) ←
`fpack_generate_video` (imports hv + wan). The other 6 generate scripts import FROM these but nothing
imports them back — so the 6 are cleanly severable.

---

## 1. Classification Table

| Candidate | In training graph? | Imported by (real `import`, non-self) | LOC | Verdict |
|---|---|---|---|---|
| **Vendored dirs** | | | | |
| `gimmvfi/` | N | `GIMMVFI.py` only | 37,592 | SAFE-DELETE |
| `codeformer/` | N | `facefix.py` only | 11,772 | SAFE-DELETE |
| `gfpgan/` | N | `facefix.py` only | 4,317 | SAFE-DELETE |
| `swinir/` | N | `upscaler.py` only | 882 | SAFE-DELETE |
| `esrgan/` | N | (no importers at all) | 414 | SAFE-DELETE |
| `config_manager/` | N | only its own tests (`tests/test_config_manager/`) | 4,298 | SAFE-DELETE (standalone tool; delete its tests too) |
| **Post-processing modules** | | | | |
| `facefix.py` | N | (none) | 131 | SAFE-DELETE |
| `upscaler.py` | N | (none) | 162 | SAFE-DELETE |
| `GIMMVFI.py` | N | (none) | 222 | SAFE-DELETE |
| `yolo_blur.py` | N | (none) | 73 | SAFE-DELETE |
| `video_to_png.py` | N | (none) | 33 | SAFE-DELETE |
| `metaview.py` | N | (none) | 174 | SAFE-DELETE |
| `taehv.py` | **Y (transitive)** | `latent_preview.py` | 267 | KEEP* (only via inference cluster — see gotcha) |
| `taesd.py` | **Y (transitive)** | `latent_preview.py` | 151 | KEEP* (only via inference cluster) |
| `video_processing_common.py` | **Y (transitive)** | `common_extensions.py`, `GIMMVFI/facefix/upscaler/yolo_blur/video_to_png` | 491 | KEEP* (reached via `common_extensions`; the post-proc importers are themselves deletable) |
| **Inference cluster (reached only via generate scripts)** | | | | |
| `latent_preview.py` | **Y (transitive)** | 6 generate scripts + `kandinsky5/generation_utils.py` | 298 | KEEP* |
| `guidance.py` | **Y** | generate scripts **AND `wan/modules/model.py`** | 149 | **KEEP (core model)** |
| `advanced_rope.py` | **Y** | `hv_generate_video.py` **AND `wan/modules/model.py`** | 107 | **KEEP (core model)** |
| `hvw_posemb_layers.py` | **Y** | `advanced_rope.py` | 292 | **KEEP (core model)** |
| `scheduling.py` | **Y (transitive)** | `wan_generate_video.py` only | 1,033 | KEEP* |
| `prompt_management.py` | **Y (transitive)** | generate scripts + `blissful_core.py` | 338 | KEEP* |
| `common_extensions.py` | **Y (transitive)** | generate scripts | 319 | KEEP* |
| `fp8_optimization.py` | **Y (transitive)** | `hv_generate_video.py` only | 488 | KEEP* |
| `blissful_core.py` | **Y (transitive)** | generate scripts, `common_extensions`, `extract_lora` | 580 | KEEP (also Args injection) |
| **Standalone generate scripts (inference-only, severable)** | | | | |
| `flux_2_generate_image.py` (+root wrapper) | N | (no non-generate importer) | 1,497 + 4 | INFERENCE-ONLY (optional delete) |
| `flux_kontext_generate_image.py` (+root) | N | (none) | 1,312 + 4 | INFERENCE-ONLY (optional delete) |
| `hv_1_5_generate_video.py` (+root) | N | (none) | 1,410 + 4 | INFERENCE-ONLY (optional delete) |
| `kandinsky5_generate_video.py` (+root) | N | (none) | 492 + 4 | INFERENCE-ONLY (optional delete) |
| `qwen_image_generate_image.py` (+root) | N | (none) | 1,792 + 4 | INFERENCE-ONLY (optional delete) |
| `zimage_generate_image.py` (+root) | N | (none) | 1,409 + 4 | INFERENCE-ONLY (optional delete) |
| **Generate scripts that TRAINING needs (NOT deletable as-is)** | | | | |
| `hv_generate_video.py` | **Y** | 3 train_network files + 6 generate scripts | 1,042 | KEEP (training imports `resize_image_to_bucket`, `save_*_grid`, `encode_to_latents`) |
| `wan_generate_video.py` | **Y** | `wan_train_network.py` + 6 generate scripts | 2,602 | KEEP (training imports `parse_one_frame_inference_args`) |
| `fpack_generate_video.py` | **Y** | `fpack_train_network.py` | 2,142 | KEEP (training imports `decode_latent`) |
| **GUI** | | | | |
| `src/musubi_tuner/gui/` | N | not imported by any training file | 2,413 | INFERENCE/GUI-ONLY (optional delete; user said no GUI use) |

`*` = reachable from training **only** because training imports the 3 generate scripts above. These are not
used by training logic; they become SAFE-DELETE the moment the ~5 borrowed helpers are relocated and the
generate-script imports are dropped. Until then, deleting them breaks training import.

`guidance.py` / `advanced_rope.py` / `hvw_posemb_layers.py` are a different class — they are imported
**directly by `src/musubi_tuner/wan/modules/model.py`** (the WAN DiT, central to WAN training/cache).
These are unconditionally KEEP regardless of the generate-script question.

---

## 2. Reclaimable LOC

**Tier A — SAFE-DELETE now (no training reach, no relocation needed):**

| Group | LOC |
|---|---|
| Vendored dirs (gimmvfi, codeformer, gfpgan, swinir, esrgan, config_manager) | 59,275 |
| Post-proc modules (facefix, upscaler, GIMMVFI, yolo_blur, video_to_png, metaview) | 795 |
| **Tier A subtotal** | **~60,070** |

Note: deleting `config_manager/` should also drop `tests/test_config_manager/` (its only consumer).

**Tier B — INFERENCE-ONLY optional delete (user says they don't do inference):**

| Group | LOC |
|---|---|
| 6 standalone generate scripts (src + root wrappers) | 7,936 |
| GUI dir (`src/musubi_tuner/gui/`) | 2,413 |
| **Tier B subtotal** | **~10,349** |

**Tier C — unlocked only after relocating ~5 borrowed helpers out of the 3 kept generate scripts,
then deleting those scripts + the inference cluster:**

| Group | LOC |
|---|---|
| `hv_generate_video` + `wan_generate_video` + `fpack_generate_video` | 5,786 |
| Inference cluster: latent_preview 298 + scheduling 1,033 + prompt_management 338 + common_extensions 319 + fp8_optimization 488 + taehv 267 + taesd 151 + video_processing_common 491 | 3,385 |
| **Tier C subtotal** | **9,171** |
| (KEEP from cluster: guidance 149 + advanced_rope 107 + hvw_posemb_layers 292 + blissful_core 580 = 1,128 — these stay) |

**Total reclaimable:**
- Tier A only (zero-risk): **~60,070 LOC**
- Tier A + B (no inference/GUI): **~70,400 LOC**
- Tier A + B + C (after relocating the borrowed helpers): **~79,600 LOC**

---

## 3. SURPRISES / GOTCHAS (would break a naive `git rm`)

1. **`guidance.py` and `advanced_rope.py` are CORE MODEL code, not inference.**
   `src/musubi_tuner/wan/modules/model.py:23-24` does
   `from blissful_tuner.advanced_rope import apply_rope_comfy, EmbedND_RifleX` and
   `from blissful_tuner.guidance import nag`. `advanced_rope` in turn imports `hvw_posemb_layers`.
   These three are loaded by every WAN training and cache run. Anyone scanning for "guidance / RoPE /
   NAG = inference feature" and deleting them breaks WAN training immediately.

2. **Training imports generate scripts.** `hv_train_network`, `wan_train_network`, `fpack_train_network`,
   and `kandinsky5_train_network` pull utility symbols (`resize_image_to_bucket`, `save_videos_grid`,
   `encode_to_latents`, `parse_one_frame_inference_args`, `decode_latent`) out of `hv_generate_video` /
   `wan_generate_video` / `fpack_generate_video`. So those 3 generate scripts are NOT inference-only and
   cannot be deleted as-is. The clean fix is to relocate ~5 helper functions into a training-side util
   module first; only then do the 3 scripts + the whole inference cluster become deletable.

3. **The "inference cluster" (latent_preview, prompt_management, common_extensions, scheduling,
   fp8_optimization, taehv, taesd, video_processing_common) shows up as reachable from training**, but
   ONLY through gotcha #2. No `*_train_network.py` / `hv_train.py` / cache script imports any of them
   directly. They look training-essential in an import graph but are byproducts of the generate-script
   coupling — easy to misclassify in either direction.

4. **`blissful_core.py` is training-coupled by reputation but NOT directly imported by any training entry
   point.** Its training reach is again only via the generate scripts (which call `add_blissful_args` /
   `parse_blissful_args`). Its own direct dep is `from blissful_tuner.prompt_management import
   process_wildcards` (line 17). The `taehv`/`latent_preview` strings inside it are argparse help text and
   config-dict keys, not imports. Keep it (it owns Args injection), but note the actual edge is via
   generate scripts, not training scripts.

5. **`config_manager/` is a self-contained standalone tool** (4,298 LOC) — imported by nothing except its
   own `tests/test_config_manager/` suite (40+ references). `registry.py`'s many `latent_preview` hits are
   dict keys, not imports. Safe to delete, but delete its test dir in the same commit or CI breaks.

6. **`video_processing_common.py` is reached via `common_extensions` (a kept cluster member), not via the
   post-proc modules.** Its other importers (GIMMVFI/facefix/upscaler/yolo_blur/video_to_png) are all
   themselves SAFE-DELETE, so after Tier A it is reachable only through `common_extensions` — i.e. it
   collapses into Tier C, not Tier A.

7. **`esrgan/` has zero importers at all** — not even from the post-proc cluster. Truly orphaned.

8. **`blissful_core`'s "Args injection" status is downstream of the generate-script coupling, NOT a direct
   training import.** `src/blissful_tuner/__init__.py` is **empty (0 lines)**, so importing any
   `blissful_tuner.X` does NOT trigger `blissful_core`. The only callers of `add_blissful_args` /
   `parse_blissful_args` are the generate scripts. The project CLAUDE.md describes `blissful_core` as
   "training-coupled (Args injection)" — that is accurate only because training imports generate scripts
   (gotcha #2); no `*_train_network.py` / `hv_train.py` / cache script imports `blissful_core` directly.

## 4. Lockstep-delete requirements (must change in the SAME commit or CI / install breaks)

- **Deleting `config_manager/`** → also delete `tests/test_config_manager/` (14 test files, its only
  consumer) AND remove 4 console-script entry points in `pyproject.toml:109-112`
  (`bt-compile`, `bt-diff`, `bt-prune`, `bt-tui` → `blissful_tuner.config_manager.cli` / `.tui.app`).
  `pip install -e .` resolves entry points at install time, so leaving them breaks reinstall.
- **Deleting vendored dirs** (`codeformer`, `esrgan`, `gfpgan`, `gimmvfi`, `swinir`) → remove their
  matching lines from `[tool.ruff.extend-exclude]` in `pyproject.toml` (lines ~219-223). Cosmetic
  (won't break install), but the exclude entries become dangling.
- **Deleting `common_extensions.py` (Tier C)** → also delete `tests/test_prepare_metadata.py`
  (imports `prepare_metadata` from it).
- **Tier B is training-import-safe but NOT test-suite-safe.** Before deleting any of the 6 standalone
  generate scripts, sweep `tests/` for imports of them (`git grep -l '_generate_image\|_generate_video' tests/`)
  and remove or rewrite those tests in the SAME commit — otherwise collection fails even though no training
  path was touched. Same applies to the `gui/` dir if any GUI test imports it.
- **No test lockstep for the post-proc top-levels or taehv/taesd/latent_preview** — a test sweep found no
  tests importing facefix/upscaler/GIMMVFI/yolo_blur/video_to_png/metaview/taehv/taesd/latent_preview.
- **False-positive test matches (no action needed):** `tests/test_prior_scheduling.py` imports
  `musubi_tuner.modules.prior_scheduling` and `tests/test_qwen_image_training.py` imports
  `musubi_tuner.modules.scheduling_flow_match_discrete` — both are KEPT training modules, NOT the
  deletable `blissful_tuner.scheduling`.

## 5. Dynamic-dispatch safety (Tier B confirmed)

Training files use `importlib.import_module` only for user-supplied optimizer / lr_scheduler / network
module paths (`hv_train_network.py:785,866,2149`, `hv_train.py:440,521`) — never for generate scripts.
No training file references the 6 standalone generate scripts by string or dynamic import. Tier B
(the 6 standalone generate scripts + GUI) is safe to delete with no relocation.
