# Upstream Refactor Landed — Migration Strategy & "Damage" Assessment

**Date:** 2026-05-23
**Status:** ✅ decision made — **MERGE + post-merge cleanup** (hybrid). Do **not** green-field rebuild.
**Revised 2026-05-23 (post-review):** estimate raised **14–19 h → 18–25 h** and several seam/dataset details sharpened after a read-only verification review ("Ampere"). The strategic call is unchanged (merge, don't rebuild); the corrections are confidence + scope, not direction. Every claim below was re-verified against `upstream/main` (`78acafb`) and blissful HEAD before landing here.
**Trigger:** kohya-ss/musubi-tuner's large refactor merged to `upstream/main` (PR #950, `78acafb`; version bumped to **0.3.0**, `55691e8`). The two pre-existing migration plans — `2026-05-06-upstream-refactor-migration-plan.md` (PEFT slice) and `2026-05-21-masked-loss-refactor-integration-plan.md` (masked loss) — had their "wait for dev→main" pre-requisite satisfied. This doc supersedes their *timing* gate and sets the overall strategy they execute under.

---

## TL;DR

- **The true "damage" is bounded.** A virtual `git merge upstream/main` produces **40 conflicts**, but the count is a vanity metric. **32 of 40 are "keep ours"** (22 files where blissful already shipped the feature *ahead* of upstream — including all 12 add/add files — plus 10 trivial docs/config). The genuinely hard work lives in **three places**: the `hv_train_network.py` structural split (its own plan), the `dataset/` refactor (a **five-file** re-home, not two), and the trainer-seam feature port (which is *not* a magic vacuum hose — see below).
- **Recommendation: merge, don't rebuild.** Starting fresh on refactored musubi-tuner would *discard* blissful's lead on 5 architectures, re-derive ~300 guarding tests, and re-debug features that already cost real effort to land. The debt you actually want gone (vendored post-processing) is severable in a **single post-merge `git rm` commit** — you do not need a green-field rebuild to get a clean tree.
- **This is a feature-port, not a model-port.** Upstream v0.3.0 ships all 9 architectures natively. You inherit the scaffolding for free and only re-hang your training deltas onto the new seams.
- **Budget ~18–25 h (still 2–3 focused days).** Don't bank on a 14 h fast path unless you're willing to defer full-finetune (Qwen/Z-Image) parity and the dead-weight cleanup. The dataset split and the trainer-seam gaps are bigger than a first read suggests.

---

## The numbers (measured, not estimated)

| Metric | Value |
|---|---|
| Merge-base | `8ec9577` (kohya-ss v0.2.15, PR #815) |
| Blissful ahead / behind | **1304 ahead / 45 behind** |
| Total fork divergence | 579 files, +160,866 / −2,778 |
| Divergence in `src/musubi_tuner` (shared zone) | 113 files, +19,244 / −2,633 |
| Upstream refactor in `src/musubi_tuner` | 52 files, +11,167 / −4,420 (the −4,420 = gutted monolith) |
| Virtual-merge conflicts (`git merge-tree`) | **40** (28 content, 12 add/add) |
| `hv_train_network.py` LOC | blissful **3865** · merge-base 3078 · upstream **502** (body moved to new `training/trainer_base.py`, 2154 LOC) |

---

## Why NOT start fresh (the strategic question, answered)

The instinct to start fresh comes from fear of merging a 1304-commit monster. The data says the monster is bounded, and a rebuild is strictly worse on every axis:

1. **Architecture parity makes rebuild redundant.** Upstream v0.3.0 has WAN, HunyuanVideo, HV1.5, FramePack, FLUX.1 Kontext, FLUX.2, Qwen-Image, Z-Image, **and** Kandinsky 5 — all native, none stubs. Blissful is *ahead* on WAN/HV/FLUX.2/Qwen/Z-Image and at parity elsewhere. A merge **keeps that lead**; a rebuild **throws it away** and re-ports model families you'd get for free anyway.
2. **The 12 add/add conflicts are blissful being ahead, not behind.** `flux_2/*`, `loha.py`, `lokr.py`, `network_arch.py`, `lora_flux_2.py` — blissful's versions are supersets (Conv2d LoRA support, 13-arch registry, additive exclude semantics, edit-2511 that upstream lacks). "Keep ours" preserves them; a rebuild re-implements them.
3. **~300 tests guard hard-won invariants.** mask-loss math, LoRA target coverage, DoRA delta path, Conv2d LoRA forward, block-swap × no_grad restore, FLUX.2 block backward anomaly, FLUX.2 compile smoke, xformers degradation, RoPE FIFO eviction, compact time embedding, the PiSSA bundle. A rebuild re-derives every one. A merge keeps them green and they become the acceptance gate.
4. **The features that cost real debugging are already debugged here.** compact time embedding (multi-GiB savings), RoPE FIFO eviction, block swap × no_grad restore (`30dfc67`), xformers graceful degradation (`ea3e1e9`), Dynamo cache-entry logging. Rebuild = re-debug.
5. **You can have the clean tree anyway.** The debt is severable post-merge (see cleanup section). Fresh-start's only real promise — shedding dead weight — is delivered by a `git rm` commit, without the fresh-start tax.

> Two corrections to received wisdom, surfaced empirically: **`RexLR` (`lr_schedulers.py`) and fused Adafactor (`adafactor_fused.py`) are byte-identical to upstream** — already upstream, zero port cost. CLAUDE.md lists them as blissful differentiators; the diff says they aren't anymore.

---

## The conflict ledger (40 files, triaged)

Full per-file ledger in `2026-05-23-refactor-migration-A-conflict-ledger.md`. Summary by category:

| Category | Count | Est. effort | Resolution posture |
|---|---:|---:|---|
| **STRUCTURAL** | 1 | (own plan) | `hv_train_network.py` — see 2026-05-21 masked-loss plan |
| **GENUINE-MERGE** | 4 | ~3.6 h | dataset split + 2 Qwen-Layered files (real hand-merge) |
| **UPSTREAM-IMPROVEMENT** | 3 | ~1.1 h | keep ours + adopt a named upstream change |
| **BLISSFUL-AHEAD** | 22 | ~3.4 h | keep ours; optional small cherry-picks |
| **TRIVIAL** | 10 | ~0.6 h | docs/config; keep ours (already supersets) |

### The only genuinely hard non-structural work
1. **`dataset/image_video_dataset.py` (~2–2.5 h)** — upstream split the monolith into **five** new files: `dataset/{architectures,bucket,cache_io,datasources,media_utils}.py`. Blissful extended the monolith in place. Adopting the split is the right move (cheaper future merges), but it is a **five-file re-home, not a two-file move** — the easy mistake is preserving cache helpers while silently dropping datasource behavior. The blissful local features that MUST survive the re-home are broader than "masks + FLUX.2 + Kandinsky": also `caption_directory`, multi-dot caption filtering, duplicate-basename protection (cache-key safety), mask lookup/fallback, cache mask-transform metadata, Qwen layered/`multiple_target` datasource behavior, and the Z-Image/FLUX.2 cache-mask conventions. **This is the highest-risk non-structural area.**
2. **`dataset/config_utils.py` (~45 min)** — upstream renamed `flux_kontext_/qwen_image_edit_*_no_resize_control` → unified `no_resize_control` / `control_resolution` and added `multiple_target`. **Less dangerous than it looks: both blissful HEAD and upstream already carry a deprecated-key→new-key shim** (`config_utils.py` maps the old names), so old TOMLs keep working. Adopt the rename, re-apply blissful's mask/alpha schema. When grepping for stragglers use the **broad** pattern — the narrow `_no_resize_control` misses `qwen_image_edit_control_resolution`:
   ```bash
   git grep -nE 'flux_kontext_no_resize_control|qwen_image_edit_no_resize_control|qwen_image_edit_control_resolution'
   ```
3. **`qwen_image_model.py` + `qwen_image_generate_image.py` (~80 min)** — both sides independently implemented Qwen-Image-Layered over the same regions; needs line-level reconciliation, not blind keep-ours. **Guardrails:** in `qwen_image_model.py`, explicitly protect blissful's padded-text / `txt_seq_lens` / varlen `cu_seqlens` attention region — upstream lacks it and it's easy to simplify toward upstream during a Layered conflict. In `qwen_image_generate_image.py`, the layered one-control guard the ledger flagged is **already present in HEAD** — likely no pending cherry-pick; keep-ours stands (blissful carries true CFG, negative-prompt defaults, CFG-normalize controls).

Everything else is "keep ours, skim upstream for a cheap cherry-pick" (e.g. upstream's `q,k,v=None` frees in `attention.py`; upstream's `rel_error` log line in the Z-Image LoRA converter).

### Trainer seam — what does NOT come for free (verified)
Upstream's `NetworkTrainer.process_batch()` (`trainer_base.py:1108`) / `compute_loss()` (`:1144`) / `on_post_optimizer_step` are the right place to re-hang LoRA masked-loss / prior / EMA — but the seam is **not** a magic vacuum hose. Four boundaries it does **not** cover, each verified on `upstream/main`:

1. **Full-finetune trainers run their own loop.** `QwenImageTrainer.train()` (`qwen_image_train.py:143`) and `ZImageTrainer.train()` (`zimage_train.py:79`) define independent `train()` methods that do **not** call `trainer_base.process_batch()`. Masked loss does **not** inherit automatically here — either port the masked path into those loops or **consciously defer full-FT mask parity** (and say so).
2. **`DiTOutput` shape adaptation is mandatory.** Blissful's `call_dit()` returns a bare tuple `model_pred, target` (`hv_train_network.py:1966`, unpacked at `:2800`); upstream's seam produces `DiTOutput(pred=…, target=…)` (`trainer_base.py:86`). The masked path must adapt explicitly — a tuple-unpack against a dataclass is a silent breakage.
3. **The block-swap restore helper is blissful-only.** `restore_block_swap_after_no_grad_forward(...)` does **not** exist in upstream `trainer_base.py`. It must be ported *before* any masked prior-teacher no-grad forward is safe (else the block-swap × no_grad crash documented in CLAUDE.md returns).
4. **Args / validation / warnings are setup-time, not per-step.** `--use_mask_loss` & friends, mask-source validation, loss-type args, and the "mask sources configured but `--use_mask_loss` off" warning are **not** reachable from `process_batch()`. They need inline ports into `parser_common.py` / `_validate_args_and_init()` / `_build_dataset()` or small new hooks.

---

## Post-merge dead-weight cleanup (the debt you wanted gone)

Full import audit in `2026-05-23-refactor-migration-C-deadweight-audit.md`. Land this as a **separate commit after the merge resolves**, so the merge stays reviewable and the deletion is atomic.

### Tier A — safe to delete now, zero training-path reach (~60,070 LOC)
- Vendored: `src/blissful_tuner/{gimmvfi (37.6K), codeformer (11.8K), gfpgan (4.3K), swinir, esrgan, config_manager (4.3K)}/`
- Post-proc modules: `facefix.py`, `upscaler.py`, `GIMMVFI.py`, `yolo_blur.py`, `video_to_png.py`, `metaview.py`

### Tier B — inference-only, optional (~10,349 LOC)
- The 6 standalone `*_generate_image.py` / generate scripts that training does **not** borrow from (see gotcha #2), plus `src/musubi_tuner/gui/` (2,413 LOC).
- ⚠️ **Tier B is training-import-safe but NOT test-suite-safe.** Deleting standalone generate scripts will break any test that imports them; those tests must be removed or rewritten **in the same commit**. This is why Tier B comes *after* the merge is green, never bundled with it.

### ⚠️ Gotchas that break a naive `git rm`
1. **`guidance.py`, `advanced_rope.py`, `hvw_posemb_layers.py` are CORE WAN model code** — imported by `src/musubi_tuner/wan/modules/model.py:23-24`. They *look* like inference features. **KEEP.**
2. **Training borrows ~5 helper symbols from 3 generate scripts** (`hv_/wan_/fpack_generate_video` → `resize_image_to_bucket`, `save_videos_grid`, `encode_to_latents`, `parse_one_frame_inference_args`, `decode_latent`). Those 3 generate scripts are **not** deletable as-is.
3. **The inference cluster** (`latent_preview`, `prompt_management`, `common_extensions`, `scheduling`, `fp8_optimization`, `taehv`, `taesd`, `video_processing_common`, `blissful_core`) is reachable from training **only via gotcha #2** — no training file imports them directly. They become deletable (a further ~9K LOC) **only after** the ~5 borrowed helpers are relocated into a training-side util module. Defer that to a follow-up; it is not free.
4. **Lockstep deletes:** removing `config_manager/` also requires deleting `tests/test_config_manager/` (14 files) and the 4 console-script entry points in `pyproject.toml:109-112`, else `pip install -e .` breaks. Removing vendored dirs → clean their `[tool.ruff.extend-exclude]` lines.

**Recommended cleanup scope for v1: Tier A only (~60K LOC, zero risk).** Tier B/C after the merge is proven green.

---

## Execution order

```
0. CAPTURE GREEN BASELINE FIRST:  ./venv314/bin/python -m pytest -q
     Record the passing count + ANY pre-existing failures (xzuyn manual tests,
     CI flakes). This exact set — not an abstract "~300" — is the acceptance
     target for step 6. Without it you can't tell a merge regression from a
     failure that was already red.
1. Branch off blissful main:  git checkout -b chore/merge-upstream-v0.3.0
2. git merge upstream/main    (expect the 40 conflicts above)
3a. RESOLVE dataset/config_utils.py FIRST (it ripples). Adopt upstream's unified
     no_resize_control / control_resolution rename (KEEP the deprecated-key shim
     both sides already have), re-apply blissful's mask/alpha schema, then grep
     with the BROAD pattern (the narrow one misses control_resolution):
       git grep -nE 'flux_kontext_no_resize_control|qwen_image_edit_no_resize_control|qwen_image_edit_control_resolution'
     and fix any straggler call sites before touching anything else.
3b. RESOLVE the dataset split as a dedicated FIVE-FILE mini-port:
     architectures.py / bucket.py / cache_io.py / datasources.py / media_utils.py
     + image_video_dataset.py. Re-home blissful's additions (masks, alpha,
     caption_directory, multi-dot caption filtering, duplicate-basename guard,
     mask fallback, cache mask metadata, Qwen layered/multiple_target, Z-Image/
     FLUX.2 cache-mask conventions) onto the split. Do NOT just diff two files.
3c. Resolve the rest by the ledger:
     - keep-ours wholesale for the 32 BLISSFUL-AHEAD + TRIVIAL files, BUT protect
       the LoRA/DoRA/Conv2d invariants and the Qwen padded-text/varlen attention
       region from accidental simplification toward upstream
     - hand-merge the 2 Qwen-Layered files
     - LEAVE hv_train_network.py for step 5
4. Adopt the named upstream improvements (attention q,k,v=None; z-image
   converter rel_error log; any pyproject dep pins).
5. Port the TRAINER SURFACES (not just process_batch — see "Trainer seam" above):
     - masked_process_batch(...) on the LoRA path (process_batch override)
     - DiTOutput adaptation (tuple → .pred/.target)
     - port the block-swap restore helper into the new base
     - EMA update on the on_post_optimizer_step hook
     - parser args + mask validation + disabled-mask warning into
       parser_common.py / _validate_args_and_init() / _build_dataset()
     - full-FT Qwen/Z-Image: port into their own train() loops OR explicitly defer
     - then 2026-05-06 PEFT plan: validate_pissa_training_args near upstream's
       SageAttention validation (pre-load); ss_base_sha256 via extra_metadata()
       for the LoRA path + explicit call sites for the full-FT trainers
6. Run the TARGETED gates below (dataset/Qwen, then LoRA/trainer) BEFORE the full
   suite — they localize fallout. Then run the full suite against the step-0 baseline.
7. Separate commit: Tier-A dead-weight git rm (+ lockstep test/pyproject/ruff edits).
8. Re-run tests again. Done. (Tier B/C only after a deliberate test+helper-relocation pass.)
```

---

## Targeted gates (the first acceptance ladder)

Run these *before* the full suite — they localize fallout to the subsystem you just touched. Capture the **step-0 green baseline** first so a regression is distinguishable from a pre-existing failure.

**After the dataset split + Qwen resolution (step 3):**
```bash
./venv314/bin/python -m pytest -q \
  tests/test_dataset_caption_directory.py \
  tests/test_cache_mask_wiring.py \
  tests/test_cache_mask_preprocessing.py \
  tests/test_mask_cache_metadata.py \
  tests/test_mask_weights_cache_dtype.py \
  tests/test_wan_dataset_loading.py \
  tests/test_qwen_cache_variant_metadata.py \
  tests/test_qwen_image_utils.py \
  tests/test_qwen_image_cfg_normalize_toggle.py \
  tests/test_qwen_image_dual_cfg.py \
  tests/test_qwen_image_generate_image_cache_key.py \
  tests/test_qwen_image_training.py
```

**After the trainer-surface port (step 5):**
```bash
./venv314/bin/python -m pytest -q \
  tests/test_mask_loss.py \
  tests/test_prior_scheduling.py \
  tests/test_lora_ema_teacher.py \
  tests/test_block_swap_prior_restore.py \
  tests/test_lora_dora_delta_path.py \
  tests/test_lora_conv2d_forward.py \
  tests/test_lora_target_coverage.py \
  tests/test_flux2_lora_target_coverage.py \
  tests/test_flux2_block_backward_anomaly.py \
  tests/test_flux2_compile_smoke.py \
  tests/test_flux2_integration_smoke.py
```

Add **one new tiny test** asserting the masked path handles a `DiTOutput`-shaped return (guards gap #2 above), and **one** for the Qwen padded-text / varlen attention region if that region is touched during the Layered reconciliation.

---

## Effort estimate (honest range — revised post-review)

Bottom-up, after the seam/dataset corrections. The earlier 14–19 h was too optimistic about the trainer refactor and under-specified the dataset split; the dataset re-home is now the largest non-structural slice.

| Phase | Estimate |
|---|---:|
| Conflict resolution, non-structural (incl. dataset split as the largest slice) | 6–9.5 h |
| Trainer / seam feature port (masked path, DiTOutput, block-swap restore, EMA hook, parser/validation, full-FT) | 8.5–11 h |
| PEFT / base-hash / PiSSA retargeting | 1–1.5 h |
| Tier-A dead-weight cleanup | 0.5–1 h |
| Targeted + full validation / fallout | 2–3 h |
| **Total** | **~18–25 h** |

Anchor: still **2–3 focused days**, single contiguous session preferred so the merge state doesn't go stale — but **do not bank on the 14 h fast path** unless you're willing to defer full-FT (Qwen/Z-Image) mask parity and the cleanup. The dataset five-file re-home and the four trainer-seam gaps are the judgment-heavy slices; everything else is mechanical or already planned.

---

## Risks / open items
- **Dataset split adoption is a fork-direction choice**, not just a merge: do you re-home onto upstream's **five-file** split (`architectures.py` / `bucket.py` / `cache_io.py` / `datasources.py` / `media_utils.py`) — more upstream-trackable future — or keep blissful's monolithic `image_video_dataset.py` (less merge work now, more drift later)? Recommend adopting the split — future upstream merges get cheaper. But it's a **~2–2.5 h five-file re-home**, not the ~90 min two-file move first estimated; the live hazard is preserving cache helpers while silently dropping datasource behavior.
- **Accepted divergent API on the LoHa/LoKr registry.** Keeping blissful's `loha.py`/`lokr.py`/`network_arch.py` (the `get_arch_config` / `ARCH_CONFIGS` API, Conv2d-capable, 13 archs) over upstream's `detect_arch_config(unet)` (Linear-only) is correct — blissful is the superset. But it means **future upstream changes in this area will not auto-merge**; they'll need manual translation between the two registry APIs. This is a cost you're choosing to pay for the richer implementation; name it so future-you isn't surprised when a network_arch merge conflicts.
- **`config_utils.py` rename is shimmed on both sides** (deprecated old keys still map to the new ones), so it's less dangerous than "rename ripples everywhere" implies. Still grep with the **broad** pattern (`flux_kontext_no_resize_control|qwen_image_edit_no_resize_control|qwen_image_edit_control_resolution`) before testing — the narrow `_no_resize_control` misses `qwen_image_edit_control_resolution`.
- **Full-finetune mask parity is an explicit decision, not a freebie.** `QwenImageTrainer`/`ZImageTrainer` have their own `train()` loops that bypass `process_batch`. Either port the masked path into them or defer full-FT mask support — but decide deliberately and document which.
- **Tier-C cleanup (the ~5 borrowed helpers)** is tempting but is a refactor, not a deletion — keep it out of the merge PR. **Tier B (generate scripts) is training-import-safe but not test-safe** — its tests must be removed/rewritten in the same commit.
- Re-run the LoRA-invariant bundle and the block-swap × no_grad test specifically after step 5 — those are the invariants most exposed by re-hanging logic onto new seams.

## Cross-references
- `2026-05-23-refactor-migration-A-conflict-ledger.md` — full 40-file triage
- `2026-05-23-refactor-migration-B-feature-inventory.md` — must-keep features + architecture parity
- `2026-05-23-refactor-migration-C-deadweight-audit.md` — import-graph dead-weight audit
- `docs/plans/2026-05-06-upstream-refactor-migration-plan.md` — PEFT slice
- `docs/plans/2026-05-21-masked-loss-refactor-integration-plan.md` — masked loss onto seams. **Verified 2026-05-23:** dev→main was a fast-forward for `trainer_base.py` (0-line diff), so that plan's seam line numbers still hold on `upstream/main` — `process_batch` @1108, `compute_loss` @1144, `on_post_optimizer_step` def @1199. The 6.5h structural estimate stands.
