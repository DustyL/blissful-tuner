# Upstream Refactor Landed — Migration Strategy & "Damage" Assessment

**Date:** 2026-05-23
**Status:** ✅ decision made — **MERGE + post-merge cleanup** (hybrid). Do **not** green-field rebuild.
**Trigger:** kohya-ss/musubi-tuner's large refactor merged to `upstream/main` (PR #950, `78acafb`; version bumped to **0.3.0**, `55691e8`). The two pre-existing migration plans — `2026-05-06-upstream-refactor-migration-plan.md` (PEFT slice) and `2026-05-21-masked-loss-refactor-integration-plan.md` (masked loss) — had their "wait for dev→main" pre-requisite satisfied. This doc supersedes their *timing* gate and sets the overall strategy they execute under.

---

## TL;DR

- **The true "damage" is bounded.** A virtual `git merge upstream/main` produces **40 conflicts**, but the count is a vanity metric. **32 of 40 are "keep ours"** (21 files where blissful already shipped the feature *ahead* of upstream — including all 12 add/add files — plus 11 trivial docs/config). The genuinely hard work lives in **two places**: the `hv_train_network.py` structural split (already has its own plan) and the `dataset/` refactor.
- **Recommendation: merge, don't rebuild.** Starting fresh on refactored musubi-tuner would *discard* blissful's lead on 5 architectures, re-derive ~300 guarding tests, and re-debug features that already cost real effort to land. The debt you actually want gone (vendored post-processing) is severable in a **single post-merge `git rm` commit** — you do not need a green-field rebuild to get a clean tree.
- **This is a feature-port, not a model-port.** Upstream v0.3.0 ships all 9 architectures natively. You inherit the scaffolding for free and only re-hang your training deltas onto the new seams.

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
| **BLISSFUL-AHEAD** | 21 | ~3.4 h | keep ours; optional small cherry-picks |
| **TRIVIAL** | 11 | ~0.6 h | docs/config; keep ours (already supersets) |

### The only genuinely hard non-structural work
1. **`dataset/image_video_dataset.py` (~90 min)** — upstream *moved* save/bucket/media functions out into new `dataset/architectures.py` + `dataset/media_utils.py`; blissful extended the monolith in place (mask_directory/alpha_mask, FLUX.2, Kandinsky). Must re-home blissful's additions onto the split layout (or consciously keep the monolith and forgo the split).
2. **`dataset/config_utils.py` (~45 min)** — upstream renamed `flux_kontext_/qwen_image_edit_*_no_resize_control` → unified `no_resize_control` / `control_resolution` and added `multiple_target`. Adopt the rename, re-apply blissful's mask/alpha schema, fix call sites (ripples into cache/generate scripts). **This is the one real "merge intelligence" decision.**
3. **`qwen_image_model.py` + `qwen_image_generate_image.py` (~80 min)** — both sides independently implemented Qwen-Image-Layered over the same regions; needs line-level reconciliation, not blind keep-ours.

Everything else is "keep ours, skim upstream for a cheap cherry-pick" (e.g. upstream's `q,k,v=None` frees in `attention.py`; upstream's `rel_error` log line in the Z-Image LoRA converter).

---

## Post-merge dead-weight cleanup (the debt you wanted gone)

Full import audit in `2026-05-23-refactor-migration-C-deadweight-audit.md`. Land this as a **separate commit after the merge resolves**, so the merge stays reviewable and the deletion is atomic.

### Tier A — safe to delete now, zero training-path reach (~60,070 LOC)
- Vendored: `src/blissful_tuner/{gimmvfi (37.6K), codeformer (11.8K), gfpgan (4.3K), swinir, esrgan, config_manager (4.3K)}/`
- Post-proc modules: `facefix.py`, `upscaler.py`, `GIMMVFI.py`, `yolo_blur.py`, `video_to_png.py`, `metaview.py`

### Tier B — inference-only, optional (~10,349 LOC)
- The 6 standalone `*_generate_image.py` / generate scripts that training does **not** borrow from (see gotcha #2), plus `src/musubi_tuner/gui/` (2,413 LOC).

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
     no_resize_control / control_resolution rename, re-apply blissful's mask/alpha
     schema, then:  git grep -n '_no_resize_control'  and fix EVERY old arg-name
     call site in cache + generate scripts before touching anything else.
     (Do this before any keep-ours pass on qwen_image_cache_latents.py, or you'll
     re-touch that file twice.)
3b. Resolve the rest by the ledger:
     - keep-ours wholesale for the 32 BLISSFUL-AHEAD + TRIVIAL files
     - hand-merge the remaining 3 GENUINE files (dataset/image_video_dataset.py
       + the 2 Qwen-Layered files)
     - LEAVE hv_train_network.py for step 5
4. Adopt the 3 named upstream improvements (attention q,k,v=None; z-image
   converter rel_error log; any pyproject dep pins).
5. Execute the structural plans against training/trainer_base.py seams:
     - 2026-05-21 masked-loss-refactor-integration-plan.md  (masked loss, EMA,
       prior, block-swap restore re-hung on process_batch/compute_loss seams)
     - 2026-05-06 PEFT plan  (validate_pissa_training_args + ss_base_sha256
       call-site re-targeting; folds into the same trainer_base edits)
6. Re-run the full guarding test suite (~300 tests). This is the acceptance gate.
7. Separate commit: Tier-A dead-weight git rm (+ lockstep test/pyproject/ruff edits).
8. Re-run tests again. Done.
```

---

## Effort estimate (honest range)

The bottom-up file-by-file estimate (Agent A) and a top-down estimate diverge on one thing: **how carefully you audit the 32 keep-ours files**. A fast `git checkout --ours` + later-skim pass collapses them to ~2 h; a careful "verify each, cherry-pick upstream wins" pass is ~8.7 h. The irreducible work is the same either way.

| Phase | Fast path | Careful path |
|---|---:|---:|
| Conflict resolution (non-structural) | ~5.5 h | ~8.7 h |
| Structural: masked-loss onto seams (2026-05-21 plan) | 6.5 h | 6.5 h |
| PEFT call-site re-targeting (2026-05-06 plan, folds in) | ~0.5 h | 0.75 h |
| Tier-A dead-weight cleanup | ~0.5 h | ~1 h |
| Full test re-run + fix fallout | ~1.5 h | ~2 h |
| **Total** | **~14 h** | **~19 h** |

Anchor: **2–3 focused days**, single contiguous session preferred so the merge state doesn't go stale. The dataset/config_utils rename is the highest-judgment item; everything else is mechanical or already planned.

---

## Risks / open items
- **Dataset split adoption is a fork-direction choice**, not just a merge: do you re-home onto upstream's `architectures.py`/`media_utils.py` split (more upstream-trackable future) or keep blissful's monolithic `image_video_dataset.py` (less merge work now, more drift later)? Recommend adopting the split — future upstream merges get cheaper, and it's a one-time ~90 min cost.
- **Accepted divergent API on the LoHa/LoKr registry.** Keeping blissful's `loha.py`/`lokr.py`/`network_arch.py` (the `get_arch_config` / `ARCH_CONFIGS` API, Conv2d-capable, 13 archs) over upstream's `detect_arch_config(unet)` (Linear-only) is correct — blissful is the superset. But it means **future upstream changes in this area will not auto-merge**; they'll need manual translation between the two registry APIs. This is a cost you're choosing to pay for the richer implementation; name it so future-you isn't surprised when a network_arch merge conflicts.
- **`config_utils.py` rename ripples** into cache + generate scripts; grep for the old `*_no_resize_control` arg names after resolving and fix all call sites before testing.
- **Tier-C cleanup (the ~5 borrowed helpers)** is tempting but is a refactor, not a deletion — keep it out of the merge PR.
- Re-run the LoRA-invariant bundle and the block-swap × no_grad test specifically after step 5 — those are the invariants most exposed by re-hanging logic onto new seams.

## Cross-references
- `2026-05-23-refactor-migration-A-conflict-ledger.md` — full 40-file triage
- `2026-05-23-refactor-migration-B-feature-inventory.md` — must-keep features + architecture parity
- `2026-05-23-refactor-migration-C-deadweight-audit.md` — import-graph dead-weight audit
- `docs/plans/2026-05-06-upstream-refactor-migration-plan.md` — PEFT slice
- `docs/plans/2026-05-21-masked-loss-refactor-integration-plan.md` — masked loss onto seams. **Verified 2026-05-23:** dev→main was a fast-forward for `trainer_base.py` (0-line diff), so that plan's seam line numbers still hold on `upstream/main` — `process_batch` @1108, `compute_loss` @1144, `on_post_optimizer_step` def @1199. The 6.5h structural estimate stands.
