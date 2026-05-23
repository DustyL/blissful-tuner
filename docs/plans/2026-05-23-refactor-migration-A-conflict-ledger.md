# Merge Conflict Triage Ledger — `blissful HEAD` ← `upstream/main`

Merge-base: `8ec9577` (kohya-ss v0.2.15). Blissful HEAD = 1304 commits ahead; upstream = 45 ahead (monolith→`training/trainer_base.py` split + `dataset/` split). Virtual merge = **40 conflicts** (28 content, 12 add/add).

Categories: **STRUCTURAL** → **GENUINE-MERGE** → **UPSTREAM-IMPROVEMENT** → **BLISSFUL-AHEAD** → **TRIVIAL**.

| File | Category | What each side did (1 line) | Resolution action | Est. min |
|------|----------|------------------------------|-------------------|---------:|
| `src/musubi_tuner/hv_train_network.py` | STRUCTURAL | Ours: monolith +1565 (mask loss, prior, EMA, block-swap restore). Theirs: gutted to ~502-line stub + new `training/trainer_base.py`. | **Do not resolve here** — dedicated split plan. Excluded from this estimate. | n/a |
| `src/musubi_tuner/dataset/image_video_dataset.py` | GENUINE-MERGE | Ours: extended monolith in place. Theirs: split monolith into **FIVE** new files — `dataset/{architectures,bucket,cache_io,datasources,media_utils}.py`. | Five-file re-home, NOT a two-file move. Preserve ALL blissful local features: masks/alpha, `caption_directory`, multi-dot caption filtering, duplicate-basename guard (cache-key safety), mask fallback, cache mask-transform metadata, Qwen layered/`multiple_target` datasource, Z-Image/FLUX.2 cache-mask conventions. Highest-risk non-structural file. | 120-150 |
| `src/musubi_tuner/dataset/config_utils.py` | GENUINE-MERGE | Ours: +220 (mask/alpha schema, new dataset params). Theirs: **renamed** `flux_kontext_/qwen_image_edit_*_no_resize_control` → unified `no_resize_control`/`control_resolution`; added `multiple_target`. | **Both sides already carry a deprecated-key→new-key shim** (old TOMLs keep working) — less dangerous than it reads. Adopt the rename, re-apply blissful's mask schema. Grep with the BROAD pattern (`...no_resize_control\|qwen_image_edit_control_resolution`) — narrow `_no_resize_control` misses `control_resolution`. | 45 |
| `src/musubi_tuner/qwen_image/qwen_image_model.py` | GENUINE-MERGE | Both independently added Layered support (`use_additional_t_cond`, `use_layer3d_rope`, 3D RoPE) to the **same** model regions. | Keep ours; line-review upstream's RoPE/block-swap region. **Guardrail: protect blissful's padded-text / `txt_seq_lens` / varlen `cu_seqlens` attention region** — upstream lacks it; easy to simplify toward upstream during a Layered conflict. | 40 |
| `src/musubi_tuner/qwen_image_generate_image.py` | GENUINE-MERGE | Ours: +1000, wildcards + edit-2509 **and** edit-2511 + layered + CFG-norm. Theirs: +399, edit-2509/layered only. | Keep ours (superset: true CFG, negative-prompt defaults, CFG-normalize controls). The layered one-control guard the ledger flagged is **already present in HEAD** — likely no pending cherry-pick. | 40 |
| `src/musubi_tuner/qwen_image_train_network.py` | UPSTREAM-IMPROVEMENT | Ours: +532 (mask loss/prior wiring). Theirs: +386, likely adopts unified `no_resize_control` schema + layered training path. | Keep ours; reconcile with renamed config schema (depends on `config_utils.py` resolution); port layered training tweaks. | 30 |
| `src/musubi_tuner/qwen_image_cache_latents.py` | UPSTREAM-IMPROVEMENT | Ours: +277 (mask weights into cache). Theirs: +155, renamed control-resolution args + layered caching. | Keep ours; rename old `qwen_image_edit_*` arg refs to unified names. Mechanical once schema decided. | 20 |
| `src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py` | UPSTREAM-IMPROVEMENT | Both added LoKr/LoHa→LoRA QKV SVD conversion; upstream has a richer per-layer rank/error log line blissful lacks. | Keep ours (has LoHa path too); pick up upstream's detailed `rel_error` log line. | 15 |
| `src/musubi_tuner/networks/loha.py` (add/add) | BLISSFUL-AHEAD | Ours: 830-line standalone `LoHaNetwork` with **Conv2d** support, uses `get_arch_config`/`ARCH_CONFIGS`. Theirs: 345-line Linear-only, uses `detect_arch_config(unet)`. | **Keep ours wholesale.** Entangled with `network_arch.py`/`lokr.py` (incompatible registry API) — these 3 flip together. | 10 |
| `src/musubi_tuner/networks/lokr.py` (add/add) | BLISSFUL-AHEAD | Ours: reuses `LoRANetwork` via injection, factor-persistence buffers, `get_arch_config`. Theirs: standalone Linear-only, `detect_arch_config`. | Keep ours (entangled group with loha/network_arch). | 8 |
| `src/musubi_tuner/networks/network_arch.py` (add/add) | BLISSFUL-AHEAD | Ours: `ARCH_CONFIGS` registry + `get_arch_config()` (13 archs, additive excludes). Theirs: single `detect_arch_config(unet)` — different API entirely. | Keep ours (drives loha/lokr/lora_flux_2). | 8 |
| `src/musubi_tuner/networks/lora_flux_2.py` (add/add) | BLISSFUL-AHEAD | Ours: additive exclude-pattern merge, BlissfulLogger note (no `basicConfig`), norms-only default. Theirs: replaces excludes, re-adds `basicConfig`, includes img_mod/txt_mod exclude. | Keep ours (deliberate exclude semantics + logger contract). | 8 |
| `src/musubi_tuner/utils/sai_model_spec.py` | BLISSFUL-AHEAD | Both added FLUX.2 + Qwen-Image-Layered arch/impl spec strings — **near-identical** values; blissful also has base `ARCHITECTURE_FLUX_2`. | Keep ours (already a superset of upstream's strings); diff to confirm no missing arch. | 10 |
| `src/musubi_tuner/networks/lora.py` | BLISSFUL-AHEAD | Ours: +1275 (DoRA delta path, Conv2d, split-dims, rank-dropout fix, block-swap invariants). Theirs: +149 (minor). | Keep ours; skim upstream's 149 lines for any non-overlapping fix. Guarded by LoRA invariant tests. | 15 |
| `src/musubi_tuner/utils/lora_utils.py` | BLISSFUL-AHEAD | Ours: +968 (extensive LoRA load/merge/convert). Theirs: +188. | Keep ours; skim upstream delta for tiny fixes. | 12 |
| `src/musubi_tuner/modules/attention.py` | BLISSFUL-AHEAD | Ours: +336 CuTe/FA4 dispatch + xformers OSError graceful-degrade. Theirs: +76, adds `q,k,v=None` frees for memory. | Keep ours; cherry-pick upstream's `q,k,v=None` dereferences (cheap VRAM win). | 12 |
| `src/musubi_tuner/zimage/zimage_utils.py` | BLISSFUL-AHEAD | Ours: +398 (siglip2, local-first tokenizer, dynamic shift, many loaders). Theirs: +84 minor. | Keep ours; skim upstream delta. | 10 |
| `src/musubi_tuner/zimage_train.py` | BLISSFUL-AHEAD | Ours: +185 full-finetune path. Theirs: +46. | Keep ours; skim. | 8 |
| `src/musubi_tuner/zimage_train_network.py` | BLISSFUL-AHEAD | Ours: +101. Theirs: +71. | Keep ours; skim upstream for any zimage fix. | 10 |
| `src/musubi_tuner/qwen_image_train.py` | BLISSFUL-AHEAD | Ours: +275 full-finetune. Theirs: +32 minor. | Keep ours; skim. | 8 |
| `src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py` | BLISSFUL-AHEAD | Ours: +122. Theirs: +26 minor. | Keep ours; skim. | 8 |
| `src/musubi_tuner/qwen_image/qwen_image_autoencoder_kl.py` | BLISSFUL-AHEAD | Both added **identical** `input_channels`/`output_channels` parameterization (layered VAE). | Keep ours; near-identical so trivially mechanical. | 8 |
| `src/musubi_tuner/cache_latents.py` | BLISSFUL-AHEAD | Ours: +315 (mask/alpha caching, arch additions). Theirs: +110. | Keep ours; reconcile with dataset-split imports if monolith dropped. | 12 |
| `src/musubi_tuner/convert_lora.py` | BLISSFUL-AHEAD | Ours: +347 (extra format conversions). Theirs: +109. | Keep ours; skim. | 10 |
| `src/musubi_tuner/flux_2/flux2_models.py` (add/add) | BLISSFUL-AHEAD | Ours: 1138 lines (FP8 exclude `_modulation`, full ModelSpec/configs). Theirs: 1020, exclude key `mod`, trimmed config region. | Keep ours; note FP8 exclude-key delta (`_modulation` vs `mod`) is intentional blissful naming. | 12 |
| `src/musubi_tuner/flux_2/flux2_utils.py` (add/add) | BLISSFUL-AHEAD | Ours: 1086 lines. Theirs: 816. Blissful superset of utils. | Keep ours; spot-check for any upstream-only helper. | 12 |
| `src/musubi_tuner/flux_2_generate_image.py` (add/add) | BLISSFUL-AHEAD | Ours: 1497 lines with wildcards + blissful_tuner imports. Theirs: 1214, none of those markers. | Keep ours wholesale. | 10 |
| `src/musubi_tuner/flux_2_train_network.py` (add/add) | BLISSFUL-AHEAD | Ours: 544 (mask/prior wiring). Theirs: 365. | Keep ours; skim. | 10 |
| `src/musubi_tuner/flux_2_cache_latents.py` (add/add) | BLISSFUL-AHEAD | Ours: 177 (mask/alpha cache). Theirs: 126. | Keep ours; skim. | 8 |
| `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py` (add/add) | BLISSFUL-AHEAD | Ours: 131. Theirs: 91. | Keep ours; skim. | 6 |
| `pyproject.toml` | TRIVIAL | Ours: +239 (extras, ruff config, deps). Theirs: +39 (FLUX.2/Kandinsky deps, version bump). | Keep ours; verify upstream's new dep pins are present (likely already are). | 5 |
| `docs/loha_lokr.md` (add/add) | TRIVIAL | Ours: 168-line guide for Conv2d-capable impl. Theirs: 341-line bilingual EN/JA guide for Linear-only impl. | Keep ours — it documents blissful's actual (different) implementation; upstream's describes Linear-only code we don't ship. | 4 |
| `docs/qwen_image.md` | TRIVIAL | Ours: +420. Theirs: +256 (layered docs). | Keep ours; optionally fold upstream's layered section. | 4 |
| `docs/zimage.md` | TRIVIAL | Ours: +197. Theirs: +236 (longer upstream doc). | Keep ours (matches blissful flags); skim upstream for any new section worth adopting. | 4 |
| `docs/dataset_config.md` | TRIVIAL | Ours: +360 (mask/alpha docs). Theirs: +319 (renamed control args, multiple_target). | Keep ours; add upstream's renamed-arg notes after schema decision. | 5 |
| `docs/flux_2.md` (add/add) | TRIVIAL | Ours: 306 lines. Theirs: 284 lines. Both flux_2 guides. | Keep ours (superset). | 3 |
| `README.md` | TRIVIAL | Ours: +107 (blissful branding/features). Theirs: +100 (new arch announcements). | Keep ours; fold any new upstream arch mentions. | 4 |
| `README.ja.md` | TRIVIAL | Same shape as README.md, JA. | Keep ours; fold. | 4 |
| `.ai/context/overview.md` | TRIVIAL | Ours: 1-line arch-list edit. Theirs: full rewrite to AI-agent-focused overview. | Internal AI-context doc only. Take upstream's rewrite OR keep ours — no code impact. | 3 |
| `.gitignore` | TRIVIAL | Ours: +48 (blissful-specific ignores; already has `references/`). Theirs: +1 (`references/`, no trailing newline). | Keep ours — already a superset. | 1 |

## Totals by category

| Category | Count | Est. min |
|----------|------:|---------:|
| STRUCTURAL | 1 | (excluded — own plan) |
| GENUINE-MERGE | 4 | ~260 (dataset re-home dominates) |
| UPSTREAM-IMPROVEMENT | 3 | 65 |
| BLISSFUL-AHEAD | 22 | 205 |
| TRIVIAL | 10 | 37 |
| **Total (excl. structural)** | **39** | **~567 min (~9.5 h)** |

(Counts re-tallied from the data rows 2026-05-23: 22 BLISSFUL-AHEAD + 10 TRIVIAL = 32 keep-ours, not the 21/11 first written. All 12 add/add files land in BLISSFUL-AHEAD or TRIVIAL — confirming the prime-suspect hypothesis: blissful authored these features ahead of upstream. GENUINE-MERGE bumped after the dataset row was corrected to a five-file re-home.)

## Bottom line

The conflict surface is **mostly bounded and mechanical**: 32 of 40 files are keep-ours (22 BLISSFUL-AHEAD where blissful already shipped the feature — all 12 add/add files included — plus 10 trivial docs/config), each resolvable in minutes with at most a small upstream cherry-pick. The genuinely hard work is concentrated in three places: the **`hv_train_network.py` structural split** (its own plan, excluded here); the **`dataset/` refactor** — corrected to a **five-file** re-home (`architectures/bucket/cache_io/datasources/media_utils` + `image_video_dataset`, ~2–2.5 h), where the live hazard is preserving cache helpers while silently dropping datasource behavior; and the **trainer-seam feature port**, which is not covered by `process_batch` alone (full-FT trainers run their own `train()` loop, the block-swap restore helper is blissful-only, and `call_dit`'s tuple return must be adapted to upstream's `DiTOutput` dataclass — see the strategy doc's "Trainer seam" section). Net: ~9.5 h of conflict resolution outside the structural file, with no architecture-feature surprises — blissful is ahead everywhere (it even carries edit-2511 that upstream lacks); the real merge intelligence is the config-schema rename (shimmed on both sides) and the five-file dataset re-home.
