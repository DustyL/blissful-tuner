# Ideogram-4 Integration: Review of the sdbds Fork + Blissful Plan

**Date:** 2026-06-07
**Reviewed fork:** `/home/dustin/musubi-tuner-forks/musubi-tuner-sdbds` @ branch `qinglong`
**Pure Ideogram commit range (de-contaminated):** `78acafb..a1a3717` (8 commits, +3,939 lines, 21 files)
**Reference impl:** `/home/dustin/Ideogram-4/Ideogram4-Github-Repo` (`src/ideogram4/`)
**Weights on disk:** HF per-row fp8 (`~/Ideogram-4/Huggingface-Model-Weights`), Comfy per-tensor fp8 (`~/Training_Models_Ideogram`)
**Method:** 6 risk-structured review agents + a two-lens adversarial refutation pass (36 agents total), plus
direct primary-source verification of the highest-stakes findings (safetensors headers, `lora.py`,
`fp8_optimization_utils.py`).

---

## TL;DR — Verdict

**Yes, we can fold Ideogram-4 into blissful-tuner — but NOT as a straight port.** The fork's *model
definition* is a faithful, near-byte-identical copy of the canonical transformer, and the VAE is the
**same `AutoencoderKLFlux2` blissful already ships in `flux_2/`**. Those are the parts worth borrowing.

The fork's *weight-loading and training harness*, however, is broken in **5 independently-confirmed
blocker ways** — including **it cannot load the published Comfy-Org fp8 weights its own docs tell you to
download** — and it **bypasses blissful's central differentiators** (mask-weighted loss, prior
preservation, the fp8 infrastructure, DoRA). The commit history is spec/review-driven with no evidence of
an end-to-end run against the real weights.

**Strategy:** treat the fork as a *reference for the transformer math only*; rebuild the harness on
blissful's existing seams. The single most important design decision — **load fp8 through blissful's
existing `nn.Linear`+`scale_weight` patch instead of the fork's bespoke `Fp8Linear` class** — neutralizes
four of the fork's problems at once.

---

## 1. Architecture ground truth (verified from `model_index.json` + `transformer/config.json`)

| Component | Value |
|---|---|
| Transformer | `Ideogram4Transformer2DModel`: 34 layers, 18 heads, head_dim 256 (emb 4608), `in_channels=128`, `intermediate_size=12288`, `adaln_dim=512`, `llm_features_dim=53248` (= 4096×13), `mrope_section=[24,20,20]`, `rope_theta=5e6`, `norm_eps=1e-5`. ~9.3B params. |
| Text encoder | `Qwen3VLModel` + `Qwen2Tokenizer` |
| VAE | **`AutoencoderKLFlux2`** — same family blissful ships in `flux_2/` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` (logit-normal sampling) |
| CFG | Asymmetric, via a **separate `unconditional_transformer`** (inference/sampling only) |
| Releases | **Only `nf4` and `fp8`** — there is **no full-precision/bf16 release.** LoRA must train over fp8 (or nf4). We target **fp8** over nf4: fp8 dequantizes to a plain `weight×scale` (trivially compatible with blissful's existing `nn.Linear` fp8 patch and LoRA wrapping), whereas nf4 (bitsandbytes 4-bit) needs the `Linear4bit`/double-quant path and a heavier dequant — more divergence for a coarser base. |

**Two distinct fp8 layouts exist, and you have both:**
- **HF `ideogram-4-fp8`** (`~/Ideogram-4/Huggingface-Model-Weights`): **per-row** `weight_scale` (shapes
  `(512,)`,`(4608,)`,`(128,)`,`(12288,)`… = each layer's `out_features`), **no `comfy_quant`**, sharded
  diffusers layout. *This is the layout the fork's `Fp8Linear` was written for.*
- **Comfy-Org** (`~/Training_Models_Ideogram`): **per-tensor scalar** `weight_scale` (shape `()`) + 211
  `comfy_quant` U8`[27]` metadata tensors, single-file. *This is the layout the fork's docs mandate — and
  the fork cannot load it.*

> **Recommended training base: the HF per-row weights.** Measured directly (dequantize the same layer from
> both files): the two layouts agree in direction (cosine **0.988–0.996** across sampled layers — i.e. the
> same model, and `comfy_quant` is droppable metadata) but the per-tensor Comfy dequant deviates **9.5–15.8%
> in L2** from per-row — the expected coarseness of one scale-per-matrix vs one-per-row. blissful can load
> *either* (§5), so there's no infra reason to accept the coarser base; the per-row HF weights are the more
> faithful foundation for a LoRA. Use Comfy only if the diffusers-sharded HF layout proves inconvenient.

---

## 2. Feasibility — the make-or-break crux is FAVORABLE

**Can blissful's LoRA train an adapter over a frozen fp8 Ideogram base? Yes.** Verified from primary
source: blissful's training-time `LoRAModule.forward` (`networks/lora.py:559-601`) is the clean
`org_forwarded = self.org_forward(x)` … `return org_forwarded + lx*multiplier*scale` path. It rides on the
base layer's own `forward` and **never reads the raw fp8 `.weight`** — the `addmm_`/`weight.t()` fast-path
hazard noted in `CLAUDE.md` is **not present in this file**. The fp8 weight + scale are frozen buffers; only
the bf16 adapter trains. So the gradient/bypass question — the one that actually decides feasibility — is
clean.

The fork's *mechanism* for reaching that path is the weak point (see §6 and §7).

---

## 3. BLOCKERS (5 — all confirmed by adversarial verification; several reproduced empirically)

> Severity legend: **BLOCKER** = the fork cannot run this path at all today.

### B1 — fp8 loader cannot load the published Comfy-Org weights (scale-shape crash)
`ideogram4/ideogram4_quantized_loading.py:197-222,257-295`
The published `ideogram4_fp8_scaled.safetensors` stores `weight_scale` as a **scalar `[]`** (per-tensor) for
all 211 Linears (verified in the safetensors header). The fork's `Fp8Linear` registers `weight_scale =
torch.empty(out_features)` (per-row) and does `weight_scale.unsqueeze(1)`. `load_state_dict(assign=True)`
does **not** skip the shape check → `size mismatch for weight_scale: copying shape [] into [N]`. Hard crash
at load. Same failure hits the Qwen3-VL fp8 text encoder (252 scalar scales).

### B2 — fp8 loader raises on the published `.comfy_quant` metadata tensors
`ideogram4/ideogram4_quantized_loading.py:282-295`
Each published quantized Linear carries a sibling `<module>.comfy_quant` U8`[27]` tensor (211 in each DiT,
252 in the TE — all verified on disk). `load_fp8_state_dict` raises on **any** unexpected key, so even with
B1 fixed, loading the real weights crashes on `unexpected keys: ['…comfy_quant', …]`. Clear evidence the
fork was written against a hypothetical layout, never the actual files.

### B3 — Training `process_batch` double-patchifies the cached latent (128→512 channel crash)
`ideogram4_train_network.py:234`; `ideogram4_autoencoder.py:337-346`
The latent cache stores `autoencoder.encode()` output, which is **already** the 128-ch patchified+BN-normed
latent (encoder mean 32-ch → `rearrange '… c (i pi) (j pj) -> … (c pi pj) i j'`, ps=[2,2] → 128). But
`process_batch` calls `patchify_vae_latents()` **again** → 512-ch, then applies 128-ch shift/scale →
`size of tensor a (512) must match tensor b (128)`. Training dies on step 1. The synthetic test hides it by
feeding a hand-built raw 32-ch latent — i.e. the test exercises the *intended* contract the cache violates.

### B4 — Generate/decode has the identical crash class
`ideogram4_utils.py:375-382`; `ideogram4_autoencoder.py:348-356`
Symmetric to B3: decode unpatchifies to 32-ch then `inv_normalize` broadcasts 128-ch BN stats → crash;
also double-unpatchifies. B3+B4 together establish, regardless of weight source, that **the train and
generate paths as written cannot complete a single step** — a strong sign the harness was never exercised
end-to-end.

### B5 — Denoising loop presents a MIRRORED timestep schedule to the frozen base
`ideogram4_utils.py:385-390,440-465` vs canonical `pipeline_ideogram4.py:587-615`
Schedulers are byte-identical (`t_=1-t`), but the fork iterates `t` **mirror-image** to canonical: canonical's
first model call is at `t≈0.0001` stepping up; the fork's first call is at `t≈0.999` stepping down (verified
by exact torch reproduction for `V4_DEFAULT_20`). The model has no internal t-flip, so the published frozen
weights are conditioned on the *opposite* t than the fork presents. **Training shares the same convention**
(`process_batch` t-mix + `noise-clean` target), so fork-train and fork-infer are self-consistent with each
other but **both disagree with the published base** — loss will still descend, masking the defect from any
smoke test that doesn't compare against canonical. A LoRA trained this way adapts a mis-conditioned base and
throws away most of the pretrained prior.

**Implication:** every blocker is in the *harness*, not the model. None are reasons to abandon the
integration; all are reasons **not to port the harness verbatim.**

---

## 4. HIGH-severity concerns (confirmed)

| # | Concern | Location | Why it matters |
|---|---|---|---|
| H1 | **Double normalization** vs canonical: fork applies BatchNorm (in `encode`) **and** `latent_norm`; canonical applies `latent_norm` **only** (its `AutoEncoder` has no encode/decode methods). Real-weight check: BN stats vs LATENT_SHIFT/SCALE differ (max\|Δ\|=0.43/0.29) — two distinct norms in series. | `ideogram4_autoencoder.py:327-356` vs canonical `pipeline_ideogram4.py:624,632` | **Silent** mis-scaling of training targets even after B3/B4 are fixed → garbage with no crash. |
| H2 | **Trainer bypasses blissful's mask-loss/prior seam** — hardcoded `F.mse_loss(reduction="mean")`. | `ideogram4_train_network.py:217-254` | `--use_mask_loss`, `--prior_preservation_weight`, EMA teacher, prior scheduling, SD3 weighting all silently dead. `validate_mask_loss_args` still passes → worst failure mode (silent no-op). |
| H3 | **Mask loss unsupported at cache AND routing level**, and Ideogram's `call_dit` returns **token-grid** space, so masks must be patchified to token resolution (more than a flux_2 copy). | `ideogram4_cache_latents.py:79`, `ideogram4_train_network.py:299-302` | blissful's headline differentiator absent for the new arch. |
| H4 | **Caption verifier hard-fails on plain-text captions** at the training data path (`verify_raw` → `raise ValueError` unless `--warn_on_caption_issues`). | `ideogram4_utils.py:470-478`, `ideogram4_cache_text_encoder_outputs.py:30` | A normal `.txt` caption (`"a photo of a cat"`) **aborts text caching for the whole dataset** by default. Bricks the standard LoRA flow. |
| H5 | **fp8 text-cache key bug**: writes `varlen_i4_llm_features_float8_e4m3fn`; the reader's single `rsplit('_',1)` only strips `…_e4m3fn`, leaving `i4_llm_features_float8` → `KeyError` at step 1. | `cache_io.py:519` write vs `bucket.py:255-263` read | The one knob that makes the 53,248-dim cache affordable (fp8) is itself broken. bf16/fp32 are safe. |
| H6 | **Tokenizer/TE config phone home** with `trust_remote_code=True` to a hardcoded HF repo even when all weights are local. | `ideogram4_utils.py:42-44,126-159` | Breaks offline/air-gapped repro; arbitrary-code + version-drift surface. Wrong default for this offline-first, source-built box. |
| H7 | **cache_io splice hazard**: the fork refactors `save_text_encoder_output_cache_common` with an **unguarded `torch.isnan`** and drops the contiguous-first step. blissful deliberately guards isnan behind `dtype.is_floating_point` (int text caches exist: Flux2 `ctx_seq_len_int32`, FramePack `llama_attention_mask`, Kandinsky `attention_mask`). | blissful `cache_io.py:506-545` vs fork hunk | Verbatim patch-apply compiles, passes an Ideogram smoke, and **regresses other architectures** on some torch builds. The canonical splice class `CLAUDE.md` warns about. |

---

## 5. The centerpiece optimization — delete `Fp8Linear`, reuse blissful's fp8 path

The fork's single worst design decision is making the fp8 layer a **distinct `nn.Module` (`Fp8Linear`)**.
That one choice is the root cause of four separate problems:

1. **B1/B2** (can't load published weights — per-row-only, no `comfy_quant` handling);
2. **Discovery break** — blissful's LoRA discovery hardcodes `__class__.__name__ == "Linear"`
   (`lora.py:1113`), so `Fp8Linear` is invisible → empty network → the fork needs an 8-line `lora.py`
   change threaded through 4 sites (a splice surface);
3. **DoRA silently disabled** — `is_linear` is False for `Fp8Linear` (`lora.py:467,472`), so
   `--network_args use_dora=True` no-ops on every Ideogram target;
4. **Per-row rigidity** — can't represent the Comfy per-tensor format.

**blissful already solved fp8 the right way, and the reusable half is the half we need.** Two parts exist:
`optimize_state_dict_with_fp8` (the *quantizer*, bf16→fp8) and `apply_fp8_monkey_patch(model, state_dict, …)`
(the *patcher* — registers `scale_weight` buffers from a loaded sd and patches `nn.Linear.forward` via
`fp8_linear_forward_patch`, `:350`). Ideogram has no bf16 to quantize, so we **skip the quantizer and reuse
the patcher** — which is already the standard pre-quantized-load entry point for 10+ arches
(flux_2/zimage/qwen/hunyuan/kandinsky, e.g. `flux2_utils.py:730`). The forward patch **already dispatches on
scale shape** (comment at `:157`/`:334`: *"keep scale shape `[1]` or `[out,1]` or `[out, num_blocks, 1]` …
determine the quantization mode from the shape of `scale_weight`"*), so per-tensor **and** per-row are
handled. **Verified empirically** (see §1 measurement): both layouts dequantize to cosine-0.99 of each other,
so blissful's plain `weight×scale` is the correct dequant for both and `comfy_quant` is droppable.

**Recommended design:** keep targets as `nn.Linear`, attach the fp8 `weight`+`scale_weight` as buffers, and
reuse `apply_fp8_monkey_patch`/`fp8_linear_forward_patch`. The **only new code is a thin layout shim**:
- `weight_scale` → `scale_weight` (key rename);
- reshape `()` → `[1]` (Comfy per-tensor) or `(out,)` → `[out,1]` (HF per-row) to the ndim the patch reads;
- **cast `scale_weight` → compute dtype (bf16)** — REQUIRED. The dequant branch
  (`fp8_optimization_utils.py:409-412`) sets the dequantized weight to `scale_weight.dtype`; the published
  scales are **float32**, so an uncast load yields an fp32 weight and `F.linear(bf16_x, fp32_weight)` crashes
  on the first forward. blissful's own arches dodge this only because they *generate* bf16 scales — we are the
  first to *load* fp32 ones. (Verified: reproduced with a patched Linear.) Cast at shim time so the registered
  buffer is bf16; leave the fp8 `weight` as float8.
- **drop `comfy_quant`** siblings before load (verified droppable);
- start with **`use_scaled_mm=False`** — `_scaled_mm` adds its own bias/shape/dtype constraints and is not
  where to spend Phase-1 risk.

This single decision:
- ✅ Loads **both** published fp8 layouts — **neutralizes B1/B2** (the shim handles scalar/vector scale and strips `comfy_quant`);
- ✅ Needs **no** `lora.py` change — targets stay class `"Linear"`, so existing discovery finds them;
- ✅ Keeps **DoRA working** (`is_linear` True);
- ✅ Reuses blissful's battle-tested `_scaled_mm`/dequant path and its 4 LoRA invariants unchanged.

> **Phase-1 validation (cheap, do it first):** confirm `apply_fp8_monkey_patch` + `load_state_dict` round-trips
> the remapped sd with **zero missing/unexpected keys** and a numerically sane forward. This is the one spot
> where the reuse claim should be exercised in code before building on it; the shim is small but load-bearing.

LoRA composition is correct as long as the fp8 patch is applied **before** `network.apply_to()`, so LoRA
captures the fp8-patched `forward` as its `org_forward` — exactly how blissful's existing wan/flux2 fp8 +
LoRA training already works.

> If we ever *do* want the fork's `Fp8Linear` route instead, B1 needs a scalar-or-vector scale and B2 needs
> a `comfy_quant` filter — but there is no reason to take on that route given the above.

---

## 6. Other optimizations & fixes to bake into our plan

- **Reuse the flux_2 VAE — but the convenience methods are themselves the trap.** The fork's 449-line
  `ideogram4_autoencoder.py` is ~95% duplication of blissful's `flux_2` `AutoencoderKLFlux2`. Two precise
  decisions:
  - **Source = Comfy native `flux2-vae.safetensors`** (the 336 MB file): it has flat musubi-style keys and
    loads `strict=True` through `flux2_utils.load_ae` (`:754`). The **HF VAE does NOT** — it carries 74
    `encoder.down_blocks.*` diffusers keys (verified) and needs `convert_diffusers_state_dict` (canonical
    `pipeline_ideogram4.py:180`). Default to the Comfy VAE; add an HF-diffusers conversion branch only if
    needed. (Note the asymmetry vs the DiT: train the **DiT** from HF per-row fp8, load the **VAE** from the
    Comfy file — different files, no conflict.)
  - **Use the raw `ae.encoder`/`ae.decoder`, NEVER `ae.encode()`/`ae.decode()`.** blissful's flux2
    `AutoEncoder.encode()` (`flux2_models.py:377-388`) bakes in **BN `normalize()` + patchify** — structurally
    identical to the fork's broken encode. Calling it would reintroduce the exact B3/H1 double-norm bug.
    Ideogram's contract (canonical `pipeline_ideogram4.py:624`): raw encoder mean (32-ch) → patchify **once**
    (→128-ch) → `latent_norm` (shift/scale); decode = denorm → unpatchify → raw `.decoder()`. **BN is never
    invoked.** The Phase-2 round-trip test must assert both the 32→128 boundary and that BN is untouched.
- **Route loss through `masked_process_batch` — and make `--use_mask_loss` binary: fully implemented OR
  loudly rejected, never a half-wired no-op.** Re-author `process_batch` to the zimage/flux_2 dispatch shape
  (`super()` when `use_mask_loss` off, else `masked_process_batch`), and factor the bespoke logit-normal
  timestep+noise prep into a `get_noisy_model_input_and_timesteps` override so the masked path can reuse
  `call_dit`. **Invariant (per second review):** because Ideogram's `call_dit` returns *token-grid* space,
  the mask must be patchified to token-grid resolution before `apply_masked_loss_with_prior`. Until that is
  implemented and tested, `--use_mask_loss` must **fail at argument/setup validation**, not silently fall
  through to mean-MSE. A seam that *looks* wired but no-ops is worse than an explicit "unsupported" because
  users would trust it. (HV/Kontext/FramePack are documented-unsupported and don't accept the flag as a
  no-op — match that.)
- **Expose blissful's flow-matching knobs.** The fork hardcodes logit-normal `mu`/`std`; thread
  `--timestep_sampling` / `--discrete_flow_shift` (or at minimum default-match the reference schedule and
  document the override).
- **Offline-first loading.** Add `--tokenizer` / `--text_encoder_config` local-path args; fall back to HF
  only if unset; document `trust_remote_code`. Primary path should not phone home (fixes H6).
- **Caption verifier off-by-default.** Warn-only for plain captions; only enforce when a caption parses as
  JSON. Document that training on plain captions is a domain shift from the structured-prompt pretraining
  (fixes H4).
- **SDPA backend guard on Blackwell.** Ideogram attention is bare SDPA with no backend guard — same
  unguarded-SDPA class as the cuDNN-SDPA NaN note in project memory. Add the fork-style import-time SDP
  guard / wire blissful's attention backend + `rope_func`/`--compile` for the 5090 path.
- **Fix the fp8 text-cache key** (H5) before recommending fp8 text cache: write a dtype-free content stem
  (dtype already lives in metadata), or special-case `i4_llm_features` in `bucket.py`.

---

## 7. Shared-file graft map (re-implement by INTENT, never patch-apply)

| File | Fork change | Graft instruction |
|---|---|---|
| `dataset/architectures.py` | `+2` — `ARCHITECTURE_IDEOGRAM4 = "i4"` / `_FULL = "ideogram4"` | **Additive append** — clean. |
| `dataset/cache_io.py` | refactor + new `save_*_ideogram4` | Keep blissful's **guarded** `save_text_encoder_output_cache_common` (contiguity loop + dtype-guarded NaN check). Add `extra_metadata: dict` param; place `metadata.update(extra_metadata)` **after** `metadata.update(existing_metadata)` (`:540`) and before `mem_eff_save_file` (`:545`). Add `save_latent_cache_ideogram4` as a new additive fn. Empty `extra_metadata` ⇒ other arches byte-identical. |
| `networks/lora.py` | `+8` — `linear_module_class_names` | **Not needed** under the §5 design (targets stay `nn.Linear`). If kept anyway: thread as an **explicit** param at all 4 sites (never via `**kwargs` — silent-drop), and port the zero-modules `RuntimeError` guard. |

---

## 8. Proposed phased plan

**Phase 0 — Reconcile the t-schedule (gate).** The discrepancy is already *proven analytically* (canonical
`V4_DEFAULT_20` starts at t≈0.000123, the fork at t≈0.999447 — reproduced independently in both reviews), so
the design rule is simply **match canonical's `(t, s, gw)` per-step sequence** and ship a parity test.
**Sequencing caveat (second review):** the fork's own generate is *dead* (B1/B2/B4), so it **cannot be the
empirical oracle** — the "coherent image vs garbage" confirmation requires our own working loader + raw VAE
decode, i.e. it lands as a **Phase-1 exit check**, not a standalone Phase-0 step. Treat Phase 0 as "lock the
convention + parity test"; treat "base model decodes a coherent image from a structured prompt" as the
Phase-1 gate that simultaneously validates fp8 load, raw-VAE decode, and the t-direction.

**Phase 1 — Loading.** fp8 load via the §5 `nn.Linear`+`scale_weight` shim (default to the **HF per-row**
base per §1; support Comfy per-tensor too); flux_2 VAE reuse; offline tokenizer/TE path. Exit: model + VAE +
TE load from local files with **no network**, and `apply_fp8_monkey_patch`+`load_state_dict` reports **zero
missing/unexpected keys** with a numerically sane forward (the §5 validation gate).

**Phase 1.5 — Guardrail pass (LANDED 2026-06-07).** Hardening so the later "finite image" pipeline can't
mask a wrong latent space: raw-helper `latent_norm` docstring warnings; **fp8 patched-count guard** in the
loader (raises if any fp8 Linear lacks a `scale_weight`); **non-unit fp8 scale test** (proves the scale is
applied, not just that forward runs); **patchify-convention pin test** (locks `(pi,pj,c)` to canonical
`pipeline_ideogram4.py:626-629`); `_is_fp8_dtype` → direct dtype comparison; load-bearing SDPA-guard import
comment. **Bonus fix (durable):** RoPE `inv_freq` was silently bf16 (the model-wide `.to(bf16)` downcast it;
the forward upcasts but can't recover precision). `Ideogram4MRoPE._apply()` now **self-heals** — any later
cast/device move rebuilds `inv_freq` float32 on the post-move device (skips pure device moves + meta), so the
invariant is enforced by the module, not remembered by the loader (third-review refinement). The fp8
patched-count guard is factored into a unit-tested `_validate_fp8_linears_patched(model)`; the non-unit fp8
scale test uses fixed literals (deterministic). Real-weight gate re-confirmed (211/211 patched; `inv_freq`
float32 after load AND after a later `.to(bf16)`). Suite: 11 tests.

**Phase 2 — Caching + the `latent_norm` semantic contract.** `latent_norm` is a **front-of-Phase-2 contract**,
not an afterthought inside generation, because it changes what a "cache-ready" token means. The invariant:

```
ENCODE: pixels[0,1] → raw AE encoder mean → patchify→128ch grid → latent_norm   → DiT/cache-ready tokens
DECODE: DiT/cache tokens                  → latent_denorm        → unpatchify→32ch → raw AE decoder → pixels[0,1]
```

The raw helpers (`encode_pixels_to_raw_vae_tokens` / `decode_raw_vae_tokens_to_pixels`) return/consume
**pre-`latent_norm`** tokens (docstring-warned in 1.5). The guard: a cache writer must **refuse** raw-token
output unless normalization has been applied — enforce via (a) distinct names (`…_raw_vae_tokens` vs
`encode_pixels_to_dit_tokens`), **and** (b) cache metadata `latent_norm_applied=true`, **and** (c) a
cache-writer assertion.

> **`latent_norm` module — LANDED 2026-06-07** (`ideogram4/latent_norm.py`): verified against canonical —
> **128-dim, post-patchify**, in `(B, L, 128)` channel-last space; `norm = (t − shift) / scale` (inverse of
> canonical decode `t*scale + shift`, `pipeline_ideogram4.py:624`). Constants vendored programmatically (no
> hand-transcription); `encode_pixels_to_dit_tokens` / `decode_dit_tokens_to_pixels` sanctioned pair +
> `assert_latent_norm_applied` convention guard (`latent_norm_applied=true`). Validated by a **direction-pinned**
> unit test (round-trip is direction-blind) **and** a real-weight gate (real image → normalized **std 1.11** vs
> inverted **3.10** → gate discriminates). Open item: VAE **mean vs sample** for the training-encode is flagged
> **RESOLVED**: use the VAE **mean** (chunk[0]) — the fork's *training* cache path confirms it
> (`ideogram4_autoencoder.py:337`), and it's guarded by a determinism/mean test. 9 tests (20 total).

Remaining Phase 2: `ideogram4_cache_latents` + `ideogram4_cache_text_encoder_outputs` (53,248-dim, fp8-key
fixed, caption verifier warn-only); cache_io by intent per §7; wire `assert_latent_norm_applied` into the
cache writer. Exit: cache→load round-trip on a **real** `encode()` shape (guards B3) **plus** an assertion
that raw tokens can't be cached as training-ready.

**Phase 3 — Training.** `ideogram4_train_network` subclassing `NetworkTrainer` with zimage/flux_2-shaped
`process_batch`, canonical t-convention, blissful flow-matching knobs. `--use_mask_loss` either fully wired
(token-grid mask patchify) **or rejected at validation** — never a silent no-op. Block swap reimplemented
against blissful's restore contract (`trainer_base.py:737`), with the no-grad-restore lifecycle test
(critical once prior preservation is enabled). Exit: 50-step run, loss descends, LoRA saves/loads.

**Phase 4 — Inference + sampling.** `ideogram4_generate_image` with asymmetric CFG (load uncond DiT in
`on_before_sample_images`, free after), canonical schedule, blissful latent-preview/guidance integration.

**Phase 5 — Hardening.** Test suite. **Landed in Phase 1.5:** fp8 shim dtype/scale tests (incl. non-unit
scale), fp8 patched-count guard, patchify-convention pin, RoPE-`inv_freq`-float32, raw-VAE BN-bypass.
**Remaining:** **fp8-LoRA gradient test** (analogous to `test_lora_conv2d_forward.py`); **VAE layout test**
(Comfy strict-loads; HF conversion branch); **canonical t-parity test**; **zero-target LoRA guard test**;
**fp8 text-cache key round-trip test** (the `bucket.py` rsplit bug); **`latent_norm` guard test** (raw tokens
cannot be cached as training-ready; `latent_norm_applied` metadata present); **cache round-trip test**
(32→128 boundary, BN untouched); **block-swap no-grad restore test**; **other-arch cache byte-identity test**
(empty `extra_metadata` = no-op). Then mask-loss as a fast-follow.

---

## 9. Storage strategy (53,248-dim text cache; ~96 GB free of 1.8 TB)

bf16 text cache ≈ **27 GB / 1k images at 256 tokens** (≈218 GB at the 2,048-token cap). On this box:
- Keep token budgets tight; cache to the SwarmUI/Downloads spindle if needed, not `/`.
- fp8 text cache halves it — **but H5 must be fixed first** (the fp8 key currently `KeyError`s).
- Quantify per-dataset before a long run; this is a genuine go/no-go input, not an afterthought.

---

## 10. Borrow vs rebuild

| Borrow (faithful) | Rebuild on blissful seams (broken/divergent) |
|---|---|
| Transformer math (`ideogram4_model.py`, byte-identical to canonical) | fp8 loading (use §5, not `Fp8Linear`) |
| `constants.py`, mRoPE/AdaLN params | VAE (reuse flux_2; drop the 449-line dup) |
| caption_verifier (lightweight, **inference-only**, off-by-default) | `process_batch`/loss routing (mask seam) |
| sampler preset *values* | denoising t-schedule (canonical convention) |
| | cache_io (by intent), tokenizer loading (offline), text-cache key |

**Bottom line:** feasible and worth doing; the fork is a useful *model reference* and a catalogue of exactly
which harness mistakes to avoid. Net new code is modest because the two heaviest pieces — the VAE and the
fp8 infrastructure — already exist in blissful and are better than the fork's versions.

---

## 11. Addendum — second-review amendments (2026-06-07, all verified against source)

A second technical review of this plan produced 7 findings. Verdicts after primary-source verification:

| # | Finding | Verdict | Action |
|---|---|---|---|
| 1 | **fp8 shim must cast `scale_weight` → compute dtype.** Published scales are fp32; dequant inherits `scale_weight.dtype` (`fp8_optimization_utils.py:409-412`) → fp32 weight → `F.linear(bf16,fp32)` crash. | ✅ **Confirmed** (genuine correction; reproduced). | Folded into §5 shim as a REQUIRED step + `use_scaled_mm=False`. |
| 2 | **VAE source/layout must be explicit.** Comfy `flux2-vae.safetensors` strict-loads via `load_ae`; HF VAE needs diffusers conversion. Use raw `ae.encoder/decoder`, not `ae.encode()/decode()` (BN baked in at `flux2_models.py:377-388`). | ✅ **Confirmed + sharpened** (HF has 74 `down_blocks` keys; blissful's `encode()` == the fork's bug shape). | Folded into §6 + Phase 1/2. |
| 3 | **`--use_mask_loss` must be binary** (implemented or rejected at validation), never a half-wired no-op. | ✅ **Agree** (better invariant than "seam wired"). | Updated §6 + Phase 3. |
| 4 | **Reuse the Blackwell SDPA guard by import, not duplication.** | ✅ **Confirmed** (guard at `attention.py`, Blackwell cuDNN-SDPA). | Route Ideogram attention through blissful's helper; no second guard. |
| 5 | **First-class `lora_ideogram4.py`** (target `Ideogram4TransformerBlock` + `attention.qkv/o`, `feed_forward.w[123]`) **with normal Linear discovery + zero-target guard** — drop `linear_module_class_names`/`Fp8Linear`. | ✅ **Agree** (consistent with §5 centerpiece). | Keep the fork module's *targets + zero-guard*; drop the Fp8Linear plumbing. |
| 6 | **Block-swap restore against blissful's contract** + lifecycle test. | ✅ **Confirmed; corrects a stale citation** — the restore hook is now `trainer_base.py:737` (this doc/`CLAUDE.md` previously said `hv_train_network.py:623`). | Updated Phase 3; flag the stale citation for a `CLAUDE.md`/memory fix. |
| 7 | **Surface the non-commercial license.** | ✅ **Confirmed** (`LICENSE.md`: "training, fine tuning, or distilling… for commercial use… is not a Non-Commercial Purpose"). | Add a license banner to `docs/ideogram4.md` and handoff notes: **Ideogram-4 LoRAs are non-commercial-only.** |

**Net:** the review found one true correction (the fp8 dtype cast) and several precision/sequencing
improvements. No finding contradicts the core verdict or the centerpiece fp8 strategy; both are reinforced.
The two amendments the user asked to fold first (fp8 dtype cast, VAE source/layout) are now **Phase-1 spec +
exit criteria**, not optional polish — they are precisely what makes Phase 1 "actually runs a forward"
instead of merely "loads," and the fp8-dtype one also unblocks the Phase-0 empirical confirmation (since the
fork's own generate is dead and cannot be the oracle).
