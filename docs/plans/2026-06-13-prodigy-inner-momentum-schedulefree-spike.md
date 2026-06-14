# ProdigyPlus Inner-Momentum Schedule-Free Spike

**Date:** 2026-06-13
**Status:** COMPLETE — A/B run 2026-06-13. Verdict: **β1=0.5 is a no-op on samples at DLAY scale (bs2, 150 img, Klein 9B, 1500 steps); keep the baseline. β1=0.9 — the paper's actual value — is the open test, not yet run.** See Results.
**Primary optimizer:** `prodigyplus.ProdigyPlusScheduleFree`
**Spike optimizer:** `musubi_tuner.optimizers.prodigy_plus_inner_momentum.ProdigyPlusInnerMomentumScheduleFree`

---

## Summary

ScheduleFree+ is not a drop-in upgrade for Blissful LoRA training. The Meta reference implementation is a
research optimizer that needs `step_func(function_value=...)`, while Blissful's training loop uses the normal
`optimizer.step()` boundary through Accelerate. More importantly, the Polyak step would be driven by a noisy
batch-1 diffusion loss that is timestep-dependent, mask/prior-weighted, sometimes Huber-shaped, and already
known to be an unreliable proxy for persona sample quality.

The useful ScheduleFree+ idea for this repo is narrower: reintroduce Adam inner momentum inside the existing
ProdigyPlusScheduleFree Schedule-Free path and test whether it improves batch-1 LoRA stability or samples.
This keeps the current Prodigy shared-d estimator, LoRA+ grouping, Schedule-Free train/eval lifecycle, and
trainer plumbing intact.

---

## Corrected Optimizer Comparison

With `split_groups_mean=True`, ProdigyPlusScheduleFree is a global-estimator path, not an independent
per-group-step path:

| Topic | Current ProdigyPlusScheduleFree | ScheduleFree+ Polyak |
|---|---|---|
| Global step scalar | `shared_d * lr` from harmonic mean of group `d` values | `polyak_lr = max(0, f + ip_term) / grad_l1_ema` |
| Per-group role | Raw group `d` is a health signal; applied step is d-raw-free under `split_groups_mean` | No per-group `d` |
| Per-group LR ratios | Preserved through group `lr` | Preserved through `group_lr = lr * polyak_lr` |
| Known risk | Harmonic-minimum collapse if one group pins near `d0` | Global L1/loss scalar dominated by the current group, timestep, mask, or prior condition |
| Integration cost | Already wired | Needs loss value threaded into optimizer stepping |

The inner-momentum spike keeps the left column and changes only the preconditioned update numerator from the
current instantaneous `d * grad` to an EMA of `d * grad`.

---

## Implementation

The spike lives in repo code, not in the shared site-packages copy of `prodigyplus`:

```toml
[optimizer]
optimizer_type = "musubi_tuner.optimizers.prodigy_plus_inner_momentum.ProdigyPlusInnerMomentumScheduleFree"
learning_rate = 1.0
optimizer_args = [
  "betas=(0.95, 0.99)",
  "d_limiter=True",
  "prodigy_steps=500",
  "weight_decay=0.0",
  "weight_decay_by_lr=True",
  "d_coef=1.0",
  "d0=1e-06",
  "eps=1e-8",
  "use_bias_correction=True",
  "schedulefree_c=12",
  "use_stableadamw=True",
  "factored=True",
  "factored_fp32=True",
  "split_groups=True",
  "split_groups_mean=True",
  "use_cautious=False",
  "stochastic_rounding=True",
  "fused_back_pass=False",
  "use_orthograd=True",
  "use_focus=False",
  "use_speed=True",
  "use_grams=False",
  "inner_beta1=0.5",
]
```

Naming matters: the class name ends in `ScheduleFree`, so `trainer_base.is_schedulefree_optimizer()` still
detects it and installs the dummy scheduler/train/eval lifecycle.

The default is `inner_beta1=0.0`, which is intended to be exact behavioral parity with upstream
`ProdigyPlusScheduleFree`.

The subclass copies `initialise_state` and `step_param_schedulefree` from `ProdigyPlusScheduleFree` at
`CoreOptimiser.VERSION == (2, 0, 0)` (upstream repo tag `v2.0.1`). Re-sync this subclass when the local
`prodigyplus` package is bumped.

### Raw-d Inertness Invariant

The first moment must be scaled by the raw group `d`, just as Prodigy's second moment is scaled by `d**2`:

```text
u = EMA(d * g) / sqrt(EMA(d**2 * g**2)) -> EMA(g) / sqrt(EMA(g**2))
```

If the first moment stored plain `g`, a stray `1/d` would survive and raw per-group `d` would reach the
weights again. The test suite explicitly checks that a forced 1000x raw-`d` gap still produces equal movement
with `inner_beta1=0.75`.

### Bias Correction Choice

The inner first moment is intentionally not bias-corrected for this initial spike. That is a simplification
relative to Meta's ScheduleFree+ reference, which bias-corrects its Adam first moment. At `inner_beta1=0.5`,
the effect is small after the early RAdam gate has opened, so this keeps the first implementation minimal while
testing the most plausible low-beta setting. Before escalating to high values such as `inner_beta1=0.9`, add or
explicitly evaluate first-moment bias correction so the experiment remains faithful to the ScheduleFree+ idea.

---

## Why Start Low

The production DLAY-style configs already use:

- Schedule-Free outer momentum via `betas=(0.95, 0.99)`.
- `schedulefree_c=12`, which makes the x/y averaging more responsive.
- Batch size / accumulation of 1, so gradients are noisy.

Inner momentum may help noisy gradients, but it stacks another memory filter on top of the Schedule-Free
outer averaging. Start at `inner_beta1=0.5`, hold every other optimizer and training option fixed, and only
try `0.75` or `0.9` if the first A/B is stable and samples justify the extra smoothing.

---

## A/B Protocol

Use a production-shaped DLAY FLUX.2 Klein config, but keep the run short enough to be a decision gate rather
than a full training commitment.

### Conditions

| Condition | Optimizer | Only changed arg |
|---|---|---|
| A: baseline | `prodigyplus.ProdigyPlusScheduleFree` | none |
| B: inner-0 parity | `ProdigyPlusInnerMomentumScheduleFree` | `inner_beta1=0.0` |
| C: inner momentum | `ProdigyPlusInnerMomentumScheduleFree` | `inner_beta1=0.5` |

Run condition B only once when validating the branch or resuming from a clean code state. If B matches A on
telemetry and smoke samples, future A/Bs can compare A vs C directly. Unit tests already prove math parity at
`inner_beta1=0.0`; condition B is mainly a pipeline integration check for optimizer loading, lifecycle switching,
scheduler logging, and sample-time behavior.

### Hold Fixed

- Dataset, cache, seed, prompt file, sample cadence, output resolution.
- LoRA rank/alpha and `loraplus_lr_ratio`.
- Mask loss, prior preservation, EMA teacher settings.
- `schedulefree_c`, `use_speed`, `use_orthograd`, `factored`, `use_bias_correction`.
- Compile/checkpointing flags.

### Required Telemetry

Inspect the exact TensorBoard event files for:

- `lr/shared_d`
- `lr/applied_dlr/*`
- `lr/applied_dlr_eff/*`
- `lr/d*lr/*` only as historical raw-d context
- `prior/teacher_mode_ema_used`
- `masked_loss/prior_fraction`
- `loss/raw_mse` and other raw loss-stat tags if `--log_loss_stats` is enabled

Do not judge the experiment by `loss/average` alone.

### Sample Gate

Compare fixed-prompt samples at matched steps, starting with:

- Step 500: early stability and likeness direction.
- Step 1000: post-warmup direction, after Prodigy freeze and early EMA/prior behavior are visible.
- Step 1500 or 2000: only if step 1000 is ambiguous.

Promote the variant only if samples improve or stability clearly improves without loss of likeness. A lower
loss curve with worse samples is a rejection.

---

## Results (2026-06-13 A/B run)

Ran under `/home/dustin/output/ab_gate_inner_momentum/` (driver + per-condition launchers, detached,
systemd 80G cap). Reused the v9 1328 / gamma-0.7 caches (resolution corrected 1024→1328 in the gate dataset
TOML). All three conditions exited 0. A & C: 1500 steps (~94 min each); B: 300 steps.

**Setup actually used (v9 recipe):** dim 64 / alpha 32 / loraplus 12 / `use_dora=True`; Prodigy
`prodigy_steps=200, schedulefree_c=12, use_speed=True, use_bias_correction=False`; EMA prior (0.9995, thr 700);
huber; compile; **batch_size 2**; seed 42; 150 images at 1328.

**B parity (inner_beta1=0 vs upstream):** PASS. `loss/average` tracked A to **max 0.24% over all 300 steps**
(max |Δ| 2.8e-4) — compile/FP cross-process noise. Confirms the subclass is behaviorally identical to upstream
in the full pipeline (DoRA 3-group + EMA + compile + bs2). Matches the unit-test `atol=0` parity.

**C vs A (inner_beta1=0.5 — the experiment): no-op on samples.** Caveat up front: β1=0.5 is a ~2-step EMA —
*barely* momentum. The paper's mechanism lives at β1≈0.9 (its reference default, and the value behind its
batch-size figures), so this arm tests **weak** inner momentum, not the paper's setting.

| metric @1500 | A | C | Δ(C−A) |
|---|---|---|---|
| `loss/average` | 0.11752 | 0.11750 | −0.00002 |
| `huber/target_linear_frac` | 0.13062 | 0.12950 | −0.0011 (~0.9% lower) |
| `prior_fraction` | 0.0036 | 0.0040 | +0.0004 |

`loss/average` and `prior_fraction` are at noise level. The huber_lin_frac delta, however, is **monotonic** —
−9e-5 → −2.5e-4 → −1.1e-3 across 500/1000/1500, sign-consistent and growing ~4× in the last 500 steps — so it
is a *faint real* signal in the helpful direction (lower = binding faster), NOT noise. It is tiny (~0.9%), below
the magnitude prior DLAY work established correlates with sample quality, and samples were equal — so it does
not move the verdict, but "all noise" would be the wrong reading. Samples (the verdict): prompt 00 (portrait,
steps 500 & 1500), prompt 05 (768×1360 pose, 1500), and prompt 01 (bare-identity, 1500) are **equivalent
identity, quality, framing, and pose stability** in A and C — no doubling, no likeness gain/loss. Gender
stability spot-checked on the bare-identity / they-them prompts 00 & 01 (the documented drift-risk class) at
step 1500: both A and C bind male consistently — no drift in either, no difference between them. Per-sample
differences (expression, hair, hand position) are variance between near-identical weight states.

**Interpretation.** Exactly what ScheduleFree+ predicts: inner momentum helps at *large* batch; at bs2/150-img
it's a no-op. The Schedule-Free outer averaging (β1=0.95) + `schedulefree_c=12` already supply the smoothing
inner momentum would add. **Decision: keep `prodigyplus.ProdigyPlusScheduleFree` as production.** The
`ProdigyPlusInnerMomentumScheduleFree` subclass stays in-tree (validated, opt-in via `inner_beta1`) for a
future *larger-batch* regime where the paper says momentum would matter — not for current DLAY runs. **Open test (the clean definitive arm):** β1=0.9 — the paper's actual value — with the **M2 bias-correction fix
applied FIRST** (at 0.9 the uncorrected first moment is suppressed for ~20 steps and would confound the early
trajectory). One ~94-min run, everything else reused. Run it only if (a) the recipe moves to larger batch
(where the paper says momentum matters) or (b) the faint monotonic huber trend warrants closure. Keep-baseline
holds regardless: even a marginal 0.9 win would carry the over-smoothing + bias-correction caveats you don't
want in a tuned production recipe, and the sample-gated standard did not move at 0.5.

**Follow-up found by running it (missed in code review):** the inner-momentum subclass initially did NOT get the
Prodigy applied-step telemetry (`lr/shared_d`, `lr/applied_dlr/*`, `lr/d*lr/*`), because `trainer_base.py`
gated those tags on `optimizer_type.lower().endswith("prodigyplusschedulefree")` — a stricter, separate name
gate than the lifecycle gate (`endswith("schedulefree")`, which the subclass clears, which is why it trains
fine). Core verdict metrics (`loss/average`, `huber_lin_frac`, `prior_fraction`) are unaffected. The adaptation
health check itself is duck-typed on `d`/`d0` param-group keys and already works for the subclass. Fixed after
the A/B by changing the step-log telemetry gate to "ScheduleFree optimizer with d-adapting param groups" and
adding regression tests for the subclass tags plus a non-d-adapting ScheduleFree negative case.

## Tests Added For The Spike

Primary test file:

```bash
./venv314/bin/python -m pytest -q tests/test_prodigy_inner_momentum_schedulefree.py
```

Coverage:

- Optimizer dotted path and `ScheduleFree` lifecycle detection.
- Upstream ProdigyPlus fork point remains at `CoreOptimiser.VERSION == (2, 0, 0)`.
- `inner_beta1=0.0` parity with upstream `ProdigyPlusScheduleFree`.
- Raw per-group `d` remains inert under `split_groups_mean=True` with `inner_beta1=0.75`.
- LoRA+ group LR ratio survives when this optimizer is used.
- Missing `exp_avg` is allocated when resuming or reusing state created with `inner_beta1=0.0`.

Recommended broader validation before a run:

```bash
./venv314/bin/python -m pytest -q \
  tests/test_prodigy_inner_momentum_schedulefree.py \
  tests/test_prodigy_d_inert.py \
  tests/test_optimizer_adaptation_check.py \
  tests/test_lora_dora_param_group.py
./venv314/bin/ruff check \
  src/musubi_tuner/optimizers/prodigy_plus_inner_momentum.py \
  src/musubi_tuner/optimizers/__init__.py \
  tests/test_prodigy_inner_momentum_schedulefree.py
./venv314/bin/python -m compileall -q src/musubi_tuner/optimizers
```

---

## Non-goals

- Do not replace ProdigyPlusScheduleFree as the default optimizer from this spike alone.
- Do not port Meta's Polyak `AdamCScheduleFreePlusPaper` into the trainer until inner momentum has been
  evaluated on samples.
- Do not edit the shared site-packages `prodigyplus` copy or require `pip install -e ~/prodigy-plus-schedule-free`
  for this experiment.
