# Krea 2 ConvRot int8 — A/B gate receipts (2026-09-11)

Empirical gate for the `--convrot_int8` port (upstream `kohya-ss/musubi-tuner` #1008 + #1063 fix,
Krea 2 wiring `fe4818da`). Artifacts: `/home/dustin/output/ab_gate_krea2_convrot/`
(`configs/`, per-condition `stdout.log` + `logs/` + `gpu_samples.csv`, `analyze.py`).

## Setup

Derived from the production `dlay_krea2_v3` recipe, with Turbo sampling removed (`--turbo_dit` is
rejected under ConvRot upstream) and no sample generation, so every condition measures training only.

| | |
|---|---|
| Base | Krea 2 **RAW** DiT, `/home/dustin/Training_Models_Krea-2-Raw/raw.safetensors` (12.8B) |
| Dataset | DLAY `subject-150` (150 images) + `masks-150`, 1024x1024, batch 1, `use_mask_loss`, no prior teacher |
| Network | `networks.lora_krea2`, dim 64 / alpha 64 -> 264 LoRA modules, 794 adapter tensors |
| Optimizer | `prodigyplus.ProdigyPlusScheduleFree`, lr 1.0, `split_groups` auto-disabled (single param group) |
| Other | bf16, `krea2_shift`, gradient checkpointing, `max_data_loader_n_workers 0`, **seed 42**, 300 steps |
| Machine | RTX 5090 32 GB (sm_120), torch `2.15.0a0+git5a7841d`, Triton 3.8.0 |

Three conditions, differing **only** in the two fields named:

| | quantization | attention |
|---|---|---|
| **A** | `--fp8_base --fp8_scaled` (the v3 production recipe) | `--sdpa` (forced: blissful rejects fp8 + fused attention) |
| **B** | `--convrot_int8` (bwd `bf16`) | `--sdpa` |
| **C** | `--convrot_int8` (bwd `bf16`) | `--flash_attn` |

## Results

Steady state measured over steps 201-300 (Triton autotune and Prodigy's initial ramp are excluded).
Peak VRAM is the max of 1 Hz `nvidia-smi memory.used` over the whole run.

| | s/it | vs A | peak MiB | vs A | wall | loss/current 101-300 | loss/average @300 | d*lr @300 |
|---|---|---|---|---|---|---|---|---|
| **A** fp8 + sdpa | 6.704 | — | 26,662 | — | 2035 s | 0.07878 | 0.07957 | 1.434e-05 |
| **B** convrot + sdpa | **2.079** | **-69.0%** | **24,332** | **-2,330** | 648 s | 0.07881 | 0.07962 | 6.754e-06 |
| **C** convrot + flash | **2.059** | **-69.3%** | **24,228** | **-2,434** | 642 s | 0.07882 | 0.07963 | 9.298e-06 |

**ConvRot is 3.26x faster and uses 2.4 GB less peak VRAM than the fp8 path, at 0.08% loss parity.**

### Structural correctness

- `apply_convrot_int8_monkey_patch` patches **224 Linears** — byte-identical to the count the proven fp8
  path reports (`apply_fp8_monkey_patch`: 224). **Zero** layers skipped: every K2 Linear's `in_features`
  is divisible by the 256 ConvRot group size.
- LoRA still builds its full **264 modules** on top of the patched Linears, and the saved adapter has an
  **identical 794-key keyset** across all three conditions. Keeping the module an `nn.Linear` with a
  patched forward (rather than a custom class) is what makes LoRA targeting, block-swap `.weight.data`
  streaming, and compile exclusion all keep working.
- The fused Triton path is active in B and C (no "triton is not available" fallback warning).

### Why the win is this large (upstream reports only ~1.2x on their Blackwell card)

K2's fp8 path **dequantizes on every forward** (`use_scaled_mm=False`, so
`fp8_linear_forward_patch` takes its `F.linear`-on-a-dequantized-weight branch), paying a dequant per
Linear and materializing a transient full-size weight; ConvRot's Triton forward runs a true int8 GEMM
and materializes no dequantized weight. That is a real difference and plausibly the dominant one.

**A single-mechanism attribution is NOT established, and an earlier revision of this file over-claimed
it.** Two corrections:

1. **A vs B changes more than the Linear implementation — it also changes the dtype flowing downstream.**
   The fp8 dequant branch runs under `torch.autocast(enabled=False)` with `linear_dtype = x.dtype`, so it
   **returns the input dtype**; ConvRot's forward explicitly casts to the autocast dtype. Measured on CPU
   with a bf16 autocast active: an **fp32** input (which K2's fp32 modulation adds produce) yields **fp32**
   out of fp8 and **bf16** out of ConvRot; a bf16 input yields bf16 from both. So condition A plausibly
   feeds fp32 activations into attention where B and C feed bf16, and the 3.26x is a sum of the quantized
   Linear *and* that precision change — not the Linear alone.
2. **B vs C therefore does not isolate attention cost in A.** It measures flash-vs-sdpa **under ConvRot**,
   i.e. at bf16. It says nothing about fp32-sdpa (what A actually runs) vs bf16-sdpa, so "the attention
   backend is nearly irrelevant" holds only for the bf16 conditions. The earlier claim that C "falsifies"
   an attention contribution in A was wrong.

Also note ConvRot does **not** materialize nothing everywhere: the default `--convrot_int8_bwd bf16`
backward explicitly builds `w_rot = wq.to(dtype) * scale`, and the no-Triton eager forward fallback
dequantizes too. "Materializes nothing" is true only of the fused Triton forward.

**What is safe to state:** the 3.26x and -2.4 GB are real, reproducible end-to-end differences between
two *legal, production-shaped configurations* (fp8 cannot use fused attention here, so A+sdpa is the
best available fp8 config). Isolating how much of it is the int8 GEMM versus the activation-precision
change would need a further condition — e.g. fp8 with a forced bf16 output cast, or ConvRot with an fp32
output cast — which has not been run.

## What the gate does NOT establish — and the control that bounds it

Loss parity at 300 steps is **weak** evidence on its own: `lora_up` starts at exactly zero, so at
`median ||lora_up|| ~ 3e-2` (against `||lora_down|| ~ 4.6`) the adapter is still nearly a no-op and both
runs are largely measuring the frozen base on identical data.

Direct adapter comparison (median over the 264 tensors of each role):

| pair | `lora_down` (shared kaiming init) | `lora_up` (everything learned) |
|---|---|---|
| A vs B | cos **0.999979** (min 0.9998) | cos 0.9157 (min 0.049) |
| A vs C | cos **0.999984** (min 0.9999) | cos 0.9181 (min 0.114) |
| **B vs C** | cos **1.000003** (min 1.0000) | cos 0.9372 (min 0.107) |

| | median `\|\|lora_up\|\|` |
|---|---|
| A | 6.2458e-02 |
| B | 2.8100e-02 |
| C | 3.3825e-02 |

ratios: A/B **2.22x**, A/C **1.85x**, **C/B 1.20x**

**B vs C is the informative row, but it does not license calling this "noise".** Those two runs share the
quantizer, the seed and the data order and differ only in the attention kernel — yet their learned side
lands 1.20x apart in magnitude at 0.937 median cosine, barely better than the A-vs-B figure (0.916)
across *different* quantizers. What that supports is a **weakened inference**: a change unrelated to the
quantizer moves the learned side by the same order as the quantizer change does, so the A-vs-B `lora_up`
(2.22x) and `d*lr` (2.12x) spreads **cannot be attributed to quantizer fidelity** on this evidence.

It does **not** establish a noise band. These are single runs per condition, and the attention backend is
itself a systematic change (different kernels and summation order), not a repeated draw from a noise
distribution. Separating run-to-run variance from a systematic quantizer effect would need repeats at
fixed configuration, which were not run. An earlier revision of this file went too far in the other
direction — first calling the A/B spread a meaningful trajectory difference, then calling it settled
noise; **both were over-readings of n=1 per condition.** The supported statement is that quantizer
effects on the learned updates are **unresolved**.

The `lora_down` tensors remain close across conditions (~2e-5 cosine), largely reflecting shared
initialization. **This does not independently validate the base computation or the gradient path** — a
gradient path that was disconnected entirely would leave `lora_down` sitting at exactly that shared
initialization and so would look identical. Structural correctness rests on the layer/keyset counts above
and on the unit tests, not on this column.

**Consequence for follow-ups:** an unquantized bf16 reference condition is staged as
`configs/D_bf16_swap.toml` (`blocks_to_swap=16` to fit 25.6 GB of bf16 weights) and was **not run**.
Declining it is a cost judgement, not a prediction — its outcome cannot be forecast from these controls.
The decisive test for production use remains a **production-length run** (the v3 recipe is 20 epochs /
~3000 steps) compared on samples.

## Verdict

Ship `--convrot_int8` as an opt-in flag. The performance measurement is solid (though its decomposition
into int8-GEMM versus activation-precision effects is not); structural correctness is verified (same 224
layers, same adapter keyset, no gross `lora_down` drift). Convergence equivalence over a full run is
**not** claimed and needs a production-length comparison before ConvRot replaces fp8 in a real DLAY recipe.

Adapters that read the raw base weight are rejected fail-fast, because under ConvRot `.weight` holds
Hadamard-rotated int8 codes: DoRA, PiSSA init, the LyCORIS bridge, `--base_weights` (whose merge runs
*after* quantization), and DoRA inferred from a `--network_weights` / `--dim_from_weights` checkpoint.
Two dtype-level guards in `networks/dora_utils.py` back this up so non-trainer callers are covered too:
`raise_if_opaque_quantized` at the read-only true-weight-space consumers (both DoRA weight norms,
`pre_calculation`, PiSSA init), and the stricter `raise_if_unmergeable_base` — which refuses fp8 as well,
since a merge must write its result back — at **every** destructive merger: `LoRAInfModule.merge_to`,
`LoHaModule.merge_to`, `LoKrInfModule.merge_to`, and all three tensor helpers behind
`merge_nonlora_to_model` (LoRA/LoHa/LoKr). The LoRA tensor helper is guarded *before* its
`compute_dtype = float16 if itemsize == 1` cast, which would otherwise erase the int8 evidence.
Each guard refuses without mutating the weight or consuming adapter keys, and stays a no-op when no
adapter key matches. See `tests/test_convrot_quantized_base_guards.py` (46 tests).

Untested combinations, deliberately not claimed as supported:

- `--block_swap_h2d_only`: the H2D ring's in-place `copy_` bumps the autograd version of the very tensor
  ConvRot's `save_for_backward` holds. Gradient checkpointing (which H2D already requires) should keep
  that consistent, but there is no test.
- `--compile`: ConvRot Linears are excluded automatically (`disable_linear`), so compile should be a
  no-op over them; not measured.
- Multi-GPU: untested upstream as well.
