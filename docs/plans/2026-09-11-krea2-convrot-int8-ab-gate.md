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

Upstream's own explanation, confirmed here: K2's fp8 path **dequantizes to bf16 on every forward**
(no `scaled_mm`), so it pays a dequant per Linear and materializes a transient bf16 weight; ConvRot runs
a true int8 GEMM and materializes nothing. That single mechanism explains the speed **and** the memory
win together.

**C falsifies the competing explanation.** A is *forced* onto `--sdpa` by blissful's fp8 + fused-attention
guard, so the fp8 result could have been an attention handicap rather than a quantization one. It is not:
C (flash) is only **1% faster** than B (sdpa), so the attention backend is nearly irrelevant at this shape
and the whole 3.26x is attributable to the quantization path. (The freedom to use fused attention under
ConvRot is therefore a correctness/ergonomics win, not a measurable speed win here.)

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

**B vs C is the control, and it is the important row.** Those two runs share the quantizer, the seed and
the data order, and differ only in the attention kernel's floating-point summation order — yet their
learned side lands 1.20x apart in magnitude at only 0.937 median cosine, with individual tensors down to
0.107. That is barely better than the A-vs-B figure (0.916) across *different* quantizers. So the
`lora_up` spread is **intrinsic 300-step run variance, not a quantization-fidelity signal**, and the
2.12x `d*lr` spread between A and B sits in the same band as the 1.38x that the kernel change alone
produces. This independently reproduces the known local result that Prodigy `d*eff_lr` differences do
not track sample quality (a prior DLAY A/B saw 22,000x with visually indistinguishable LoRAs).

The `lora_down` column is the meaningful correctness read: the side dominated by its shared initialization
is identical to ~2e-5 across every pair, i.e. base quantization is not structurally corrupting training.

**Consequence for follow-ups:** an unquantized bf16 reference condition (staged as
`configs/D_bf16_swap.toml`, `blocks_to_swap=16` to fit 25.6 GB of bf16 weights) was considered and is
**not worth running** — the B/C control shows it would land inside the same noise band and could not
adjudicate which quantizer's gradients are "truer". The decisive test is a **production-length run**
(the v3 recipe is 20 epochs / ~3000 steps) compared on samples, not another 300-step condition.

## Verdict

Ship `--convrot_int8` as an opt-in flag. The performance claim is solid and mechanistically explained;
structural correctness is verified (same 224 layers, same adapter keyset, `lora_down` parity). Convergence
equivalence over a full run is **not** claimed and needs a production-length comparison before ConvRot
replaces fp8 in a real DLAY recipe.

Untested combinations, deliberately not claimed as supported:

- `--block_swap_h2d_only`: the H2D ring's in-place `copy_` bumps the autograd version of the very tensor
  ConvRot's `save_for_backward` holds. Gradient checkpointing (which H2D already requires) should keep
  that consistent, but there is no test.
- `--compile`: ConvRot Linears are excluded automatically (`disable_linear`), so compile should be a
  no-op over them; not measured.
- Multi-GPU: untested upstream as well.
