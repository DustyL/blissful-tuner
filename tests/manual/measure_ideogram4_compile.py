"""Manual gate: Ideogram-4 torch.compile on the REAL prequantized-fp8 DiT (backlog P1-1).

Validates the 4-way interaction (compile x fp8 monkey-patched forwards x DoRA LoRA x gradient
checkpointing) that the unit smoke (tests/test_ideogram4_compile_smoke.py) cannot reach on CPU,
and measures the wall-clock win that justifies the hook.

The PASS/FAIL criterion is LOSS-TRAJECTORY PARITY, not bitwise forward parity — matching the
repo's established A/B methodology (CLAUDE.md "Performance Investigation Pattern", FLUX.2 Klein
v8 gate). Rationale, established empirically on 2026-06-11: eager and compiled are each bit-
deterministic, but Inductor's fusion/reduction reordering (and SDPA backend dispatch) produces a
systematic bf16 forward deviation of ~1.4% mean-rel with a rare ~5% tail (0.2% of elements >1%).
That tail is invisible at the training level: 20 AdamW steps from identical LoRA init with
identical per-step inputs tracked to 0.037% mean / 0.227% max per-step loss difference. A bitwise
forward gate would reject a configuration that trains identically.

First gate run (2026-06-11, 5090, rank-16 DoRA, GC on, 1024 res, 93-token captions):
eager 3.396 s/step -> compiled 2.444 s/step = +28.0% throughput. On a 5000-step DLAY run that is
~80 minutes saved.

    ./venv314/bin/python tests/manual/measure_ideogram4_compile.py \
        --model-path ~/Training_Models_Ideogram/diffusion_models/ideogram4_fp8_scaled.safetensors \
        --resolution 1024 --steps 10 --parity-steps 20
"""

import argparse
import copy
import os
import statistics
import time
from types import SimpleNamespace

import torch

import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.ideogram4.ideogram4_utils import load_ideogram4_transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

IDEOGRAM4_IMAGE_PATCH = 16


def _step_inputs(grid, text_len, device, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, 128, grid, grid, generator=g).to(device)
    noise = torch.randn(1, 128, grid, grid, generator=g).to(device)
    text_features = [torch.randn(text_len, 53248, generator=g).to(device=device, dtype=dtype)]
    t = torch.rand(1, generator=g).to(device)
    return latents, noise, text_features, t


def _loss_trajectory(model, network, init_state, grid, text_len, device, dtype, steps):
    """Run `steps` AdamW steps from `init_state` with per-step fixed seeds; return the loss curve."""
    network.load_state_dict(init_state)
    opt = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    for i in range(steps):
        latents, noise, text_features, t = _step_inputs(grid, text_len, device, dtype, seed=1000 + i)
        pred, target = ideogram4_flow_matching_target(model, latents, text_features, noise, t, network_dtype=dtype, device=device)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


def _timed_steps(model, network, grid, text_len, device, dtype, steps, seed0=5000):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(steps):
        latents, noise, text_features, t = _step_inputs(grid, text_len, device, dtype, seed=seed0 + i)
        pred, target = ideogram4_flow_matching_target(model, latents, text_features, noise, t, network_dtype=dtype, device=device)
        torch.nn.functional.mse_loss(pred.float(), target.float()).backward()
        network.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--text-len", type=int, default=93)  # real DLAY caption length
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10, help="timed throughput steps per mode")
    ap.add_argument("--parity-steps", type=int, default=20, help="AdamW steps for the loss-parity gate")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    grid = args.resolution // IDEOGRAM4_IMAGE_PATCH

    model = load_ideogram4_transformer(os.path.expanduser(args.model_path), dtype=dtype, loading_device=device)
    network = lora_ideogram4.create_arch_network(1.0, args.rank, args.rank * 2.0, None, None, model, use_dora="True")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    model.requires_grad_(False)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    model.enable_gradient_checkpointing()
    init_state = copy.deepcopy(network.state_dict())
    print(f"DiT(fp8) + DoRA LoRA rank {args.rank} resident: {torch.cuda.memory_allocated() / 1e9:.2f} GB, GC on")

    # --- eager phase --------------------------------------------------------------------------------
    eager_losses = _loss_trajectory(model, network, init_state, grid, args.text_len, device, dtype, args.parity_steps)
    _timed_steps(model, network, grid, args.text_len, device, dtype, args.warmup)
    eager_s = _timed_steps(model, network, grid, args.text_len, device, dtype, args.steps)
    print(f"eager:    {eager_s:.3f} s/step")

    # --- compiled phase (production-default args, via the trainer hook) ------------------------------
    compile_args = SimpleNamespace(
        compile=True,
        compile_backend="inductor",
        compile_mode="default",
        compile_dynamic=None,
        compile_fullgraph=False,
        compile_cache_size_limit=None,
    )
    t0 = time.perf_counter()
    Ideogram4NetworkTrainer().compile_transformer(compile_args, model)
    _timed_steps(model, network, grid, args.text_len, device, dtype, args.warmup)  # triggers compilation
    print(f"compile + warmup: {time.perf_counter() - t0:.0f} s")

    compiled_losses = _loss_trajectory(model, network, init_state, grid, args.text_len, device, dtype, args.parity_steps)
    compiled_s = _timed_steps(model, network, grid, args.text_len, device, dtype, args.steps)
    print(f"compiled: {compiled_s:.3f} s/step")

    # --- verdict --------------------------------------------------------------------------------------
    rels = [abs(e - c) / max(abs(e), 1e-9) for e, c in zip(eager_losses, compiled_losses)]
    finite = all(map(lambda x: x == x and abs(x) != float("inf"), compiled_losses))
    speedup = (eager_s - compiled_s) / eager_s * 100
    print(
        f"\nloss parity over {args.parity_steps} AdamW steps: mean rel {statistics.mean(rels) * 100:.3f}%  max rel {max(rels) * 100:.3f}%"
    )
    print(f"mean loss: eager {statistics.mean(eager_losses):.6f} vs compiled {statistics.mean(compiled_losses):.6f}")
    print(f"speedup: {speedup:+.1f}%  ({eager_s:.3f} -> {compiled_s:.3f} s/step)")
    ok = finite and statistics.mean(rels) < 0.01 and max(rels) < 0.05
    print(f"GATE: {'PASS' if ok else 'FAIL'} (finite losses + mean loss-rel < 1% + max loss-rel < 5%)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
