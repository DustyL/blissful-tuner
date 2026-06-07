"""Manual VRAM measurement: Ideogram-4 gradient-checkpointing activation reclaim.

Loads the REAL prequantized-fp8 DiT, applies a LoRA (base frozen, as in training), and runs the actual
flow-matching forward+backward at each resolution with GC toggled — reporting peak `max_memory_allocated`.
This is the datapoint that justifies Slice 1 and sizes Slice 2 (block swap): if GC reclaims little here, stop
and investigate before adding block swap.

This is the **model forward/backward peak** — a LOWER BOUND on the full-trainer peak. It omits optimizer state
and keeps LoRA params in bf16 (the real trainer allocates Adam moments and generally keeps adapters in fp32). A
real cached 1024 trainer smoke (GC + adamw, rank 32, real ~93-token DLAY captions, optimizer state) measured
**~15.3 GB peak / 32 GB** — i.e. ~3 GB above this harness's 1024 number (12.3 GB), but still ~16 GB under budget.
Conclusion: GC ALONE comfortably enables native 1024 (batch=1) on a 32 GB 5090; block swap is optional headroom.

Peak memory is hit in the first backward, so a couple of steps suffice — no long training run needed.

    ./venv314/bin/python tests/manual/measure_ideogram4_gc_memory.py \
        --model-path /home/dustin/Ideogram-4/Huggingface-Model-Weights/transformer/diffusion_pytorch_model.safetensors \
        --resolutions 512 768 1024
"""

import argparse

import torch

import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.ideogram4.ideogram4_utils import load_ideogram4_transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target

IDEOGRAM4_IMAGE_PATCH = 16


def _is_oom(err: RuntimeError) -> bool:
    return "out of memory" in str(err).lower()


def measure(model, network, grid, text_len, gc, device, dtype, steps=2):
    (model.enable_gradient_checkpointing if gc else model.disable_gradient_checkpointing)()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        latents = torch.randn(1, 128, grid, grid, device=device)
        noise = torch.randn_like(latents)
        text_features = [torch.randn(text_len, 53248, device=device, dtype=dtype)]
        t = torch.tensor([0.5], device=device)
        pred, target = ideogram4_flow_matching_target(model, latents, text_features, noise, t, network_dtype=dtype, device=device)
        torch.nn.functional.mse_loss(pred.float(), target.float()).backward()
        network.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--text-len", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required for VRAM measurement"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = load_ideogram4_transformer(args.model_path, dtype=dtype, loading_device=device)
    network = lora_ideogram4.create_arch_network(1.0, args.rank, args.rank * 2.0, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    model.requires_grad_(False)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)  # LoRA params in the model's compute dtype (real training uses autocast)
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"DiT + LoRA resident: {weights_gb:.2f} GB (fp8 weights)\n")

    print(f"{'res':>6} {'grid':>9} {'tokens':>8} {'GC-off':>10} {'GC-on':>10} {'saved':>9} {'saved %':>8}")
    for res in args.resolutions:
        grid = res // IDEOGRAM4_IMAGE_PATCH
        tokens = grid * grid

        def _try(gc):
            try:
                return measure(model, network, grid, args.text_len, gc, device, dtype)
            except RuntimeError as e:
                if _is_oom(e):
                    torch.cuda.empty_cache()
                    return None
                raise

        off, on = _try(False), _try(True)
        off_s = "OOM" if off is None else f"{off:.2f}"
        on_s = "OOM" if on is None else f"{on:.2f}"
        if off is not None and on is not None:
            saved, pct = off - on, 100.0 * (off - on) / off
            print(f"{res:>6} {f'{grid}x{grid}':>9} {tokens:>8} {off_s:>10} {on_s:>10} {saved:>8.2f}G {pct:>7.1f}%")
        else:
            print(f"{res:>6} {f'{grid}x{grid}':>9} {tokens:>8} {off_s:>10} {on_s:>10} {'-':>9} {'-':>8}")


if __name__ == "__main__":
    main()
