"""Per-block CUDA peak memory measurement for FLUX.2 Klein-9B-shaped blocks.

Compares `torch.cuda.max_memory_allocated()` between this branch and main on a
single DoubleStreamBlock + SingleStreamBlock forward+backward at production
shape (Klein-9B, bf16 mixed precision, gradient_checkpointing=True, optional
LoRA adapter wrapping).

Architectural reasoning: trained weights are irrelevant for activation memory
profiling. Random-initialized blocks at the right shape capture the same peak
memory profile as a real Klein-9B checkpoint. This lets the measurement run
without the 8.5 GB safetensors file and without affecting the live training
config.

Usage (run on the current branch first, then `git checkout main` and rerun;
compare the two `BRANCH_RESULT` outputs):

    cd ~/blissful-tuner
    ./venv314/bin/python tests/manual/measure_flux2_block_peak_memory.py \
      --batch 3 --img-seq 4096 --txt-seq 512 --with-lora --iters 3

Decision rule (per the round-2 plan): if per-block peak drops by >=200 MB
between main and this branch, escalate to a short ab-gate with
--blocks_to_swap reduced by 1. If <200 MB, the win is below noise and we
close the door.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from dataclasses import dataclass

import torch

from musubi_tuner.flux_2.flux2_models import DoubleStreamBlock, SingleStreamBlock, rope
from musubi_tuner.modules.attention import AttentionParams


# Klein-9B shape (from Klein9BParams in flux2_models.py)
KLEIN_9B_HIDDEN_SIZE = 4096
KLEIN_9B_NUM_HEADS = 32
KLEIN_9B_HEAD_DIM = KLEIN_9B_HIDDEN_SIZE // KLEIN_9B_NUM_HEADS  # 128
KLEIN_9B_MLP_RATIO = 4.0  # standard transformer MLP ratio


@dataclass
class MeasureResult:
    label: str
    iters: list[int]  # bytes per iter

    @property
    def mb_mean(self) -> float:
        return statistics.mean(self.iters) / (1024 * 1024)

    @property
    def mb_min(self) -> float:
        return min(self.iters) / (1024 * 1024)

    @property
    def mb_stdev(self) -> float:
        return statistics.stdev(self.iters) / (1024 * 1024) if len(self.iters) > 1 else 0.0


def _build_pe(batch: int, seq_len: int, head_dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).expand(batch, -1)
    pe = rope(pos, head_dim, theta=10_000).unsqueeze(1)  # (B, 1, L, D//2, 2, 2)
    return pe


def _wrap_with_lora(module: torch.nn.Module) -> torch.nn.Module:
    """Wrap qkv / proj / linear1 / linear2 / mlp linears in a Klein-9B block with
    tiny LoRA adapters so the measurement includes the LoRA forward path's memory
    cost (which is what production uses)."""
    from musubi_tuner.networks.lora import LoRAModule

    lora_dim, alpha = 16, 16.0
    wrapped = []

    def maybe_wrap(name: str, child: torch.nn.Module):
        if isinstance(child, torch.nn.Linear) and child.in_features > 0:
            lora = LoRAModule(
                lora_name=name.replace(".", "_"),
                org_module=child,
                multiplier=1.0,
                lora_dim=lora_dim,
                alpha=alpha,
                use_dora=True,  # production uses DoRA
            )
            lora.apply_to()
            wrapped.append(lora)

    for name, child in module.named_modules():
        if isinstance(child, torch.nn.Linear):
            maybe_wrap(name, child)

    return wrapped  # return list so they're rooted and not garbage-collected


def measure_double_block(
    batch: int,
    img_seq: int,
    txt_seq: int,
    with_lora: bool,
    iters: int,
) -> MeasureResult:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    block = DoubleStreamBlock(
        hidden_size=KLEIN_9B_HIDDEN_SIZE,
        num_heads=KLEIN_9B_NUM_HEADS,
        mlp_ratio=KLEIN_9B_MLP_RATIO,
    ).to(device=device, dtype=dtype)
    block.enable_gradient_checkpointing()
    block.train()

    loras = _wrap_with_lora(block) if with_lora else []
    for lora in loras:
        lora.to(device=device, dtype=dtype)

    attn_params = AttentionParams.create_attention_params("torch", split_attn=False)

    results: list[int] = []
    for i in range(iters + 1):  # +1 warmup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()

        img = torch.randn(batch, img_seq, KLEIN_9B_HIDDEN_SIZE, device=device, dtype=dtype, requires_grad=True)
        txt = torch.randn(batch, txt_seq, KLEIN_9B_HIDDEN_SIZE, device=device, dtype=dtype, requires_grad=True)
        pe_img = _build_pe(batch, img_seq, KLEIN_9B_HEAD_DIM, device)
        pe_txt = _build_pe(batch, txt_seq, KLEIN_9B_HEAD_DIM, device)

        def mod_triple():
            return tuple(torch.randn(batch, 1, KLEIN_9B_HIDDEN_SIZE, device=device, dtype=dtype) for _ in range(3))

        mod_img = (mod_triple(), mod_triple())
        mod_txt = (mod_triple(), mod_triple())

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            out_img, out_txt = block(img, txt, pe_img, pe_txt, mod_img, mod_txt, attn_params)
            loss = out_img.float().square().mean() + out_txt.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated(device)
        if i > 0:  # skip warmup
            results.append(peak)

    return MeasureResult(label="DoubleStreamBlock (Klein-9B)", iters=results)


def measure_single_block(
    batch: int,
    seq_len: int,
    with_lora: bool,
    iters: int,
) -> MeasureResult:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    block = SingleStreamBlock(
        hidden_size=KLEIN_9B_HIDDEN_SIZE,
        num_heads=KLEIN_9B_NUM_HEADS,
        mlp_ratio=KLEIN_9B_MLP_RATIO,
    ).to(device=device, dtype=dtype)
    block.enable_gradient_checkpointing()
    block.train()

    loras = _wrap_with_lora(block) if with_lora else []
    for lora in loras:
        lora.to(device=device, dtype=dtype)

    attn_params = AttentionParams.create_attention_params("torch", split_attn=False)

    results: list[int] = []
    for i in range(iters + 1):  # +1 warmup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()

        x = torch.randn(batch, seq_len, KLEIN_9B_HIDDEN_SIZE, device=device, dtype=dtype, requires_grad=True)
        pe = _build_pe(batch, seq_len, KLEIN_9B_HEAD_DIM, device)
        mod = tuple(torch.randn(batch, 1, KLEIN_9B_HIDDEN_SIZE, device=device, dtype=dtype) for _ in range(3))

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            out = block(x, pe, mod, attn_params)
            loss = out.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated(device)
        if i > 0:
            results.append(peak)

    return MeasureResult(label="SingleStreamBlock (Klein-9B)", iters=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=3, help="Production batch size (default 3 — Klein-9B v8 config)")
    parser.add_argument("--img-seq", type=int, default=4096, help="Image token sequence length")
    parser.add_argument("--txt-seq", type=int, default=512, help="Text token sequence length")
    parser.add_argument("--with-lora", action="store_true", help="Wrap Linears in DoRA-LoRA (matches production)")
    parser.add_argument("--iters", type=int, default=3, help="Measurement iterations (after one warmup)")
    parser.add_argument("--blocks", choices=["single", "double", "both"], default="both", help="Which block(s) to measure")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    branch = "unknown"
    try:
        import subprocess
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        pass

    print(f"# Branch: {branch}")
    print(f"# Config: batch={args.batch} img_seq={args.img_seq} txt_seq={args.txt_seq} with_lora={args.with_lora}")
    print(f"# Klein-9B shape: hidden={KLEIN_9B_HIDDEN_SIZE} heads={KLEIN_9B_NUM_HEADS} head_dim={KLEIN_9B_HEAD_DIM} mlp_ratio={KLEIN_9B_MLP_RATIO}")
    print()

    results = {}

    if args.blocks in ("double", "both"):
        r = measure_double_block(args.batch, args.img_seq, args.txt_seq, args.with_lora, args.iters)
        results["double"] = {"mb_min": r.mb_min, "mb_mean": r.mb_mean, "mb_stdev": r.mb_stdev, "iters": r.iters}
        print(f"DoubleStreamBlock peak: min={r.mb_min:7.1f} MB  mean={r.mb_mean:7.1f} MB  stdev={r.mb_stdev:.2f} MB")

    if args.blocks in ("single", "both"):
        r = measure_single_block(args.batch, args.img_seq, args.with_lora, args.iters)
        results["single"] = {"mb_min": r.mb_min, "mb_mean": r.mb_mean, "mb_stdev": r.mb_stdev, "iters": r.iters}
        print(f"SingleStreamBlock peak: min={r.mb_min:7.1f} MB  mean={r.mb_mean:7.1f} MB  stdev={r.mb_stdev:.2f} MB")

    if args.json:
        print()
        print("JSON:", json.dumps({"branch": branch, "args": vars(args), "results": results}))


if __name__ == "__main__":
    main()
