#!/usr/bin/env python3
"""
Verify FLUX.2 architecture parameters and LoRA target shapes.

Prints expected weight paths, shapes, and LoRA targeting for each FLUX.2 variant
(dev, klein-9b, klein-4b) WITHOUT loading model weights. Useful for verifying
documentation accuracy against the actual code.

Usage:
    python tools/verify_flux2_architecture.py
    python tools/verify_flux2_architecture.py --variant dev
    python tools/verify_flux2_architecture.py --variant klein-9b --lora-keys
"""

import argparse
import re
import sys
from pathlib import Path

# Add src to path so we can import without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from musubi_tuner.flux_2.flux2_models import (
    AutoEncoderParams,
    Flux2Params,
    Klein4BParams,
    Klein9BParams,
)

# Default LoRA exclude patterns from lora_flux_2.py
LORA_DEFAULT_EXCLUDE = [
    r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*",
    r".*(norm).*",
]

VARIANTS = {
    "dev": Flux2Params(),
    "klein-9b": Klein9BParams(),
    "klein-4b": Klein4BParams(),
}

TEXT_ENCODER_INFO = {
    "dev": {"name": "Mistral3", "hidden": 5120, "layers": 40, "extract": [10, 20, 30]},
    "klein-9b": {"name": "Qwen3-8B", "hidden": 4096, "layers": 36, "extract": [9, 18, 27]},
    "klein-4b": {"name": "Qwen3-4B", "hidden": 2560, "layers": 28, "extract": [9, 18, 27]},
}


def compute_shapes(p: Flux2Params) -> dict:
    """Derive all weight shapes from Flux2Params without instantiation."""
    h = p.hidden_size
    heads = p.num_heads
    head_dim = h // heads
    mlp_hidden = int(h * p.mlp_ratio)
    mlp_mult = 2  # SiLU gating factor (hardcoded in both block types)
    ctx_dim = p.context_in_dim

    # Invariant checks
    assert h % heads == 0, f"hidden_size ({h}) must be divisible by num_heads ({heads})"
    assert sum(p.axes_dim) == head_dim, (
        f"sum(axes_dim) = {sum(p.axes_dim)} != head_dim = {head_dim}; RoPE dims must cover all head dimensions"
    )

    shapes = {}

    # === Top-level embeddings ===
    shapes["img_in.weight"] = (h, p.in_channels)
    shapes["txt_in.weight"] = (h, ctx_dim)
    shapes["time_in.in_layer.weight"] = (h, 256)
    shapes["time_in.out_layer.weight"] = (h, h)
    if p.use_guidance_embed:
        shapes["guidance_in.in_layer.weight"] = (h, 256)
        shapes["guidance_in.out_layer.weight"] = (h, h)

    # === Double-stream modulation (shared, not per-block) ===
    shapes["double_stream_modulation_img.lin.weight"] = (6 * h, h)
    shapes["double_stream_modulation_txt.lin.weight"] = (6 * h, h)
    shapes["single_stream_modulation.lin.weight"] = (3 * h, h)

    # === Double-stream blocks ===
    for i in range(p.depth):
        pfx = f"double_blocks.{i}"
        # Fused QKV (img and txt both project from hidden_size)
        shapes[f"{pfx}.img_attn.qkv.weight"] = (3 * h, h)
        shapes[f"{pfx}.img_attn.proj.weight"] = (h, h)
        shapes[f"{pfx}.img_attn.norm.query_norm.scale"] = (head_dim,)
        shapes[f"{pfx}.img_attn.norm.key_norm.scale"] = (head_dim,)
        # MLP: SiLU gating → first linear is 2× mlp_hidden
        shapes[f"{pfx}.img_mlp.0.weight"] = (mlp_hidden * mlp_mult, h)
        shapes[f"{pfx}.img_mlp.2.weight"] = (h, mlp_hidden)
        # img_norm1, img_norm2: LayerNorm(elementwise_affine=False) → NO parameters
        # txt stream (identical structure)
        shapes[f"{pfx}.txt_attn.qkv.weight"] = (3 * h, h)
        shapes[f"{pfx}.txt_attn.proj.weight"] = (h, h)
        shapes[f"{pfx}.txt_attn.norm.query_norm.scale"] = (head_dim,)
        shapes[f"{pfx}.txt_attn.norm.key_norm.scale"] = (head_dim,)
        shapes[f"{pfx}.txt_mlp.0.weight"] = (mlp_hidden * mlp_mult, h)
        shapes[f"{pfx}.txt_mlp.2.weight"] = (h, mlp_hidden)

    # === Single-stream blocks ===
    for i in range(p.depth_single_blocks):
        pfx = f"single_blocks.{i}"
        # linear1: fused QKV + MLP_in
        qkv_dim = 3 * h
        mlp_in_dim = mlp_hidden * mlp_mult
        shapes[f"{pfx}.linear1.weight"] = (qkv_dim + mlp_in_dim, h)
        # linear2: concat(attn_out, mlp_activated) → hidden
        shapes[f"{pfx}.linear2.weight"] = (h, h + mlp_hidden)
        # QK norm
        shapes[f"{pfx}.norm.query_norm.scale"] = (head_dim,)
        shapes[f"{pfx}.norm.key_norm.scale"] = (head_dim,)
        # pre_norm: LayerNorm(elementwise_affine=False) → NO parameters

    # === Final layer ===
    shapes["final_layer.linear.weight"] = (p.in_channels, h)
    shapes["final_layer.adaLN_modulation.1.weight"] = (2 * h, h)

    return shapes


def classify_lora_target(name: str, exclude_patterns: list[str]) -> str:
    """Check if a weight path would be targeted by LoRA."""
    # LoRA only targets nn.Linear (not RMSNorm .scale params)
    if name.endswith(".scale"):
        return "skip (not Linear)"

    for pat in exclude_patterns:
        if re.match(pat, name):
            return f"excluded ({pat})"

    # Must be inside a target class (DoubleStreamBlock or SingleStreamBlock)
    if name.startswith("double_blocks.") or name.startswith("single_blocks."):
        return "TARGETED"

    return "skip (not in target module)"


def print_variant(variant: str, show_lora: bool, show_shapes: bool):
    """Print architecture info for a variant."""
    p = VARIANTS[variant]
    te = TEXT_ENCODER_INFO[variant]
    shapes = compute_shapes(p)

    h = p.hidden_size
    heads = p.num_heads
    head_dim = h // heads
    mlp_hidden = int(h * p.mlp_ratio)

    print(f"{'=' * 72}")
    print(f"  FLUX.2 {variant.upper()}")
    print(f"{'=' * 72}")

    # Transformer params
    print("\n--- Transformer Params ---")
    print(f"  hidden_size:        {h}")
    print(f"  num_heads:          {heads}")
    print(f"  head_dim:           {head_dim}")
    print(f"  in_channels:        {p.in_channels}")
    print(f"  context_in_dim:     {p.context_in_dim}")
    print(f"  depth (double):     {p.depth}")
    print(f"  depth_single:       {p.depth_single_blocks}")
    print(f"  total blocks:       {p.depth + p.depth_single_blocks}")
    print(f"  mlp_ratio:          {p.mlp_ratio}")
    print(f"  mlp_hidden_dim:     {mlp_hidden}")
    print("  mlp_mult_factor:    2 (SiLU gating)")
    print(f"  axes_dim (RoPE):    {p.axes_dim}")
    print(f"  theta (RoPE):       {p.theta}")
    print(f"  guidance_embed:     {p.use_guidance_embed}")

    # Text encoder
    print("\n--- Text Encoder ---")
    print(f"  model:              {te['name']}")
    print(f"  hidden_size:        {te['hidden']}")
    print(f"  num_layers:         {te['layers']}")
    print(f"  extract_layers:     {te['extract']}")
    print(f"  context_in_dim:     {len(te['extract'])} x {te['hidden']} = {len(te['extract']) * te['hidden']}")
    expected_ctx = len(te["extract"]) * te["hidden"]
    if expected_ctx != p.context_in_dim:
        print(f"  ** MISMATCH: expected {expected_ctx}, params says {p.context_in_dim} **")

    # VAE
    print("\n--- VAE ---")
    print(f"  z_channels:         {AutoEncoderParams.z_channels}")
    print("  patch_size:         [2, 2]")
    print("  encoder stride:     8x (3 downsamples)")
    print("  post-encoder patch: 2x (2x2 rearrange)")
    print("  total stride:       16x")
    print(f"  latent channels:    {AutoEncoderParams.z_channels * 2 * 2} (= {AutoEncoderParams.z_channels} x 2 x 2)")
    print(f"  1024x1024 image ->  {1024 // 16}x{1024 // 16} latent")

    if show_shapes:
        print(f"\n--- Weight Shapes ({len(shapes)} tensors) ---")
        # Group by prefix
        current_group = ""
        for name, shape in shapes.items():
            group = name.split(".")[0]
            if group != current_group:
                current_group = group
                print()
            shape_str = "x".join(str(s) for s in shape)
            params = 1
            for s in shape:
                params *= s
            print(f"  {name:<55} {shape_str:>15}  ({params:>12,} params)")

        # Total params (rough, transformer only)
        total_params = 0
        for shape in shapes.values():
            n = 1
            for s in shape:
                n *= s
            total_params += n
        print(f"\n  Total DiT params (excl. text encoder + VAE): {total_params:,}")

    if show_lora:
        print("\n--- LoRA Targeting ---")
        print("  Target classes:     ['DoubleStreamBlock', 'SingleStreamBlock']")
        print(f"  Exclude patterns:   {LORA_DEFAULT_EXCLUDE}")
        print()

        targeted = []
        excluded = []
        skipped = []

        for name in shapes:
            status = classify_lora_target(name, LORA_DEFAULT_EXCLUDE)
            if status == "TARGETED":
                targeted.append(name)
            elif status.startswith("excluded"):
                excluded.append((name, status))
            else:
                skipped.append((name, status))

        print(f"  TARGETED ({len(targeted)} modules):")
        for name in targeted:
            shape = shapes[name]
            shape_str = "x".join(str(s) for s in shape)
            # LoRA names are derived from the module path (without .weight suffix)
            # See lora.py:800 — named_modules() gives module paths, not param keys
            module_path = name.removesuffix(".weight")
            lora_name = f"lora_unet_{module_path}".replace(".", "_")
            print(f"    {name:<50} {shape_str:>15}")
            print(f"      -> {lora_name}.lora_down.weight")
            print(f"         {lora_name}.lora_up.weight")
            print(f"         {lora_name}.alpha")

        print(f"\n  EXCLUDED by pattern ({len(excluded)} modules):")
        for name, reason in excluded:
            print(f"    {name:<50} {reason}")

        print(f"\n  NOT IN TARGET SCOPE ({len(skipped)} modules):")
        for name, reason in skipped:
            print(f"    {name:<50} {reason}")


def main():
    parser = argparse.ArgumentParser(description="Verify FLUX.2 architecture params and LoRA shapes (no model weights needed)")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()) + ["all"],
        default="all",
        help="Model variant to inspect (default: all)",
    )
    parser.add_argument(
        "--lora-keys",
        action="store_true",
        help="Show LoRA targeting (which modules get LoRA, which are excluded, and safetensors key names)",
    )
    parser.add_argument(
        "--no-shapes",
        action="store_true",
        help="Skip printing individual weight shapes",
    )
    args = parser.parse_args()

    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    for v in variants:
        print_variant(v, show_lora=args.lora_keys, show_shapes=not args.no_shapes)
        print()


if __name__ == "__main__":
    main()
