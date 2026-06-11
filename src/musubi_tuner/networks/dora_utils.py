import torch

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def dequantize_fp8_weight(
    weight: torch.Tensor, scale_weight: torch.Tensor | None, compute_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize a raw fp8 weight to true values using its ``scale_weight`` buffer.

    Mirrors the dequant semantics of ``fp8_optimization_utils.fp8_linear_forward_patch``: scale
    shape ``[1]`` (per-tensor) or ``[out, 1]`` (per-row) broadcasts directly; ``[out, num_blocks, 1]``
    (block-wise) reshapes the weight into blocks along the input dim first.

    Raw fp8 values are NOT true weights — a plain ``.float()`` cast without the scale multiply
    yields quantization-lattice magnitudes (empirically ~200x off for the Ideogram 4 DiT), which
    for DoRA means silently wrong row norms and wrong magnitude scaling. Non-fp8 weights pass
    through unchanged.
    """
    if weight.dtype not in FP8_DTYPES:
        return weight
    if scale_weight is None:
        raise ValueError(
            "fp8 weight has no scale_weight buffer — cannot dequantize. DoRA on an fp8 base requires the "
            "scale_weight installed by apply_fp8_monkey_patch (or the Ideogram 4 pre-quantized shim)."
        )
    w = weight.to(compute_dtype)
    scale = scale_weight.to(device=weight.device, dtype=compute_dtype)
    if scale.ndim < 3:
        return w * scale
    out_features, num_blocks, _ = scale.shape
    return (w.contiguous().view(out_features, num_blocks, -1) * scale).view(weight.shape)


def dora_weight_norm_materialized(
    weight: torch.Tensor,
    lora_weight: torch.Tensor,
    scaling: float,
    eps: float = 1e-6,
    scale_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate row-wise L2 norm by materializing a LoRA delta.

    fp8 base weights are dequantized via ``scale_weight`` first (``weight + scaling * lora_weight``
    would otherwise raise "Promotion for Float8 Types is not supported"). For fp8 inputs the norm
    is returned in the LoRA dtype — returning fp8 would re-poison the downstream
    ``magnitude / weight_norm`` division with the same promotion failure this path avoids.
    """
    is_fp8 = weight.dtype in FP8_DTYPES
    if is_fp8:
        weight = dequantize_fp8_weight(weight, scale_weight)
    if lora_weight.device != weight.device:
        lora_weight = lora_weight.to(weight.device)
    combined = weight + scaling * lora_weight
    weight_norm = torch.linalg.norm(combined, dim=1).clamp_min(eps)
    return weight_norm.to(lora_weight.dtype if is_fp8 else weight.dtype)
