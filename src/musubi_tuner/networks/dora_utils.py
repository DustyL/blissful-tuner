import torch

# All four fp8 variants the repo recognizes elsewhere (flux_utils.is_fp8, model_utils.str_to_dtype):
# the fn/fnuz pairs differ only in their value lattice (e.g. e4m3fnuz max 240 vs e4m3fn 448), which
# matters at QUANTIZATION time — dequant is a dtype-agnostic scale multiply, so all four are safe
# here. getattr guards keep this importable on torch builds that lack the ROCm-targeted fnuz pair.
FP8_DTYPES = tuple(
    dt
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dt := getattr(torch, name, None)) is not None
)


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
    out_features = weight.shape[0]
    # Strict shape contract, anchored to the weight: a raw [out] vector would broadcast over the LAST
    # axis — silently scaling columns instead of rows on square matrices (and erroring on others).
    # The repo's producers normalize to these shapes (fp8_optimization_utils, the Ideogram 4 shim);
    # anything else is a caller bug we refuse rather than mis-scale.
    if scale.numel() == 1 or scale.shape == (out_features, 1):
        return w * scale
    if scale.ndim == 3 and scale.shape[0] == out_features and scale.shape[2] == 1 and weight.shape[1] % scale.shape[1] == 0:
        num_blocks = scale.shape[1]
        return (w.contiguous().view(out_features, num_blocks, -1) * scale).view(weight.shape)
    raise ValueError(
        f"unsupported scale_weight shape {tuple(scale.shape)} for weight shape {tuple(weight.shape)}: expected [1], "
        "[out, 1], or [out, num_blocks, 1] with in_features divisible by num_blocks. A raw [out] vector would "
        "silently scale the wrong axis on square matrices — reshape it to [out, 1]."
    )


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
