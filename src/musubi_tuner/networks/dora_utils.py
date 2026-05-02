import torch


def dora_weight_norm_materialized(
    weight: torch.Tensor, lora_weight: torch.Tensor, scaling: float, eps: float = 1e-6
) -> torch.Tensor:
    """Calculate row-wise L2 norm by materializing a LoRA delta."""
    if lora_weight.device != weight.device:
        lora_weight = lora_weight.to(weight.device)
    combined = weight + scaling * lora_weight
    weight_norm = torch.linalg.norm(combined, dim=1).clamp_min(eps)
    return weight_norm.to(weight.dtype)
