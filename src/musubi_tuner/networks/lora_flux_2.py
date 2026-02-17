# LoRA module for FLUX.2

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
# Note: logging.basicConfig removed to avoid conflicts with BlissfulLogger - configure at entry points

import musubi_tuner.networks.lora as lora


FLUX_2_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    # Exclude patterns are additive: merge user-supplied patterns with defaults.
    # NOTE: FLUX.2 modulation layers are not inside the LoRA target blocks, so we only exclude norms by default.
    exclude_patterns_arg = kwargs.get("exclude_patterns", None)
    if exclude_patterns_arg is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns_arg, str):
        exclude_patterns = ast.literal_eval(exclude_patterns_arg)
    else:
        exclude_patterns = list(exclude_patterns_arg)

    exclude_patterns.append(r".*(norm).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        FLUX_2_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        FLUX_2_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
