# LoRA module for Z-Image

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
# Note: logging.basicConfig removed to avoid conflicts with BlissfulLogger - configure at entry points

import musubi_tuner.networks.lora as lora


ZIMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]


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
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # Always exclude _modulation (adaLN low-rank projections — training these causes instability).
    # By default also exclude _refiner (noise_refiner / context_refiner layers). Override with
    # `include_refiner=True` in `network_args` to target them (matches diffusers' LoRA behaviour).
    include_refiner = kwargs.pop("include_refiner", False)
    if isinstance(include_refiner, str):
        include_refiner = ast.literal_eval(include_refiner)

    if include_refiner:
        exclude_patterns.append(r".*_modulation.*")
        logger.info("Z-Image LoRA: including noise_refiner / context_refiner layers (include_refiner=True)")
    else:
        exclude_patterns.append(r".*(_modulation|_refiner).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        ZIMAGE_TARGET_REPLACE_MODULES,
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
        ZIMAGE_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
