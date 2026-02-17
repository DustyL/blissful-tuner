# LoRA module for FLUX.1 Kontext

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import musubi_tuner.networks.lora as lora
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")

FLUX_KONTEXT_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]


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
    exclude_patterns = [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*"]

    exclude_patterns_arg = kwargs.get("exclude_patterns", None)
    if exclude_patterns_arg is None:
        pass
    elif isinstance(exclude_patterns_arg, str):
        exclude_patterns.extend(ast.literal_eval(exclude_patterns_arg))
    else:
        exclude_patterns.extend(list(exclude_patterns_arg))

    exclude_patterns.append(r".*(norm).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        FLUX_KONTEXT_TARGET_REPLACE_MODULES,
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
        FLUX_KONTEXT_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
