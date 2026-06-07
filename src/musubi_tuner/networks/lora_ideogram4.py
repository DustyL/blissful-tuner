# LoRA module for Ideogram 4

import ast
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora

logger = logging.getLogger(__name__)

# Targets attention.qkv / attention.o and feed_forward.w1/w2/w3 inside each Ideogram4TransformerBlock.
# These are plain nn.Linear (the fp8 base is loaded via apply_fp8_monkey_patch, which keeps the class as
# "Linear"), so standard LoRA discovery + DoRA work with NO Fp8Linear special-casing. adaln_modulation is
# excluded (training the adaLN projection destabilizes — matches the other image arches' _modulation exclude).
IDEOGRAM4_TARGET_REPLACE_MODULES = ["Ideogram4TransformerBlock"]


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
    exclude_patterns = kwargs.get("exclude_patterns", None)
    exclude_patterns = [] if exclude_patterns is None else ast.literal_eval(exclude_patterns)
    exclude_patterns.append(r".*adaln_modulation.*")
    kwargs["exclude_patterns"] = exclude_patterns

    network = lora.create_network(
        IDEOGRAM4_TARGET_REPLACE_MODULES,
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
    # Fail loudly rather than silently train nothing (e.g. if the block class name or structure changed).
    if len(network.unet_loras) == 0:
        raise RuntimeError(
            "Ideogram 4 LoRA found zero target modules — check Ideogram4TransformerBlock discovery / exclude patterns."
        )
    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        IDEOGRAM4_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
