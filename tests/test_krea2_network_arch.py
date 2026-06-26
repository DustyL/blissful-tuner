"""Krea 2 LoHa/LoKr registration in the architecture registry (Phase 2).

K2's plain-LoRA module sets ``KREA2_TARGET_REPLACE_MODULES = None`` (wrap every nn.Linear).
This verifies the registry entry resolves and that ``None`` drives loha/lokr's "all modules"
walker to wrap the Linears, rather than raising "unsupported architecture".
"""

import torch.nn as nn

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_KREA2
from musubi_tuner.networks import loha, lokr
from musubi_tuner.networks.network_arch import get_arch_config


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(8, 8, bias=False)
        self.wk = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Linear(8, 16, bias=False)


def test_krea2_registered_with_none_target():
    config = get_arch_config(ARCHITECTURE_KREA2)
    assert config["target_modules"] is None  # wrap-all-Linear contract (matches lora_krea2)
    assert config["exclude_patterns"] == []


def test_loha_builds_on_krea2_all_linear():
    net = loha.create_arch_network(1.0, 4, 4, None, [], _TinyDiT(), architecture=ARCHITECTURE_KREA2)
    assert len(net.unet_loras) == 3  # all three Linears wrapped via the None ("all modules") walker


def test_lokr_builds_on_krea2_all_linear():
    net = lokr.create_arch_network(1.0, 4, 4, None, [], _TinyDiT(), architecture=ARCHITECTURE_KREA2)
    assert len(net.unet_loras) == 3
