"""Guards and reporting in the LyCORIS bridge (``networks/lycoris.py``), from the 2026-09-13 LyCORIS 4.0 review.

Two silent-failure modes in LyCORIS itself that the bridge must refuse rather than pass through:

1. fp8 base + rebuild path. LyCORIS recognises weight-only fp8 only by class name ``Fp8Linear`` with a
   ``weight_scale`` attribute; blissful's fp8-scaled Linears stay ``nn.Linear`` with ``scale_weight``.
   ``_current_weight()`` then returns the raw unscaled fp8 tensor and ``_rebuild_forward`` does
   ``base_weight + diff_weight.to(fp8)`` — fails or merges an unscaled weight. Only ``bypass_mode``
   (``org_forward(x) + diff(x)``) is representation-agnostic.
2. ``bypass_mode`` + ``dora_wd``. ``bypass_forward`` has no DoRA branch in LoKr/LoHa/LoCon and
   ``lycoris.kohya.create_network`` has no guard, so DoRA is silently dropped.

Plus: the run log must record which LyCORIS build and kernel backend produced the arithmetic, since
4.0 can run the same config through eager, Inductor or Triton depending on wrapping.
"""

import pytest
import torch
import torch.nn as nn

lycoris = pytest.importorskip("lycoris")

from musubi_tuner.networks.lycoris import create_network, describe_lycoris_runtime  # noqa: E402


class Block(nn.Module):
    def __init__(self, fp8: bool):
        super().__init__()
        self.proj = nn.Linear(64, 64)
        if fp8:
            self.proj.weight = nn.Parameter(self.proj.weight.to(torch.float8_e4m3fn), requires_grad=False)
            self.proj.register_buffer("scale_weight", torch.ones(1))  # blissful's fp8-scaled layout


class Model(nn.Module):
    def __init__(self, fp8: bool):
        super().__init__()
        self.blocks = nn.ModuleList([Block(fp8)])


def _create(model, **network_args):
    return create_network(1.0, 4, 1.0, None, None, model, extra_unet_targets=["Block"], algo="lokr", **network_args)


def test_fp8_base_without_bypass_mode_is_rejected():
    with pytest.raises(ValueError, match="bypass_mode"):
        _create(Model(fp8=True))


def test_fp8_base_with_bypass_mode_and_dora_is_rejected():
    with pytest.raises(ValueError, match="dora_wd"):
        _create(Model(fp8=True), bypass_mode="True", dora_wd="True")


def test_bypass_mode_with_dora_is_rejected_on_any_base():
    with pytest.raises(ValueError, match="dora_wd"):
        _create(Model(fp8=False), bypass_mode="True", dora_wd="True")


def test_fp8_base_with_bypass_mode_creates_network():
    net = _create(Model(fp8=True), bypass_mode="True")
    assert len(net.unet_loras) == 1
    assert net.unet_loras[0].bypass_mode is True


def test_dora_without_bypass_on_bf16_base_is_still_allowed():
    net = _create(Model(fp8=False), dora_wd="True")
    assert len(net.unet_loras) == 1


def test_describe_lycoris_runtime_names_version_and_backend():
    from importlib.metadata import version

    from lycoris.kernels import resolve_backend

    text = describe_lycoris_runtime()
    assert version("lycoris_lora") in text
    assert resolve_backend() in text
