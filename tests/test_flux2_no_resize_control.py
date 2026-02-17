from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from musubi_tuner import flux_2_generate_image


class _DummyAE:
    def __init__(self) -> None:
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def to(self, device):  # noqa: ANN001 - mirrors torch API flexibility
        self.device = torch.device(device)
        return self

    def encode(self, x: torch.Tensor):  # noqa: ANN001 - mirrors real API
        # Return shape is irrelevant for this test; only that [0] exists and is a Tensor.
        return (torch.zeros((x.shape[0], 1), dtype=self.dtype, device=self.device),)


def _write_image(path) -> None:  # noqa: ANN001 - pytest tmp_path type is PathLike
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(path)


def test_prepare_image_inputs_no_resize_control_sets_limit_pixels_none(tmp_path):
    image_path = tmp_path / "control.png"
    _write_image(image_path)

    args = SimpleNamespace(image_size=[64, 64], control_image_path=[str(image_path)], no_resize_control=True)
    device = torch.device("cpu")

    captured = {}

    def _fake_default_prep(*, img, limit_pixels, ensure_multiple=16):  # noqa: ANN001
        captured["limit_pixels"] = limit_pixels
        return torch.zeros((3, 64, 64))

    ae = _DummyAE()
    with mock.patch.object(flux_2_generate_image.flux2_utils, "default_prep", side_effect=_fake_default_prep):
        result = flux_2_generate_image.prepare_image_inputs(args, device, ae)

    assert captured["limit_pixels"] is None
    assert result["control_latent"] is not None


def test_prepare_image_inputs_default_resize_capping_keeps_limit_pixels(tmp_path):
    image_path = tmp_path / "control.png"
    _write_image(image_path)

    args = SimpleNamespace(image_size=[64, 64], control_image_path=[str(image_path)], no_resize_control=False)
    device = torch.device("cpu")

    captured = {}

    def _fake_default_prep(*, img, limit_pixels, ensure_multiple=16):  # noqa: ANN001
        captured["limit_pixels"] = limit_pixels
        return torch.zeros((3, 64, 64))

    ae = _DummyAE()
    with mock.patch.object(flux_2_generate_image.flux2_utils, "default_prep", side_effect=_fake_default_prep):
        result = flux_2_generate_image.prepare_image_inputs(args, device, ae)

    assert captured["limit_pixels"] == 2024**2
    assert result["control_latent"] is not None
