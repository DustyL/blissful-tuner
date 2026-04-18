"""Regression tests: FLUX.2 CuTE availability preflight checks.

Ensures that both training and generation fail early with a clear error
when --cute / --attn_mode cute is requested but CuTE is not installed or
its kernels cannot execute on the user's GPU, rather than failing deep in
the attention kernels after expensive setup.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

try:
    from musubi_tuner import flux_2_generate_image
    from musubi_tuner.modules import attention as attn_mod
except (OSError, ImportError) as _err:
    pytest.skip(f"FLUX.2 modules not importable: {_err}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    attn_mod._CUTE_PROBE_CACHE.clear()
    yield
    attn_mod._CUTE_PROBE_CACHE.clear()


_FAKE_CAP = (12, 0)  # arbitrary; must match the cap the probe sees under mock


def _preload_probe_cache_ok(needs_backward: bool, cap: tuple[int, int] = _FAKE_CAP) -> None:
    """Preload the probe cache with a success result so positive-path tests
    don't actually execute CuTE kernels (which may fail on some GPU/build
    combinations even when CuTE is 'available')."""
    attn_mod._CUTE_PROBE_CACHE[(cap, needs_backward, torch.bfloat16)] = (True, "")


# ---------------------------------------------------------------------------
# Training preflight now lives in NetworkTrainer.train() (hv_train_network.py),
# upstream of Flux2NetworkTrainer.load_transformer(). It is exercised by the
# base trainer and covered at the probe level in tests/test_cute_probe.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Generation: flux_2_generate_image.parse_args()
# ---------------------------------------------------------------------------


def test_flux2_generation_rejects_cute_when_unavailable():
    """Generation must raise ValueError during arg parsing if CuTE is missing."""
    argv = [
        "flux_2_generate_image.py",
        "--text_encoder", "/fake/te",
        "--save_path", "/fake/out",
        "--attn_mode", "cute",
        "--device", "cuda",
    ]

    with (
        mock.patch("sys.argv", argv),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("musubi_tuner.modules.attention.CUTE_AVAILABLE", False),
    ):
        with pytest.raises(ValueError, match=r"CuTE"):
            flux_2_generate_image.parse_args()


def test_flux2_generation_passes_cute_preflight_when_available():
    """Generation CuTE preflight must not reject when CuTE imports *and* probe passes."""
    argv = [
        "flux_2_generate_image.py",
        "--text_encoder", "/fake/te",
        "--save_path", "/fake/out",
        "--attn_mode", "cute",
        "--device", "cuda",
        "--prompt", "test",
    ]

    _preload_probe_cache_ok(needs_backward=False)
    with (
        mock.patch("sys.argv", argv),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.get_device_capability", return_value=_FAKE_CAP),
        mock.patch("musubi_tuner.modules.attention.CUTE_AVAILABLE", True),
    ):
        args = flux_2_generate_image.parse_args()
        assert args.attn_mode == "cute"
