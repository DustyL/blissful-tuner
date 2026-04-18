"""Tests for the runtime CuTE preflight probe.

`probe_cute_runtime` runs a minimal CuTE kernel to catch failures where
`CUTE_AVAILABLE == True` but the kernel JIT crashes on the user's actual GPU
(e.g., SM120 with non-TMA-optimized builds). Covers:

- CUTE_AVAILABLE=False shortcut (no CUDA access required)
- non-CUDA device shortcut (no CUDA access required)
- exception capture when kernel raises
- successful probe path
- result caching (probe runs at most once per key)
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from musubi_tuner.modules import attention as attn_mod


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    attn_mod._CUTE_PROBE_CACHE.clear()
    yield
    attn_mod._CUTE_PROBE_CACHE.clear()


def test_probe_rejects_when_cute_unavailable():
    with mock.patch.object(attn_mod, "CUTE_AVAILABLE", False):
        ok, detail = attn_mod.probe_cute_runtime(torch.device("cuda"))
    assert ok is False
    assert "CuTE not importable" in detail


def test_probe_rejects_non_cuda_device():
    with mock.patch.object(attn_mod, "CUTE_AVAILABLE", True):
        ok, detail = attn_mod.probe_cute_runtime(torch.device("cpu"))
    assert ok is False
    assert "CUDA" in detail


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_probe_captures_kernel_exception():
    fake_err = RuntimeError("synthetic kernel compile failure")
    with (
        mock.patch.object(attn_mod, "CUTE_AVAILABLE", True),
        mock.patch.object(attn_mod, "_cute_attention", side_effect=fake_err),
    ):
        ok, detail = attn_mod.probe_cute_runtime(torch.device("cuda"))
    assert ok is False
    assert "RuntimeError" in detail
    assert "synthetic kernel compile failure" in detail


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_probe_reports_success_when_kernel_runs():
    def fake_cute(q, k, v, *, causal=False):
        # Match `_cute_attention`'s (out, lse) return convention.
        return (torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1], device=q.device))

    with (
        mock.patch.object(attn_mod, "CUTE_AVAILABLE", True),
        mock.patch.object(attn_mod, "_cute_attention", side_effect=fake_cute),
    ):
        ok, detail = attn_mod.probe_cute_runtime(torch.device("cuda"), needs_backward=False)
    assert ok is True
    assert detail == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_probe_caches_by_key():
    call_count = {"n": 0}

    def fake_cute(q, k, v, *, causal=False):
        call_count["n"] += 1
        return (torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1], device=q.device))

    with (
        mock.patch.object(attn_mod, "CUTE_AVAILABLE", True),
        mock.patch.object(attn_mod, "_cute_attention", side_effect=fake_cute),
    ):
        ok1, _ = attn_mod.probe_cute_runtime(torch.device("cuda"), needs_backward=False)
        ok2, _ = attn_mod.probe_cute_runtime(torch.device("cuda"), needs_backward=False)
    assert ok1 is True and ok2 is True
    assert call_count["n"] == 1, "second probe call must hit cache, not re-run the kernel"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_probe_distinguishes_backward_from_forward():
    """Forward-only and forward+backward probes cache independently."""
    call_count = {"n": 0}

    def fake_cute(q, k, v, *, causal=False):
        call_count["n"] += 1
        return (torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1], device=q.device))

    with (
        mock.patch.object(attn_mod, "CUTE_AVAILABLE", True),
        mock.patch.object(attn_mod, "_cute_attention", side_effect=fake_cute),
    ):
        attn_mod.probe_cute_runtime(torch.device("cuda"), needs_backward=False)
        attn_mod.probe_cute_runtime(torch.device("cuda"), needs_backward=True)
    assert call_count["n"] == 2, "fwd-only and fwd+bwd probes must be cached separately"
