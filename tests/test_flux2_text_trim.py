"""Tests for FLUX.2 text sequence trimming (Phase 1).

Covers:
- compute_trimmed_text_len() pure math
- maybe_trim_ctx_vec() trim + fallback logic
- Trim-through-prc_txt integration (no GPU)
- Cache round-trip with ctx_seq_len
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

try:
    from musubi_tuner.flux_2 import flux2_utils
    from musubi_tuner.flux_2.text_trim import compute_trimmed_text_len, maybe_trim_ctx_vec
except (OSError, ImportError) as _err:
    pytest.skip(f"FLUX.2 modules not importable: {_err}", allow_module_level=True)


# ---------------------------------------------------------------------------
# compute_trimmed_text_len unit tests
# ---------------------------------------------------------------------------


def test_trim_to_exact_length():
    """ctx_seq_len=[64], pad_multiple=1 → trimmed_len=64."""
    assert compute_trimmed_text_len(512, torch.tensor([64], dtype=torch.int32), 1) == 64


def test_trim_with_rounding():
    """ctx_seq_len=[65], pad_multiple=32 → trimmed_len=96."""
    assert compute_trimmed_text_len(512, torch.tensor([65], dtype=torch.int32), 32) == 96


def test_trim_no_op_when_full():
    """ctx_seq_len=[512] → trimmed_len=512."""
    assert compute_trimmed_text_len(512, torch.tensor([512], dtype=torch.int32), 1) == 512


def test_trim_batch_uses_max():
    """ctx_seq_len=[64, 128], pad_multiple=1 → trimmed_len=128 (batch max)."""
    assert compute_trimmed_text_len(512, torch.tensor([64, 128], dtype=torch.int32), 1) == 128


def test_trim_clamps_to_full_len():
    """ctx_seq_len=[500], pad_multiple=32 → 512 (clamped to full_len)."""
    assert compute_trimmed_text_len(512, torch.tensor([500], dtype=torch.int32), 32) == 512


def test_trim_rounding_16_vs_32():
    """ctx_seq_len=[65], pad_multiple=16 → 80; pad_multiple=32 → 96."""
    seq = torch.tensor([65], dtype=torch.int32)
    assert compute_trimmed_text_len(512, seq, 16) == 80
    assert compute_trimmed_text_len(512, seq, 32) == 96


# ---------------------------------------------------------------------------
# maybe_trim_ctx_vec tests (exercises the real trainer trim/fallback logic)
# ---------------------------------------------------------------------------


def test_maybe_trim_trims_with_ctx_seq_len():
    """Trim mode with ctx_seq_len present trims ctx_vec."""
    ctx_vec = torch.randn(1, 512, 128)
    batch = {"ctx_seq_len": torch.tensor([64], dtype=torch.int32)}
    args = SimpleNamespace(flux2_text_seqlen_mode="trim", compile=False, pad_text_seq_len_multiple=None)

    result = maybe_trim_ctx_vec(ctx_vec, batch, args, {})
    assert result.shape == (1, 64, 128)


def test_maybe_trim_rounds_with_compile():
    """Trim mode with --compile defaults to pad_multiple=32."""
    ctx_vec = torch.randn(1, 512, 128)
    batch = {"ctx_seq_len": torch.tensor([65], dtype=torch.int32)}
    args = SimpleNamespace(flux2_text_seqlen_mode="trim", compile=True, pad_text_seq_len_multiple=None)

    result = maybe_trim_ctx_vec(ctx_vec, batch, args, {})
    assert result.shape == (1, 96, 128)


def test_maybe_trim_noop_in_fixed_mode():
    """Fixed mode never trims, even with ctx_seq_len present."""
    ctx_vec = torch.randn(1, 512, 128)
    batch = {"ctx_seq_len": torch.tensor([64], dtype=torch.int32)}
    args = SimpleNamespace(flux2_text_seqlen_mode="fixed", compile=False, pad_text_seq_len_multiple=None)

    result = maybe_trim_ctx_vec(ctx_vec, batch, args, {})
    assert result.shape == (1, 512, 128)


def test_maybe_trim_fallback_warns_once(caplog):
    """Trim mode without ctx_seq_len warns once then stays silent."""
    ctx_vec = torch.randn(1, 512, 128)
    batch = {}  # no ctx_seq_len
    args = SimpleNamespace(flux2_text_seqlen_mode="trim", compile=False, pad_text_seq_len_multiple=None)
    warned_state = {}

    with caplog.at_level(logging.WARNING, logger="musubi_tuner.flux_2_train_network"):
        result1 = maybe_trim_ctx_vec(ctx_vec, batch, args, warned_state)

    # No trim occurred
    assert result1.shape == (1, 512, 128)
    # Warning was logged
    assert "ctx_seq_len" in caplog.text
    assert warned_state["fallback_warned"] is True

    # Second call should NOT warn again
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="musubi_tuner.flux_2_train_network"):
        result2 = maybe_trim_ctx_vec(ctx_vec, batch, args, warned_state)
    assert result2.shape == (1, 512, 128)
    assert caplog.text == ""


def test_maybe_trim_explicit_pad_multiple():
    """--pad_text_seq_len_multiple overrides the compile-based default."""
    ctx_vec = torch.randn(1, 512, 128)
    batch = {"ctx_seq_len": torch.tensor([65], dtype=torch.int32)}
    args = SimpleNamespace(flux2_text_seqlen_mode="trim", compile=True, pad_text_seq_len_multiple=16)

    result = maybe_trim_ctx_vec(ctx_vec, batch, args, {})
    # 16 wins over the compile default of 32: ceil(65/16)*16 = 80
    assert result.shape == (1, 80, 128)


def test_maybe_trim_skips_mixed_batch(caplog):
    """Mixed-schema batch (some items lack ctx_seq_len) must skip trim and warn once.

    Simulates what BucketBatchManager produces when item 0 is re-cached with
    ctx_seq_len_int32 but item 1 is a legacy cache: ctx_vec has batch dim 2,
    but ctx_seq_len has batch dim 1. Trimming to the partial max would
    silently truncate real tokens from item 1.
    """
    ctx_vec = torch.randn(2, 512, 128)  # full batch
    batch = {"ctx_seq_len": torch.tensor([64], dtype=torch.int32)}  # only 1 item annotated
    args = SimpleNamespace(flux2_text_seqlen_mode="trim", compile=False, pad_text_seq_len_multiple=None)
    warned_state = {}

    with caplog.at_level(logging.WARNING, logger="musubi_tuner.flux_2_train_network"):
        result = maybe_trim_ctx_vec(ctx_vec, batch, args, warned_state)

    # Full shape preserved — no trimming happened.
    assert result.shape == (2, 512, 128)
    # Warning fired and names the ratio.
    assert "1/2" in caplog.text
    assert "mixed cache schemas" in caplog.text
    assert warned_state["partial_cache_warned"] is True

    # Second call must NOT re-warn.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="musubi_tuner.flux_2_train_network"):
        result2 = maybe_trim_ctx_vec(ctx_vec, batch, args, warned_state)
    assert result2.shape == (2, 512, 128)
    assert caplog.text == ""


def test_maybe_trim_happy_path_multi_item_batch():
    """Well-formed batch with ctx_seq_len covering every item still trims correctly."""
    ctx_vec = torch.randn(2, 512, 128)
    batch = {"ctx_seq_len": torch.tensor([40, 64], dtype=torch.int32)}
    args = SimpleNamespace(flux2_text_seqlen_mode="trim", compile=False, pad_text_seq_len_multiple=None)

    result = maybe_trim_ctx_vec(ctx_vec, batch, args, {})
    # Batch max is 64, pad_multiple=1 → trim to 64.
    assert result.shape == (2, 64, 128)


# ---------------------------------------------------------------------------
# Integration: trim through prc_txt
# ---------------------------------------------------------------------------


def test_trim_through_prc_txt():
    """Trim ctx_vec then run batched_prc_txt — shapes and position IDs must be consistent."""
    B, full_len, D = 1, 512, 128
    ctx_vec = torch.randn(B, full_len, D)
    ctx_seq_len = torch.tensor([64], dtype=torch.int32)

    trimmed_len = compute_trimmed_text_len(full_len, ctx_seq_len, pad_multiple=1)
    ctx_vec_trimmed = ctx_vec[:, :trimmed_len, :]

    ctx, ctx_ids = flux2_utils.batched_prc_txt(ctx_vec_trimmed)

    assert ctx.shape == (B, trimmed_len, D)
    assert ctx_ids.shape == (B, trimmed_len, 4)

    # l-axis position IDs should be arange(trimmed_len)
    l_positions = ctx_ids[0, :, 3]  # 4th column is the l axis
    expected = torch.arange(trimmed_len, device=l_positions.device)
    assert torch.equal(l_positions, expected)


def test_trim_through_prc_txt_with_rounding():
    """Trimmed + rounded length flows correctly through prc_txt."""
    B, full_len, D = 1, 512, 128
    ctx_vec = torch.randn(B, full_len, D)
    ctx_seq_len = torch.tensor([65], dtype=torch.int32)

    trimmed_len = compute_trimmed_text_len(full_len, ctx_seq_len, pad_multiple=32)
    assert trimmed_len == 96

    ctx_vec_trimmed = ctx_vec[:, :trimmed_len, :]
    ctx, ctx_ids = flux2_utils.batched_prc_txt(ctx_vec_trimmed)

    assert ctx.shape == (B, 96, D)
    assert ctx_ids.shape == (B, 96, 4)
