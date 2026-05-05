"""Offline merge PiSSA base-hash preflight tests.

Pins _pissa_base_hash_preflight in src/musubi_tuner/merge_lora.py — the
function that fails fast on PiSSA-tagged LoRAs against a wrong base
BEFORE the multi-GB load_transformer cost. Strict-by-default: no
--no-* flag in v1 (offline merge writes a derived checkpoint, so
artifact-safety failures should be loud).

Synthetic safetensors fixtures so tests run in <1s without touching
real DiT files. The base file is a tiny synthetic .safetensors so
compute_base_hash has bytes to hash; the LoRA file is a tiny synthetic
LoRA-shaped .safetensors with metadata that triggers (or doesn't
trigger) the PiSSA branch.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from musubi_tuner.merge_lora import _pissa_base_hash_preflight
from musubi_tuner.utils.lora_utils import compute_base_hash


def _write_synthetic_dit(path: Path, *, seed: int = 0) -> None:
    torch.manual_seed(seed)
    sd = {"layer.weight": torch.randn(8, 4), "layer.bias": torch.randn(8)}
    save_file(sd, str(path))


def _write_synthetic_lora(path: Path, *, metadata: dict[str, str], seed: int = 1) -> None:
    """LoRA-shaped tensors are not load-bearing here — only the metadata
    matters for the preflight. Write a minimal tensor block so safe_open
    has something to read, plus the metadata header that triggers the
    PiSSA branch (or doesn't)."""
    torch.manual_seed(seed)
    sd = {
        "lora_unet_test.lora_down.weight": torch.randn(2, 4),
        "lora_unet_test.lora_up.weight": torch.randn(8, 2),
    }
    save_file(sd, str(path), metadata=metadata)


def _ns(**kwargs) -> argparse.Namespace:
    """argparse.Namespace with sensible defaults for the preflight."""
    base = dict(dit=None, lora_weight=None)
    base.update(kwargs)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# No-op cases (back-compat — standard merge users see no new behavior)
# ---------------------------------------------------------------------------


class TestPreflightNoOpForNonPissa(unittest.TestCase):
    """Pin: standard-LoRA users see ZERO new behavior from this preflight.
    No metadata read fires for missing args.lora_weight, no hash compute
    fires for non-PiSSA LoRAs. The load-bearing back-compat invariant
    for every existing merge_lora.py invocation."""

    def test_no_lora_weight_arg_is_no_op(self) -> None:
        args = _ns(dit="/nonexistent/dit", lora_weight=None)
        _pissa_base_hash_preflight(args)  # must not raise, must not touch dit

    def test_empty_lora_weight_list_is_no_op(self) -> None:
        args = _ns(dit="/nonexistent/dit", lora_weight=[])
        _pissa_base_hash_preflight(args)

    def test_kaiming_lora_does_not_trigger_hash_compute(self) -> None:
        """The dit path is intentionally invalid — if compute_base_hash
        fired, this test would FileNotFoundError. Passing means the
        non-PiSSA short-circuit worked."""
        with tempfile.TemporaryDirectory() as td:
            lora_path = Path(td) / "kaiming.safetensors"
            _write_synthetic_lora(lora_path, metadata={"ss_init_lora_weights": "kaiming"})
            args = _ns(dit="/nonexistent/dit", lora_weight=[str(lora_path)])
            _pissa_base_hash_preflight(args)

    def test_orthogonal_lora_does_not_trigger_hash_compute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lora_path = Path(td) / "ortho.safetensors"
            _write_synthetic_lora(lora_path, metadata={"ss_init_lora_weights": "orthogonal"})
            args = _ns(dit="/nonexistent/dit", lora_weight=[str(lora_path)])
            _pissa_base_hash_preflight(args)

    def test_lora_with_no_metadata_does_not_trigger_hash_compute(self) -> None:
        """LoRAs predating ss_* metadata land in the back-compat path — no
        ss_init_lora_weights means no PiSSA detection means no hash compute."""
        with tempfile.TemporaryDirectory() as td:
            lora_path = Path(td) / "legacy.safetensors"
            _write_synthetic_lora(lora_path, metadata={})
            args = _ns(dit="/nonexistent/dit", lora_weight=[str(lora_path)])
            _pissa_base_hash_preflight(args)

    def test_lora_with_unrelated_metadata_does_not_trigger_hash_compute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lora_path = Path(td) / "other.safetensors"
            _write_synthetic_lora(lora_path, metadata={"ss_other_key": "value"})
            args = _ns(dit="/nonexistent/dit", lora_weight=[str(lora_path)])
            _pissa_base_hash_preflight(args)


# ---------------------------------------------------------------------------
# PiSSA happy path
# ---------------------------------------------------------------------------


class TestPreflightPissaHappyPath(unittest.TestCase):
    """When the LoRA is PiSSA-tagged AND the base hash matches, preflight
    silently passes — the existing load_transformer flow runs unchanged."""

    def test_matching_hash_passes_silently(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "base.safetensors"
            _write_synthetic_dit(dit_path)
            base_hash = compute_base_hash([str(dit_path)])

            lora_path = Path(td) / "pissa.safetensors"
            _write_synthetic_lora(
                lora_path,
                metadata={"ss_init_lora_weights": "pissa", "ss_base_sha256": base_hash},
            )
            args = _ns(dit=str(dit_path), lora_weight=[str(lora_path)])
            _pissa_base_hash_preflight(args)  # must not raise

    def test_pissa_niter_variant_passes_silently_on_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "base.safetensors"
            _write_synthetic_dit(dit_path)
            base_hash = compute_base_hash([str(dit_path)])

            lora_path = Path(td) / "pissa_niter.safetensors"
            _write_synthetic_lora(
                lora_path,
                metadata={"ss_init_lora_weights": "pissa_niter_8", "ss_base_sha256": base_hash},
            )
            args = _ns(dit=str(dit_path), lora_weight=[str(lora_path)])
            _pissa_base_hash_preflight(args)


# ---------------------------------------------------------------------------
# PiSSA reject paths (the load-bearing safety pins)
# ---------------------------------------------------------------------------


class TestPreflightPissaRejects(unittest.TestCase):
    """The whole reason this preflight exists — fail BEFORE load_transformer."""

    def test_mismatched_base_hash_raises(self) -> None:
        """The defining safety failure: PiSSA LoRA + wrong base = wrong math.
        Must fail before load_transformer, regardless of any flag."""
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "wrong_base.safetensors"
            _write_synthetic_dit(dit_path, seed=99)  # different from training base

            lora_path = Path(td) / "pissa.safetensors"
            _write_synthetic_lora(
                lora_path,
                metadata={
                    "ss_init_lora_weights": "pissa",
                    "ss_base_sha256": "0" * 64,  # deliberately wrong hash
                },
            )
            args = _ns(dit=str(dit_path), lora_weight=[str(lora_path)])
            with self.assertRaisesRegex(ValueError, "PiSSA-trained adapters require an exact base match"):
                _pissa_base_hash_preflight(args)

    def test_missing_base_hash_raises(self) -> None:
        """Strict-by-default for offline merge: no --no-allow_pissa_missing
        flag in v1. Missing hash on a PiSSA-tagged LoRA suggests external
        metadata stripping; refuse to write a derived checkpoint when the
        provenance is broken."""
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "base.safetensors"
            _write_synthetic_dit(dit_path)

            lora_path = Path(td) / "pissa_no_hash.safetensors"
            _write_synthetic_lora(
                lora_path,
                metadata={"ss_init_lora_weights": "pissa"},  # no ss_base_sha256
            )
            args = _ns(dit=str(dit_path), lora_weight=[str(lora_path)])
            with self.assertRaisesRegex(ValueError, "base-hash metadata appears stripped"):
                _pissa_base_hash_preflight(args)

    def test_pissa_niter_mismatch_routes_through_same_reject(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "base.safetensors"
            _write_synthetic_dit(dit_path)

            lora_path = Path(td) / "pissa_niter.safetensors"
            _write_synthetic_lora(
                lora_path,
                metadata={
                    "ss_init_lora_weights": "pissa_niter_5",
                    "ss_base_sha256": "0" * 64,
                },
            )
            args = _ns(dit=str(dit_path), lora_weight=[str(lora_path)])
            with self.assertRaises(ValueError):
                _pissa_base_hash_preflight(args)


# ---------------------------------------------------------------------------
# Multi-LoRA coordination
# ---------------------------------------------------------------------------


class TestPreflightMultiLora(unittest.TestCase):
    """Real merge runs often pass multiple --lora_weight paths. Pin the
    coordination contract: hash computed once, only PiSSA LoRAs checked,
    first failure halts."""

    def test_mixed_pissa_and_standard_only_pissa_checked(self) -> None:
        """Standard LoRA in the mix doesn't trigger a hash compute for itself,
        and the PiSSA LoRA gets validated against the base."""
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "base.safetensors"
            _write_synthetic_dit(dit_path)
            base_hash = compute_base_hash([str(dit_path)])

            kaiming = Path(td) / "kaiming.safetensors"
            _write_synthetic_lora(kaiming, metadata={"ss_init_lora_weights": "kaiming"})
            pissa = Path(td) / "pissa.safetensors"
            _write_synthetic_lora(
                pissa,
                metadata={"ss_init_lora_weights": "pissa", "ss_base_sha256": base_hash},
            )
            args = _ns(dit=str(dit_path), lora_weight=[str(kaiming), str(pissa)])
            _pissa_base_hash_preflight(args)  # both pass

    def test_two_pissa_loras_share_one_base_hash_compute(self) -> None:
        """Coordination invariant: compute_base_hash runs once even when
        N PiSSA LoRAs are validated. Verified by patching compute_base_hash
        to count calls."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "base.safetensors"
            _write_synthetic_dit(dit_path)
            base_hash = compute_base_hash([str(dit_path)])

            pissa_a = Path(td) / "pissa_a.safetensors"
            pissa_b = Path(td) / "pissa_b.safetensors"
            _write_synthetic_lora(
                pissa_a,
                metadata={"ss_init_lora_weights": "pissa", "ss_base_sha256": base_hash},
            )
            _write_synthetic_lora(
                pissa_b,
                metadata={"ss_init_lora_weights": "pissa_niter_5", "ss_base_sha256": base_hash},
            )
            args = _ns(dit=str(dit_path), lora_weight=[str(pissa_a), str(pissa_b)])

            # Patch compute_base_hash inside merge_lora's namespace and count calls
            with patch("musubi_tuner.merge_lora.compute_base_hash", wraps=compute_base_hash) as mock_hash:
                _pissa_base_hash_preflight(args)
                self.assertEqual(mock_hash.call_count, 1, "base hash should be computed exactly once across N PiSSA LoRAs")

    def test_first_pissa_failure_halts_subsequent_check(self) -> None:
        """If the first PiSSA LoRA mismatches, the function raises before
        even checking the second. Pin so a future refactor can't silently
        accumulate errors and write a half-validated checkpoint."""
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "base.safetensors"
            _write_synthetic_dit(dit_path)
            base_hash = compute_base_hash([str(dit_path)])

            bad = Path(td) / "bad_pissa.safetensors"
            good = Path(td) / "good_pissa.safetensors"
            _write_synthetic_lora(bad, metadata={"ss_init_lora_weights": "pissa", "ss_base_sha256": "0" * 64})
            _write_synthetic_lora(good, metadata={"ss_init_lora_weights": "pissa", "ss_base_sha256": base_hash})
            args = _ns(dit=str(dit_path), lora_weight=[str(bad), str(good)])
            with self.assertRaises(ValueError):
                _pissa_base_hash_preflight(args)


if __name__ == "__main__":
    unittest.main()
