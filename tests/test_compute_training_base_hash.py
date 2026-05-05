"""Round-trip test for ss_base_sha256 write/read contract.

Pins the contract that compute_training_base_hash (write side) produces
a hash that _check_lora_base_hash (read side, hotswap) accepts when the
same DiT path is used. Also pins the Option D scope decision: WAN
dual-expert returns None and warns, deferred to follow-up PR.

If a future change moves the read-side hash semantic without updating
the write-side helper (or vice versa), this round-trip suite breaks at
the exact line that names the broken invariant.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import save_file
import torch

from musubi_tuner.utils.lora_utils import (
    _check_lora_base_hash,
    compute_base_hash,
    compute_training_base_hash,
)
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _write_synthetic_dit(path: Path, *, seed: int = 0) -> None:
    """Tiny synthetic DiT-shaped safetensors so the file-content hash has something to hash."""
    torch.manual_seed(seed)
    sd = {"layer.weight": torch.randn(8, 4), "layer.bias": torch.randn(8)}
    save_file(sd, str(path))


class TestComputeTrainingBaseHashSingleDit(unittest.TestCase):
    """Single-DiT trainers (eight LoRA + three full-FT)."""

    def test_returns_compute_base_hash_of_args_dit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "model.safetensors"
            _write_synthetic_dit(dit_path)
            args = _ns(dit=str(dit_path))

            actual = compute_training_base_hash(args)

            expected = compute_base_hash([str(dit_path)])
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), 64)  # SHA256 hex

    def test_dit_high_noise_unset_attribute_falls_back_to_single_dit(self) -> None:
        """Trainers that don't even define args.dit_high_noise (Qwen, Z-Image, etc.) still work."""
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "model.safetensors"
            _write_synthetic_dit(dit_path)
            args = _ns(dit=str(dit_path))  # no dit_high_noise attribute at all

            actual = compute_training_base_hash(args)

            self.assertEqual(actual, compute_base_hash([str(dit_path)]))

    def test_dit_high_noise_none_falls_back_to_single_dit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "model.safetensors"
            _write_synthetic_dit(dit_path)
            args = _ns(dit=str(dit_path), dit_high_noise=None)

            actual = compute_training_base_hash(args)

            self.assertEqual(actual, compute_base_hash([str(dit_path)]))


class TestComputeTrainingBaseHashMissingDit(unittest.TestCase):
    """Trainers with args.dit unset / empty (model-from-config flows)."""

    def test_dit_none_returns_none(self) -> None:
        args = _ns(dit=None)
        self.assertIsNone(compute_training_base_hash(args))

    def test_dit_empty_string_returns_none(self) -> None:
        args = _ns(dit="")
        self.assertIsNone(compute_training_base_hash(args))

    def test_dit_attribute_missing_returns_none(self) -> None:
        """getattr fallback: namespace without `dit` at all."""
        args = _ns()
        self.assertIsNone(compute_training_base_hash(args))


class TestWanDualExpertDeferral(unittest.TestCase):
    """Option D scope: WAN dual-expert returns None — follow-up PR territory.

    Pins the deferral so a future change that adds combined-list hashing
    here without also updating the read side (which expects per-expert
    keys) breaks at this test rather than silently shipping LoRAs that
    can never pass hotswap strict mode.
    """

    def test_dit_high_noise_truthy_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            low = Path(td) / "low.safetensors"
            high = Path(td) / "high.safetensors"
            _write_synthetic_dit(low, seed=1)
            _write_synthetic_dit(high, seed=2)
            args = _ns(dit=str(low), dit_high_noise=str(high))

            self.assertIsNone(compute_training_base_hash(args))

    def test_dit_high_noise_empty_string_treated_same_as_none(self) -> None:
        """Empty-string defense (advisor): == "" must be equivalent to is None.

        wan_generate_video.py:613 already gates on truthy, so we mirror.
        Pins so a future refactor to `is not None` would silently regress
        this case.
        """
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "model.safetensors"
            _write_synthetic_dit(dit_path)
            none_args = _ns(dit=str(dit_path), dit_high_noise=None)
            empty_args = _ns(dit=str(dit_path), dit_high_noise="")

            self.assertEqual(compute_training_base_hash(none_args), compute_training_base_hash(empty_args))


class TestRoundTripWithHotswapReadSide(unittest.TestCase):
    """The load-bearing pin: a LoRA written with the new helper passes
    hotswap strict-mode validation against the same DiT.

    This test is what would have caught the original [low, high] combined-
    hash bug: the trainer-side hash MUST match what the hotswap-side
    state.base_sha256 stores when given the same path.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dit_path = Path(self.tmpdir.name) / "base.safetensors"
        _write_synthetic_dit(self.dit_path)
        self.args = _ns(dit=str(self.dit_path))
        self.hash = compute_training_base_hash(self.args)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_match_passes_strict(self) -> None:
        """Trainer-saved hash + same-base hotswap → silent OK under strict."""
        metadata = {"ss_base_sha256": self.hash}
        # Simulating: hotswap state computed compute_base_hash([dit_path]) at prepare time
        expected = compute_base_hash([str(self.dit_path)])

        # Should not raise
        _check_lora_base_hash(metadata, "synthetic.lora", expected, strict=True)

    def test_mismatch_raises_under_strict(self) -> None:
        metadata = {"ss_base_sha256": self.hash}

        with self.assertRaises(ValueError) as ctx:
            _check_lora_base_hash(metadata, "synthetic.lora", "0" * 64, strict=True)

        self.assertIn("base-hash mismatch", str(ctx.exception))

    def test_mismatch_warns_under_non_strict(self) -> None:
        metadata = {"ss_base_sha256": self.hash}
        # _check_lora_base_hash uses the BlissfulLogger; check no raise + return None
        result = _check_lora_base_hash(metadata, "synthetic.lora", "0" * 64, strict=False)
        self.assertIsNone(result)

    def test_helper_output_byte_identical_to_compute_base_hash_single_path(self) -> None:
        """The contract that closes the loop: compute_training_base_hash(args) MUST
        equal compute_base_hash([args.dit]) when high noise is not set.

        Any drift here breaks every hotswap on every newly-trained LoRA simultaneously.
        """
        from_helper = compute_training_base_hash(self.args)
        from_direct = compute_base_hash([str(self.dit_path)])
        self.assertEqual(from_helper, from_direct)


class TestSafetensorsMetadataRoundTrip(unittest.TestCase):
    """mem_eff_save_file vs save_file: assert both writers preserve __metadata__
    identically. Code-read confirms mem_eff_save_file:53 writes __metadata__ but
    pin the equivalence so a future memory-efficient writer refactor that drops
    metadata can't silently regress full-FT trainers (zimage_train.py:505,
    qwen_image_train.py uses mem_eff_save_file)."""

    def test_both_writers_preserve_ss_base_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dit_path = Path(td) / "src.safetensors"
            _write_synthetic_dit(dit_path)
            base_hash = compute_base_hash([str(dit_path)])

            tensors = {"w": torch.randn(4, 4)}
            metadata = {"ss_base_sha256": base_hash, "ss_other": "preserved"}

            std_out = Path(td) / "via_save_file.safetensors"
            mem_out = Path(td) / "via_mem_eff_save_file.safetensors"
            save_file(tensors, str(std_out), metadata=metadata)
            mem_eff_save_file(tensors, str(mem_out), metadata=metadata)

            from safetensors import safe_open

            for path in (std_out, mem_out):
                with safe_open(str(path), framework="pt") as f:
                    md = f.metadata() or {}
                    self.assertEqual(md.get("ss_base_sha256"), base_hash)
                    self.assertEqual(md.get("ss_other"), "preserved")

    def test_value_normalization_to_str(self) -> None:
        """Both writers normalize non-string metadata values to str. Regression
        guard: if someone passes a bytes hash by mistake, the resulting key
        should still survive through mem_eff_save_file's validate_metadata
        (which normalizes to str via the conversion warning path)."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.safetensors"
            tensors = {"w": torch.randn(2, 2)}
            # mem_eff_save_file accepts non-str values and normalizes
            mem_eff_save_file(tensors, str(out), metadata={"ss_steps": 42})

            from safetensors import safe_open

            with safe_open(str(out), framework="pt") as f:
                md = f.metadata() or {}
                self.assertEqual(md.get("ss_steps"), "42")


if __name__ == "__main__":
    unittest.main()
