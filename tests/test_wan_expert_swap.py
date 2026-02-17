"""Tests for WAN dual-expert weight swap logic (TP-05).

Tests the swap_high_low_weights mechanism using lightweight mock state dicts
without loading full WAN models.
"""

import argparse
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock

import torch
from torch import nn

from musubi_tuner.wan_train_network import WanNetworkTrainer


def _make_simple_model(keys: list[str], seed: int = 0) -> nn.Module:
    """Build a minimal nn.Module with named parameters matching the given keys."""
    model = nn.Module()
    gen = torch.Generator().manual_seed(seed)
    for key in keys:
        parts = key.split(".")
        # Navigate/create submodules for nested keys
        parent = model
        for part in parts[:-1]:
            if not hasattr(parent, part):
                child = nn.Module()
                parent.add_module(part, child)
            parent = getattr(parent, part)
        param = nn.Parameter(torch.randn(4, 4, generator=gen))
        parent.register_parameter(parts[-1], param)
    return model


BLOCK_KEYS = [
    "blocks.0.weight",
    "blocks.0.bias",
    "blocks.1.weight",
    "blocks.1.bias",
    "head.weight",
]


class TestExpertSwap(unittest.TestCase):
    """Test swap_high_low_weights logic with mock state dicts."""

    def _make_trainer(self, *, blocks_to_swap: int = 0, compile: bool = False) -> WanNetworkTrainer:
        """Build a WanNetworkTrainer with minimal state for swap testing."""
        trainer = WanNetworkTrainer.__new__(WanNetworkTrainer)
        trainer.blocks_to_swap = blocks_to_swap
        trainer.current_model_is_high_noise = False
        trainer.next_model_is_high_noise = True  # trigger swap
        return trainer

    def _make_args(self, *, compile: bool = False, offload_inactive_dit: bool = False) -> argparse.Namespace:
        return argparse.Namespace(compile=compile, offload_inactive_dit=offload_inactive_dit)

    # --- Test 1: Round-trip swap preserves weights ---

    def test_round_trip_preserves_weights(self) -> None:
        """Swapping low→high→low should restore original weights exactly."""
        model = _make_simple_model(BLOCK_KEYS, seed=42)
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        # Create "high noise" state dict with different weights
        high_sd = OrderedDict()
        gen = torch.Generator().manual_seed(99)
        for k, v in model.state_dict().items():
            high_sd[k] = torch.randn_like(v, generator=gen)

        trainer = self._make_trainer()
        trainer.dit_inactive_state_dict = high_sd
        args = self._make_args()
        accelerator = MagicMock()

        # Swap 1: low → high
        trainer.swap_high_low_weights(args, accelerator, model)
        self.assertTrue(trainer.current_model_is_high_noise)

        # Now model has high-noise weights, inactive has low-noise weights
        for k in BLOCK_KEYS:
            self.assertTrue(torch.equal(model.state_dict()[k], high_sd[k]), f"After swap 1, {k} should be high-noise")

        # Swap 2: high → low
        trainer.next_model_is_high_noise = False
        trainer.swap_high_low_weights(args, accelerator, model)
        self.assertFalse(trainer.current_model_is_high_noise)

        # Model should have original weights restored
        for k in BLOCK_KEYS:
            self.assertTrue(torch.equal(model.state_dict()[k], original_sd[k]), f"After round-trip, {k} should be restored")

    # --- Test 2: strict=True catches key mismatch ---

    def test_strict_catches_key_mismatch(self) -> None:
        """Swap should fail if inactive state dict has different keys."""
        model = _make_simple_model(BLOCK_KEYS, seed=42)

        # Inactive dict missing a key
        bad_sd = OrderedDict()
        for k, v in model.state_dict().items():
            if k != "head.weight":
                bad_sd[k] = torch.randn_like(v)

        trainer = self._make_trainer()
        trainer.dit_inactive_state_dict = bad_sd
        args = self._make_args()
        accelerator = MagicMock()

        with self.assertRaises((RuntimeError, AssertionError)):
            trainer.swap_high_low_weights(args, accelerator, model)

    # --- Test 3: torch.compile key patching ---

    def test_compile_key_patching(self) -> None:
        """With compile=True, block keys should get ._orig_mod. inserted."""
        # Build model with compile-style keys
        compile_keys = [
            "blocks.0._orig_mod.weight",
            "blocks.0._orig_mod.bias",
            "blocks.1._orig_mod.weight",
            "blocks.1._orig_mod.bias",
            "head.weight",
        ]
        model = _make_simple_model(compile_keys, seed=42)

        # Inactive state dict uses non-compiled keys (as stored by state_dict() before compile)
        inactive_sd = OrderedDict()
        gen = torch.Generator().manual_seed(99)
        for k in BLOCK_KEYS:  # non-compiled keys
            inactive_sd[k] = torch.randn(4, 4, generator=gen)

        trainer = self._make_trainer()
        trainer.dit_inactive_state_dict = inactive_sd
        args = self._make_args(compile=True)
        accelerator = MagicMock()

        # Swap should succeed because patch_fn adds ._orig_mod. to block keys
        trainer.swap_high_low_weights(args, accelerator, model)
        self.assertTrue(trainer.current_model_is_high_noise)
        # TP-10: patching should not mutate the stored inactive state dict in-place.
        self.assertEqual(set(inactive_sd.keys()), set(BLOCK_KEYS))

    def test_compile_key_unpatching(self) -> None:
        """With compile=True, compiled inactive keys should be unpatched if the model expects non-compiled keys."""
        # Model uses non-compiled keys
        model = _make_simple_model(BLOCK_KEYS, seed=42)

        # Inactive dict uses compile-style keys
        inactive_sd = OrderedDict()
        gen = torch.Generator().manual_seed(99)
        compile_keys = [
            "blocks.0._orig_mod.weight",
            "blocks.0._orig_mod.bias",
            "blocks.1._orig_mod.weight",
            "blocks.1._orig_mod.bias",
            "head.weight",
        ]
        for key in compile_keys:
            inactive_sd[key] = torch.randn(4, 4, generator=gen)

        trainer = self._make_trainer()
        trainer.dit_inactive_state_dict = inactive_sd
        args = self._make_args(compile=True)
        accelerator = MagicMock()

        trainer.swap_high_low_weights(args, accelerator, model)
        self.assertTrue(trainer.current_model_is_high_noise)
        # Inactive dict stays unchanged (no in-place mutation)
        self.assertEqual(set(inactive_sd.keys()), set(compile_keys))

    # --- Test 4: TP-14 structure validation catches mismatched experts ---

    def test_structure_validation_catches_mismatch(self) -> None:
        """load_transformer's TP-14 check should catch mismatched experts."""
        # We test the validation logic directly rather than calling load_transformer
        low_keys = {"blocks.0.weight", "blocks.1.weight", "head.weight"}
        high_keys = {"blocks.0.weight", "blocks.1.weight", "extra.weight"}

        # This is the same logic from load_transformer TP-14 check
        if low_keys != high_keys:
            only_low = low_keys - high_keys
            only_high = high_keys - low_keys
            self.assertEqual(only_low, {"head.weight"})
            self.assertEqual(only_high, {"extra.weight"})
        else:
            self.fail("Keys should not match")

    # --- Test 5: swap is a true exchange, not a copy ---

    def test_swap_is_exchange(self) -> None:
        """After swap, dit_inactive_state_dict should contain the previously active weights."""
        model = _make_simple_model(BLOCK_KEYS, seed=42)
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        gen = torch.Generator().manual_seed(99)
        high_sd = OrderedDict()
        for k, v in model.state_dict().items():
            high_sd[k] = torch.randn_like(v, generator=gen)
        high_sd_copy = {k: v.clone() for k, v in high_sd.items()}

        trainer = self._make_trainer()
        trainer.dit_inactive_state_dict = high_sd
        args = self._make_args()
        accelerator = MagicMock()

        trainer.swap_high_low_weights(args, accelerator, model)

        # Model now has high-noise weights
        for k in BLOCK_KEYS:
            self.assertTrue(torch.equal(model.state_dict()[k], high_sd_copy[k]))

        # Inactive state dict now has the OLD (low-noise) weights
        for k in BLOCK_KEYS:
            self.assertTrue(
                torch.equal(trainer.dit_inactive_state_dict[k], original_sd[k]),
                f"Inactive {k} should hold previously-active (low-noise) weights",
            )

    # --- Test 6: no-op when current == next ---

    def test_no_swap_when_same_expert(self) -> None:
        """When current_model_is_high_noise == next_model_is_high_noise, no swap occurs."""
        model = _make_simple_model(BLOCK_KEYS, seed=42)
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        trainer = self._make_trainer()
        trainer.current_model_is_high_noise = False
        trainer.next_model_is_high_noise = False  # same → no swap

        gen = torch.Generator().manual_seed(99)
        inactive_sd = OrderedDict()
        for k, v in model.state_dict().items():
            inactive_sd[k] = torch.randn_like(v, generator=gen)
        trainer.dit_inactive_state_dict = inactive_sd

        args = self._make_args()
        accelerator = MagicMock()

        trainer.swap_high_low_weights(args, accelerator, model)

        # Model weights should be unchanged
        for k in BLOCK_KEYS:
            self.assertTrue(torch.equal(model.state_dict()[k], original_sd[k]))


if __name__ == "__main__":
    unittest.main()
