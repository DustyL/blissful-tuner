"""Tests for require_mask_weights_if_enabled enforcement on every batch, not just step 0."""

from __future__ import annotations

import argparse

import pytest
import torch

from musubi_tuner.modules.mask_loss import require_mask_weights_if_enabled


class TestRequireMaskWeightsEnforcement:
    """Verify require_mask_weights_if_enabled raises on any batch missing masks."""

    def _make_args(self, use_mask_loss: bool = True) -> argparse.Namespace:
        return argparse.Namespace(use_mask_loss=use_mask_loss)

    def test_raises_when_mask_loss_enabled_but_no_masks(self):
        args = self._make_args(use_mask_loss=True)
        batch = {"latents": "dummy"}
        with pytest.raises(ValueError, match="no mask_weights"):
            require_mask_weights_if_enabled(batch, args)

    def test_passes_when_mask_weights_present(self):
        args = self._make_args(use_mask_loss=True)
        batch = {"mask_weights": torch.ones(1, 1, 4, 4)}
        require_mask_weights_if_enabled(batch, args)

    def test_skips_when_mask_loss_disabled(self):
        args = self._make_args(use_mask_loss=False)
        batch = {"latents": "dummy"}
        require_mask_weights_if_enabled(batch, args)

    def test_cache_hint_included_in_error(self):
        args = self._make_args(use_mask_loss=True)
        batch = {}
        with pytest.raises(ValueError, match="custom hint"):
            require_mask_weights_if_enabled(batch, args, cache_hint="custom hint")
