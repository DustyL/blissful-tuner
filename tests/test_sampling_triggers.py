"""Tests for the separated sampling trigger helpers.

Verifies that should_sample_on_step and should_sample_on_epoch produce
unambiguous, non-overlapping decisions — the core fix for duplicate
sample generation when both step-based and epoch-based sampling are
configured simultaneously.
"""

import argparse
import unittest


def _make_args(**overrides):
    """Build a minimal args namespace with sampling-related fields."""
    defaults = {
        "sample_at_first": False,
        "sample_every_n_steps": None,
        "sample_every_n_epochs": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestShouldSampleOnStep(unittest.TestCase):
    def setUp(self):
        from musubi_tuner.hv_train_network import should_sample_on_step

        self.fn = should_sample_on_step

    def test_step_zero_without_sample_at_first(self):
        args = _make_args(sample_every_n_steps=10)
        self.assertFalse(self.fn(args, 0))

    def test_step_zero_with_sample_at_first(self):
        args = _make_args(sample_at_first=True)
        # step=0 + sample_at_first => True
        self.assertTrue(self.fn(args, 0))

    def test_step_matches_interval(self):
        args = _make_args(sample_every_n_steps=50)
        self.assertTrue(self.fn(args, 50))
        self.assertTrue(self.fn(args, 100))

    def test_step_does_not_match_interval(self):
        args = _make_args(sample_every_n_steps=50)
        self.assertFalse(self.fn(args, 25))
        self.assertFalse(self.fn(args, 99))

    def test_step_none_interval_always_false(self):
        args = _make_args(sample_every_n_steps=None)
        self.assertFalse(self.fn(args, 10))
        self.assertFalse(self.fn(args, 100))

    def test_step_ignores_epoch_setting(self):
        """Step trigger must NOT activate based on epoch config."""
        args = _make_args(sample_every_n_steps=None, sample_every_n_epochs=1)
        self.assertFalse(self.fn(args, 10))


class TestShouldSampleOnEpoch(unittest.TestCase):
    def setUp(self):
        from musubi_tuner.hv_train_network import should_sample_on_epoch

        self.fn = should_sample_on_epoch

    def test_epoch_zero_without_sample_at_first(self):
        args = _make_args(sample_every_n_epochs=1)
        self.assertFalse(self.fn(args, 0))

    def test_epoch_zero_with_sample_at_first(self):
        args = _make_args(sample_at_first=True)
        self.assertTrue(self.fn(args, 0))

    def test_epoch_matches_interval(self):
        args = _make_args(sample_every_n_epochs=2)
        self.assertTrue(self.fn(args, 2))
        self.assertTrue(self.fn(args, 4))

    def test_epoch_does_not_match_interval(self):
        args = _make_args(sample_every_n_epochs=2)
        self.assertFalse(self.fn(args, 1))
        self.assertFalse(self.fn(args, 3))

    def test_epoch_none_interval_always_false(self):
        args = _make_args(sample_every_n_epochs=None)
        self.assertFalse(self.fn(args, 1))
        self.assertFalse(self.fn(args, 5))

    def test_epoch_ignores_step_setting(self):
        """Epoch trigger must NOT activate based on step config."""
        args = _make_args(sample_every_n_steps=10, sample_every_n_epochs=None)
        self.assertFalse(self.fn(args, 1))


class TestNoOverlap(unittest.TestCase):
    """When both step and epoch sampling are configured, the two helpers
    should never both return True for the same call — they check
    orthogonal axes."""

    def test_step_trigger_never_considers_epochs(self):
        from musubi_tuner.hv_train_network import should_sample_on_step

        args = _make_args(sample_every_n_steps=10, sample_every_n_epochs=1)
        # At step 10, step trigger fires.
        self.assertTrue(should_sample_on_step(args, 10))
        # But it doesn't take an epoch argument — no epoch-based activation.

    def test_epoch_trigger_never_considers_steps(self):
        from musubi_tuner.hv_train_network import should_sample_on_epoch

        args = _make_args(sample_every_n_steps=10, sample_every_n_epochs=1)
        # Epoch 1 fires.
        self.assertTrue(should_sample_on_epoch(args, 1))
        # But it doesn't take a step argument — no step-based activation.


class TestCompatibilityWrapper(unittest.TestCase):
    """The legacy should_sample_images wrapper still works for callers
    that haven't been updated (e.g. some subclass overrides)."""

    def test_legacy_step_trigger(self):
        from musubi_tuner.hv_train_network import should_sample_images

        args = _make_args(sample_every_n_steps=10)
        self.assertTrue(should_sample_images(args, 10))
        self.assertFalse(should_sample_images(args, 7))

    def test_legacy_epoch_trigger(self):
        from musubi_tuner.hv_train_network import should_sample_images

        args = _make_args(sample_every_n_epochs=2)
        self.assertTrue(should_sample_images(args, 100, epoch=2))
        self.assertFalse(should_sample_images(args, 100, epoch=1))

    def test_legacy_sample_at_first(self):
        from musubi_tuner.hv_train_network import should_sample_images

        args = _make_args(sample_at_first=True)
        self.assertTrue(should_sample_images(args, 0))

        args2 = _make_args(sample_at_first=False)
        self.assertFalse(should_sample_images(args2, 0))


if __name__ == "__main__":
    unittest.main()
