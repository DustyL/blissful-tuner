"""Tests for the idempotent FBGEMM_GPU root-logger suppression helper."""

import logging
import unittest


class TestEarlyRuntimeFilters(unittest.TestCase):
    def setUp(self):
        """Reset the module-level guard before each test."""
        import musubi_tuner._early_runtime_filters as mod

        mod._applied = False
        # Remove any _FBGEMMRootFilter instances left on the root logger.
        root = logging.getLogger()
        root.filters = [f for f in root.filters if not isinstance(f, mod._FBGEMMRootFilter)]

    def test_filter_suppresses_fbgemm_messages(self):
        from musubi_tuner._early_runtime_filters import apply_early_runtime_filters

        apply_early_runtime_filters()

        root = logging.getLogger()
        record = logging.LogRecord(
            name="root",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="FBGEMM_GPU version is 1.5.0, which is not guaranteed to be compatible",
            args=(),
            exc_info=None,
        )
        # The filter should reject this record.
        self.assertFalse(root.filter(record))

    def test_filter_passes_unrelated_messages(self):
        from musubi_tuner._early_runtime_filters import apply_early_runtime_filters

        apply_early_runtime_filters()

        root = logging.getLogger()
        record = logging.LogRecord(
            name="root",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Some other warning message",
            args=(),
            exc_info=None,
        )
        self.assertTrue(root.filter(record))

    def test_idempotent_no_duplicate_filters(self):
        from musubi_tuner._early_runtime_filters import _FBGEMMRootFilter, apply_early_runtime_filters

        apply_early_runtime_filters()
        apply_early_runtime_filters()
        apply_early_runtime_filters()

        root = logging.getLogger()
        fbgemm_filters = [f for f in root.filters if isinstance(f, _FBGEMMRootFilter)]
        self.assertEqual(len(fbgemm_filters), 1)


if __name__ == "__main__":
    unittest.main()
