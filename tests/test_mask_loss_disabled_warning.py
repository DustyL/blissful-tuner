"""Tests for warning when mask sources are configured but mask loss is disabled (DS-01)."""

import argparse
import unittest

from musubi_tuner.dataset.config_utils import (
    Blueprint,
    DatasetBlueprint,
    DatasetGroupBlueprint,
    ImageDatasetParams,
    get_mask_loss_disabled_warning,
)


class TestMaskLossDisabledWarning(unittest.TestCase):
    def test_returns_warning_when_mask_directory_configured(self) -> None:
        params = ImageDatasetParams(mask_directory="/tmp/masks")
        blueprint = Blueprint(
            dataset_group=DatasetGroupBlueprint(datasets=[DatasetBlueprint(is_image_dataset=True, params=params)])
        )

        args = argparse.Namespace(use_mask_loss=False)
        msg = get_mask_loss_disabled_warning(args, blueprint)

        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertIn("mask_directory", msg)
        self.assertIn("--use_mask_loss", msg)

    def test_returns_none_when_mask_loss_enabled(self) -> None:
        params = ImageDatasetParams(mask_directory="/tmp/masks")
        blueprint = Blueprint(
            dataset_group=DatasetGroupBlueprint(datasets=[DatasetBlueprint(is_image_dataset=True, params=params)])
        )

        args = argparse.Namespace(use_mask_loss=True)
        msg = get_mask_loss_disabled_warning(args, blueprint)
        self.assertIsNone(msg)

    def test_returns_none_when_no_mask_sources_configured(self) -> None:
        params = ImageDatasetParams(mask_directory=None, alpha_mask=False, require_mask=False)
        blueprint = Blueprint(
            dataset_group=DatasetGroupBlueprint(datasets=[DatasetBlueprint(is_image_dataset=True, params=params)])
        )

        args = argparse.Namespace(use_mask_loss=False)
        msg = get_mask_loss_disabled_warning(args, blueprint)
        self.assertIsNone(msg)


if __name__ == "__main__":
    unittest.main()
