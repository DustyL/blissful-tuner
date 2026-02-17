import unittest
from types import SimpleNamespace

import numpy as np
import torch

from musubi_tuner.flux_2_cache_latents import preprocess_contents_flux_2


def _make_item(*, has_control: bool):
    content = np.zeros((64, 64, 3), dtype=np.uint8)
    if has_control:
        control_content = [np.zeros((64, 64, 3), dtype=np.uint8)]
    else:
        control_content = None
    return SimpleNamespace(content=content, control_content=control_content)


class TestFlux2CacheLatentsAlignment(unittest.TestCase):
    def test_mixed_control_batch_alignment(self):
        """C3: controls list must align to batch indices even when some items have no control images."""
        batch = [_make_item(has_control=False), _make_item(has_control=True)]
        contents, controls = preprocess_contents_flux_2(batch)

        self.assertIsInstance(contents, torch.Tensor)
        self.assertEqual(contents.shape[0], len(batch))

        self.assertIsNotNone(controls)
        assert controls is not None  # for type checkers
        self.assertEqual(len(controls), len(batch))
        self.assertIsNone(controls[0])
        self.assertIsInstance(controls[1], list)
        self.assertIsInstance(controls[1][0], torch.Tensor)

    def test_all_no_control_returns_none(self):
        """C3: when no items have control images, controls must be None (not an all-None list)."""
        batch = [_make_item(has_control=False), _make_item(has_control=False)]
        _, controls = preprocess_contents_flux_2(batch)
        self.assertIsNone(controls)


if __name__ == "__main__":
    unittest.main()
