"""Tests for WAN text encoder output caching warning on empty captions (TC-02)."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from musubi_tuner.dataset.image_video_dataset import ItemInfo


class TestWanTextCacheEmptyCaptionsWarning(unittest.TestCase):
    @patch("musubi_tuner.wan_cache_text_encoder_outputs.save_text_encoder_output_cache_wan")
    @patch("musubi_tuner.wan_cache_text_encoder_outputs.logger.warning")
    def test_empty_caption_triggers_warning(self, mock_warn: MagicMock, _mock_save: MagicMock) -> None:
        from musubi_tuner.wan_cache_text_encoder_outputs import encode_and_save_batch

        text_encoder = MagicMock()
        text_encoder.return_value = [torch.zeros(1), torch.zeros(1)]

        item_a = MagicMock(spec=ItemInfo)
        item_a.caption = ""
        item_a.item_key = "a"
        item_b = MagicMock(spec=ItemInfo)
        item_b.caption = "hello"
        item_b.item_key = "b"

        encode_and_save_batch(text_encoder, [item_a, item_b], device=torch.device("cpu"), accelerator=None)

        self.assertTrue(mock_warn.called)
        self.assertIn("Empty captions", mock_warn.call_args[0][0])

    @patch("musubi_tuner.wan_cache_text_encoder_outputs.save_text_encoder_output_cache_wan")
    @patch("musubi_tuner.wan_cache_text_encoder_outputs.logger.warning")
    def test_nonempty_captions_do_not_warn(self, mock_warn: MagicMock, _mock_save: MagicMock) -> None:
        from musubi_tuner.wan_cache_text_encoder_outputs import encode_and_save_batch

        text_encoder = MagicMock()
        text_encoder.return_value = [torch.zeros(1)]

        item = MagicMock(spec=ItemInfo)
        item.caption = "ok"
        item.item_key = "a"

        encode_and_save_batch(text_encoder, [item], device=torch.device("cpu"), accelerator=None)

        mock_warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
