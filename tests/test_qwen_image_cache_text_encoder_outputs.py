import unittest

import torch

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.qwen_image_cache_text_encoder_outputs import encode_and_save_batch


class TestQwenImageTextCacheEditValidation(unittest.TestCase):
    def test_missing_control_content_raises_in_edit_mode(self):
        item = ItemInfo("item0", "caption", (64, 64))
        item.control_content = None

        with self.assertRaises(ValueError):
            encode_and_save_batch(
                tokenizer=None,  # not used before validation
                text_encoder=None,  # not used before validation
                vl_processor=object(),  # non-None => edit mode
                model_version="edit-2511",
                batch=[item],
                device=torch.device("cpu"),
                accelerator=None,
            )


if __name__ == "__main__":
    unittest.main()
