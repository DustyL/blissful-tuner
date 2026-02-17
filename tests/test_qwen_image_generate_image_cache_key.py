import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from musubi_tuner import qwen_image_generate_image


class _DummyTextEncoder:
    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self


class TestQwenImageEmbedsCacheKey(unittest.TestCase):
    def _parse(self, argv):
        with patch.object(sys, "argv", argv):
            return qwen_image_generate_image.parse_args()

    def test_cache_key_separates_model_versions(self):
        shared_models = {
            "tokenizer": object(),
            "text_encoder": _DummyTextEncoder(),
            "vl_processor": object(),
            "conds_cache": {},
        }
        images = [np.zeros((32, 32, 3), dtype=np.uint8)]

        def fake_get_embeds_with_image(vl_processor, text_encoder, prompt, ims, model_version=None):
            value = 1 if model_version == "edit" else 2
            embed = torch.tensor([[value]], dtype=torch.float32)
            mask = torch.tensor([[1]], dtype=torch.float32)
            return embed, mask

        with patch.object(
            qwen_image_generate_image.qwen_image_utils,
            "get_qwen_prompt_embeds_with_image",
            side_effect=fake_get_embeds_with_image,
        ) as mocked:
            args_edit = self._parse(["prog", "--model_version", "edit", "--text_encoder", "x", "--save_path", "x", "--prompt", "A"])
            args_edit.control_image_path = ["a.png"]
            qwen_image_generate_image.prepare_text_inputs(
                args_edit, images, device=torch.device("cpu"), shared_models=shared_models
            )
            self.assertEqual(len(shared_models["conds_cache"]), 2)

            args_2511 = self._parse(
                ["prog", "--model_version", "edit-2511", "--text_encoder", "x", "--save_path", "x", "--prompt", "A"]
            )
            args_2511.control_image_path = ["a.png"]
            qwen_image_generate_image.prepare_text_inputs(
                args_2511, images, device=torch.device("cpu"), shared_models=shared_models
            )
            self.assertEqual(len(shared_models["conds_cache"]), 4)
            self.assertEqual(mocked.call_count, 4)

    def test_cache_key_includes_resize_flags(self):
        shared_models = {
            "tokenizer": object(),
            "text_encoder": _DummyTextEncoder(),
            "vl_processor": object(),
            "conds_cache": {},
        }
        images = [np.zeros((32, 32, 3), dtype=np.uint8)]

        def fake_get_embeds_with_image(vl_processor, text_encoder, prompt, ims, model_version=None):
            embed = torch.randn(1, 1)
            mask = torch.ones(1, 1)
            return embed, mask

        with patch.object(
            qwen_image_generate_image.qwen_image_utils,
            "get_qwen_prompt_embeds_with_image",
            side_effect=fake_get_embeds_with_image,
        ) as mocked:
            args_default = self._parse(
                ["prog", "--model_version", "edit", "--text_encoder", "x", "--save_path", "x", "--prompt", "A"]
            )
            args_default.control_image_path = ["a.png"]
            qwen_image_generate_image.prepare_text_inputs(
                args_default, images, device=torch.device("cpu"), shared_models=shared_models
            )
            self.assertEqual(len(shared_models["conds_cache"]), 2)

            args_resize = self._parse(
                [
                    "prog",
                    "--model_version",
                    "edit",
                    "--text_encoder",
                    "x",
                    "--save_path",
                    "x",
                    "--prompt",
                    "A",
                    "--resize_control_to_image_size",
                ]
            )
            args_resize.control_image_path = ["a.png"]
            qwen_image_generate_image.prepare_text_inputs(
                args_resize, images, device=torch.device("cpu"), shared_models=shared_models
            )
            self.assertEqual(len(shared_models["conds_cache"]), 4)
            self.assertEqual(mocked.call_count, 4)


if __name__ == "__main__":
    unittest.main()
