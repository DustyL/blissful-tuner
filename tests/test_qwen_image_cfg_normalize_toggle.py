import sys
import unittest
from unittest.mock import patch

from musubi_tuner import qwen_image_generate_image


class TestQwenImageCfgNormalizeToggle(unittest.TestCase):
    def _parse(self, argv):
        with patch.object(sys, "argv", argv):
            return qwen_image_generate_image.parse_args()

    def test_default_cfg_normalize_enabled_for_non_layered(self):
        args = self._parse(["prog", "--model_version", "original", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertTrue(args.cfg_normalize)

    def test_default_cfg_normalize_disabled_for_layered(self):
        args = self._parse(["prog", "--model_version", "layered", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertFalse(args.cfg_normalize)

    def test_cfg_normalize_can_be_forced_on_for_layered(self):
        args = self._parse(
            ["prog", "--model_version", "layered", "--cfg_normalize", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"]
        )
        self.assertTrue(args.cfg_normalize)

    def test_cfg_normalize_can_be_forced_off_for_non_layered(self):
        args = self._parse(
            ["prog", "--model_version", "original", "--no_cfg_normalize", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"]
        )
        self.assertFalse(args.cfg_normalize)

    def test_prompt_line_overrides_cfg_normalize(self):
        base = self._parse(["prog", "--model_version", "layered", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertFalse(base.cfg_normalize)

        prompt_data = qwen_image_generate_image.parse_prompt_line("A --cfg_normalize")
        prompt_args = qwen_image_generate_image.apply_overrides(base, prompt_data)
        self.assertTrue(prompt_args.cfg_normalize)

        prompt_data = qwen_image_generate_image.parse_prompt_line("A --no_cfg_normalize")
        prompt_args = qwen_image_generate_image.apply_overrides(base, prompt_data)
        self.assertFalse(prompt_args.cfg_normalize)


if __name__ == "__main__":
    unittest.main()

