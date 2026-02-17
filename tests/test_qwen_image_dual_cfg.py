import sys
import unittest
from unittest.mock import patch

from musubi_tuner import qwen_image_generate_image


class TestQwenImageDualCfg(unittest.TestCase):
    def _parse(self, argv):
        with patch.object(sys, "argv", argv):
            return qwen_image_generate_image.parse_args()

    def test_edit_2509_defaults_use_dual_cfg(self):
        args = self._parse(["prog", "--model_version", "edit-2509", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertEqual(args.guidance_scale, 1.0)
        self.assertEqual(args.true_cfg_scale, 4.0)

    def test_edit_2511_defaults_use_dual_cfg(self):
        args = self._parse(["prog", "--model_version", "edit-2511", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertEqual(args.guidance_scale, 1.0)
        self.assertEqual(args.true_cfg_scale, 4.0)

    def test_original_defaults_true_cfg_follows_guidance_scale(self):
        args = self._parse(["prog", "--model_version", "original", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertEqual(args.guidance_scale, 4.0)
        self.assertEqual(args.true_cfg_scale, 4.0)

    def test_guidance_scale_override_sets_true_cfg_when_true_cfg_not_provided(self):
        args = self._parse(
            ["prog", "--model_version", "edit-2509", "--text_encoder", "x", "--save_path", "x", "--prompt", "x", "--guidance_scale", "7"]
        )
        self.assertEqual(args.guidance_scale, 7.0)
        self.assertEqual(args.true_cfg_scale, 7.0)

    def test_true_cfg_scale_override_keeps_edit_guidance_default(self):
        args = self._parse(
            ["prog", "--model_version", "edit-2509", "--text_encoder", "x", "--save_path", "x", "--prompt", "x", "--true_cfg_scale", "6"]
        )
        self.assertEqual(args.guidance_scale, 1.0)
        self.assertEqual(args.true_cfg_scale, 6.0)

    def test_prompt_line_guidance_scale_updates_true_cfg_scale(self):
        base = self._parse(["prog", "--model_version", "edit-2509", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        prompt_data = qwen_image_generate_image.parse_prompt_line("A --g 9")
        prompt_args = qwen_image_generate_image.apply_overrides(base, prompt_data)
        self.assertEqual(prompt_args.guidance_scale, 9.0)
        self.assertEqual(prompt_args.true_cfg_scale, 9.0)

    def test_prompt_line_true_cfg_scale_override(self):
        base = self._parse(["prog", "--model_version", "edit-2509", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        prompt_data = qwen_image_generate_image.parse_prompt_line("A --tcfg 3.5")
        prompt_args = qwen_image_generate_image.apply_overrides(base, prompt_data)
        self.assertEqual(prompt_args.guidance_scale, 1.0)
        self.assertEqual(prompt_args.true_cfg_scale, 3.5)


if __name__ == "__main__":
    unittest.main()

