import sys
import unittest
from unittest.mock import patch

import torch

from musubi_tuner import (
    flux_2_generate_image,
    flux_kontext_generate_image,
    fpack_generate_video,
    hv_generate_video,
    qwen_image_generate_image,
    zimage_generate_image,
)


class TestLoraMultiplierDefaults(unittest.TestCase):
    def _parse(self, module, argv):
        with patch.object(sys, "argv", argv):
            return module.parse_args()

    def test_lora_multiplier_default_is_none(self):
        """C5: nargs='*' must use default=None (not 1.0) to avoid TypeError when calling len()."""
        cases = [
            (
                flux_2_generate_image,
                ["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"],
            ),
            (
                hv_generate_video,
                [
                    "prog",
                    "--dit",
                    "x",
                    "--vae",
                    "x",
                    "--text_encoder1",
                    "x",
                    "--text_encoder2",
                    "x",
                    "--prompt",
                    "x",
                    "--save_path",
                    "x",
                ],
            ),
            (
                fpack_generate_video,
                [
                    "prog",
                    "--text_encoder1",
                    "x",
                    "--text_encoder2",
                    "x",
                    "--image_encoder",
                    "x",
                    "--save_path",
                    "x",
                    "--prompt",
                    "x",
                ],
            ),
            (
                zimage_generate_image,
                ["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"],
            ),
            (
                flux_kontext_generate_image,
                ["prog", "--text_encoder1", "x", "--text_encoder2", "x", "--save_path", "x", "--prompt", "x"],
            ),
            (
                qwen_image_generate_image,
                ["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"],
            ),
        ]

        for module, argv in cases:
            with self.subTest(module=module.__name__):
                args = self._parse(module, argv)
                self.assertIsNone(args.lora_multiplier)

    def test_qwen_negative_prompt_default_is_string(self):
        args = self._parse(qwen_image_generate_image, ["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertIsInstance(args.negative_prompt, str)
        self.assertNotEqual(args.negative_prompt, "")


class TestFlux2ArgDefaultsAndEnforcement(unittest.TestCase):
    def _parse_flux2(self, argv):
        with patch.object(sys, "argv", argv):
            return flux_2_generate_image.parse_args()

    def test_defaults_filled_from_model_info_dev(self):
        """W8/W18: defaults are filled post-parse from FLUX2_MODEL_INFO for DEV."""
        args = self._parse_flux2(["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertEqual(args.model_version, "dev")
        self.assertEqual(args.infer_steps, 50)
        self.assertEqual(args.embedded_cfg_scale, 4.0)
        self.assertEqual(args.guidance_scale, 4.0)

    def test_klein_distilled_fixed_params_enforced(self):
        """W8/W9: Klein distilled models must enforce fixed steps/guidance regardless of user input."""
        args = self._parse_flux2(
            [
                "prog",
                "--model_version",
                "klein-4b",
                "--text_encoder",
                "x",
                "--save_path",
                "x",
                "--prompt",
                "x",
                "--infer_steps",
                "50",
                "--embedded_cfg_scale",
                "4.0",
                "--guidance_scale",
                "4.0",
            ]
        )
        self.assertEqual(args.model_version, "klein-4b")
        self.assertEqual(args.infer_steps, 4)
        self.assertEqual(args.embedded_cfg_scale, 1.0)
        self.assertEqual(args.guidance_scale, 1.0)

    def test_prompt_file_can_override_embedded_cfg_scale(self):
        """W19: prompt-file mode supports --e shorthand for embedded_cfg_scale."""
        overrides = flux_2_generate_image.parse_prompt_line("A cat --e 3.0")
        self.assertEqual(overrides["prompt"], "A cat")
        self.assertEqual(overrides["embedded_cfg_scale"], 3.0)

    def test_dev_guidance_scale_alias_sets_embedded_cfg_scale(self):
        """Guidance-distilled models use embedded guidance; --guidance_scale should alias to --embedded_cfg_scale."""
        args = self._parse_flux2(
            ["prog", "--model_version", "dev", "--text_encoder", "x", "--save_path", "x", "--prompt", "x", "--guidance_scale", "7"]
        )
        self.assertEqual(args.guidance_scale, 7.0)
        self.assertEqual(args.embedded_cfg_scale, 7.0)

    def test_base_embedded_cfg_scale_alias_sets_guidance_scale(self):
        """Base (non-distilled) models use guidance_scale; --embedded_cfg_scale should alias to guidance_scale."""
        args = self._parse_flux2(
            [
                "prog",
                "--model_version",
                "klein-base-9b",
                "--text_encoder",
                "x",
                "--save_path",
                "x",
                "--prompt",
                "x",
                "--embedded_cfg_scale",
                "6",
            ]
        )
        self.assertEqual(args.guidance_scale, 6.0)
        self.assertEqual(args.embedded_cfg_scale, 6.0)

    def test_dev_warns_on_mismatched_guidance_knobs(self):
        """If both knobs are set and differ, warn which one is used."""
        with self.assertLogs("musubi_tuner.flux_2_generate_image", level="WARNING") as cm:
            args = self._parse_flux2(
                [
                    "prog",
                    "--model_version",
                    "dev",
                    "--text_encoder",
                    "x",
                    "--save_path",
                    "x",
                    "--prompt",
                    "x",
                    "--embedded_cfg_scale",
                    "4",
                    "--guidance_scale",
                    "7",
                ]
            )
        self.assertEqual(args.embedded_cfg_scale, 4.0)
        self.assertEqual(args.guidance_scale, 7.0)
        self.assertTrue(any("ignored for guidance-distilled" in m for m in cm.output), cm.output)

    def test_dev_warns_negative_prompt_ignored(self):
        with self.assertLogs("musubi_tuner.flux_2_generate_image", level="WARNING") as cm:
            self._parse_flux2(
                [
                    "prog",
                    "--model_version",
                    "dev",
                    "--text_encoder",
                    "x",
                    "--save_path",
                    "x",
                    "--prompt",
                    "x",
                    "--negative_prompt",
                    "bad",
                ]
            )
        self.assertTrue(any("negative_prompt is ignored" in m for m in cm.output), cm.output)


class TestZImageCfgArgs(unittest.TestCase):
    def _parse_zimage(self, argv):
        with patch.object(sys, "argv", argv):
            return zimage_generate_image.parse_args()

    def test_cfg_defaults(self):
        args = self._parse_zimage(["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x"])
        self.assertEqual(args.cfg_truncation, zimage_generate_image.zimage_config.DEFAULT_CFG_TRUNCATION)
        self.assertIsNone(args.cfg_normalization)
        self.assertIsNone(args.embedded_cfg_scale)

    def test_cfg_normalization_flag_defaults_to_one(self):
        args = self._parse_zimage(["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x", "--cfg_normalization"])
        self.assertEqual(args.cfg_normalization, 1.0)

    def test_cfg_normalization_value(self):
        args = self._parse_zimage(
            ["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x", "--cfg_normalization", "0.7"]
        )
        self.assertEqual(args.cfg_normalization, 0.7)

    def test_cfg_truncation_range_enforced(self):
        with self.assertRaises(ValueError):
            self._parse_zimage(["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x", "--cfg_truncation", "1.5"])

        with self.assertRaises(ValueError):
            self._parse_zimage(
                ["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x", "--cfg_truncation", "-0.1"]
            )

    def test_cfg_normalization_range_enforced(self):
        with self.assertRaises(ValueError):
            self._parse_zimage(
                ["prog", "--text_encoder", "x", "--save_path", "x", "--prompt", "x", "--cfg_normalization", "0.0"]
            )

    def test_fp8_llm_dtype_resolves_cuda_only(self):
        cpu_args = type("Args", (), {"fp8_llm": True})()
        self.assertEqual(
            zimage_generate_image.resolve_text_encoder_weight_dtype(cpu_args, torch.device("cpu")),
            torch.bfloat16,
        )
        self.assertEqual(
            zimage_generate_image.resolve_text_encoder_weight_dtype(cpu_args, torch.device("cuda")),
            torch.float8_e4m3fn,
        )


if __name__ == "__main__":
    unittest.main()
