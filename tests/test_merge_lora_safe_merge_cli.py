import sys
import unittest
from unittest.mock import Mock, patch

import torch

from musubi_tuner import merge_lora


class TestMergeLoRASafeMergeCLI(unittest.TestCase):
    def _parse(self, extra_args: list[str]):
        argv = ["merge_lora.py", "--dit", "base.safetensors", "--save_merged_model", "merged.safetensors", *extra_args]
        with patch.object(sys, "argv", argv):
            return merge_lora.parse_args()

    def test_safe_merge_defaults_on(self):
        args = self._parse([])

        self.assertTrue(args.safe_merge)

    def test_boolean_optional_opt_out_disables_safe_merge(self):
        args = self._parse(["--no-safe_merge"])

        self.assertFalse(args.safe_merge)

    def test_underscore_opt_out_alias_disables_safe_merge(self):
        args = self._parse(["--no_safe_merge"])

        self.assertFalse(args.safe_merge)

    def test_standard_lora_path_passes_safe_merge_to_network_merge(self):
        transformer = torch.nn.Linear(1, 1, bias=False)
        weights_sd = {"lora_unet_proj.lora_down.weight": torch.zeros(1, 1)}
        network = Mock()
        network.unet_loras = [object()]

        argv = [
            "merge_lora.py",
            "--dit",
            "base.safetensors",
            "--architecture",
            "hv",
            "--save_merged_model",
            "merged.safetensors",
            "--lora_weight",
            "adapter.safetensors",
            "--device",
            "cpu",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(merge_lora, "load_transformer", return_value=transformer),
            patch.object(merge_lora, "load_file", return_value=weights_sd),
            patch.object(merge_lora, "detect_network_type", return_value="lora"),
            patch.object(merge_lora, "convert_diffusers_if_needed", return_value=weights_sd),
            patch.object(merge_lora.lora, "create_arch_network_from_weights", return_value=network),
            patch.object(merge_lora, "mem_eff_save_file"),
        ):
            merge_lora.main()

        network.merge_to.assert_called_once()
        self.assertTrue(network.merge_to.call_args.kwargs["safe_merge"])

    def test_nonstandard_path_passes_safe_merge_to_direct_merge(self):
        transformer = torch.nn.Linear(1, 1, bias=False)
        weights_sd = {"lora_unet_proj.lokr_w1": torch.zeros(1, 1)}

        argv = [
            "merge_lora.py",
            "--dit",
            "base.safetensors",
            "--architecture",
            "hv",
            "--save_merged_model",
            "merged.safetensors",
            "--lora_weight",
            "adapter.safetensors",
            "--device",
            "cpu",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(merge_lora, "load_transformer", return_value=transformer),
            patch.object(merge_lora, "load_file", return_value=weights_sd),
            patch.object(merge_lora, "detect_network_type", return_value="lokr"),
            patch.object(merge_lora, "convert_diffusers_if_needed", return_value=weights_sd),
            patch.object(merge_lora, "merge_nonlora_to_model", return_value=1) as merge_mock,
            patch.object(merge_lora, "mem_eff_save_file"),
        ):
            merge_lora.main()

        self.assertTrue(merge_mock.call_args.kwargs["safe_merge"])


if __name__ == "__main__":
    unittest.main()
