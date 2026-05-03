import argparse
import unittest

from blissful_tuner.common_extensions import prepare_metadata


class TestPrepareMetadataLoRAMultipliers(unittest.TestCase):
    def test_lora_multiplier_defaults_to_one_when_omitted(self):
        args = argparse.Namespace(
            seed=1234,
            lora_weight=["/private/path/dlay_man_wan22_lora.safetensors"],
            lora_multiplier=None,
        )

        metadata = prepare_metadata(args)

        self.assertEqual(metadata["bt_lora_0"], "dlay_man_wan22_lora.safetensors: 1.0")

    def test_short_lora_multiplier_list_defaults_missing_entries(self):
        args = argparse.Namespace(
            seed=1234,
            task="t2v-A14B",
            lora_weight=["/private/path/low_a.safetensors", "/private/path/low_b.safetensors"],
            lora_multiplier=[0.5],
            lora_weight_high_noise=["/private/path/high_a.safetensors"],
            lora_multiplier_high_noise=None,
        )

        metadata = prepare_metadata(args)

        self.assertEqual(metadata["bt_lora_0"], "low_a.safetensors: 0.5")
        self.assertEqual(metadata["bt_lora_1"], "low_b.safetensors: 1.0")
        self.assertEqual(metadata["bt_lora_high_0"], "high_a.safetensors: 1.0")
        self.assertEqual(metadata["bt_model_type"], "Wan 2.2")


if __name__ == "__main__":
    unittest.main()
