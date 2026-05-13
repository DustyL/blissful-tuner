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


class TestPrepareMetadataKandinsky5Detection(unittest.TestCase):
    """Lock down the K5 task -> bt_model_type dispatch added alongside the prefix refactor.

    The reviewer's concern with the original substring check was that future K5 task
    names containing `k5-pro`/`k5-lite` *anywhere* in the string would match. Using
    str.startswith() encodes the invariant tightly, and these tests assert both the
    direct cases and totality over the current TASK_CONFIGS registry.
    """

    def test_k5_pro_task_classified_as_kandinsky_5_pro(self):
        args = argparse.Namespace(seed=42, task="k5-pro-t2v-5s-sd")
        metadata = prepare_metadata(args)
        self.assertEqual(metadata["bt_model_type"], "Kandinsky 5 Pro")

    def test_k5_lite_task_classified_as_kandinsky_5_lite(self):
        args = argparse.Namespace(seed=42, task="k5-lite-t2v-5s-distil-sd")
        metadata = prepare_metadata(args)
        self.assertEqual(metadata["bt_model_type"], "Kandinsky 5 Lite")

    def test_all_kandinsky5_task_configs_classified(self):
        """Totality check: every TASK_CONFIGS key must resolve to a K5 model type,
        not the Wan 2.1 fallback. Regresses if a new K5 task name slips a prefix."""
        from musubi_tuner.kandinsky5.configs import TASK_CONFIGS

        for task_name in TASK_CONFIGS:
            with self.subTest(task=task_name):
                args = argparse.Namespace(seed=42, task=task_name)
                metadata = prepare_metadata(args)
                self.assertIn(
                    metadata["bt_model_type"],
                    ("Kandinsky 5 Pro", "Kandinsky 5 Lite"),
                    f"task {task_name!r} classified as {metadata['bt_model_type']!r}, "
                    "expected 'Kandinsky 5 Pro' or 'Kandinsky 5 Lite'",
                )


if __name__ == "__main__":
    unittest.main()
