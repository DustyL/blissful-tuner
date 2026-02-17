import tempfile
import unittest

import torch
from safetensors.torch import save_file

from musubi_tuner.merge_lora import _infer_architecture_from_lora_file, _resolve_architecture


class TestMergeLoraArchitectureDetection(unittest.TestCase):
    def _write_lora(self, path: str, keys: list[str], metadata: dict[str, str] | None = None):
        sd = {}
        for k in keys:
            sd[k] = torch.zeros((4, 4), dtype=torch.float32)
        save_file(sd, path, metadata=metadata)

    def test_infer_qwen_image_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/qwen.safetensors"
            self._write_lora(
                path,
                keys=["lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight"],
                metadata={"ss_network_module": "networks.lora_qwen_image"},
            )
            self.assertEqual(_infer_architecture_from_lora_file(path), "qwen_image")

    def test_infer_hv_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/hv.safetensors"
            self._write_lora(
                path,
                keys=["lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"],
                metadata={"ss_network_module": "networks.lora"},
            )
            self.assertEqual(_infer_architecture_from_lora_file(path), "hv")

    def test_infer_qwen_image_from_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/qwen_keys.safetensors"
            self._write_lora(
                path,
                keys=["lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight"],
                metadata=None,
            )
            self.assertEqual(_infer_architecture_from_lora_file(path), "qwen_image")

    def test_resolve_architecture_auto_defaults_to_hv_without_lora(self):
        self.assertEqual(_resolve_architecture("auto", None), "hv")
        self.assertEqual(_resolve_architecture("auto", []), "hv")


if __name__ == "__main__":
    unittest.main()
