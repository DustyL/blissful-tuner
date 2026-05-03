"""Regression tests for orthogonal LoRA initialization."""

import tempfile
import unittest

import torch
from safetensors import safe_open

from musubi_tuner.networks.lora import LoRAModule, _init_orthogonal_lora_pair, create_network, parse_init_lora_weights_arg


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias=False)
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=(1, 1), bias=False)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ToyBlock()


class NotLoRAModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class TestOrthogonalInitMath(unittest.TestCase):
    def test_ba_is_zero_at_init(self):
        torch.manual_seed(42)
        base = torch.nn.Linear(64, 32, bias=False)

        lora = LoRAModule("test", base, multiplier=1.0, lora_dim=8, alpha=4.0, init_lora_weights="orthogonal")

        product = lora.lora_up.weight @ lora.lora_down.weight
        self.assertTrue(
            torch.allclose(product, torch.zeros_like(product), atol=1e-5),
            f"B @ A should be zero at orthogonal init; max abs = {product.abs().max().item()}",
        )

    def test_both_matrices_are_nonzero(self):
        torch.manual_seed(42)
        base = torch.nn.Linear(64, 32, bias=False)

        lora = LoRAModule("test", base, multiplier=1.0, lora_dim=8, alpha=4.0, init_lora_weights="orthogonal")

        self.assertGreater(lora.lora_down.weight.abs().mean().item(), 1e-4)
        self.assertGreater(lora.lora_up.weight.abs().mean().item(), 1e-4)

    def test_odd_rank_raises_actionable_error(self):
        base = torch.nn.Linear(64, 32, bias=False)

        with self.assertRaisesRegex(ValueError, r"even rank, got 5.*network_dim.*kaiming"):
            LoRAModule("test", base, multiplier=1.0, lora_dim=5, alpha=4.0, init_lora_weights="orthogonal")

    def test_helper_preserves_module_dtype(self):
        down = torch.nn.Linear(64, 8, bias=False).to(dtype=torch.bfloat16)
        up = torch.nn.Linear(8, 32, bias=False).to(dtype=torch.bfloat16)

        _init_orthogonal_lora_pair(down, up, in_features=64, out_features=32, rank=8)

        self.assertEqual(down.weight.dtype, torch.bfloat16)
        self.assertEqual(up.weight.dtype, torch.bfloat16)


class TestKaimingDefaultUnchanged(unittest.TestCase):
    def test_kaiming_default_leaves_lora_up_zero(self):
        torch.manual_seed(42)
        base = torch.nn.Linear(64, 32, bias=False)

        lora = LoRAModule("test", base, multiplier=1.0, lora_dim=8, alpha=4.0)

        self.assertGreater(lora.lora_down.weight.abs().mean().item(), 0.0)
        self.assertTrue(torch.allclose(lora.lora_up.weight, torch.zeros_like(lora.lora_up.weight)))

    def test_true_alias_maps_to_kaiming(self):
        self.assertEqual(parse_init_lora_weights_arg(True), "kaiming")
        self.assertEqual(parse_init_lora_weights_arg("true"), "kaiming")

        base = torch.nn.Linear(64, 32, bias=False)
        lora = LoRAModule("test", base, multiplier=1.0, lora_dim=8, alpha=4.0, init_lora_weights="true")

        self.assertEqual(lora.init_lora_weights, "kaiming")
        self.assertTrue(torch.allclose(lora.lora_up.weight, torch.zeros_like(lora.lora_up.weight)))


class TestSplitDimsOrthogonal(unittest.TestCase):
    def test_each_split_satisfies_ba_zero(self):
        torch.manual_seed(0)
        base = torch.nn.Linear(64, 30, bias=False)

        lora = LoRAModule(
            "test",
            base,
            multiplier=1.0,
            lora_dim=8,
            alpha=4.0,
            split_dims=[10, 10, 10],
            init_lora_weights="orthogonal",
        )

        for i, (down, up) in enumerate(zip(lora.lora_down, lora.lora_up)):
            product = up.weight @ down.weight
            self.assertTrue(torch.allclose(product, torch.zeros_like(product), atol=1e-5), f"split {i}: B @ A != 0")
            self.assertGreater(down.weight.abs().mean().item(), 1e-4)
            self.assertGreater(up.weight.abs().mean().item(), 1e-4)


class TestConv2dFallback(unittest.TestCase):
    def test_conv2d_falls_back_to_kaiming(self):
        torch.manual_seed(0)
        base = torch.nn.Conv2d(8, 4, kernel_size=(3, 3), padding=(1, 1), bias=False)

        lora = LoRAModule("test", base, multiplier=1.0, lora_dim=8, alpha=4.0, init_lora_weights="orthogonal")

        self.assertEqual(lora.orthogonal_init_fallback_reason, "Conv2d")
        self.assertTrue(torch.allclose(lora.lora_up.weight, torch.zeros_like(lora.lora_up.weight)))

    def test_network_counts_conv2d_fallbacks_once(self):
        torch.manual_seed(0)
        network = create_network(
            ["ToyBlock"],
            "lora_unet",
            multiplier=1.0,
            network_dim=8,
            network_alpha=4.0,
            vae=None,
            text_encoders=[],
            unet=ToyModel(),
            init_lora_weights="orthogonal",
        )

        self.assertEqual(network._orthogonal_conv2d_fallback_count, 1)
        linear_lora = next(lora for lora in network.unet_loras if lora.org_module.__class__.__name__ == "Linear")
        conv_lora = next(lora for lora in network.unet_loras if lora.org_module.__class__.__name__ == "Conv2d")
        self.assertTrue(torch.allclose(linear_lora.lora_up.weight @ linear_lora.lora_down.weight, torch.zeros(8, 16), atol=1e-5))
        self.assertTrue(torch.allclose(conv_lora.lora_up.weight, torch.zeros_like(conv_lora.lora_up.weight)))


class TestInitDispatch(unittest.TestCase):
    def test_unknown_init_raises_with_listed_options(self):
        with self.assertRaisesRegex(ValueError, r"init_lora_weights.*kaiming.*orthogonal.*true"):
            parse_init_lora_weights_arg("bogus")

    def test_orthogonal_rejected_for_non_standard_lora_module_class(self):
        with self.assertRaisesRegex(ValueError, r"standard LoRA.*NotLoRAModule"):
            create_network(
                ["ToyBlock"],
                "lora_unet",
                multiplier=1.0,
                network_dim=8,
                network_alpha=4.0,
                vae=None,
                text_encoders=[],
                unet=ToyModel(),
                module_class=NotLoRAModule,
                init_lora_weights="orthogonal",
            )


class TestMetadataPersistence(unittest.TestCase):
    def test_save_weights_records_init_scheme_metadata(self):
        network = create_network(
            ["ToyBlock"],
            "lora_unet",
            multiplier=1.0,
            network_dim=8,
            network_alpha=4.0,
            vae=None,
            text_encoders=[],
            unet=ToyModel(),
            init_lora_weights="orthogonal",
        )

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            network.save_weights(f.name, dtype=None, metadata={})
            with safe_open(f.name, framework="pt") as sf:
                metadata = sf.metadata()

        self.assertEqual(metadata["ss_init_lora_weights"], "orthogonal")


if __name__ == "__main__":
    unittest.main()
