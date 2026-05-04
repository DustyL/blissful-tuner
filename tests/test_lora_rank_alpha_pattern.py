"""Regression tests for static rank_pattern / alpha_pattern LoRA creation."""

import json
import tempfile
import unittest

import torch
from safetensors import safe_open

from musubi_tuner.networks import loha
from musubi_tuner.networks.lora import (
    RANK_PATTERN_MATCH_SEMANTICS,
    create_network,
    create_network_from_weights,
    parse_rank_or_alpha_pattern_arg,
)


class ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(16, 8, bias=False)
        self.to_k = torch.nn.Linear(16, 8, bias=False)


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = ToyAttention()
        self.mlp = torch.nn.Linear(16, 8, bias=False)
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=(3, 3), padding=(1, 1), bias=False)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ToyBlock()


class NotLoRAModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _create_network(**kwargs):
    return create_network(
        ["ToyBlock"],
        "lora_unet",
        multiplier=1.0,
        network_dim=8,
        network_alpha=4.0,
        vae=None,
        text_encoders=[],
        unet=ToyModel(),
        **kwargs,
    )


def _lora_by_name(network, suffix):
    return next(lora for lora in network.unet_loras if lora.lora_name.endswith(suffix))


class TestRankPatternResolution(unittest.TestCase):
    def test_no_rank_pattern_uses_network_dim(self):
        network = _create_network()

        self.assertEqual(_lora_by_name(network, "block_attn_to_q").lora_dim, 8)
        self.assertEqual(_lora_by_name(network, "block_attn_to_k").lora_dim, 8)
        self.assertEqual(_lora_by_name(network, "block_mlp").lora_dim, 8)

    def test_rank_pattern_overrides_matching_module(self):
        network = _create_network(rank_pattern=r"{'.*attn\\.to_q': 16}")

        self.assertEqual(_lora_by_name(network, "block_attn_to_q").lora_dim, 16)
        self.assertEqual(_lora_by_name(network, "block_attn_to_k").lora_dim, 8)

    def test_fullmatch_requires_pattern_to_cover_original_name(self):
        network = _create_network(rank_pattern=r"{'to_q': 16}")

        self.assertEqual(_lora_by_name(network, "block_attn_to_q").lora_dim, 8)

    def test_first_matching_rank_pattern_wins(self):
        network = _create_network(rank_pattern=r"{'.*attn\\..*': 16, '.*attn\\.to_q': 32}")

        self.assertEqual(_lora_by_name(network, "block_attn_to_q").lora_dim, 16)

    def test_rank_pattern_can_target_conv2d_without_conv_dim(self):
        network = _create_network(rank_pattern=r"{'.*conv': 12}")

        conv_lora = _lora_by_name(network, "block_conv")
        self.assertEqual(conv_lora.lora_dim, 12)
        self.assertEqual(conv_lora.org_module.__class__.__name__, "Conv2d")


class TestAlphaPatternResolution(unittest.TestCase):
    def test_alpha_pattern_overrides_default_alpha(self):
        network = _create_network(alpha_pattern=r"{'.*attn\\.to_q': 6.5}")

        self.assertEqual(float(_lora_by_name(network, "block_attn_to_q").alpha.item()), 6.5)
        self.assertEqual(float(_lora_by_name(network, "block_attn_to_k").alpha.item()), 4.0)

    def test_alpha_pattern_accepts_int_and_float_values(self):
        network = _create_network(alpha_pattern=r"{'.*attn\\.to_q': 8, '.*attn\\.to_k': 8.5}")

        self.assertEqual(float(_lora_by_name(network, "block_attn_to_q").alpha.item()), 8.0)
        self.assertEqual(float(_lora_by_name(network, "block_attn_to_k").alpha.item()), 8.5)

    def test_alpha_pattern_uses_fullmatch(self):
        network = _create_network(alpha_pattern=r"{'to_q': 8.5}")

        self.assertEqual(float(_lora_by_name(network, "block_attn_to_q").alpha.item()), 4.0)

    def test_first_matching_alpha_pattern_wins(self):
        network = _create_network(alpha_pattern=r"{'.*attn\\..*': 6.0, '.*attn\\.to_q': 8.5}")

        self.assertEqual(float(_lora_by_name(network, "block_attn_to_q").alpha.item()), 6.0)

    def test_alpha_only_pattern_does_not_create_skipped_conv2d(self):
        network = _create_network(alpha_pattern=r"{'.*conv': 8.5}")

        self.assertFalse(any(lora.lora_name.endswith("block_conv") for lora in network.unet_loras))


class TestPatternParsing(unittest.TestCase):
    def test_json_dict_string_parses(self):
        parsed = parse_rank_or_alpha_pattern_arg(r'{".*": 16}', "rank_pattern")

        [(pattern, value)] = parsed.items()
        self.assertEqual(pattern.pattern, ".*")
        self.assertEqual(value, 16)

    def test_python_literal_dict_string_parses(self):
        parsed = parse_rank_or_alpha_pattern_arg(r"{'.*attn\\.to_q': 16}", "rank_pattern")

        [(pattern, value)] = parsed.items()
        self.assertEqual(pattern.pattern, r".*attn\.to_q")
        self.assertEqual(value, 16)

    def test_invalid_regex_raises(self):
        with self.assertRaisesRegex(ValueError, "did not compile"):
            parse_rank_or_alpha_pattern_arg({"[": 16}, "rank_pattern")

    def test_negative_rank_raises(self):
        with self.assertRaisesRegex(ValueError, "positive int"):
            parse_rank_or_alpha_pattern_arg({".*": -1}, "rank_pattern")

    def test_zero_rank_raises(self):
        with self.assertRaisesRegex(ValueError, "positive int"):
            parse_rank_or_alpha_pattern_arg({".*": 0}, "rank_pattern")

    def test_float_rank_raises(self):
        with self.assertRaisesRegex(ValueError, "positive int"):
            parse_rank_or_alpha_pattern_arg({".*": 4.5}, "rank_pattern")

    def test_negative_alpha_raises(self):
        with self.assertRaisesRegex(ValueError, "positive number"):
            parse_rank_or_alpha_pattern_arg({".*": -1.0}, "alpha_pattern")

    def test_bool_values_raise(self):
        with self.assertRaisesRegex(ValueError, "positive int"):
            parse_rank_or_alpha_pattern_arg({".*": True}, "rank_pattern")
        with self.assertRaisesRegex(ValueError, "positive number"):
            parse_rank_or_alpha_pattern_arg({".*": False}, "alpha_pattern")

    def test_patterns_rejected_for_non_standard_lora_module_class(self):
        with self.assertRaisesRegex(ValueError, r"standard LoRA.*NotLoRAModule"):
            _create_network(module_class=NotLoRAModule, rank_pattern=r"{'.*': 16}")

    def test_patterns_rejected_for_loha_network(self):
        with self.assertRaisesRegex(ValueError, r"standard LoRA.*LoHa"):
            loha.create_network(
                ["ToyBlock"],
                "lora_unet",
                multiplier=1.0,
                network_dim=8,
                network_alpha=4.0,
                vae=None,
                text_encoders=[],
                unet=ToyModel(),
                rank_pattern=r"{'.*': 16}",
            )


class TestOrthogonalInteraction(unittest.TestCase):
    def test_orthogonal_odd_rank_pattern_raises_with_module_and_pattern(self):
        with self.assertRaisesRegex(ValueError, r"block\.attn\.to_q.*rank 5.*rank_pattern.*attn.*even rank"):
            _create_network(init_lora_weights="orthogonal", rank_pattern=r"{'.*attn\\.to_q': 5}")

    def test_orthogonal_even_rank_pattern_passes(self):
        network = _create_network(init_lora_weights="orthogonal", rank_pattern=r"{'.*attn\\.to_q': 16}")

        self.assertEqual(_lora_by_name(network, "block_attn_to_q").lora_dim, 16)

    def test_orthogonal_off_means_odd_rank_pattern_passes(self):
        network = _create_network(init_lora_weights="kaiming", rank_pattern=r"{'.*attn\\.to_q': 5}")

        self.assertEqual(_lora_by_name(network, "block_attn_to_q").lora_dim, 5)


class TestMetadataPersistence(unittest.TestCase):
    def test_metadata_includes_rank_and_alpha_pattern_json(self):
        network = _create_network(
            rank_pattern=r"{'.*attn\\.to_q': 16}",
            alpha_pattern=r"{'.*attn\\.to_q': 6.5}",
        )

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            network.save_weights(f.name, dtype=None, metadata={})
            with safe_open(f.name, framework="pt") as sf:
                metadata = sf.metadata()

        self.assertEqual(json.loads(metadata["ss_rank_pattern"]), {r".*attn\.to_q": 16})
        self.assertEqual(json.loads(metadata["ss_alpha_pattern"]), {r".*attn\.to_q": 6.5})

    def test_metadata_omits_patterns_when_unused(self):
        network = _create_network()

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            network.save_weights(f.name, dtype=None, metadata={})
            with safe_open(f.name, framework="pt") as sf:
                metadata = sf.metadata()

        self.assertNotIn("ss_rank_pattern", metadata)
        self.assertNotIn("ss_alpha_pattern", metadata)
        self.assertNotIn("ss_rank_pattern_match_semantics", metadata)

    def test_metadata_includes_match_semantics_when_patterns_used(self):
        network = _create_network(rank_pattern=r"{'.*attn\\.to_q': 16}")

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            network.save_weights(f.name, dtype=None, metadata={})
            with safe_open(f.name, framework="pt") as sf:
                metadata = sf.metadata()

        self.assertEqual(metadata["ss_rank_pattern_match_semantics"], RANK_PATTERN_MATCH_SEMANTICS)


class TestLoadFromWeightsIgnoresPattern(unittest.TestCase):
    def test_modules_dim_wins_over_rank_pattern_at_load(self):
        weights_sd = {
            "lora_unet_block_attn_to_q.lora_down.weight": torch.zeros(4, 16),
            "lora_unet_block_attn_to_q.alpha": torch.tensor(2.0),
        }

        network = create_network_from_weights(
            ["ToyBlock"],
            multiplier=1.0,
            weights_sd=weights_sd,
            text_encoders=[],
            unet=ToyModel(),
            rank_pattern=r"{'.*attn\\.to_q': 16}",
        )

        lora = _lora_by_name(network, "block_attn_to_q")
        self.assertEqual(lora.lora_dim, 4)
        self.assertEqual(float(lora.alpha.item()), 2.0)


if __name__ == "__main__":
    unittest.main()
