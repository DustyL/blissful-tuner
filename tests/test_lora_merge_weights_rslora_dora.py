"""Regression tests for rsLoRA and DoRA support in the fast LoRA merge helper."""

import math
import unittest

import torch

from musubi_tuner.networks.lora import LoRAInfModule
from musubi_tuner.utils.lora_utils import lora_merge_weights_to_tensor, merge_nonlora_to_model


def _build_lora_sd(
    name: str,
    in_dim: int,
    out_dim: int,
    rank: int,
    alpha: float,
    *,
    use_rslora: bool = False,
    use_dora: bool = False,
    dora_magnitude: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    sd = {
        f"{name}.lora_down.weight": torch.randn(rank, in_dim),
        f"{name}.lora_up.weight": torch.randn(out_dim, rank),
        f"{name}.alpha": torch.tensor(float(alpha)),
    }
    if use_rslora:
        sd["use_rslora_flag"] = torch.tensor(True, dtype=torch.bool)
    if use_dora:
        sd["use_dora_flag"] = torch.tensor(True, dtype=torch.bool)
        if dora_magnitude is not None:
            sd[f"{name}.dora_layer.weight"] = dora_magnitude
    return sd


class TestRSLoRAFastMerge(unittest.TestCase):
    def test_plain_lora_scale_is_unchanged(self) -> None:
        torch.manual_seed(0)
        rank = 4
        alpha = 4.0
        sd = _build_lora_sd("test", 8, 4, rank, alpha)
        keys = set(sd.keys())

        out = lora_merge_weights_to_tensor(torch.zeros(4, 8), "test", sd, keys, 1.0, torch.device("cpu"))
        expected = sd["test.lora_up.weight"] @ sd["test.lora_down.weight"]

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_rslora_uses_sqrt_rank_scale(self) -> None:
        torch.manual_seed(0)
        rank = 4
        alpha = 4.0
        sd = _build_lora_sd("test", 8, 4, rank, alpha, use_rslora=True)
        keys = set(sd.keys())

        out = lora_merge_weights_to_tensor(torch.zeros(4, 8), "test", sd, keys, 1.0, torch.device("cpu"))
        expected = (sd["test.lora_up.weight"] @ sd["test.lora_down.weight"]) * (alpha / math.sqrt(rank))

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_root_rslora_flag_is_consumed(self) -> None:
        sd = _build_lora_sd("test", 8, 4, 4, 4.0, use_rslora=True)
        keys = set(sd.keys())

        lora_merge_weights_to_tensor(torch.zeros(4, 8), "test", sd, keys, 1.0, torch.device("cpu"))

        self.assertNotIn("use_rslora_flag", keys)


class TestDoRAFastMerge(unittest.TestCase):
    def test_dora_merge_matches_networks_lora_merge_to(self) -> None:
        torch.manual_seed(0)
        in_dim = 8
        out_dim = 4
        rank = 4
        alpha = 4.0
        multiplier = 0.75

        base = torch.nn.Linear(in_dim, out_dim, bias=False)
        original_weight = base.weight.detach().clone()
        lora = LoRAInfModule("test", base, multiplier=multiplier, lora_dim=rank, alpha=alpha, use_dora=True)
        with torch.no_grad():
            lora.lora_down.weight.normal_(0.0, 0.2)
            lora.lora_up.weight.normal_(0.0, 0.2)
            delta = multiplier * (lora.lora_up.weight @ lora.lora_down.weight) * lora.scale
            base_norm = torch.linalg.norm(original_weight.float() + delta.float(), dim=1)
            lora.dora_layer.weight.copy_(base_norm * torch.linspace(0.8, 1.2, out_dim))

        module_sd = {
            "lora_down.weight": lora.lora_down.weight.detach().clone(),
            "lora_up.weight": lora.lora_up.weight.detach().clone(),
            "alpha": lora.alpha.detach().clone(),
            "dora_layer.weight": lora.dora_layer.weight.detach().clone(),
        }
        lora.merge_to(module_sd, dtype=None, device=torch.device("cpu"))
        expected = base.weight.detach().clone()

        full_sd = {
            "test.lora_down.weight": module_sd["lora_down.weight"],
            "test.lora_up.weight": module_sd["lora_up.weight"],
            "test.alpha": module_sd["alpha"],
            "test.dora_layer.weight": module_sd["dora_layer.weight"],
            "use_dora_flag": torch.tensor(True, dtype=torch.bool),
        }
        out = lora_merge_weights_to_tensor(
            original_weight.clone(), "test", full_sd, set(full_sd.keys()), multiplier, torch.device("cpu")
        )

        self.assertTrue(torch.allclose(out, expected, atol=1e-5), f"out={out}\nexpected={expected}")

    def test_dora_without_magnitude_raises(self) -> None:
        sd = _build_lora_sd("test", 8, 4, 4, 4.0)
        sd["use_dora_flag"] = torch.tensor(True, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "dora_layer.weight.*missing"):
            lora_merge_weights_to_tensor(torch.zeros(4, 8), "test", sd, set(sd.keys()), 1.0, torch.device("cpu"))

    def test_dora_magnitude_without_flag_uses_plain_lora_merge(self) -> None:
        torch.manual_seed(0)
        sd = _build_lora_sd("test", 8, 4, 4, 4.0)
        sd["test.dora_layer.weight"] = torch.ones(4)
        keys = set(sd.keys())

        out = lora_merge_weights_to_tensor(torch.zeros(4, 8), "test", sd, keys, 1.0, torch.device("cpu"))
        expected = sd["test.lora_up.weight"] @ sd["test.lora_down.weight"]

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))
        self.assertIn("test.dora_layer.weight", keys)

    def test_dora_on_conv2d_raises(self) -> None:
        sd = {
            "test.lora_down.weight": torch.randn(4, 8, 1, 1),
            "test.lora_up.weight": torch.randn(4, 4, 1, 1),
            "test.alpha": torch.tensor(4.0),
            "test.dora_layer.weight": torch.ones(4),
            "use_dora_flag": torch.tensor(True, dtype=torch.bool),
        }

        with self.assertRaises(NotImplementedError):
            lora_merge_weights_to_tensor(torch.zeros(4, 8, 1, 1), "test", sd, set(sd.keys()), 1.0, torch.device("cpu"))

    def test_global_dora_flag_does_not_make_regular_conv2d_dora(self) -> None:
        torch.manual_seed(0)
        multiplier = 0.5
        sd = {
            "test.lora_down.weight": torch.randn(4, 8, 1, 1),
            "test.lora_up.weight": torch.randn(4, 4, 1, 1),
            "test.alpha": torch.tensor(4.0),
            "use_dora_flag": torch.tensor(True, dtype=torch.bool),
        }
        base_weight = torch.randn(4, 8, 1, 1)

        out = lora_merge_weights_to_tensor(base_weight.clone(), "test", sd, set(sd.keys()), multiplier, torch.device("cpu"))
        expected = base_weight + (
            multiplier
            * (sd["test.lora_up.weight"].squeeze(3).squeeze(2) @ sd["test.lora_down.weight"].squeeze(3).squeeze(2))
            .unsqueeze(2)
            .unsqueeze(3)
            * (sd["test.alpha"].item() / sd["test.lora_down.weight"].size(0))
        )

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


class TestSafeMerge(unittest.TestCase):
    def test_safe_merge_off_default_preserves_existing_behavior(self) -> None:
        sd = _build_lora_sd("test", 8, 4, 4, 4.0)
        sd["test.lora_up.weight"][0, 0] = float("nan")

        out = lora_merge_weights_to_tensor(torch.zeros(4, 8), "test", sd, set(sd.keys()), 1.0, torch.device("cpu"))

        self.assertTrue(torch.isnan(out).any())

    def test_safe_merge_on_raises_on_nonfinite_result(self) -> None:
        sd = _build_lora_sd("test", 8, 4, 4, 4.0)
        sd["test.lora_up.weight"][0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "non-finite"):
            lora_merge_weights_to_tensor(torch.zeros(4, 8), "test", sd, set(sd.keys()), 1.0, torch.device("cpu"), safe_merge=True)

    def test_merge_nonlora_safe_merge_does_not_commit_nonfinite_result(self) -> None:
        model = torch.nn.Module()
        model.to_q = torch.nn.Linear(8, 4, bias=False)
        original = model.to_q.weight.detach().clone()
        sd = _build_lora_sd("lora_unet_to_q", 8, 4, 4, 4.0)
        sd["lora_unet_to_q.lora_up.weight"][0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "non-finite"):
            merge_nonlora_to_model(model, sd, 1.0, torch.device("cpu"), safe_merge=True)

        self.assertTrue(torch.allclose(model.to_q.weight, original))

    def test_lora_inf_merge_to_safe_merge_does_not_commit_nonfinite_result(self) -> None:
        base = torch.nn.Linear(8, 4, bias=False)
        original = base.weight.detach().clone()
        module = LoRAInfModule("test", base, multiplier=1.0, lora_dim=4, alpha=4.0)
        sd = {
            "lora_down.weight": torch.randn(4, 8),
            "lora_up.weight": torch.randn(4, 4),
            "alpha": torch.tensor(4.0),
        }
        sd["lora_up.weight"][0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "non-finite"):
            module.merge_to(sd, dtype=None, device=torch.device("cpu"), safe_merge=True)

        self.assertTrue(torch.allclose(base.weight, original))


if __name__ == "__main__":
    unittest.main()
