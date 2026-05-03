"""Tier 1 hotswap regression tests.

Pins the load-bearing assumptions of the compile-friendly LoRA hotswap
design (see docs/plans/2026-05-02-peft-tier1-hotswap.md):

  - param.data.copy_() does NOT trigger a Dynamo recompile (CUDA-skip)
  - Forward-output parity: hotswap-merged model == fresh-merged model
  - No delta accumulation across N sequential hotswaps (cache + reload)
  - Cached base snapshot is never mutated in place
  - Non-standard LoRA networks (loha/lokr/hybrid/unknown) are rejected
  - Hash validation (default-on, --no-... opt-out, missing = warn)
  - Multi-LoRA hotswap, rsLoRA hotswap, DoRA hotswap

All CPU-deterministic except TestDynamoNoRecompileOnDataCopy.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.utils.lora_utils import (
    HotswapState,
    _assert_standard_lora_only,
    _check_lora_base_hash,
    compute_base_hash,
    hotswap_lora,
    prepare_for_hotswap,
    setup_parser_hotswap,
)


CUDA_AVAILABLE = torch.cuda.is_available()


# -----------------------------
# Test fixtures: synthetic LoRA + base in the format the helpers expect
# -----------------------------


def _make_synthetic_base_module(in_dim: int = 16, out_dim: int = 8) -> torch.nn.Module:
    """A tiny stand-in for a DiT block. The merge code keys on dotted
    parameter names matching the lora_unet_<name>_weight pattern."""
    torch.manual_seed(0)
    mod = torch.nn.Sequential()
    # The naming `to_q.weight` → `lora_unet_to_q_weight` lora-name lookup matches
    # the helper at `merge_nonlora_to_model:lora_name` construction.
    mod.add_module("to_q", torch.nn.Linear(in_dim, out_dim, bias=False))
    return mod


def _make_lora_sd_for_module(
    module_name: str, in_dim: int, out_dim: int, rank: int = 4, alpha: float = 4.0, *, seed: int = 1
) -> dict:
    torch.manual_seed(seed)
    name = f"lora_unet_{module_name}"
    return {
        f"{name}.lora_down.weight": torch.randn(rank, in_dim) * 0.05,
        f"{name}.lora_up.weight": torch.randn(out_dim, rank) * 0.05,
        f"{name}.alpha": torch.tensor(float(alpha)),
    }


def _make_loha_sd(module_name: str, in_dim: int, out_dim: int, rank: int = 4) -> dict:
    name = f"lora_unet_{module_name}"
    return {
        f"{name}.hada_w1_a": torch.randn(out_dim, rank) * 0.05,
        f"{name}.hada_w1_b": torch.randn(rank, in_dim) * 0.05,
        f"{name}.hada_w2_a": torch.randn(out_dim, rank) * 0.05,
        f"{name}.hada_w2_b": torch.randn(rank, in_dim) * 0.05,
        f"{name}.alpha": torch.tensor(4.0),
    }


def _make_lokr_sd(module_name: str, in_dim: int, out_dim: int) -> dict:
    name = f"lora_unet_{module_name}"
    return {
        f"{name}.lokr_w1": torch.randn(2, 2) * 0.05,
        f"{name}.lokr_w2": torch.randn(out_dim // 2, in_dim // 2) * 0.05,
        f"{name}.alpha": torch.tensor(4.0),
    }


def _save_sd(sd: dict, dirpath: Path, filename: str, metadata: dict | None = None) -> str:
    """Write a state dict as safetensors and return the path."""
    path = dirpath / filename
    save_file(sd, str(path), metadata=metadata)
    return str(path)


# -----------------------------
# TestDynamoNoRecompileOnDataCopy — pins the load-bearing assumption
# -----------------------------


class TestDynamoNoRecompileOnDataCopy(unittest.TestCase):
    """If this test ever fails, the ENTIRE Approach A hotswap design is
    invalidated. param.data.copy_() must not trigger Dynamo recompile."""

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA required for torch.compile recompile probe")
    def test_data_copy_does_not_recompile(self) -> None:
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = 8

        linear = torch.nn.Linear(128, 128, device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(linear, fullgraph=True)
        x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)

        _ = compiled(x).sum().item()
        _ = compiled(x).sum().item()

        # Snapshot Dynamo's internal compile counter
        compile_id_before = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)

        new_weight = torch.randn_like(linear.weight)
        linear.weight.data.copy_(new_weight)

        y_after = compiled(x)
        compile_id_after = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)

        self.assertEqual(
            compile_id_before,
            compile_id_after,
            "param.data.copy_() triggered a new Dynamo trace — hotswap design assumption broken",
        )
        # And the new weight must actually be reflected in the output
        expected = torch.nn.functional.linear(x, new_weight, linear.bias)
        self.assertTrue(torch.allclose(y_after, expected, atol=1e-2))


# -----------------------------
# TestComputeBaseHash
# -----------------------------


class TestComputeBaseHash(unittest.TestCase):
    def test_hash_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            sd = {"x": torch.zeros(4)}
            p = _save_sd(sd, tdpath, "base.safetensors")
            self.assertEqual(compute_base_hash([p]), compute_base_hash([p]))

    def test_hash_changes_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            p1 = _save_sd({"x": torch.zeros(4)}, tdpath, "a.safetensors")
            p2 = _save_sd({"x": torch.ones(4)}, tdpath, "b.safetensors")
            self.assertNotEqual(compute_base_hash([p1]), compute_base_hash([p2]))

    def test_hash_concatenates_in_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            p1 = _save_sd({"x": torch.zeros(4)}, tdpath, "a.safetensors")
            p2 = _save_sd({"x": torch.ones(4)}, tdpath, "b.safetensors")
            h_ab = compute_base_hash([p1, p2])
            h_ba = compute_base_hash([p2, p1])
            self.assertNotEqual(h_ab, h_ba, "hash must depend on path order so caller controls semantic")

    def test_hash_expands_split_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            p1 = _save_sd({"x": torch.zeros(4)}, tdpath, "model-00001-of-00002.safetensors")
            p2 = _save_sd({"y": torch.ones(4)}, tdpath, "model-00002-of-00002.safetensors")

            manual = hashlib.sha256()
            for path in (p1, p2):
                with open(path, "rb") as f:
                    manual.update(f.read())

            self.assertEqual(compute_base_hash([p1]), manual.hexdigest())


# -----------------------------
# TestStandardLoRAOnlyRejection — Phase 1 contract
# -----------------------------


class TestStandardLoRAOnlyRejection(unittest.TestCase):
    def test_standard_lora_passes(self) -> None:
        sd = _make_lora_sd_for_module("to_q", 16, 8)
        _assert_standard_lora_only(sd, "fake.safetensors")  # must not raise

    def test_loha_raises_with_actionable_message(self) -> None:
        sd = _make_loha_sd("to_q", 16, 8)
        with self.assertRaisesRegex(ValueError, r"Phase 1.*standard LoRA only.*loha"):
            _assert_standard_lora_only(sd, "loha_adapter.safetensors")

    def test_lokr_raises(self) -> None:
        sd = _make_lokr_sd("to_q", 16, 8)
        with self.assertRaisesRegex(ValueError, r"Phase 1.*standard LoRA only.*lokr"):
            _assert_standard_lora_only(sd, "lokr_adapter.safetensors")

    def test_hybrid_raises(self) -> None:
        sd = {**_make_lora_sd_for_module("a", 16, 8), **_make_loha_sd("b", 16, 8)}
        with self.assertRaisesRegex(ValueError, r"Phase 1.*standard LoRA only.*hybrid"):
            _assert_standard_lora_only(sd, "hybrid.safetensors")

    def test_unknown_raises(self) -> None:
        sd = {"foo.bar": torch.randn(4, 4)}
        with self.assertRaisesRegex(ValueError, r"Phase 1.*standard LoRA only.*unknown"):
            _assert_standard_lora_only(sd, "mystery.safetensors")


# -----------------------------
# TestHashValidation — default-on with sane back-compat
# -----------------------------


class TestHashValidation(unittest.TestCase):
    def test_matching_hash_passes_silently(self) -> None:
        _check_lora_base_hash({"ss_base_sha256": "abc"}, "lora.st", "abc", strict=True)

    def test_missing_metadata_warns_only(self) -> None:
        # Should not raise — back-compat for old checkpoints
        _check_lora_base_hash(None, "lora.st", "abc", strict=True)
        _check_lora_base_hash({}, "lora.st", "abc", strict=True)
        _check_lora_base_hash({"other_key": "x"}, "lora.st", "abc", strict=True)

    def test_mismatch_strict_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "base-hash mismatch"):
            _check_lora_base_hash({"ss_base_sha256": "wrong"}, "lora.st", "right", strict=True)

    def test_mismatch_nonstrict_warns_only(self) -> None:
        # Should not raise
        _check_lora_base_hash({"ss_base_sha256": "wrong"}, "lora.st", "right", strict=False)


# -----------------------------
# TestPrepareForHotswapCacheModes
# -----------------------------


class TestPrepareForHotswapCacheModes(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)
        # Build a synthetic base file whose hash we can compute
        base_sd = {"to_q.weight": torch.randn(8, 16) * 0.1}
        self.base_path = _save_sd(base_sd, self.tdpath, "base.safetensors")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_cache_in_ram_snapshots_parameters(self) -> None:
        model = _make_synthetic_base_module(16, 8)
        state = prepare_for_hotswap(model, dit_paths=[self.base_path], cache_in_ram=True)
        self.assertIsNotNone(state.cached_base_sd)
        self.assertIn("to_q.weight", state.cached_base_sd)
        self.assertTrue(state.cache_in_ram)
        self.assertEqual(state.base_sha256, compute_base_hash([self.base_path]))

    def test_reload_mode_stores_only_paths(self) -> None:
        model = _make_synthetic_base_module(16, 8)
        state = prepare_for_hotswap(model, dit_paths=[self.base_path], cache_in_ram=False)
        self.assertIsNone(state.cached_base_sd)
        self.assertFalse(state.cache_in_ram)
        self.assertEqual(state.base_dit_paths, [self.base_path])


# -----------------------------
# TestNoAccumulation — load-bearing correctness invariant
# -----------------------------


class TestNoAccumulation(unittest.TestCase):
    """Three sequential hotswaps must produce the same final state as a
    single fresh merge of the third LoRA. If deltas accumulate across
    swaps, the test fails."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)

        # Build a synthetic base
        torch.manual_seed(42)
        self.in_dim = 16
        self.out_dim = 8
        self.base_param = torch.randn(self.out_dim, self.in_dim) * 0.1
        # The model expects parameters via named_parameters with name "to_q.weight"
        # which maps to lora_name "lora_unet_to_q" via merge_nonlora_to_model.
        base_sd_for_disk = {"to_q.weight": self.base_param.clone()}
        self.base_path = _save_sd(base_sd_for_disk, self.tdpath, "base.safetensors")

        # Three different LoRAs targeting the same module
        self.lora_paths = []
        for i in range(3):
            lora_sd = _make_lora_sd_for_module("to_q", self.in_dim, self.out_dim, seed=10 + i)
            p = _save_sd(lora_sd, self.tdpath, f"lora_{i}.safetensors")
            self.lora_paths.append(p)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_model_at_base(self) -> torch.nn.Module:
        model = _make_synthetic_base_module(self.in_dim, self.out_dim)
        with torch.no_grad():
            model.to_q.weight.copy_(self.base_param)
        return model

    def _do_hotswap_chain(self, cache_in_ram: bool) -> torch.Tensor:
        model = self._make_model_at_base()
        state = prepare_for_hotswap(model, dit_paths=[self.base_path], cache_in_ram=cache_in_ram, strict_base_hash=False)
        for path in self.lora_paths:
            hotswap_lora(model, state, [path], [1.0], calc_device=torch.device("cpu"))
        return model.to_q.weight.detach().clone()

    def _fresh_merge_third_lora(self) -> torch.Tensor:
        from musubi_tuner.utils.lora_utils import merge_nonlora_to_model
        from safetensors.torch import load_file

        model = self._make_model_at_base()
        sd = load_file(self.lora_paths[-1])
        merge_nonlora_to_model(model, sd, 1.0, torch.device("cpu"))
        return model.to_q.weight.detach().clone()

    def test_no_accumulation_cache_mode(self) -> None:
        after_chain = self._do_hotswap_chain(cache_in_ram=True)
        fresh = self._fresh_merge_third_lora()
        self.assertTrue(
            torch.allclose(after_chain, fresh, atol=1e-5),
            f"cache-mode hotswap accumulated deltas across swaps; max diff {(after_chain - fresh).abs().max().item()}",
        )

    def test_no_accumulation_reload_mode(self) -> None:
        after_chain = self._do_hotswap_chain(cache_in_ram=False)
        fresh = self._fresh_merge_third_lora()
        self.assertTrue(
            torch.allclose(after_chain, fresh, atol=1e-5),
            f"reload-mode hotswap accumulated deltas across swaps; max diff {(after_chain - fresh).abs().max().item()}",
        )

    def test_cached_base_snapshot_is_not_mutated(self) -> None:
        model = self._make_model_at_base()
        state = prepare_for_hotswap(model, dit_paths=[self.base_path], cache_in_ram=True, strict_base_hash=False)
        snapshot_before = {k: v.clone() for k, v in state.cached_base_sd.items()}
        hotswap_lora(model, state, [self.lora_paths[0]], [1.0], calc_device=torch.device("cpu"))
        for k, v in state.cached_base_sd.items():
            self.assertTrue(
                torch.equal(snapshot_before[k], v),
                f"cached_base_sd[{k}] was mutated by hotswap_lora — accumulation bug",
            )

    def test_reload_mode_accepts_wan_prefixed_base_keys(self) -> None:
        prefixed_base = _save_sd(
            {"model.diffusion_model.to_q.weight": self.base_param.clone()},
            self.tdpath,
            "prefixed_base.safetensors",
        )
        model = self._make_model_at_base()
        with torch.no_grad():
            model.to_q.weight.add_(10.0)  # prove reset reads disk, not current model params

        state = prepare_for_hotswap(model, dit_paths=[prefixed_base], cache_in_ram=False, strict_base_hash=False)
        hotswap_lora(model, state, [self.lora_paths[-1]], [1.0], calc_device=torch.device("cpu"))

        fresh = self._fresh_merge_third_lora()
        self.assertTrue(torch.allclose(model.to_q.weight, fresh, atol=1e-5))

    def test_multiplier_zero_resets_to_base_without_merge(self) -> None:
        model = self._make_model_at_base()
        state = prepare_for_hotswap(model, dit_paths=[self.base_path], cache_in_ram=True, strict_base_hash=False)
        hotswap_lora(model, state, [self.lora_paths[0]], [1.0], calc_device=torch.device("cpu"))

        hotswap_lora(model, state, [self.lora_paths[1]], [0.0], calc_device=torch.device("cpu"))

        self.assertTrue(torch.allclose(model.to_q.weight, self.base_param, atol=1e-5))


# -----------------------------
# TestForwardParity — hotswap == fresh-merge by forward output
# -----------------------------


class TestForwardParity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)
        torch.manual_seed(123)
        self.in_dim = 16
        self.out_dim = 8
        self.base_param = torch.randn(self.out_dim, self.in_dim) * 0.1
        self.base_path = _save_sd({"to_q.weight": self.base_param.clone()}, self.tdpath, "base.safetensors")
        lora_sd = _make_lora_sd_for_module("to_q", self.in_dim, self.out_dim, seed=99)
        self.lora_path = _save_sd(lora_sd, self.tdpath, "lora.safetensors")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_model(self) -> torch.nn.Module:
        m = _make_synthetic_base_module(self.in_dim, self.out_dim)
        with torch.no_grad():
            m.to_q.weight.copy_(self.base_param)
        return m

    def test_forward_parity_after_hotswap(self) -> None:
        from musubi_tuner.utils.lora_utils import merge_nonlora_to_model
        from safetensors.torch import load_file

        # Path A: fresh merge
        model_a = self._make_model()
        merge_nonlora_to_model(model_a, load_file(self.lora_path), 1.0, torch.device("cpu"))

        # Path B: prepare_for_hotswap (no initial LoRA), then hotswap
        model_b = self._make_model()
        state = prepare_for_hotswap(model_b, dit_paths=[self.base_path], cache_in_ram=True, strict_base_hash=False)
        hotswap_lora(model_b, state, [self.lora_path], [1.0], calc_device=torch.device("cpu"))

        x = torch.randn(2, self.in_dim)
        y_a = model_a(x)
        y_b = model_b(x)
        self.assertTrue(
            torch.allclose(y_a, y_b, atol=1e-5),
            f"hotswap forward output differs from fresh-merge; max diff {(y_a - y_b).abs().max().item()}",
        )


# -----------------------------
# TestNonStandardLoRARejection — hotswap-call-time
# -----------------------------


class TestNonStandardLoRARejection(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)
        self.base_path = _save_sd({"to_q.weight": torch.zeros(8, 16)}, self.tdpath, "base.safetensors")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _state_for(self, model: torch.nn.Module) -> HotswapState:
        return prepare_for_hotswap(model, dit_paths=[self.base_path], cache_in_ram=True, strict_base_hash=False)

    def test_loha_state_dict_rejected(self) -> None:
        model = _make_synthetic_base_module(16, 8)
        state = self._state_for(model)
        loha_path = _save_sd(_make_loha_sd("to_q", 16, 8), self.tdpath, "loha.safetensors")
        with self.assertRaisesRegex(ValueError, r"Phase 1.*standard LoRA only.*loha"):
            hotswap_lora(model, state, [loha_path], [1.0], calc_device=torch.device("cpu"))

    def test_lokr_state_dict_rejected(self) -> None:
        model = _make_synthetic_base_module(16, 8)
        state = self._state_for(model)
        lokr_path = _save_sd(_make_lokr_sd("to_q", 16, 8), self.tdpath, "lokr.safetensors")
        with self.assertRaisesRegex(ValueError, r"Phase 1.*standard LoRA only.*lokr"):
            hotswap_lora(model, state, [lokr_path], [1.0], calc_device=torch.device("cpu"))

    def test_hybrid_state_dict_rejected(self) -> None:
        model = _make_synthetic_base_module(16, 8)
        state = self._state_for(model)
        hybrid_sd = {**_make_lora_sd_for_module("a", 16, 8), **_make_loha_sd("b", 16, 8)}
        hybrid_path = _save_sd(hybrid_sd, self.tdpath, "hybrid.safetensors")
        with self.assertRaisesRegex(ValueError, r"Phase 1.*standard LoRA only.*hybrid"):
            hotswap_lora(model, state, [hybrid_path], [1.0], calc_device=torch.device("cpu"))

    def test_unknown_state_dict_rejected(self) -> None:
        model = _make_synthetic_base_module(16, 8)
        state = self._state_for(model)
        unknown_path = _save_sd({"foo.bar": torch.zeros(4)}, self.tdpath, "unknown.safetensors")
        with self.assertRaisesRegex(ValueError, r"Phase 1.*standard LoRA only.*unknown"):
            hotswap_lora(model, state, [unknown_path], [1.0], calc_device=torch.device("cpu"))


# -----------------------------
# TestParserHotswap — argparse-time wiring
# -----------------------------


class TestParserHotswap(unittest.TestCase):
    def _build_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser()
        setup_parser_hotswap(p)
        return p

    def _parse_wan(self, extra_args: list[str]):
        from musubi_tuner import wan_generate_video

        argv = ["prog", "--save_path", "x", "--prompt", "x", *extra_args]
        with patch.object(sys, "argv", argv):
            return wan_generate_video.parse_args()

    def test_default_off(self) -> None:
        ns = self._build_parser().parse_args([])
        self.assertFalse(ns.prepare_for_hotswap)
        self.assertFalse(ns.cache_unmerged_base)
        # Default-on for strict
        self.assertTrue(ns.hotswap_strict_base_hash)

    def test_prepare_for_hotswap_on(self) -> None:
        ns = self._build_parser().parse_args(["--prepare_for_hotswap"])
        self.assertTrue(ns.prepare_for_hotswap)

    def test_no_strict_disables(self) -> None:
        ns = self._build_parser().parse_args(["--no-hotswap_strict_base_hash"])
        self.assertFalse(ns.hotswap_strict_base_hash)

    def test_cache_unmerged_base(self) -> None:
        ns = self._build_parser().parse_args(["--cache_unmerged_base"])
        self.assertTrue(ns.cache_unmerged_base)

    def test_setup_parser_compile_exposes_hotswap_flags(self) -> None:
        from musubi_tuner.hv_generate_video import setup_parser_compile

        p = argparse.ArgumentParser()
        setup_parser_compile(p)

        ns = p.parse_args(["--compile", "--prepare_for_hotswap", "--cache_unmerged_base", "--no-hotswap_strict_base_hash"])

        self.assertTrue(ns.compile)
        self.assertTrue(ns.prepare_for_hotswap)
        self.assertTrue(ns.cache_unmerged_base)
        self.assertFalse(ns.hotswap_strict_base_hash)

    def test_wan_parse_accepts_hotswap_flags(self) -> None:
        args = self._parse_wan(["--prepare_for_hotswap", "--cache_unmerged_base", "--no-hotswap_strict_base_hash"])

        self.assertTrue(args.prepare_for_hotswap)
        self.assertTrue(args.cache_unmerged_base)
        self.assertFalse(args.hotswap_strict_base_hash)

    def test_wan_parse_rejects_prefer_lycoris_hotswap(self) -> None:
        with self.assertRaisesRegex(ValueError, "prepare_for_hotswap.*prefer_lycoris"):
            self._parse_wan(["--prepare_for_hotswap", "--prefer_lycoris"])

    def test_wan_parse_rejects_fp8_scaled_hotswap(self) -> None:
        with self.assertRaisesRegex(ValueError, "prepare_for_hotswap.*fp8_scaled"):
            self._parse_wan(["--prepare_for_hotswap", "--fp8_scaled"])

    def test_wan_parse_rejects_save_merged_model_hotswap(self) -> None:
        with self.assertRaisesRegex(ValueError, "prepare_for_hotswap.*save_merged_model"):
            self._parse_wan(["--prepare_for_hotswap", "--save_merged_model", "merged.safetensors"])


class TestWanDualExpertHotswapPlumbing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)
        self.in_dim = 16
        self.out_dim = 8
        self.low_base = torch.zeros(self.out_dim, self.in_dim)
        self.high_base = torch.ones(self.out_dim, self.in_dim)
        self.low_path = _save_sd({"to_q.weight": self.low_base.clone()}, self.tdpath, "low.safetensors")
        self.high_path = _save_sd({"to_q.weight": self.high_base.clone()}, self.tdpath, "high.safetensors")
        self.low_lora = _save_sd(
            _make_lora_sd_for_module("to_q", self.in_dim, self.out_dim, seed=111),
            self.tdpath,
            "low_lora.safetensors",
        )
        self.high_lora = _save_sd(
            _make_lora_sd_for_module("to_q", self.in_dim, self.out_dim, seed=222),
            self.tdpath,
            "high_lora.safetensors",
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            attn_mode="torch",
            blocks_to_swap=0,
            cache_unmerged_base=True,
            compile=False,
            compile_args=None,
            disable_linear_for_compile=False,
            disable_numpy_memmap=False,
            dit=self.low_path,
            dit_high_noise=self.high_path,
            exclude_patterns=None,
            fp8_fast=False,
            fp8_scaled=False,
            hotswap_strict_base_hash=False,
            include_patterns=None,
            lazy_loading=False,
            lora_multiplier=[0.5],
            lora_multiplier_high_noise=[0.75],
            lora_weight=[self.low_lora],
            lora_weight_high_noise=[self.high_lora],
            mixed_precision_transformer=False,
            offload_inactive_dit=False,
            prefer_lycoris=False,
            prepare_for_hotswap=True,
            save_merged_model=None,
            use_pinned_memory_for_block_swap=False,
        )

    def _fake_load_wan_model(self, _config, _device, dit_path, *_args, **kwargs) -> torch.nn.Module:
        self.assertIsNone(
            kwargs.get("lora_weights_list"),
            "WAN hotswap must suppress load-time LoRA preload before capturing the base",
        )
        model = _make_synthetic_base_module(self.in_dim, self.out_dim)
        with torch.no_grad():
            model.to_q.weight.copy_(load_file(dit_path)["to_q.weight"])
        return model

    def test_dual_expert_models_get_independent_hotswap_state(self) -> None:
        from musubi_tuner import wan_generate_video

        args = self._args()
        with (
            patch.object(wan_generate_video, "load_wan_model", side_effect=self._fake_load_wan_model) as load_mock,
            patch.object(wan_generate_video, "merge_lora_weights") as merge_mock,
        ):
            models = wan_generate_video.load_dit_models(
                args,
                config=object(),
                device=torch.device("cpu"),
                dit_dtype=torch.float32,
                dit_weight_dtype=torch.float32,
            )

        self.assertEqual(load_mock.call_count, 2)
        self.assertEqual(merge_mock.call_count, 2)
        self.assertEqual(merge_mock.call_args_list[0].args[2], [self.low_lora])
        self.assertEqual(merge_mock.call_args_list[1].args[2], [self.high_lora])
        self.assertTrue(merge_mock.call_args_list[0].kwargs["standard_lora_only"])
        self.assertTrue(merge_mock.call_args_list[1].kwargs["standard_lora_only"])

        self.assertEqual(len(models), 2)
        high_model, low_model = models
        self.assertIsNotNone(high_model.hotswap_state)
        self.assertIsNotNone(low_model.hotswap_state)
        self.assertEqual(high_model.hotswap_state.base_dit_paths, [self.high_path])
        self.assertEqual(low_model.hotswap_state.base_dit_paths, [self.low_path])
        self.assertIsNot(high_model.hotswap_state, low_model.hotswap_state)
        self.assertIsNot(high_model.hotswap_state.cached_base_sd, low_model.hotswap_state.cached_base_sd)
        self.assertEqual(high_model.hotswap_state.active_lora_paths, [self.high_lora])
        self.assertEqual(high_model.hotswap_state.active_lora_multipliers, [0.75])
        self.assertEqual(low_model.hotswap_state.active_lora_paths, [self.low_lora])
        self.assertEqual(low_model.hotswap_state.active_lora_multipliers, [0.5])

        low_before = low_model.to_q.weight.detach().clone()
        high_before = high_model.to_q.weight.detach().clone()
        hotswap_lora(high_model, high_model.hotswap_state, [self.high_lora], [1.0], calc_device=torch.device("cpu"))

        self.assertTrue(torch.equal(low_model.to_q.weight, low_before))
        self.assertFalse(torch.equal(high_model.to_q.weight, high_before))


if __name__ == "__main__":
    unittest.main()
