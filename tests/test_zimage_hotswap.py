"""Tier 1 #1c Z-Image hotswap regression tests.

Pins the Z-Image generalization of the compile-friendly LoRA hotswap design
(see docs/plans/2026-05-04-peft-tier1c-zimage-qwenimage-hotswap.md):

  - Parser accepts hotswap flags via inherited setup_parser_compile()
  - Parser rejects --prefer_lycoris, --fp8_scaled, --fp8, --save_merged_model,
    --latent_path when --prepare_for_hotswap is set
  - prepare_zimage_hotswap_state() sets model.hotswap_state to None when off
  - prepare_zimage_hotswap_state() captures un-merged base when on
  - note_zimage_initial_loras() records the initial active LoRA set
  - note_zimage_initial_loras() pads multipliers to 1.0 (matches WAN convention)
  - load_dit_model lifecycle: suppress→capture→merge→note ordering preserved

All CPU-deterministic. Lifecycle tests use source-text introspection on
load_dit_model rather than executing the heavy model load.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file


def _save_tiny_dit(dirpath: Path, filename: str = "tiny_dit.safetensors") -> str:
    """Write a tiny safetensors file to stand in for a Z-Image DiT.

    compute_base_hash() reads file bytes, so the path must exist on disk.
    """
    path = dirpath / filename
    save_file({"to_q.weight": torch.zeros(8, 16)}, str(path))
    return str(path)


class _TinyZImageStub(torch.nn.Module):
    """Stand-in for a Z-Image DiT for hotswap-state capture tests.

    prepare_for_hotswap() iterates model.named_parameters(), so the stub must
    expose at least one nn.Parameter.
    """

    def __init__(self, in_dim: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(in_dim, out_dim, bias=False)


# ============================================================================
# Parser-level tests
# ============================================================================


class TestZImageHotswapParser(unittest.TestCase):
    """Z-Image parser inherits hotswap flags via setup_parser_compile()."""

    def _parse_zimage(self, extra_args: list[str]):
        from musubi_tuner import zimage_generate_image

        argv = [
            "prog",
            "--text_encoder",
            "x",
            "--save_path",
            "x",
            "--prompt",
            "x",
            *extra_args,
        ]
        with patch.object(sys, "argv", argv):
            return zimage_generate_image.parse_args()

    def test_zimage_parse_accepts_hotswap_flags(self) -> None:
        args = self._parse_zimage(["--prepare_for_hotswap", "--cache_unmerged_base", "--no-hotswap_strict_base_hash"])
        self.assertTrue(args.prepare_for_hotswap)
        self.assertTrue(args.cache_unmerged_base)
        self.assertFalse(args.hotswap_strict_base_hash)

    def test_zimage_parse_default_off(self) -> None:
        args = self._parse_zimage([])
        self.assertFalse(args.prepare_for_hotswap)
        self.assertFalse(args.cache_unmerged_base)
        self.assertTrue(args.hotswap_strict_base_hash)

    def test_zimage_hotswap_rejects_prefer_lycoris(self) -> None:
        with self.assertRaisesRegex(ValueError, r"prepare_for_hotswap.*prefer_lycoris"):
            self._parse_zimage(["--prepare_for_hotswap", "--prefer_lycoris"])

    def test_zimage_hotswap_rejects_fp8_scaled(self) -> None:
        with self.assertRaisesRegex(ValueError, r"prepare_for_hotswap.*--fp8_scaled"):
            self._parse_zimage(["--prepare_for_hotswap", "--fp8_scaled"])

    def test_zimage_hotswap_rejects_fp8(self) -> None:
        # Tighter than `fp8` to avoid matching the fp8_scaled error message.
        with self.assertRaisesRegex(ValueError, r"prepare_for_hotswap.*--fp8 in Z-Image"):
            self._parse_zimage(["--prepare_for_hotswap", "--fp8"])

    def test_zimage_hotswap_rejects_save_merged_model(self) -> None:
        with self.assertRaisesRegex(ValueError, r"prepare_for_hotswap.*save_merged_model"):
            self._parse_zimage(["--prepare_for_hotswap", "--save_merged_model", "merged.safetensors"])

    def test_zimage_hotswap_rejects_latent_path(self) -> None:
        from musubi_tuner import zimage_generate_image

        argv = [
            "prog",
            "--text_encoder",
            "x",
            "--save_path",
            "x",
            "--latent_path",
            "x.safetensors",
            "--prepare_for_hotswap",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, r"prepare_for_hotswap.*latent_path"):
                zimage_generate_image.parse_args()


# ============================================================================
# Helper / lifecycle tests
# ============================================================================


class TestPrepareZImageHotswapState(unittest.TestCase):
    """prepare_zimage_hotswap_state() shape contract."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)
        self.dit_path = _save_tiny_dit(self.tdpath)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _args(self, **overrides) -> argparse.Namespace:
        ns = argparse.Namespace(
            prepare_for_hotswap=False,
            cache_unmerged_base=False,
            hotswap_strict_base_hash=True,
            dit=self.dit_path,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_off_sets_hotswap_state_to_none(self) -> None:
        from musubi_tuner.zimage_generate_image import prepare_zimage_hotswap_state

        model = _TinyZImageStub()
        result = prepare_zimage_hotswap_state(model, self._args(prepare_for_hotswap=False))

        self.assertIsNone(result)
        self.assertIsNone(model.hotswap_state)

    def test_on_with_cache_in_ram_captures_state(self) -> None:
        from musubi_tuner.zimage_generate_image import prepare_zimage_hotswap_state

        model = _TinyZImageStub()
        with torch.no_grad():
            model.to_q.weight.fill_(0.5)

        result = prepare_zimage_hotswap_state(model, self._args(prepare_for_hotswap=True, cache_unmerged_base=True))

        self.assertIsNotNone(result)
        self.assertIs(model.hotswap_state, result)
        self.assertEqual(result.base_dit_paths, [self.dit_path])
        self.assertTrue(result.cache_in_ram)
        self.assertIsNotNone(result.cached_base_sd)
        self.assertIn("to_q.weight", result.cached_base_sd)
        self.assertTrue(torch.equal(result.cached_base_sd["to_q.weight"], torch.full((8, 16), 0.5)))

    def test_on_with_reload_mode_stores_only_paths(self) -> None:
        from musubi_tuner.zimage_generate_image import prepare_zimage_hotswap_state

        model = _TinyZImageStub()
        result = prepare_zimage_hotswap_state(model, self._args(prepare_for_hotswap=True, cache_unmerged_base=False))

        self.assertIsNotNone(result)
        self.assertFalse(result.cache_in_ram)
        self.assertIsNone(result.cached_base_sd)
        self.assertEqual(result.base_dit_paths, [self.dit_path])

    def test_strict_base_hash_threaded_through(self) -> None:
        from musubi_tuner.zimage_generate_image import prepare_zimage_hotswap_state

        model = _TinyZImageStub()
        result = prepare_zimage_hotswap_state(
            model,
            self._args(prepare_for_hotswap=True, hotswap_strict_base_hash=False),
        )

        self.assertFalse(result.strict_base_hash)


class TestNoteZImageInitialLoras(unittest.TestCase):
    """note_zimage_initial_loras() bookkeeping shape."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)
        self.dit_path = _save_tiny_dit(self.tdpath)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _model_with_state(self, args: argparse.Namespace) -> _TinyZImageStub:
        from musubi_tuner.zimage_generate_image import prepare_zimage_hotswap_state

        model = _TinyZImageStub()
        prepare_zimage_hotswap_state(model, args)
        return model

    def _args(self, **overrides) -> argparse.Namespace:
        ns = argparse.Namespace(
            prepare_for_hotswap=True,
            cache_unmerged_base=False,
            hotswap_strict_base_hash=True,
            dit=self.dit_path,
            lora_weight=None,
            lora_multiplier=None,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_off_is_noop(self) -> None:
        from musubi_tuner.zimage_generate_image import note_zimage_initial_loras

        args = self._args(prepare_for_hotswap=False)
        model = _TinyZImageStub()
        model.hotswap_state = None

        note_zimage_initial_loras(model, args)

        self.assertIsNone(model.hotswap_state)

    def test_records_paths_and_multipliers(self) -> None:
        from musubi_tuner.zimage_generate_image import note_zimage_initial_loras

        args = self._args(lora_weight=["a.safetensors", "b.safetensors"], lora_multiplier=[0.7, 0.3])
        model = self._model_with_state(args)

        note_zimage_initial_loras(model, args)

        self.assertEqual(model.hotswap_state.active_lora_paths, ["a.safetensors", "b.safetensors"])
        self.assertEqual(model.hotswap_state.active_lora_multipliers, [0.7, 0.3])

    def test_pads_missing_multipliers_to_one(self) -> None:
        from musubi_tuner.zimage_generate_image import note_zimage_initial_loras

        args = self._args(lora_weight=["a.safetensors", "b.safetensors"], lora_multiplier=[0.5])
        model = self._model_with_state(args)

        note_zimage_initial_loras(model, args)

        self.assertEqual(model.hotswap_state.active_lora_multipliers, [0.5, 1.0])

    def test_truncates_extra_multipliers(self) -> None:
        from musubi_tuner.zimage_generate_image import note_zimage_initial_loras

        args = self._args(lora_weight=["a.safetensors", "b.safetensors"], lora_multiplier=[0.5, 0.6, 0.7])
        model = self._model_with_state(args)

        note_zimage_initial_loras(model, args)

        self.assertEqual(model.hotswap_state.active_lora_multipliers, [0.5, 0.6])

    def test_none_multipliers_means_all_one(self) -> None:
        from musubi_tuner.zimage_generate_image import note_zimage_initial_loras

        args = self._args(lora_weight=["a.safetensors", "b.safetensors"], lora_multiplier=None)
        model = self._model_with_state(args)

        note_zimage_initial_loras(model, args)

        self.assertEqual(model.hotswap_state.active_lora_multipliers, [1.0, 1.0])

    def test_missing_state_raises_actionable(self) -> None:
        from musubi_tuner.zimage_generate_image import note_zimage_initial_loras

        args = self._args(lora_weight=["a.safetensors"], lora_multiplier=[1.0])
        model = _TinyZImageStub()
        model.hotswap_state = None

        with self.assertRaisesRegex(RuntimeError, "hotswap_state missing"):
            note_zimage_initial_loras(model, args)

    def test_no_initial_lora_leaves_active_lists_empty(self) -> None:
        """Sweep-script use case: prepare hotswap but start from base, no initial LoRA."""
        from musubi_tuner.zimage_generate_image import prepare_zimage_hotswap_state

        args = self._args(lora_weight=None, lora_multiplier=None)
        model = _TinyZImageStub()

        prepare_zimage_hotswap_state(model, args)

        self.assertIsNotNone(model.hotswap_state)
        self.assertEqual(model.hotswap_state.active_lora_paths, [])
        self.assertEqual(model.hotswap_state.active_lora_multipliers, [])


# ============================================================================
# Lifecycle ordering tests (suppress→capture→merge→note in load_dit_model)
# ============================================================================


class TestZImageLoadDitModelLifecycleOrdering(unittest.TestCase):
    """load_dit_model must thread suppress→capture→merge→note in order.

    Source-text introspection: parse load_dit_model and verify the relative line
    positions of the four hotswap-relevant statements. Cheap, stable, and pinned
    against future refactors that might silently reorder the lifecycle.
    """

    def _load_dit_model_lines(self) -> list[str]:
        import musubi_tuner.zimage_generate_image as mod

        src_path = Path(mod.__file__)
        text = src_path.read_text()
        lines = text.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("def load_dit_model("))
        end = next(
            (i for i, ln in enumerate(lines[start + 1 :], start=start + 1) if ln.startswith("def ")),
            len(lines),
        )
        return lines[start:end]

    def test_suppresses_lora_preload_when_hotswap_enabled(self) -> None:
        lines = self._load_dit_model_lines()
        # The hotswap branch must set lora_weights_list to None
        joined = "\n".join(lines)
        self.assertIn("if prepare_for_hotswap:", joined)
        self.assertIn("lora_weights_list = None", joined)

    def test_capture_after_load_before_optimize(self) -> None:
        lines = self._load_dit_model_lines()
        load_idx = next(i for i, ln in enumerate(lines) if "zimage_model.load_zimage_model" in ln)
        capture_idx = next(i for i, ln in enumerate(lines) if "prepare_zimage_hotswap_state(model, args)" in ln)
        # Block-swap is the canonical "first optimize step" for Z-Image
        block_swap_idx = next(i for i, ln in enumerate(lines) if "model.enable_block_swap" in ln)
        self.assertLess(load_idx, capture_idx)
        self.assertLess(capture_idx, block_swap_idx)

    def test_initial_merge_uses_standard_lora_only_under_hotswap(self) -> None:
        lines = self._load_dit_model_lines()
        joined = "\n".join(lines)
        # Must explicitly pass standard_lora_only=True for hotswap-mode merge
        self.assertIn("standard_lora_only=True", joined)
        # And the merge must reference lora_zimage as the network module
        self.assertIn("merge_lora_weights(\n            lora_zimage,", joined)

    def test_note_after_initial_merge(self) -> None:
        lines = self._load_dit_model_lines()
        # Find the hotswap-block merge call (the one that uses standard_lora_only=True)
        merge_idx = None
        for i, ln in enumerate(lines):
            if "standard_lora_only=True" in ln:
                merge_idx = i
                break
        self.assertIsNotNone(merge_idx, "Hotswap merge call (standard_lora_only=True) not found")
        note_idx = next(i for i, ln in enumerate(lines) if "note_zimage_initial_loras(model, args)" in ln)
        self.assertLess(merge_idx, note_idx)


class TestZImageLoadDitModelMockedIntegration(unittest.TestCase):
    """Execute the hotswap load branch with the heavy model loader mocked out."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)
        self.dit_path = _save_tiny_dit(self.tdpath)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            prepare_for_hotswap=True,
            cache_unmerged_base=False,
            hotswap_strict_base_hash=True,
            dit=self.dit_path,
            blocks_to_swap=0,
            prefer_lycoris=False,
            lora_weight=["initial.safetensors"],
            lora_multiplier=[0.25],
            include_patterns=None,
            exclude_patterns=None,
            fp8_scaled=False,
            fp8=False,
            save_merged_model=None,
            disable_numpy_memmap=False,
            use_32bit_attention=False,
            attn_mode="torch",
            use_pinned_memory_for_block_swap=False,
            compile=False,
        )

    def test_load_branch_suppresses_preload_then_captures_merges_and_notes(self) -> None:
        import musubi_tuner.zimage_generate_image as mod

        args = self._args()
        device = torch.device("cpu")
        model = _TinyZImageStub()
        events: list[str] = []
        original_prepare = mod.prepare_zimage_hotswap_state
        original_note = mod.note_zimage_initial_loras

        def load_spy(*_args, **kwargs):
            events.append("load")
            self.assertIsNone(kwargs["lora_weights_list"])
            return model

        def prepare_spy(loaded_model, loaded_args):
            events.append("capture")
            self.assertIs(loaded_model, model)
            return original_prepare(loaded_model, loaded_args)

        def merge_spy(*merge_args, **kwargs):
            events.append("merge")
            self.assertIs(merge_args[1], model)
            self.assertIsNotNone(model.hotswap_state)
            self.assertEqual(merge_args[2], ["initial.safetensors"])
            self.assertEqual(merge_args[3], [0.25])
            self.assertTrue(kwargs["standard_lora_only"])

        def note_spy(loaded_model, loaded_args):
            events.append("note")
            return original_note(loaded_model, loaded_args)

        with (
            patch.object(mod.zimage_model, "load_zimage_model", side_effect=load_spy),
            patch.object(mod, "prepare_zimage_hotswap_state", side_effect=prepare_spy),
            patch.object(mod, "merge_lora_weights", side_effect=merge_spy),
            patch.object(mod, "note_zimage_initial_loras", side_effect=note_spy),
            patch.object(mod, "clean_memory_on_device"),
        ):
            result = mod.load_dit_model(args, device, torch.float32)

        self.assertIs(result, model)
        self.assertEqual(events, ["load", "capture", "merge", "note"])
        self.assertEqual(model.hotswap_state.active_lora_paths, ["initial.safetensors"])
        self.assertEqual(model.hotswap_state.active_lora_multipliers, [0.25])


if __name__ == "__main__":
    unittest.main()
