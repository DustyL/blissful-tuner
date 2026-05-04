"""Tier 1 #1b FLUX.2 hotswap regression tests.

Pins the FLUX.2 generalization of the compile-friendly LoRA hotswap design
(see docs/plans/2026-05-04-peft-tier1b-flux2-hotswap.md):

  - Parser accepts hotswap flags via inherited setup_parser_compile()
  - Parser rejects --prefer_lycoris, --fp8_scaled, --fp8, --save_merged_model
    when --prepare_for_hotswap is set
  - prepare_flux2_hotswap_state() sets model.hotswap_state to None when off
  - prepare_flux2_hotswap_state() captures un-merged base when on
  - note_flux2_initial_loras() records the initial active LoRA set
  - note_flux2_initial_loras() pads multipliers to 1.0 (matches WAN convention)
  - The lifecycle ordering (prepare BEFORE merge, note AFTER merge) is preserved

All CPU-deterministic. The lifecycle tests stub heavy pieces and exercise the
real orchestrator helpers; we do not load FLUX.2 DiT/VAE/text encoders here.
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
    """Write a tiny safetensors file to stand in for a FLUX.2 DiT.

    compute_base_hash() reads file bytes, so the path must exist on disk.
    """
    path = dirpath / filename
    save_file({"to_q.weight": torch.zeros(8, 16)}, str(path))
    return str(path)


class _TinyFlux2Stub(torch.nn.Module):
    """Stand-in for a FLUX.2 DiT for hotswap-state capture tests.

    prepare_for_hotswap() iterates model.named_parameters(), so the stub must
    expose at least one nn.Parameter. We mirror a Linear named to_q.
    """

    def __init__(self, in_dim: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(in_dim, out_dim, bias=False)


# ============================================================================
# Parser-level tests
# ============================================================================


class TestFlux2HotswapParser(unittest.TestCase):
    """FLUX.2 parser inherits hotswap flags via setup_parser_compile()."""

    def _parse_flux2(self, extra_args: list[str]):
        from musubi_tuner import flux_2_generate_image

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
            return flux_2_generate_image.parse_args()

    def test_flux2_parse_accepts_hotswap_flags(self) -> None:
        args = self._parse_flux2(["--prepare_for_hotswap", "--cache_unmerged_base", "--no-hotswap_strict_base_hash"])
        self.assertTrue(args.prepare_for_hotswap)
        self.assertTrue(args.cache_unmerged_base)
        self.assertFalse(args.hotswap_strict_base_hash)

    def test_flux2_parse_default_off(self) -> None:
        args = self._parse_flux2([])
        self.assertFalse(args.prepare_for_hotswap)
        self.assertFalse(args.cache_unmerged_base)
        self.assertTrue(args.hotswap_strict_base_hash)

    def test_flux2_hotswap_rejects_prefer_lycoris(self) -> None:
        with self.assertRaisesRegex(ValueError, "prepare_for_hotswap.*prefer_lycoris"):
            self._parse_flux2(["--prepare_for_hotswap", "--prefer_lycoris"])

    def test_flux2_hotswap_rejects_fp8_scaled(self) -> None:
        with self.assertRaisesRegex(ValueError, r"prepare_for_hotswap.*--fp8_scaled"):
            self._parse_flux2(["--prepare_for_hotswap", "--fp8_scaled"])

    def test_flux2_hotswap_rejects_fp8(self) -> None:
        # Tighter than `fp8` to avoid accidentally matching the fp8_scaled error message.
        with self.assertRaisesRegex(ValueError, r"prepare_for_hotswap.*--fp8 in FLUX\.2"):
            self._parse_flux2(["--prepare_for_hotswap", "--fp8"])

    def test_flux2_hotswap_rejects_save_merged_model(self) -> None:
        with self.assertRaisesRegex(ValueError, r"prepare_for_hotswap.*save_merged_model"):
            self._parse_flux2(["--prepare_for_hotswap", "--save_merged_model", "merged.safetensors"])

    def test_flux2_hotswap_rejects_latent_path(self) -> None:
        # Latent-only decode never loads the DiT, so hotswap state cannot be prepared.
        # Reject at parse time rather than silently ignoring the flag.
        # We need a real-ish argv: latent_path is given so --prompt is not required.
        from musubi_tuner import flux_2_generate_image

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
                flux_2_generate_image.parse_args()


# ============================================================================
# Helper / lifecycle tests
# ============================================================================


class TestPrepareFlux2HotswapState(unittest.TestCase):
    """prepare_flux2_hotswap_state() shape contract."""

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
        from musubi_tuner.flux_2_generate_image import prepare_flux2_hotswap_state

        model = _TinyFlux2Stub()
        result = prepare_flux2_hotswap_state(model, self._args(prepare_for_hotswap=False))

        self.assertIsNone(result)
        self.assertIsNone(model.hotswap_state)

    def test_on_with_cache_in_ram_captures_state(self) -> None:
        from musubi_tuner.flux_2_generate_image import prepare_flux2_hotswap_state

        model = _TinyFlux2Stub()
        # Set known weight so we can verify capture
        with torch.no_grad():
            model.to_q.weight.fill_(0.5)

        result = prepare_flux2_hotswap_state(model, self._args(prepare_for_hotswap=True, cache_unmerged_base=True))

        self.assertIsNotNone(result)
        self.assertIs(model.hotswap_state, result)
        self.assertEqual(result.base_dit_paths, [self.dit_path])
        self.assertTrue(result.cache_in_ram)
        self.assertIsNotNone(result.cached_base_sd)
        self.assertIn("to_q.weight", result.cached_base_sd)
        self.assertTrue(torch.equal(result.cached_base_sd["to_q.weight"], torch.full((8, 16), 0.5)))

    def test_on_with_reload_mode_stores_only_paths(self) -> None:
        from musubi_tuner.flux_2_generate_image import prepare_flux2_hotswap_state

        model = _TinyFlux2Stub()
        result = prepare_flux2_hotswap_state(model, self._args(prepare_for_hotswap=True, cache_unmerged_base=False))

        self.assertIsNotNone(result)
        self.assertFalse(result.cache_in_ram)
        self.assertIsNone(result.cached_base_sd)
        self.assertEqual(result.base_dit_paths, [self.dit_path])

    def test_strict_base_hash_threaded_through(self) -> None:
        from musubi_tuner.flux_2_generate_image import prepare_flux2_hotswap_state

        model = _TinyFlux2Stub()
        result = prepare_flux2_hotswap_state(
            model,
            self._args(prepare_for_hotswap=True, hotswap_strict_base_hash=False),
        )

        self.assertFalse(result.strict_base_hash)


class TestNoteFlux2InitialLoras(unittest.TestCase):
    """note_flux2_initial_loras() bookkeeping shape."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tdpath = Path(self.tmpdir.name)
        self.dit_path = _save_tiny_dit(self.tdpath)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _model_with_state(self, args: argparse.Namespace) -> _TinyFlux2Stub:
        from musubi_tuner.flux_2_generate_image import prepare_flux2_hotswap_state

        model = _TinyFlux2Stub()
        prepare_flux2_hotswap_state(model, args)
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
        from musubi_tuner.flux_2_generate_image import note_flux2_initial_loras

        # Hotswap off: model.hotswap_state stays None and note is a no-op
        args = self._args(prepare_for_hotswap=False)
        model = _TinyFlux2Stub()
        model.hotswap_state = None

        note_flux2_initial_loras(model, args)

        self.assertIsNone(model.hotswap_state)

    def test_records_paths_and_multipliers(self) -> None:
        from musubi_tuner.flux_2_generate_image import note_flux2_initial_loras

        args = self._args(lora_weight=["a.safetensors", "b.safetensors"], lora_multiplier=[0.7, 0.3])
        model = self._model_with_state(args)

        note_flux2_initial_loras(model, args)

        self.assertEqual(model.hotswap_state.active_lora_paths, ["a.safetensors", "b.safetensors"])
        self.assertEqual(model.hotswap_state.active_lora_multipliers, [0.7, 0.3])

    def test_pads_missing_multipliers_to_one(self) -> None:
        from musubi_tuner.flux_2_generate_image import note_flux2_initial_loras

        # Two LoRAs, only one multiplier — second should default to 1.0
        args = self._args(lora_weight=["a.safetensors", "b.safetensors"], lora_multiplier=[0.5])
        model = self._model_with_state(args)

        note_flux2_initial_loras(model, args)

        self.assertEqual(model.hotswap_state.active_lora_multipliers, [0.5, 1.0])

    def test_truncates_extra_multipliers(self) -> None:
        from musubi_tuner.flux_2_generate_image import note_flux2_initial_loras

        # Two LoRAs, three multipliers — third should be dropped
        args = self._args(lora_weight=["a.safetensors", "b.safetensors"], lora_multiplier=[0.5, 0.6, 0.7])
        model = self._model_with_state(args)

        note_flux2_initial_loras(model, args)

        self.assertEqual(model.hotswap_state.active_lora_multipliers, [0.5, 0.6])

    def test_none_multipliers_means_all_one(self) -> None:
        from musubi_tuner.flux_2_generate_image import note_flux2_initial_loras

        args = self._args(lora_weight=["a.safetensors", "b.safetensors"], lora_multiplier=None)
        model = self._model_with_state(args)

        note_flux2_initial_loras(model, args)

        self.assertEqual(model.hotswap_state.active_lora_multipliers, [1.0, 1.0])

    def test_missing_state_raises_actionable(self) -> None:
        from musubi_tuner.flux_2_generate_image import note_flux2_initial_loras

        # Hotswap on but caller forgot prepare → state is None
        args = self._args(lora_weight=["a.safetensors"], lora_multiplier=[1.0])
        model = _TinyFlux2Stub()
        model.hotswap_state = None

        with self.assertRaisesRegex(RuntimeError, "hotswap_state missing"):
            note_flux2_initial_loras(model, args)

    def test_no_initial_lora_leaves_active_lists_empty(self) -> None:
        """Sweep-script use case: prepare hotswap but start from base, no initial LoRA.

        The shipped lifecycle in generate() only calls note_flux2_initial_loras() inside
        the `if args.lora_weight is not None and len(args.lora_weight) > 0:` block. This
        test pins the contract that under hotswap=True with no LoRAs, the state is
        prepared and active lists are empty (HotswapState dataclass defaults). A future
        agent moving the note() call outside the if-block would silently call note() with
        empty paths — still correct here, but this test makes the contract explicit.
        """
        from musubi_tuner.flux_2_generate_image import prepare_flux2_hotswap_state

        args = self._args(lora_weight=None, lora_multiplier=None)
        model = _TinyFlux2Stub()

        prepare_flux2_hotswap_state(model, args)

        # State exists but active lists are empty (no merge happened, no note() call)
        self.assertIsNotNone(model.hotswap_state)
        self.assertEqual(model.hotswap_state.active_lora_paths, [])
        self.assertEqual(model.hotswap_state.active_lora_multipliers, [])


# ============================================================================
# Single-path lifecycle ordering tests (prepare BEFORE merge, note AFTER merge)
# ============================================================================


class TestFlux2GenerateLifecycleOrdering(unittest.TestCase):
    """The single-path generate() must call prepare_flux2_hotswap_state() before
    merge_lora_weights(), and note_flux2_initial_loras() after.

    Source-level ordering check: parse the file and verify the relative line
    positions of the three calls in generate(). Cheap and stable; the actual
    runtime invocation is exercised by integration during real-weights smoke.
    """

    def _generate_block_lines(self) -> list[str]:
        import musubi_tuner.flux_2_generate_image as mod

        src_path = Path(mod.__file__)
        text = src_path.read_text()

        # Locate `def generate(` and read until the next top-level `def `
        lines = text.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("def generate("))
        end = next(
            (i for i, ln in enumerate(lines[start + 1 :], start=start + 1) if ln.startswith("def ")),
            len(lines),
        )
        return lines[start:end]

    def _batch_block_lines(self) -> list[str]:
        import musubi_tuner.flux_2_generate_image as mod

        src_path = Path(mod.__file__)
        text = src_path.read_text()
        lines = text.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("def process_batch_prompts("))
        end = next(
            (i for i, ln in enumerate(lines[start + 1 :], start=start + 1) if ln.startswith("def ")),
            len(lines),
        )
        return lines[start:end]

    def test_single_path_prepares_before_merge(self) -> None:
        lines = self._generate_block_lines()
        prepare_idx = next(i for i, ln in enumerate(lines) if "prepare_flux2_hotswap_state(model" in ln)
        merge_idx = next(i for i, ln in enumerate(lines) if "merge_lora_weights(" in ln)
        self.assertLess(prepare_idx, merge_idx)

    def test_single_path_notes_after_merge(self) -> None:
        lines = self._generate_block_lines()
        merge_idx = next(i for i, ln in enumerate(lines) if "merge_lora_weights(" in ln)
        note_idx = next(i for i, ln in enumerate(lines) if "note_flux2_initial_loras(model" in ln)
        self.assertLess(merge_idx, note_idx)

    def test_single_path_passes_standard_lora_only_under_hotswap(self) -> None:
        lines = self._generate_block_lines()
        # Must explicitly pass standard_lora_only based on prepare_for_hotswap
        joined = "\n".join(lines)
        self.assertIn('standard_lora_only=getattr(args, "prepare_for_hotswap", False)', joined)

    def test_batch_path_prepares_before_merge_with_first_prompt_args(self) -> None:
        lines = self._batch_block_lines()
        prepare_idx = next(i for i, ln in enumerate(lines) if "prepare_flux2_hotswap_state(dit_model, first_prompt_args)" in ln)
        merge_idx = next(i for i, ln in enumerate(lines) if "merge_lora_weights(" in ln)
        self.assertLess(prepare_idx, merge_idx)

    def test_batch_path_notes_after_merge(self) -> None:
        lines = self._batch_block_lines()
        merge_idx = next(i for i, ln in enumerate(lines) if "merge_lora_weights(" in ln)
        note_idx = next(i for i, ln in enumerate(lines) if "note_flux2_initial_loras(dit_model, first_prompt_args)" in ln)
        self.assertLess(merge_idx, note_idx)

    def test_batch_path_passes_standard_lora_only_under_hotswap(self) -> None:
        lines = self._batch_block_lines()
        joined = "\n".join(lines)
        self.assertIn('standard_lora_only=getattr(first_prompt_args, "prepare_for_hotswap", False)', joined)


if __name__ == "__main__":
    unittest.main()
