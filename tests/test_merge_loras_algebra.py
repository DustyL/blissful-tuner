"""Regression tests for the offline LoRA merge algebra CLI."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import os
import tempfile
import unittest
import warnings
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools import merge_loras_algebra as mla


def _lora_sd(
    name: str = "lora_unet_block",
    *,
    in_dim: int = 3,
    out_dim: int = 2,
    rank: int = 2,
    alpha: float | None = None,
    down: torch.Tensor | None = None,
    up: torch.Tensor | None = None,
    use_rslora: bool = False,
) -> dict[str, torch.Tensor]:
    down = torch.arange(1, rank * in_dim + 1, dtype=torch.float32).reshape(rank, in_dim) if down is None else down
    up = torch.arange(1, out_dim * rank + 1, dtype=torch.float32).reshape(out_dim, rank) if up is None else up
    sd = {
        f"{name}.lora_down.weight": down.clone(),
        f"{name}.lora_up.weight": up.clone(),
        f"{name}.alpha": torch.tensor(float(rank if alpha is None else alpha)),
    }
    if use_rslora:
        sd["use_rslora_flag"] = torch.tensor(True, dtype=torch.bool)
    return sd


def _dora_lora_sd(
    name: str = "lora_unet_block",
    *,
    in_dim: int = 3,
    out_dim: int = 2,
    rank: int = 2,
    alpha: float | None = None,
    down: torch.Tensor | None = None,
    up: torch.Tensor | None = None,
    magnitude: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build a well-formed DoRA adapter fixture: standard LoRA factors + magnitude + global flag.

    Used by v1.5 #4 tests. Magnitude defaults to ``ones(out_dim)`` so DoRA materialization
    produces predictable values for shape/dtype checks; override for parity tests that need
    specific magnitudes.
    """
    sd = _lora_sd(name=name, in_dim=in_dim, out_dim=out_dim, rank=rank, alpha=alpha, down=down, up=up)
    sd["use_dora_flag"] = torch.tensor(True)
    if magnitude is None:
        magnitude = torch.ones(out_dim, dtype=torch.float32)
    sd[f"{name}.dora_layer.weight"] = magnitude.clone()
    return sd


def _save_sd(tmp: Path, filename: str, sd: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> str:
    path = tmp / filename
    save_file(sd, str(path), metadata=metadata)
    return str(path)


def _explicit_delta(sd: dict[str, torch.Tensor], name: str = "lora_unet_block") -> torch.Tensor:
    down = sd[f"{name}.lora_down.weight"].float()
    up = sd[f"{name}.lora_up.weight"].float()
    rank = down.shape[0]
    alpha = float(sd.get(f"{name}.alpha", torch.tensor(float(rank))).item())
    scale = alpha / (math.sqrt(rank) if sd.get("use_rslora_flag", torch.tensor(False)).item() else rank)
    return (up @ down) * scale


def _output_delta(sd: dict[str, torch.Tensor], name: str = "lora_unet_block") -> torch.Tensor:
    down = sd[f"{name}.lora_down.weight"].float()
    up = sd[f"{name}.lora_up.weight"].float()
    alpha = float(sd[f"{name}.alpha"].item())
    return (up @ down) * (alpha / down.shape[0])


def _config(*args: str) -> mla.MergeConfig:
    return mla.validate_args(mla.parse_args(list(args)))


class TestCliInputPairing(unittest.TestCase):
    def test_repeated_input_path_weight_pairs_parse_correctly(self) -> None:
        config = _config(
            "--method",
            "linear",
            "--input",
            "a.safetensors",
            "0.6",
            "--input",
            "b.safetensors",
            "0.4",
            "--output",
            "out.safetensors",
            "--output_rank",
            "2",
        )

        self.assertEqual([(spec.path, spec.weight) for spec in config.inputs], [("a.safetensors", 0.6), ("b.safetensors", 0.4)])

    def test_missing_weight_in_pair_rejects(self) -> None:
        with self.assertRaises(ValueError):
            _config("--method", "linear", "--input", "a.safetensors", "--preview_spectrum")

    def test_negative_weight_warns_but_proceeds(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = _config("--method", "linear", "--input", "a.safetensors", "-0.5", "--preview_spectrum")

        self.assertEqual(config.inputs[0].weight, -0.5)
        self.assertIn("negative", str(caught[0].message))

    def test_non_finite_weight_rejects(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _config("--method", "linear", "--input", "a.safetensors", "nan", "--preview_spectrum")


class TestMethodValidation(unittest.TestCase):
    def test_ties_requires_density(self) -> None:
        with self.assertRaisesRegex(ValueError, "--density"):
            _config("--method", "ties", "--input", "a.safetensors", "1.0", "--preview_spectrum")

    def test_dare_linear_requires_drop_prob_and_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "--drop_prob"):
            _config("--method", "dare_linear", "--input", "a.safetensors", "1.0", "--preview_spectrum")
        with self.assertRaisesRegex(ValueError, "--seed"):
            _config(
                "--method",
                "dare_linear",
                "--input",
                "a.safetensors",
                "1.0",
                "--drop_prob",
                "0.5",
                "--preview_spectrum",
            )

    def test_dare_ties_requires_all_method_args(self) -> None:
        with self.assertRaisesRegex(ValueError, "--density"):
            _config(
                "--method",
                "dare_ties",
                "--input",
                "a.safetensors",
                "1.0",
                "--drop_prob",
                "0.5",
                "--seed",
                "1",
                "--preview_spectrum",
            )
        with self.assertRaisesRegex(ValueError, "--drop_prob"):
            _config(
                "--method", "dare_ties", "--input", "a.safetensors", "1.0", "--density", "0.5", "--seed", "1", "--preview_spectrum"
            )
        with self.assertRaisesRegex(ValueError, "--seed"):
            _config(
                "--method",
                "dare_ties",
                "--input",
                "a.safetensors",
                "1.0",
                "--density",
                "0.5",
                "--drop_prob",
                "0.5",
                "--preview_spectrum",
            )

    def test_linear_does_not_require_method_args(self) -> None:
        config = _config("--method", "linear", "--input", "a.safetensors", "1.0", "--preview_spectrum")

        self.assertEqual(config.method, "linear")

    def test_linear_rejects_irrelevant_method_args(self) -> None:
        for flag, value in (("--density", "2.0"), ("--drop_prob", "0.5"), ("--seed", "123")):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, f"{flag} is not used"):
                _config("--method", "linear", "--input", "a.safetensors", "1.0", flag, value, "--preview_spectrum")

    def test_ties_rejects_irrelevant_dare_args(self) -> None:
        for flag, value in (("--drop_prob", "0.5"), ("--seed", "123")):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, f"{flag} is not used"):
                _config(
                    "--method", "ties", "--input", "a.safetensors", "1.0", "--density", "0.5", flag, value, "--preview_spectrum"
                )

    def test_dare_linear_rejects_irrelevant_density(self) -> None:
        with self.assertRaisesRegex(ValueError, "--density is not used"):
            _config(
                "--method",
                "dare_linear",
                "--input",
                "a.safetensors",
                "1.0",
                "--density",
                "2.0",
                "--drop_prob",
                "0.5",
                "--seed",
                "123",
                "--preview_spectrum",
            )

    def test_dare_ties_accepts_all_method_args(self) -> None:
        config = _config(
            "--method",
            "dare_ties",
            "--input",
            "a.safetensors",
            "1.0",
            "--density",
            "0.5",
            "--drop_prob",
            "0.5",
            "--seed",
            "123",
            "--preview_spectrum",
        )

        self.assertEqual(config.method, "dare_ties")
        self.assertEqual(config.density, 0.5)
        self.assertEqual(config.drop_prob, 0.5)
        self.assertEqual(config.seed, 123)

    def test_output_requires_output_rank_but_preview_does_not(self) -> None:
        with self.assertRaisesRegex(ValueError, "--output_rank"):
            _config("--method", "linear", "--input", "a.safetensors", "1.0", "--output", "out.safetensors")

        config = _config("--method", "linear", "--input", "a.safetensors", "1.0", "--preview_spectrum")
        self.assertIsNone(config.output_rank)


class TestAdapterFormatRejection(unittest.TestCase):
    def test_partial_dora_global_flag_without_magnitude_rejects_at_load(self) -> None:
        # v1.5 #4 decision #15: global use_dora_flag=True means EVERY module needs a
        # matching .dora_layer.weight. The existing fixture (global flag, no magnitudes)
        # now hits the partial-DoRA validation in _collect_modules instead of the
        # legacy "DoRA family rejected" wall.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_sd(Path(tmpdir), "dora.safetensors", _lora_sd() | {"use_dora_flag": torch.tensor(True)})

            with self.assertRaises(ValueError) as cm:
                mla.load_adapter(path, 1.0)

        self.assertIn("use_dora_flag but missing", str(cm.exception))
        self.assertIn("lora_unet_block.dora_layer.weight", str(cm.exception))
        self.assertIn("partial-DoRA", str(cm.exception))
        self.assertIn("decision #15", str(cm.exception))

    def test_dora_cli_rejection_has_no_traceback(self) -> None:
        # CLI no-traceback contract preserved: a DoRA-input rejection (any kind) must
        # exit through SystemExit with the actionable message, not a Python traceback.
        # Fixture is well-formed DoRA (passes partial-DoRA validation) so we hit the
        # post-load preflight "DoRA requires --base_dit" reject — the rejection layer
        # most users will encounter in practice.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_sd(
                Path(tmpdir),
                "dora.safetensors",
                _lora_sd() | {
                    "use_dora_flag": torch.tensor(True),
                    "lora_unet_block.dora_layer.weight": torch.zeros(2),
                },
            )

            with self.assertRaises(SystemExit) as cm:
                mla.main(["--method", "linear", "--input", path, "1.0", "--preview_spectrum"])

        self.assertIn("--base_dit", str(cm.exception))
        self.assertIn("dora.safetensors", str(cm.exception))
        self.assertNotIn("Traceback", str(cm.exception))

    def test_per_module_dora_flag_without_magnitude_rejects_at_load(self) -> None:
        # _detect_dora's per-module branch detects per-module *.use_dora_flag, and
        # _collect_modules' partial-DoRA validation rejects when the matching
        # magnitude key is missing. The original step-2-corrections test for this
        # now exercises the partial-DoRA reject (decision #15) instead of the
        # legacy "DoRA family rejected" wall.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_sd(
                Path(tmpdir),
                "per_module_dora.safetensors",
                _lora_sd() | {"lora_unet_block.use_dora_flag": torch.tensor(True)},
            )
            with self.assertRaises(ValueError) as cm:
                mla.load_adapter(path, 1.0)
        self.assertIn("use_dora_flag but missing", str(cm.exception))
        self.assertIn("lora_unet_block.dora_layer.weight", str(cm.exception))
        self.assertIn("decision #15", str(cm.exception))

    def test_well_formed_dora_input_rejected_at_preflight_without_base_dit(self) -> None:
        # v1.5 #4 decision #1: DoRA input passes load_adapter (no longer rejected at
        # the family wall, has all required magnitudes) but is rejected at post-load
        # preflight when --base_dit is missing. This is the contract a real DoRA user
        # will encounter when they forget to pass --base_dit.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_sd(
                Path(tmpdir),
                "well_formed_dora.safetensors",
                _lora_sd() | {
                    "use_dora_flag": torch.tensor(True),
                    "lora_unet_block.dora_layer.weight": torch.zeros(2),
                },
            )
            # load_adapter itself succeeds — DoRA is now allowed through.
            adapter = mla.load_adapter(path, 1.0)
            self.assertTrue(adapter.is_dora)
            self.assertTrue(adapter.modules["lora_unet_block"].is_dora)
            self.assertTrue(adapter.modules["lora_unet_block"].has_dora_magnitude)

            # The rejection fires at merge_adapters' post-load preflight.
            config = _config("--method", "linear", "--input", path, "1.0", "--preview_spectrum")
            with self.assertRaisesRegex(ValueError, r"DoRA input.*require --base_dit"):
                mla.run(config)

    def test_magnitude_only_module_treated_as_dora(self) -> None:
        # Finding 1 fix: a `.dora_layer.weight` key alone (no use_dora_flag anywhere)
        # marks the module as DoRA — matches _detect_dora's broader detection. Without
        # this, the adapter would be tagged DoRA at the adapter level (requiring
        # --base_dit) but silently merged as standard LoRA at the module level (the
        # magnitude vector ignored). Locks the post-fix unified detection.
        with tempfile.TemporaryDirectory() as tmpdir:
            sd = _lora_sd() | {"lora_unet_block.dora_layer.weight": torch.ones(2, dtype=torch.float32)}
            path = _save_sd(Path(tmpdir), "magnitude_only.safetensors", sd)
            adapter = mla.load_adapter(path, 1.0)

        # Adapter is tagged DoRA (via _detect_dora).
        self.assertTrue(adapter.is_dora)
        # AND the module itself is tagged DoRA (no longer split-brain).
        self.assertTrue(adapter.modules["lora_unet_block"].is_dora)
        self.assertTrue(adapter.modules["lora_unet_block"].has_dora_magnitude)

    def test_magnitude_only_module_rejects_at_preflight_without_base_dit(self) -> None:
        # End-to-end consequence of finding 1 fix: a magnitude-only adapter must
        # require --base_dit through the same preflight path as a flag-marked DoRA
        # adapter. Pre-fix, this would silently merge as standard LoRA.
        with tempfile.TemporaryDirectory() as tmpdir:
            sd = _lora_sd() | {"lora_unet_block.dora_layer.weight": torch.ones(2, dtype=torch.float32)}
            path = _save_sd(Path(tmpdir), "magnitude_only.safetensors", sd)
            config = _config("--method", "linear", "--input", path, "1.0", "--preview_spectrum")
            with self.assertRaisesRegex(ValueError, r"DoRA input.*require --base_dit"):
                mla.run(config)

    def test_orphan_dora_magnitude_without_lora_module_rejects(self) -> None:
        # Finding 1 second-half: a `.dora_layer.weight` key with no matching LoRA
        # module would tag the adapter DoRA via _detect_dora but never reach the
        # per-module DoRA branch — silently leaving the magnitude unused.
        with tempfile.TemporaryDirectory() as tmpdir:
            sd = _lora_sd() | {"orphan_module.dora_layer.weight": torch.ones(2, dtype=torch.float32)}
            path = _save_sd(Path(tmpdir), "orphan_mag.safetensors", sd)
            with self.assertRaisesRegex(ValueError, r"orphan DoRA magnitude key.*orphan_module\.dora_layer\.weight"):
                mla.load_adapter(path, 1.0)

    def test_dora_magnitude_shape_mismatch_rejects_with_actionable_message(self) -> None:
        # Finding 2 fix: malformed magnitude tensor (wrong row-norm length) must reject
        # at load with an actionable ValueError, not escape as a torch RuntimeError
        # mid-materialization. The default _lora_sd has out_dim=2, so magnitude must
        # be shape (2,); we provide (3,) to trigger the check.
        with tempfile.TemporaryDirectory() as tmpdir:
            sd = _lora_sd() | {
                "use_dora_flag": torch.tensor(True),
                "lora_unet_block.dora_layer.weight": torch.ones(3, dtype=torch.float32),
            }
            path = _save_sd(Path(tmpdir), "bad_mag.safetensors", sd)
            with self.assertRaisesRegex(ValueError, r"DoRA magnitude shape mismatch.*\(2,\).*\(3,\)"):
                mla.load_adapter(path, 1.0)

    def test_dora_magnitude_shape_mismatch_cli_no_traceback(self) -> None:
        # Finding 2 CLI contract: malformed magnitude → SystemExit with the actionable
        # message, not a Python traceback. Locks the no-traceback contract for the
        # newly-added shape validation.
        with tempfile.TemporaryDirectory() as tmpdir:
            sd = _lora_sd() | {
                "use_dora_flag": torch.tensor(True),
                "lora_unet_block.dora_layer.weight": torch.ones(3, dtype=torch.float32),
            }
            path = _save_sd(Path(tmpdir), "bad_mag.safetensors", sd)
            with self.assertRaises(SystemExit) as cm:
                mla.main(["--method", "linear", "--input", path, "1.0", "--preview_spectrum"])
        self.assertIn("DoRA magnitude shape mismatch", str(cm.exception))
        self.assertNotIn("Traceback", str(cm.exception))

    def test_conv2d_dora_input_rejected_at_load(self) -> None:
        # Decision #8: Conv2d DoRA pre-decided as preflight reject, mirroring the
        # production helper's NotImplementedError. Fires from _collect_modules during
        # load_adapter, not later in the pipeline.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_sd(
                Path(tmpdir),
                "conv_dora.safetensors",
                _lora_sd(
                    name="lora_unet_conv",
                    rank=2, in_dim=3, out_dim=2,
                    down=torch.zeros(2, 3, 1, 1),
                    up=torch.zeros(2, 2, 1, 1),
                ) | {
                    "use_dora_flag": torch.tensor(True),
                    "lora_unet_conv.dora_layer.weight": torch.zeros(2),
                },
            )
            with self.assertRaisesRegex(ValueError, r"Conv2d DoRA.*decision #8"):
                mla.load_adapter(path, 1.0)

    def test_false_per_module_dora_flag_without_magnitude_does_not_mark_dora(self) -> None:
        # Inverse boundary: per-module *.use_dora_flag=False (explicit, no other
        # DoRA markers) must NOT trigger DoRA detection. _bool_tensor_value honors
        # the actual boolean value, so a False flag is just unused metadata.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_sd(
                Path(tmpdir),
                "false_flag.safetensors",
                _lora_sd() | {"lora_unet_block.use_dora_flag": torch.tensor(False)},
            )
            adapter = mla.load_adapter(path, 1.0)
        self.assertIn("lora_unet_block", adapter.modules)

    def test_loha_lokr_hybrid_and_unknown_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixtures = {
                "loha.safetensors": {"x.hada_w1_a": torch.zeros(1, 1), "x.hada_w2_a": torch.zeros(1, 1)},
                "lokr.safetensors": {"x.lokr_w1": torch.zeros(1, 1), "x.lokr_w2": torch.zeros(1, 1)},
                "hybrid.safetensors": _lora_sd() | {"x.lokr_w1": torch.zeros(1, 1)},
                "unknown.safetensors": {"x.weight": torch.zeros(1, 1)},
            }
            for filename, sd in fixtures.items():
                path = _save_sd(tmp, filename, sd)
                with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "standard LoRA only"):
                    mla.load_adapter(path, 1.0)

    def test_standard_and_rslora_inputs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            standard = mla.load_adapter(_save_sd(tmp, "standard.safetensors", _lora_sd()), 1.0)
            rslora = mla.load_adapter(_save_sd(tmp, "rslora.safetensors", _lora_sd(use_rslora=True)), 1.0)

        self.assertIn("lora_unet_block", standard.modules)
        self.assertTrue(rslora.modules["lora_unet_block"].use_rslora)


class TestMaterializedDeltaAlgebra(unittest.TestCase):
    def test_linear_two_inputs_matches_explicit_full_delta_math(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd_a = _lora_sd(down=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), up=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), in_dim=2)
            sd_b = _lora_sd(down=torch.tensor([[2.0, 0.0], [0.0, 2.0]]), up=torch.tensor([[0.5, 1.0], [1.5, 2.0]]), in_dim=2)
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "a.safetensors", sd_a),
                "0.25",
                "--input",
                _save_sd(tmp, "b.safetensors", sd_b),
                "0.75",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )

            mla.run(config)
            merged = load_file(str(out))

        expected = 0.25 * _explicit_delta(sd_a) + 0.75 * _explicit_delta(sd_b)
        self.assertTrue(torch.allclose(_output_delta(merged), expected, atol=1e-5))

    def test_ties_sign_election_dominates_minority_signs(self) -> None:
        deltas = [torch.tensor([[10.0, -4.0]]), torch.tensor([[5.0, 3.0]]), torch.tensor([[-1.0, 2.0]])]

        merged = mla.combine_deltas("ties", deltas, [1.0, 1.0, 1.0], density=1.0)

        # Column 0 elects positive sign, column 1 elects positive sign.
        self.assertTrue(torch.allclose(merged, torch.tensor([[7.5, 2.5]])))

    def test_dare_linear_drop_prob_zero_equals_linear(self) -> None:
        deltas = [torch.randn(3, 4), torch.randn(3, 4)]
        linear = mla.combine_deltas("linear", deltas, [0.25, 0.75])
        dare = mla.combine_deltas(
            "dare_linear",
            deltas,
            [0.25, 0.75],
            drop_prob=0.0,
            generator=torch.Generator().manual_seed(123),
        )

        self.assertTrue(torch.allclose(dare, linear))

    def test_dare_seed_reproducible(self) -> None:
        deltas = [torch.arange(12, dtype=torch.float32).reshape(3, 4), torch.ones(3, 4)]

        first = mla.combine_deltas(
            "dare_linear",
            deltas,
            [1.0, 1.0],
            drop_prob=0.5,
            generator=torch.Generator().manual_seed(123),
        )
        second = mla.combine_deltas(
            "dare_linear",
            deltas,
            [1.0, 1.0],
            drop_prob=0.5,
            generator=torch.Generator().manual_seed(123),
        )
        third = mla.combine_deltas(
            "dare_linear",
            deltas,
            [1.0, 1.0],
            drop_prob=0.5,
            generator=torch.Generator().manual_seed(124),
        )

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

    def test_dare_output_invariant_under_input_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd_a = _lora_sd(
                down=torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]),
                up=torch.tensor([[2.0, 0.5], [1.0, 3.0]]),
            )
            sd_b = _lora_sd(
                down=torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0]]),
                up=torch.tensor([[1.5, 0.25], [2.0, 1.0]]),
            )
            path_a = _save_sd(tmp, "a.safetensors", sd_a)
            path_b = _save_sd(tmp, "b.safetensors", sd_b)
            for method in ("dare_linear", "dare_ties"):
                with self.subTest(method=method):
                    out_ab = tmp / f"{method}_ab.safetensors"
                    out_ba = tmp / f"{method}_ba.safetensors"
                    method_args = ["--density", "1.0"] if method == "dare_ties" else []

                    mla.run(
                        _config(
                            "--method",
                            method,
                            "--input",
                            path_a,
                            "1.0",
                            "--input",
                            path_b,
                            "1.0",
                            *method_args,
                            "--drop_prob",
                            "0.5",
                            "--seed",
                            "123",
                            "--output",
                            str(out_ab),
                            "--output_rank",
                            "2",
                        )
                    )
                    mla.run(
                        _config(
                            "--method",
                            method,
                            "--input",
                            path_b,
                            "1.0",
                            "--input",
                            path_a,
                            "1.0",
                            *method_args,
                            "--drop_prob",
                            "0.5",
                            "--seed",
                            "123",
                            "--output",
                            str(out_ba),
                            "--output_rank",
                            "2",
                        )
                    )
                    merged_ab = load_file(str(out_ab))
                    merged_ba = load_file(str(out_ba))

                    self.assertTrue(torch.allclose(_output_delta(merged_ab), _output_delta(merged_ba), atol=1e-5))

    def test_cross_rank_inputs_merge_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd_a = _lora_sd(rank=1, down=torch.tensor([[1.0, 2.0, 3.0]]), up=torch.tensor([[2.0], [3.0]]))
            sd_b = _lora_sd(rank=2)
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "a.safetensors", sd_a),
                "1.0",
                "--input",
                _save_sd(tmp, "b.safetensors", sd_b),
                "1.0",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )

            mla.run(config)
            merged = load_file(str(out))

        expected = _explicit_delta(sd_a) + _explicit_delta(sd_b)
        self.assertTrue(torch.allclose(_output_delta(merged), expected, atol=1e-5))


class TestMissingModulesAndShapeMismatch(unittest.TestCase):
    def test_missing_module_contributes_zero_and_output_is_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd_a = _lora_sd(name="lora_unet_a")
            sd_b = _lora_sd(name="lora_unet_b")
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "a.safetensors", sd_a),
                "1.0",
                "--input",
                _save_sd(tmp, "b.safetensors", sd_b),
                "1.0",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )

            mla.run(config)
            merged = load_file(str(out))

        self.assertIn("lora_unet_a.lora_down.weight", merged)
        self.assertIn("lora_unet_b.lora_down.weight", merged)
        self.assertTrue(torch.allclose(_output_delta(merged, "lora_unet_a"), _explicit_delta(sd_a, "lora_unet_a"), atol=1e-5))

    def test_same_module_different_shape_hard_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd_a = _lora_sd(in_dim=3, out_dim=2)
            sd_b = _lora_sd(in_dim=4, out_dim=2)
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "a.safetensors", sd_a),
                "1.0",
                "--input",
                _save_sd(tmp, "b.safetensors", sd_b),
                "1.0",
                "--preview_spectrum",
            )

            with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                mla.merge_adapters(config)

    def test_output_rank_too_large_rejects_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = _save_sd(tmp, "a.safetensors", _lora_sd(in_dim=3, out_dim=2))
            config = _config(
                "--method", "linear", "--input", source, "1.0", "--output", str(tmp / "out.safetensors"), "--output_rank", "3"
            )

            with (
                patch.object(mla, "materialize_module_delta", side_effect=AssertionError("materialization should not run")),
                self.assertRaisesRegex(ValueError, "--output_rank 3 is too large"),
            ):
                mla.merge_adapters(config)


class TestAllZeroModuleSkip(unittest.TestCase):
    def test_exact_zero_merged_module_omitted_but_tiny_nonzero_kept(self) -> None:
        zero_sd = _lora_sd(name="lora_unet_zero", down=torch.zeros(2, 3), up=torch.zeros(2, 2))
        tiny_sd = _lora_sd(name="lora_unet_tiny", down=torch.full((2, 3), 1e-10), up=torch.eye(2))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "zero.safetensors", zero_sd | tiny_sd),
                "1.0",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )

            mla.run(config)
            merged = load_file(str(out))

        self.assertNotIn("lora_unet_zero.lora_down.weight", merged)
        self.assertIn("lora_unet_tiny.lora_down.weight", merged)

    def test_all_zero_output_prints_warning(self) -> None:
        zero_sd = _lora_sd(down=torch.zeros(2, 3), up=torch.zeros(2, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "zero.safetensors", zero_sd),
                "1.0",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )
            buf = io.StringIO()

            with contextlib.redirect_stdout(buf):
                mla.run(config)

            merged = load_file(str(out))

        self.assertEqual(merged, {})
        self.assertIn("Warning: all merged modules were exact-zero", buf.getvalue())


class TestNonFiniteMergedDelta(unittest.TestCase):
    def test_non_finite_delta_cli_error_has_no_traceback(self) -> None:
        bad_sd = _lora_sd(down=torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0]]), up=torch.eye(2))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = _save_sd(tmp, "nan_lora.safetensors", bad_sd)
            out = tmp / "out.safetensors"

            with self.assertRaises(SystemExit) as cm:
                mla.main(["--method", "linear", "--input", source, "1.0", "--output", str(out), "--output_rank", "2"])

        message = str(cm.exception)
        self.assertIn("Non-finite merged delta", message)
        self.assertIn("lora_unet_block", message)
        self.assertNotIn("Traceback", message)


class TestPruneThreshold(unittest.TestCase):
    """v1.5 #1: --prune_threshold for fuzzy-zero module skip.

    See `docs/plans/2026-05-04-peft-tier2-merge-algebra.md` "v1.5 #1" section
    for the locked decision contract.
    """

    def _tiny_sd(self, name: str, magnitude: float) -> dict[str, torch.Tensor]:
        # Construct a LoRA whose merged delta has abs().max() == magnitude.
        # With down = magnitude * eye and up = eye and alpha == rank (scale=1),
        # delta = up @ down = magnitude * eye, so abs().max() = magnitude.
        down = torch.eye(2) * magnitude
        up = torch.eye(2)
        return _lora_sd(name=name, down=down, up=up)

    def test_default_preserves_exact_zero_behavior(self) -> None:
        # At default 0.0, exact-zero modules skipped, near-zero kept (v1 behavior)
        zero_sd = _lora_sd(name="lora_unet_zero", down=torch.zeros(2, 3), up=torch.zeros(2, 2))
        tiny_sd = _lora_sd(name="lora_unet_tiny", down=torch.full((2, 3), 1e-10), up=torch.eye(2))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "src.safetensors", zero_sd | tiny_sd),
                "1.0",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )
            mla.run(config)
            merged = load_file(str(out))

        self.assertNotIn("lora_unet_zero.lora_down.weight", merged)
        self.assertIn("lora_unet_tiny.lora_down.weight", merged)
        self.assertEqual(config.prune_threshold, 0.0)

    def test_skips_below_magnitude(self) -> None:
        # Module with abs().max() == 1e-5 skipped when threshold = 1e-4
        below_sd = self._tiny_sd("lora_unet_below", 1e-5)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", below_sd),
                    "1.0",
                    "--prune_threshold",
                    "1e-4",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                )
            )
            merged = load_file(str(out))

        self.assertNotIn("lora_unet_below.lora_down.weight", merged)

    def test_keeps_above_magnitude(self) -> None:
        # Module with abs().max() == 1e-3 kept when threshold = 1e-4
        above_sd = self._tiny_sd("lora_unet_above", 1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", above_sd),
                    "1.0",
                    "--prune_threshold",
                    "1e-4",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                )
            )
            merged = load_file(str(out))

        self.assertIn("lora_unet_above.lora_down.weight", merged)

    def test_skips_equal_magnitude(self) -> None:
        """Pin the `<=` boundary semantics: skip when abs().max() exactly equals threshold.

        A future refactor that swapped `<=` for `<` would silently change behavior;
        without this test, both directions would still pass test_skips_below_magnitude
        and test_keeps_above_magnitude. This test fails loudly if `<=` becomes `<`.
        """
        threshold = 1e-4
        equal_sd = self._tiny_sd("lora_unet_equal", threshold)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", equal_sd),
                    "1.0",
                    "--prune_threshold",
                    str(threshold),
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                )
            )
            merged = load_file(str(out))

        # |delta|.max() == threshold → contract says skip (`<=` boundary)
        self.assertNotIn("lora_unet_equal.lora_down.weight", merged)

    def test_negative_rejects(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative and finite"):
            _config("--method", "linear", "--input", "a.safetensors", "1.0", "--prune_threshold", "-0.1", "--preview_spectrum")

    def test_nan_rejects(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative and finite"):
            _config("--method", "linear", "--input", "a.safetensors", "1.0", "--prune_threshold", "nan", "--preview_spectrum")

    def test_inf_rejects(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative and finite"):
            _config("--method", "linear", "--input", "a.safetensors", "1.0", "--prune_threshold", "inf", "--preview_spectrum")

    def test_excludes_pruned_from_spectrum(self) -> None:
        # Pruned module must not contribute to spectrum_energy aggregate or per_module_energy
        below_sd = self._tiny_sd("lora_unet_pruned", 1e-5)
        kept_sd = self._tiny_sd("lora_unet_kept", 1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "src.safetensors", below_sd | kept_sd),
                "1.0",
                "--prune_threshold",
                "1e-4",
                "--preview_spectrum",
            )
            result = mla.merge_adapters(config)

        # Both modules counted in modules_processed (incremented before the prune check).
        # Spectrum stats receive only the kept module — pruned module excluded from
        # spectrum_energy aggregate AND per_module_energy, mirroring the v1
        # zero-delta-preview-stat fix.
        self.assertEqual(result.modules_processed, 2)
        self.assertEqual(len(result.spectrum_energy[mla.SPECTRUM_RANKS[0]]), 1)
        self.assertNotIn("lora_unet_pruned", result.per_module_energy)
        self.assertIn("lora_unet_kept", result.per_module_energy)

    def test_metadata_recorded_at_default(self) -> None:
        # ss_merge_prune_threshold persisted as "0.0" even when flag not passed
        sd = self._tiny_sd("lora_unet_x", 1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", sd),
                    "1.0",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                )
            )
            with safe_open(str(out), framework="pt") as f:
                metadata = f.metadata()

        self.assertIn("ss_merge_prune_threshold", metadata)
        self.assertEqual(metadata["ss_merge_prune_threshold"], "0.0")

    def test_metadata_recorded_at_user_value(self) -> None:
        # ss_merge_prune_threshold reflects user-supplied value
        sd = self._tiny_sd("lora_unet_x", 1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", sd),
                    "1.0",
                    "--prune_threshold",
                    "0.001",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                )
            )
            with safe_open(str(out), framework="pt") as f:
                metadata = f.metadata()

        self.assertEqual(metadata["ss_merge_prune_threshold"], "0.001")

    def test_all_pruned_warning_includes_threshold_value(self) -> None:
        # When all modules are pruned, warning message names the threshold value
        below_sd = self._tiny_sd("lora_unet_x", 1e-5)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "src.safetensors", below_sd),
                "1.0",
                "--prune_threshold",
                "1e-4",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mla.run(config)

        text = buf.getvalue()
        self.assertIn("exact-zero or below --prune_threshold", text)
        self.assertIn("0.0001", text)  # threshold value rendered

    def test_all_pruned_warning_default_zero_omits_threshold_phrase(self) -> None:
        # At default 0.0, original "exact-zero" wording preserved (no threshold mention)
        zero_sd = _lora_sd(name="lora_unet_x", down=torch.zeros(2, 3), up=torch.zeros(2, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "src.safetensors", zero_sd),
                "1.0",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mla.run(config)

        text = buf.getvalue()
        self.assertIn("all merged modules were exact-zero", text)
        self.assertNotIn("--prune_threshold", text)


class TestRsLoRAInputStandardOutput(unittest.TestCase):
    def test_rslora_input_scale_absorbed_and_output_standard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd = _lora_sd(use_rslora=True, alpha=math.sqrt(2.0))
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "linear",
                "--input",
                _save_sd(tmp, "rslora.safetensors", sd),
                "1.0",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )

            mla.run(config)
            merged = load_file(str(out))

        self.assertTrue(torch.allclose(_output_delta(merged), _explicit_delta(sd), atol=1e-5))
        self.assertNotIn("use_rslora_flag", merged)
        self.assertEqual(float(merged["lora_unet_block.alpha"].item()), 2.0)


def _rslora_output_delta(sd: dict[str, torch.Tensor], name: str = "lora_unet_block") -> torch.Tensor:
    """Reconstruct delta from rsLoRA-shaped output (alpha / sqrt(rank) scale).

    Counterpart to _output_delta() which assumes standard alpha/rank scaling.
    """
    down = sd[f"{name}.lora_down.weight"].float()
    up = sd[f"{name}.lora_up.weight"].float()
    alpha = float(sd[f"{name}.alpha"].item())
    return (up @ down) * (alpha / math.sqrt(down.shape[0]))


class TestOutputUseRsLoRA(unittest.TestCase):
    """v1.5 #2: --output_use_rslora for rsLoRA-shaped output.

    See `docs/plans/2026-05-04-peft-tier2-merge-algebra.md` "v1.5 #2" section
    for the locked decision contract (12 forks).
    """

    def test_rejects_without_output(self) -> None:
        # Self-application of the "accepted-but-ignored" lens: --output_use_rslora is
        # output-only, so passing it under --preview_spectrum has no meaning.
        with self.assertRaisesRegex(ValueError, r"--output_use_rslora requires --output"):
            _config(
                "--method",
                "linear",
                "--input",
                "a.safetensors",
                "1.0",
                "--output_use_rslora",
                "--preview_spectrum",
            )

    def test_standard_input_to_rslora_output_writes_flag(self) -> None:
        # Output safetensors gains a global use_rslora_flag=True boolean tensor
        sd = _lora_sd()  # standard input (no use_rslora_flag)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", sd),
                    "1.0",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                    "--output_use_rslora",
                )
            )
            merged = load_file(str(out))

        self.assertIn("use_rslora_flag", merged)
        self.assertTrue(bool(merged["use_rslora_flag"].item()))
        self.assertEqual(merged["use_rslora_flag"].dtype, torch.bool)

    def test_standard_input_to_rslora_output_reconstructs_delta(self) -> None:
        # Reading the rsLoRA-shaped output with alpha/sqrt(rank) scale must recover
        # the same materialized delta the standard input encoded with alpha/rank scale.
        sd = _lora_sd()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", sd),
                    "1.0",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                    "--output_use_rslora",
                )
            )
            merged = load_file(str(out))

        self.assertTrue(torch.allclose(_rslora_output_delta(merged), _explicit_delta(sd), atol=1e-5))

    def test_rslora_input_to_rslora_output_round_trip(self) -> None:
        # rsLoRA input → rsLoRA output produces equivalent materialized delta
        sd = _lora_sd(use_rslora=True, alpha=math.sqrt(2.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "rslora.safetensors", sd),
                    "1.0",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                    "--output_use_rslora",
                )
            )
            merged = load_file(str(out))

        self.assertIn("use_rslora_flag", merged)
        self.assertTrue(torch.allclose(_rslora_output_delta(merged), _explicit_delta(sd), atol=1e-5))

    def test_output_alpha_override_in_rslora_preserves_delta(self) -> None:
        # User-supplied --output_alpha in rsLoRA mode still reconstructs the target
        # delta correctly (factors absorb whatever scale the user picked).
        sd = _lora_sd()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", sd),
                    "1.0",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                    "--output_alpha",
                    "1.0",  # explicit override
                    "--output_use_rslora",
                )
            )
            merged = load_file(str(out))

        self.assertEqual(float(merged["lora_unet_block.alpha"].item()), 1.0)
        self.assertTrue(torch.allclose(_rslora_output_delta(merged), _explicit_delta(sd), atol=1e-5))

    def test_standard_output_records_false_metadata_and_omits_flag(self) -> None:
        # Default off: metadata records "false", output safetensors has no use_rslora_flag
        sd = _lora_sd()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "src.safetensors", sd),
                    "1.0",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                )
            )
            merged = load_file(str(out))
            with safe_open(str(out), framework="pt") as f:
                metadata = f.metadata()

        self.assertNotIn("use_rslora_flag", merged)
        self.assertEqual(metadata["ss_merge_output_use_rslora"], "false")

    def test_all_pruned_with_rslora_records_metadata_but_no_lone_flag(self) -> None:
        # When all modules pruned, metadata says "true" but output doesn't write
        # a lone use_rslora_flag tensor — artifact hygiene per locked decision #5.
        zero_sd = _lora_sd(name="lora_unet_zero", down=torch.zeros(2, 3), up=torch.zeros(2, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out.safetensors"
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    _save_sd(tmp, "zero.safetensors", zero_sd),
                    "1.0",
                    "--output",
                    str(out),
                    "--output_rank",
                    "2",
                    "--output_use_rslora",
                )
            )
            merged = load_file(str(out))
            with safe_open(str(out), framework="pt") as f:
                metadata = f.metadata()

        # Metadata reflects the user's intent
        self.assertEqual(metadata["ss_merge_output_use_rslora"], "true")
        # But no lone flag tensor in an otherwise-empty safetensors
        self.assertEqual(merged, {})

    def test_parser_help_includes_rslora_flag(self) -> None:
        # Help string mentions both the math (alpha / sqrt(rank)) and the loader requirement
        parser = mla.build_parser()
        help_text = parser.format_help()
        self.assertIn("--output_use_rslora", help_text)
        self.assertIn("alpha / sqrt(rank)", help_text)
        self.assertIn("downstream loaders", help_text)


class TestMetadata(unittest.TestCase):
    def test_metadata_includes_required_provenance_keys_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = _save_sd(tmp, "a.safetensors", _lora_sd())
            out = tmp / "out.safetensors"
            config = _config("--method", "linear", "--input", source, "0.5", "--output", str(out), "--output_rank", "2")

            mla.run(config)
            with safe_open(str(out), framework="pt") as f:
                metadata = f.metadata()

        for key in [
            "ss_merge_tool",
            "ss_merge_tool_version",
            "ss_merge_method",
            "ss_merge_output_format",
            "ss_merge_output_rank",
            "ss_merge_output_alpha",
            "ss_merge_output_dtype",
            "ss_merge_inputs",
            "ss_merge_input_count",
            "ss_merge_match_semantics",
            "ss_merge_recompression",
            "ss_merge_rejects_dora",
        ]:
            self.assertIn(key, metadata)
        inputs = json.loads(metadata["ss_merge_inputs"])
        self.assertEqual(inputs[0]["basename"], "a.safetensors")
        self.assertEqual(inputs[0]["weight"], 0.5)
        self.assertEqual(inputs[0]["rank"], 2)
        self.assertEqual(inputs[0]["alpha"], 2.0)
        # Per-input format tag (v1.5 #4 decision #10) — standard LoRA inputs always "lora".
        self.assertEqual(inputs[0]["format"], "lora")
        self.assertEqual(metadata["ss_merge_match_semantics"], mla.MATCH_SEMANTICS)
        self.assertEqual(metadata["ss_merge_recompression"], mla.RECOMPRESSION_SEMANTICS)
        # Backward-compat lock (v1.5 #4 finding-3 follow-up): standard-LoRA-only runs
        # still record "true" for downstream readers that grep on this key. Only DoRA
        # runs flip to "false". The companion DoRA-side test is
        # test_dora_metadata_does_not_claim_rejects_dora_when_dora_input_present.
        self.assertEqual(metadata["ss_merge_rejects_dora"], "true")
        self.assertEqual(metadata["ss_merge_dora_inputs"], "false")

    def test_metadata_method_specific_args_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = _save_sd(tmp, "a.safetensors", _lora_sd())
            out = tmp / "out.safetensors"
            config = _config(
                "--method",
                "dare_ties",
                "--input",
                source,
                "1.0",
                "--density",
                "1.0",
                "--drop_prob",
                "0.0",
                "--seed",
                "42",
                "--output",
                str(out),
                "--output_rank",
                "2",
            )

            mla.run(config)
            with safe_open(str(out), framework="pt") as f:
                metadata = f.metadata()

        self.assertEqual(metadata["ss_merge_density"], "1.0")
        self.assertEqual(metadata["ss_merge_drop_prob"], "0.0")
        self.assertEqual(metadata["ss_merge_seed"], "42")


class TestMetadataPathPrivacy(unittest.TestCase):
    def test_metadata_inputs_contains_basename_not_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nested = tmp / "private" / "path"
            nested.mkdir(parents=True)
            source = _save_sd(nested, "secret_lora.safetensors", _lora_sd())
            out = tmp / "out.safetensors"
            config = _config("--method", "linear", "--input", source, "1.0", "--output", str(out), "--output_rank", "2")

            mla.run(config)
            with safe_open(str(out), framework="pt") as f:
                metadata = f.metadata()

        values = list(metadata.values())
        self.assertFalse(any(tmpdir in value for value in values))
        self.assertEqual(json.loads(metadata["ss_merge_inputs"])[0]["basename"], "secret_lora.safetensors")

    def test_metadata_no_path_separator_outside_inputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = _save_sd(tmp, "secret_lora.safetensors", _lora_sd())
            out = tmp / "out.safetensors"
            config = _config("--method", "linear", "--input", source, "1.0", "--output", str(out), "--output_rank", "2")

            mla.run(config)
            with safe_open(str(out), framework="pt") as f:
                metadata = f.metadata()

        for key, value in metadata.items():
            if key == "ss_merge_inputs":
                continue
            self.assertNotIn("/", value, msg=f"{key} leaked a path-like separator")


class TestPreviewSpectrum(unittest.TestCase):
    def test_preview_aggregate_default_and_no_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = _save_sd(tmp, "a.safetensors", _lora_sd())
            out = tmp / "should_not_exist.safetensors"
            config = _config("--method", "linear", "--input", source, "1.0", "--preview_spectrum")
            buf = io.StringIO()

            with contextlib.redirect_stdout(buf):
                result = mla.run(config)

        text = buf.getvalue()
        self.assertIn("Aggregate energy captured", text)
        self.assertIn("rank=  8", text)
        self.assertNotIn("lora_unet_block:", text)
        self.assertFalse(out.exists())
        energies = [result.spectrum_energy[rank][0] for rank in mla.SPECTRUM_RANKS]
        self.assertEqual(energies, sorted(energies))

    def test_zero_delta_modules_do_not_inflate_preview_spectrum(self) -> None:
        zero_sd = _lora_sd(name="lora_unet_zero", down=torch.zeros(2, 3), up=torch.zeros(2, 2))
        nonzero_sd = _lora_sd(name="lora_unet_nonzero")
        with tempfile.TemporaryDirectory() as tmpdir:
            source = _save_sd(Path(tmpdir), "mixed.safetensors", zero_sd | nonzero_sd)
            config = _config("--method", "linear", "--input", source, "1.0", "--preview_spectrum")

            result = mla.merge_adapters(config)

        self.assertEqual(result.modules_processed, 2)
        self.assertEqual(len(result.spectrum_energy[mla.SPECTRUM_RANKS[0]]), 1)
        self.assertNotIn("lora_unet_zero", result.per_module_energy)

    def test_preview_per_module_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = _save_sd(Path(tmpdir), "a.safetensors", _lora_sd())
            config = _config("--method", "linear", "--input", source, "1.0", "--preview_spectrum", "--preview_per_module")
            buf = io.StringIO()

            with contextlib.redirect_stdout(buf):
                mla.run(config)

        self.assertIn("lora_unet_block:", buf.getvalue())

    def test_preview_header_echoes_prune_threshold_when_set(self) -> None:
        """Provenance: pruning happens before the spectrum is computed, so a pasted
        preview transcript must record the threshold value that shaped the stats.
        Header echoes prune_threshold=N alongside density/drop_prob/seed when > 0;
        omits it at default 0.0 to keep header noise low."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = _save_sd(Path(tmpdir), "a.safetensors", _lora_sd())
            # Run 1: with --prune_threshold 1e-4 → header includes the value
            config_with = _config(
                "--method",
                "linear",
                "--input",
                source,
                "1.0",
                "--prune_threshold",
                "1e-4",
                "--preview_spectrum",
            )
            buf_with = io.StringIO()
            with contextlib.redirect_stdout(buf_with):
                mla.run(config_with)
            text_with = buf_with.getvalue()

            # Run 2: at default 0.0 → header does NOT mention prune_threshold
            config_default = _config("--method", "linear", "--input", source, "1.0", "--preview_spectrum")
            buf_default = io.StringIO()
            with contextlib.redirect_stdout(buf_default):
                mla.run(config_default)
            text_default = buf_default.getvalue()

        self.assertIn("prune_threshold=", text_with)
        self.assertIn("0.0001", text_with)  # %g formatting of 1e-4
        self.assertNotIn("prune_threshold=", text_default)


class TestOutputDtypeAndMaterializationShape(unittest.TestCase):
    def test_default_output_is_fp32_and_lower_precision_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = _save_sd(tmp, "a.safetensors", _lora_sd())
            out_fp32 = tmp / "fp32.safetensors"
            out_bf16 = tmp / "bf16.safetensors"
            out_fp16 = tmp / "fp16.safetensors"

            mla.run(_config("--method", "linear", "--input", source, "1.0", "--output", str(out_fp32), "--output_rank", "2"))
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    source,
                    "1.0",
                    "--output",
                    str(out_bf16),
                    "--output_rank",
                    "2",
                    "--output_dtype",
                    "bf16",
                )
            )
            mla.run(
                _config(
                    "--method",
                    "linear",
                    "--input",
                    source,
                    "1.0",
                    "--output",
                    str(out_fp16),
                    "--output_rank",
                    "2",
                    "--output_dtype",
                    "fp16",
                )
            )
            fp32 = load_file(str(out_fp32))
            bf16 = load_file(str(out_bf16))
            fp16 = load_file(str(out_fp16))

        self.assertEqual(fp32["lora_unet_block.lora_down.weight"].dtype, torch.float32)
        self.assertEqual(bf16["lora_unet_block.lora_down.weight"].dtype, torch.bfloat16)
        self.assertEqual(fp16["lora_unet_block.lora_down.weight"].dtype, torch.float16)

    def test_weakref_probe_confirms_previous_module_deltas_are_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd = _lora_sd(name="lora_unet_a") | _lora_sd(name="lora_unet_b") | _lora_sd(name="lora_unet_c")
            source_a = _save_sd(tmp, "a.safetensors", sd)
            source_b = _save_sd(tmp, "b.safetensors", sd)
            config = _config("--method", "linear", "--input", source_a, "1.0", "--input", source_b, "1.0", "--preview_spectrum")
            seen: list[str] = []
            refs: list[weakref.ReferenceType[torch.Tensor]] = []
            original = mla.materialize_module_delta

            def callback(module_name: str) -> None:
                live = [ref for ref in refs if ref() is not None]
                self.assertEqual(live, [], f"Previous module deltas still alive at start of {module_name}")
                seen.append(module_name)

            def wrapped_materialize(
                adapter: mla.AdapterInfo,
                module_name: str,
                *,
                dora_base_tensor: torch.Tensor | None = None,
            ) -> torch.Tensor | None:
                live = [ref for ref in refs if ref() is not None]
                self.assertLessEqual(len(live), len(config.inputs))
                delta = original(adapter, module_name, dora_base_tensor=dora_base_tensor)
                if delta is not None:
                    refs.append(weakref.ref(delta))
                return delta

            with patch.object(mla, "materialize_module_delta", side_effect=wrapped_materialize):
                mla.merge_adapters(config, module_callback=callback)

        self.assertEqual(seen, ["lora_unet_a", "lora_unet_b", "lora_unet_c"])


class TestFoldIntoValidation(unittest.TestCase):
    """Pins the --fold_into parser/config/validation surface from Tier 2 #5 v1.5 #3 step 2.

    Step 2 installs the CLI contract (flag, MergeConfig field, sentinel dtype, validation
    reordering, fold-mode rejections, run() guard) without any base-loading or fold execution.
    These tests freeze that contract before step 3 expands the diff with base-loading helpers.
    """

    def test_fold_requires_output(self) -> None:
        with self.assertRaisesRegex(ValueError, r"--fold_into requires --output"):
            _config(
                "--method",
                "linear",
                "--input",
                "a.safetensors",
                "1.0",
                "--fold_into",
                "base.safetensors",
            )

    def test_fold_rejects_lora_only_flags(self) -> None:
        # Each rejection must name both the conflicting flag and --fold_into so the user
        # learns what to remove and why. Lock that invariant via subTest per flag.
        cases = (
            (("--output_rank", "32"), r"--output_rank.*--fold_into"),
            (("--output_alpha", "16"), r"--output_alpha.*--fold_into"),
            (("--output_use_rslora",), r"--output_use_rslora.*--fold_into"),
            (("--preview_spectrum",), r"--preview_spectrum.*--fold_into"),
            (("--preview_per_module",), r"--preview_per_module.*--fold_into"),
        )
        base_argv = (
            "--method",
            "linear",
            "--input",
            "a.safetensors",
            "1.0",
            "--fold_into",
            "base.safetensors",
            "--output",
            "folded.safetensors",
        )
        for extra, pattern in cases:
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, pattern):
                    _config(*base_argv, *extra)

    def test_fold_valid_surface_does_not_require_output_rank(self) -> None:
        # Pins the validation reordering: in fold mode, the LoRA-mode rule
        # "--output_rank is required when --output is set" must NOT fire.
        config = _config(
            "--method",
            "linear",
            "--input",
            "a.safetensors",
            "1.0",
            "--fold_into",
            "base.safetensors",
            "--output",
            "folded.safetensors",
        )
        self.assertEqual(config.fold_into, "base.safetensors")
        self.assertEqual(config.output, "folded.safetensors")
        self.assertIsNone(config.output_rank)
        self.assertIsNone(config.output_alpha)

    def test_output_dtype_defaults_by_mode(self) -> None:
        # LoRA mode default → "fp32" / torch.float32
        cfg_lora = _config(
            "--method",
            "linear",
            "--input",
            "a.safetensors",
            "1.0",
            "--output",
            "out.safetensors",
            "--output_rank",
            "8",
        )
        self.assertEqual(cfg_lora.output_dtype_name, "fp32")
        self.assertIs(cfg_lora.output_dtype, torch.float32)

        # Fold mode default → "base" sentinel / None
        cfg_fold = _config(
            "--method",
            "linear",
            "--input",
            "a.safetensors",
            "1.0",
            "--fold_into",
            "base.safetensors",
            "--output",
            "folded.safetensors",
        )
        self.assertEqual(cfg_fold.output_dtype_name, "base")
        self.assertIsNone(cfg_fold.output_dtype)

        # Fold mode + explicit override → concrete dtype, no sentinel
        cfg_fold_bf16 = _config(
            "--method",
            "linear",
            "--input",
            "a.safetensors",
            "1.0",
            "--fold_into",
            "base.safetensors",
            "--output",
            "folded.safetensors",
            "--output_dtype",
            "bf16",
        )
        self.assertEqual(cfg_fold_bf16.output_dtype_name, "bf16")
        self.assertIs(cfg_fold_bf16.output_dtype, torch.bfloat16)

        # Invariant: output_dtype is None iff output_dtype_name == "base"
        for cfg in (cfg_lora, cfg_fold, cfg_fold_bf16):
            with self.subTest(cfg=cfg):
                self.assertEqual(cfg.output_dtype is None, cfg.output_dtype_name == "base")

    def test_output_dtype_base_is_not_argparse_choice(self) -> None:
        # "base" is an internal sentinel only — must never appear as a user-facing choice.
        # If it did, --output_dtype base would be accepted in LoRA mode and become an
        # accepted-but-ignored trap (the LoRA-output writer always needs a concrete dtype).
        self.assertNotIn("base", mla.OUTPUT_DTYPES)

        # Defense in depth: argparse should reject --output_dtype base via SystemExit.
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                mla.parse_args(
                    [
                        "--method",
                        "linear",
                        "--input",
                        "a.safetensors",
                        "1.0",
                        "--preview_spectrum",
                        "--output_dtype",
                        "base",
                    ]
                )

    def test_lora_output_rank_requirement_unchanged(self) -> None:
        # After the validate_args branch reordering, the LoRA-mode rule
        # "--output_rank is required when --output is set" must still fire when
        # --fold_into is absent.
        with self.assertRaisesRegex(ValueError, r"--output_rank is required when --output is set"):
            _config(
                "--method",
                "linear",
                "--input",
                "a.safetensors",
                "1.0",
                "--output",
                "out.safetensors",
            )


def _write_split_shard(tmp: Path, prefix: str, idx: int, count: int, sd: dict[str, torch.Tensor]) -> str:
    """Write one shard of a split set following the 00001-of-NNNNN naming scheme."""
    path = tmp / f"{prefix}-{idx:05d}-of-{count:05d}.safetensors"
    save_file(sd, str(path))
    return str(path)


class TestBaseDitValidation(unittest.TestCase):
    """Pins --base_dit CLI surface and mutex contracts for v1.5 #4.

    Tests in this class lock the parse-time portion of the v1.5 #4 contract — flag
    presence, mutex with --fold_into, and config field plumbing. Adapter-content-dependent
    rejections (DoRA + no --base_dit, --base_dit + no DoRA, etc.) live in later test
    classes that land alongside the loader-wall opening and post-load preflight code.
    """

    def test_base_dit_plus_fold_into_rejects_separate_surfaces(self) -> None:
        # Decision #4: fold mode and DoRA mode are separate surfaces in v1.5 #4.
        # Both flags are file-path arguments → mutex is parse-time (no file introspection).
        with self.assertRaisesRegex(ValueError, r"--base_dit and --fold_into are mutually exclusive"):
            _config(
                "--method", "linear",
                "--input", "a.safetensors", "1.0",
                "--base_dit", "base.safetensors",
                "--fold_into", "fold_base.safetensors",
                "--output", "out.safetensors",
            )

    def test_base_dit_field_populated_in_config(self) -> None:
        config = _config(
            "--method", "linear",
            "--input", "a.safetensors", "1.0",
            "--base_dit", "/path/to/base.safetensors",
            "--output", "out.safetensors",
            "--output_rank", "8",
        )
        self.assertEqual(config.base_dit, "/path/to/base.safetensors")

    def test_base_dit_default_is_none(self) -> None:
        config = _config(
            "--method", "linear",
            "--input", "a.safetensors", "1.0",
            "--output", "out.safetensors",
            "--output_rank", "8",
        )
        self.assertIsNone(config.base_dit)

    def test_base_dit_without_dora_input_rejects(self) -> None:
        # Decision #11: --base_dit without DoRA inputs is the accepted-but-ignored case.
        # Loud reject so a misconfigured pipeline surfaces immediately.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            standard_lora_path = _save_sd(tmp, "standard.safetensors", _lora_sd())
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3)})
            config = _config(
                "--method", "linear",
                "--input", standard_lora_path, "1.0",
                "--base_dit", base_path,
                "--output", str(tmp / "out.safetensors"),
                "--output_rank", "2",
            )
            with self.assertRaisesRegex(ValueError, r"--base_dit was provided but no input adapter is DoRA"):
                mla.run(config)

    def test_base_dit_validation_happens_after_adapter_load_not_at_parse_time(self) -> None:
        # Decision #11 wording lock: argparse cannot see file contents — DoRA detection
        # requires opening the safetensors and inspecting use_dora_flag / .dora_layer.weight.
        # The "no DoRA inputs" check must fire AFTER load_adapter, not at validate_args.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            standard_lora_path = _save_sd(tmp, "standard.safetensors", _lora_sd())
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3)})
            # validate_args succeeds (parse-time): base_dit is just a path string at this layer.
            config = _config(
                "--method", "linear",
                "--input", standard_lora_path, "1.0",
                "--base_dit", base_path,
                "--output", str(tmp / "out.safetensors"),
                "--output_rank", "2",
            )
            # Validation accepts the args; merge_adapters does the post-load preflight reject.
            self.assertEqual(config.base_dit, base_path)
            with self.assertRaises(ValueError):
                mla.run(config)


class TestExpandFoldBasePaths(unittest.TestCase):
    """Pins shard expansion semantics for --fold_into base paths."""

    def test_single_file_returns_singleton_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            path = _save_sd(tmp, "base.safetensors", {"x": torch.zeros(2)})
            self.assertEqual(mla._expand_fold_base_paths(path), [path])

    def test_split_first_shard_returns_all_in_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard_1 = _write_split_shard(tmp, "tiny", 1, 3, {"a": torch.zeros(2)})
            shard_2 = _write_split_shard(tmp, "tiny", 2, 3, {"b": torch.zeros(2)})
            shard_3 = _write_split_shard(tmp, "tiny", 3, 3, {"c": torch.zeros(2)})
            self.assertEqual(mla._expand_fold_base_paths(shard_1), [shard_1, shard_2, shard_3])

    def test_split_non_first_shard_still_returns_all_in_canonical_order(self) -> None:
        # Guardrail: get_split_weight_filenames is shard-name agnostic. Passing any shard
        # of the split set must rebuild the canonical 1..N sequence from the prefix.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard_1 = _write_split_shard(tmp, "tiny", 1, 2, {"a": torch.zeros(2)})
            shard_2 = _write_split_shard(tmp, "tiny", 2, 2, {"b": torch.zeros(2)})
            self.assertEqual(mla._expand_fold_base_paths(shard_2), [shard_1, shard_2])

    def test_missing_shard_raises_with_fold_into_hint(self) -> None:
        # Save only the first shard of an expected 3-shard set. Expansion must raise
        # FileNotFoundError with --fold_into context, not a bare path-not-found.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard_1 = _write_split_shard(tmp, "tiny", 1, 3, {"a": torch.zeros(2)})
            with self.assertRaisesRegex(FileNotFoundError, r"--fold_into base shard missing"):
                mla._expand_fold_base_paths(shard_1)


class TestLoadBaseAsStored(unittest.TestCase):
    """Pins dtype-preserving load semantics for --fold_into base checkpoints."""

    def test_single_file_loads_all_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd = {"a": torch.zeros(2, 3), "b.c": torch.ones(4)}
            path = _save_sd(tmp, "base.safetensors", sd)
            loaded = mla._load_base_as_stored(path)
            self.assertEqual(set(loaded), {"a", "b.c"})
            self.assertTrue(torch.equal(loaded["a"], sd["a"]))
            self.assertTrue(torch.equal(loaded["b.c"], sd["b.c"]))

    def test_split_file_merges_keys_from_all_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard_1 = _write_split_shard(tmp, "tiny", 1, 2, {"a": torch.zeros(2)})
            _write_split_shard(tmp, "tiny", 2, 2, {"b": torch.ones(3)})
            loaded = mla._load_base_as_stored(shard_1)
            self.assertEqual(set(loaded), {"a", "b"})
            self.assertTrue(torch.equal(loaded["a"], torch.zeros(2)))
            self.assertTrue(torch.equal(loaded["b"], torch.ones(3)))

    def test_split_file_duplicate_key_hard_rejects(self) -> None:
        # Silent-corruption guard: two shards declaring the same key would have the
        # second overwrite the first via dict.update, losing evidence before any
        # downstream fold-plan check could see it. Hard-reject at load.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard_1 = _write_split_shard(tmp, "tiny", 1, 2, {"shared.weight": torch.zeros(2)})
            _write_split_shard(tmp, "tiny", 2, 2, {"shared.weight": torch.ones(2)})
            with self.assertRaisesRegex(ValueError, r"duplicate tensor keys across shards"):
                mla._load_base_as_stored(shard_1)

    def test_mixed_dtypes_preserved_no_fp32_promotion(self) -> None:
        # Critical: the fold writer's "base" dtype sentinel only works if the loader
        # preserves per-tensor dtype. Any silent fp32 promotion here would produce a
        # 2-4x larger output checkpoint and lie about the source dtypes.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sd = {
                "fp32_tensor": torch.zeros(2, dtype=torch.float32),
                "bf16_tensor": torch.zeros(2, dtype=torch.bfloat16),
                "fp16_tensor": torch.zeros(2, dtype=torch.float16),
            }
            path = _save_sd(tmp, "base.safetensors", sd)
            loaded = mla._load_base_as_stored(path)
            self.assertIs(loaded["fp32_tensor"].dtype, torch.float32)
            self.assertIs(loaded["bf16_tensor"].dtype, torch.bfloat16)
            self.assertIs(loaded["fp16_tensor"].dtype, torch.float16)


class TestCompositeBaseHash(unittest.TestCase):
    """Pins file-bytes hash semantics for --fold_into base provenance."""

    def test_single_file_hash_equals_file_sha256(self) -> None:
        # Single-file invariant: composite hash matches `sha256sum` so users can verify
        # the base provenance string against their local file with shell tools.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            path = _save_sd(tmp, "base.safetensors", {"x": torch.arange(8, dtype=torch.float32)})
            self.assertEqual(mla._composite_base_hash(path), mla._file_sha256(path))

    def test_split_file_hash_is_deterministic_and_independent_of_shard_passed(self) -> None:
        # The composite hash must depend only on the shard contents (in canonical order),
        # not on which shard the user happened to pass.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard_1 = _write_split_shard(tmp, "tiny", 1, 2, {"a": torch.zeros(2)})
            shard_2 = _write_split_shard(tmp, "tiny", 2, 2, {"b": torch.ones(3)})
            self.assertEqual(mla._composite_base_hash(shard_1), mla._composite_base_hash(shard_2))

    def test_split_file_hash_changes_when_any_shard_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard_1_a = _write_split_shard(tmp, "alpha", 1, 2, {"a": torch.zeros(2)})
            _write_split_shard(tmp, "alpha", 2, 2, {"b": torch.ones(3)})
            hash_a = mla._composite_base_hash(shard_1_a)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shard_1_b = _write_split_shard(tmp, "beta", 1, 2, {"a": torch.zeros(2)})
            # Differ only in shard 2 contents.
            _write_split_shard(tmp, "beta", 2, 2, {"b": torch.full((3,), 2.0)})
            hash_b = mla._composite_base_hash(shard_1_b)

        self.assertNotEqual(hash_a, hash_b)


class TestAssertNoFp8InBase(unittest.TestCase):
    """Pins fold-mode rejection of fp8-quantized base checkpoints."""

    def test_no_fp8_does_not_raise(self) -> None:
        sd = {
            "fc.weight": torch.zeros(4, 4, dtype=torch.float32),
            "fc.bias": torch.zeros(4, dtype=torch.bfloat16),
            "norm.weight": torch.ones(4, dtype=torch.float16),
        }
        # Must not raise.
        mla._assert_no_fp8_in_base(sd, "/tmp/base.safetensors")

    def test_fp8_e4m3fn_present_raises_with_fold_into_and_basename(self) -> None:
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch build lacks float8_e4m3fn")
        sd = {
            "fc.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn),
            "fc.bias": torch.zeros(4, dtype=torch.float32),
        }
        with self.assertRaisesRegex(ValueError, r"--fold_into base 'base\.safetensors'.*fp8"):
            mla._assert_no_fp8_in_base(sd, "/tmp/some/dir/base.safetensors")

    def test_fp8_e5m2_present_raises_with_actionable_resolution_text(self) -> None:
        if not hasattr(torch, "float8_e5m2"):
            self.skipTest("torch build lacks float8_e5m2")
        sd = {"fc.weight": torch.zeros(4, 4, dtype=torch.float8_e5m2)}
        with self.assertRaisesRegex(ValueError, r"re-quantize downstream"):
            mla._assert_no_fp8_in_base(sd, "/tmp/base.safetensors")


class TestBuildBaseLoraIndex(unittest.TestCase):
    """Pins the forward-mapping base-key index used by --fold_into resolution."""

    def test_index_only_includes_weight_keys(self) -> None:
        # Biases, scales, alphas, and any non-".weight" tensor must be skipped — LoRA
        # adapters target weight matrices only.
        base_sd = {
            "block.weight": torch.zeros(2, 3),
            "block.bias": torch.zeros(2),
            "block.scale": torch.tensor(1.0),
            "norm.running_mean": torch.zeros(2),
        }
        index = mla.build_base_lora_index(base_sd)
        self.assertEqual(set(index), {"lora_unet_block"})
        self.assertEqual(index["lora_unet_block"], ["block.weight"])

    def test_index_uses_forward_mapping_with_underscored_base_keys(self) -> None:
        # Critical correctness: base keys with REAL underscores (not dot-replacements)
        # must forward-map correctly. A naive inverse parser would invent the wrong key.
        base_sd = {"double_blocks.0.img_attn.qkv.weight": torch.zeros(2, 3)}
        index = mla.build_base_lora_index(base_sd)
        self.assertEqual(
            index,
            {"lora_unet_double_blocks_0_img_attn_qkv": ["double_blocks.0.img_attn.qkv.weight"]},
        )

    def test_index_returns_lists_to_preserve_ambiguity_evidence(self) -> None:
        # When two distinct base keys forward-map to the same LoRA name, BOTH must be
        # preserved in the list. A dict[str, str] would silently keep only the last one.
        base_sd = {
            "double_blocks.0.img_attn.qkv.weight": torch.zeros(2, 3),
            "double.blocks.0.img.attn.qkv.weight": torch.zeros(2, 3),
        }
        index = mla.build_base_lora_index(base_sd)
        candidates = index["lora_unet_double_blocks_0_img_attn_qkv"]
        self.assertEqual(set(candidates), {"double_blocks.0.img_attn.qkv.weight", "double.blocks.0.img.attn.qkv.weight"})
        self.assertEqual(len(candidates), 2)

    def test_index_strips_model_diffusion_model_prefix(self) -> None:
        # WAN-style checkpoints prepend "model.diffusion_model." which the production
        # helper strips before mapping. The index must inherit that behavior.
        base_sd = {"model.diffusion_model.blocks.0.attn.q.weight": torch.zeros(2, 3)}
        index = mla.build_base_lora_index(base_sd)
        self.assertEqual(set(index), {"lora_unet_blocks_0_attn_q"})


class TestResolveFoldPlan(unittest.TestCase):
    """Pins preflight orphan / ambiguity / non-floating / shape-mismatch rejections."""

    def _make_adapter(self, tmp: Path, *, name: str, **lora_kwargs: object) -> mla.AdapterInfo:
        sd = _lora_sd(name=name, **lora_kwargs)  # type: ignore[arg-type]
        path = _save_sd(tmp, f"{name}.safetensors", sd)
        return mla.load_adapter(path, 1.0)

    def test_resolve_returns_deterministic_plan_in_module_union_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Two adapters covering modules {a, b} and {b, c}; union should be a, b, c sorted.
            sd_ab = _lora_sd(name="lora_unet_a") | _lora_sd(name="lora_unet_b")
            sd_bc = _lora_sd(name="lora_unet_b") | _lora_sd(name="lora_unet_c")
            adapter_ab = mla.load_adapter(_save_sd(tmp, "ab.safetensors", sd_ab), 1.0)
            adapter_bc = mla.load_adapter(_save_sd(tmp, "bc.safetensors", sd_bc), 1.0)
            base_sd = {
                "a.weight": torch.zeros(2, 3),
                "b.weight": torch.zeros(2, 3),
                "c.weight": torch.zeros(2, 3),
            }
            plan = mla.resolve_fold_plan([adapter_ab, adapter_bc], base_sd)
            self.assertEqual(list(plan), ["lora_unet_a", "lora_unet_b", "lora_unet_c"])

    def test_resolve_orphan_lora_module_hard_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter = self._make_adapter(tmp, name="lora_unet_nonexistent")
            base_sd = {"some_other.weight": torch.zeros(2, 3)}
            with self.assertRaisesRegex(ValueError, r"--fold_into orphan.*lora_unet_nonexistent"):
                mla.resolve_fold_plan([adapter], base_sd)

    def test_resolve_ambiguous_base_key_hard_rejects(self) -> None:
        # Both base keys forward-map to lora_unet_double_blocks_0_img_attn_qkv.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter = self._make_adapter(tmp, name="lora_unet_double_blocks_0_img_attn_qkv")
            base_sd = {
                "double_blocks.0.img_attn.qkv.weight": torch.zeros(2, 3),
                "double.blocks.0.img.attn.qkv.weight": torch.zeros(2, 3),
            }
            with self.assertRaisesRegex(ValueError, r"--fold_into ambiguity.*forward-maps to multiple base keys"):
                mla.resolve_fold_plan([adapter], base_sd)

    def test_resolve_non_floating_base_target_hard_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter = self._make_adapter(tmp, name="lora_unet_block")
            base_sd = {"block.weight": torch.zeros(2, 3, dtype=torch.int32)}
            with self.assertRaisesRegex(ValueError, r"non-floating target.*torch\.int32"):
                mla.resolve_fold_plan([adapter], base_sd)

    def test_resolve_shape_mismatch_hard_rejects(self) -> None:
        # User-specified shape: LoRA delta (2, 3), base tensor (2, 4). Differs in in_dim
        # at the materialized-delta level, not just at the LoRA factor level.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter = self._make_adapter(tmp, name="lora_unet_block", in_dim=3, out_dim=2)
            base_sd = {"block.weight": torch.zeros(2, 4)}
            with self.assertRaisesRegex(ValueError, r"shape mismatch.*\(2, 3\).*\(2, 4\)"):
                mla.resolve_fold_plan([adapter], base_sd)

    def test_resolve_uses_base_key_forward_index_not_underscore_inverse(self) -> None:
        # THE load-bearing test for the forward-mapping decision (locked plan, decision #11).
        # A naive inverse parser would map "lora_unet_double_blocks_0_img_attn_qkv" to a
        # nonexistent "double.blocks.0.img.attn.qkv.weight" and reject as orphan. The forward
        # mapping correctly picks the actually-present "double_blocks.0.img_attn.qkv.weight".
        # No decoy in base_sd here — that case is covered separately as ambiguity.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter = self._make_adapter(tmp, name="lora_unet_double_blocks_0_img_attn_qkv")
            base_sd = {"double_blocks.0.img_attn.qkv.weight": torch.zeros(2, 3)}
            plan = mla.resolve_fold_plan([adapter], base_sd)
            self.assertEqual(
                plan["lora_unet_double_blocks_0_img_attn_qkv"].base_key,
                "double_blocks.0.img_attn.qkv.weight",
            )

    def test_fold_target_carries_base_shape_and_dtype(self) -> None:
        # FoldTarget must record what the writer will need: base_shape (for re-validation
        # against the materialized delta) and base_dtype (for the "base" sentinel path).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter = self._make_adapter(tmp, name="lora_unet_block")
            base_sd = {"block.weight": torch.zeros(2, 3, dtype=torch.bfloat16)}
            plan = mla.resolve_fold_plan([adapter], base_sd)
            target = plan["lora_unet_block"]
            self.assertEqual(target.lora_name, "lora_unet_block")
            self.assertEqual(target.base_key, "block.weight")
            self.assertEqual(target.base_shape, (2, 3))
            self.assertIs(target.base_dtype, torch.bfloat16)

    def test_resolve_rejects_unsupported_conv2d_up_kernel(self) -> None:
        # Mirrors materialize_module_delta's contract: Conv2d LoRA must have a 1x1 up kernel.
        # Without preflight enforcement, an unsupported geometry would slip through resolution
        # and only fail mid-pipeline during materialization. Catch it at the preflight boundary.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rank, in_dim, out_dim = 2, 3, 2
            sd = _lora_sd(
                name="lora_unet_conv",
                rank=rank,
                in_dim=in_dim,
                out_dim=out_dim,
                down=torch.zeros(rank, in_dim, 1, 1),
                up=torch.zeros(out_dim, rank, 3, 3),  # 3x3 up kernel — unsupported
            )
            adapter = mla.load_adapter(_save_sd(tmp, "conv.safetensors", sd), 1.0)
            base_sd = {"conv.weight": torch.zeros(out_dim, in_dim, 1, 1)}
            with self.assertRaisesRegex(ValueError, r"unsupported Conv2d LoRA up kernel"):
                mla.resolve_fold_plan([adapter], base_sd)

    def test_resolve_accepts_conv2d_with_matching_4d_base(self) -> None:
        # 1x1 Conv2d LoRA: down=(rank, in_dim, 1, 1), up=(out_dim, rank, 1, 1) →
        # delta=(out_dim, in_dim, 1, 1). Base tensor of matching 4D shape must resolve.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rank, in_dim, out_dim = 2, 3, 2
            sd = _lora_sd(
                name="lora_unet_conv",
                rank=rank,
                in_dim=in_dim,
                out_dim=out_dim,
                down=torch.zeros(rank, in_dim, 1, 1),
                up=torch.zeros(out_dim, rank, 1, 1),
            )
            adapter = mla.load_adapter(_save_sd(tmp, "conv.safetensors", sd), 1.0)
            base_sd = {"conv.weight": torch.zeros(out_dim, in_dim, 1, 1)}
            plan = mla.resolve_fold_plan([adapter], base_sd)
            self.assertEqual(plan["lora_unet_conv"].base_shape, (2, 3, 1, 1))


class TestSaveAtomic(unittest.TestCase):
    """Pins crash-safe write semantics for fold-mode output."""

    def test_save_atomic_writes_temp_then_replace(self) -> None:
        # Happy-path + ordering evidence in one test:
        #   * save_file is invoked with a path != output_path (must be the temp file)
        #   * after save_atomic returns, output_path exists with the right content
        #   * the temp path no longer exists (it was renamed away by os.replace)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "out.safetensors")
            seen_paths: list[str] = []
            real_save = mla.save_file

            def recording_save(state_dict, path, metadata=None):
                seen_paths.append(path)
                real_save(state_dict, path, metadata=metadata)

            with patch.object(mla, "save_file", side_effect=recording_save):
                mla.save_atomic(
                    {"x": torch.arange(4, dtype=torch.float32)},
                    out_path,
                    {"ss_merge_tool": "test"},
                )

            self.assertEqual(len(seen_paths), 1)
            temp_path = seen_paths[0]
            self.assertNotEqual(temp_path, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertFalse(os.path.exists(temp_path))
            loaded = load_file(out_path)
            self.assertTrue(torch.equal(loaded["x"], torch.arange(4, dtype=torch.float32)))
            with safe_open(out_path, framework="pt") as f:
                self.assertEqual(f.metadata().get("ss_merge_tool"), "test")

    def test_save_atomic_temp_file_in_same_directory_as_output(self) -> None:
        # Atomicity guarantee: os.replace is only atomic across the same filesystem.
        # The temp file must live in the SAME directory as the destination, not /tmp.
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "subdir_target.safetensors")
            seen_paths: list[str] = []
            real_save = mla.save_file

            def recording_save(state_dict, path, metadata=None):
                seen_paths.append(path)
                real_save(state_dict, path, metadata=metadata)

            with patch.object(mla, "save_file", side_effect=recording_save):
                mla.save_atomic({"x": torch.zeros(2)}, out_path, {})

            self.assertEqual(os.path.dirname(seen_paths[0]), os.path.dirname(os.path.abspath(out_path)))

    def test_save_atomic_cleans_temp_on_save_failure(self) -> None:
        # If save_file raises, the temp file must be removed before the exception
        # propagates — otherwise a failed fold leaves a half-written file behind that
        # could be confused with a successful one on a retry.
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "out.safetensors")
            with patch.object(mla, "save_file", side_effect=RuntimeError("simulated disk full")):
                with self.assertRaisesRegex(RuntimeError, "simulated disk full"):
                    mla.save_atomic({"x": torch.zeros(2)}, out_path, {})
            # Output file was never created and no temp residue remains.
            self.assertFalse(os.path.exists(out_path))
            self.assertEqual(list(Path(tmpdir).iterdir()), [])


class TestBuildFoldMetadata(unittest.TestCase):
    """Pins the fold-mode metadata contract: format, provenance, privacy, count serialization."""

    def _setup_fold_config(self, tmp: Path) -> tuple[mla.MergeConfig, list[mla.AdapterInfo], str]:
        adapter_path = _save_sd(tmp, "adapter.safetensors", _lora_sd())
        adapter = mla.load_adapter(adapter_path, 1.0)
        base_path = _save_sd(tmp, "base.safetensors", {"x": torch.zeros(4, dtype=torch.float32)})
        config = _config(
            "--method",
            "linear",
            "--input",
            adapter_path,
            "1.0",
            "--fold_into",
            base_path,
            "--output",
            str(tmp / "folded.safetensors"),
        )
        return config, [adapter], base_path

    def test_fold_metadata_records_checkpoint_format_and_base_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config, adapters, base_path = self._setup_fold_config(tmp)
            stats = mla.FoldStats(
                modules_resolved=3, modules_folded=2, modules_pruned=1,
                base_tensors_total=10, base_tensors_modified=2,
            )
            meta = mla.build_fold_metadata(config, adapters, base_path, stats)

            self.assertEqual(meta["ss_merge_output_format"], "checkpoint")
            self.assertEqual(meta["ss_merge_base_basename"], "base.safetensors")
            self.assertEqual(meta["ss_merge_base_sha256"], mla._file_sha256(base_path))
            # Common metadata still present.
            self.assertEqual(meta["ss_merge_tool"], "blissful-tuner")
            self.assertEqual(meta["ss_merge_method"], "linear")
            self.assertEqual(meta["ss_merge_rejects_dora"], "true")

    def test_fold_metadata_does_not_include_absolute_paths(self) -> None:
        # Privacy: tmpdir absolute path must not leak into any metadata value.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config, adapters, base_path = self._setup_fold_config(tmp)
            stats = mla.FoldStats(0, 0, 0, 0, 0)
            meta = mla.build_fold_metadata(config, adapters, base_path, stats)
            for key, value in meta.items():
                with self.subTest(key=key):
                    self.assertNotIn(tmpdir, value, f"absolute path leaked into metadata[{key!r}] = {value!r}")

    def test_fold_metadata_omits_lora_recompression_keys(self) -> None:
        # LoRA-only metadata keys have no meaning for a folded checkpoint.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config, adapters, base_path = self._setup_fold_config(tmp)
            stats = mla.FoldStats(0, 0, 0, 0, 0)
            meta = mla.build_fold_metadata(config, adapters, base_path, stats)
            for forbidden in (
                "ss_merge_output_rank",
                "ss_merge_output_alpha",
                "ss_merge_output_use_rslora",
                "ss_merge_recompression",
            ):
                with self.subTest(forbidden=forbidden):
                    self.assertNotIn(forbidden, meta)

    def test_fold_metadata_records_counts_from_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config, adapters, base_path = self._setup_fold_config(tmp)
            stats = mla.FoldStats(
                modules_resolved=42, modules_folded=30, modules_pruned=12,
                base_tensors_total=100, base_tensors_modified=30,
            )
            meta = mla.build_fold_metadata(config, adapters, base_path, stats)
            self.assertEqual(meta["ss_merge_modules_resolved"], "42")
            self.assertEqual(meta["ss_merge_modules_folded"], "30")
            self.assertEqual(meta["ss_merge_modules_pruned"], "12")
            self.assertEqual(meta["ss_merge_base_tensors_total"], "100")
            self.assertEqual(meta["ss_merge_base_tensors_modified"], "30")

    def test_fold_stats_enforces_invariants_at_construction(self) -> None:
        # Invariants documented on the dataclass MUST be enforced; otherwise a fold-loop
        # accounting bug ships nonsensical metadata strings instead of failing loudly.
        with self.subTest("modules_resolved != modules_folded + modules_pruned"):
            with self.assertRaisesRegex(ValueError, r"modules_resolved.*must equal"):
                mla.FoldStats(
                    modules_resolved=5, modules_folded=2, modules_pruned=2,
                    base_tensors_total=10, base_tensors_modified=2,
                )
        with self.subTest("base_tensors_modified != modules_folded"):
            with self.assertRaisesRegex(ValueError, r"base_tensors_modified.*must equal"):
                mla.FoldStats(
                    modules_resolved=4, modules_folded=2, modules_pruned=2,
                    base_tensors_total=10, base_tensors_modified=3,
                )
        with self.subTest("base_tensors_modified > base_tensors_total"):
            with self.assertRaisesRegex(ValueError, r"must not exceed"):
                mla.FoldStats(
                    modules_resolved=4, modules_folded=4, modules_pruned=0,
                    base_tensors_total=3, base_tensors_modified=4,
                )

    def test_fold_stats_accepts_valid_boundary_cases(self) -> None:
        # All zero (no modules processed at all).
        mla.FoldStats(0, 0, 0, 0, 0)
        # All folded, none pruned, modified == total.
        mla.FoldStats(modules_resolved=5, modules_folded=5, modules_pruned=0,
                      base_tensors_total=5, base_tensors_modified=5)
        # All pruned, none folded; modified == 0 < total.
        mla.FoldStats(modules_resolved=5, modules_folded=0, modules_pruned=5,
                      base_tensors_total=10, base_tensors_modified=0)
        # Asymmetric: more total tensors than modules (the realistic case).
        mla.FoldStats(modules_resolved=42, modules_folded=30, modules_pruned=12,
                      base_tensors_total=100, base_tensors_modified=30)

    def test_fold_metadata_records_composite_base_hash_for_split_input(self) -> None:
        # Split base: the recorded hash must be the composite-of-shards, not the
        # bare file SHA of whichever shard was passed via --fold_into.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adapter_path = _save_sd(tmp, "adapter.safetensors", _lora_sd())
            adapter = mla.load_adapter(adapter_path, 1.0)
            shard_1 = _write_split_shard(tmp, "base", 1, 2, {"a": torch.zeros(2)})
            _write_split_shard(tmp, "base", 2, 2, {"b": torch.zeros(2)})
            config = _config(
                "--method",
                "linear",
                "--input",
                adapter_path,
                "1.0",
                "--fold_into",
                shard_1,
                "--output",
                str(tmp / "folded.safetensors"),
            )
            stats = mla.FoldStats(0, 0, 0, 0, 0)
            meta = mla.build_fold_metadata(config, [adapter], shard_1, stats)

            self.assertEqual(meta["ss_merge_base_sha256"], mla._composite_base_hash(shard_1))
            # Composite hash differs from any single-shard SHA in the multi-shard case.
            self.assertNotEqual(meta["ss_merge_base_sha256"], mla._file_sha256(shard_1))


class TestCastUntouchedFloatingTensors(unittest.TestCase):
    """Pins the helper that handles untouched floating tensors in explicit --output_dtype mode."""

    def test_casts_untouched_floating_tensors_only(self) -> None:
        # Three classes of tensor:
        #   * modified (in modified_keys) → preserved untouched (the fold loop already cast it)
        #   * untouched + floating → cast to target_dtype
        #   * untouched + non-floating → preserved untouched (would be lossy or fail)
        sd = {
            "modified.weight": torch.zeros(2, dtype=torch.float32),
            "untouched_floating.weight": torch.ones(2, dtype=torch.float16),
            "untouched_int.weight": torch.tensor([1, 2], dtype=torch.int32),
            "untouched_bool": torch.tensor([True, False]),
        }
        modified = {"modified.weight"}
        count = mla._cast_untouched_floating_tensors(sd, modified, torch.bfloat16)

        self.assertIs(sd["modified.weight"].dtype, torch.float32, "modified key should not be re-cast")
        self.assertIs(sd["untouched_floating.weight"].dtype, torch.bfloat16, "untouched fp should be cast")
        self.assertIs(sd["untouched_int.weight"].dtype, torch.int32, "untouched int must stay int")
        self.assertIs(sd["untouched_bool"].dtype, torch.bool, "untouched bool must stay bool")
        self.assertEqual(count, 1)

    def test_raises_on_non_finite_after_cast(self) -> None:
        # 1e5 is finite in fp32 but exceeds fp16 max (~65504), so the cast produces inf.
        # The helper must raise rather than silently writing a non-finite tensor.
        sd = {"big.weight": torch.full((2,), 1e5, dtype=torch.float32)}
        with self.assertRaisesRegex(ValueError, r"non-finite values when casting untouched"):
            mla._cast_untouched_floating_tensors(sd, set(), torch.float16)


class TestFoldPipeline(unittest.TestCase):
    """End-to-end tests for fold_adapters_into_base and run() dispatch.

    Each test runs a real fold operation against synthetic adapters and a synthetic base,
    then inspects the written checkpoint via load_file. All tests redirect stdout to keep
    the test runner output clean (the pipeline emits 'fold plan' and 'fold summary' lines).
    """

    def _run_fold(self, *, lora_sd: dict, base_sd: dict, tmp: Path, extra_args: tuple = ()) -> mla.FoldResult:
        lora_path = _save_sd(tmp, "lora.safetensors", lora_sd)
        base_path = _save_sd(tmp, "base.safetensors", base_sd)
        out_path = str(tmp / "folded.safetensors")
        config = _config(
            "--method",
            "linear",
            "--input",
            lora_path,
            "1.0",
            "--fold_into",
            base_path,
            "--output",
            out_path,
            *extra_args,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return mla.fold_adapters_into_base(config)

    def test_fold_happy_path_applies_delta_and_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lora_sd = _lora_sd(name="lora_unet_block")
            expected_delta = _explicit_delta(lora_sd, "lora_unet_block")
            base_block = torch.full_like(expected_delta, 0.5)
            other = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            result = self._run_fold(
                lora_sd=lora_sd,
                base_sd={"block.weight": base_block.clone(), "other.weight": other.clone()},
                tmp=tmp,
            )

            loaded = load_file(result.output_path)
            self.assertTrue(torch.allclose(loaded["block.weight"], base_block + expected_delta, atol=1e-5))
            self.assertTrue(torch.equal(loaded["other.weight"], other))
            self.assertIsInstance(result, mla.FoldResult)
            self.assertEqual(result.stats.modules_folded, 1)
            self.assertEqual(result.stats.modules_pruned, 0)
            self.assertEqual(result.stats.base_tensors_total, 2)
            self.assertEqual(result.stats.base_tensors_modified, 1)

    def test_fold_default_dtype_preserves_base_dtype_for_touched_and_untouched(self) -> None:
        # Sentinel "base" mode (no --output_dtype): each tensor keeps its stored dtype.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_sd = {
                "block.weight": torch.zeros(2, 3, dtype=torch.bfloat16),       # touched
                "norm.weight": torch.ones(2, dtype=torch.float16),             # untouched, floating
                "step_count": torch.zeros(1, dtype=torch.int64),               # untouched, non-floating
            }
            result = self._run_fold(lora_sd=_lora_sd(name="lora_unet_block"), base_sd=base_sd, tmp=tmp)

            loaded = load_file(result.output_path)
            self.assertIs(loaded["block.weight"].dtype, torch.bfloat16)
            self.assertIs(loaded["norm.weight"].dtype, torch.float16)
            self.assertIs(loaded["step_count"].dtype, torch.int64)

    def test_fold_output_dtype_override_casts_floating_tensors_only(self) -> None:
        # Explicit --output_dtype fp32: touched + untouched-floating cast; non-floating preserved.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_sd = {
                "block.weight": torch.zeros(2, 3, dtype=torch.bfloat16),
                "norm.weight": torch.ones(2, dtype=torch.float16),
                "step_count": torch.zeros(1, dtype=torch.int64),
            }
            result = self._run_fold(
                lora_sd=_lora_sd(name="lora_unet_block"),
                base_sd=base_sd,
                tmp=tmp,
                extra_args=("--output_dtype", "fp32"),
            )

            loaded = load_file(result.output_path)
            self.assertIs(loaded["block.weight"].dtype, torch.float32, "touched tensor cast to override dtype")
            self.assertIs(loaded["norm.weight"].dtype, torch.float32, "untouched floating cast to override dtype")
            self.assertIs(loaded["step_count"].dtype, torch.int64, "non-floating preserved across cast")

    def test_fold_pruned_module_leaves_base_unchanged(self) -> None:
        # All-zero up tensor → merged delta is exactly 0 → pruned at default threshold (0.0).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lora_sd = _lora_sd(name="lora_unet_block", up=torch.zeros(2, 2))
            original_block = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
            result = self._run_fold(
                lora_sd=lora_sd,
                base_sd={"block.weight": original_block.clone()},
                tmp=tmp,
            )

            loaded = load_file(result.output_path)
            self.assertTrue(torch.equal(loaded["block.weight"], original_block))
            self.assertEqual(result.stats.modules_folded, 0)
            self.assertEqual(result.stats.modules_pruned, 1)
            self.assertEqual(result.stats.base_tensors_modified, 0)

    def test_fold_rejects_non_finite_folded_tensor(self) -> None:
        # base + delta overflows fp32 (both ~2e38, sum ~4e38 = inf). The first
        # finite check inside the fold loop must fire; no output file should be written.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # rank=1, in_dim=2, out_dim=2 → delta[i,j] = up[i,0] * down[0,j] * (alpha/rank)
            # alpha defaults to rank, so scale = 1. With up=2e38, down=1.0 → delta = 2e38.
            lora_sd = _lora_sd(
                name="lora_unet_block",
                rank=1, in_dim=2, out_dim=2,
                up=torch.full((2, 1), 2e38, dtype=torch.float32),
                down=torch.full((1, 2), 1.0, dtype=torch.float32),
            )
            base_sd = {"block.weight": torch.full((2, 2), 2e38, dtype=torch.float32)}
            lora_path = _save_sd(tmp, "lora.safetensors", lora_sd)
            base_path = _save_sd(tmp, "base.safetensors", base_sd)
            out_path = str(tmp / "folded.safetensors")
            config = _config(
                "--method", "linear", "--input", lora_path, "1.0",
                "--fold_into", base_path, "--output", out_path,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, r"non-finite folded tensor for module"):
                    mla.fold_adapters_into_base(config)
            # Atomic write contract: failed fold leaves no output file behind.
            self.assertFalse(os.path.exists(out_path))

    def test_fold_output_dtype_override_rejects_non_finite_after_cast(self) -> None:
        # Fold result is finite in fp32 (~1e5) but exceeds fp16 max (~65504), so the cast
        # to fp16 produces inf. The SECOND finite check (post-cast) must fire — distinct
        # failure mode from the first finite check (which guards fp32 overflow during add).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lora_sd = _lora_sd(name="lora_unet_block")  # tiny default delta (max ~33)
            base_sd = {"block.weight": torch.full((2, 3), 1e5, dtype=torch.float32)}
            lora_path = _save_sd(tmp, "lora.safetensors", lora_sd)
            base_path = _save_sd(tmp, "base.safetensors", base_sd)
            out_path = str(tmp / "folded.safetensors")
            config = _config(
                "--method", "linear", "--input", lora_path, "1.0",
                "--fold_into", base_path, "--output", out_path,
                "--output_dtype", "fp16",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, r"non-finite folded tensor after cast to torch\.float16"):
                    mla.fold_adapters_into_base(config)
            self.assertFalse(os.path.exists(out_path))

    def test_fold_output_file_contains_full_checkpoint_not_sparse_delta(self) -> None:
        # Catches a potential "fold mode wrote only modified tensors" regression. The
        # output is a full checkpoint, NOT a sparse adapter — every base key must be present.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_sd = {
                "block.weight": torch.zeros(2, 3),         # touched
                "untouched_a.weight": torch.ones(4),       # untouched, weight-suffixed
                "untouched_b": torch.tensor([42.0]),       # untouched, no .weight suffix
                "untouched_int": torch.tensor([1, 2], dtype=torch.int32),
            }
            result = self._run_fold(
                lora_sd=_lora_sd(name="lora_unet_block"),
                base_sd=base_sd,
                tmp=tmp,
            )

            loaded = load_file(result.output_path)
            self.assertEqual(set(loaded.keys()), set(base_sd.keys()))

    def test_fold_prints_plan_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lora_sd = _lora_sd(name="lora_unet_block")
            base_sd = {"block.weight": torch.zeros(2, 3), "other.weight": torch.zeros(4)}
            lora_path = _save_sd(tmp, "lora.safetensors", lora_sd)
            base_path = _save_sd(tmp, "base.safetensors", base_sd)
            out_path = str(tmp / "folded.safetensors")
            config = _config(
                "--method", "linear", "--input", lora_path, "1.0",
                "--fold_into", base_path, "--output", out_path,
            )
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                mla.fold_adapters_into_base(config)
            output = captured.getvalue()

            plan_pos = output.find("fold plan")
            summary_pos = output.find("fold summary")
            self.assertGreaterEqual(plan_pos, 0, f"missing 'fold plan' in stdout: {output!r}")
            self.assertGreaterEqual(summary_pos, 0, f"missing 'fold summary' in stdout: {output!r}")
            self.assertLess(plan_pos, summary_pos, "plan must precede summary")
            # "not delta-modified" wording (honest in both sentinel and explicit dtype mode).
            self.assertIn("not delta-modified", output)
            # Plan reports module + base-tensor count; summary reports folded/pruned/untouched.
            self.assertIn("modules resolved", output)
            self.assertIn("modules folded", output)

    def test_fold_plan_and_summary_prints_flush_for_long_running_cli(self) -> None:
        # Real fold smokes can spend minutes between preflight and atomic write. In
        # non-interactive subprocesses stdout is buffered, so the plan/summary lines must
        # flush explicitly or users won't see the preflight line until process exit.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lora_path = _save_sd(tmp, "lora.safetensors", _lora_sd(name="lora_unet_block"))
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3)})
            out_path = str(tmp / "folded.safetensors")
            config = _config(
                "--method", "linear", "--input", lora_path, "1.0",
                "--fold_into", base_path, "--output", out_path,
            )

            with patch("builtins.print") as mock_print:
                mla.fold_adapters_into_base(config)

            fold_calls = [
                call
                for call in mock_print.call_args_list
                if call.args and str(call.args[0]).startswith(("fold plan", "fold summary"))
            ]
            self.assertEqual(len(fold_calls), 2)
            for call in fold_calls:
                with self.subTest(message=call.args[0]):
                    self.assertIs(call.kwargs.get("flush"), True)

    def test_fold_writes_checkpoint_metadata_through_to_safetensors_file(self) -> None:
        # Integration seam test: FoldStats → build_fold_metadata → save_atomic → file metadata.
        # Helper-level tests can all pass while the WIRING between them silently drifts (e.g.
        # stats not threaded into the metadata call, or save_atomic dropping the metadata kwarg).
        # This test asserts the actual on-disk metadata matches what FoldResult reports.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lora_sd = _lora_sd(name="lora_unet_block")
            base_sd = {"block.weight": torch.zeros(2, 3), "other.weight": torch.zeros(4)}
            lora_path = _save_sd(tmp, "lora.safetensors", lora_sd)
            base_path = _save_sd(tmp, "base.safetensors", base_sd)
            out_path = str(tmp / "folded.safetensors")
            config = _config(
                "--method", "linear", "--input", lora_path, "1.0",
                "--fold_into", base_path, "--output", out_path,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = mla.fold_adapters_into_base(config)

            with safe_open(out_path, framework="pt") as f:
                file_metadata = f.metadata()

            # Spot-check the fold-specific fields landed correctly through the pipeline.
            self.assertEqual(file_metadata["ss_merge_output_format"], "checkpoint")
            self.assertEqual(file_metadata["ss_merge_modules_folded"], "1")
            self.assertEqual(file_metadata["ss_merge_modules_pruned"], "0")
            self.assertEqual(file_metadata["ss_merge_modules_resolved"], "1")
            self.assertEqual(file_metadata["ss_merge_base_basename"], "base.safetensors")
            self.assertEqual(file_metadata["ss_merge_base_sha256"], mla._file_sha256(base_path))
            self.assertEqual(file_metadata["ss_merge_base_tensors_total"], "2")
            self.assertEqual(file_metadata["ss_merge_base_tensors_modified"], "1")
            # Full-dict equality: catches drift in either direction (extra keys in file or lost keys in result).
            self.assertEqual(file_metadata, result.metadata)

    def test_run_dispatches_fold_mode_to_fold_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lora_path = _save_sd(tmp, "lora.safetensors", _lora_sd(name="lora_unet_block"))
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3)})
            out_path = str(tmp / "folded.safetensors")
            config = _config(
                "--method", "linear", "--input", lora_path, "1.0",
                "--fold_into", base_path, "--output", out_path,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = mla.run(config)

            self.assertIsInstance(result, mla.FoldResult)
            self.assertNotIsInstance(result, mla.MergeResult)
            self.assertTrue(os.path.exists(out_path))
            self.assertEqual(result.output_path, out_path)
            self.assertEqual(result.base_path, base_path)


class TestBaseDitDoraPipeline(unittest.TestCase):
    """End-to-end tests for v1.5 #4 --base_dit DoRA materialization.

    The load-bearing test is ``test_dora_materialization_matches_production_merge_path``
    — direct parity against the production runtime DoRA merge formula at
    ``src/musubi_tuner/utils/lora_utils.py:401-405``. If that test passes, v1.5 #4's
    DoRA math is correct. If it fails, the entire feature is wrong.
    """

    def test_dora_materialization_matches_production_merge_path(self) -> None:
        # THE load-bearing test — v1.5 #4 decision #14 lock at the math level.
        # Construct a known-good DoRA module and a known-good base tensor; compute the
        # delta two ways and assert byte-identity:
        #   Path 1: production formula (lora_utils.py:401-405) reimplemented inline
        #   Path 2: v1.5 #4 materialize_module_delta with dora_base_tensor
        rank, in_dim, out_dim = 2, 4, 3
        down = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
        up = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
        alpha = 4.0
        magnitude = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        base_weight = torch.tensor(
            [[0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]],
            dtype=torch.float32,
        )

        # Path 1: production formula
        scale = alpha / rank
        lora_delta = (up @ down) * scale
        weight_norm = mla.dora_weight_norm_materialized(base_weight, lora_delta, 1.0)
        dora_factor = magnitude / weight_norm
        merged_weight = dora_factor.view(-1, 1) * (base_weight + lora_delta)
        expected_delta = merged_weight - base_weight

        # Path 2: v1.5 #4 materialization
        with tempfile.TemporaryDirectory() as tmpdir:
            sd = _dora_lora_sd(
                name="lora_unet_test",
                rank=rank, in_dim=in_dim, out_dim=out_dim,
                alpha=alpha, down=down, up=up, magnitude=magnitude,
            )
            adapter = mla.load_adapter(_save_sd(Path(tmpdir), "dora.safetensors", sd), 1.0)
            actual_delta = mla.materialize_module_delta(
                adapter, "lora_unet_test", dora_base_tensor=base_weight
            )

        self.assertIsNotNone(actual_delta)
        self.assertTrue(
            torch.allclose(actual_delta, expected_delta, atol=1e-6),
            f"DoRA materialization parity broken: max diff = {(actual_delta - expected_delta).abs().max().item()}",
        )

    def test_dora_materialization_returns_delta_not_merged_weight(self) -> None:
        # Decision #14 explicit lock: materialize_module_delta MUST return DELTA
        # (merged - base), not merged_weight. Surgical assertion: compute both delta
        # and merged inline using the production formula, then assert the actual return
        # matches DELTA but NOT MERGED. Catches the off-by-one-concept bug where someone
        # returns merged_weight by mistake even when both have similar magnitudes.
        base_weight = torch.full((2, 3), 5.0, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            sd = _dora_lora_sd()  # default: in_dim=3, out_dim=2, rank=2
            adapter = mla.load_adapter(_save_sd(Path(tmpdir), "dora.safetensors", sd), 1.0)
            actual = mla.materialize_module_delta(adapter, "lora_unet_block", dora_base_tensor=base_weight)

        # Recompute production formula inline.
        info = adapter.modules["lora_unet_block"]
        down_t = adapter.state_dict["lora_unet_block.lora_down.weight"].float()
        up_t = adapter.state_dict["lora_unet_block.lora_up.weight"].float()
        scale = info.alpha / info.rank
        lora_delta = (up_t @ down_t) * scale
        magnitude = adapter.state_dict["lora_unet_block.dora_layer.weight"].float()
        weight_norm = mla.dora_weight_norm_materialized(base_weight, lora_delta, 1.0)
        dora_factor = magnitude / weight_norm
        expected_merged = dora_factor.view(-1, 1) * (base_weight + lora_delta)
        expected_delta = expected_merged - base_weight

        # The load-bearing assertion: actual matches DELTA, not MERGED.
        self.assertTrue(torch.allclose(actual, expected_delta, atol=1e-6))
        # Sanity: delta and merged are sufficiently different to make this test meaningful
        # (otherwise the test couldn't distinguish them and would silently pass).
        self.assertFalse(
            torch.allclose(actual, expected_merged, atol=0.01),
            "Test fixture is degenerate: delta ≈ merged, so the test cannot detect the bug",
        )

    def test_mixed_lora_and_dora_inputs_merge_uniformly(self) -> None:
        # Decision #5: a single invocation can mix standard LoRA and DoRA inputs in the
        # same --input list. Both become materialized deltas; combine_deltas treats them
        # uniformly. Proves the dora_base_lookup correctly returns None for non-DoRA
        # adapters and the resolved base tensor for DoRA adapters.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            standard_path = _save_sd(tmp, "standard.safetensors", _lora_sd())
            dora_path = _save_sd(tmp, "dora.safetensors", _dora_lora_sd())
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3, dtype=torch.float32)})
            out_path = str(tmp / "merged.safetensors")
            config = _config(
                "--method", "linear",
                "--input", standard_path, "1.0",
                "--input", dora_path, "1.0",
                "--base_dit", base_path,
                "--output", out_path,
                "--output_rank", "2",
            )
            result = mla.run(config)

            self.assertEqual(result.modules_written, 1)
            # Output exists and is a standard LoRA file (has lora_down/up/alpha keys).
            loaded = load_file(out_path)
            self.assertIn("lora_unet_block.lora_down.weight", loaded)
            self.assertIn("lora_unet_block.lora_up.weight", loaded)
            self.assertIn("lora_unet_block.alpha", loaded)

    def test_dora_output_does_not_include_use_dora_flag_or_dora_layer_keys(self) -> None:
        # Decision #13: standard-LoRA output by definition has no DoRA marker tensors.
        # The SVD recompression naturally produces only lora_down/up/alpha keys, so this
        # is structurally enforced — but locking it as a test prevents any future
        # "preserve metadata" regression that might forward DoRA tensors.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dora_path = _save_sd(tmp, "dora.safetensors", _dora_lora_sd())
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3, dtype=torch.float32)})
            out_path = str(tmp / "merged.safetensors")
            config = _config(
                "--method", "linear",
                "--input", dora_path, "1.0",
                "--base_dit", base_path,
                "--output", out_path,
                "--output_rank", "2",
            )
            mla.run(config)

            loaded = load_file(out_path)
            for key in loaded:
                with self.subTest(key=key):
                    self.assertFalse(key.endswith(".dora_layer.weight"), f"DoRA magnitude key leaked: {key}")
                    self.assertNotIn("use_dora_flag", key, f"DoRA flag key leaked: {key}")

    def test_dora_metadata_records_base_dit_basename_sha256_and_inputs_flag(self) -> None:
        # Decision #10 metadata contract: ss_merge_dora_inputs aggregate flag,
        # ss_merge_dora_output_format, ss_merge_base_dit_basename (privacy-safe),
        # ss_merge_base_dit_sha256, plus per-input "format" field in ss_merge_inputs.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dora_path = _save_sd(tmp, "dora.safetensors", _dora_lora_sd())
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3, dtype=torch.float32)})
            out_path = str(tmp / "merged.safetensors")
            config = _config(
                "--method", "linear",
                "--input", dora_path, "1.0",
                "--base_dit", base_path,
                "--output", out_path,
                "--output_rank", "2",
            )
            mla.run(config)

            with safe_open(out_path, framework="pt") as f:
                meta = f.metadata()

            self.assertEqual(meta["ss_merge_dora_inputs"], "true")
            self.assertEqual(meta["ss_merge_dora_output_format"], "standard_lora")
            self.assertEqual(meta["ss_merge_base_dit_basename"], "base.safetensors")
            self.assertEqual(meta["ss_merge_base_dit_sha256"], mla._file_sha256(base_path))
            # Privacy: no absolute paths leaked.
            self.assertNotIn(tmpdir, meta["ss_merge_base_dit_basename"])
            # Per-input "format" field present and correctly tagged.
            inputs = json.loads(meta["ss_merge_inputs"])
            self.assertEqual(inputs[0]["format"], "dora")

    def test_dora_with_split_safetensors_base_dit(self) -> None:
        # Decision #7: --base_dit reuses _load_base_as_stored, which handles single-file
        # AND multi-shard bases. Pass any shard, the helper expands to the canonical set.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dora_path = _save_sd(tmp, "dora.safetensors", _dora_lora_sd())
            shard_1 = _write_split_shard(tmp, "base", 1, 2, {"block.weight": torch.zeros(2, 3, dtype=torch.float32)})
            _write_split_shard(tmp, "base", 2, 2, {"other.weight": torch.zeros(4, dtype=torch.float32)})
            out_path = str(tmp / "merged.safetensors")
            config = _config(
                "--method", "linear",
                "--input", dora_path, "1.0",
                "--base_dit", shard_1,
                "--output", out_path,
                "--output_rank", "2",
            )
            mla.run(config)

            with safe_open(out_path, framework="pt") as f:
                meta = f.metadata()
            # Hash recorded is the COMPOSITE hash, not the bare file SHA of shard 1.
            self.assertEqual(meta["ss_merge_base_dit_sha256"], mla._composite_base_hash(shard_1))
            self.assertNotEqual(meta["ss_merge_base_dit_sha256"], mla._file_sha256(shard_1))

    def test_dora_metadata_does_not_claim_rejects_dora_when_dora_input_present(self) -> None:
        # Finding 3 fix: ss_merge_rejects_dora must NOT be "true" in DoRA-output runs.
        # The pre-fix behavior unconditionally wrote "true", which contradicted
        # ss_merge_dora_inputs="true" and was false provenance. This test pins the
        # consistent metadata contract. Standard-LoRA-only runs still record "true"
        # for backward compat — a separate test would verify that, but it's already
        # implicit in TestMetadata's existing "key is present" check.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dora_path = _save_sd(tmp, "dora.safetensors", _dora_lora_sd())
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3, dtype=torch.float32)})
            out_path = str(tmp / "merged.safetensors")
            config = _config(
                "--method", "linear",
                "--input", dora_path, "1.0",
                "--base_dit", base_path,
                "--output", out_path,
                "--output_rank", "2",
            )
            mla.run(config)

            with safe_open(out_path, framework="pt") as f:
                meta = f.metadata()

            self.assertEqual(meta["ss_merge_dora_inputs"], "true")
            self.assertEqual(meta["ss_merge_rejects_dora"], "false")

    def test_fp8_base_dit_error_message_names_correct_flag(self) -> None:
        # Finding 4 fix: _assert_no_fp8_in_base now takes a flag_name parameter so
        # the DoRA path's error message says "--base_dit" instead of the fold-mode
        # default "--fold_into". Direct unit test on the helper.
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch build lacks float8_e4m3fn")
        sd = {"fc.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn)}
        with self.assertRaisesRegex(ValueError, r"--base_dit base 'base\.safetensors'"):
            mla._assert_no_fp8_in_base(sd, "/some/dir/base.safetensors", flag_name="--base_dit")
        # And the default still says --fold_into.
        with self.assertRaisesRegex(ValueError, r"--fold_into base 'base\.safetensors'"):
            mla._assert_no_fp8_in_base(sd, "/some/dir/base.safetensors")

    def test_dora_input_with_output_use_rslora_writes_rslora_standard_lora_output(self) -> None:
        # Decision #16: DoRA inputs are compatible with --output_use_rslora because output
        # is LoRA-shaped regardless of input mode. DoRA factors become materialized deltas
        # first; rsLoRA scaling is a post-SVD output convention. Output has rsLoRA flag
        # AND no DoRA markers (decision #13 still applies).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dora_path = _save_sd(tmp, "dora.safetensors", _dora_lora_sd())
            base_path = _save_sd(tmp, "base.safetensors", {"block.weight": torch.zeros(2, 3, dtype=torch.float32)})
            out_path = str(tmp / "merged.safetensors")
            config = _config(
                "--method", "linear",
                "--input", dora_path, "1.0",
                "--base_dit", base_path,
                "--output", out_path,
                "--output_rank", "2",
                "--output_use_rslora",
            )
            mla.run(config)

            loaded = load_file(out_path)
        # rsLoRA flag present.
        self.assertIn("use_rslora_flag", loaded)
        self.assertTrue(bool(loaded["use_rslora_flag"].item()))
        # No DoRA markers (decision #13).
        for key in loaded:
            self.assertFalse(key.endswith(".dora_layer.weight"))
            self.assertNotIn("use_dora_flag", key)


class TestFoldResultShape(unittest.TestCase):
    """Pins the FoldResult dataclass shape — load-bearing memory-footprint design."""

    def test_fold_result_does_not_carry_state_dict(self) -> None:
        # Critical memory-footprint invariant: FoldResult must NOT retain the full base
        # checkpoint after save_atomic completes. A field named state_dict (or any other
        # full-tensor-dict shape) would keep the checkpoint alive in the run() caller's
        # frame just because the result is returned. The whole point of separating
        # FoldResult from MergeResult.
        fields = {f.name for f in dataclasses.fields(mla.FoldResult)}
        for forbidden in ("state_dict", "base_sd", "tensors", "checkpoint"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, fields)

    def test_fold_result_carries_output_path_metadata_stats_base_path(self) -> None:
        stats = mla.FoldStats(0, 0, 0, 0, 0)
        result = mla.FoldResult(
            output_path="/some/path/folded.safetensors",
            metadata={"ss_merge_tool": "test"},
            stats=stats,
            base_path="/some/path/base.safetensors",
        )
        self.assertEqual(result.output_path, "/some/path/folded.safetensors")
        self.assertEqual(result.metadata, {"ss_merge_tool": "test"})
        self.assertIs(result.stats, stats)
        self.assertEqual(result.base_path, "/some/path/base.safetensors")


if __name__ == "__main__":
    unittest.main()
