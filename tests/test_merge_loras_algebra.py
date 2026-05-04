"""Regression tests for the offline LoRA merge algebra CLI."""

from __future__ import annotations

import contextlib
import io
import json
import math
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
    def test_dora_input_rejected_with_actionable_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_sd(Path(tmpdir), "dora.safetensors", _lora_sd() | {"use_dora_flag": torch.tensor(True)})

            with self.assertRaises(ValueError) as cm:
                mla.load_adapter(path, 1.0)

        self.assertIn("DoRA adapter merge algebra", str(cm.exception))
        self.assertIn("Rejected input: dora.safetensors", str(cm.exception))

    def test_dora_cli_rejection_has_no_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_sd(Path(tmpdir), "dora.safetensors", _lora_sd() | {"use_dora_flag": torch.tensor(True)})

            with self.assertRaises(SystemExit) as cm:
                mla.main(["--method", "linear", "--input", path, "1.0", "--preview_spectrum"])

        self.assertIn("DoRA adapter merge algebra", str(cm.exception))
        self.assertNotIn("Traceback", str(cm.exception))

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
        self.assertEqual(metadata["ss_merge_match_semantics"], mla.MATCH_SEMANTICS)
        self.assertEqual(metadata["ss_merge_recompression"], mla.RECOMPRESSION_SEMANTICS)

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

            def wrapped_materialize(adapter: mla.AdapterInfo, module_name: str) -> torch.Tensor | None:
                live = [ref for ref in refs if ref() is not None]
                self.assertLessEqual(len(live), len(config.inputs))
                delta = original(adapter, module_name)
                if delta is not None:
                    refs.append(weakref.ref(delta))
                return delta

            with patch.object(mla, "materialize_module_delta", side_effect=wrapped_materialize):
                mla.merge_adapters(config, module_callback=callback)

        self.assertEqual(seen, ["lora_unet_a", "lora_unet_b", "lora_unet_c"])


if __name__ == "__main__":
    unittest.main()
