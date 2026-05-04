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

    def test_fold_run_guard_raises_not_implemented(self) -> None:
        config = _config(
            "--method", "linear", "--input", "a.safetensors", "1.0",
            "--fold_into", "base.safetensors", "--output", "folded.safetensors",
        )
        with self.assertRaisesRegex(NotImplementedError, r"--fold_into execution is not yet implemented"):
            mla.run(config)

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

if __name__ == "__main__":
    unittest.main()
