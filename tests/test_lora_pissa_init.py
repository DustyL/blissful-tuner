"""Math parity + contract tests for PiSSA initialization helper.

Pins blissful-tuner's _init_pissa_lora_pair against PEFT's pissa_init
(/home/dustin/peft/src/peft/tuners/lora/layer.py:360-393) on synthetic
Linear weights. The helper is the contract surface for Tier 2 #6b
PiSSA — every other commit (LoRAModule wiring, hash hard-reject,
parser rejects) builds on the math being correct here.

Structure mirrors tests/test_compute_training_base_hash.py: small
self-contained fixtures, deterministic seeds, every test isolatable
in <1s.
"""

from __future__ import annotations

import re
import unittest

import torch

from musubi_tuner.networks.lora import (
    _init_pissa_lora_pair,
    parse_init_lora_weights_arg,
)


def _make_lora_pair(in_dim: int, out_dim: int, rank: int, dtype: torch.dtype = torch.float32):
    """Synthetic Linear-shaped lora_down / lora_up pair, matching
    LoRAModule's allocation pattern (lora_down: (rank, in_dim),
    lora_up: (out_dim, rank), bias=False)."""
    lora_down = torch.nn.Linear(in_dim, rank, bias=False, dtype=dtype)
    lora_up = torch.nn.Linear(rank, out_dim, bias=False, dtype=dtype)
    return lora_down, lora_up


def _make_base_weight(in_dim: int, out_dim: int, *, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Linear weight shape: (out_dim, in_dim)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(out_dim, in_dim, generator=g, dtype=dtype)


# ---------------------------------------------------------------------------
# parse_init_lora_weights_arg extension (Fork 2)
# ---------------------------------------------------------------------------


class TestParseInitLoraWeightsArgPissaExtension(unittest.TestCase):
    """The string-arg parser must accept the PEFT-compatible spellings."""

    def test_pissa_value_normalized(self) -> None:
        self.assertEqual(parse_init_lora_weights_arg("pissa"), "pissa")
        self.assertEqual(parse_init_lora_weights_arg("PISSA"), "pissa")
        self.assertEqual(parse_init_lora_weights_arg(" pissa "), "pissa")

    def test_pissa_niter_value_preserves_n(self) -> None:
        self.assertEqual(parse_init_lora_weights_arg("pissa_niter_5"), "pissa_niter_5")
        self.assertEqual(parse_init_lora_weights_arg("PISSA_NITER_20"), "pissa_niter_20")
        self.assertEqual(parse_init_lora_weights_arg("pissa_niter_1"), "pissa_niter_1")

    def test_pissa_niter_zero_or_negative_rejects(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            parse_init_lora_weights_arg("pissa_niter_0")

    def test_pissa_niter_garbage_rejects(self) -> None:
        for bad in ("pissa_niter_", "pissa_niter_x", "pissa_niter_5.5", "pissa_niter_-3", "pissa5", "pissaniter5"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    parse_init_lora_weights_arg(bad)

    def test_existing_values_unchanged(self) -> None:
        """Back-compat invariant: existing accepted values still parse."""
        self.assertEqual(parse_init_lora_weights_arg(None), "kaiming")
        self.assertEqual(parse_init_lora_weights_arg(True), "kaiming")
        self.assertEqual(parse_init_lora_weights_arg("true"), "kaiming")
        self.assertEqual(parse_init_lora_weights_arg("kaiming"), "kaiming")
        self.assertEqual(parse_init_lora_weights_arg("orthogonal"), "orthogonal")


# ---------------------------------------------------------------------------
# Math parity vs PEFT (Fork 5)
# ---------------------------------------------------------------------------


def _peft_pissa_reference(
    base_weight: torch.Tensor, rank: int, scaling: float, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce PEFT's pissa_init math on the same inputs.

    Returns (lora_A, lora_B, residual) so blissful-tuner's helper output
    can be compared element-wise. Avoids importing PEFT directly so this
    test runs even if PEFT is not installed; PEFT's reference algorithm
    is small enough to inline. The inline copy is line-for-line identical
    to peft/src/peft/tuners/lora/layer.py:368-393 modulo the pure-function
    return shape.
    """
    weight_fp32 = base_weight.to(torch.float32)
    if mode == "pissa":
        V, S, Uh = torch.linalg.svd(weight_fp32, full_matrices=False)
        Vr = V[:, :rank]
        Sr = S[:rank].clone()
        Sr /= scaling
        Uhr = Uh[:rank, :]
    else:
        m = re.match(r"^pissa_niter_(\d+)$", mode)
        assert m is not None, f"Bad mode: {mode}"
        niter = int(m.group(1))
        Vr, Sr, Ur = torch.svd_lowrank(weight_fp32, rank, niter=niter)
        Sr = Sr / scaling
        Uhr = Ur.t()
    sqrt_sr = torch.sqrt(Sr)
    lora_A = torch.diag(sqrt_sr) @ Uhr
    lora_B = Vr @ torch.diag(sqrt_sr)
    residual = weight_fp32 - scaling * lora_B @ lora_A
    return lora_A, lora_B, residual


class TestPissaMathParityVsPeft(unittest.TestCase):
    """Element-wise parity between blissful-tuner's helper and PEFT's
    reference pissa_init on identical inputs. SVD is sign-flip-equivalent
    so we compare via reconstruction (the residual + scale*B@A equals
    the original) rather than direct A/B equality."""

    def test_full_svd_pissa_matches_peft_via_reconstruction(self) -> None:
        torch.manual_seed(42)
        base = _make_base_weight(in_dim=8, out_dim=4, seed=42)
        rank = 2
        scaling = 1.0  # alpha=rank=2 standard LoRA scale

        # Blissful-tuner helper
        lora_down, lora_up = _make_lora_pair(in_dim=8, out_dim=4, rank=rank)
        bt_residual = _init_pissa_lora_pair(lora_down, lora_up, base, rank, scaling, "pissa")

        # PEFT reference on identical inputs
        peft_A, peft_B, peft_residual = _peft_pissa_reference(base, rank, scaling, "pissa")

        # Reconstruction check: helper output reconstructs the same original weight
        bt_reconstructed = bt_residual.to(torch.float32) + scaling * lora_up.weight.to(torch.float32) @ lora_down.weight.to(torch.float32)
        peft_reconstructed = peft_residual + scaling * peft_B @ peft_A

        self.assertTrue(torch.allclose(bt_reconstructed, base.to(torch.float32), rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(peft_reconstructed, base.to(torch.float32), rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(bt_residual.to(torch.float32), peft_residual, rtol=1e-5, atol=1e-5))

    def test_full_svd_pissa_lora_factors_equal_peft_modulo_sign(self) -> None:
        """SVD has a sign-flip ambiguity per singular vector; lora_A and lora_B
        from the two implementations may differ by per-rank sign flips. The
        reconstruction is sign-invariant. Pin the per-rank-magnitude equivalence
        as a stronger pin than reconstruction alone."""
        torch.manual_seed(0)
        base = _make_base_weight(in_dim=16, out_dim=8, seed=11)
        rank = 4
        scaling = 0.5

        lora_down, lora_up = _make_lora_pair(in_dim=16, out_dim=8, rank=rank)
        _init_pissa_lora_pair(lora_down, lora_up, base, rank, scaling, "pissa")
        bt_A = lora_down.weight.detach().to(torch.float32)
        bt_B = lora_up.weight.detach().to(torch.float32)

        peft_A, peft_B, _ = _peft_pissa_reference(base, rank, scaling, "pissa")

        # Per-rank magnitude (Frobenius norm of each row of A or column of B) is sign-invariant
        for r in range(rank):
            self.assertAlmostEqual(bt_A[r].norm().item(), peft_A[r].norm().item(), places=5)
            self.assertAlmostEqual(bt_B[:, r].norm().item(), peft_B[:, r].norm().item(), places=5)

    def test_pissa_niter_5_matches_peft_via_reconstruction(self) -> None:
        """Newton-Schulz approximate SVD path. Wider tolerance — niter=5 is
        approximate. Both implementations call torch.svd_lowrank with the same
        niter, but the underlying torch.randn-driven init may make the per-call
        result non-deterministic without seeding. Use a fresh seed at each
        call to keep the pair comparable."""
        torch.manual_seed(7)
        base = _make_base_weight(in_dim=32, out_dim=16, seed=99)
        rank = 4
        scaling = 1.0

        # NOTE: torch.svd_lowrank uses an internal random init. We seed before
        # each call to match.
        lora_down, lora_up = _make_lora_pair(in_dim=32, out_dim=16, rank=rank)
        torch.manual_seed(7)
        bt_residual = _init_pissa_lora_pair(lora_down, lora_up, base, rank, scaling, "pissa_niter_5")

        torch.manual_seed(7)
        peft_A, peft_B, peft_residual = _peft_pissa_reference(base, rank, scaling, "pissa_niter_5")

        bt_reconstructed = bt_residual.to(torch.float32) + scaling * lora_up.weight.to(torch.float32) @ lora_down.weight.to(torch.float32)

        # niter=5 reconstruction is approximate — wider tolerance
        self.assertTrue(torch.allclose(bt_reconstructed, base.to(torch.float32), rtol=1e-3, atol=1e-3))
        # And the two implementations should agree to fp32 precision when seeded identically
        self.assertTrue(torch.allclose(bt_residual.to(torch.float32), peft_residual, rtol=1e-5, atol=1e-5))


# ---------------------------------------------------------------------------
# In-place mutation contract (test bar item 2)
# ---------------------------------------------------------------------------


class TestPissaInPlaceMutationContract(unittest.TestCase):
    """The helper does NOT mutate base_weight in place — it returns the
    residual for the caller to write back. This is the design decision
    that keeps the helper pure (testable in isolation). The caller
    (LoRAModule.__init__, commit 2) is responsible for the actual
    org_module.weight.data mutation."""

    def test_helper_does_not_mutate_input_base_weight(self) -> None:
        base = _make_base_weight(in_dim=8, out_dim=4, seed=1)
        snapshot = base.clone()
        lora_down, lora_up = _make_lora_pair(in_dim=8, out_dim=4, rank=2)

        _ = _init_pissa_lora_pair(lora_down, lora_up, base, rank=2, scaling=1.0, mode="pissa")

        self.assertTrue(torch.equal(base, snapshot), "helper unexpectedly mutated input base_weight")

    def test_returned_residual_equals_base_minus_scale_times_BA(self) -> None:
        """The whole point of returning the residual: applying the standard
        LoRA forward against (residual + scale * B @ A) reconstructs the
        original base. Pin this equation directly."""
        base = _make_base_weight(in_dim=12, out_dim=6, seed=3)
        rank = 3
        scaling = 0.75
        lora_down, lora_up = _make_lora_pair(in_dim=12, out_dim=6, rank=rank)

        residual = _init_pissa_lora_pair(lora_down, lora_up, base, rank, scaling, "pissa")

        # Reconstruct: residual + scale * (B @ A) should equal original base
        reconstructed = residual.to(torch.float32) + scaling * lora_up.weight.to(torch.float32) @ lora_down.weight.to(torch.float32)
        self.assertTrue(torch.allclose(reconstructed, base.to(torch.float32), rtol=1e-5, atol=1e-5))

    def test_lora_factors_have_expected_shapes(self) -> None:
        """lora_down: (rank, in_features), lora_up: (out_features, rank).
        Pin the shape contract because PEFT and blissful-tuner agree but
        a future refactor that swaps A/B conventions would silently break
        the merge math everywhere."""
        in_dim, out_dim, rank = 24, 12, 4
        base = _make_base_weight(in_dim=in_dim, out_dim=out_dim, seed=5)
        lora_down, lora_up = _make_lora_pair(in_dim=in_dim, out_dim=out_dim, rank=rank)

        _init_pissa_lora_pair(lora_down, lora_up, base, rank, scaling=1.0, mode="pissa")

        self.assertEqual(lora_down.weight.shape, (rank, in_dim))
        self.assertEqual(lora_up.weight.shape, (out_dim, rank))


# ---------------------------------------------------------------------------
# Forward equivalence at step 0 (test bar item 3)
# ---------------------------------------------------------------------------


class TestPissaForwardEquivalenceAtStepZero(unittest.TestCase):
    """The defining empirical invariant of PiSSA: at step 0, the model
    output is identical (within fp32 tolerance) to the un-LoRA'd base
    forward. Training perturbs lora_A/lora_B from this principal-rank-r
    init, which is what gives PiSSA its fast-convergence property.

    Test simulates the LoRA forward pass: `out = x @ residual^T + scale * (x @ A^T) @ B^T`
    (matches blissful-tuner's existing LoRAModule forward semantics)."""

    def test_forward_equals_original_at_step_zero_full_svd(self) -> None:
        torch.manual_seed(11)
        in_dim, out_dim, rank = 16, 8, 4
        scaling = 1.0
        base = _make_base_weight(in_dim=in_dim, out_dim=out_dim, seed=22)
        x = torch.randn(3, in_dim, dtype=torch.float32)  # batch=3
        original_out = x @ base.t()

        lora_down, lora_up = _make_lora_pair(in_dim=in_dim, out_dim=out_dim, rank=rank)
        residual = _init_pissa_lora_pair(lora_down, lora_up, base, rank, scaling, "pissa")

        # Standard LoRA forward: residual.forward(x) + scale * up.forward(down.forward(x))
        residual_out = x @ residual.to(torch.float32).t()
        lora_path_out = scaling * lora_up(lora_down(x)).to(torch.float32)
        pissa_out = residual_out + lora_path_out

        self.assertTrue(torch.allclose(pissa_out, original_out, rtol=1e-5, atol=1e-5))

    def test_forward_equivalence_with_rslora_scaling(self) -> None:
        """rsLoRA changes the scaling formula (alpha/sqrt(r) instead of
        alpha/r). PiSSA must respect whatever scale is passed — the
        forward equivalence holds for either."""
        torch.manual_seed(13)
        in_dim, out_dim, rank = 16, 8, 4
        alpha = 4.0
        rslora_scaling = alpha / (rank ** 0.5)  # rsLoRA scale
        base = _make_base_weight(in_dim=in_dim, out_dim=out_dim, seed=33)
        x = torch.randn(2, in_dim, dtype=torch.float32)
        original_out = x @ base.t()

        lora_down, lora_up = _make_lora_pair(in_dim=in_dim, out_dim=out_dim, rank=rank)
        residual = _init_pissa_lora_pair(lora_down, lora_up, base, rank, rslora_scaling, "pissa")

        residual_out = x @ residual.to(torch.float32).t()
        lora_path_out = rslora_scaling * lora_up(lora_down(x)).to(torch.float32)
        pissa_out = residual_out + lora_path_out

        self.assertTrue(torch.allclose(pissa_out, original_out, rtol=1e-5, atol=1e-5))


# ---------------------------------------------------------------------------
# dtype + shape rejects (test bar items 4, 11)
# ---------------------------------------------------------------------------


class TestPissaInputValidation(unittest.TestCase):
    """The helper enforces two invariants at the boundary: 2D base weight
    (Conv2d 4D rejected) and fp32/fp16/bf16 base dtype (PEFT mirror)."""

    def test_conv2d_4d_weight_hard_rejects(self) -> None:
        """Conv2d weights are (out, in, kH, kW). PEFT's pissa_init crashes on
        these at torch.linalg.svd. Blissful-tuner gives an actionable error."""
        # Synthetic 4D weight (Conv2d shape)
        base_4d = torch.randn(8, 4, 3, 3)
        lora_down, lora_up = _make_lora_pair(in_dim=4, out_dim=8, rank=2)

        with self.assertRaises(NotImplementedError) as ctx:
            _init_pissa_lora_pair(lora_down, lora_up, base_4d, rank=2, scaling=1.0, mode="pissa")
        self.assertIn("2D", str(ctx.exception))
        self.assertIn("Conv2d", str(ctx.exception))

    def test_unsupported_dtype_hard_rejects(self) -> None:
        """fp64 is not in the allowed set even though SVD would work — the
        contract mirrors PEFT's exact dtype filter for forward-compat with
        any fp8/int8 base that might hit this code path."""
        base_fp64 = _make_base_weight(in_dim=8, out_dim=4, seed=1, dtype=torch.float64)
        lora_down, lora_up = _make_lora_pair(in_dim=8, out_dim=4, rank=2, dtype=torch.float32)

        with self.assertRaises(TypeError) as ctx:
            _init_pissa_lora_pair(lora_down, lora_up, base_fp64, rank=2, scaling=1.0, mode="pissa")
        self.assertIn("float32/float16/bfloat16", str(ctx.exception))

    def test_invalid_mode_string_hard_rejects(self) -> None:
        """Defense in depth: the public-API parse_init_lora_weights_arg
        gates string formatting, but the helper itself should still reject
        garbage if called directly via internal code."""
        base = _make_base_weight(in_dim=8, out_dim=4, seed=1)
        lora_down, lora_up = _make_lora_pair(in_dim=8, out_dim=4, rank=2)

        for bad_mode in ("pissa_v2", "PISSA", "kaiming"):
            with self.subTest(mode=bad_mode):
                with self.assertRaises(ValueError):
                    _init_pissa_lora_pair(lora_down, lora_up, base, rank=2, scaling=1.0, mode=bad_mode)


# ---------------------------------------------------------------------------
# bf16 base support (PEFT compat)
# ---------------------------------------------------------------------------


class TestPissaBfloat16BaseSupport(unittest.TestCase):
    """Blissful-tuner trains in bf16 widely. PiSSA must work on bf16 base
    weights (with the SVD computed in fp32 internally and cast back)."""

    def test_bf16_base_round_trips_through_fp32_svd(self) -> None:
        in_dim, out_dim, rank = 16, 8, 4
        base_bf16 = _make_base_weight(in_dim=in_dim, out_dim=out_dim, seed=7, dtype=torch.bfloat16)
        lora_down, lora_up = _make_lora_pair(in_dim=in_dim, out_dim=out_dim, rank=rank, dtype=torch.bfloat16)

        residual = _init_pissa_lora_pair(lora_down, lora_up, base_bf16, rank, scaling=1.0, mode="pissa")

        # Output dtype matches base
        self.assertEqual(residual.dtype, torch.bfloat16)
        self.assertEqual(lora_down.weight.dtype, torch.bfloat16)
        self.assertEqual(lora_up.weight.dtype, torch.bfloat16)

        # Reconstruction in bf16: wider tolerance than fp32
        reconstructed = residual.to(torch.float32) + lora_up.weight.to(torch.float32) @ lora_down.weight.to(torch.float32)
        self.assertTrue(torch.allclose(reconstructed, base_bf16.to(torch.float32), rtol=1e-2, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
