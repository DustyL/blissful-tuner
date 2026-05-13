"""Tests for optional Liger RMSNorm + SiLU-gated MLP in flux2_models.

The Liger code path is opt-in via BLISSFUL_USE_LIGER_FLUX2=1 and CUDA-only
(Triton). State-dict tests run on CPU and verify that the saved-parameter
contract (`scale` key, not `weight`) holds regardless of whether the Liger
gate was on when the module was instantiated — this is what protects
existing FLUX.2-Klein checkpoints from silent corruption.

Numerical parity tests run only on CUDA and only when liger-kernel is
importable. They compare the Liger-active forward output against the
fallback forward output, asserting closeness within bf16-friendly
tolerance.
"""

import importlib
import os
import unittest

import torch

from musubi_tuner.flux_2 import flux2_models


CUDA_AVAILABLE = torch.cuda.is_available()
LIGER_IMPORTABLE = flux2_models._LIGER_AVAILABLE
# Liger CUDA tests trigger Triton JIT + GPU kernels. Per the repo's local-safe
# policy, default pytest runs should not silently fire those. Opt-in via env var.
CUDA_TESTS_ENABLED = os.environ.get("BLISSFUL_RUN_CUDA_TESTS", "0") == "1"


def _reload_with_env(env_value: str) -> object:
    """Reload flux2_models with a specific BLISSFUL_USE_LIGER_FLUX2 value so
    we can flip the module-level _LIGER_ENABLED constant for a single test
    without contaminating the rest of the suite."""
    prior = os.environ.get("BLISSFUL_USE_LIGER_FLUX2")
    os.environ["BLISSFUL_USE_LIGER_FLUX2"] = env_value
    try:
        reloaded = importlib.reload(flux2_models)
    finally:
        if prior is None:
            os.environ.pop("BLISSFUL_USE_LIGER_FLUX2", None)
        else:
            os.environ["BLISSFUL_USE_LIGER_FLUX2"] = prior
    return reloaded


class RMSNormStateDictCompat(unittest.TestCase):
    """The load-bearing property: saved checkpoints use `scale` as the
    parameter key regardless of whether Liger is active at save time, and
    load cleanly under either backend at load time."""

    def test_fallback_state_dict_has_scale_key(self):
        norm = flux2_models.RMSNorm(16)
        sd = norm.state_dict()
        self.assertIn("scale", sd)
        self.assertNotIn("weight", sd, "RMSNorm must persist as 'scale', never 'weight'")

    def test_fallback_load_from_scale_state_dict(self):
        norm_a = flux2_models.RMSNorm(16)
        with torch.no_grad():
            norm_a.scale.normal_()
        sd = norm_a.state_dict()

        norm_b = flux2_models.RMSNorm(16)
        norm_b.load_state_dict(sd)
        self.assertTrue(torch.equal(norm_a.scale, norm_b.scale))

    @unittest.skipUnless(LIGER_IMPORTABLE, "liger-kernel not installed")
    def test_liger_active_state_dict_has_scale_key(self):
        """When Liger is gated on, the saved state_dict must STILL use
        `scale` — otherwise existing checkpoints would silently fail to
        load. The implementation uses LigerRMSNormFunction (the underlying
        autograd.Function) directly with `self.scale` as the weight
        argument, so the parameter name is unaffected by the gate."""
        reloaded = _reload_with_env("1")
        try:
            self.assertTrue(reloaded._LIGER_ENABLED)
            norm = reloaded.RMSNorm(16)
            sd = norm.state_dict()
            self.assertIn("scale", sd)
            self.assertNotIn("weight", sd)
        finally:
            # Reset the module back to disabled state.
            _reload_with_env("0")


class SiLUActivationFallbackParity(unittest.TestCase):
    """The fallback SiLUActivation path must produce identical output to
    the original (pre-port) implementation `nn.SiLU()(x1) * x2`. Pins the
    fallback equivalence so a future refactor cannot drift the no-Liger
    path."""

    def test_fallback_matches_reference_silu_gated(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16)

        act = flux2_models.SiLUActivation()
        actual = act(x)

        x1, x2 = x.chunk(2, dim=-1)
        reference = torch.nn.functional.silu(x1) * x2

        self.assertTrue(torch.equal(actual, reference))


@unittest.skipUnless(
    CUDA_AVAILABLE and LIGER_IMPORTABLE and CUDA_TESTS_ENABLED,
    "Liger active path requires CUDA + liger-kernel + BLISSFUL_RUN_CUDA_TESTS=1",
)
class LigerActiveNumericalParity(unittest.TestCase):
    """Compare the Liger-active forward against the fallback forward.
    Tolerance is loose-ish because Liger uses fused Triton kernels with
    slightly different floating-point ordering — bf16-friendly bounds."""

    def setUp(self):
        self.reloaded = _reload_with_env("1")
        self.assertTrue(self.reloaded._LIGER_ENABLED)
        self.assertTrue(CUDA_AVAILABLE)

    def tearDown(self):
        _reload_with_env("0")

    def test_rmsnorm_liger_matches_fallback_within_tolerance(self):
        torch.manual_seed(0)
        dim = 32
        x = torch.randn(2, 8, dim, device="cuda", dtype=torch.float32)
        scale = torch.randn(dim, device="cuda", dtype=torch.float32) * 0.1 + 1.0

        # Liger-active path (re-imported with gate on).
        norm_liger = self.reloaded.RMSNorm(dim).cuda()
        with torch.no_grad():
            norm_liger.scale.copy_(scale)
        out_liger = norm_liger(x)

        # Fallback formula (computed inline so we don't have to wrestle
        # with re-disabling the gate just to call the fallback).
        x_dtype = x.dtype
        xf = x.float()
        rrms = torch.rsqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
        out_fallback = (xf * rrms).to(dtype=x_dtype) * scale

        max_diff = (out_liger - out_fallback).abs().max().item()
        self.assertLess(max_diff, 1e-3, f"RMSNorm parity drift too large: max abs diff {max_diff:.2e}")

    def test_silu_gated_liger_matches_fallback_within_tolerance(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32)

        act = self.reloaded.SiLUActivation()
        out_liger = act(x)

        x1, x2 = x.chunk(2, dim=-1)
        out_fallback = torch.nn.functional.silu(x1) * x2

        max_diff = (out_liger - out_fallback).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"SiLU-gated parity drift too large: max abs diff {max_diff:.2e}")

    def test_rmsnorm_liger_backward_produces_scale_gradient(self):
        """End-to-end sanity: Liger autograd reaches the scale parameter."""
        torch.manual_seed(0)
        dim = 16
        norm = self.reloaded.RMSNorm(dim).cuda()
        x = torch.randn(2, 4, dim, device="cuda", requires_grad=True)
        loss = norm(x).sum()
        loss.backward()
        self.assertIsNotNone(norm.scale.grad)
        self.assertGreater(norm.scale.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
