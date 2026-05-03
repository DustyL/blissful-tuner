"""Regression tests for the DoRA init device-mismatch bug.

In the FLUX.2 trainer flow the base DiT is moved to CUDA via
`transformer.move_to_device_except_swap_blocks()` *before* the LoRA network's
`apply_to()` runs DoRA initialization (see `hv_train_network.py:2041` and
`:2127`). Because the freshly-created LoRA Linear layers and the `DoRALayer`
magnitude vector live on CPU at that point, the materialized `B @ A` path
used to crash with:

    RuntimeError: Expected all tensors to be on the same device,
    but found at least two devices, cuda:0 and cpu!

The fix in `DoRALayer.get_weight_norm_materialized()` casts `lora_weight`
onto `weight.device` before the addition, making the helper device-robust
without forcing any change to the trainer's model-placement order.

These tests pin that fix in place. The CUDA-specific cases skip cleanly when
no GPU is available (e.g. CI runners).
"""

import math
import unittest

import torch

from musubi_tuner.networks.lora import LoRAModule


CUDA_AVAILABLE = torch.cuda.is_available()


def _make_lora(base: torch.nn.Linear, lora_dim: int = 8, alpha: float = 4.0) -> LoRAModule:
    return LoRAModule(
        lora_name="test.layer",
        org_module=base,
        multiplier=1.0,
        lora_dim=lora_dim,
        alpha=alpha,
        use_dora=True,
    )


class TestDoRAInitDeviceMismatch(unittest.TestCase):
    """Reproduces and pins the bug scenario: base on CUDA, LoRA modules on CPU."""

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_apply_to_with_cuda_base_does_not_crash(self) -> None:
        base = torch.nn.Linear(64, 32, bias=True).to(device="cuda", dtype=torch.bfloat16)
        lora = _make_lora(base)

        # Pre-condition: LoRA tensors are still on CPU at this point. This is
        # the exact state the FLUX.2 trainer creates between step
        # `move_to_device_except_swap_blocks()` and `network.apply_to()`.
        self.assertEqual(lora.lora_down.weight.device.type, "cpu")
        self.assertEqual(lora.lora_up.weight.device.type, "cpu")
        self.assertIsNotNone(lora.dora_layer)
        self.assertEqual(lora.dora_layer.weight.device.type, "cpu")

        # Without the fix this raises RuntimeError; with the fix it returns
        # cleanly and the magnitude vector is populated.
        lora.apply_to()

        # DoRA still active and magnitude got written.
        self.assertTrue(lora.use_dora)
        self.assertFalse(torch.equal(lora.dora_layer.weight, torch.ones_like(lora.dora_layer.weight)))

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_dora_magnitude_equals_base_row_norm_at_init(self) -> None:
        """At init, `lora_up` is zeros, so DoRA magnitude == row-wise L2 norm of the base weight."""
        torch.manual_seed(0)
        base = torch.nn.Linear(16, 8, bias=False).to(device="cuda", dtype=torch.bfloat16)
        lora = _make_lora(base, lora_dim=4, alpha=4.0)
        lora.apply_to()

        with torch.no_grad():
            expected = torch.linalg.norm(base.weight.float(), dim=1)
        magnitude = lora.dora_layer.weight.detach().cpu().float()

        # Tolerance accommodates bf16 precision in the materialized norm.
        self.assertTrue(
            torch.allclose(magnitude, expected.cpu(), rtol=2e-2, atol=2e-3),
            f"magnitude={magnitude}\nexpected={expected.cpu()}",
        )

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_forward_after_init_and_move(self) -> None:
        """End-to-end: bug-scenario init -> move LoRA child to CUDA -> bf16 autocast forward."""
        torch.manual_seed(0)
        base = torch.nn.Linear(64, 32, bias=True).to(device="cuda", dtype=torch.bfloat16)
        lora = _make_lora(base)
        lora.apply_to()

        # Mirror what `accelerator.prepare(network)` would do for a real run:
        # bring the LoRA child modules onto the same device/dtype as the base.
        lora.to(device="cuda", dtype=torch.bfloat16)

        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        # `apply_to()` rewires `base.forward` to `lora.forward`, so this routes
        # through DoRA's runtime delta path.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = base(x)

        self.assertEqual(y.shape, (2, 32))
        self.assertEqual(y.device.type, "cuda")
        self.assertTrue(torch.isfinite(y).all())

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_init_idempotent_under_device_cast_guard(self) -> None:
        """The device-cast guard must not perturb the math vs. all-same-device init."""
        torch.manual_seed(123)
        weights = torch.randn(8, 16, dtype=torch.bfloat16)

        # Path A: base CUDA + LoRA CPU (the patched path).
        base_cuda = torch.nn.Linear(16, 8, bias=False)
        with torch.no_grad():
            base_cuda.weight.copy_(weights)
        base_cuda = base_cuda.to(device="cuda", dtype=torch.bfloat16)
        lora_a = _make_lora(base_cuda, lora_dim=4, alpha=4.0)
        # Match init RNG by manually zeroing lora_up and using a known down init.
        lora_a.apply_to()
        magnitude_a = lora_a.dora_layer.weight.detach().cpu().float()

        # Path B: everything on CPU (the unaffected path).
        base_cpu = torch.nn.Linear(16, 8, bias=False).to(dtype=torch.bfloat16)
        with torch.no_grad():
            base_cpu.weight.copy_(weights)
        lora_b = _make_lora(base_cpu, lora_dim=4, alpha=4.0)
        lora_b.apply_to()
        magnitude_b = lora_b.dora_layer.weight.detach().cpu().float()

        # Magnitudes are derived from base_weight (lora_up=0), so both paths must agree
        # within bf16 norm tolerance regardless of device-cast ordering.
        self.assertTrue(
            torch.allclose(magnitude_a, magnitude_b, rtol=2e-2, atol=2e-3),
            f"Path-A (cuda base) {magnitude_a}\n!= Path-B (cpu base) {magnitude_b}",
        )


class TestDoRAInitSameDeviceStillWorks(unittest.TestCase):
    """Sanity: ensure the all-same-device path was not regressed by the patch."""

    def test_apply_to_all_cpu(self) -> None:
        base = torch.nn.Linear(16, 8, bias=False)
        lora = _make_lora(base, lora_dim=4, alpha=4.0)

        # No-op for the device-cast guard; equivalent to pre-patch behavior.
        lora.apply_to()
        self.assertTrue(lora.use_dora)
        self.assertEqual(lora.dora_layer.weight.device.type, "cpu")
        self.assertFalse(torch.equal(lora.dora_layer.weight, torch.ones_like(lora.dora_layer.weight)))

    def test_alpha_zero_means_no_scale(self) -> None:
        """Spot-check unrelated DoRA-init invariant survived the patch."""
        base = torch.nn.Linear(16, 8, bias=False)
        lora = LoRAModule(
            lora_name="test.alpha_zero",
            org_module=base,
            multiplier=1.0,
            lora_dim=4,
            alpha=0,
            use_dora=True,
        )
        # alpha=0 should be normalized to lora_dim (so scale = 1.0).
        self.assertEqual(float(lora.alpha.item()), 4.0)
        self.assertAlmostEqual(lora.scale, 1.0)
        lora.apply_to()
        self.assertTrue(lora.use_dora)


if __name__ == "__main__":
    unittest.main()
