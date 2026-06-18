"""LoRAModule residual add: scale+add the delta in the (higher-precision) adapter dtype and round the
SUM to the base dtype ONCE.

Guards the change from `org + lx.to(base_dtype) * m * s` (prematurely quantize+scale the delta in the
base dtype, then round the sum again) to `(org + lx * m * s).to(base_dtype)` (one rounding). This is a
training-arithmetic change: it alters BOTH the forward output AND the adapter gradients in the mixed
base/adapter-dtype regime, so both are pinned here. Scope: the non-split (lora.py:660) and split_dims
(lora.py:694) training paths only -- DoRA (separate `_dora_delta` norm path, guarded by
test_lora_dora_delta_path) and LoRAInfModule (already adds in the adapter dtype) are out of scope.

Deterministic (RNG-independent) fixtures are used so a future torch RNG/kernel change can't turn the
old/new divergence into a spurious failure. Benefit is correctness hygiene, not a quality win.
"""

import itertools
import unittest

import torch

from musubi_tuner.networks.lora import LoRAModule

_MULT = 4.0  # scaled up so the delta is comparable to org -> the two rounding orders genuinely diverge


def _fixed(shape, dtype):
    """A deterministic, RNG-independent tensor with a mix of signs/magnitudes (asymmetric range so the
    matmuls don't trivially cancel)."""
    n = 1
    for s in shape:
        n *= s
    return torch.linspace(-1.3, 1.7, n, dtype=torch.float32).reshape(shape).to(dtype)


def _make_lora(in_f, out_f, lora_dim, base_dtype, *, split_dims=None):
    base = torch.nn.Linear(in_f, out_f, bias=False)
    lora = LoRAModule(lora_name="t", org_module=base, multiplier=_MULT, lora_dim=lora_dim, alpha=lora_dim, split_dims=split_dims)
    lora.apply_to()
    base.to(base_dtype)  # base in base_dtype; the adapter (a submodule of `lora`) stays fp32
    with torch.no_grad():
        base.weight.copy_(_fixed(base.weight.shape, base_dtype))
        for mod in _adapter_mods(lora, split_dims):
            mod.weight.copy_(_fixed(mod.weight.shape, torch.float32))  # non-zero (lora_up is zeros-init)
    lora.eval()  # no dropout
    return base, lora


def _adapter_mods(lora, split_dims):
    downs = list(lora.lora_down) if split_dims else [lora.lora_down]
    ups = list(lora.lora_up) if split_dims else [lora.lora_up]
    return downs + ups


def _recompute_lx(lora, x):
    xf = x.to(torch.float32)
    if lora.split_dims is None:
        return lora.lora_up(lora.lora_down(xf))
    return torch.cat([up(down(xf)) for down, up in zip(lora.lora_down, lora.lora_up)], dim=-1)


def _grads(lora, split_dims):
    return [m.weight.grad.detach().clone() for m in _adapter_mods(lora, split_dims)]


def _zero(lora, split_dims):
    for m in _adapter_mods(lora, split_dims):
        m.weight.grad = None


class LoRAResidualForward(unittest.TestCase):
    def _check(self, in_f, out_f, lora_dim, split_dims):
        base, lora = _make_lora(in_f, out_f, lora_dim, torch.bfloat16, split_dims=split_dims)
        x = _fixed((4, in_f), torch.bfloat16)

        org = lora.org_forward(x)
        lx = _recompute_lx(lora, x)
        old = org + lx.to(org.dtype) * lora.multiplier * lora.scale  # quantize delta first (OLD)
        new = (org + lx * lora.multiplier * lora.scale).to(org.dtype)  # add in adapter dtype, round once (NEW)

        actual = base(x)
        self.assertEqual(actual.dtype, torch.bfloat16, "must return the base dtype, not promote to fp32")
        self.assertTrue(torch.equal(actual, new), "forward must use the one-rounding add-then-cast form")
        self.assertFalse(torch.equal(new, old), "old/new must diverge here, else the test can't detect the change")

    def test_normal_path(self):
        self._check(in_f=64, out_f=64, lora_dim=8, split_dims=None)

    def test_split_dims_path(self):
        self._check(in_f=64, out_f=96, lora_dim=8, split_dims=[32, 32, 32])


class LoRAResidualBackward(unittest.TestCase):
    """The change moves the scale/multiply into the adapter dtype, which alters the adapter GRADIENTS --
    pin them across bf16/fp16 bases and normal/split paths."""

    def _check_grad_health(self, base_dtype, split_dims):
        in_f, out_f, lora_dim = 64, (96 if split_dims else 64), 8
        base, lora = _make_lora(in_f, out_f, lora_dim, base_dtype, split_dims=split_dims)
        x = _fixed((4, in_f), base_dtype)

        out = base(x)
        self.assertEqual(out.dtype, base_dtype, "output stays in the base dtype")
        out.float().square().mean().backward()

        for mod in _adapter_mods(lora, split_dims):
            g = mod.weight.grad
            self.assertIsNotNone(g, "every adapter param must receive a gradient")
            self.assertTrue(torch.isfinite(g).all(), "gradients must be finite")
            self.assertGreater(g.abs().sum().item(), 0.0, "gradients must be nonzero")
            self.assertEqual(g.dtype, torch.float32, "gradients stay in the adapter (fp32) dtype")

    def test_grad_health_all_regimes(self):
        for base_dtype, split_dims in itertools.product([torch.bfloat16, torch.float16], [None, [32, 32, 32]]):
            with self.subTest(base_dtype=base_dtype, split_dims=split_dims):
                self._check_grad_health(base_dtype, split_dims)

    def test_backward_matches_add_then_cast_and_differs_from_round_first(self):
        # The module's gradients must equal the add-then-cast (new) reference AND differ from the
        # round-delta-first (old) reference -- the backward analogue of the forward divergence proof.
        base, lora = _make_lora(64, 64, 8, torch.bfloat16)
        x = _fixed((4, 64), torch.bfloat16)
        m, s = lora.multiplier, lora.scale

        def grads_of(out):
            _zero(lora, None)
            out.float().square().mean().backward()
            return _grads(lora, None)

        g_code = grads_of(base(x))
        g_new = grads_of((lora.org_forward(x) + _recompute_lx(lora, x) * m * s).to(x.dtype))
        g_old = grads_of(lora.org_forward(x) + _recompute_lx(lora, x).to(x.dtype) * m * s)

        for gc, gn in zip(g_code, g_new):
            self.assertTrue(torch.equal(gc, gn), "module backward must match the add-then-cast reference")
        self.assertTrue(
            any(not torch.equal(gn, go) for gn, go in zip(g_new, g_old)),
            "new/old backward must diverge, else the test can't detect the gradient change",
        )


class LoRAResidualNoOps(unittest.TestCase):
    def test_multiplier_zero_is_bitwise_noop(self):
        base, lora = _make_lora(64, 64, 8, torch.bfloat16)
        lora.multiplier = 0.0
        x = _fixed((4, 64), torch.bfloat16)
        self.assertTrue(torch.equal(base(x), lora.org_forward(x)))  # early return at top of forward()

    def test_scale_zero_is_bitwise_noop(self):
        # scale==0 with multiplier!=0 exercises the else-branch arithmetic: (org + lx*m*0).to(bf16) == org.
        base, lora = _make_lora(64, 64, 8, torch.bfloat16)
        lora.scale = 0.0
        x = _fixed((4, 64), torch.bfloat16)
        self.assertTrue(torch.equal(base(x), lora.org_forward(x)))

    def test_disabled_is_bitwise_noop(self):
        base, lora = _make_lora(64, 64, 8, torch.bfloat16)
        lora.enabled = False
        x = _fixed((4, 64), torch.bfloat16)
        self.assertTrue(torch.equal(base(x), lora.org_forward(x)))


class LoRAResidualAutocastNoOp(unittest.TestCase):
    """Under autocast, org_forward(x) and the adapter output share the autocast dtype, so 'round the
    delta then add' and 'add then round' collapse to the SAME expression -- the change is a bitwise
    no-op for every autocast architecture (WAN / HV / FLUX.2). This is the standing guard for that
    dtype-confluence invariant; the divergence tests above only cover the no-autocast Ideogram-class
    regime where the two output dtypes differ. Runs on CPU autocast -- no GPU needed."""

    def _check(self, split_dims):
        in_f, out_f, lora_dim = 64, (96 if split_dims else 64), 8
        # fp32 master weights + adapter under bf16 autocast == the canonical mixed-precision setup.
        base, lora = _make_lora(in_f, out_f, lora_dim, torch.float32, split_dims=split_dims)
        x = _fixed((4, in_f), torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = base(x)  # NEW code: the real wrapped forward (also exercises the x.to(lora_dtype) cast)
            org = lora.org_forward(x)
            lx = _recompute_lx(lora, x)
            old = org + lx.to(org.dtype) * lora.multiplier * lora.scale  # round-delta-first (OLD)
        self.assertEqual(actual.dtype, torch.bfloat16, "autocast output is the autocast dtype")
        self.assertTrue(torch.equal(actual, old), "under autocast the change is a bitwise no-op (dtype confluence)")

    def test_normal_path_autocast_is_noop(self):
        self._check(split_dims=None)

    def test_split_dims_path_autocast_is_noop(self):
        self._check(split_dims=[32, 32, 32])


if __name__ == "__main__":
    unittest.main()
