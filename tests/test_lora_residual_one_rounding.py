"""LoRAModule residual add: keep the delta in fp32 and round the sum to the base dtype ONCE.

Guards the change from `org + lx.to(base_dtype) * m * s` (round the delta to bf16, then add and round
the sum again) to `(org + lx * m * s).to(base_dtype)` (one rounding). The non-split (lora.py:660) and
split_dims (lora.py:694) training paths only. DoRA's delta path and the inference module are out of
scope (DoRA: separate norm decomposition guarded by test_lora_dora_delta_path; inference: already adds
in fp32). multiplier==0 / scale==0 / disabled stay bitwise no-ops.

Benefit is correctness hygiene, not a quality win: ~0.001% at delta-scale 0.001, ~2% at 0.1.
"""

import unittest

import torch

from musubi_tuner.networks.lora import LoRAModule


def _bf16_lora(in_f, out_f, lora_dim, multiplier, *, split_dims=None, seed=0):
    """bf16 base Linear + fp32 LoRAModule (the no-autocast mixed-dtype case), with non-zero adapter
    weights so the delta is real (lora_up is zeros-init by default)."""
    torch.manual_seed(seed)
    base = torch.nn.Linear(in_f, out_f, bias=False)
    lora = LoRAModule(
        lora_name="t", org_module=base, multiplier=multiplier, lora_dim=lora_dim, alpha=lora_dim, split_dims=split_dims
    )
    lora.apply_to()
    base.to(torch.bfloat16)  # base in bf16; the adapter (a submodule of `lora`) stays fp32
    with torch.no_grad():
        downs = lora.lora_down if split_dims else [lora.lora_down]
        ups = lora.lora_up if split_dims else [lora.lora_up]
        for m in list(downs) + list(ups):
            m.weight.copy_(torch.randn_like(m.weight))
    lora.eval()  # no dropout
    return base, lora


def _recompute_lx(lora, x):
    xf = x.to(torch.float32)
    if lora.split_dims is None:
        return lora.lora_up(lora.lora_down(xf))
    return torch.cat([up(down(xf)) for down, up in zip(lora.lora_down, lora.lora_up)], dim=-1)


class LoRAResidualOneRounding(unittest.TestCase):
    def _check_add_then_cast(self, in_f, out_f, lora_dim, split_dims):
        # multiplier scaled up so the delta is comparable to org -> the two rounding orders genuinely diverge.
        base, lora = _bf16_lora(in_f, out_f, lora_dim, multiplier=4.0, split_dims=split_dims)
        x = torch.randn(4, in_f).to(torch.bfloat16)

        org = lora.org_forward(x)
        lx = _recompute_lx(lora, x)
        old = org + lx.to(org.dtype) * lora.multiplier * lora.scale  # round delta first (the OLD behavior)
        new = (org + lx * lora.multiplier * lora.scale).to(org.dtype)  # add in fp32, round once (NEW)

        actual = base(x)
        self.assertEqual(actual.dtype, torch.bfloat16, "residual add must return the base dtype, not promote to fp32")
        self.assertTrue(torch.equal(actual, new), "forward must use the one-rounding add-then-cast form")
        # The test only proves something if the two formulas actually differ for this input.
        self.assertFalse(torch.equal(new, old), "old/new must diverge here, else the test cannot detect the change")

    def test_normal_path_is_add_then_cast(self):
        self._check_add_then_cast(in_f=64, out_f=64, lora_dim=8, split_dims=None)

    def test_split_dims_path_is_add_then_cast(self):
        self._check_add_then_cast(in_f=64, out_f=96, lora_dim=8, split_dims=[32, 32, 32])

    def test_multiplier_zero_is_bitwise_noop(self):
        base, lora = _bf16_lora(64, 64, 8, multiplier=0.0)
        x = torch.randn(4, 64).to(torch.bfloat16)
        self.assertTrue(torch.equal(base(x), lora.org_forward(x)))  # early return at top of forward()

    def test_scale_zero_is_bitwise_noop(self):
        # scale==0 with multiplier!=0 bypasses the top guard and exercises the else-branch arithmetic:
        # (org + lx*m*0).to(bf16) == org bit-for-bit (bf16 -> fp32 -> bf16 is exact).
        base, lora = _bf16_lora(64, 64, 8, multiplier=2.0)
        lora.scale = 0.0
        x = torch.randn(4, 64).to(torch.bfloat16)
        self.assertTrue(torch.equal(base(x), lora.org_forward(x)))

    def test_disabled_is_bitwise_noop(self):
        base, lora = _bf16_lora(64, 64, 8, multiplier=4.0)
        lora.enabled = False
        x = torch.randn(4, 64).to(torch.bfloat16)
        self.assertTrue(torch.equal(base(x), lora.org_forward(x)))


if __name__ == "__main__":
    unittest.main()
