"""Regression tests for LoHaModule.forward honoring self.enabled.

Same bug class as LoRA's: LoHaModule.forward at networks/loha.py:145 previously returned
self.default_forward(x) unconditionally, ignoring self.enabled. The attribute existed on the
training class (initialized in __init__ at line 50) but was only consulted by the inference
class (LoHaInfModule at line 199). LoHaNetwork.set_enabled (line 532) propagates the attribute
but the training forward ignored it — exact same shape as the LoRA silent-no-op bug discovered
via DLAY v5 telemetry (commit 2c53a1b).

This test file is the LoHa-equivalent of the test_lora_forward_respects_enabled_flag_numerical_effect
tests in test_lora_dtype_robust_forward.py. The reviewer flagged the missing LoHa coverage as test
debt — closing it here so future regressions in either family's forward fail loudly.

Contract: when enabled=False, LoHaModule.forward MUST produce output bit-identical to
org_forward(x); when enabled=True with trained (non-zero hada_w1_a) weights, output MUST differ
from org_forward(x).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from musubi_tuner.networks.loha import LoHaModule


def _make_loha(in_dim: int = 16, out_dim: int = 32, rank: int = 4) -> tuple[nn.Linear, LoHaModule]:
    """Build a LoHaModule wrapping a Linear. Returns (base, loha).

    base.forward is monkey-patched to loha.forward by apply_to(); calling base(x) dispatches
    through LoHa. The reference org_forward is preserved on the loha module after apply_to (it
    deletes org_module but keeps the saved forward closure)."""
    base = nn.Linear(in_dim, out_dim, bias=False)
    loha = LoHaModule(
        lora_name="loha.test",
        org_module=base,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    )
    loha.apply_to()
    return base, loha


def test_loha_forward_respects_enabled_flag_numerical_effect():
    """The LoHa numerical contract that mirrors LoRA's: enabled=False -> base output;
    enabled=True with trained weights -> base + LoHa delta. Either direction broken is a silent
    failure mode for prior preservation (set_enabled(False) had no effect, teacher == student,
    masked_loss/prior collapses to 0 — same shape as the DLAY v5 bug).

    Critical: this test must use TRAINED (non-zero) hada_w1_a, not the zeros-init state. LoHa
    initializes hada_w1_a to zeros (loha.py:85) so the diff_weight = (w1_a @ w1_b) * (w2_a @ w2_b)
    starts at exactly zero — the buggy code path would be undetectable at init because the
    contribution is zero anyway. We force a non-zero hada_w1_a to simulate the post-warmup state
    where the bug actually matters."""
    base, loha = _make_loha(in_dim=16, out_dim=32, rank=4)

    # Simulate trained state: non-zero hada_w1_a so diff_weight != 0 and the LoHa contribution
    # is detectable. Other params (w1_b, w2_a, w2_b) were already non-zero from normal-init.
    with torch.no_grad():
        loha.hada_w1_a.copy_(torch.randn_like(loha.hada_w1_a) * 0.1)
    assert torch.any(loha.hada_w1_a != 0.0), "test precondition: hada_w1_a must be non-zero (trained state)"

    x = torch.randn(2, 8, 16, dtype=torch.float32)
    base_out = loha.org_forward(x)  # the saved Linear forward (no LoHa delta)

    # enabled=True (default) with trained weights: forward includes the LoHa delta.
    out_enabled = base(x)
    assert not torch.equal(out_enabled, base_out), (
        "LoHa enabled=True with trained weights should produce output != base. If they're equal, "
        "either default_forward isn't adding the diff_weight contribution or hada_w1_a wasn't "
        "actually populated. Without this precondition the disabled-case assertion is uninformative."
    )

    # enabled=False MUST produce bit-identical org_forward output.
    loha.enabled = False
    out_disabled = base(x)
    assert torch.equal(out_disabled, base_out), (
        f"LoHa enabled=False must produce output IDENTICAL to org_forward. Max diff: "
        f"{(out_disabled - base_out).abs().max().item():.3e}. If non-zero, LoHaModule.forward "
        "is ignoring self.enabled — same bug class as LoRAModule had until commit 2c53a1b. "
        "prior_model_context's set_enabled(False) would have no numerical effect on training, "
        "making teacher and student forwards identical and collapsing masked_loss/prior to 0."
    )

    # Reversibility: re-enable must restore exactly the same output as before disable.
    loha.enabled = True
    out_re_enabled = base(x)
    assert torch.equal(out_re_enabled, out_enabled), (
        "Re-enabling must restore the same output as before disable. If different, set_enabled "
        "toggles have side effects on internal state — a serious correctness issue beyond the "
        "original bug."
    )


def test_loha_network_set_enabled_propagates_to_module_forward():
    """LoHaNetwork.set_enabled propagates `lora.enabled` across all wrapped LoHa modules.
    Verify the network-level call changes the per-module forward behavior numerically — the
    API contract prior_model_context exercises via the shared mask_loss_process_batch.py path."""
    from musubi_tuner.networks.loha import LoHaNetwork

    base, loha = _make_loha()
    with torch.no_grad():
        loha.hada_w1_a.copy_(torch.randn_like(loha.hada_w1_a) * 0.1)

    class _NetShell:
        def __init__(self, loha_module):
            self.text_encoder_loras = []
            self.unet_loras = [loha_module]

        # Bind the production set_enabled by class reference so we exercise the real code path.
        set_enabled = LoHaNetwork.set_enabled

    net = _NetShell(loha)
    x = torch.randn(2, 8, 16, dtype=torch.float32)
    base_out = loha.org_forward(x)

    out_initial = base(x)
    assert not torch.equal(out_initial, base_out), "test precondition: trained LoHa must contribute non-zero"

    net.set_enabled(False)
    out_off = base(x)
    assert torch.equal(out_off, base_out), (
        "Network-level set_enabled(False) must propagate to per-module forward — the contract "
        "prior_model_context depends on. If this fails, LoHa prior preservation is silently broken "
        "for every architecture (same bug class LoRA had until commit 2c53a1b)."
    )

    net.set_enabled(True)
    out_on = base(x)
    assert torch.equal(out_on, out_initial), "Re-enable via network must restore original output"
