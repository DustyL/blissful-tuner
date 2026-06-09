"""Regression tests for LoRAModule.forward dtype harmonization.

Background: LoRAModule's `lora_down` / `lora_up` are `nn.Linear` modules created with PyTorch's
default fp32 weights. When the base model runs in bf16 (e.g., Ideogram 4 with fp8-patched
Linears producing bf16 outputs), the input `x` to `lora_down` is bf16. Under accelerator's
mixed_precision autocast, `F.linear` silently casts the fp32 weight to bf16 at call time and
the forward works. Without autocast — exactly the case for architectures that disable it to
preserve training-inference parity, like Ideogram 4 since commit 55e4d79 — `F.linear` errors:
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float`.

The fix in lora.py:571-... explicitly casts `x` to lora_down's dtype before the call and casts
`lx` back to base's dtype before the final add. Under autocast both casts are no-ops (same
dtype); without autocast they make the math work. This test file locks down the invariant.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from musubi_tuner.networks.lora import LoRAModule


def _make_lora(in_dim: int = 16, out_dim: int = 32, rank: int = 4, *, dora: bool = False) -> tuple[nn.Linear, LoRAModule]:
    """Build a LoRAModule wrapping a Linear, with the standard kaiming/zeros init.

    Returns (base, lora). `base.forward` is monkey-patched to LoRA's forward, so calling
    `base(x)` dispatches through LoRA. The LoRA module deletes its `org_module` ref in
    apply_to(), so we hold the base separately for dtype mutations like `base.to(bf16)`.
    """
    base = nn.Linear(in_dim, out_dim, bias=False)
    lora = LoRAModule(
        lora_name="lora.test",
        org_module=base,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        use_dora=dora,
    )
    lora.apply_to()  # monkey-patches base.forward -> lora.forward, deletes lora.org_module
    return base, lora


def test_lora_forward_handles_bf16_input_with_fp32_lora_weights():
    """The bug from v4 launch: fp32 LoRA weights × bf16 base input with autocast off.

    Without the fix this raises:
        RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float

    Reproduces the exact configuration from the Ideogram 4 training crash:
    - base model in bf16 (its Linear is cast to bf16 to match the fp8 dequant output dtype)
    - LoRA in fp32 (the default for nn.Linear and the dtype Prodigy needs for master weights)
    - no autocast (Ideogram 4 disables it at the do_inference / process_batch boundary)
    """
    base, lora = _make_lora()
    # Base in bf16 (matches Ideogram 4's fp8-dequant -> bf16 output)
    base.to(dtype=torch.bfloat16)
    x = torch.randn(2, 8, 16, dtype=torch.bfloat16)

    # Without autocast — the autocast-disabled context Ideogram 4's helper installs
    out = base(x)

    assert out.shape == (2, 8, 32)
    assert out.dtype == torch.bfloat16, (
        "Output dtype must match base's dtype. If LoRA contribution promotes to fp32, downstream "
        "layers expecting bf16 (norms, attention, etc.) will hit the same dtype mismatch the LoRA "
        "fix was supposed to eliminate."
    )


def test_lora_forward_under_autocast_is_unchanged():
    """Sanity: with autocast active (the original codepath), the new explicit casts are no-ops
    and forward produces the same shape/dtype as before. Locks down the invariant that this
    fix is backward-compatible with the historical autocast-on path."""
    base, lora = _make_lora()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.to(device=device, dtype=torch.bfloat16)
    lora.to(device=device)  # LoRA stays fp32 (default) so autocast actually has work to do
    x = torch.randn(2, 8, 16, dtype=torch.bfloat16, device=device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = base(x)

    assert out.shape == (2, 8, 32)
    assert out.dtype == torch.bfloat16


def test_lora_forward_step_zero_is_exact_noop():
    """At step 0, lora_up.weight = zeros exactly (kaiming + zeros init pattern). The full LoRA
    contribution must be exactly zero — equal to the base output to bit-identical precision.

    This is the load-bearing invariant for sample_at_first: any LoRA-induced perturbation at
    step 0 would change the sampled image relative to the base (no-LoRA) generation, defeating
    the diagnostic value of sample_at_first."""
    base, lora = _make_lora()
    base.to(dtype=torch.bfloat16)

    # Sanity: confirm step-0 init pattern actually puts lora_up at exact zero
    assert torch.all(lora.lora_up.weight == 0.0), (
        "Test precondition: lora_up at init should be all zeros (kaiming/zeros pattern). "
        "If this assertion fails, the init pattern has changed and step-0 sampling no longer "
        "produces no-LoRA output."
    )

    x = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    base_out = lora.org_forward(x)  # the saved original Linear forward (no LoRA add)
    lora_out = base(x)  # dispatches through lora.forward (with LoRA add)

    assert torch.equal(base_out, lora_out), (
        f"At step 0 (lora_up=0), LoRA output must equal base output bit-for-bit. "
        f"Diff: max={(base_out - lora_out).abs().max().item():.3e}. If the cast-then-multiply path "
        f"introduces non-zero perturbation here, sample_at_first becomes misleading."
    )


def test_lora_forward_handles_fp32_input_with_fp32_lora_weights():
    """Regression guard for the standard case: fp32 input + fp32 LoRA. The explicit cast
    `x.to(lora_dtype) if x.dtype != lora_dtype else x` should be a no-op here (no copy,
    same tensor reference), preserving existing behavior on fully-fp32 paths."""
    base, lora = _make_lora()
    x = torch.randn(2, 8, 16, dtype=torch.float32)

    out = base(x)
    assert out.shape == (2, 8, 32)
    assert out.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DoRA test uses CUDA for realistic fp8 base layer.")
def test_lora_forward_with_dora_handles_dtype_mismatch():
    """DoRA path also needs the cast-to-base-dtype guard. DoRA's _dora_delta computes
    `mag_norm_scale * lora_result * s` where mag_norm_scale may be fp32 (DoRA layer weight)
    — without casting lora_result to base's dtype first, the delta dtype becomes fp32 and
    the final add `org_forwarded + dora_delta` promotes the output to fp32 unexpectedly."""
    base, lora = _make_lora(dora=True)
    base.cuda().to(dtype=torch.bfloat16)
    lora.cuda()  # DoRA layer initialized with magnitudes — needs to be on device too
    x = torch.randn(2, 8, 16, dtype=torch.bfloat16, device="cuda")

    out = base(x)
    assert out.shape == (2, 8, 32)
    # DoRA path should still preserve base's dtype after the fix; without the cast guard,
    # mag_norm_scale (fp32) * bf16 base would have promoted the delta to fp32.
    assert out.dtype == torch.bfloat16, (
        "DoRA output dtype must match base. If this is fp32, the cast-to-base-dtype guard inside "
        "the DoRA branch is missing or stripped — _dora_delta's mag_norm_scale (fp32) will promote "
        "the delta and the final addition will widen the output dtype."
    )


# ---------------------------------------------------------------------------------------------------
# LoRAModule.forward must honor self.enabled — NUMERICAL-effect test
# ---------------------------------------------------------------------------------------------------
#
# Discovered 2026-06-09 via DLAY v5 training telemetry: masked_loss/prior = 0 across 2145 steps
# despite prior_preservation_weight = 0.5 and prior/teacher_ran = 1. Root cause: LoRANetwork.set_enabled
# (lora.py:1309) sets `lora.enabled = is_enabled` on every wrapped LoRAModule, but the training-time
# LoRAModule.forward was only consulting `self.multiplier == 0` — it never checked `self.enabled`.
# So `prior_model_context` (in mask_loss_process_batch.py) calling `set_enabled(False)` around the
# teacher pass became a silent no-op: teacher and student forwards saw the same LoRA-active model,
# their predictions were identical, and `prior_loss_unreduced = MSE(student, teacher) = 0`.
#
# Affects ALL architectures using the shared `prior_model_context` — WAN, FLUX.2, HV1.5, Qwen,
# Z-Image, Ideogram 4. LoHaModule had the same bug; LoKrModule was already correct.
#
# The pre-existing call-pattern tests (test_ideogram4_prior_preservation::test_lora_disabled_only_
# around_teacher_forward, test_masked_process_batch's _FakeNetwork.history assertions) verified the
# set_enabled CALLS but not their numerical effect. This new test exists specifically to catch the
# numerical contract: when enabled=False, LoRA contribution is exactly zero regardless of training
# state.


def test_lora_forward_respects_enabled_flag_numerical_effect():
    """LoRAModule.forward with self.enabled=False must produce IDENTICAL output to org_forward.

    Critical: this test must use TRAINED (non-zero) LoRA weights, not the zeros-init state, because
    the buggy code path is invisible at init (lora_up=0 makes the contribution zero anyway). We
    manually populate lora_up.weight with non-zero values to simulate post-training state, then
    assert the forward output equals org_forward output (bit-identical) when enabled=False, AND
    that they DIFFER when enabled=True. This dual assertion catches both directions of the bug —
    a forward that ignores enabled (the actual bug) AND a forward that always returns org_forwarded
    (the over-correction that would silently disable LoRA training entirely)."""
    base, lora = _make_lora(in_dim=16, out_dim=32, rank=4)

    # Simulate trained LoRA: non-zero lora_up so the contribution is detectable.
    with torch.no_grad():
        lora.lora_up.weight.copy_(torch.randn_like(lora.lora_up.weight) * 0.1)
    assert torch.any(lora.lora_up.weight != 0.0), "test precondition: lora_up must be non-zero (trained state)"

    x = torch.randn(2, 8, 16, dtype=torch.float32)
    base_out = lora.org_forward(x)  # the pre-patched Linear forward (no LoRA add)

    # With enabled=True (default), forward includes LoRA contribution — output must differ from base.
    out_enabled = base(x)
    assert not torch.equal(out_enabled, base_out), (
        "LoRA enabled=True with trained weights should produce output != base. If they're equal, "
        "either the LoRA contribution is being silently dropped (the over-correction failure mode) "
        "or lora_up.weight wasn't actually populated."
    )

    # With enabled=False, forward MUST equal org_forward bit-identically.
    lora.enabled = False
    out_disabled = base(x)
    assert torch.equal(out_disabled, base_out), (
        f"LoRA enabled=False must produce output IDENTICAL to org_forward. Max diff: "
        f"{(out_disabled - base_out).abs().max().item():.3e}. If non-zero, LoRAModule.forward is "
        "ignoring self.enabled — this is the DLAY v5 prior preservation silent-no-op bug. "
        "prior_model_context's set_enabled(False) call would have no numerical effect on training, "
        "making teacher and student forwards identical, collapsing masked_loss/prior to 0."
    )

    # Restore and verify the toggle is reversible (no state corruption).
    lora.enabled = True
    out_re_enabled = base(x)
    assert torch.equal(out_re_enabled, out_enabled), (
        "Re-enabling must restore the same output as before disable. If different, set_enabled toggles "
        "have side effects on internal state — a serious correctness issue beyond the original bug."
    )


def test_lora_network_set_enabled_propagates_to_module_forward():
    """LoRANetwork.set_enabled(False) iterates `lora.enabled = False` across all wrapped modules
    (lora.py:1309). Verify the network-level call propagates to per-module forward behavior, not
    just the attribute. This is the API contract `prior_model_context` actually exercises."""
    from musubi_tuner.networks.lora import LoRANetwork

    # Build a tiny network with one wrapped Linear so we can exercise set_enabled end-to-end.
    base, lora = _make_lora()
    with torch.no_grad():
        lora.lora_up.weight.copy_(torch.randn_like(lora.lora_up.weight) * 0.1)

    # Construct a minimal LoRANetwork shell. We only need set_enabled to iterate one entry.
    class _NetShell:
        def __init__(self, lora_module):
            self.text_encoder_loras = []
            self.unet_loras = [lora_module]

        # Bind the real method by class reference so we exercise the production code path.
        set_enabled = LoRANetwork.set_enabled

    net = _NetShell(lora)
    x = torch.randn(2, 8, 16, dtype=torch.float32)
    base_out = lora.org_forward(x)

    out_initial = base(x)
    assert not torch.equal(out_initial, base_out), "test precondition: trained LoRA must contribute non-zero"

    net.set_enabled(False)
    out_off = base(x)
    assert torch.equal(out_off, base_out), (
        "Network-level set_enabled(False) must propagate to per-module forward — the contract "
        "prior_model_context depends on. If this fails, every architecture's prior preservation is "
        "silently broken."
    )

    net.set_enabled(True)
    out_on = base(x)
    assert torch.equal(out_on, out_initial), "Re-enable via network must restore original output"
