"""Regression tests for disable_accelerate_forward_autocast.

`Accelerator.prepare(model)` with `mixed_precision="bf16"` installs autocast via
forward-method REPLACEMENT, not a `torch.autocast` context manager. The model's
`__init__`-time forward is saved as `model._original_forward`, and `model.forward` is
replaced with a function that wraps the original in `autocast_context(model_forward_func)`.
At dispatch, the wrapper enters autocast(bf16) inside the call, after any caller-installed
`torch.autocast(enabled=False)` context.

The first PR (#14) addressed only the context-manager mechanism: an outer
`with torch.autocast(enabled=False):` correctly defeats a `with torch.autocast(bf16):`
in caller scope (validated by 11 synthetic fp8 tests). But that's the wrong mechanism for
Accelerate-prepared models — `torch.autocast` nesting can't reach inside the wrapped forward
because the inner autocast enters AFTER dispatch.

This test file proves both:
  - The bug shape: outer `autocast(enabled=False)` does NOT defeat Accelerate's wrapper.
  - The fix: `disable_accelerate_forward_autocast` swaps `forward` back to `_original_forward`
    for the helper's scope and restores the wrapped forward on exit (or on exception).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from musubi_tuner.utils.accelerate_utils import disable_accelerate_forward_autocast


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Accelerator's autocast wrapper is installed only on CUDA-aware setups with native_amp.",
)


class _SeeAutocast(nn.Module):
    """Tiny module that records the autocast state observed inside its forward.

    `torch.is_autocast_enabled(device_type)` returns True when an autocast context is
    active for that device at the time of the query. We use it as the canary: when the
    Accelerate-installed wrapper enters its inner autocast context, the canary fires.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.observed: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observed.append(torch.is_autocast_enabled("cuda"))
        return self.linear(x)


def _make_prepared(device: str = "cuda") -> tuple[object, _SeeAutocast, _SeeAutocast]:
    """Build a model + Accelerator(prepare) pair. Returns (accelerator, prepared, unwrapped)."""
    from accelerate import Accelerator

    model = _SeeAutocast().to(device)
    accelerator = Accelerator(mixed_precision="bf16")
    prepared = accelerator.prepare(model)
    unwrapped = accelerator.unwrap_model(prepared)
    return accelerator, prepared, unwrapped


def test_accelerate_prepare_installs_forward_wrapper_with_original_forward_saved():
    """Documents Accelerator.prepare_model's installation mechanism. Required precondition
    for the helper's design: the helper relies on `_original_forward` being attached and
    `model.forward` being a different (wrapping) callable."""
    _, prepared, unwrapped = _make_prepared()

    # Postcondition: native_amp path attached _original_forward and replaced .forward
    assert hasattr(unwrapped, "_original_forward"), (
        "Accelerator.prepare_model under mixed_precision='bf16' should attach _original_forward. "
        "If this assertion fails, accelerate's installation API has changed and the helper's "
        "design assumption is broken."
    )
    assert unwrapped.forward is not unwrapped._original_forward, (
        "Accelerator should have REPLACED forward with an autocast-wrapping callable. "
        "If they're the same, prepare didn't actually wrap — the helper would be a no-op for the wrong reason."
    )


def test_outer_autocast_disable_DOES_NOT_defeat_accelerate_wrapper():
    """The bug shape: torch.autocast nesting cannot defeat Accelerate's forward wrapper.

    This is the empirical fact the user verified with accelerate 1.14.0.dev0 and the
    reason the PR #14 sample-time/training-time fixes were insufficient on their own.
    Without this regression test we'd silently lose the discovery on a future upgrade.
    """
    _, prepared, unwrapped = _make_prepared()
    x = torch.randn(2, 8, device="cuda")

    unwrapped.observed.clear()
    with torch.autocast(device_type="cuda", enabled=False):
        prepared(x)

    assert unwrapped.observed == [True], (
        "Expected the inner accelerate-installed autocast(bf16) context to OVERRIDE the outer "
        f"torch.autocast(enabled=False), but observed={unwrapped.observed}. If this changes, either "
        "accelerate's installation no longer wraps in an inner context, or torch.autocast stacking "
        "semantics changed — re-evaluate whether disable_accelerate_forward_autocast is still needed."
    )


def test_disable_accelerate_forward_autocast_defeats_the_wrapper():
    """The fix: with the helper active, forward dispatches to _original_forward directly,
    bypassing accelerate's autocast wrapper entirely.

    Combined with torch.autocast(enabled=False) at the call site, this covers BOTH autocast
    installation mechanisms (the context manager path was already handled by PR #14).
    """
    accelerator, prepared, unwrapped = _make_prepared()
    x = torch.randn(2, 8, device="cuda")

    unwrapped.observed.clear()
    wrapped_forward_before = unwrapped.forward
    with disable_accelerate_forward_autocast(accelerator, prepared):
        with torch.autocast(device_type="cuda", enabled=False):
            prepared(x)

    assert unwrapped.observed == [False], (
        f"Expected helper to bypass accelerate's autocast wrapper (observed={unwrapped.observed}). "
        "If autocast is still True inside forward, the helper's forward-swap mechanism is failing."
    )
    assert unwrapped.forward is wrapped_forward_before, (
        "Helper must restore the wrapped forward on exit so subsequent training-loop forwards still "
        "go through Accelerate's autocast path. If the restoration fails, every step after a sample "
        "would silently train without autocast (different from the explicit fix scope)."
    )


def test_helper_restores_wrapped_forward_even_on_exception():
    """Failure-path correctness: if the body raises, the wrapped forward MUST still be restored.

    Block-swap discipline in CLAUDE.md calls out this exact failure mode — a try/finally
    around state-mutating helpers is the only way training continues correctly after an
    exception in sampling.
    """
    accelerator, prepared, unwrapped = _make_prepared()
    wrapped_forward_before = unwrapped.forward

    with pytest.raises(RuntimeError, match="forced"):
        with disable_accelerate_forward_autocast(accelerator, prepared):
            raise RuntimeError("forced")

    assert unwrapped.forward is wrapped_forward_before, (
        "Helper must restore the wrapped forward in a finally block; otherwise an exception in the "
        "denoise loop would leave the transformer with autocast permanently bypassed for the rest "
        "of training."
    )


def test_helper_is_noop_when_accelerator_is_none():
    """Helper should gracefully degrade when called without an Accelerator (e.g., from a
    standalone script or a test using a raw model). This makes the helper safe to use at
    every call site without conditional wrapping."""
    model = _SeeAutocast().cuda()
    x = torch.randn(2, 8, device="cuda")

    model.observed.clear()
    with disable_accelerate_forward_autocast(None, model):
        with torch.autocast(device_type="cuda", enabled=False):
            model(x)

    assert model.observed == [False], "With no accelerator and no wrapper, autocast should already be off — the helper just yields."


def test_helper_is_noop_when_model_has_no_original_forward():
    """Helper should gracefully degrade when `_original_forward` is absent (e.g., the model
    was passed through a non-native-amp prepare or wasn't prepared at all).

    AcceleratorState is a process singleton, so we can't construct a second Accelerator
    with `mixed_precision="no"` mid-session — instead we simulate the no-native-amp case
    by removing the attribute the helper checks for. This tests the same code branch
    (`original_forward is None` → yield model unchanged) without fighting the singleton."""
    accelerator, prepared, unwrapped = _make_prepared()

    # Simulate the absence of native_amp by clearing the saved-original sentinel.
    # The helper checks for `_original_forward` via getattr(..., None); deleting the attribute
    # exercises the same code path the no-native-amp accelerator would have triggered.
    saved_original = unwrapped._original_forward
    saved_wrapped = unwrapped.forward
    delattr(unwrapped, "_original_forward")

    try:
        unwrapped.observed.clear()
        with disable_accelerate_forward_autocast(accelerator, prepared):
            assert unwrapped.forward is saved_wrapped, (
                "Helper must yield without touching forward when _original_forward is missing — "
                "the wrapped forward stays installed for the with-block scope."
            )
    finally:
        unwrapped._original_forward = saved_original  # restore so other tests aren't affected
