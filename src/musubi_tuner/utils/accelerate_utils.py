"""Accelerator-aware context managers for code paths that need the original (unwrapped) forward.

`Accelerator.prepare(model)` with `mixed_precision="bf16"` REPLACES `model.forward` with
`autocast_context(model._original_forward)` (see accelerate.Accelerator.prepare_model:50-60 in
accelerate >= 1.x). The wrapper enters `torch.autocast("cuda", dtype=bf16)` INSIDE `model.forward`
after dispatch, so an outer `with torch.autocast(..., enabled=False):` at the caller does not defeat
it — `torch.autocast` is a stacking context and the inner accelerate-installed setting overrides the
outer disable for its scope.

Architectures whose forward is numerically incompatible with bf16 autocast (Ideogram 4 was the first
case: `Ideogram4RMSNorm`/`F.rms_norm`, `F.scaled_dot_product_attention`, custom `_apply_rotary_pos_emb`
all diverge under autocast enough that iterative CFG amplification collapses sampling output to flat
gray; the training forward diverges by ~Max abs diff 3.8 even at 512×512) need to bypass the wrapper
at the call site. The right boundary is to temporarily restore `model._original_forward`, run the
computation, and restore the wrapped forward in a finally block. Combined with an explicit
`torch.autocast(..., enabled=False)` at the same site, this defeats BOTH autocast installation
mechanisms (context manager AND forward wrapper).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch.nn as nn


@contextmanager
def disable_accelerate_forward_autocast(accelerator, model: nn.Module) -> Iterator[nn.Module]:
    """Temporarily restore the unwrapped forward on a model prepared by `accelerator.prepare`.

    Accelerator.prepare_model with native_amp stores the original forward as
    `model._original_forward` and replaces `model.forward` with an autocast-wrapping callable.
    This context manager swaps `forward` back to `_original_forward` for its scope, then restores
    the wrapped forward on exit (even on exception).

    No-op when:
      - `accelerator is None` (e.g., standalone test or direct script call).
      - The model has no `_original_forward` attribute (e.g., accelerator with native_amp=False,
        or the model was never passed through prepare()).

    Args:
        accelerator: `accelerate.Accelerator` instance, or None. Used only for unwrap_model so we
            also handle DDP-wrapped models correctly (`unwrap_model` strips DDP first, then we
            operate on the inner model where prepare_model attached `_original_forward`).
        model: The (possibly wrapped) model to call inside the context.

    Yields:
        The same `model` reference. Callers should call `model(...)` (or pass it through any helper
        that ultimately calls `model.__call__`) inside the with-block; dispatch will hit the
        restored unwrapped forward instead of the autocast-wrapping callable.

    Example:
        >>> with disable_accelerate_forward_autocast(accelerator, transformer):
        ...     with torch.autocast(device_type=device.type, enabled=False):
        ...         output = transformer(x)  # runs in the dtype the caller intended, no bf16 autocast
    """
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    original_forward = getattr(unwrapped, "_original_forward", None)

    if original_forward is None:
        # Either not prepared by accelerate, or prepared without native_amp — nothing to swap.
        yield model
        return

    wrapped_forward = unwrapped.forward
    unwrapped.forward = original_forward
    try:
        yield model
    finally:
        unwrapped.forward = wrapped_forward
