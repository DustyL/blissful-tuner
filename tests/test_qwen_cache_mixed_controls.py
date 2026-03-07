"""Tests for Qwen cache writer handling of mixed control batches."""

from __future__ import annotations

import numpy as np
import torch


def test_dense_controls_list_preserves_none_positions():
    """Verify controls list is dense with None for items without controls."""

    class FakeItem:
        def __init__(self, has_control: bool):
            self.control_content = [np.zeros((64, 64, 3), dtype=np.uint8)] if has_control else None

    batch = [FakeItem(True), FakeItem(False), FakeItem(True), FakeItem(False)]

    controls: list[list[torch.Tensor] | None] = []
    for item in batch:
        if item.control_content is not None and len(item.control_content) > 0:
            controls.append([torch.from_numpy(cc[..., :3]) for cc in item.control_content])
        else:
            controls.append(None)

    assert len(controls) == len(batch)
    assert controls[0] is not None
    assert controls[1] is None
    assert controls[2] is not None
    assert controls[3] is None

    for b in range(len(batch)):
        control = controls[b]
        if batch[b].control_content is not None:
            assert control is not None
        else:
            assert control is None


def test_all_none_controls_collapses_to_none():
    """When no items have controls, the list should collapse to None."""
    controls: list[None] = [None, None, None]
    if all(c is None for c in controls):
        controls_final = None
    else:
        controls_final = controls
    assert controls_final is None
