"""Native LoKr must start as an identity adapter: ΔW == 0 at step 0 in BOTH w2 branches.

LyCORIS's ``weight_gen`` zero-inits the full ``w2`` (and ``w2_b`` when factored). The native module
kaiming-initialised a full ``lokr_w2``, which for Wan 5120² Linears (factor -1 → out_k = in_n = 64)
is selected at ``lora_dim >= 64`` and produced a random step-0 delta with mean magnitude ~25% of the
base weight at dim 64 / alpha 32. Found in the 2026-09-13 LyCORIS 4.0 review.
"""

import torch
import torch.nn as nn

from musubi_tuner.networks.lokr import LoKrModule


def _delta(in_dim, out_dim, lora_dim, alpha):
    torch.manual_seed(0)
    module = LoKrModule("t", nn.Linear(in_dim, out_dim), lora_dim=lora_dim, alpha=alpha)
    return module, module.get_weight()


def test_full_w2_branch_starts_at_zero_delta():
    # 5120 -> (80, 64): dim 64 >= max(64, 64) selects the full-matrix w2 branch
    module, delta = _delta(5120, 5120, lora_dim=64, alpha=32.0)
    assert module.lokr_w2 is not None, "test premise: this shape/dim must take the full-w2 branch"
    assert torch.count_nonzero(delta) == 0


def test_factored_w2_branch_starts_at_zero_delta():
    module, delta = _delta(3072, 3072, lora_dim=8, alpha=4.0)
    assert module.lokr_w2 is None, "test premise: this shape/dim must take the factored w2 branch"
    assert torch.count_nonzero(delta) == 0


def test_full_w2_branch_still_trains():
    """Zero-init must not zero the gradient path: w1 is non-zero, so w2 receives gradient."""
    module, _ = _delta(256, 256, lora_dim=64, alpha=32.0)
    assert module.lokr_w2 is not None
    x = torch.randn(4, 256)
    torch.nn.functional.linear(x, module.get_weight()).sum().backward()
    assert torch.count_nonzero(module.lokr_w2.grad) > 0
