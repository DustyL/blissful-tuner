# Prodigy per-group raw d is OPERATIONALLY INERT under split_groups_mean (backlog P0-8 follow-up).
#
# The Klein v9 false-alarm cycle (2026-06-11) hinged on one question: does a group's raw Prodigy d
# reach the weights? Three analyses answered it three different ways from prose readings of the
# source, and it took this experiment to settle. The answer is NO — it cancels.
# ProdigyPlusScheduleFree accumulates the second moment with ``d_k = d**2`` (core_optimiser.py:485/506),
# so ``denom`` scales with d, while the numerator is ``d*grad`` (prodigy_plus_schedulefree.py:398):
# ``update = (d*grad)/(d*rms) = grad/rms`` — independent of d. The applied step is set by
# ``dlr = shared_d * lr`` alone. This test pins that contract numerically so the question is a
# two-second check, never a multi-analyst churn, again.

import pytest
import torch


@pytest.mark.parametrize("factored", [False, True])
def test_raw_per_group_d_is_inert_under_split_groups_mean(factored):
    # factored=True is the PRODUCTION configuration (every DLAY Prodigy config sets it); the factored
    # second-moment path applies the same d_k = d**2 scaling (core_optimiser.py:497-504), so the
    # cancellation must hold there too — proven here rather than prose-read, per the lesson of this
    # whole episode.
    pytest.importorskip("prodigyplus")
    from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

    torch.manual_seed(0)
    p_hi = torch.nn.Parameter(torch.zeros(100, 20))  # 2D so the factored path actually factorizes
    p_lo = torch.nn.Parameter(torch.zeros(100, 20))
    opt = ProdigyPlusScheduleFree(
        [{"params": [p_hi]}, {"params": [p_lo]}],
        lr=1.0,
        betas=(0.95, 0.99),
        prodigy_steps=1,  # freeze adaptation after step 1 so a forced d override persists
        eps=1e-8,
        use_stableadamw=True,
        split_groups=True,
        split_groups_mean=True,
        factored=factored,
    )
    opt.train()
    g = torch.randn(100, 20)

    def step():
        p_hi.grad = g.clone()
        p_lo.grad = g.clone()
        opt.step()
        opt.zero_grad(set_to_none=False)

    for _ in range(3):  # initialise optimiser state + freeze adaptation
        step()

    # Force a 1000x raw per-group d gap by hand (adaptation is frozen, so it persists). 1.769 is the
    # literal Klein v9 down-group raw d that triggered the whole false-alarm cycle.
    opt.param_groups[0]["d"] = 1.769
    opt.param_groups[1]["d"] = 1.769e-3

    # Let each group's second moment re-settle to its new d^2 (~1/(1-beta2) steps), then measure one step.
    before_hi = before_lo = None
    for _ in range(400):
        before_hi = p_hi.detach().clone()
        before_lo = p_lo.detach().clone()
        step()
    move_hi = (p_hi.detach() - before_hi).norm().item()
    move_lo = (p_lo.detach() - before_lo).norm().item()

    assert move_lo > 0, "control group did not move; test is not exercising the step path"
    # A 1000x raw-d gap MUST produce the same update — the raw per-group d cancels in the Adam ratio,
    # and the applied step depends only on shared_d * lr (identical lr here). If this ever fails, the
    # 'applied scale = d_raw * shared_d * lr' model is back and the P0-8 telemetry/warning are wrong.
    assert move_hi == pytest.approx(move_lo, rel=1e-3), (
        f"raw per-group d is NOT inert: a 1000x d gap gave updates {move_hi:.4e} vs {move_lo:.4e} "
        f"(ratio {move_hi / move_lo:.3f}); the applied step should depend only on shared_d * lr."
    )
