from types import SimpleNamespace

import pytest
import torch

import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer
from musubi_tuner.optimizers.prodigy_plus_inner_momentum import (
    UPSTREAM_PRODIGYPLUS_VERSION,
    ProdigyPlusInnerMomentumScheduleFree,
)


OPTIMIZER_TYPE = "musubi_tuner.optimizers.prodigy_plus_inner_momentum.ProdigyPlusInnerMomentumScheduleFree"


def _base_kwargs(factored):
    return dict(
        lr=1.0,
        betas=(0.95, 0.99),
        d0=1e-6,
        prodigy_steps=6,
        eps=1e-8,
        use_bias_correction=True,
        schedulefree_c=12,
        use_stableadamw=True,
        split_groups=True,
        split_groups_mean=True,
        factored=factored,
        stochastic_rounding=False,
        use_speed=True,
    )


def _two_param_groups(shape=(48, 40)):
    p0 = torch.nn.Parameter(torch.zeros(shape))
    p1 = torch.nn.Parameter(torch.zeros(shape))
    return p0, p1, [{"params": [p0]}, {"params": [p1]}]


def test_optimizer_type_keeps_schedulefree_lifecycle_detection():
    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(optimizer_type=OPTIMIZER_TYPE)

    assert OPTIMIZER_TYPE.lower().endswith("schedulefree")
    assert ProdigyPlusInnerMomentumScheduleFree.__name__.endswith("ScheduleFree")
    assert trainer.is_schedulefree_optimizer(None, args)


def test_inner_momentum_optimizer_emits_prodigy_applied_step_logs():
    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(optimizer_type=OPTIMIZER_TYPE)
    descs = ["unet", "unet plus"]
    shared = 6.43e-5
    groups = [
        {"d": d, "d0": 1e-6, "lr": lr, "effective_lr": lr * 0.5, "k": 500, "prodigy_steps": 500, "shared_d": shared}
        for d, lr in ((1.769, 1.0), (2.197e-5, 12.0))
    ]
    optimizer = SimpleNamespace(param_groups=groups)
    scheduler = SimpleNamespace(get_last_lr=lambda: [1.0, 12.0])

    logs = trainer.generate_step_logs(args, 0.1, 0.1, scheduler, descs, optimizer)

    assert logs["lr/d*lr/unet"] == pytest.approx(1.769)
    assert logs["lr/d*eff_lr/unet"] == pytest.approx(1.769 * 0.5)
    assert logs["lr/shared_d"] == pytest.approx(shared)
    assert logs["lr/applied_dlr/unet"] == pytest.approx(shared)
    assert logs["lr/applied_dlr/unet plus"] == pytest.approx(shared * 12.0)
    assert logs["lr/applied_dlr_eff/unet"] == pytest.approx(shared * 0.5)
    assert logs["lr/applied_dlr_eff/unet plus"] == pytest.approx(shared * 12.0 * 0.5)


def test_non_d_adapting_schedulefree_optimizer_skips_prodigy_logs():
    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(optimizer_type="schedulefree.AdamWScheduleFree")
    optimizer = SimpleNamespace(param_groups=[{"lr": 1.0}])
    scheduler = SimpleNamespace(get_last_lr=lambda: [1.0])

    logs = trainer.generate_step_logs(args, 0.1, 0.1, scheduler, ["unet"], optimizer)

    assert logs["lr/unet"] == 1.0
    assert not any(key.startswith("lr/d*") or key.startswith("lr/applied_") for key in logs)
    assert "lr/shared_d" not in logs


def test_upstream_prodigyplus_fork_point_is_still_current():
    pytest.importorskip("prodigyplus")
    from prodigyplus.core_optimiser import CoreOptimiser

    assert CoreOptimiser.VERSION == UPSTREAM_PRODIGYPLUS_VERSION


@pytest.mark.parametrize("factored", [False, True])
def test_inner_beta1_zero_matches_upstream_prodigy_schedulefree(factored):
    pytest.importorskip("prodigyplus")
    from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

    torch.manual_seed(0)
    p_base0, p_base1, base_groups = _two_param_groups()
    p_inner0, p_inner1, inner_groups = _two_param_groups()

    base = ProdigyPlusScheduleFree(base_groups, **_base_kwargs(factored))
    inner = ProdigyPlusInnerMomentumScheduleFree(inner_groups, inner_beta1=0.0, **_base_kwargs(factored))
    base.train()
    inner.train()

    grads = [(torch.randn_like(p_base0), torch.randn_like(p_base1)) for _ in range(12)]
    for g0, g1 in grads:
        p_base0.grad = g0.clone()
        p_base1.grad = g1.clone()
        p_inner0.grad = g0.clone()
        p_inner1.grad = g1.clone()

        base.step()
        inner.step()
        base.zero_grad(set_to_none=False)
        inner.zero_grad(set_to_none=False)

    assert torch.allclose(p_base0, p_inner0, atol=0, rtol=0)
    assert torch.allclose(p_base1, p_inner1, atol=0, rtol=0)
    assert torch.allclose(base.state[p_base0]["z"], inner.state[p_inner0]["z"], atol=0, rtol=0)
    assert torch.allclose(base.state[p_base1]["z"], inner.state[p_inner1]["z"], atol=0, rtol=0)
    assert inner.param_groups[0]["inner_beta1"] == 0.0
    assert "exp_avg" not in inner.state[p_inner0]
    for base_group, inner_group in zip(base.param_groups, inner.param_groups):
        assert inner_group["d"] == pytest.approx(base_group["d"])
        assert inner_group["shared_d"] == pytest.approx(base_group["shared_d"])
        assert inner_group["effective_lr"] == pytest.approx(base_group["effective_lr"])


@pytest.mark.parametrize("factored", [False, True])
def test_raw_d_remains_inert_with_inner_momentum(factored):
    torch.manual_seed(0)
    p_hi, p_lo, groups = _two_param_groups(shape=(100, 40))
    opt = ProdigyPlusInnerMomentumScheduleFree(
        groups,
        **{
            **_base_kwargs(factored),
            "prodigy_steps": 1,
        },
        inner_beta1=0.75,
    )
    opt.train()
    g = torch.randn_like(p_hi)

    def step():
        p_hi.grad = g.clone()
        p_lo.grad = g.clone()
        opt.step()
        opt.zero_grad(set_to_none=False)

    for _ in range(3):
        step()

    opt.param_groups[0]["d"] = 1.769
    opt.param_groups[1]["d"] = 1.769e-3

    before_hi = before_lo = None
    for _ in range(400):
        before_hi = p_hi.detach().clone()
        before_lo = p_lo.detach().clone()
        step()

    move_hi = (p_hi.detach() - before_hi).norm().item()
    move_lo = (p_lo.detach() - before_lo).norm().item()

    assert "exp_avg" in opt.state[p_hi]
    assert move_lo > 0, "control group did not move; test is not exercising the step path"
    assert move_hi == pytest.approx(move_lo, rel=1e-3), (
        f"inner momentum reintroduced raw-d sensitivity: a 1000x d gap gave updates "
        f"{move_hi:.4e} vs {move_lo:.4e} (ratio {move_hi / move_lo:.3f})"
    )


def _tiny_ideogram4_model():
    cfg = Ideogram4Config(
        emb_dim=32,
        num_layers=2,
        num_heads=2,
        intermediate_size=48,
        adanln_dim=16,
        in_channels=128,
        llm_features_dim=16,
        mrope_section=(2, 2, 2),
    )
    return Ideogram4Transformer(cfg)


def test_loraplus_groups_are_preserved_for_inner_momentum_optimizer():
    torch.manual_seed(0)
    model = _tiny_ideogram4_model()
    net = lora_ideogram4.create_arch_network(
        1.0,
        4,
        4.0,
        None,
        None,
        model,
        loraplus_lr_ratio="12",
        use_dora="False",
    )
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)

    params, descriptions = net.prepare_optimizer_params(unet_lr=1.0)
    params = [{**group, "params": list(group["params"])} for group in params]

    assert descriptions == ["unet", "unet plus"]
    assert [group["lr"] for group in params] == [1.0, 12.0]

    opt = ProdigyPlusInnerMomentumScheduleFree(
        params,
        lr=1.0,
        betas=(0.95, 0.99),
        d0=1e-6,
        split_groups=True,
        split_groups_mean=True,
        use_stableadamw=True,
        stochastic_rounding=False,
        inner_beta1=0.5,
    )
    opt.train()

    for group_idx, group in enumerate(opt.param_groups):
        for param in group["params"]:
            param.grad = torch.full_like(param, 0.01 * (group_idx + 1))

    opt.step()
    opt.zero_grad(set_to_none=True)

    assert [group["lr"] for group in opt.param_groups] == [1.0, 12.0]
    assert all("inner_beta1" in group for group in opt.param_groups)
    assert any("exp_avg" in opt.state[param] for group in opt.param_groups for param in group["params"])


def test_resume_from_inner_beta1_zero_allocates_missing_exp_avg():
    torch.manual_seed(0)
    p0, p1, groups = _two_param_groups()
    opt = ProdigyPlusInnerMomentumScheduleFree(groups, inner_beta1=0.0, **_base_kwargs(factored=False))
    opt.train()

    p0.grad = torch.randn_like(p0)
    p1.grad = torch.randn_like(p1)
    opt.step()
    opt.zero_grad(set_to_none=False)

    assert "exp_avg" not in opt.state[p0]

    for group in opt.param_groups:
        group["inner_beta1"] = 0.5
    p0.grad = torch.randn_like(p0)
    p1.grad = torch.randn_like(p1)
    opt.step()

    assert "exp_avg" in opt.state[p0]
    assert "exp_avg" in opt.state[p1]
