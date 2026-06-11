# DoRA magnitudes -> separate optimizer param group (backlog P0-7, DLAY v8 post-mortems).
#
# DoRA magnitude vectors initialize at weight-row-norm scale (~1-10) while LoRA matrices start near
# zero. Mixing the two populations in one param group corrupts adaptive optimizers' per-group
# step-size estimation: Prodigy's shared d collapsed to ~d0 (3.9x under the no-DoRA baseline) on
# Ideogram 4 fp8 and inflated 1660x over its own up-group on FLUX.2 Klein bf16 — the same mechanism,
# opposite symptoms, condition-dependent direction. These tests pin the new grouping contract.

import pytest
import torch

import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target


def _tiny_model():
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
    return Ideogram4Transformer(cfg), cfg


def _make_network(model, use_dora=True, loraplus_ratio="16"):
    kwargs = {"use_dora": str(use_dora)}
    if loraplus_ratio is not None:
        kwargs["loraplus_lr_ratio"] = loraplus_ratio
    net = lora_ideogram4.create_arch_network(1.0, 4, 4.0, None, None, model, **kwargs)
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    return net


def _param_ids(group):
    return {id(p) for p in group["params"]}


def test_dora_magnitudes_get_their_own_param_group():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    net = _make_network(model, use_dora=True, loraplus_ratio="16")
    params, descs = net.prepare_optimizer_params(unet_lr=1.0)
    # materialize the dict_values views so they can be inspected repeatedly
    params = [{**g, "params": list(g["params"])} for g in params]

    assert descs == ["unet", "unet plus", "unet dora"]
    assert [len(g["params"]) for g in params] == [10, 10, 10]  # down / up / magnitudes
    assert [g["lr"] for g in params] == [1.0, 16.0, 1.0]  # dora group takes BASE lr, no ratio

    # Identity: the dora group is exactly the set of dora_layer.weight parameters; none leak into
    # the down group (the leak is what corrupted Prodigy's shared d in both v8 runs).
    dora_param_ids = {id(m.dora_layer.weight) for m in net.unet_loras}
    assert _param_ids(params[2]) == dora_param_ids
    assert not (_param_ids(params[0]) & dora_param_ids)
    assert not (_param_ids(params[1]) & dora_param_ids)


def test_dora_without_loraplus_still_isolated():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    net = _make_network(model, use_dora=True, loraplus_ratio=None)
    params, descs = net.prepare_optimizer_params(unet_lr=1.0)
    params = [{**g, "params": list(g["params"])} for g in params]

    assert descs == ["unet", "unet dora"]
    assert [len(g["params"]) for g in params] == [20, 10]  # down+up together, magnitudes apart
    dora_param_ids = {id(m.dora_layer.weight) for m in net.unet_loras}
    assert _param_ids(params[1]) == dora_param_ids


def test_no_dora_grouping_unchanged():
    # Regression: plain LoRA(+) runs keep the exact pre-change structure — the empty dora bucket
    # is skipped, so optimizer-state resume compatibility for non-DoRA runs is untouched.
    torch.manual_seed(0)
    model, _ = _tiny_model()
    net = _make_network(model, use_dora=False, loraplus_ratio="16")
    params, descs = net.prepare_optimizer_params(unet_lr=1.0)
    params = [{**g, "params": list(g["params"])} for g in params]

    assert descs == ["unet", "unet plus"]
    assert [len(g["params"]) for g in params] == [10, 10]
    assert [g["lr"] for g in params] == [1.0, 16.0]


def test_prodigy_estimates_independent_d_per_group():
    prodigyplus = pytest.importorskip("prodigyplus")

    torch.manual_seed(0)
    model, cfg = _tiny_model()
    net = _make_network(model, use_dora=True, loraplus_ratio="16")
    model.requires_grad_(False)
    net.requires_grad_(True)
    params, _ = net.prepare_optimizer_params(unet_lr=1.0)

    opt = prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree(
        params, lr=1.0, betas=(0.95, 0.99), d0=1e-6, split_groups=True, use_stableadamw=True
    )
    assert len(opt.param_groups) == 3
    for i in range(8):
        latents = torch.randn(1, 128, 2, 2)
        noise = torch.randn(1, 128, 2, 2)
        text_features = [torch.randn(3, cfg.llm_features_dim)]
        t = torch.rand(1)
        pred, target = ideogram4_flow_matching_target(
            model, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu"
        )
        torch.nn.functional.mse_loss(pred.float(), target.float()).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Each group carries its OWN d under split_groups — the whole point of the partition. We assert
    # structure, not dynamics (d trajectories on a tiny synthetic model are not meaningful).
    ds = [g.get("d") for g in opt.param_groups]
    assert all(d is not None for d in ds), "every group should carry its own Prodigy d estimate"
    assert all(torch.isfinite(torch.tensor(float(d))) for d in ds)
