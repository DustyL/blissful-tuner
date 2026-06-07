"""Slice 1 — Ideogram-4 gradient checkpointing.

Block-internal GC gated on ``torch.is_grad_enabled() and self.gradient_checkpointing`` (NOT self.training), with
``use_reentrant=False``. The tests are built to actually exercise the checkpoint path — a no_grad parity test would
be vacuous (the gate skips checkpoint under no_grad, so it would compare two identical non-checkpointed runs).
"""

import torch

import musubi_tuner.ideogram4.modeling_ideogram4 as m4
import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target


def _tiny_model():
    # in_channels=128 + mrope_section=(2,2,2) so ideogram4_flow_matching_target works on (1, 128, 2, 2) latents.
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


def _inputs(cfg):
    latents = torch.randn(1, 128, 2, 2)
    noise = torch.randn(1, 128, 2, 2)
    text_features = [torch.randn(3, cfg.llm_features_dim)]
    t = torch.tensor([0.5])
    return latents, noise, text_features, t


def _run(model, cfg):
    latents, noise, text_features, t = _inputs(cfg)
    return ideogram4_flow_matching_target(model, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu")


def test_gc_forward_parity_with_grad_enabled():
    # MUST run with grad enabled: under no_grad the gate (torch.is_grad_enabled()) is False, so GC-on would not
    # checkpoint and this would compare two identical non-checkpointed runs (a vacuous pass). The bare model has
    # no dropout, so the forward is deterministic; checkpoint changes only the backward, so outputs must match.
    assert torch.is_grad_enabled()
    torch.manual_seed(0)
    model, cfg = _tiny_model()
    latents, noise, text_features, t = _inputs(cfg)

    pred_off, _ = ideogram4_flow_matching_target(model, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu")
    model.enable_gradient_checkpointing()
    pred_on, _ = ideogram4_flow_matching_target(model, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu")

    assert torch.allclose(pred_off, pred_on, atol=1e-5), (pred_off - pred_on).abs().max().item()


def test_gc_actually_checkpoints_under_grad_and_skips_under_no_grad(monkeypatch):
    # Partner to parity: prove checkpoint fires once per block under grad, and ZERO times under no_grad. Without
    # this, a wrong gate could pass parity yet never checkpoint. modeling_ideogram4 does `from torch.utils.checkpoint
    # import checkpoint`, so the name to patch is m4.checkpoint.
    torch.manual_seed(0)
    model, cfg = _tiny_model()
    model.enable_gradient_checkpointing()

    calls = {"n": 0}
    real = m4.checkpoint

    def counting(fn, *args, **kwargs):
        calls["n"] += 1
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(m4, "checkpoint", counting)

    _run(model, cfg)
    assert calls["n"] == cfg.num_layers, f"expected one checkpoint per block, got {calls['n']}"

    calls["n"] = 0
    with torch.no_grad():
        _run(model, cfg)
    assert calls["n"] == 0, "checkpoint must NOT fire under no_grad (gate is torch.is_grad_enabled())"


def test_gc_lora_grads_flow_through_checkpointed_block():
    # The reentrant trap: base frozen, only LoRA adapters trainable. use_reentrant=False must let grads reach the
    # REAL lora_ideogram4 adapters recomputed inside the checkpointed block (a toy trainable param would not prove
    # the adapter path survives recompute).
    torch.manual_seed(0)
    model, cfg = _tiny_model()
    model.requires_grad_(False)  # freeze base, as in training
    network = lora_ideogram4.create_arch_network(1.0, 8, 4.0, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    model.enable_gradient_checkpointing()

    pred, target = _run(model, cfg)
    torch.nn.functional.mse_loss(pred, target).backward()

    lora_grads = [p.grad for p in network.parameters() if p.grad is not None]
    assert lora_grads, "no LoRA grads — checkpoint dropped the adapter graph (the reentrant trap)"
    assert any(g.abs().sum() > 0 for g in lora_grads), "all LoRA grads zero — adapters not in the recomputed graph"


def test_enable_disable_toggles_flag_on_every_block():
    model, _ = _tiny_model()
    assert not model.gradient_checkpointing and all(not b.gradient_checkpointing for b in model.layers)
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing and all(b.gradient_checkpointing for b in model.layers)
    model.disable_gradient_checkpointing()
    assert not model.gradient_checkpointing and all(not b.gradient_checkpointing for b in model.layers)


def test_cpu_offload_branch_is_wired_only_not_validated():
    # WIRING-ONLY: create_cpu_offloading_wrapper(device="cpu") moves nothing, so this proves the cpu_offload branch
    # executes and is parity-finite, NOT that GPU offloading works. The --gradient_checkpointing_cpu_offload flag
    # stays REJECTED at the trainer until a dedicated CUDA backward test lands.
    torch.manual_seed(0)
    model, cfg = _tiny_model()
    model.enable_gradient_checkpointing(cpu_offload=True)
    assert all(b.activation_cpu_offloading for b in model.layers)
    pred, _ = _run(model, cfg)
    assert torch.isfinite(pred).all()
