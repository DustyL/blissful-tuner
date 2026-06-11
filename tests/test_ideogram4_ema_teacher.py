# EMA teacher mode for Ideogram 4 prior preservation (backlog P0-6).
#
# Mirrors the shared masked-path lifecycle (mask_loss_process_batch.py) exactly: the teacher runs
# in BASE mode (LoRA disabled via prior_model_context) until the EMA lazily initializes at
# max(100, prior-decay warmup) optimizer steps; from then on the teacher forward runs under
# LoRAEmaTeacher.apply_to (LoRA ENABLED, weights swapped in-place to EMA values), and
# on_post_optimizer_step updates the EMA on real optimizer steps only. The user's standard recipe
# (Klein configs) is prior_teacher_mode="ema", decay=0.9995 — this unblocks it for Ideogram.

import logging
from types import SimpleNamespace

import torch

import musubi_tuner.ideogram4_train_network as itn

TRAINER_LOGGER = "musubi_tuner.ideogram4_train_network"


class _ParamNetwork(torch.nn.Module):
    """Stub adapter network: a real trainable parameter (so LoRAEmaTeacher can track it) plus the
    set_enabled recording the base-teacher context uses (prior_model_context disables/re-enables)."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.enabled_calls: list[bool] = []

    def set_enabled(self, enabled: bool):
        self.enabled_calls.append(enabled)


def _make_accel(with_trackers=True):
    return SimpleNamespace(
        device=torch.device("cpu"),
        trackers=[object()] if with_trackers else [],
        unwrap_model=lambda m: m,
    )


def _args(**overrides):
    base = dict(
        use_mask_loss=True,
        mask_gamma=1.0,
        mask_min_weight=0.0,
        mask_blur_kernel_size=0,
        mask_blur_radius=0.0,
        mask_area_scale_beta=0.0,
        normalize_per_sample=False,
        prior_preservation_weight=0.5,
        prior_teacher_mode="ema",
        prior_teacher_ema_decay=0.9995,
        prior_teacher_eval=False,
        prior_decay_schedule="constant",
        prior_decay_timestep_start=300.0,
        prior_decay_warmup_ratio=0.0,
        prior_preservation_timestep_threshold=None,
        loss_type="mse",
        loss_delta=1.0,
        ideogram4_timestep_mu=0.0,
        ideogram4_timestep_std=1.0,
        max_train_steps=5000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _batch(gh=4, gw=6):
    latents = torch.zeros(1, 128, gh, gw)
    mask = torch.ones(1, 1, 1, gh, gw)
    mask[..., :, : gw // 2] = 0.0  # partial mask so the no-prior-region skip doesn't fire
    return latents, {"i4_llm_features": [torch.zeros(1, 53248)], "mask_weights": mask}


def _fake_flow(transformer, latents, text_features, noise, timesteps, *, network_dtype, device):
    gh, gw = int(latents.shape[2]), int(latents.shape[3])
    return torch.zeros(1, gh * gw, 128, dtype=network_dtype), torch.full((1, gh * gw, 128), 0.5, dtype=network_dtype)


def _run_process_batch(trainer, args, network, monkeypatch, global_step):
    monkeypatch.setattr(itn, "ideogram4_flow_matching_target", _fake_flow)
    monkeypatch.setattr(itn, "get_schedule_for_resolution", lambda reso, *, known_mean, std: lambda u: u)
    latents, batch = _batch()
    return trainer.process_batch(
        args,
        _make_accel(),
        "TRANSFORMER",
        network,
        batch,
        latents,
        torch.ones_like(latents),
        None,
        torch.float32,
        torch.float32,
        None,
        global_step,
    )


def test_base_fallback_before_ema_init_step(monkeypatch):
    # Before max(100, warmup) steps, EMA mode runs the BASE teacher: prior_model_context disables
    # then re-enables the adapters, and no EMA object exists yet.
    trainer = itn.Ideogram4NetworkTrainer()
    net = _ParamNetwork()
    _, stats = _run_process_batch(trainer, _args(), net, monkeypatch, global_step=0)

    assert trainer.prior_lora_ema is None
    assert net.enabled_calls == [False, True]  # base-teacher context signature
    assert stats["prior/teacher_ran"] == 1.0
    assert stats["prior/teacher_mode_ema_used"] == 0.0


def test_lazy_init_and_ema_context_at_init_step(monkeypatch):
    # At global_step >= 100 (no warmup configured) the EMA initializes from the live adapter and
    # the teacher forward switches to apply_to: adapters stay ENABLED (no set_enabled calls) with
    # weights swapped in-place.
    trainer = itn.Ideogram4NetworkTrainer()
    net = _ParamNetwork()
    _, stats = _run_process_batch(trainer, _args(), net, monkeypatch, global_step=100)

    assert trainer.prior_lora_ema is not None
    assert float(trainer.prior_lora_ema.decay) == 0.9995
    assert net.enabled_calls == []  # EMA context never touches set_enabled
    assert stats["prior/teacher_mode_ema_used"] == 1.0
    # init_from captured the live parameter value
    ema_buf = next(iter(trainer.prior_lora_ema._ema_fp32.values()))
    assert torch.allclose(ema_buf, torch.ones(4))


def test_warmup_pushes_init_step_back(monkeypatch):
    # warmup_ratio 0.05 of 5000 steps = 250 > the 100-step floor: at step 100 EMA must NOT init yet.
    trainer = itn.Ideogram4NetworkTrainer()
    net = _ParamNetwork()
    # warmup scheduling produces per-sample prior weights, which the reducer requires
    # normalize_per_sample for (mask_loss.py guard) — same pairing real configs use.
    args = _args(prior_decay_warmup_ratio=0.05, normalize_per_sample=True)
    _run_process_batch(trainer, args, net, monkeypatch, global_step=100)
    assert trainer.prior_lora_ema is None
    _run_process_batch(trainer, args, net, monkeypatch, global_step=250)
    assert trainer.prior_lora_ema is not None


def test_base_mode_never_creates_ema(monkeypatch):
    trainer = itn.Ideogram4NetworkTrainer()
    net = _ParamNetwork()
    _, stats = _run_process_batch(trainer, _args(prior_teacher_mode="base"), net, monkeypatch, global_step=5000)
    assert trainer.prior_lora_ema is None
    assert stats["prior/teacher_mode_ema_used"] == 0.0
    assert net.enabled_calls == [False, True]


def test_on_post_optimizer_step_updates_only_on_sync_gradients():
    trainer = itn.Ideogram4NetworkTrainer()
    net = _ParamNetwork()
    args = _args()
    # Initialize the EMA as process_batch would, then move the live weights.
    from musubi_tuner.modules.lora_ema_teacher import LoRAEmaTeacher

    trainer.prior_lora_ema = LoRAEmaTeacher(decay=0.5)
    trainer.prior_lora_ema.init_from(net)
    with torch.no_grad():
        net.weight.fill_(3.0)

    accel = _make_accel()
    trainer.on_post_optimizer_step(args, accel, net, None, sync_gradients=False, global_step=1)
    ema_buf = next(iter(trainer.prior_lora_ema._ema_fp32.values()))
    assert torch.allclose(ema_buf, torch.ones(4))  # micro-batch (no sync): no update

    trainer.on_post_optimizer_step(args, accel, net, None, sync_gradients=True, global_step=1)
    assert torch.allclose(ema_buf, torch.full((4,), 2.0))  # 0.5*1 + 0.5*3


def test_on_post_optimizer_step_noop_before_init():
    trainer = itn.Ideogram4NetworkTrainer()
    net = _ParamNetwork()
    trainer.on_post_optimizer_step(_args(), _make_accel(), net, None, sync_gradients=True, global_step=1)  # no raise
    assert trainer.prior_lora_ema is None


def test_warns_once_when_ema_cannot_init_within_run(monkeypatch, caplog):
    trainer = itn.Ideogram4NetworkTrainer()
    net = _ParamNetwork()
    args = _args(max_train_steps=50)  # < the 100-step init floor
    with caplog.at_level(logging.WARNING, logger=TRAINER_LOGGER):
        _run_process_batch(trainer, args, net, monkeypatch, global_step=0)
        _run_process_batch(trainer, args, net, monkeypatch, global_step=1)
    warnings = [r for r in caplog.records if "will not initialize" in r.message]
    assert len(warnings) == 1  # one-shot
