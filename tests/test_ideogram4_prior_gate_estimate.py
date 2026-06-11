# Prior-gate skip-rate startup estimate (backlog P0-8b).
#
# The v8 run set prior_preservation_timestep_threshold=250 expecting ~25-40% teacher savings and
# measured 3.3%: at 1024x1024 the resolution-aware schedule shifts the logit-normal mean by
# 0.5*ln(1024^2/512^2) ~= 0.693, and P(traditional_t <= 250) = Phi((ln(1/3) - 0.693)/1) ~= 3.7% —
# the analytic value matches the run telemetry. estimate_prior_gate_skip_rate simulates the EXACT
# production sampling path so the logged expectation can never drift from what training does.

import logging
from types import SimpleNamespace

import torch

import musubi_tuner.ideogram4_train_network as itn
from musubi_tuner.ideogram4.training import estimate_prior_gate_skip_rate

TRAINER_LOGGER = "musubi_tuner.ideogram4_train_network"


# --- the estimator itself ---------------------------------------------------------------------------


def test_matches_v8_empirical_rate_at_1024():
    # Pins both the v8 run telemetry (3.3%) and the analytic value (~3.7%) — the whole reason this
    # feature exists. A change to the schedule or the remap that moves this materially is a real
    # training-behavior change and should fail here first.
    rate = estimate_prior_gate_skip_rate(1024, 1024, timestep_mu=0.0, timestep_std=1.0, threshold=250.0)
    assert 0.02 <= rate <= 0.05, rate


def test_monotonic_in_threshold():
    rates = [
        estimate_prior_gate_skip_rate(1024, 1024, timestep_mu=0.0, timestep_std=1.0, threshold=t)
        for t in (0.0, 250.0, 500.0, 700.0, 1000.0)
    ]
    assert rates == sorted(rates)
    assert rates[0] <= 0.001  # threshold 0 skips (essentially) nothing
    assert rates[-1] >= 0.999  # threshold 1000 skips (essentially) everything


def test_resolution_shifts_the_rate():
    # The resolution shift is the part naive round-number reasoning misses: smaller images put MORE
    # mass at low traditional_t (higher cleanness), so the same threshold skips more.
    at_512 = estimate_prior_gate_skip_rate(512, 512, timestep_mu=0.0, timestep_std=1.0, threshold=250.0)
    at_1024 = estimate_prior_gate_skip_rate(1024, 1024, timestep_mu=0.0, timestep_std=1.0, threshold=250.0)
    assert at_512 > at_1024


def test_deterministic():
    a = estimate_prior_gate_skip_rate(1024, 1024, timestep_mu=0.0, timestep_std=1.0, threshold=250.0)
    b = estimate_prior_gate_skip_rate(1024, 1024, timestep_mu=0.0, timestep_std=1.0, threshold=250.0)
    assert a == b  # fixed seed: a property of the configuration, not a sample


# --- trainer wiring (one-shot, only when the gate is active) ----------------------------------------
# Mirrors the harness in tests/test_ideogram4_prior_preservation.py.


def _make_accel():
    return SimpleNamespace(device=torch.device("cpu"), trackers=[], unwrap_model=lambda m: m)


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
        prior_teacher_mode="base",
        prior_decay_schedule="constant",
        prior_decay_timestep_start=300.0,
        prior_decay_warmup_ratio=0.0,
        prior_preservation_timestep_threshold=250.0,
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


def _run_process_batch(trainer, args, monkeypatch):
    monkeypatch.setattr(itn, "ideogram4_flow_matching_target", _fake_flow)
    monkeypatch.setattr(itn, "get_schedule_for_resolution", lambda reso, *, known_mean, std: lambda u: u)
    latents, batch = _batch()
    trainer.process_batch(
        args, _make_accel(), "TRANSFORMER", _StubNetwork(), batch, latents, torch.ones_like(latents),
        None, torch.float32, torch.float32, None, 0,
    )


class _StubNetwork:
    def set_enabled(self, enabled):
        pass


def test_estimate_logged_once_when_gate_active(monkeypatch, caplog):
    trainer = itn.Ideogram4NetworkTrainer()
    args = _args()
    with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
        _run_process_batch(trainer, args, monkeypatch)
        _run_process_batch(trainer, args, monkeypatch)
    estimates = [r for r in caplog.records if "expected to skip" in r.message]
    assert len(estimates) == 1  # one-shot
    assert "threshold=250" in estimates[0].message


def test_no_estimate_without_threshold(monkeypatch, caplog):
    trainer = itn.Ideogram4NetworkTrainer()
    args = _args(prior_preservation_timestep_threshold=None)
    with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
        _run_process_batch(trainer, args, monkeypatch)
    assert not [r for r in caplog.records if "expected to skip" in r.message]


def test_no_estimate_without_prior_preservation(monkeypatch, caplog):
    trainer = itn.Ideogram4NetworkTrainer()
    args = _args(prior_preservation_weight=0.0)
    with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
        _run_process_batch(trainer, args, monkeypatch)
    assert not [r for r in caplog.records if "expected to skip" in r.message]


def test_mu_none_tolerated(monkeypatch, caplog):
    # _base_args-style configs (and the parser pre-defaulting) can leave mu as None; the trainer
    # call defends with `or 0.0` rather than crashing the first training step.
    trainer = itn.Ideogram4NetworkTrainer()
    args = _args(ideogram4_timestep_mu=None)
    with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
        _run_process_batch(trainer, args, monkeypatch)
    assert [r for r in caplog.records if "expected to skip" in r.message]
