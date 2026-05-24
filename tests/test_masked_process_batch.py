"""Runtime coverage for the masked_process_batch orchestration body.

The dispatch test mocks ``masked_process_batch`` and the existing mask tests call
``apply_masked_loss_with_prior`` directly, so without this file the orchestration body
(prior-teacher no-grad forward, adapter-disable context, block-swap restore between the
teacher and student forwards, telemetry assembly) is only validated by py_compile.

These tests drive the real ``masked_process_batch`` through a minimal fake trainer that
exposes only the seams it calls. They assert *mechanics* — call counts, adapter state
during each forward, restore timing, telemetry key presence — not masked-loss numerics
(those are covered by ``test_wan_mask_loss_integration`` / ``test_mask_loss``).
"""

import argparse
from contextlib import contextmanager

import pytest
import torch

import blissful_tuner.mask_loss_process_batch as mpb
from blissful_tuner.mask_loss_process_batch import masked_process_batch
from musubi_tuner.hv_train_network import DiTOutput


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        use_mask_loss=True,
        loss_type="mse",
        loss_delta=1.0,
        weighting_scheme="none",
        max_train_steps=1000,
        prior_preservation_weight=0.0,
        prior_teacher_mode="base",
        prior_teacher_ema_decay=0.999,
        prior_decay_schedule="constant",
        prior_decay_timestep_start=300.0,
        prior_decay_warmup_ratio=0.0,
        prior_preservation_timestep_threshold=None,
        prior_mask_threshold=None,
        prior_teacher_eval=False,
        is_layered=False,
        mask_gamma=1.0,
        mask_min_weight=0.0,
        normalize_per_sample=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _FakeNetwork:
    """Records adapter enable/disable so we can assert the teacher pass ran with LoRA off."""

    def __init__(self) -> None:
        self.enabled = True
        self.history: list[bool] = []

    def set_enabled(self, value: bool) -> None:
        self.enabled = value
        self.history.append(value)


class _FakeAccelerator:
    def __init__(self, with_trackers: bool = True) -> None:
        self.trackers = ["tb"] if with_trackers else []
        self.device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


class _FakeTrainer:
    """Minimal trainer exposing only the seams ``masked_process_batch`` invokes."""

    def __init__(self, pred: torch.Tensor, target: torch.Tensor, network: _FakeNetwork) -> None:
        self._pred = pred
        self._target = target
        self._network = network
        self.call_dit_count = 0
        self.restore_calls = 0
        self.adapter_enabled_per_call: list[bool] = []

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps_in, noise_scheduler, device, dit_dtype):
        bsz = latents.shape[0]
        timesteps = torch.full((bsz,), 500.0)  # mid-range; structural enough for any gating
        return latents, timesteps

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy, timesteps, network_dtype):
        # Snapshot adapter state so we can prove the teacher pass ran with LoRA disabled.
        self.adapter_enabled_per_call.append(self._network.enabled)
        self.call_dit_count += 1
        return DiTOutput(pred=self._pred.clone(), target=self._target.clone())

    def restore_block_swap_after_no_grad_forward(self, accelerator, transformer):
        self.restore_calls += 1


class _SpyEma:
    """Stand-in for LoRAEmaTeacher that records lifecycle calls without touching EMA internals."""

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.init_from_calls = 0
        self.apply_to_entered = 0
        self.update_calls = 0

    def init_from(self, network) -> None:
        self.init_from_calls += 1

    @contextmanager
    def apply_to(self, network):
        self.apply_to_entered += 1
        yield

    def update(self, network) -> None:
        self.update_calls += 1


def _make_batch(mask: torch.Tensor) -> dict:
    return {"timesteps": None, "mask_weights": mask}


def _run_at_step(trainer, network, args, mask, global_step):
    acc = _FakeAccelerator()
    latents = torch.zeros(1, 4, 1, 4, 4)
    noise = torch.zeros_like(latents)
    return masked_process_batch(
        trainer,
        args,
        acc,
        "transformer",
        network,
        _make_batch(mask),
        latents,
        noise,
        noise_scheduler=None,
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
        vae=None,
        global_step=global_step,
    )


def _run(trainer, network, args, mask):
    return _run_at_step(trainer, network, args, mask, global_step=10)


def test_no_prior_runs_student_only():
    """prior_preservation_weight=0 → one (student) forward, no teacher, no block-swap restore."""
    network = _FakeNetwork()
    pred = torch.zeros(1, 4, 1, 4, 4)
    target = torch.ones(1, 4, 1, 4, 4)
    trainer = _FakeTrainer(pred, target, network)
    args = _args(prior_preservation_weight=0.0)
    mask = torch.ones(1, 1, 1, 4, 4)  # full-weight mask

    loss, metrics = _run(trainer, network, args, mask)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert trainer.call_dit_count == 1, "no prior → student forward only"
    assert trainer.restore_calls == 0, "no teacher forward → no block-swap restore"
    assert network.history == [], "adapter must never be toggled when prior is off"
    # Telemetry: masked_loss/* present, prior/* absent (prior weight == 0).
    assert any(k.startswith("masked_loss/") for k in metrics)
    assert not any(k.startswith("prior/") for k in metrics)


def test_base_prior_runs_teacher_then_student_with_restore():
    """Base-mode prior → teacher forward (LoRA disabled) then student forward (LoRA on),
    with the block-swap restore firing exactly once between them."""
    network = _FakeNetwork()
    pred = torch.zeros(1, 4, 1, 4, 4)
    target = torch.ones(1, 4, 1, 4, 4)
    trainer = _FakeTrainer(pred, target, network)
    args = _args(prior_preservation_weight=1.0, prior_teacher_mode="base")
    # Mask with a background (zero) region so a prior region exists and need_prior stays True.
    mask = torch.ones(1, 1, 1, 4, 4)
    mask[..., 2:, :] = 0.0

    loss, metrics = _run(trainer, network, args, mask)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert trainer.call_dit_count == 2, "prior → teacher + student forwards"
    assert trainer.restore_calls == 1, "block-swap restore fires once, between teacher and student"
    # Teacher forward ran with adapter disabled; student forward ran with it re-enabled.
    assert trainer.adapter_enabled_per_call == [False, True]
    # Context manager left the adapter enabled afterwards.
    assert network.enabled is True
    assert metrics.get("prior/teacher_ran") == 1.0
    assert metrics.get("prior/teacher_mode_ema_used") == 0.0


def test_block_swap_restore_runs_even_if_teacher_forward_raises():
    """Regression (restore-in-finally): if the prior-teacher ``call_dit`` raises mid-forward,
    the block-swap restore must STILL run — otherwise swapped blocks stay on CPU and the next
    training forward crashes with "mat2 is on cpu". The restore lives in a ``finally`` so the
    raising teacher path can't strand them. See [[project_block_swap_no_grad_invariant]]."""

    class _RaisingTeacherTrainer(_FakeTrainer):
        def call_dit(self, *args, **kwargs):
            # Raise on the (first) teacher forward, before the student forward is reached.
            self.call_dit_count += 1
            raise RuntimeError("boom in teacher forward")

    network = _FakeNetwork()
    pred = torch.zeros(1, 4, 1, 4, 4)
    target = torch.ones(1, 4, 1, 4, 4)
    trainer = _RaisingTeacherTrainer(pred, target, network)
    args = _args(prior_preservation_weight=1.0, prior_teacher_mode="base")
    mask = torch.ones(1, 1, 1, 4, 4)
    mask[..., 2:, :] = 0.0  # background region → need_prior True → teacher forward attempted

    with pytest.raises(RuntimeError, match="boom in teacher forward"):
        _run(trainer, network, args, mask)

    assert trainer.call_dit_count == 1, "raised during the teacher forward, before the student forward"
    assert trainer.restore_calls == 1, "block-swap restore must fire in finally even when the teacher raises"
    assert network.enabled is True, "prior_model_context finally must re-enable the adapter despite the raise"


def test_all_ones_mask_skips_teacher_forward():
    """Continuous-mode optimization: an all-ones mask has no prior region → teacher skipped."""
    network = _FakeNetwork()
    pred = torch.zeros(1, 4, 1, 4, 4)
    target = torch.ones(1, 4, 1, 4, 4)
    trainer = _FakeTrainer(pred, target, network)
    args = _args(prior_preservation_weight=1.0, prior_teacher_mode="base")
    mask = torch.ones(1, 1, 1, 4, 4)  # no background → no prior region

    loss, metrics = _run(trainer, network, args, mask)

    assert trainer.call_dit_count == 1, "all-ones mask → prior region empty → student only"
    assert trainer.restore_calls == 0
    assert metrics.get("prior/teacher_ran") == 0.0


def test_no_trackers_skips_telemetry_build():
    """With no accelerator trackers, telemetry (and its GPU syncs) is skipped → empty metrics."""
    network = _FakeNetwork()
    pred = torch.zeros(1, 4, 1, 4, 4)
    target = torch.ones(1, 4, 1, 4, 4)
    trainer = _FakeTrainer(pred, target, network)
    args = _args(prior_preservation_weight=0.0)
    mask = torch.ones(1, 1, 1, 4, 4)

    acc = _FakeAccelerator(with_trackers=False)
    latents = torch.zeros(1, 4, 1, 4, 4)
    loss, metrics = masked_process_batch(
        trainer,
        args,
        acc,
        "transformer",
        network,
        _make_batch(mask),
        latents,
        torch.zeros_like(latents),
        None,
        torch.float32,
        torch.float32,
        None,
        10,
    )

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics == {}, "no trackers → no telemetry assembled"


def _prior_region_mask() -> torch.Tensor:
    mask = torch.ones(1, 1, 1, 4, 4)
    mask[..., 2:, :] = 0.0  # background half → a prior region exists, need_prior stays True
    return mask


def test_ema_teacher_lazy_init_and_apply_at_warmup(monkeypatch):
    """EMA mode past warmup: teacher lazily constructed + init_from'd, applied via apply_to
    (NOT the base set_enabled toggle), and telemetry reports EMA was used."""
    monkeypatch.setattr(mpb, "LoRAEmaTeacher", _SpyEma)
    network = _FakeNetwork()
    trainer = _FakeTrainer(torch.zeros(1, 4, 1, 4, 4), torch.ones(1, 4, 1, 4, 4), network)
    args = _args(prior_preservation_weight=1.0, prior_teacher_mode="ema", prior_teacher_ema_decay=0.999)

    # global_step=150 >= prior_teacher_ema_init_step (max(100, 0)) → EMA lazy-init fires.
    loss, metrics = _run_at_step(trainer, network, args, _prior_region_mask(), global_step=150)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert isinstance(trainer.prior_lora_ema, _SpyEma)
    assert trainer.prior_lora_ema.decay == 0.999
    assert trainer.prior_lora_ema.init_from_calls == 1, "EMA teacher initialized once at warmup"
    assert trainer.prior_lora_ema.apply_to_entered == 1, "teacher forward ran under EMA apply_to context"
    assert trainer.call_dit_count == 2 and trainer.restore_calls == 1
    assert network.history == [], "EMA mode must use apply_to, not the base set_enabled toggle"
    assert metrics.get("prior/teacher_ran") == 1.0
    assert metrics.get("prior/teacher_mode_ema_used") == 1.0


def test_ema_mode_before_warmup_falls_back_to_base(monkeypatch):
    """EMA mode before warmup step: teacher not yet initialized → base-mode fallback
    (adapter toggled off/on) and telemetry reports EMA not used."""
    monkeypatch.setattr(mpb, "LoRAEmaTeacher", _SpyEma)
    network = _FakeNetwork()
    trainer = _FakeTrainer(torch.zeros(1, 4, 1, 4, 4), torch.ones(1, 4, 1, 4, 4), network)
    args = _args(prior_preservation_weight=1.0, prior_teacher_mode="ema")

    # global_step=10 < 100 → no EMA init; teacher runs base-mode (LoRA disabled then restored).
    loss, metrics = _run_at_step(trainer, network, args, _prior_region_mask(), global_step=10)

    assert getattr(trainer, "prior_lora_ema", None) is None, "EMA must not initialize before warmup"
    assert trainer.call_dit_count == 2 and trainer.restore_calls == 1
    assert trainer.adapter_enabled_per_call == [False, True], "base-mode fallback toggles the adapter"
    assert metrics.get("prior/teacher_mode_ema_used") == 0.0
