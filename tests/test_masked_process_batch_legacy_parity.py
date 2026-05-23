"""Behavioral no-op parity: pre-merge inline masked-loss orchestration vs masked_process_batch.

The v0.3.0 migration moved blissful-tuner's masked-loss + prior-preservation orchestration out
of the monolithic ``hv_train_network.py`` training loop into
``blissful_tuner.mask_loss_process_batch.masked_process_batch``. This test answers the narrow
question the GPU 20-step gate would otherwise answer, but without its confounders: **did that
move change behavior?**

``_legacy_masked_process`` below is a hermetic copy of the pre-merge inline orchestration
(``hv_train_network.py`` at ``1c3fb0d`` / pre-merge ``HEAD``, the block around lines 2594-2881
plus the per-step telemetry at 2940-2989). It is intentionally narrow — orchestration parity
only, NOT trainer/model/optimizer/dataloader integration. The legacy block keeps its original
shape: tuple ``call_dit`` returns, ``n_dim=latents.ndim`` loss weighting, local-then-self EMA.

Both paths run against identical deterministic fakes (same noisy/timesteps, same per-call
``call_dit`` outputs, same masks) so any divergence in the migrated loss/telemetry surfaces as a
parity failure. The expensive end-to-end WAN GPU parity run is deferred for environment reasons
(finder-based editable installs prevent a sibling worktree from sharing this venv); see the
merge commit body.
"""

import argparse
from contextlib import contextmanager

import pytest
import torch

import blissful_tuner.mask_loss_process_batch as mpb
from blissful_tuner.mask_loss_process_batch import masked_process_batch, prior_model_context
from musubi_tuner.hv_train_network import DiTOutput
from musubi_tuner.modules.loss_utils import compute_unreduced_target_loss
from musubi_tuner.modules.mask_loss import apply_masked_loss_with_prior, require_mask_weights_if_enabled
from musubi_tuner.modules.prior_scheduling import compute_prior_weight_per_sample
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3


# ---------------------------------------------------------------------------
# Deterministic fakes shared by both paths (identical inputs => identical math)
# ---------------------------------------------------------------------------


class _SpyEma:
    """Stand-in for LoRAEmaTeacher; the counter-based fake call_dit ignores adapter weights,
    so the EMA's only parity-relevant effect is that the teacher forward happens at all."""

    def __init__(self, decay: float) -> None:
        self.decay = decay

    def init_from(self, network) -> None:
        pass

    @contextmanager
    def apply_to(self, network):
        yield

    def update(self, network) -> None:
        pass


class _FakeNetwork:
    def __init__(self) -> None:
        self.enabled = True

    def set_enabled(self, value: bool) -> None:
        self.enabled = value


class _FakeAccelerator:
    def __init__(self) -> None:
        self.trackers = ["tb"]
        self.device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


class _FakeTrainer:
    """Exposes only the seams both the legacy block and masked_process_batch call.

    ``dit_mode`` selects the call_dit return contract: ``"legacy"`` returns a bare
    ``(pred, target)`` tuple (pre-merge shape); ``"new"`` returns a ``DiTOutput``. Both modes
    yield the SAME per-call pred/target tensors (seeded by call index) so the two paths see
    identical model outputs in identical order (teacher forward = call 0, student = call 1)."""

    def __init__(self, dit_mode: str) -> None:
        self.dit_mode = dit_mode
        self.call_count = 0
        self.restore_calls = 0
        self.prior_lora_ema = None

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps_in, noise_scheduler, device, dtype):
        return latents, torch.full((latents.shape[0],), 500.0)

    def _pred_target(self):
        idx = self.call_count
        self.call_count += 1
        g = torch.Generator().manual_seed(1000 + idx)
        pred = torch.randn(1, 4, 1, 4, 4, generator=g)
        target = torch.randn(1, 4, 1, 4, 4, generator=g)
        return pred, target

    def call_dit(self, *args, **kwargs):
        pred, target = self._pred_target()
        if self.dit_mode == "new":
            return DiTOutput(pred=pred, target=target)
        return pred, target  # pre-merge tuple contract

    def restore_block_swap_after_no_grad_forward(self, accelerator, transformer):
        self.restore_calls += 1


# ---------------------------------------------------------------------------
# Legacy reference: hermetic copy of the pre-merge inline orchestration.
# Source: hv_train_network.py @ 1c3fb0d (pre-merge HEAD), ~lines 2594-2881 (orchestration)
# and 2940-2989 (per-step masked-loss telemetry). Trimmed to orchestration only — no
# scale_shift/noise/backward/optimizer/scale_weight_norms (the base loop owned those).
# Kept intentionally narrow: a behavioral reference, not an embedded copy of the monolith.
# ---------------------------------------------------------------------------


def _legacy_masked_process(self, args, accelerator, transformer, network, batch, latents, noise, noise_scheduler, dit_dtype, network_dtype, global_step):
    # === Prior scheduling / EMA teacher configuration (pre-merge :2594-2613) ===
    prior_decay_schedule = str(getattr(args, "prior_decay_schedule", "constant"))
    prior_decay_timestep_start = float(getattr(args, "prior_decay_timestep_start", 300.0))
    prior_decay_warmup_ratio = float(getattr(args, "prior_decay_warmup_ratio", 0.0))
    prior_schedule_enabled = (prior_decay_schedule != "constant") or (prior_decay_warmup_ratio > 0.0)
    prior_decay_warmup_steps = int(args.max_train_steps * prior_decay_warmup_ratio) if prior_decay_warmup_ratio > 0 else 0

    prior_teacher_mode = str(getattr(args, "prior_teacher_mode", "base"))
    prior_teacher_ema_decay = float(getattr(args, "prior_teacher_ema_decay", 0.999))
    prior_teacher_ema_min_init_steps = 100
    prior_teacher_ema_init_step = max(prior_teacher_ema_min_init_steps, prior_decay_warmup_steps)

    require_mask_weights_if_enabled(batch, args, cache_hint="(legacy parity fixture)")

    # calculate model input and timesteps (pre-merge :2655)
    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
        args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
    )

    # pre-merge passed n_dim=latents.ndim (W14). The new path uses loss_unreduced.ndim; equal value.
    weighting = compute_loss_weighting_for_sd3(
        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype, n_dim=latents.ndim
    )

    prior_pred = None
    prior_teacher_uses_ema = False
    prior_preservation_weight = getattr(args, "prior_preservation_weight", 0.0)
    mask_weights = batch.get("mask_weights")
    need_prior_base = prior_preservation_weight > 0 and getattr(args, "use_mask_loss", False) and mask_weights is not None

    prior_weight_per_sample = None
    if need_prior_base and prior_schedule_enabled:
        prior_weight_per_sample = compute_prior_weight_per_sample(
            timesteps,
            base_weight=float(prior_preservation_weight),
            schedule=prior_decay_schedule,
            pivot_timestep=prior_decay_timestep_start,
            global_step=global_step,
            warmup_steps=prior_decay_warmup_steps,
        )

    if (
        need_prior_base
        and prior_teacher_mode == "ema"
        and self.prior_lora_ema is None
        and global_step >= prior_teacher_ema_init_step
    ):
        self.prior_lora_ema = _SpyEma(decay=prior_teacher_ema_decay)
        self.prior_lora_ema.init_from(accelerator.unwrap_model(network))

    prior_timestep_threshold = getattr(args, "prior_preservation_timestep_threshold", None)
    prior_sample_mask = None
    need_prior = need_prior_base
    if need_prior and (prior_timestep_threshold is not None or prior_weight_per_sample is not None):
        if prior_timestep_threshold is not None:
            do_prior = timesteps > float(prior_timestep_threshold)
        else:
            do_prior = torch.ones_like(timesteps, dtype=torch.bool)
        if prior_weight_per_sample is not None:
            do_prior = do_prior & (prior_weight_per_sample > 0)
        if mask_weights is not None:
            if mask_weights.ndim == 5:
                mask_min_per_sample = mask_weights.amin(dim=(1, 2, 3, 4))
            elif mask_weights.ndim == 4:
                mask_min_per_sample = mask_weights.amin(dim=(1, 2, 3))
            else:
                mask_min_per_sample = mask_weights.view(mask_weights.shape[0], -1).amin(dim=1)
        else:
            mask_min_per_sample = None
        prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
        if mask_min_per_sample is not None:
            if prior_mask_threshold is not None:
                has_prior_region = mask_min_per_sample < float(prior_mask_threshold)
            else:
                has_prior_region = mask_min_per_sample < (1.0 - 1e-6)
            has_prior_region = has_prior_region.to(device=do_prior.device)
            do_prior = do_prior & has_prior_region
        prior_sample_mask = do_prior
        need_prior = bool(do_prior.any().item())
    elif need_prior:
        mask_min = float(mask_weights.min())
        prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
        if prior_mask_threshold is not None:
            if mask_min >= float(prior_mask_threshold):
                need_prior = False
        else:
            if mask_min >= (1.0 - 1e-6):
                need_prior = False

    if need_prior:
        prior_teacher_eval = bool(getattr(args, "prior_teacher_eval", False))
        transformer_was_training = transformer.training if prior_teacher_eval else None
        if prior_teacher_eval:
            transformer.eval()
        try:
            with torch.no_grad():
                unwrapped_network = accelerator.unwrap_model(network)
                prior_teacher_uses_ema = prior_teacher_mode == "ema" and self.prior_lora_ema is not None
                prior_context = (
                    self.prior_lora_ema.apply_to(unwrapped_network)
                    if prior_teacher_uses_ema
                    else prior_model_context(unwrapped_network)
                )
                with prior_context:
                    prior_pred_raw, _ = self.call_dit(  # pre-merge tuple unpack
                        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype
                    )
                self.restore_block_swap_after_no_grad_forward(accelerator, transformer)
            prior_pred = prior_pred_raw.detach()
        finally:
            if prior_teacher_eval and transformer_was_training:
                transformer.train()

    model_pred, target = self.call_dit(  # pre-merge tuple unpack
        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype
    )
    loss_type = getattr(args, "loss_type", "mse")
    loss_delta = getattr(args, "loss_delta", 1.0)
    model_pred_cast = model_pred.to(network_dtype)
    target_cast = target.to(network_dtype)
    loss_unreduced = compute_unreduced_target_loss(model_pred_cast, target_cast, loss_type=loss_type, loss_delta=loss_delta)

    log_huber_stats = str(loss_type).lower() == "huber" and len(accelerator.trackers) > 0 and bool(getattr(args, "use_mask_loss", False))
    huber_threshold = 0.5 * float(loss_delta) * float(loss_delta)
    target_huber_is_linear = (loss_unreduced > huber_threshold) if log_huber_stats else None

    loss = loss_unreduced
    if weighting is not None:
        loss = loss * weighting

    prior_loss_unreduced = None
    prior_huber_is_linear = None
    if prior_pred is not None:
        prior_loss_unreduced = compute_unreduced_target_loss(
            model_pred_cast, prior_pred.to(network_dtype), loss_type=loss_type, loss_delta=loss_delta
        )
        prior_huber_is_linear = (prior_loss_unreduced > huber_threshold) if log_huber_stats else None
        if weighting is not None:
            prior_loss_unreduced = prior_loss_unreduced * weighting

    layout = "layered" if getattr(args, "is_layered", False) else "video"
    drop_base_frame = bool(getattr(args, "remove_first_image_from_target", False)) if layout == "layered" else False
    mask_loss_stats = {} if len(accelerator.trackers) > 0 and bool(getattr(args, "use_mask_loss", False)) else None
    loss = apply_masked_loss_with_prior(
        loss,
        mask_weights,
        prior_loss_unreduced=prior_loss_unreduced,
        prior_sample_mask=prior_sample_mask,
        prior_weight_per_sample=prior_weight_per_sample,
        target_huber_is_linear=target_huber_is_linear,
        prior_huber_is_linear=prior_huber_is_linear,
        stats=mask_loss_stats,
        args=args,
        layout=layout,
        drop_base_frame=drop_base_frame,
    )

    # === per-step masked-loss telemetry (pre-merge :2940-2989) ===
    metrics: dict[str, float] = {}
    if len(accelerator.trackers) > 0:
        if mask_loss_stats:
            for k, v in mask_loss_stats.items():
                if isinstance(v, torch.Tensor):
                    metrics[f"masked_loss/{k}"] = float(v.detach().float().item())
            if "target" in mask_loss_stats and "prior" in mask_loss_stats:
                t = float(mask_loss_stats["target"].detach().float().item())
                p = float(mask_loss_stats["prior"].detach().float().item())
                denom = t + p + 1e-8
                metrics["masked_loss/target_fraction"] = t / denom
                metrics["masked_loss/prior_fraction"] = p / denom
        if isinstance(timesteps, torch.Tensor) and timesteps.numel() > 0:
            t_stat = timesteps.detach().to(dtype=torch.float32)
            metrics["timestep/mean"] = float(t_stat.mean().item())
            metrics["timestep/min"] = float(t_stat.min().item())
            metrics["timestep/max"] = float(t_stat.max().item())
        prior_weight = float(getattr(args, "prior_preservation_weight", 0.0))
        if prior_weight > 0:
            metrics["prior/teacher_ran"] = float(prior_pred is not None)
            metrics["prior/teacher_mode_ema_used"] = float(prior_teacher_uses_ema)
            if prior_sample_mask is not None and isinstance(prior_sample_mask, torch.Tensor):
                metrics["prior/gate_frac"] = float(prior_sample_mask.detach().to(dtype=torch.float32).mean().item())
            else:
                metrics["prior/gate_frac"] = 1.0 if prior_pred is not None else 0.0
            if prior_weight_per_sample is not None and isinstance(prior_weight_per_sample, torch.Tensor):
                w = prior_weight_per_sample.detach().to(dtype=torch.float32)
                metrics["prior/w_prior_mean"] = float(w.mean().item())
                metrics["prior/w_prior_min"] = float(w.min().item())
                metrics["prior/w_prior_max"] = float(w.max().item())
            else:
                metrics["prior/w_prior_mean"] = float(prior_weight)

    return loss, metrics


# ---------------------------------------------------------------------------
# Parity harness
# ---------------------------------------------------------------------------


def _args(**overrides) -> argparse.Namespace:
    base = dict(
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
    base.update(overrides)
    return argparse.Namespace(**base)


def _mask(prior_region: bool) -> torch.Tensor:
    m = torch.ones(1, 1, 1, 4, 4)
    if prior_region:
        m[..., 2:, :] = 0.0
    return m


def _run_new(args, mask, global_step, monkeypatch):
    monkeypatch.setattr(mpb, "LoRAEmaTeacher", _SpyEma)
    network = _FakeNetwork()
    trainer = _FakeTrainer("new")
    acc = _FakeAccelerator()
    latents = torch.zeros(1, 4, 1, 4, 4)
    loss, metrics = masked_process_batch(
        trainer, args, acc, "transformer", network, {"timesteps": None, "mask_weights": mask},
        latents, torch.zeros_like(latents), None, torch.float32, torch.float32, None, global_step,
    )
    return loss, metrics, trainer


def _run_legacy(args, mask, global_step):
    network = _FakeNetwork()
    trainer = _FakeTrainer("legacy")
    acc = _FakeAccelerator()
    latents = torch.zeros(1, 4, 1, 4, 4)
    loss, metrics = _legacy_masked_process(
        trainer, args, acc, "transformer", network, {"timesteps": None, "mask_weights": mask},
        latents, torch.zeros_like(latents), None, torch.float32, torch.float32, global_step,
    )
    return loss, metrics, trainer


def _assert_parity(new, legacy):
    new_loss, new_metrics, new_trainer = new
    old_loss, old_metrics, old_trainer = legacy
    torch.testing.assert_close(new_loss, old_loss, atol=1e-6, rtol=0)
    assert new_metrics.keys() == old_metrics.keys(), f"telemetry key drift: {new_metrics.keys()} vs {old_metrics.keys()}"
    for key in old_metrics:
        assert new_metrics[key] == pytest.approx(old_metrics[key], abs=1e-6), f"metric {key}: {new_metrics[key]} != {old_metrics[key]}"
    # Same number of forwards + same block-swap restore count (no-grad teacher restore parity).
    assert new_trainer.call_count == old_trainer.call_count
    assert new_trainer.restore_calls == old_trainer.restore_calls


@pytest.mark.parametrize("prior_weight", [0.0, 1.0])
def test_parity_base_mode(prior_weight, monkeypatch):
    """No-prior and base-mode prior: new masked_process_batch == pre-merge inline orchestration."""
    args = _args(prior_preservation_weight=prior_weight, prior_teacher_mode="base")
    mask = _mask(prior_region=prior_weight > 0)
    _assert_parity(_run_new(args, mask, 10, monkeypatch), _run_legacy(args, mask, 10))


def test_parity_ema_mode_at_warmup(monkeypatch):
    """EMA-mode prior past warmup: parity including EMA lazy-init + apply_to teacher path."""
    args = _args(prior_preservation_weight=1.0, prior_teacher_mode="ema")
    mask = _mask(prior_region=True)
    _assert_parity(_run_new(args, mask, 150, monkeypatch), _run_legacy(args, mask, 150))


def test_parity_ema_mode_before_warmup(monkeypatch):
    """EMA-mode before warmup: both fall back to base-mode teacher (LoRA-disabled) identically."""
    args = _args(prior_preservation_weight=1.0, prior_teacher_mode="ema")
    mask = _mask(prior_region=True)
    _assert_parity(_run_new(args, mask, 10, monkeypatch), _run_legacy(args, mask, 10))


def test_parity_timestep_scheduled_prior(monkeypatch):
    """Prior-decay schedule active: per-sample prior_weight + gating parity."""
    args = _args(
        prior_preservation_weight=1.0,
        prior_teacher_mode="base",
        prior_decay_schedule="cosine",
        prior_decay_timestep_start=600.0,
        normalize_per_sample=True,  # apply_masked_loss_with_prior requires this when prior_weight_per_sample is set
    )
    mask = _mask(prior_region=True)
    _assert_parity(_run_new(args, mask, 50, monkeypatch), _run_legacy(args, mask, 50))
