"""Regression tests for Ideogram 4 prior preservation (Slice 2).

The implementation lives in `src/musubi_tuner/ideogram4_train_network.py:process_batch` and the
shared `apply_masked_loss_with_prior`. These tests focus on the Ideogram-specific routing layer
(teacher forward, t-convention remap, autocast bypass propagation to the teacher) rather than
re-testing the shared reducer's reduction math (covered by `test_mask_loss.py`,
`test_wan_mask_loss_integration.py`, and the masked-loss/prior tests under those names).

The contract being locked down:

1. ``ideogram4_cleanness_to_noise_timestep`` produces the correct (1 - t) * 1000 mapping so that
   FLUX.2-trained scheduling args (--prior_decay_timestep_start=300 etc.) keep their semantics.
2. Teacher forward fires when prior_weight > 0 with masks present; doesn't when prior_weight == 0.
3. LoRA is disabled only around the teacher forward (set_enabled(False) -> teacher -> set_enabled(True)).
4. Both teacher and student go through ``_run_i4_flow_forward`` so they share the autocast bypass.
5. Prior pred is converted to grid layout (B, 128, gh, gw) before reduction — passing token-shape
   would broadcast the mask incorrectly.
6. The timestep threshold uses the REMAPPED t, not Ideogram's cleanness, so user args mean what they say.
7. apply_masked_loss_with_prior receives a non-None prior_loss_unreduced when teacher ran.
8. EMA teacher mode is rejected LOUDLY in handle_model_specific_args (no silent fallback to base).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.ideogram4.training import ideogram4_cleanness_to_noise_timestep


# ---------------------------------------------------------------------------------------------------
# Test 1 — Remap math is the named cross-system contract
# ---------------------------------------------------------------------------------------------------


def test_cleanness_to_noise_remap_endpoints_and_midpoint():
    """Pin the (1 - t) * 1000 mapping at the three diagnostic points. If this assertion ever fails,
    the entire prior scheduling/gating logic for Ideogram 4 is silently inverted — schedule fires at
    LOW noise instead of HIGH noise, gate threshold rejects the wrong half of timesteps. Catches sign
    inversion at the contract boundary, where it is cheapest to fix."""
    t = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float32)
    out = ideogram4_cleanness_to_noise_timestep(t)
    expected = torch.tensor([1000.0, 700.0, 0.0])
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_cleanness_to_noise_remap_preserves_shape_and_promotes_dtype():
    """Helper accepts arbitrary tensor shapes (per-sample timesteps are (B,)) and promotes to fp32
    so downstream comparisons against fp32 scheduler args (pivot, threshold) avoid silent
    bf16-rounding surprises at boundary values (e.g., a bf16 0.3 doesn't quantize cleanly to 700.0)."""
    t = torch.rand(8, dtype=torch.bfloat16)
    out = ideogram4_cleanness_to_noise_timestep(t)
    assert out.shape == (8,), f"shape preservation broken: {out.shape}"
    assert out.dtype == torch.float32, f"dtype promotion broken: {out.dtype}"


# ---------------------------------------------------------------------------------------------------
# Test infrastructure — shared stubs for the trainer routing tests
# ---------------------------------------------------------------------------------------------------


class _RecordingNetwork:
    """Mock LoRA network that records set_enabled() call order, so tests can assert LoRA was
    disabled only around the teacher forward (the symmetric set_enabled(True) restore in
    prior_model_context's finally block proves restoration on the happy path)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def set_enabled(self, enabled: bool) -> None:
        self.calls.append(f"set_enabled({enabled})")


def _make_accel(trackers=()):
    """Tiny Accelerator stub. unwrap_model is identity (no-DDP single-device). trackers=[obj] enables
    stats; empty trackers suppresses .item() syncs in process_batch's telemetry block."""
    return SimpleNamespace(device=torch.device("cpu"), trackers=list(trackers), unwrap_model=lambda m: m)


def _base_args(**overrides):
    """Standard args namespace for process_batch. Tests override specific fields per their scenario."""
    base = dict(
        use_mask_loss=True,
        mask_gamma=1.0,
        mask_min_weight=0.0,
        mask_blur_kernel_size=0,
        mask_blur_radius=0.0,
        mask_area_scale_beta=0.0,
        normalize_per_sample=False,
        prior_preservation_weight=0.0,
        prior_teacher_mode="base",
        prior_decay_schedule="constant",
        prior_decay_timestep_start=300.0,
        prior_decay_warmup_ratio=0.0,
        prior_preservation_timestep_threshold=None,
        loss_type="mse",
        loss_delta=1.0,
        ideogram4_timestep_mu=None,
        ideogram4_timestep_std=1.0,
        max_train_steps=5000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_batch(gh=4, gw=6, *, masked=True):
    """Latent batch shaped like the bucket reader produces. mask_weights is (B, 1, F=1, gh, gw); when
    `masked=True` the mask defaults to a PARTIAL mask (left half background = 0, right half foreground = 1)
    so the no-prior-region skip optimization doesn't fire and teacher tests can observe teacher behavior.
    The all-ones-mask case (which exercises the skip) is tested explicitly in
    test_teacher_skipped_when_all_ones_mask_continuous_mode."""
    latents = torch.zeros(1, 128, gh, gw)
    if masked:
        mask = torch.ones(1, 1, 1, gh, gw)
        mask[..., :, : gw // 2] = 0.0  # left half background — guarantees a prior region exists
    else:
        mask = None
    batch = {"i4_llm_features": [torch.zeros(1, 53248)]}
    if mask is not None:
        batch["mask_weights"] = mask
    return latents, batch


def _make_flow_target(model_pred_value=0.0, target_value=0.5):
    """Factory for the monkey-patched ``ideogram4_flow_matching_target``: returns a closure that
    records each call and returns deterministic token tensors. Returned closure captures `calls`
    so tests can assert how many forwards ran (= 1 for student only, = 2 for teacher + student)."""
    calls: list[dict] = []

    def _flow(transformer, latents, text_features, noise, timesteps, *, network_dtype, device):
        gh, gw = int(latents.shape[2]), int(latents.shape[3])
        calls.append(
            {
                "timesteps": timesteps.detach().clone(),
                "autocast_enabled": torch.is_autocast_enabled("cpu"),
            }
        )
        return (
            torch.full((1, gh * gw, 128), model_pred_value, dtype=network_dtype),
            torch.full((1, gh * gw, 128), target_value, dtype=network_dtype),
        )

    return _flow, calls


def _install_stubs(monkeypatch, flow_callable):
    """Monkey-patch ideogram4_flow_matching_target and get_schedule_for_resolution so process_batch
    doesn't try to load the real DiT or resolve a real resolution schedule."""
    import musubi_tuner.ideogram4_train_network as itn

    monkeypatch.setattr(itn, "ideogram4_flow_matching_target", flow_callable)
    # identity schedule — timesteps == input uniforms, so we can control their values
    monkeypatch.setattr(itn, "get_schedule_for_resolution", lambda reso, *, known_mean, std: lambda u: u)


# ---------------------------------------------------------------------------------------------------
# Test 2 — Teacher fires when prior_weight > 0
# ---------------------------------------------------------------------------------------------------


def test_teacher_fires_when_prior_weight_positive(monkeypatch):
    """With --prior_preservation_weight=0.5, --use_mask_loss=true, and a mask present, the teacher
    forward must run before the student. Total forwards = 2 (teacher under LoRA-disabled context,
    then student under LoRA-enabled). Without this routing the prior_loss_unreduced argument to
    apply_masked_loss_with_prior is None and the prior_preservation_weight value silently has no
    effect — exactly the failure mode the EMA rejection guard also defends against."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    flow, calls = _make_flow_target()
    _install_stubs(monkeypatch, flow)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5)
    latents, batch = _make_batch()
    network = _RecordingNetwork()

    loss, _ = trainer.process_batch(
        args, _make_accel(), None, network, batch, latents, None, None, torch.float32, torch.float32, None, 0
    )

    assert len(calls) == 2, f"expected 2 forwards (teacher + student), got {len(calls)}"
    assert torch.is_tensor(loss)


# ---------------------------------------------------------------------------------------------------
# Test 3 — Teacher does not fire when prior_weight is zero
# ---------------------------------------------------------------------------------------------------


def test_teacher_does_not_fire_when_prior_weight_zero(monkeypatch):
    """Default config (prior_preservation_weight=0.0) must run only the student forward. Skipping
    the teacher when prior is disabled avoids the ~2x training-time cost of the no-grad teacher
    forward on every step. This is also why prior_weight=0 is the safe default — no user opts into
    the cost until they explicitly enable it."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    flow, calls = _make_flow_target()
    _install_stubs(monkeypatch, flow)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.0)
    latents, batch = _make_batch()
    network = _RecordingNetwork()

    trainer.process_batch(args, _make_accel(), None, network, batch, latents, None, None, torch.float32, torch.float32, None, 0)

    assert len(calls) == 1, f"expected 1 forward (student only), got {len(calls)}"
    # Network's set_enabled() must NOT have been called at all (no teacher path entered).
    assert network.calls == [], f"prior_model_context fired despite prior_weight=0: {network.calls}"


# ---------------------------------------------------------------------------------------------------
# Test 4 — LoRA is disabled only for the teacher forward
# ---------------------------------------------------------------------------------------------------


def test_lora_disabled_only_around_teacher_forward(monkeypatch):
    """The sequence MUST be: set_enabled(False) -> teacher forward -> set_enabled(True) -> student
    forward. If the student runs with LoRA disabled, it computes against the base model and the
    LoRA receives no gradient signal — training appears to proceed but the adapter never learns.
    If LoRA is never disabled, the teacher computes against the LoRA-active model and the prior
    target becomes "what the LoRA already predicts," collapsing the prior loss to ~0 and
    eliminating the preservation effect.

    We check ordering by interleaving network.set_enabled() calls with forward records."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    network = _RecordingNetwork()
    events: list[str] = []

    def _flow(transformer, latents, text_features, noise, timesteps, *, network_dtype, device):
        gh, gw = int(latents.shape[2]), int(latents.shape[3])
        events.append("forward")
        return (
            torch.zeros(1, gh * gw, 128, dtype=network_dtype),
            torch.full((1, gh * gw, 128), 0.5, dtype=network_dtype),
        )

    # Wrap network so set_enabled call interleaves with forward events
    original_set_enabled = network.set_enabled

    def _recording_set_enabled(enabled: bool) -> None:
        events.append(f"set_enabled({enabled})")
        original_set_enabled(enabled)

    network.set_enabled = _recording_set_enabled  # type: ignore[method-assign]

    _install_stubs(monkeypatch, _flow)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5)
    latents, batch = _make_batch()

    trainer.process_batch(args, _make_accel(), None, network, batch, latents, None, None, torch.float32, torch.float32, None, 0)

    assert events == ["set_enabled(False)", "forward", "set_enabled(True)", "forward"], (
        f"Unexpected event order: {events}. The teacher must run with LoRA disabled and the student "
        "with LoRA enabled — any other interleaving silently breaks either the prior target (LoRA "
        "leaks into prior) or the student gradient (no LoRA to learn)."
    )


# ---------------------------------------------------------------------------------------------------
# Test 5 — Autocast bypass propagates to both teacher and student
# ---------------------------------------------------------------------------------------------------


def test_autocast_bypass_propagates_to_teacher_and_student(monkeypatch):
    """Both forwards must run with autocast disabled. If the teacher runs under autocast while the
    student doesn't (or vice versa), they evaluate different effective models — the same parity
    failure the v1-v3 sample/train split hit (commit 55e4d79). The `_run_i4_flow_forward` private
    helper makes drift impossible by routing both call sites through one wrapping.

    Here we install a torch.autocast(cpu, bf16) context that wraps process_batch itself; if the
    helper's disable_accelerate_forward_autocast + nested autocast(enabled=False) work, both
    captured forwards see autocast_enabled=False."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    flow, calls = _make_flow_target()
    _install_stubs(monkeypatch, flow)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5)
    latents, batch = _make_batch()
    network = _RecordingNetwork()

    # Wrap process_batch in a torch.autocast context — simulates accelerator's outer wrapping
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        trainer.process_batch(args, _make_accel(), None, network, batch, latents, None, None, torch.float32, torch.float32, None, 0)

    assert len(calls) == 2, "test precondition: expected 2 forwards"
    for i, call in enumerate(calls):
        assert call["autocast_enabled"] is False, (
            f"forward #{i} ran with autocast enabled — _run_i4_flow_forward isn't propagating the "
            f"disable to the {'teacher' if i == 0 else 'student'} path. This is the v3 sample-time "
            "parity failure mode applied to training-time prior preservation."
        )


# ---------------------------------------------------------------------------------------------------
# Test 6 — Prior loss is reduced via grid layout, not token layout
# ---------------------------------------------------------------------------------------------------


def test_prior_loss_uses_grid_layout(monkeypatch):
    """The shared apply_masked_loss_with_prior with layout='video' expects (B, C, F, gh, gw)
    tensors and a (B, 1, F, gh, gw) mask. Passing token-shape (B, L, C) instead would broadcast
    the mask across the wrong dimension and silently weight the wrong pixels. Ideogram's process_batch
    converts model_pred + target + prior_pred from token shape to grid shape via dit_tokens_to_grid
    before calling the reducer; this test asserts the conversion fires on the prior path too."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer
    import musubi_tuner.ideogram4_train_network as itn

    captured = {}

    def _spy_apply(loss_unreduced, mask_weights, *, prior_loss_unreduced=None, **kwargs):
        captured["target_shape"] = tuple(loss_unreduced.shape)
        captured["prior_shape"] = tuple(prior_loss_unreduced.shape) if prior_loss_unreduced is not None else None
        return torch.tensor(0.0)

    flow, _ = _make_flow_target()
    _install_stubs(monkeypatch, flow)
    monkeypatch.setattr(itn, "apply_masked_loss_with_prior", _spy_apply)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5)
    latents, batch = _make_batch(gh=4, gw=6)
    network = _RecordingNetwork()

    trainer.process_batch(args, _make_accel(), None, network, batch, latents, None, None, torch.float32, torch.float32, None, 0)

    # Both target and prior MUST be grid-shaped (B, C, gh, gw) — dit_tokens_to_grid produces 4D.
    # If they were token-shaped, the shape would be (B, L, C) = (1, 24, 128) instead of (1, 128, 4, 6).
    assert captured["target_shape"] == (1, 128, 4, 6), f"target shape wrong: {captured['target_shape']}"
    assert captured["prior_shape"] == (1, 128, 4, 6), f"prior shape wrong (token-shape leak?): {captured['prior_shape']}"


# ---------------------------------------------------------------------------------------------------
# Test 7 — The timestep threshold uses REMAPPED t, not Ideogram cleanness
# ---------------------------------------------------------------------------------------------------


def test_timestep_threshold_uses_remapped_t(monkeypatch):
    """User args have FLUX.2 semantics: --prior_preservation_timestep_threshold=300 means "fire prior
    at noise level > 300". For Ideogram (t=cleanness), this MUST map through (1-t)*1000.

    - ideogram_t=0.9 (low noise) -> traditional=100 -> below 300 -> teacher skipped.
    - ideogram_t=0.1 (high noise) -> traditional=900 -> above 300 -> teacher runs.

    If the remap is missing or inverted, the threshold either gates wrong half of timesteps (silent
    semantic error — the most expensive bug class for the user to debug)."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    # ideogram_t=0.9 -> traditional=100, below threshold 300 -> teacher should NOT fire
    flow_low, calls_low = _make_flow_target()
    _install_stubs(monkeypatch, flow_low)
    monkeypatch.setattr(
        "musubi_tuner.ideogram4_train_network.get_schedule_for_resolution",
        lambda reso, *, known_mean, std: lambda u: torch.full_like(u, 0.9),
    )

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5, prior_preservation_timestep_threshold=300.0)
    latents, batch = _make_batch()
    trainer.process_batch(
        args, _make_accel(), None, _RecordingNetwork(), batch, latents, None, None, torch.float32, torch.float32, None, 0
    )
    assert len(calls_low) == 1, (
        f"With ideogram_t=0.9 (remapped to 100) and threshold=300, teacher should have been "
        f"gated out. Got {len(calls_low)} forwards. If 2, the threshold is being applied against "
        "cleanness directly — the remap is missing."
    )

    # ideogram_t=0.1 -> traditional=900, above threshold 300 -> teacher SHOULD fire
    flow_hi, calls_hi = _make_flow_target()
    _install_stubs(monkeypatch, flow_hi)
    monkeypatch.setattr(
        "musubi_tuner.ideogram4_train_network.get_schedule_for_resolution",
        lambda reso, *, known_mean, std: lambda u: torch.full_like(u, 0.1),
    )
    trainer.process_batch(
        args, _make_accel(), None, _RecordingNetwork(), batch, latents, None, None, torch.float32, torch.float32, None, 0
    )
    assert len(calls_hi) == 2, (
        f"With ideogram_t=0.1 (remapped to 900) and threshold=300, teacher should have fired. "
        f"Got {len(calls_hi)} forwards. If 1, the threshold is being applied against cleanness "
        "directly (which is < threshold) — the remap is missing."
    )


# ---------------------------------------------------------------------------------------------------
# Test 8 — Loss includes prior contribution
# ---------------------------------------------------------------------------------------------------


def test_loss_routes_prior_loss_into_reducer(monkeypatch):
    """End-to-end shape: model_pred (zero), target (0.5), prior_pred (different value). Make the
    target loss numerically distinguishable from the prior loss; assert apply_masked_loss_with_prior
    receives a non-None prior_loss_unreduced. This is the contract that proves prior_preservation_weight
    actually influences the scalar loss returned to the optimizer."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer
    import musubi_tuner.ideogram4_train_network as itn

    received_prior = {"value": None}

    def _spy_apply(loss_unreduced, mask_weights, *, prior_loss_unreduced=None, **kwargs):
        received_prior["value"] = prior_loss_unreduced
        # Return a sentinel scalar that proves the spy fired, not None
        return torch.tensor(1.0)

    # Teacher emits prediction that's different from student's (so the prior_loss is non-zero).
    call_count = {"n": 0}

    def _flow(transformer, latents, text_features, noise, timesteps, *, network_dtype, device):
        gh, gw = int(latents.shape[2]), int(latents.shape[3])
        # Teacher is call #0 (returns 0.3), student is call #1 (returns 0.0)
        teacher_pred = 0.3 if call_count["n"] == 0 else 0.0
        call_count["n"] += 1
        return (
            torch.full((1, gh * gw, 128), teacher_pred, dtype=network_dtype),
            torch.full((1, gh * gw, 128), 0.5, dtype=network_dtype),
        )

    _install_stubs(monkeypatch, _flow)
    monkeypatch.setattr(itn, "apply_masked_loss_with_prior", _spy_apply)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5)
    latents, batch = _make_batch()
    network = _RecordingNetwork()

    trainer.process_batch(args, _make_accel(), None, network, batch, latents, None, None, torch.float32, torch.float32, None, 0)

    assert received_prior["value"] is not None, (
        "prior_loss_unreduced was None despite prior_preservation_weight > 0. The teacher forward "
        "ran but its result wasn't routed into apply_masked_loss_with_prior — the prior preservation "
        "weight is silently inert in this code path."
    )
    assert tuple(received_prior["value"].shape) == (1, 128, 4, 6), (
        f"prior loss shape wrong: {received_prior['value'].shape}; expected grid (1, 128, 4, 6)."
    )


# ---------------------------------------------------------------------------------------------------
# Test 9 — EMA teacher mode is rejected loudly
# ---------------------------------------------------------------------------------------------------


def test_ema_teacher_mode_rejected_loudly():
    """The global parser accepts --prior_teacher_mode=ema (registered in modules/mask_loss.py), but
    Ideogram 4 Phase 1 only supports base mode. handle_model_specific_args must reject EMA mode with
    an actionable error rather than silently fall through to base — the latter would be the exact
    "looks configured, does the wrong thing" failure class the v1-v3 investigation killed.

    The follow-up PR that adds EMA must also remove this rejection."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(
        gradient_checkpointing_cpu_offload=False,
        blocks_to_swap=0,
        compile=False,
        use_mask_loss=True,
        prior_preservation_weight=0.5,
        prior_teacher_mode="ema",
        sample_prompts=None,
    )

    with pytest.raises(ValueError, match=r"EMA teacher mode is deferred"):
        trainer.handle_model_specific_args(args)


def test_base_teacher_mode_accepted_with_prior_weight():
    """The complement of the EMA rejection: prior_teacher_mode='base' (or omitted, defaulting to
    'base') with prior_preservation_weight>0 and use_mask_loss=True must be ACCEPTED. The previous
    Slice-1 guard rejected this combination unconditionally; Slice 2 must allow it.

    handle_model_specific_args also does a sample_prompts requirement check and a mixed_precision
    default-fill at the end; we supply just enough args for it to reach the early-return paths and
    not blow up. The point is to test that this method no longer rejects the use_mask_loss +
    prior_weight>0 + teacher_mode='base' combination."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(
        gradient_checkpointing_cpu_offload=False,
        blocks_to_swap=0,
        compile=False,
        use_mask_loss=True,
        prior_preservation_weight=0.5,
        prior_teacher_mode="base",
        sample_prompts=None,
        mixed_precision="bf16",  # explicit so the fp32-default fallback path is bypassed
    )

    # Must not raise — proves the v1 fail-fast was removed
    trainer.handle_model_specific_args(args)


# ---------------------------------------------------------------------------------------------------
# Test 10 — prior_teacher_eval honored (reviewer finding #1)
# ---------------------------------------------------------------------------------------------------


def test_prior_teacher_eval_toggles_model_mode(monkeypatch):
    """The shared masked path honors --prior_teacher_eval by calling transformer.eval() around the
    teacher forward, restoring train() in finally (mask_loss_process_batch.py:255-258). Ideogram 4
    must implement the same toggle. The flag is a harmless no-op for Ideogram's DiT today (no
    dropout/BN), but silently no-oping a CLI-accepted knob is the "looks configured, does the wrong
    thing" failure class commit 55e4d79 was killing.

    We assert transformer.training is False during the teacher forward and True after, given the
    transformer entered process_batch in train() mode."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    class _ModeTrackingTransformer(torch.nn.Module):
        """Records transformer.training state at every flow call. We pass this object as the
        `transformer` arg to process_batch and let the flow stub introspect it."""

        def __init__(self) -> None:
            super().__init__()
            self.observed_modes: list[bool] = []

        def forward(self, *a, **kw):
            return torch.zeros(1)  # never called; the flow stub captures `transformer`

    trf = _ModeTrackingTransformer()
    trf.train()  # the trainer's outer loop sets this; mimic it

    def _flow_capturing_mode(transformer, latents, text_features, noise, timesteps, *, network_dtype, device):
        trf.observed_modes.append(transformer.training)
        gh, gw = int(latents.shape[2]), int(latents.shape[3])
        return (
            torch.zeros(1, gh * gw, 128, dtype=network_dtype),
            torch.full((1, gh * gw, 128), 0.5, dtype=network_dtype),
        )

    _install_stubs(monkeypatch, _flow_capturing_mode)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5, prior_teacher_eval=True)
    latents, batch = _make_batch()
    network = _RecordingNetwork()

    trainer.process_batch(args, _make_accel(), trf, network, batch, latents, None, None, torch.float32, torch.float32, None, 0)

    # Order is teacher then student. The student call should see transformer.training=True (restored
    # after the teacher's eval() toggle); the teacher should see training=False.
    assert trf.observed_modes == [False, True], (
        f"Expected [teacher in eval, student in train], got {trf.observed_modes}. Either the eval "
        "toggle didn't fire (knob silently ignored) or the restore in finally didn't run (subsequent "
        "training steps would silently train in eval mode after the first sample-with-eval)."
    )
    # After process_batch returns, transformer.training must be True (the trainer's outer loop expects this).
    assert trf.training is True, (
        "transformer.train() restoration missing after process_batch — every step after the first prior teacher would run in eval mode"
    )


def test_prior_teacher_eval_restores_train_mode_on_exception(monkeypatch):
    """Failure-path correctness: if the teacher forward raises, transformer.train() MUST still be
    restored. Without the finally block, an exception in the teacher would strand the transformer in
    eval mode for the rest of training — every subsequent step would silently train with eval-mode
    semantics. The finally guard is the only thing standing between a transient teacher failure
    (e.g., transient OOM) and a corrupted training run."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    class _ModeTrackingTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, *a, **kw):
            return torch.zeros(1)

    trf = _ModeTrackingTransformer()
    trf.train()

    def _flow_that_raises(transformer, latents, text_features, noise, timesteps, *, network_dtype, device):
        # Detect we're in the teacher path (LoRA disabled) and raise. The teacher path runs first
        # so this fires on the first call.
        raise RuntimeError("simulated teacher OOM")

    _install_stubs(monkeypatch, _flow_that_raises)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5, prior_teacher_eval=True)
    latents, batch = _make_batch()

    with pytest.raises(RuntimeError, match="simulated teacher OOM"):
        trainer.process_batch(
            args, _make_accel(), trf, _RecordingNetwork(), batch, latents, None, None, torch.float32, torch.float32, None, 0
        )

    # Without the finally, transformer.training would be False here and every subsequent step would
    # silently run in eval mode.
    assert trf.training is True, (
        "transformer.train() not restored after teacher raised — subsequent training steps would run in eval mode"
    )


# ---------------------------------------------------------------------------------------------------
# Test 11 — Teacher forward skipped when no prior region exists (reviewer finding #2)
# ---------------------------------------------------------------------------------------------------


def test_teacher_skipped_when_all_ones_mask_continuous_mode(monkeypatch):
    """When mask_weights is all-ones (full mask, no background), prior_mask = 1 - mask_processed
    = 0 everywhere — the reducer would weight prior_loss by zero, contributing nothing to the
    scalar loss. Running the teacher forward in this case is pure waste and pollutes telemetry
    (prior/teacher_ran=1 while masked_loss/prior=0 looks like "prior is configured but
    ineffective" when actually it's "running pointlessly"). Mirrors the shared path's optimization
    at mask_loss_process_batch.py:241-251."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    flow, calls = _make_flow_target()
    _install_stubs(monkeypatch, flow)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5)
    latents, batch = _make_batch()
    # OVERRIDE the default partial mask with an actual all-ones mask — this test's whole point is
    # to exercise the all-ones skip path; the helper default is partial precisely so OTHER tests
    # observe the teacher firing on the common-case workload.
    batch["mask_weights"] = torch.ones_like(batch["mask_weights"])
    assert float(batch["mask_weights"].min()) >= (1.0 - 1e-6), "test precondition: mask must be all-ones"

    trainer.process_batch(
        args, _make_accel(), None, _RecordingNetwork(), batch, latents, None, None, torch.float32, torch.float32, None, 0
    )

    assert len(calls) == 1, (
        f"All-ones mask should skip the teacher (no prior region exists). Got {len(calls)} forwards. "
        "This is a wasted full DiT forward per step and clutters telemetry; the shared path mirrors "
        "this optimization at mask_loss_process_batch.py:241-251."
    )


def test_teacher_skipped_when_threshold_mode_no_background(monkeypatch):
    """Threshold-mode variant: with --prior_mask_threshold=0.1 and a mask where every pixel >= 0.1,
    the prior region (raw_mask < threshold) is empty — teacher should skip. This catches the case
    where a user enables threshold mode on a small persona dataset whose masks happen to be high
    everywhere (e.g., portraits where the face mask covers a large fraction)."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    flow, calls = _make_flow_target()
    _install_stubs(monkeypatch, flow)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5, prior_mask_threshold=0.1)
    latents, batch = _make_batch()
    # Mask with min=0.5: every pixel >= threshold=0.1, so prior region is empty
    batch["mask_weights"] = torch.full_like(batch["mask_weights"], 0.5)

    trainer.process_batch(
        args, _make_accel(), None, _RecordingNetwork(), batch, latents, None, None, torch.float32, torch.float32, None, 0
    )

    assert len(calls) == 1, (
        f"With prior_mask_threshold=0.1 and mask.min()=0.5 (every pixel >= threshold), no background "
        f"region exists. Got {len(calls)} forwards instead of 1 (student only). Wasted teacher pass."
    )


def test_teacher_still_runs_when_partial_mask_continuous_mode(monkeypatch):
    """Inverse of the all-ones test: with a mask that has background regions (min < 1.0), the
    teacher MUST run. Catches the case where the skip optimization is too aggressive and
    inadvertently disables prior preservation on the normal use case."""
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    flow, calls = _make_flow_target()
    _install_stubs(monkeypatch, flow)

    trainer = Ideogram4NetworkTrainer()
    args = _base_args(prior_preservation_weight=0.5)
    latents, batch = _make_batch()
    # Mask with background: half ones, half zeros
    mask = batch["mask_weights"].clone()
    mask[..., :, : mask.shape[-1] // 2] = 0.0
    batch["mask_weights"] = mask
    assert float(batch["mask_weights"].min()) < (1.0 - 1e-6), "test precondition: mask must have background"

    trainer.process_batch(
        args, _make_accel(), None, _RecordingNetwork(), batch, latents, None, None, torch.float32, torch.float32, None, 0
    )

    assert len(calls) == 2, (
        f"With a partial mask (background present), teacher must run. Got {len(calls)} forwards. "
        "If 1, the skip optimization is too aggressive — it's disabling prior preservation on the "
        "normal-case workload, defeating the whole point of this PR."
    )


# ---------------------------------------------------------------------------------------------------
# Test 12 — _run_i4_flow_forward defeats a real Accelerator-prepared forward wrapper (reviewer finding #3)
# ---------------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Accelerator's autocast wrapper is installed only on CUDA-aware setups with native_amp.",
)
def test_run_i4_flow_forward_defeats_real_accelerator_wrapper():
    """Direct test of _run_i4_flow_forward against a real Accelerator(mixed_precision="bf16").prepare()
    canary. The earlier autocast bypass test wrapped process_batch in a torch.autocast context manager,
    which is NOT the same mechanism Accelerator uses (Accelerator replaces model.forward with an
    autocast-wrapping callable; torch.autocast nesting CAN'T defeat that — proven by
    tests/test_accelerate_autocast_helper.py).

    The new private helper _run_i4_flow_forward inside process_batch is the integration point where
    that fix must propagate. Without this test, a future refactor that drops the
    disable_accelerate_forward_autocast wrap from _run_i4_flow_forward would pass the existing
    integration tests (which use fake accelerators) while silently re-introducing the v1-v3 broken
    sample/train parity. This test fails loudly in that scenario.

    The actual forward we run is the Ideogram-specific ideogram4_flow_matching_target with a canary
    transformer that records the autocast state observed inside its forward; we pass it as the
    'conditional_model' arg, so the autocast-bypass path either reaches the canary's forward or it
    doesn't."""
    from accelerate import Accelerator

    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    class _AutocastCanary(torch.nn.Module):
        """Records autocast state observed during its forward. Used as the conditional model in
        ideogram4_flow_matching_target — the latter passes (x, t, llm_features, position_ids,
        segment_ids, indicator) and expects a tensor back at image positions."""

        def __init__(self) -> None:
            super().__init__()
            self.observed_autocast: list[bool] = []
            # Trivial linear so accelerator.prepare has something to attach to
            self.linear = torch.nn.Linear(128, 128).to(torch.bfloat16)

        def forward(self, *, x, t, llm_features, position_ids, segment_ids, indicator):
            self.observed_autocast.append(torch.is_autocast_enabled("cuda"))
            # Return a tensor shaped like the joint sequence — ideogram4_flow_matching_target's
            # extract_image_tokens then takes image positions. Easier to just return x unchanged
            # (after the linear, to exercise some compute under whatever dtype regime is active).
            return self.linear(x.to(torch.bfloat16))

    accelerator = Accelerator(mixed_precision="bf16")
    canary = _AutocastCanary().cuda()
    prepared_canary = accelerator.prepare(canary)

    trainer = Ideogram4NetworkTrainer()
    # Build minimal valid inputs for ideogram4_flow_matching_target — this is the real function,
    # not a stub. We're proving _run_i4_flow_forward's autocast bypass plumbs through to a real
    # forward call on a real Accelerator-prepared model.
    gh, gw = 4, 6
    latents = torch.zeros(1, 128, gh, gw, device="cuda", dtype=torch.bfloat16)
    text_features = [torch.zeros(1, 53248, device="cuda", dtype=torch.bfloat16)]
    noise = torch.ones_like(latents)
    timesteps = torch.tensor([0.5], device="cuda")

    # Call _run_i4_flow_forward directly under an outer-set autocast context (simulating accelerator's
    # outer training loop). The helper should defeat BOTH the outer torch.autocast AND the inner
    # accelerate forward wrapper, so the canary observes autocast_enabled=False.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        trainer._run_i4_flow_forward(
            accelerator,
            prepared_canary,
            latents,
            text_features,
            noise,
            timesteps,
            network_dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )

    assert canary.observed_autocast == [False], (
        f"Expected canary forward to observe autocast disabled (observed={canary.observed_autocast}). "
        "If True, _run_i4_flow_forward's disable_accelerate_forward_autocast call is missing or not "
        "reaching the prepared forward — this is the v3 sample-time parity failure mode applied to "
        "training-time prior preservation, EXACTLY the bug class commit 55e4d79 fixed."
    )
