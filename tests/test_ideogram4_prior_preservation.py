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
    `masked=True` the mask is all ones (active, non-degenerate). The shape MUST match the latent grid
    or apply_masked_loss_with_prior broadcasts incorrectly."""
    latents = torch.zeros(1, 128, gh, gw)
    mask = torch.ones(1, 1, 1, gh, gw) if masked else None
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
