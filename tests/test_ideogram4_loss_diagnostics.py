# --log_loss_stats raw flow diagnostics (ported from sdbds qinglong 99ad6c5).
#
# compute_flow_loss_diagnostics runs on token-space (pred, target) BEFORE Huber / mask weighting /
# area-scale beta, so its values are comparable across runs regardless of loss configuration —
# the cross-run baseline the DLAY v8-v12 likeness investigation lacked. These tests pin the math
# (baselines, cosine, RMS), the t-convention remap of the timestep tags, the float (non-tensor)
# return contract, and the parser wiring.

import math

import pytest
import torch

from musubi_tuner.ideogram4.training import compute_flow_loss_diagnostics

EXPECTED_KEYS = {
    "loss/raw_mse",
    "loss/raw_mse_over_zero_pred",
    "loss/zero_pred",
    "loss/flipped_pred",
    "loss/pred_rms",
    "loss/target_rms",
    "loss/pred_target_cosine",
    "timestep/traditional_t_mean",
    "timestep/traditional_t_min",
    "timestep/traditional_t_max",
}


def _make_inputs(seed=0, b=2, n=12, c=128):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(b, n, c, generator=g)
    target = torch.randn(b, n, c, generator=g)
    timesteps = torch.rand(b, generator=g)
    return pred, target, timesteps


def test_key_set_and_float_contract():
    pred, target, timesteps = _make_inputs()
    out = compute_flow_loss_diagnostics(pred, target, timesteps)
    assert set(out.keys()) == EXPECTED_KEYS
    # Plain python floats (TensorBoard-ready), all finite.
    for k, v in out.items():
        assert isinstance(v, float), f"{k} is {type(v)}, expected float"
        assert math.isfinite(v), f"{k} is not finite: {v}"


def test_perfect_prediction_baselines():
    # pred == target: raw_mse = 0 (and ratio 0), cosine = 1, RMS equal, flipped_pred = E[(2t)^2] = 4 * zero_pred.
    _, target, timesteps = _make_inputs(seed=1)
    out = compute_flow_loss_diagnostics(target.clone(), target, timesteps)
    assert out["loss/raw_mse"] == pytest.approx(0.0, abs=1e-8)
    assert out["loss/raw_mse_over_zero_pred"] == pytest.approx(0.0, abs=1e-8)
    assert out["loss/pred_target_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert out["loss/pred_rms"] == pytest.approx(out["loss/target_rms"], rel=1e-6)
    assert out["loss/flipped_pred"] == pytest.approx(4.0 * out["loss/zero_pred"], rel=1e-5)


def test_inverted_prediction_is_visible():
    # pred == -target (the inverted-target failure class): cosine = -1, flipped_pred = 0,
    # raw_mse = 4 * zero_pred (ratio = 4 — far above the predict-zero baseline of 1.0).
    _, target, timesteps = _make_inputs(seed=2)
    out = compute_flow_loss_diagnostics(-target, target, timesteps)
    assert out["loss/pred_target_cosine"] == pytest.approx(-1.0, abs=1e-5)
    assert out["loss/flipped_pred"] == pytest.approx(0.0, abs=1e-6)
    assert out["loss/raw_mse_over_zero_pred"] == pytest.approx(4.0, rel=1e-5)


def test_zero_pred_baseline_is_target_power():
    # zero_pred = MSE(0, target) = mean(target^2); exact for a known constant tensor. A zero
    # prediction also makes raw_mse == zero_pred by definition, so the ratio reads exactly 1.0.
    target = torch.full((2, 8, 128), 3.0)
    pred = torch.zeros_like(target)
    timesteps = torch.tensor([0.25, 0.75])
    out = compute_flow_loss_diagnostics(pred, target, timesteps)
    assert out["loss/zero_pred"] == pytest.approx(9.0, rel=1e-6)
    assert out["loss/raw_mse"] == pytest.approx(9.0, rel=1e-6)
    assert out["loss/raw_mse_over_zero_pred"] == pytest.approx(1.0, rel=1e-6)
    assert out["loss/pred_rms"] == pytest.approx(0.0, abs=1e-8)
    assert out["loss/target_rms"] == pytest.approx(3.0, rel=1e-6)
    # Zero pred -> zero numerator; the clamped denominator keeps this finite and exactly 0.
    assert out["loss/pred_target_cosine"] == pytest.approx(0.0, abs=1e-8)


def test_timestep_tags_use_traditional_noise_convention():
    # Ideogram t = cleanness in [0,1]; tags report (1 - t) * 1000 to match prior/traditional_t_*.
    pred, target, _ = _make_inputs(seed=3)
    timesteps = torch.tensor([0.0, 0.25, 1.0])  # cleanness: pure noise, mid, clean
    out = compute_flow_loss_diagnostics(pred[:1].expand(3, -1, -1), target[:1].expand(3, -1, -1), timesteps)
    assert out["timestep/traditional_t_max"] == pytest.approx(1000.0, abs=1e-4)  # t=0 cleanness = full noise
    assert out["timestep/traditional_t_min"] == pytest.approx(0.0, abs=1e-4)  # t=1 cleanness = clean
    assert out["timestep/traditional_t_mean"] == pytest.approx((1000.0 + 750.0 + 0.0) / 3.0, rel=1e-5)


def test_autograd_isolation():
    # Diagnostics must not require grad, retain graph, or disturb the caller's autograd state.
    pred, target, timesteps = _make_inputs(seed=4)
    pred_req = pred.clone().requires_grad_(True)
    loss = (pred_req - target).square().mean()
    out = compute_flow_loss_diagnostics(pred_req, target, timesteps)
    assert all(isinstance(v, float) for v in out.values())
    loss.backward()  # graph still intact after diagnostics
    assert pred_req.grad is not None


def test_bf16_inputs_computed_in_float32():
    # bf16 pred/target must not degrade the baselines (math runs on .float() copies).
    target = torch.full((1, 4, 128), 3.0, dtype=torch.bfloat16)
    out = compute_flow_loss_diagnostics(torch.zeros_like(target), target, torch.tensor([0.5]))
    assert out["loss/zero_pred"] == pytest.approx(9.0, rel=1e-2)


def test_parser_wiring_default_off():
    from musubi_tuner.ideogram4_train_network import ideogram4_setup_parser
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = ideogram4_setup_parser(setup_parser_common())
    base = ["--dataset_config", "x.toml", "--dit", "x.safetensors"]
    assert parser.parse_args(base).log_loss_stats is False
    assert parser.parse_args(base + ["--log_loss_stats"]).log_loss_stats is True


# --- process_batch call-site wiring guards ---------------------------------------------------------
#
# The helper math above can stay green while a refactor silently moves the computation after
# Huber/mask/beta or drops it from one return path. These tests pin the load-bearing call-site
# claims with a sentinel: diagnostics are computed once after the student forward and surface in
# BOTH return paths, and the helper is not called at all when the flag or trackers gate it off.
# Scaffolding mirrors test_ideogram4_masked_loss.py::test_process_batch_returns_masked_loss_telemetry.

from types import SimpleNamespace  # noqa: E402

SENTINEL = {"loss/sentinel": 123.0}


def _trainer_args(**over):
    args = SimpleNamespace(
        use_mask_loss=False,
        mask_gamma=1.0,
        mask_min_weight=0.0,
        mask_blur_kernel_size=0,
        mask_blur_radius=0.0,
        mask_area_scale_beta=0.0,
        normalize_per_sample=False,
        prior_preservation_weight=0.0,
        loss_type="mse",
        loss_delta=1.0,
        ideogram4_timestep_mu=None,
        ideogram4_timestep_std=1.0,
        log_loss_stats=True,
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def _run_process_batch(monkeypatch, args, *, trackers, with_mask, diag_calls):
    import musubi_tuner.ideogram4_train_network as itn
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    gh, gw = 4, 6
    latents = torch.zeros(1, 128, gh, gw)
    batch = {"i4_llm_features": [torch.zeros(1, 53248)]}
    if with_mask:
        batch["mask_weights"] = torch.ones(1, 1, 1, gh, gw)

    def _flow(transformer, latents, text_features, noise, timesteps, *, network_dtype, device):
        return torch.zeros(1, gh * gw, 128, dtype=network_dtype), torch.full((1, gh * gw, 128), 0.5, dtype=network_dtype)

    def _fake_diagnostics(model_pred, target, timesteps):
        diag_calls.append((model_pred.shape, target.shape))
        return dict(SENTINEL)

    monkeypatch.setattr(itn, "ideogram4_flow_matching_target", _flow)
    monkeypatch.setattr(itn, "get_schedule_for_resolution", lambda reso, *, known_mean, std: lambda u: u)
    monkeypatch.setattr(itn, "compute_flow_loss_diagnostics", _fake_diagnostics)

    accelerator_stub = SimpleNamespace(device=torch.device("cpu"), trackers=trackers, unwrap_model=lambda m: m)
    trainer = Ideogram4NetworkTrainer()
    return trainer.process_batch(
        args, accelerator_stub, None, None, batch, latents, None, None, torch.float32, torch.float32, None, 0
    )


def test_process_batch_unmasked_path_returns_diagnostics(monkeypatch):
    calls = []
    loss, metrics = _run_process_batch(
        monkeypatch, _trainer_args(use_mask_loss=False), trackers=[object()], with_mask=False, diag_calls=calls
    )
    assert torch.is_tensor(loss)
    assert len(calls) == 1, "diagnostics must be computed exactly once"
    assert metrics == SENTINEL, f"unmasked path must return the diagnostics dict, got {sorted(metrics)}"
    # Pinned placement: the helper receives TOKEN-space tensors (B, L, 128) from the student
    # forward, not grid-space (B, 128, gh, gw) — i.e. it runs BEFORE the masked grid reconstruction.
    assert calls[0][0][-1] == 128 and calls[0][1][-1] == 128


def test_process_batch_masked_path_merges_diagnostics(monkeypatch):
    calls = []
    loss, metrics = _run_process_batch(
        monkeypatch, _trainer_args(use_mask_loss=True), trackers=[object()], with_mask=True, diag_calls=calls
    )
    assert torch.is_tensor(loss)
    assert len(calls) == 1
    assert metrics["loss/sentinel"] == 123.0, "masked path must preserve the diagnostics"
    assert any(k.startswith("masked_loss/") for k in metrics), "masked telemetry must coexist with the diagnostics"


@pytest.mark.parametrize(
    "flag,trackers",
    [(False, [object()]), (True, []), (False, [])],
    ids=["flag-off", "no-trackers", "both-off"],
)
def test_process_batch_diagnostics_gated_off(monkeypatch, flag, trackers):
    calls = []
    loss, metrics = _run_process_batch(
        monkeypatch, _trainer_args(use_mask_loss=False, log_loss_stats=flag), trackers=trackers, with_mask=False, diag_calls=calls
    )
    assert torch.is_tensor(loss)
    assert calls == [], "helper must not be invoked when the flag or trackers gate it off"
    assert "loss/sentinel" not in metrics
