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
    # pred == target: cosine = 1, RMS equal, flipped_pred = E[(2t)^2] = 4 * zero_pred.
    _, target, timesteps = _make_inputs(seed=1)
    out = compute_flow_loss_diagnostics(target.clone(), target, timesteps)
    assert out["loss/pred_target_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert out["loss/pred_rms"] == pytest.approx(out["loss/target_rms"], rel=1e-6)
    assert out["loss/flipped_pred"] == pytest.approx(4.0 * out["loss/zero_pred"], rel=1e-5)


def test_inverted_prediction_is_visible():
    # pred == -target (the inverted-target failure class): cosine = -1, flipped_pred = 0.
    _, target, timesteps = _make_inputs(seed=2)
    out = compute_flow_loss_diagnostics(-target, target, timesteps)
    assert out["loss/pred_target_cosine"] == pytest.approx(-1.0, abs=1e-5)
    assert out["loss/flipped_pred"] == pytest.approx(0.0, abs=1e-6)


def test_zero_pred_baseline_is_target_power():
    # zero_pred = MSE(0, target) = mean(target^2); exact for a known constant tensor.
    target = torch.full((2, 8, 128), 3.0)
    pred = torch.zeros_like(target)
    timesteps = torch.tensor([0.25, 0.75])
    out = compute_flow_loss_diagnostics(pred, target, timesteps)
    assert out["loss/zero_pred"] == pytest.approx(9.0, rel=1e-6)
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
