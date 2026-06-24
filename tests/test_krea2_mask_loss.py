"""Krea 2 mask-weighted loss: cache payload + process_batch wiring (Phase 2).

The discriminating checks (per review): not just that ``--use_mask_loss`` parses, but that the
mask actually changes the numerical loss and weights the correct spatial region, and that prior
preservation runs the teacher forward and contributes a prior term. Mask math itself is covered
by tests/test_mask_loss.py; this file pins K2's *wiring* (cache key + process_batch dispatch).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dataset.cache_io import save_latent_cache_krea2
from musubi_tuner.hv_train_network import DiTOutput


def _item(tmp_path, key="item"):
    return SimpleNamespace(
        latent_cache_path=str(tmp_path / f"{key}_krea2.safetensors"),
        original_size=(64, 64),
        frame_count=None,
        item_key=key,
    )


# ----------------------------------------------------------------------------- cache payload
def test_save_latent_cache_writes_mask_weights(tmp_path):
    item = _item(tmp_path)
    latent = torch.randn(16, 1, 8, 8).to(torch.bfloat16)  # (C, F=1, H, W)
    save_latent_cache_krea2(item, latent, mask_weights=torch.rand(1, 1, 8, 8))
    sd = load_file(item.latent_cache_path)
    assert "mask_weights_1x8x8_float16" in sd, sorted(sd)
    assert tuple(sd["mask_weights_1x8x8_float16"].shape) == (1, 1, 8, 8)
    assert sd["mask_weights_1x8x8_float16"].dtype == torch.float16


def test_save_latent_cache_no_mask_omits_key(tmp_path):
    item = _item(tmp_path, key="nomask")
    save_latent_cache_krea2(item, torch.randn(16, 1, 8, 8).to(torch.bfloat16))
    sd = load_file(item.latent_cache_path)
    assert not any(k.startswith("mask_weights_") for k in sd), sorted(sd)


def test_save_latent_cache_rejects_bad_mask_shape(tmp_path):
    item = _item(tmp_path, key="bad")
    with pytest.raises(ValueError, match=r"\(1, 1, 8, 8\)"):
        save_latent_cache_krea2(item, torch.zeros(16, 1, 8, 8).to(torch.bfloat16), mask_weights=torch.zeros(8, 8))


def test_cache_downsamples_mask_to_latent_grid(tmp_path):
    """The mask must downsample to the /8 latent grid (8x8), not the raw 64x64 image."""
    import musubi_tuner.krea2_cache_latents as cl

    class _VAE:
        device = torch.device("cpu")
        dtype = torch.float32

        def encode_pixels_to_latents(self, x):  # x: (B, C, 1, H, W) -> latent at /8
            b, _, _, h, w = x.shape
            return torch.randn(b, 16, 1, h // 8, w // 8)

    item = _item(tmp_path, key="ds")
    item.content = np.zeros((64, 64, 3), dtype=np.uint8)
    item.mask_content = (np.random.rand(64, 64) * 255).astype(np.uint8)
    item.cache_mask_gamma, item.cache_mask_min_weight = 1.0, 0.0

    cl.encode_and_save_batch(_VAE(), [item])
    sd = load_file(item.latent_cache_path)
    assert "mask_weights_1x8x8_float16" in sd, "mask must downsample to the 8x8 latent grid, not 64x64"
    assert tuple(sd["mask_weights_1x8x8_float16"].shape) == (1, 1, 8, 8)


# ------------------------------------------------------------------- process_batch numerical gate
def _mask_args(**over):
    base = dict(
        use_mask_loss=True,
        mask_gamma=1.0,
        mask_min_weight=0.0,
        mask_blur_kernel_size=0,
        mask_blur_radius=0.0,
        mask_area_scale_beta=0.0,
        normalize_per_sample=False,
        prior_preservation_weight=0.0,
        prior_mask_threshold=None,
        prior_preservation_timestep_threshold=None,
        prior_decay_schedule="constant",
        prior_decay_timestep_start=300.0,
        prior_decay_warmup_ratio=0.0,
        prior_teacher_mode="base",
        prior_teacher_ema_decay=0.999,
        prior_teacher_eval=False,
        is_layered=False,
        loss_type="mse",
        loss_delta=1.0,
        weighting_scheme="none",
        max_train_steps=1000,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _accel(stats=True):
    return SimpleNamespace(
        device=torch.device("cpu"),
        trackers=[object()] if stats else [],
        unwrap_model=lambda m: m,
    )


def _stub_trainer(*, teacher_pred=None):
    """Krea2NetworkTrainer with the model/timestep machinery stubbed to controlled tensors.

    call_dit returns ``teacher_pred`` while grad is disabled (the no-grad teacher forward) and
    the student pred otherwise, so the prior term is computable without a real DiT.
    """
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    class _Stub(Krea2NetworkTrainer):
        def __init__(self):
            super().__init__()
            self.blocks_to_swap = 0
            self.call_dit_count = 0
            self._student_pred = None
            self._target = None

        def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dit_dtype):
            return latents, timesteps

        def call_dit(
            self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kw
        ):
            self.call_dit_count += 1
            pred = teacher_pred if (teacher_pred is not None and not torch.is_grad_enabled()) else self._student_pred
            return DiTOutput(pred=pred, target=self._target)

        def restore_block_swap_after_no_grad_forward(self, accelerator, transformer):
            pass

    return _Stub()


def _run(trainer, args, *, pred, target, mask, network=None):
    trainer._student_pred = pred
    trainer._target = target
    batch = {"mask_weights": mask, "timesteps": torch.tensor([500.0])}
    latents = torch.zeros_like(target)
    noise = torch.zeros_like(target)
    return trainer.process_batch(
        args, _accel(), None, network, batch, latents, noise, None, torch.float32, torch.float32, None, 500
    )


def test_masked_process_batch_tracks_region_and_differs_from_vanilla():
    """The gate: masking concentrates loss on the masked region and differs from plain-mean MSE."""
    target = torch.zeros(1, 4, 1, 4, 4)
    mask = torch.zeros(1, 1, 1, 4, 4)
    mask[..., :2, :2] = 1.0  # top-left quadrant is the trained region

    pred_in = torch.zeros(1, 4, 1, 4, 4)
    pred_in[..., :2, :2] = 3.0  # error INSIDE the mask
    pred_out = torch.zeros(1, 4, 1, 4, 4)
    pred_out[..., 2:, 2:] = 3.0  # error OUTSIDE the mask

    masked_in, metrics = _run(_stub_trainer(), _mask_args(), pred=pred_in, target=target, mask=mask)
    masked_out, _ = _run(_stub_trainer(), _mask_args(), pred=pred_out, target=target, mask=mask)
    vanilla_in, _ = _run(_stub_trainer(), _mask_args(use_mask_loss=False), pred=pred_in, target=target, mask=mask)

    # (a) mask weights the right region: error inside drives loss, error outside is ignored.
    assert float(masked_in) > 1.0, float(masked_in)
    assert float(masked_out) < 1e-4, float(masked_out)
    # (b) masking is not a no-op vs plain mean: concentrating on the masked region raises the loss.
    assert float(masked_in) > float(vanilla_in) + 1e-3, (float(masked_in), float(vanilla_in))
    # (c) telemetry is emitted so long runs can confirm the mask is active.
    assert any(k.startswith("masked_loss/") for k in metrics), sorted(metrics)


def test_prior_preservation_runs_teacher_and_contributes():
    """prior_preservation_weight>0 must run the teacher forward (2 call_dit) and add a prior term."""
    target = torch.zeros(1, 4, 1, 4, 4)
    mask = torch.zeros(1, 1, 1, 4, 4)
    mask[..., :2, :2] = 1.0  # partial mask -> an unmasked region where the prior applies
    student_pred = torch.full((1, 4, 1, 4, 4), 2.0)
    teacher_pred = torch.zeros(1, 4, 1, 4, 4)  # teacher disagrees with student -> nonzero prior term

    class _Net:
        def set_enabled(self, flag):
            pass

    # With prior: teacher runs (2 forwards), prior telemetry present.
    trainer = _stub_trainer(teacher_pred=teacher_pred)
    loss_prior, metrics = _run(
        trainer, _mask_args(prior_preservation_weight=1.0), pred=student_pred, target=target, mask=mask, network=_Net()
    )
    assert trainer.call_dit_count == 2, f"teacher + student forwards expected, got {trainer.call_dit_count}"
    assert metrics.get("prior/teacher_ran") == 1.0, metrics
    assert metrics.get("prior/teacher_mode_ema_used") == 0.0  # base mode, not EMA

    # Without prior: teacher does not run (1 forward), and the loss differs (no prior term).
    trainer_np = _stub_trainer(teacher_pred=teacher_pred)
    loss_noprior, _ = _run(trainer_np, _mask_args(prior_preservation_weight=0.0), pred=student_pred, target=target, mask=mask)
    assert trainer_np.call_dit_count == 1, trainer_np.call_dit_count
    assert abs(float(loss_prior) - float(loss_noprior)) > 1e-4, (float(loss_prior), float(loss_noprior))
