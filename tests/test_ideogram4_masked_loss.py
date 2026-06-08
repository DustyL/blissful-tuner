"""Ideogram-4 weighted masked loss (Slice 1, target-only).

The centerpiece is the NON-SQUARE + ASYMMETRIC alignment gate: a square/symmetric test would pass even if
dit_tokens_to_grid transposed gh/gw, which is the silent "trains fine, weights the wrong region" failure. So the
gate uses gh != gw and an off-center masked region and asserts the reduced loss tracks where the error lands.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dataset.cache_io import save_latent_cache_ideogram4
from musubi_tuner.ideogram4.ideogram4_utils import (
    dit_tokens_to_grid,
    grid_to_dit_tokens,
    preflight_ideogram4_latent_cache,
)
from musubi_tuner.modules.loss_utils import compute_unreduced_target_loss
from musubi_tuner.modules.mask_loss import apply_masked_loss_with_prior, require_mask_weights_if_enabled


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
    )
    base.update(over)
    return SimpleNamespace(**base)


def _item(tmp_path, key="item"):
    return SimpleNamespace(
        latent_cache_path=str(tmp_path / f"{key}_i4.safetensors"), original_size=(16, 16), frame_count=None, item_key=key
    )


def test_dit_tokens_to_grid_is_exact_inverse_nonsquare():
    grid = torch.randn(2, 128, 4, 6)  # gh != gw
    tokens = grid_to_dit_tokens(grid)
    assert tokens.shape == (2, 24, 128)
    assert torch.equal(dit_tokens_to_grid(tokens, 4, 6), grid), "dit_tokens_to_grid must exactly invert grid_to_dit_tokens"


def test_masked_loss_tracks_spatial_region_nonsquare():
    # THE alignment gate. gh=4, gw=6 (non-square); mask = top 2 rows only (asymmetric).
    gh, gw = 4, 6
    target_grid = torch.zeros(1, 128, gh, gw)
    pred_in = target_grid.clone()
    pred_in[:, :, :2, :] = 5.0  # error INSIDE the masked (top) region
    pred_out = target_grid.clone()
    pred_out[:, :, 2:, :] = 5.0  # error OUTSIDE the mask (bottom)

    target_tokens = grid_to_dit_tokens(target_grid)  # tokens are what the trainer's model_pred/target look like
    mask = torch.zeros(1, 1, gh, gw)
    mask[:, :, :2, :] = 1.0
    args = _mask_args()

    def reduce(pred_tokens):
        pg = dit_tokens_to_grid(pred_tokens, gh, gw)
        tg = dit_tokens_to_grid(target_tokens, gh, gw)
        lu = compute_unreduced_target_loss(pg, tg, loss_type="mse")
        return float(apply_masked_loss_with_prior(lu, mask, prior_loss_unreduced=None, args=args, layout="video"))

    loss_in = reduce(grid_to_dit_tokens(pred_in))
    loss_out = reduce(grid_to_dit_tokens(pred_out))
    assert loss_in > 1.0, f"error inside the mask must drive loss, got {loss_in}"
    assert loss_out < 1e-4, f"error outside the mask must be ignored, got {loss_out}"


def test_allones_mask_equals_plain_mean():
    # use_mask_loss=True with a uniform mask must reduce to ~plain mean MSE (pins num_channels compensation).
    gh, gw = 4, 6
    pred = torch.randn(1, gh * gw, 128)
    target = torch.randn(1, gh * gw, 128)
    lu = compute_unreduced_target_loss(dit_tokens_to_grid(pred, gh, gw), dit_tokens_to_grid(target, gh, gw), loss_type="mse")
    masked = apply_masked_loss_with_prior(
        lu, torch.ones(1, 1, gh, gw), prior_loss_unreduced=None, args=_mask_args(), layout="video"
    )
    plain = torch.nn.functional.mse_loss(pred, target, reduction="mean")
    assert torch.allclose(masked.float(), plain.float(), atol=1e-5), (float(masked), float(plain))


def test_use_mask_loss_requires_mask_weights():
    with pytest.raises(ValueError, match="mask_weights"):
        require_mask_weights_if_enabled({}, SimpleNamespace(use_mask_loss=True))
    # present -> no raise; disabled -> no raise even without masks.
    require_mask_weights_if_enabled({"mask_weights": torch.ones(1, 1, 1, 4, 6)}, SimpleNamespace(use_mask_loss=True))
    require_mask_weights_if_enabled({}, SimpleNamespace(use_mask_loss=False))


def test_save_latent_cache_writes_mask_weights(tmp_path):
    item = _item(tmp_path)
    save_latent_cache_ideogram4(item, torch.randn(128, 4, 6).to(torch.bfloat16), mask_weights=torch.rand(1, 1, 4, 6))
    sd = load_file(item.latent_cache_path)
    assert "mask_weights_4x6_float16" in sd, sorted(sd)
    assert tuple(sd["mask_weights_4x6_float16"].shape) == (1, 1, 4, 6)


def test_cache_downsamples_mask_to_token_grid_not_raw(tmp_path, monkeypatch):
    # The mask must downsample to the LATENT token grid (gh, gw), not the raw image or the /8 VAE grid.
    import musubi_tuner.ideogram4_cache_latents as cl

    monkeypatch.setattr(cl.ideogram4_utils, "encode_pixels_to_latent_grid", lambda ae, px: (torch.randn(1, 128, 4, 6), 4, 6))
    item = _item(tmp_path)
    item.content = np.zeros((96, 96, 3), dtype=np.uint8)  # raw 96x96 image
    item.mask_content = (np.random.rand(96, 96) * 255).astype(np.uint8)  # raw 96x96 mask
    item.cache_mask_gamma, item.cache_mask_min_weight = 1.0, 0.0

    cl.encode_and_save_batch(object(), [item])
    sd = load_file(item.latent_cache_path)
    assert "mask_weights_4x6_float16" in sd, "mask must be keyed/downsampled to the 4x6 token grid (not 96x96 or /8)"
    assert tuple(sd["mask_weights_4x6_float16"].shape) == (1, 1, 4, 6)


def test_preflight_requires_mask_only_when_enabled(tmp_path):
    no_mask = _item(tmp_path, key="nomask")
    save_latent_cache_ideogram4(no_mask, torch.randn(128, 4, 6).to(torch.bfloat16))  # no mask
    preflight_ideogram4_latent_cache(no_mask.latent_cache_path)  # default: accepts maskless
    with pytest.raises(ValueError, match="mask_weights"):
        preflight_ideogram4_latent_cache(no_mask.latent_cache_path, require_mask_weights=True)

    masked = _item(tmp_path, key="masked")
    save_latent_cache_ideogram4(masked, torch.randn(128, 4, 6).to(torch.bfloat16), mask_weights=torch.rand(1, 1, 4, 6))
    preflight_ideogram4_latent_cache(masked.latent_cache_path, require_mask_weights=True)  # ok with a mask


def _write_cache_with_extra_tensors(latent_cache_path, latent, mask_overrides):
    """Bypass the write-side shape assert by writing through safetensors directly — used to forge the kinds of
    stale/foreign caches that the preflight has to catch on read. Mirrors save_latent_cache_ideogram4's keys."""
    from safetensors.torch import save_file

    from musubi_tuner.ideogram4 import constants as i4c

    _, gh, gw = latent.shape
    sd = {f"latents_{gh}x{gw}_bfloat16": latent.contiguous()}
    for key, tensor in mask_overrides.items():
        sd[key] = tensor
    save_file(
        sd,
        str(latent_cache_path),
        metadata={
            i4c.IDEOGRAM4_LATENT_NORM_METADATA_KEY: "true",
            i4c.IDEOGRAM4_LATENT_LAYOUT_KEY: i4c.IDEOGRAM4_LATENT_LAYOUT_GRID_CHW,
            i4c.IDEOGRAM4_LATENT_SPACE_KEY: i4c.IDEOGRAM4_LATENT_SPACE_DIT_TOKENS,
            "architecture": "ideogram4",
        },
    )


def test_preflight_rejects_mask_grid_mismatch(tmp_path):
    # The fail-fast headline scenario: a stale 128x128 mask paired with a fresh 4x6 latent (different
    # resolution, different cache). Without this check, preflight passes, the 8B DiT loads, and training
    # crashes at the first loss reduction inside apply_masked_loss_with_prior. We want the abort BEFORE load.
    path = tmp_path / "mismatch_4x6_i4.safetensors"
    _write_cache_with_extra_tensors(
        path,
        torch.zeros(128, 4, 6, dtype=torch.bfloat16),
        {"mask_weights_128x128_float16": torch.zeros(1, 1, 128, 128, dtype=torch.float16)},
    )
    with pytest.raises(ValueError, match=r"grid 128x128 != latent grid 4x6"):
        preflight_ideogram4_latent_cache(str(path), require_mask_weights=True)


def test_preflight_rejects_multiple_mask_tensors(tmp_path):
    # A cache should hold exactly one mask. Two is either a writer bug or a corrupted manual edit; either way
    # apply_masked_loss_with_prior only reads the one the shared bucket reader picks first, so the others are
    # ghosts. Refuse to train against ambiguous state.
    path = tmp_path / "dual_4x6_i4.safetensors"
    _write_cache_with_extra_tensors(
        path,
        torch.zeros(128, 4, 6, dtype=torch.bfloat16),
        {
            "mask_weights_4x6_float16": torch.zeros(1, 1, 4, 6, dtype=torch.float16),
            "mask_weights_8x12_float16": torch.zeros(1, 1, 8, 12, dtype=torch.float16),
        },
    )
    with pytest.raises(ValueError, match="exactly one mask_weights"):
        preflight_ideogram4_latent_cache(str(path), require_mask_weights=True)


def test_preflight_rejects_malformed_mask_key(tmp_path):
    # A non-parseable mask key (no "x" in the geometry token) is a corrupted writer or a foreign file. Reject
    # explicitly so the error points at the key, not at a downstream tensor op three layers deep.
    path = tmp_path / "malformed_4x6_i4.safetensors"
    _write_cache_with_extra_tensors(
        path,
        torch.zeros(128, 4, 6, dtype=torch.bfloat16),
        {"mask_weights_GARBAGE_float16": torch.zeros(1, 1, 4, 6, dtype=torch.float16)},
    )
    with pytest.raises(ValueError, match="malformed mask key"):
        preflight_ideogram4_latent_cache(str(path), require_mask_weights=True)


def test_preflight_rejects_wrong_mask_tensor_shape(tmp_path):
    # The mask must be (1, 1, gh, gw). A (1, 1, gh, gw, 1) etc. shape (cache corruption / partial write) would
    # broadcast unpredictably inside apply_masked_loss_with_prior. Catch it at preflight so the abort message
    # names "mask shape," not "loss reduction shape mismatch."
    path = tmp_path / "wrongshape_4x6_i4.safetensors"
    _write_cache_with_extra_tensors(
        path,
        torch.zeros(128, 4, 6, dtype=torch.bfloat16),
        {"mask_weights_4x6_float16": torch.zeros(4, 6, dtype=torch.float16)},  # 2D, missing (1, 1, ...) prefix
    )
    with pytest.raises(ValueError, match=r"expected shape \(1, 1, 4, 6\)"):
        preflight_ideogram4_latent_cache(str(path), require_mask_weights=True)


def test_save_latent_cache_rejects_bad_mask_shape(tmp_path):
    # The write-side guard: docstring promises (1, 1, gh, gw); enforce it at write time so a future caller can
    # not plant a malformed mask that the preflight then has to field. ValueError (not assert) so python -O
    # cannot strip the data-integrity contract at this I/O boundary.
    item = _item(tmp_path, key="badwrite")
    with pytest.raises(ValueError, match=r"\(1, 1, 4, 6\)"):
        save_latent_cache_ideogram4(item, torch.zeros(128, 4, 6).to(torch.bfloat16), mask_weights=torch.zeros(4, 6))


def test_process_batch_returns_masked_loss_telemetry():
    # When trackers are wired, process_batch must return loss_metrics with masked_loss/* keys so TensorBoard
    # can confirm the mask is active and non-degenerate during long likeness runs. Without this, the only
    # signal is the loss number itself, which looks plausible even with an all-zero or all-one mask.
    from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer

    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(
        use_mask_loss=True,
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
    )

    gh, gw = 4, 6
    latents = torch.zeros(1, 128, gh, gw)
    mask = torch.ones(1, 1, 1, gh, gw)  # (B, 1, F=1, gh, gw), what the bucket reader stacks to
    batch = {"mask_weights": mask, "i4_llm_features": [torch.zeros(1, 53248)]}

    # Stub out ideogram4_flow_matching_target + the resolution scheduler so we don't load the real DiT.
    import musubi_tuner.ideogram4_train_network as itn

    accelerator_stub = SimpleNamespace(device=torch.device("cpu"), trackers=[object()])  # nonempty -> stats on
    model_pred_tokens = torch.zeros(1, gh * gw, 128)
    target_tokens = torch.full((1, gh * gw, 128), 0.5)

    def _flow(transformer, latents, text_features, noise, timesteps, *, network_dtype, device):
        return model_pred_tokens.to(network_dtype), target_tokens.to(network_dtype)

    def _schedule(reso, *, known_mean, std):
        return lambda u: u  # identity; timesteps shape is what matters

    orig_flow = itn.ideogram4_flow_matching_target
    orig_sched = itn.get_schedule_for_resolution
    itn.ideogram4_flow_matching_target = _flow
    itn.get_schedule_for_resolution = _schedule
    try:
        loss, metrics = trainer.process_batch(
            args, accelerator_stub, None, None, batch, latents, None, None, torch.float32, torch.float32, None, 0
        )
    finally:
        itn.ideogram4_flow_matching_target = orig_flow
        itn.get_schedule_for_resolution = orig_sched

    assert torch.is_tensor(loss)
    assert any(k.startswith("masked_loss/") for k in metrics), f"expected masked_loss/* telemetry, got {sorted(metrics)}"
    assert "masked_loss/target" in metrics
    assert "masked_loss/mask/raw_mean" in metrics  # uniform-one mask -> raw_mean == 1.0
    assert abs(metrics["masked_loss/mask/raw_mean"] - 1.0) < 1e-6
