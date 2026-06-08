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
