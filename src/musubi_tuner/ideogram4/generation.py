"""Ideogram 4 flow-matching denoise loop (asymmetric CFG), faithful to canonical pipeline_ideogram4.__call__.

Canonical t-convention (the B5 fix): z starts as pure noise presented at t≈0.0001 and is integrated toward
clean at t≈0.999 (`z += v * (s_val - t_val)`, t increasing). The CONDITIONAL branch runs the full
[text][image] sequence; the UNCONDITIONAL branch is IMAGE-ONLY (a separate transformer, zero llm_features,
image-slice positions) — that is the asymmetry. Guidance is `v = gw*pos_v + (1-gw)*neg_v` (standard CFG).
"""

from __future__ import annotations

import torch

from musubi_tuner.ideogram4.sampler_configs import SamplerParameters
from musubi_tuner.ideogram4.scheduler import LogitNormalSchedule, get_schedule_for_resolution, make_step_intervals
from musubi_tuner.ideogram4.sequence import (
    Ideogram4Sequence,
    build_ideogram4_conditioning,
    build_image_input,
    extract_image_tokens,
    image_grid_dims,
)

IDEOGRAM4_LATENT_DIM = 128  # DiT in_channels (patchified token channels)


@torch.no_grad()
def denoise_ideogram4_tokens(
    conditional_model,
    unconditional_model,
    sequence: Ideogram4Sequence,
    *,
    num_steps: int,
    guidance_schedule,
    schedule: LogitNormalSchedule,
    device: torch.device | str | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    latent_dim: int = IDEOGRAM4_LATENT_DIM,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run the canonical denoise loop. Returns the denoised normalized DiT tokens z (B, num_image, latent_dim)."""
    device = torch.device(device) if device is not None else sequence.llm_features.device
    batch_size = sequence.llm_features.shape[0]
    num_image = sequence.num_image_tokens
    image_start = sequence.image_start
    total = sequence.total_seq_len
    feature_dim = sequence.llm_features.shape[-1]

    guidance = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)
    if guidance.shape != (num_steps,):
        raise ValueError(f"guidance_schedule must have shape ({num_steps},), got {tuple(guidance.shape)}")
    step_intervals = make_step_intervals(num_steps).to(device)

    # Unconditional branch: image-only, zero conditioning, image-slice positions/segment/indicator.
    neg_llm_features = torch.zeros(batch_size, num_image, feature_dim, dtype=sequence.llm_features.dtype, device=device)
    neg_position_ids = sequence.position_ids[:, image_start:]
    neg_segment_ids = sequence.segment_ids[:, image_start:]
    neg_indicator = sequence.indicator[:, image_start:]

    z = torch.randn(batch_size, num_image, latent_dim, dtype=torch.float32, device=device, generator=generator)

    for i in range(num_steps - 1, -1, -1):
        t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
        s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
        t = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

        # Conditional: scatter z into the full [text][image] sequence; the DiT prediction is at image positions.
        pos_x = build_image_input(z.to(compute_dtype), total, image_start)
        pos_out = conditional_model(
            x=pos_x,
            t=t,
            llm_features=sequence.llm_features,
            position_ids=sequence.position_ids,
            segment_ids=sequence.segment_ids,
            indicator=sequence.indicator,
        )
        pos_v = extract_image_tokens(pos_out, image_start, num_image).float()

        # Unconditional: image-only forward on z directly.
        neg_out = unconditional_model(
            x=z.to(compute_dtype),
            t=t,
            llm_features=neg_llm_features,
            position_ids=neg_position_ids,
            segment_ids=neg_segment_ids,
            indicator=neg_indicator,
        )
        neg_v = neg_out.float()

        gw_i = guidance[i]
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        z = z + v * (s_val - t_val)  # canonical: integrate toward clean (delta > 0), noise@t~0 -> clean@t~1

    return z


def denoise_ideogram4_to_tokens(
    conditional_model,
    unconditional_model,
    text_features: torch.Tensor,
    *,
    height: int,
    width: int,
    preset: SamplerParameters,
    device: torch.device | str,
    compute_dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Build conditioning -> asymmetric-CFG denoise -> (z normalized DiT tokens, grid_h, grid_w).

    The SINGLE source of the validated B5 t-convention for BOTH manual generation and sampling-during-training,
    so it can never drift between them. Decode is a SEPARATE call (decode_dit_tokens_to_pixels) so a caller can
    free the unconditional DiT BEFORE the VAE decode — keeping both 9.4 GB DiTs resident through a 1024 decode
    peaks ~30 GB, which OOMs sampling on a 32 GB card. ``text_features`` is one prompt's (L, 53248) features.
    """
    grid_h, grid_w = image_grid_dims(height, width)
    sequence = build_ideogram4_conditioning([text_features], grid_h, grid_w, device=device, dtype=compute_dtype)
    schedule = get_schedule_for_resolution((height, width), known_mean=preset.mu, std=preset.std)
    z = denoise_ideogram4_tokens(
        conditional_model,
        unconditional_model,
        sequence,
        num_steps=preset.num_steps,
        guidance_schedule=preset.guidance_schedule,
        schedule=schedule,
        device=device,
        compute_dtype=compute_dtype,
        generator=generator,
    )
    return z, grid_h, grid_w
