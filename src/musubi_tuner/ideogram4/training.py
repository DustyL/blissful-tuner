"""Ideogram 4 training flow-matching (the canonical, generate-VALIDATED convention).

Convention (confirmed by the Phase-0 generate gate, opposite of the fork's mirrored B5 bug):
    t in [0, 1] is the "cleanness" coefficient; noisy = (1 - t) * noise + t * clean  (noise@t=0, clean@t=1).
    velocity target = clean - noise.  loss = MSE(model_pred, target) at OUTPUT_IMAGE positions.

The cached latents are the already-patchified + latent_norm'd grid (B, 128, gh, gw); the trainer flattens them
with grid_to_dit_tokens (NEVER patchify or latent_norm again — that is the B3 bug) and scatters the noised
image tokens into the joint [pad][text][image] sequence (shared sequence helpers).
"""

from __future__ import annotations

import torch

from musubi_tuner.ideogram4.ideogram4_utils import grid_to_dit_tokens
from musubi_tuner.ideogram4.sequence import build_ideogram4_conditioning, build_image_input, extract_image_tokens


def ideogram4_flow_matching_target(
    conditional_model,
    latents: torch.Tensor,
    text_features: list[torch.Tensor],
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    network_dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the conditional DiT on noised tokens and return (model_pred, target) at image positions.

    latents / noise: cached grids (B, 128, gh, gw). text_features: list of B varlen (L_text_i, 53248).
    timesteps: (B,) cleanness coefficients in [0, 1].
    """
    device = torch.device(device) if device is not None else latents.device
    grid_h, grid_w = int(latents.shape[2]), int(latents.shape[3])

    clean = grid_to_dit_tokens(latents.to(device))  # (B, num_image, 128) — already patchified + normalized
    noise_tokens = grid_to_dit_tokens(noise.to(device))

    sequence = build_ideogram4_conditioning(text_features, grid_h, grid_w, device=device, dtype=network_dtype)

    t = timesteps.to(device=device, dtype=torch.float32).view(-1, 1, 1)
    noisy = (1.0 - t) * noise_tokens + t * clean  # noise@t=0 -> clean@t=1 (VALIDATED canonical convention)
    target = clean - noise_tokens  # velocity = d(noisy)/dt

    x = build_image_input(noisy.to(network_dtype), sequence.total_seq_len, sequence.image_start)
    output = conditional_model(
        x=x,
        t=timesteps.to(device=device, dtype=torch.float32),
        llm_features=sequence.llm_features,
        position_ids=sequence.position_ids,
        segment_ids=sequence.segment_ids,
        indicator=sequence.indicator,
    )
    model_pred = extract_image_tokens(output, sequence.image_start, sequence.num_image_tokens)
    return model_pred, target
