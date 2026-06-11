# Vendored verbatim from canonical src/ideogram4/sampler_configs.py (logit-normal schedule + v4 sampler presets),
# plus the blissful training-sample guidance override below the preset registry.

"""Named sampler configurations for Ideogram 4 inference."""

from __future__ import annotations

import dataclasses

from musubi_tuner.ideogram4.scheduler import SamplerParameters

# guidance_schedule is in loop-INDEX order: index 0 is the LAST (polish) step.
# Each preset does the first N_main sampling steps at gw=7, then N_cleanup
# polish steps at gw=3.
PRESETS: dict[str, SamplerParameters] = {
    "V4_QUALITY_48": SamplerParameters(
        num_steps=48,
        guidance_schedule=(3.0,) * 3 + (7.0,) * 45,
        mu=0.0,
        std=1.5,
    ),
    "V4_DEFAULT_20": SamplerParameters(
        num_steps=20,
        guidance_schedule=(3.0,) * 2 + (7.0,) * 18,
        mu=0.0,
        std=1.75,
    ),
    "V4_TURBO_12": SamplerParameters(
        num_steps=12,
        guidance_schedule=(3.0,) * 1 + (7.0,) * 11,
        mu=0.5,
        std=1.75,
    ),
}


# --- Blissful training-sample guidance override (backlog P0-4) ------------------------------------
# Ideogram 4's asymmetric CFG runs a separate unconditional DiT with NO LoRA against the conditional
# DiT WITH LoRA, so a conditional-only LoRA perturbation cannot cancel in `cond - uncond` and is
# instead multiplied by the guidance weight at every step. At the presets' production gw=7.0 an
# early-training LoRA (loud but uninformed) compounds into coherence collapse in TRAINING samples
# even when the adapter is healthy. A uniform low-CFG override (recommended: 3.0) keeps in-training
# probes interpretable. Production generation (ideogram4_generate_image.py) never calls these.


def apply_uniform_guidance(preset: SamplerParameters, guidance: float) -> SamplerParameters:
    """Return a copy of ``preset`` with every step's guidance replaced by ``guidance``."""
    return dataclasses.replace(preset, guidance_schedule=(float(guidance),) * preset.num_steps)


def resolve_training_sample_preset(args) -> SamplerParameters:
    """Select the training-sample preset, applying ``--ideogram4_sample_guidance`` when set."""
    preset = PRESETS[args.sampler_preset]
    override = getattr(args, "ideogram4_sample_guidance", None)
    if override is not None:
        preset = apply_uniform_guidance(preset, override)
    return preset
