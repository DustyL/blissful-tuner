# Tests for the training-sample guidance override (backlog P0-4).
#
# Ideogram 4's asymmetric CFG (separate unconditional DiT with no LoRA vs. conditional DiT with LoRA)
# amplifies conditional-only LoRA noise by the guidance weight at every step, so training-time samples
# need a low uniform CFG to be interpretable during early training. These tests cover the override
# helper, the trainer wiring seam, the fail-fast validation, and that the production generation
# script does not grow the knob.

import argparse
from types import SimpleNamespace

import pytest

from musubi_tuner.ideogram4.sampler_configs import (
    PRESETS,
    apply_uniform_guidance,
    resolve_training_sample_preset,
)
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer, ideogram4_setup_parser


def test_apply_uniform_guidance_overrides_every_step_for_all_presets():
    for name, preset in PRESETS.items():
        out = apply_uniform_guidance(preset, 3.0)
        assert out.guidance_schedule == (3.0,) * preset.num_steps, name
        # Everything except the schedule is preserved.
        assert out.num_steps == preset.num_steps
        assert out.mu == preset.mu
        assert out.std == preset.std


def test_apply_uniform_guidance_does_not_mutate_registry():
    before = {name: p.guidance_schedule for name, p in PRESETS.items()}
    apply_uniform_guidance(PRESETS["V4_TURBO_12"], 1.5)
    assert {name: p.guidance_schedule for name, p in PRESETS.items()} == before


def test_apply_uniform_guidance_casts_to_float():
    out = apply_uniform_guidance(PRESETS["V4_TURBO_12"], 3)
    assert all(isinstance(g, float) for g in out.guidance_schedule)


def test_resolve_training_sample_preset_default_unchanged():
    # No override → the exact registry object, schedule untouched (production parity).
    args = SimpleNamespace(sampler_preset="V4_TURBO_12")
    assert resolve_training_sample_preset(args) is PRESETS["V4_TURBO_12"]
    args = SimpleNamespace(sampler_preset="V4_DEFAULT_20", ideogram4_sample_guidance=None)
    assert resolve_training_sample_preset(args) is PRESETS["V4_DEFAULT_20"]


def test_resolve_training_sample_preset_applies_override():
    args = SimpleNamespace(sampler_preset="V4_TURBO_12", ideogram4_sample_guidance=3.0)
    out = resolve_training_sample_preset(args)
    assert out.guidance_schedule == (3.0,) * 12
    assert out.num_steps == 12


def test_parser_default_is_none_and_flag_parses():
    parser = ideogram4_setup_parser(argparse.ArgumentParser())
    assert parser.parse_args([]).ideogram4_sample_guidance is None
    assert parser.parse_args(["--ideogram4_sample_guidance", "3.0"]).ideogram4_sample_guidance == 3.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_handle_model_specific_args_rejects_nonpositive_guidance(bad):
    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(ideogram4_sample_guidance=bad, mixed_precision="bf16")
    with pytest.raises(ValueError, match="ideogram4_sample_guidance"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_accepts_positive_guidance_without_sample_prompts():
    # Legal without --sample_prompts (it then simply never fires).
    trainer = Ideogram4NetworkTrainer()
    trainer.handle_model_specific_args(SimpleNamespace(ideogram4_sample_guidance=3.0, mixed_precision="bf16"))


def test_production_generation_parser_has_no_sample_guidance_flag():
    # The override is a TRAINING-sample knob; the standalone generation script must not grow it
    # silently (its presets stay production-faithful).
    from musubi_tuner.ideogram4_generate_image import setup_parser

    option_strings = [opt for action in setup_parser()._actions for opt in action.option_strings]
    assert "--ideogram4_sample_guidance" not in option_strings
