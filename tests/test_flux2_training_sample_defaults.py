"""FLUX.2 training sample prompt defaults.

These tests ensure sample image generation defaults match FLUX2_MODEL_INFO per variant.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer


def test_flux2_sample_defaults_dev_filled_when_missing():
    model_info = flux2_utils.FLUX2_MODEL_INFO["dev"]
    prompt_dict = {"prompt": "a cat"}
    Flux2NetworkTrainer._apply_flux2_sample_defaults("dev", model_info, prompt_dict)

    assert prompt_dict["sample_steps"] == 50
    assert prompt_dict["guidance_scale"] == 4.0


def test_flux2_sample_defaults_klein_9b_filled_when_missing():
    model_info = flux2_utils.FLUX2_MODEL_INFO["klein-9b"]
    prompt_dict = {"prompt": "a cat"}
    Flux2NetworkTrainer._apply_flux2_sample_defaults("klein-9b", model_info, prompt_dict)

    assert prompt_dict["sample_steps"] == 4
    assert prompt_dict["guidance_scale"] == 1.0


def test_flux2_sample_defaults_klein_base_9b_filled_when_missing():
    model_info = flux2_utils.FLUX2_MODEL_INFO["klein-base-9b"]
    prompt_dict = {"prompt": "a cat"}
    Flux2NetworkTrainer._apply_flux2_sample_defaults("klein-base-9b", model_info, prompt_dict)

    assert prompt_dict["sample_steps"] == 50
    assert prompt_dict["guidance_scale"] == 4.0


def test_flux2_sample_defaults_does_not_override_explicit_prompt_values():
    model_info = flux2_utils.FLUX2_MODEL_INFO["dev"]
    prompt_dict = {"prompt": "a cat", "sample_steps": 12, "guidance_scale": 2.5}
    Flux2NetworkTrainer._apply_flux2_sample_defaults("dev", model_info, prompt_dict)

    assert prompt_dict["sample_steps"] == 12
    assert prompt_dict["guidance_scale"] == 2.5


def test_flux2_trainer_default_guidance_scale_matches_model_default():
    trainer = Flux2NetworkTrainer()

    args = SimpleNamespace(
        model_version="klein-9b",
        fp8_text_encoder=False,
        mixed_precision="bf16",
        split_attn=True,
    )
    trainer.handle_model_specific_args(args)
    assert trainer.default_guidance_scale == 1.0


def test_flux2_sample_defaults_enforces_fixed_params_for_distilled_variants(caplog):
    model_info = flux2_utils.FLUX2_MODEL_INFO["klein-9b"]
    prompt_dict = {"prompt": "a cat", "sample_steps": 12, "guidance_scale": 2.5}

    with caplog.at_level(logging.WARNING):
        Flux2NetworkTrainer._apply_flux2_sample_defaults("klein-9b", model_info, prompt_dict)
    assert prompt_dict["sample_steps"] == 4
    assert prompt_dict["guidance_scale"] == 1.0

    warning_text = "\n".join(r.message for r in caplog.records)
    assert "fixed for klein-9b" in warning_text
