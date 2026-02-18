from __future__ import annotations

from types import SimpleNamespace

import torch

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer


class _GuidanceCaptureModel:
    def __init__(self):
        self.guidance: torch.Tensor | None = None

    def __call__(self, *, x, x_ids, timesteps, ctx, ctx_ids, guidance):  # noqa: ANN001
        self.guidance = guidance
        return torch.zeros_like(x)


def _run_call_dit_with_guidance_scale(*, training_guidance_scale: float | None) -> torch.Tensor:
    trainer = Flux2NetworkTrainer()
    trainer.model_version_info = flux2_utils.FLUX2_MODEL_INFO["dev"]

    args = SimpleNamespace(gradient_checkpointing=False)
    if training_guidance_scale is not None:
        args.training_guidance_scale = training_guidance_scale

    accelerator = SimpleNamespace(device=torch.device("cpu"))
    model = _GuidanceCaptureModel()

    bsize = 2
    latents = torch.zeros((bsize, 4, 2, 2), dtype=torch.float32)
    noise = torch.zeros_like(latents)
    noisy_model_input = torch.zeros_like(latents)
    timesteps = torch.zeros((bsize,), dtype=torch.float32)

    ctx_dim = int(trainer.model_version_info.params.context_in_dim)
    batch = {"ctx_vec": torch.zeros((bsize, 1, ctx_dim), dtype=torch.bfloat16)}

    trainer.call_dit(
        args=args,
        accelerator=accelerator,
        transformer=model,
        latents=latents,
        batch=batch,
        noise=noise,
        noisy_model_input=noisy_model_input,
        timesteps=timesteps,
        network_dtype=torch.bfloat16,
    )
    assert model.guidance is not None
    return model.guidance


def test_flux2_training_guidance_scale_is_used():
    """HC-1: training guidance embeddings must be configurable (DEV)."""
    guidance = _run_call_dit_with_guidance_scale(training_guidance_scale=2.5)
    assert guidance.shape == (2,)
    assert guidance.dtype == torch.bfloat16
    assert guidance.device.type == "cpu"
    assert torch.allclose(guidance, torch.tensor([2.5, 2.5], dtype=torch.bfloat16))


def test_flux2_training_guidance_scale_defaults_to_one_when_missing():
    guidance = _run_call_dit_with_guidance_scale(training_guidance_scale=None)
    assert torch.allclose(guidance, torch.tensor([1.0, 1.0], dtype=torch.bfloat16))
