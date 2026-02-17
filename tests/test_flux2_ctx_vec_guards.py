from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from musubi_tuner.flux_2 import flux2_utils


def test_validate_ctx_vec_dim_raises_on_mismatch():
    model_info = flux2_utils.FLUX2_MODEL_INFO["klein-9b"]
    # dev dim (15360) mismatches klein-9b expected (12288)
    bad_ctx = torch.zeros((1, 512, 15360), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"ctx_vec dim mismatch"):
        flux2_utils.validate_ctx_vec_dim(bad_ctx, model_info, source="test")


def test_prepare_text_inputs_raises_on_ctx_dim_mismatch():
    from musubi_tuner import flux_2_generate_image

    class FakeEmbedder:
        def __init__(self, dim: int):
            self._device = torch.device("cpu")
            self._dim = dim
            self._dtype = torch.bfloat16

        @property
        def device(self):
            return self._device

        @property
        def dtype(self):
            return self._dtype

        def to(self, *args, **kwargs):
            if args:
                self._device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
            if "dtype" in kwargs and kwargs["dtype"] is not None:
                self._dtype = kwargs["dtype"]
            return self

        def __call__(self, prompts: list[str]):
            bsz = len(prompts)
            return torch.zeros((bsz, 512, self._dim), dtype=self._dtype, device=self._device)

    args = SimpleNamespace(
        model_version="dev",
        fp8_text_encoder=False,
        text_encoder="/fake/te",
        prompt="hello",
        negative_prompt=None,
        blocks_to_swap=0,
    )

    with mock.patch.object(flux2_utils, "load_text_embedder", return_value=FakeEmbedder(dim=123)):
        with pytest.raises(ValueError, match=r"ctx_vec dim mismatch"):
            flux_2_generate_image.prepare_text_inputs(args, device=torch.device("cpu"), shared_models=None)


def test_cache_text_encoder_outputs_casts_bf16_and_validates_dim():
    from musubi_tuner import flux_2_cache_text_encoder_outputs as cache_te

    class FakeEmbedder:
        def __init__(self):
            self._dtype = torch.bfloat16

        @property
        def dtype(self):
            return self._dtype

        def __call__(self, prompts: list[str]):
            # Return float32 to ensure the caching path normalizes dtype to bf16.
            return torch.zeros((len(prompts), 8, 4), dtype=torch.float32)

    # Use a tiny dummy model_info so the test tensor stays small.
    cache_te._model_version_info = SimpleNamespace(params=SimpleNamespace(context_in_dim=4), architecture_full="flux_2_dev")

    saved = []

    def fake_save(item_info, ctx_vec, arch_full):
        saved.append((ctx_vec.dtype, tuple(ctx_vec.shape), arch_full))

    batch = [SimpleNamespace(caption="a"), SimpleNamespace(caption="b")]

    with mock.patch.object(cache_te, "save_text_encoder_output_cache_flux_2", side_effect=fake_save):
        cache_te.encode_and_save_batch(
            text_embedder=FakeEmbedder(),
            guidance_distilled=True,
            batch=batch,
            device=torch.device("cpu"),
            arch_full="flux_2_dev",
        )

    assert saved, "Expected save_text_encoder_output_cache_flux_2 to be called"
    assert all(dtype == torch.bfloat16 for (dtype, _shape, _arch) in saved)
    assert all(shape == (8, 4) for (_dtype, shape, _arch) in saved)


def test_training_raises_on_ctx_dim_mismatch_early():
    from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer

    trainer = Flux2NetworkTrainer()
    trainer.model_version_info = flux2_utils.FLUX2_MODEL_INFO["dev"]

    args = SimpleNamespace(gradient_checkpointing=False)
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    latents = torch.zeros((1, 128, 2, 2), dtype=torch.float32)
    noise = torch.zeros_like(latents)
    noisy_model_input = torch.zeros_like(latents)
    timesteps = torch.tensor([0.0])

    batch = {"ctx_vec": torch.zeros((1, 8, 1), dtype=torch.bfloat16)}

    with pytest.raises(ValueError, match=r"ctx_vec dim mismatch"):
        trainer.call_dit(
            args=args,
            accelerator=accelerator,
            transformer=None,
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            network_dtype=torch.bfloat16,
        )
