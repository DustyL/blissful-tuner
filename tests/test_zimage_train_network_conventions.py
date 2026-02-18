import argparse
import contextlib
import math

import pytest
import torch

from musubi_tuner.zimage import zimage_config
from musubi_tuner.zimage_train_network import ZImageNetworkTrainer


class _DummyAccelerator:
    def __init__(self, device: torch.device):
        self.device = device

    def unwrap_model(self, model):
        return model

    @contextlib.contextmanager
    def autocast(self):
        yield


class _DummyTransformer(torch.nn.Module):
    def __init__(self, output_value: float = 1.0, all_patch_size: tuple[int, ...] = (2,)):
        super().__init__()
        self.all_patch_size = all_patch_size
        self._output_value = output_value

        self.captured_cap_mask = None
        self.captured_cap_feats_shape = None

    def forward(self, x, t, cap_feats, cap_mask):
        self.captured_cap_mask = cap_mask
        self.captured_cap_feats_shape = tuple(cap_feats.shape)
        return x.new_full(x.shape, self._output_value)


def test_zimage_call_dit_standard_sign_convention():
    trainer = ZImageNetworkTrainer()
    args = argparse.Namespace(split_attn=False, gradient_checkpointing=False)
    accelerator = _DummyAccelerator(device=torch.device("cpu"))
    transformer = _DummyTransformer(output_value=2.0, all_patch_size=(2,))

    latents = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    noise = torch.ones_like(latents)
    noisy_model_input = latents + 0.5 * noise
    timesteps = torch.tensor([500.0], dtype=torch.float32)
    network_dtype = torch.float32

    batch = {"llm_embed": [torch.zeros((3, 2560), dtype=torch.float32)]}

    model_pred, target = trainer.call_dit(
        args=args,
        accelerator=accelerator,
        transformer=transformer,
        latents=latents,
        batch=batch,
        noise=noise,
        noisy_model_input=noisy_model_input,
        timesteps=timesteps,
        network_dtype=network_dtype,
    )

    assert model_pred.shape == latents.shape
    assert torch.allclose(model_pred, torch.full_like(latents, -2.0))
    assert torch.allclose(target, noise - latents)


@pytest.mark.parametrize("split_attn", [False, True])
def test_zimage_call_dit_masks_caption_padding_when_batched(split_attn: bool):
    trainer = ZImageNetworkTrainer()
    args = argparse.Namespace(split_attn=split_attn, gradient_checkpointing=False)
    accelerator = _DummyAccelerator(device=torch.device("cpu"))
    transformer = _DummyTransformer(output_value=0.0, all_patch_size=(2,))

    # Two samples with different caption lengths -> requires padding and an attention mask.
    latents = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
    noise = torch.zeros_like(latents)
    noisy_model_input = latents
    timesteps = torch.tensor([500.0, 500.0], dtype=torch.float32)
    network_dtype = torch.float32

    txt0 = torch.zeros((3, 2560), dtype=torch.float32)
    txt1 = torch.zeros((5, 2560), dtype=torch.float32)
    batch = {"llm_embed": [txt0, txt1]}

    trainer.call_dit(
        args=args,
        accelerator=accelerator,
        transformer=transformer,
        latents=latents,
        batch=batch,
        noise=noise,
        noisy_model_input=noisy_model_input,
        timesteps=timesteps,
        network_dtype=network_dtype,
    )

    cap_mask = transformer.captured_cap_mask
    assert cap_mask is not None, "Expected cap_mask to be created when batching variable-length captions"
    assert cap_mask.dtype == torch.bool

    image_seq_len = (latents.shape[2] // transformer.all_patch_size[0]) * (latents.shape[3] // transformer.all_patch_size[0])
    padded_total_len = math.ceil((5 + image_seq_len) / zimage_config.SEQ_MULTI_OF) * zimage_config.SEQ_MULTI_OF
    expected_max_len = int(padded_total_len) - image_seq_len

    assert cap_mask.shape == (2, expected_max_len)
    assert transformer.captured_cap_feats_shape == (2, expected_max_len, 2560)

    assert bool(cap_mask[0, :3].all().item())
    assert not bool(cap_mask[0, 3:].any().item())
    assert bool(cap_mask[1, :5].all().item())
    assert not bool(cap_mask[1, 5:].any().item())
