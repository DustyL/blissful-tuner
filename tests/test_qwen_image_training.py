import contextlib
import unittest
from types import SimpleNamespace

import torch

from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer


class _FakeAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()


class _CapturingDummyQwenImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        self.last_kwargs = kwargs
        return torch.zeros_like(hidden_states)


class TestQwenImageTrainingCallDit(unittest.TestCase):
    def _make_common_inputs(self, *, bsz: int, channels: int, height: int, width: int):
        latents = torch.zeros(bsz, channels, 1, height, width, dtype=torch.float32)
        noise = torch.zeros_like(latents)
        noisy_model_input = torch.zeros_like(latents)
        timesteps = torch.zeros(bsz, dtype=torch.float32)
        batch = {
            "latents": latents,
            "vl_embed": [torch.zeros(2, 8, dtype=torch.float32) for _ in range(bsz)],  # list[(txt_len, dim)]
        }
        return latents, noise, noisy_model_input, timesteps, batch

    def test_t2i_img_shapes_is_nested_list(self):
        trainer = QwenImageNetworkTrainer()
        trainer.is_edit = False
        trainer.is_layered = False

        latents, noise, noisy_model_input, timesteps, batch = self._make_common_inputs(bsz=1, channels=16, height=4, width=4)
        args = SimpleNamespace(
            is_layered=False, remove_first_image_from_target=False, split_attn=False, gradient_checkpointing=False
        )
        accelerator = _FakeAccelerator()
        model = _CapturingDummyQwenImageModel()

        model_pred, target = trainer.call_dit(
            args=args,
            accelerator=accelerator,
            transformer=model,
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            network_dtype=torch.float32,
        )

        self.assertEqual(model_pred.shape, latents.shape)
        self.assertEqual(target.shape, latents.shape)
        self.assertEqual(model.last_kwargs["img_shapes"], [[(1, 2, 2)]])

    def test_edit_img_shapes_includes_control(self):
        trainer = QwenImageNetworkTrainer()
        trainer.is_edit = True
        trainer.is_layered = False

        latents, noise, noisy_model_input, timesteps, batch = self._make_common_inputs(bsz=1, channels=16, height=4, width=4)
        batch["latents_control_0"] = torch.zeros_like(latents)

        args = SimpleNamespace(
            is_layered=False, remove_first_image_from_target=False, split_attn=False, gradient_checkpointing=False
        )
        accelerator = _FakeAccelerator()
        model = _CapturingDummyQwenImageModel()

        model_pred, target = trainer.call_dit(
            args=args,
            accelerator=accelerator,
            transformer=model,
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            network_dtype=torch.float32,
        )

        self.assertEqual(model_pred.shape, latents.shape)
        self.assertEqual(target.shape, latents.shape)
        self.assertEqual(model.last_kwargs["img_shapes"], [[(1, 2, 2), (1, 2, 2)]])

    def test_layered_img_shapes_counts_base_plus_layers_plus_control(self):
        trainer = QwenImageNetworkTrainer()
        trainer.is_edit = False
        trainer.is_layered = True

        bsz, channels, num_images, height, width = 1, 16, 3, 4, 4  # 1 base + 2 layers
        latents = torch.zeros(bsz, channels, num_images, height, width, dtype=torch.float32)
        noise = torch.zeros_like(latents)
        noisy_model_input = torch.zeros_like(latents)
        timesteps = torch.zeros(bsz, dtype=torch.float32)
        batch = {"latents": latents, "vl_embed": [torch.zeros(2, 8, dtype=torch.float32)]}

        args = SimpleNamespace(
            is_layered=True, remove_first_image_from_target=False, split_attn=False, gradient_checkpointing=False
        )
        accelerator = _FakeAccelerator()
        model = _CapturingDummyQwenImageModel()

        model_pred, target = trainer.call_dit(
            args=args,
            accelerator=accelerator,
            transformer=model,
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            network_dtype=torch.float32,
        )

        # Layered call_dit permutes from (B, C, L, H, W) -> (B, L, C, H, W)
        self.assertEqual(model_pred.shape, (bsz, num_images, channels, height, width))
        self.assertEqual(target.shape, (bsz, num_images, channels, height, width))
        # img_shapes: base+layers (3) + control (1) == 4 entries
        self.assertEqual(len(model.last_kwargs["img_shapes"][0]), 4)

    def test_layered_remove_first_image_from_target_reduces_output_layers(self):
        trainer = QwenImageNetworkTrainer()
        trainer.is_edit = False
        trainer.is_layered = True

        bsz, channels, num_images, height, width = 1, 16, 3, 4, 4  # 1 base + 2 layers
        latents = torch.zeros(bsz, channels, num_images, height, width, dtype=torch.float32)
        noise = torch.zeros_like(latents)
        noisy_model_input = torch.zeros_like(latents)
        timesteps = torch.zeros(bsz, dtype=torch.float32)
        batch = {"latents": latents, "vl_embed": [torch.zeros(2, 8, dtype=torch.float32)]}

        args = SimpleNamespace(is_layered=True, remove_first_image_from_target=True, split_attn=False, gradient_checkpointing=False)
        accelerator = _FakeAccelerator()
        model = _CapturingDummyQwenImageModel()

        model_pred, target = trainer.call_dit(
            args=args,
            accelerator=accelerator,
            transformer=model,
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            network_dtype=torch.float32,
        )

        # Base image is dropped from the target for loss and prediction: 2 remaining "layers".
        self.assertEqual(model_pred.shape, (bsz, num_images - 1, channels, height, width))
        self.assertEqual(target.shape, (bsz, num_images - 1, channels, height, width))
        # img_shapes: remaining targets (2) + control (1) == 3 entries
        self.assertEqual(len(model.last_kwargs["img_shapes"][0]), 3)


if __name__ == "__main__":
    unittest.main()
