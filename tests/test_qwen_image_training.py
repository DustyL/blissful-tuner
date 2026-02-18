import contextlib
import unittest
from types import SimpleNamespace

import torch

from musubi_tuner.hv_train_network import NetworkTrainer
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
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


class TestTimestepSigmaConsistency(unittest.TestCase):
    """Tests for the _map_continuous_t_to_sigma_and_timesteps hook."""

    def test_qwen_timestep_sigma_consistency(self):
        """Verify sigma used for mixing equals timesteps/1000 (used for embedding)."""
        trainer = QwenImageNetworkTrainer()
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        timesteps = t * 1000.0
        sigma, adjusted_timesteps = trainer._map_continuous_t_to_sigma_and_timesteps(t, timesteps)

        sigma_embed = adjusted_timesteps / 1000.0
        torch.testing.assert_close(sigma, sigma_embed)

        # Verify range [0.001, 1.0]
        self.assertGreaterEqual(sigma.min().item(), 0.001 - 1e-7)
        self.assertLessEqual(sigma.max().item(), 1.0 + 1e-7)

    def test_qwen_timestep_range_boundaries(self):
        """Verify exact boundary values: t=0 → sigma=0.001, t=1 → sigma=1.0."""
        trainer = QwenImageNetworkTrainer()
        t = torch.tensor([0.0, 1.0])
        timesteps = t * 1000.0
        sigma, adjusted_timesteps = trainer._map_continuous_t_to_sigma_and_timesteps(t, timesteps)

        self.assertAlmostEqual(sigma[0].item(), 0.001, places=6)
        self.assertAlmostEqual(sigma[1].item(), 1.0, places=6)
        self.assertAlmostEqual(adjusted_timesteps[0].item(), 1.0, places=4)
        self.assertAlmostEqual(adjusted_timesteps[1].item(), 1000.0, places=4)

    def test_base_trainer_preserves_legacy_plus_one(self):
        """Base NetworkTrainer still adds +1 to timesteps (legacy behavior)."""
        trainer = NetworkTrainer()
        t = torch.tensor([0.5])
        timesteps = t * 1000.0  # 500.0
        sigma, adjusted_timesteps = trainer._map_continuous_t_to_sigma_and_timesteps(t, timesteps)

        self.assertEqual(sigma.item(), 0.5)  # unchanged
        self.assertEqual(adjusted_timesteps.item(), 501.0)  # +1

    def test_qwen_get_noisy_model_input_uses_mapped_sigma(self):
        """Integration: noisy_model_input from the real method uses the mapped sigma.

        Uses latents=0 and noise=1 so noisy = (1-sigma)*0 + sigma*1 = sigma,
        making the mixing sigma directly readable from the output tensor.
        """
        trainer = QwenImageNetworkTrainer()
        noise_scheduler = FlowMatchDiscreteScheduler(shift=1.0, reverse=True, solver="euler")

        # 4D image latents (not 5D video)
        latents = torch.zeros(1, 16, 8, 8)
        noise = torch.ones(1, 16, 8, 8)

        args = SimpleNamespace(
            timestep_sampling="uniform",
            discrete_flow_shift=1.0,
            sigmoid_scale=1.0,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.0,
            min_timestep=None,
            max_timestep=None,
            preserve_distribution_shape=False,
        )
        # Pass org_timesteps=[0.5] to force t=0.5
        noisy_model_input, returned_timesteps = trainer.get_noisy_model_input_and_timesteps(
            args, noise, latents, timesteps=[0.5], noise_scheduler=noise_scheduler, device=torch.device("cpu"), dtype=torch.float32
        )

        # With latents=0 and noise=1: noisy[i] == sigma at every element
        actual_sigma_mix = noisy_model_input[0, 0, 0, 0].item()
        actual_sigma_embed = returned_timesteps[0].item() / 1000.0

        # They must match
        self.assertAlmostEqual(actual_sigma_mix, actual_sigma_embed, places=5)
        # And both should be in [0.001, 1.0]
        self.assertGreaterEqual(actual_sigma_mix, 0.001 - 1e-7)
        self.assertLessEqual(actual_sigma_mix, 1.0 + 1e-7)


if __name__ == "__main__":
    unittest.main()
