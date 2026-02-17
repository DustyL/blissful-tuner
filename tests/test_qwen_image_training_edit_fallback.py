import contextlib
import unittest
from types import SimpleNamespace

import torch

from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer


class _FakeAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()


class _DummyQwenImageModel(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_kwargs) -> torch.Tensor:
        # Return zeros with the expected packed-latents shape: (B, Seq, C)
        return torch.zeros_like(hidden_states)


class TestQwenImageEditFallback(unittest.TestCase):
    def _make_common_inputs(self):
        bsz, channels, height, width = 1, 16, 4, 4
        latents = torch.zeros(bsz, channels, 1, height, width, dtype=torch.float32)
        noise = torch.zeros_like(latents)
        noisy_model_input = torch.zeros_like(latents)
        timesteps = torch.zeros(bsz, dtype=torch.float32)
        batch = {
            "latents": latents,
            "vl_embed": [torch.zeros(2, 8, dtype=torch.float32)],  # (txt_len, dim)
        }
        return latents, noise, noisy_model_input, timesteps, batch

    def test_edit_batch_without_controls_errors_by_default(self):
        trainer = QwenImageNetworkTrainer()
        trainer.is_edit = True
        trainer.is_layered = False

        latents, noise, noisy_model_input, timesteps, batch = self._make_common_inputs()

        args = SimpleNamespace(
            is_layered=False,
            remove_first_image_from_target=False,
            split_attn=False,
            gradient_checkpointing=False,
        )
        accelerator = _FakeAccelerator()

        with self.assertRaises(ValueError) as ctx:
            trainer.call_dit(
                args=args,
                accelerator=accelerator,
                transformer=_DummyQwenImageModel(),
                latents=latents,
                batch=batch,
                noise=noise,
                noisy_model_input=noisy_model_input,
                timesteps=timesteps,
                network_dtype=torch.float32,
            )
        self.assertIn("latents_control_0", str(ctx.exception))

    def test_edit_batch_without_controls_can_fallback_when_allowed(self):
        trainer = QwenImageNetworkTrainer()
        trainer.is_edit = True
        trainer.is_layered = False

        latents, noise, noisy_model_input, timesteps, batch = self._make_common_inputs()

        args = SimpleNamespace(
            is_layered=False,
            remove_first_image_from_target=False,
            split_attn=False,
            gradient_checkpointing=False,
            allow_edit_fallback_to_t2i=True,
            model_version="edit-2511",
        )
        accelerator = _FakeAccelerator()

        model_pred, target = trainer.call_dit(
            args=args,
            accelerator=accelerator,
            transformer=_DummyQwenImageModel(),
            latents=latents,
            batch=batch,
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            network_dtype=torch.float32,
        )

        self.assertEqual(model_pred.shape, latents.shape)
        self.assertEqual(target.shape, latents.shape)


if __name__ == "__main__":
    unittest.main()
