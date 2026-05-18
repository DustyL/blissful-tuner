import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn

from musubi_tuner.hv_train_network import NetworkTrainer
from musubi_tuner.modules.lora_ema_teacher import LoRAEmaTeacher
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.networks.lora import LoRAModule


class _FakeAccelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()


class _FakeTransformer:
    def __init__(self):
        self.prepare_calls = 0

    def prepare_block_swap_before_forward(self):
        self.prepare_calls += 1


class TestBlockSwapPriorRestoreHelper(unittest.TestCase):
    def test_restore_is_noop_when_block_swap_disabled(self):
        trainer = NetworkTrainer()
        trainer.blocks_to_swap = 0
        transformer = _FakeTransformer()

        trainer.restore_block_swap_after_no_grad_forward(_FakeAccelerator(), transformer)

        self.assertEqual(transformer.prepare_calls, 0)

    def test_restore_prepares_transformer_when_block_swap_enabled(self):
        trainer = NetworkTrainer()
        trainer.blocks_to_swap = 4
        transformer = _FakeTransformer()

        trainer.restore_block_swap_after_no_grad_forward(_FakeAccelerator(), transformer)

        self.assertEqual(transformer.prepare_calls, 1)


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))


class _TinyBlockSwapTransformer(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock() for _ in range(4)])
        self.blocks_to_swap = 1
        self.offloader = ModelOffloader("tiny", self.blocks, len(self.blocks), self.blocks_to_swap, True, device, False)
        self.requires_grad_(False)

    def prepare_block_swap_before_forward(self):
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_inference(self):
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()

    def forward(self, x):
        for block_idx, block in enumerate(self.blocks):
            self.offloader.wait_for_block(block_idx)
            x = block(x)
            self.offloader.submit_move_blocks_forward(self.blocks, block_idx)
        return x


class _TinyLoraNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))


class _FakeSampleTransformer:
    def __init__(self):
        self.inference_calls = 0
        self.training_calls = 0
        self.in_inference_mode = False

    def switch_block_swap_for_inference(self):
        self.inference_calls += 1
        self.in_inference_mode = True

    def switch_block_swap_for_training(self):
        self.training_calls += 1
        self.in_inference_mode = False


class _FailingSampleTrainer(NetworkTrainer):
    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
        self.assert_sample_state(transformer)
        torch.rand(4)
        raise RuntimeError("sample failure")

    @staticmethod
    def assert_sample_state(transformer):
        if not transformer.in_inference_mode:
            raise AssertionError("sample_image_inference must run after switch_block_swap_for_inference()")


class TestSampleImagesCleanup(unittest.TestCase):
    def test_exception_restores_rng_and_training_block_swap_mode(self):
        torch.manual_seed(123)
        expected_rng_state = torch.get_rng_state().clone()
        transformer = _FakeSampleTransformer()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=tmpdir,
                sample_at_first=False,
                sample_every_n_steps=1,
                sample_every_n_epochs=None,
                sample_prompts="sample_prompts.txt",
            )

            trainer = _FailingSampleTrainer()
            with self.assertRaisesRegex(RuntimeError, "sample failure"):
                trainer.sample_images(
                    _FakeAccelerator(),
                    args,
                    epoch=None,
                    steps=1,
                    vae=None,
                    transformer=transformer,
                    sample_parameters=[{}],
                    dit_dtype=torch.float32,
                    trigger="step",
                )

        self.assertEqual(transformer.inference_calls, 1)
        self.assertEqual(transformer.training_calls, 1)
        self.assertFalse(transformer.in_inference_mode)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng_state))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for block-swap device restore regression")
class TestBlockSwapPriorRestoreCuda(unittest.TestCase):
    def test_no_grad_forward_is_restored_for_next_training_forward(self):
        device = torch.device("cuda")
        trainer = NetworkTrainer()
        trainer.blocks_to_swap = 1
        transformer = _TinyBlockSwapTransformer(device)
        transformer.prepare_block_swap_before_forward()

        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cuda")
        self.assertEqual(transformer.blocks[-1].linear.weight.device.type, "cpu")

        with torch.no_grad():
            x = torch.randn(2, 4, device=device)
            transformer(x)
        transformer.offloader.wait_for_block(len(transformer.blocks) - 1)
        torch.cuda.synchronize()

        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cpu")
        self.assertEqual(transformer.blocks[-1].linear.weight.device.type, "cuda")

        trainer.restore_block_swap_after_no_grad_forward(_FakeAccelerator(), transformer)

        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cuda")
        self.assertEqual(transformer.blocks[-1].linear.weight.device.type, "cpu")

    def test_sample_inference_switch_restores_training_layout(self):
        device = torch.device("cuda")
        transformer = _TinyBlockSwapTransformer(device)
        transformer.prepare_block_swap_before_forward()

        transformer.switch_block_swap_for_inference()
        with torch.no_grad():
            x = torch.randn(2, 4, device=device)
            transformer(x)

        transformer.switch_block_swap_for_training()

        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cuda")
        self.assertEqual(transformer.blocks[-1].linear.weight.device.type, "cpu")

        x = torch.randn(2, 4, device=device, requires_grad=True)
        out = transformer(x)
        out.sum().backward()
        torch.cuda.synchronize()

        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cuda")

    def test_ema_update_and_apply_do_not_change_block_swap_layout(self):
        device = torch.device("cuda")
        transformer = _TinyBlockSwapTransformer(device)
        transformer.prepare_block_swap_before_forward()
        network = _TinyLoraNetwork().to(device)

        ema = LoRAEmaTeacher(decay=0.9)
        ema.init_from(network)

        ema.update(network)
        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cuda")
        self.assertEqual(transformer.blocks[-1].linear.weight.device.type, "cpu")

        with ema.apply_to(network):
            self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cuda")
            self.assertEqual(transformer.blocks[-1].linear.weight.device.type, "cpu")

        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cuda")
        self.assertEqual(transformer.blocks[-1].linear.weight.device.type, "cpu")

    def test_ema_prior_context_restore_then_student_forward_with_lora(self):
        device = torch.device("cuda")
        trainer = NetworkTrainer()
        trainer.blocks_to_swap = 1
        transformer = _TinyBlockSwapTransformer(device)
        lora = LoRAModule("lora_tiny_block0_linear", transformer.blocks[0].linear, 1.0, 2, 2, use_dora=True)
        lora.apply_to()
        lora.to(device)
        transformer.prepare_block_swap_before_forward()

        ema = LoRAEmaTeacher(decay=0.9)
        ema.init_from(lora)

        with torch.no_grad(), ema.apply_to(lora):
            x = torch.randn(2, 4, device=device)
            transformer(x)
        transformer.offloader.wait_for_block(len(transformer.blocks) - 1)
        torch.cuda.synchronize()

        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cpu")

        trainer.restore_block_swap_after_no_grad_forward(_FakeAccelerator(), transformer)

        self.assertEqual(transformer.blocks[0].linear.weight.device.type, "cuda")

        x = torch.randn(2, 4, device=device, requires_grad=True)
        out = transformer(x)
        out.sum().backward()
        torch.cuda.synchronize()

        self.assertIsNotNone(lora.lora_up.weight.grad)


if __name__ == "__main__":
    unittest.main()
