"""Manual FLUX.2 block-swap smoke for the prior-teacher restore path.

Default test collection skips this file. To replay against a real DiT checkpoint:

    ./venv314/bin/python -m pytest -q tests/manual/test_flux2_block_swap_smoke.py \
      --run-manual \
      --model-path /path/to/FLUX2-Klein-Base-9B.safetensors \
      --model-version klein-base-9b \
      --blocks-to-swap 4 \
      --compile-blocks
"""

import pytest
import torch

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.hv_train_network import NetworkTrainer
from musubi_tuner.modules.lora_ema_teacher import LoRAEmaTeacher
from musubi_tuner.networks import lora_flux_2
from musubi_tuner.utils import model_utils


def _flush_offloader(offloader):
    for block_idx in list(getattr(offloader, "futures", {}).keys()):
        offloader.wait_for_block(block_idx)


def _first_double_qkv_device(model):
    block = model.double_blocks[0]
    block = getattr(block, "_orig_mod", block)
    return block.img_attn.qkv.weight.device


class _ManualAccelerator:
    def unwrap_model(self, model):
        return model


def _inputs(params, device, *, requires_grad=False):
    batch = 1
    img_seq = 4
    txt_seq = 3
    x = torch.randn(batch, img_seq, params.in_channels, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
    x_ids = torch.zeros(batch, img_seq, len(params.axes_dim), device=device)
    x_ids[..., 0] = torch.arange(img_seq, device=device, dtype=torch.float32)
    ctx = torch.randn(batch, txt_seq, params.context_in_dim, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
    ctx_ids = torch.zeros(batch, txt_seq, len(params.axes_dim), device=device)
    ctx_ids[..., 0] = torch.arange(txt_seq, device=device, dtype=torch.float32)
    timesteps = torch.tensor([0.5], device=device)
    return x, x_ids, timesteps, ctx, ctx_ids


def test_flux2_block_swap_prior_teacher_restore_smoke(pytestconfig):
    if not pytestconfig.getoption("--run-manual"):
        pytest.skip("manual smoke: pass --run-manual and --model-path to run")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model_path = pytestconfig.getoption("--model-path")
    if not model_path:
        pytest.skip("manual smoke requires --model-path /path/to/dit.safetensors")

    device = torch.device("cuda")
    model_version = flux2_utils.resolve_model_version(pytestconfig.getoption("--model-version"))
    model_info = flux2_utils.FLUX2_MODEL_INFO[model_version]
    blocks_to_swap = int(pytestconfig.getoption("--blocks-to-swap"))

    model = flux2_utils.load_flow_model(
        device=device,
        model_version_info=model_info,
        dit_path=model_path,
        attn_mode="torch",
        split_attn=False,
        loading_device="cpu",
        dit_weight_dtype=torch.bfloat16,
        fp8_scaled=False,
        disable_numpy_memmap=False,
    )
    model.requires_grad_(False)
    model.enable_block_swap(blocks_to_swap, device, supports_backward=True, use_pinned_memory=False)
    model.move_to_device_except_swap_blocks(device)

    network = lora_flux_2.create_arch_network(1.0, 4, 4, None, [], model, use_dora=True)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.to(device)
    model.enable_gradient_checkpointing(False)
    model.prepare_block_swap_before_forward()

    if pytestconfig.getoption("--compile-blocks"):
        for blocks in (model.double_blocks, model.single_blocks):
            for block in blocks:
                model_utils.disable_linear_from_compile(block)
        for blocks in (model.double_blocks, model.single_blocks):
            for i, block in enumerate(blocks):
                blocks[i] = torch.compile(block, mode="default")

    trainer = NetworkTrainer()
    trainer.blocks_to_swap = blocks_to_swap
    ema = LoRAEmaTeacher(decay=0.9)
    ema.init_from(network)

    model.train()
    network.train()
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        with torch.no_grad(), ema.apply_to(network):
            prior_out = model(*_inputs(model_info.params, device), guidance=None)
            assert prior_out.shape == (1, 4, model_info.params.in_channels)

    _flush_offloader(model.offloader_double)
    _flush_offloader(model.offloader_single)
    torch.cuda.synchronize()
    assert _first_double_qkv_device(model).type == "cpu"

    trainer.restore_block_swap_after_no_grad_forward(_ManualAccelerator(), model)
    assert _first_double_qkv_device(model).type == "cuda"

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(*_inputs(model_info.params, device, requires_grad=True), guidance=None)
        loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    assert any(param.grad is not None for param in network.parameters() if param.requires_grad)
