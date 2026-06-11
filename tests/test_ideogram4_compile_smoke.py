# torch.compile smoke for the Ideogram 4 trainer (backlog P1-1).
#
# The compile hook compiles each transformer block individually (mirrors flux_2_train_network.py);
# the root forward stays eager so the fp8 monkey-patched Linear forwards and the
# disable_accelerate_forward_autocast root-level forward swap during sampling keep working. These
# tests cover the hook wiring, the lifted fail-fast, and an eager-vs-compiled parity smoke with a
# DoRA LoRA applied (the CPU half of the 4-way compile x fp8 x DoRA x masked-loss interaction; the
# fp8 half needs real weights and lives in tests/manual/measure_ideogram4_compile.py).

from types import SimpleNamespace

import pytest
import torch

import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer


def _tiny_model():
    # Mirrors tests/test_ideogram4_gradient_checkpointing.py: in_channels=128 + mrope_section=(2,2,2)
    # so ideogram4_flow_matching_target works on (1, 128, 2, 2) latents.
    cfg = Ideogram4Config(
        emb_dim=32,
        num_layers=2,
        num_heads=2,
        intermediate_size=48,
        adanln_dim=16,
        in_channels=128,
        llm_features_dim=16,
        mrope_section=(2, 2, 2),
    )
    return Ideogram4Transformer(cfg), cfg


def _compile_args(**overrides):
    # Production defaults from training/parser_common.py:_add_compile_and_dynamo_args.
    base = dict(
        compile=True,
        compile_backend="inductor",
        compile_mode="default",
        compile_dynamic=None,
        compile_fullgraph=False,
        compile_cache_size_limit=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _inputs(cfg):
    latents = torch.randn(1, 128, 2, 2)
    noise = torch.randn(1, 128, 2, 2)
    text_features = [torch.randn(3, cfg.llm_features_dim)]
    t = torch.tensor([0.5])
    return latents, noise, text_features, t


def test_handle_model_specific_args_accepts_compile():
    trainer = Ideogram4NetworkTrainer()
    trainer.handle_model_specific_args(SimpleNamespace(compile=True, mixed_precision="bf16"))  # no raise


def test_compile_hook_compiles_each_block_not_the_root():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    trainer = Ideogram4NetworkTrainer()
    out = trainer.compile_transformer(_compile_args(), model)
    assert out is model  # compiled in place, same root object
    # Every block is an OptimizedModule wrapper; the root module is NOT compiled (its forward must
    # stay swappable by disable_accelerate_forward_autocast during sampling).
    for block in model.layers:
        assert hasattr(block, "_orig_mod"), "each transformer block should be torch.compile-wrapped"
    assert not hasattr(model, "_orig_mod")


def test_compiled_forward_backward_parity_with_dora_lora():
    torch.manual_seed(0)
    eager_model, cfg = _tiny_model()

    # Apply a DoRA LoRA exactly like training does, BEFORE compile (matches trainer_base ordering:
    # network.apply_to happens before the compile_transformer call at prepare time).
    network = lora_ideogram4.create_arch_network(1.0, 4, 8.0, None, None, eager_model, use_dora="True")
    network.apply_to(None, eager_model, apply_text_encoder=False, apply_unet=True)
    eager_model.requires_grad_(False)
    network.requires_grad_(True)

    torch.manual_seed(42)
    latents, noise, text_features, t = _inputs(cfg)
    pred_eager, target = ideogram4_flow_matching_target(
        eager_model, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu"
    )

    trainer = Ideogram4NetworkTrainer()
    try:
        trainer.compile_transformer(_compile_args(), eager_model)
        pred_compiled, _ = ideogram4_flow_matching_target(
            eager_model, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu"
        )
    except Exception as e:  # pragma: no cover - env-specific inductor availability
        pytest.skip(f"torch.compile unavailable in this environment: {e!r}")

    # Same weights, same inputs: compile must not change the math.
    assert torch.allclose(pred_eager, pred_compiled, atol=1e-4), (pred_eager - pred_compiled).abs().max().item()

    # Backward through the compiled blocks reaches the LoRA adapters.
    torch.nn.functional.mse_loss(pred_compiled.float(), target.float()).backward()
    lora_grads = [p.grad for p in network.parameters() if p.grad is not None]
    assert lora_grads, "LoRA parameters should receive gradients through compiled blocks"
    assert all(torch.isfinite(g).all() for g in lora_grads)


def test_compile_composes_with_gradient_checkpointing():
    # Production configs run compile + GC together; the GC gate lives INSIDE the compiled block
    # forward (torch.is_grad_enabled() and self.gradient_checkpointing), so dynamo must trace it.
    torch.manual_seed(0)
    model, cfg = _tiny_model()
    model.enable_gradient_checkpointing()
    trainer = Ideogram4NetworkTrainer()
    try:
        trainer.compile_transformer(_compile_args(), model)
        latents, noise, text_features, t = _inputs(cfg)
        pred, target = ideogram4_flow_matching_target(
            model, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu"
        )
        torch.nn.functional.mse_loss(pred.float(), target.float()).backward()
    except Exception as e:  # pragma: no cover - env-specific inductor availability
        pytest.skip(f"torch.compile unavailable in this environment: {e!r}")
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
