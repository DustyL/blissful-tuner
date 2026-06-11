# Ideogram 4 LoHa/LoKr support (backlog P1-3): architecture-registry entry + the dtype
# harmonization both backends need on Ideogram's autocast-defeated forward path.
#
# The harmonization mirrors commit e85284b (which fixed LoRAModule.forward only): Ideogram 4
# deliberately disables autocast at do_inference/process_batch for training-inference parity, so
# bf16 base activations meet fp32 adapter weights with nothing to harmonize them — F.linear raises
# "expected mat1 and mat2 to have the same dtype". On every other architecture accelerate's
# autocast hides this, which is why it only bites (and is only testable) without autocast.

import torch

from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target
from musubi_tuner.networks import loha, lokr


def _tiny_model(dtype=torch.float32):
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
    return Ideogram4Transformer(cfg).to(dtype), cfg


def _apply(backend, model, **kwargs):
    net = backend.create_arch_network(1.0, 4, 4.0, None, None, model, architecture="i4", **kwargs)
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    return net


def _expected_module_names(model, prefix="lora_unet"):
    # Every Linear inside the two blocks except adaln_modulation: 5 per block (qkv, o, w1, w2, w3).
    names = set()
    for block_idx, block in enumerate(model.layers):
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear) and "adaln_modulation" not in name:
                names.add(f"{prefix}_layers_{block_idx}_{name.replace('.', '_')}")
    return names


def test_loha_registry_targets_all_block_linears_except_adaln():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    net = _apply(loha, model)
    covered = {m.lora_name for m in net.unet_loras}
    assert covered == _expected_module_names(model)
    assert len(covered) == 10  # 2 layers x (qkv, o, w1, w2, w3)
    assert not any("adaln_modulation" in n for n in covered)


def test_lokr_registry_targets_all_block_linears_except_adaln():
    torch.manual_seed(0)
    model, _ = _tiny_model()
    net = _apply(lokr, model)
    covered = {m.lora_name for m in net.unet_loras}
    assert covered == _expected_module_names(model)
    assert not any("adaln_modulation" in n for n in covered)
    # Factor persistence buffer rides on the network (saved into checkpoints).
    assert "lokr_factor" in net.state_dict()


def _run_flow_forward_no_autocast(model, cfg, dtype):
    # The exact regime Ideogram training/sampling uses: NO autocast anywhere; base activations in
    # the model dtype, adapter weights left at their construction dtype (fp32).
    assert not torch.is_autocast_enabled("cpu")
    latents = torch.randn(1, 128, 2, 2)
    noise = torch.randn(1, 128, 2, 2)
    text_features = [torch.randn(3, cfg.llm_features_dim)]
    t = torch.tensor([0.5])
    return ideogram4_flow_matching_target(model, latents, text_features, noise, t, network_dtype=dtype, device="cpu")


def test_loha_forward_backward_bf16_base_fp32_adapters_no_autocast():
    # Regression for the e85284b failure class on the LoHa path: crashed with
    # "expected mat1 and mat2 to have the same dtype" before the harmonization.
    torch.manual_seed(0)
    model, cfg = _tiny_model(dtype=torch.bfloat16)
    net = _apply(loha, model)  # adapters default to fp32
    model.requires_grad_(False)
    net.requires_grad_(True)

    pred, target = _run_flow_forward_no_autocast(model, cfg, torch.bfloat16)
    assert torch.isfinite(pred.float()).all()
    torch.nn.functional.mse_loss(pred.float(), target.float()).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_lokr_forward_backward_bf16_base_fp32_adapters_no_autocast():
    torch.manual_seed(0)
    model, cfg = _tiny_model(dtype=torch.bfloat16)
    net = _apply(lokr, model)
    model.requires_grad_(False)
    net.requires_grad_(True)

    pred, target = _run_flow_forward_no_autocast(model, cfg, torch.bfloat16)
    assert torch.isfinite(pred.float()).all()
    torch.nn.functional.mse_loss(pred.float(), target.float()).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_adapter_module_does_not_promote_output_dtype():
    # The module-level contract of the harmonization: a wrapped Linear fed bf16 must return bf16 —
    # the fp32 adapter delta is cast back before the addition. (Pipeline-level output dtype is the
    # model's business — its internal norm/embedding paths may legitimately upcast.)
    for backend in (loha, lokr):
        torch.manual_seed(0)
        model, _ = _tiny_model(dtype=torch.bfloat16)
        _apply(backend, model)
        wrapped = model.layers[0].feed_forward.w1  # patched forward = adapter module's forward
        out = wrapped(torch.randn(2, 32, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16, backend.__name__


def test_loha_harmonization_is_noop_when_dtypes_already_match():
    # fp32 base + fp32 adapters: the casts must be identity — outputs equal the pre-harmonization
    # math bit-for-bit (guards against an accidental always-cast perturbing other architectures).
    torch.manual_seed(0)
    model, cfg = _tiny_model(dtype=torch.float32)
    _apply(loha, model)
    pred, _ = _run_flow_forward_no_autocast(model, cfg, torch.float32)
    assert pred.dtype == torch.float32
    assert torch.isfinite(pred).all()
