# Runtime LoRA application in standalone Ideogram 4 generation (backlog P2-7).
#
# The fp8-prequantized base CANNOT be merged into (networks/lora.py refuses by design — float-merged
# values written back as fp8 without inverse re-scaling corrupt the checkpoint), so the generation
# script runtime-applies adapters exactly like sampling-during-training: the network monkey-patches
# the conditional DiT's module forwards and stays live through the denoise. The unconditional DiT
# stays adapter-free (training never touches it — asymmetric CFG).

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import musubi_tuner.networks.lora_ideogram4 as lora_ideogram4
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target
from musubi_tuner.ideogram4_generate_image import apply_lora_weights_runtime, setup_parser
from musubi_tuner.networks import lokr

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _tiny_model():
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
    torch.manual_seed(0)  # identical base weights across instances
    return Ideogram4Transformer(cfg), cfg


def _forward(model, cfg):
    g = torch.Generator().manual_seed(7)
    latents = torch.randn(1, 128, 2, 2, generator=g)
    noise = torch.randn(1, 128, 2, 2, generator=g)
    text = [torch.randn(3, cfg.llm_features_dim, generator=g)]
    t = torch.tensor([0.5])
    with torch.no_grad():
        pred, _ = ideogram4_flow_matching_target(model, latents, text, noise, t, network_dtype=DTYPE, device="cpu")
    return pred


def _save_lora_checkpoint(tmp_path, use_dora=False, name="lora.safetensors"):
    # Build on a THROWAWAY model instance; the test then applies the saved file to a fresh one.
    donor, _ = _tiny_model()
    net = lora_ideogram4.create_arch_network(1.0, 4, 4.0, None, None, donor, use_dora=str(use_dora))
    net.apply_to(None, donor, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for m in net.unet_loras:
            m.lora_up.weight.normal_(0.0, 0.05)  # zero-init up => zero delta; randomize so apply is visible
    sd = {k: v.detach().clone().contiguous() for k, v in net.state_dict().items()}
    path = tmp_path / name
    save_file(sd, str(path))
    return str(path)


def _save_lokr_checkpoint(tmp_path, name="lokr.safetensors"):
    donor, _ = _tiny_model()
    net = lokr.create_arch_network(1.0, 4, 4.0, None, None, donor, architecture="i4")
    net.apply_to(None, donor, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0.0, 0.05)
    sd = {k: v.detach().clone().contiguous() for k, v in net.state_dict().items()}
    path = tmp_path / name
    save_file(sd, str(path))
    return str(path)


def test_parser_has_runtime_lora_flags():
    parser = setup_parser()
    options = [opt for action in parser._actions for opt in action.option_strings]
    assert "--lora_weight" in options and "--lora_multiplier" in options


def test_runtime_apply_changes_output_and_multiplier_zero_is_noop(tmp_path):
    path = _save_lora_checkpoint(tmp_path)

    base_model, cfg = _tiny_model()
    base_pred = _forward(base_model, cfg)

    applied_model, _ = _tiny_model()
    nets = apply_lora_weights_runtime(applied_model, [path], None, DEVICE, DTYPE)  # default multiplier 1.0
    assert len(nets) == 1
    applied_pred = _forward(applied_model, cfg)
    assert not torch.allclose(applied_pred, base_pred), "runtime-applied LoRA must change the forward"

    zero_model, _ = _tiny_model()
    apply_lora_weights_runtime(zero_model, [path], [0.0], DEVICE, DTYPE)
    zero_pred = _forward(zero_model, cfg)
    assert torch.equal(zero_pred, base_pred), "multiplier 0 must be a bitwise no-op"


def test_dora_checkpoint_autodetected_and_applies(tmp_path):
    path = _save_lora_checkpoint(tmp_path, use_dora=True, name="dora.safetensors")

    base_model, cfg = _tiny_model()
    base_pred = _forward(base_model, cfg)

    model, _ = _tiny_model()
    nets = apply_lora_weights_runtime(model, [path], [1.0], DEVICE, DTYPE)
    # DoRA reconstructed from the checkpoint's flag buffer, not from any caller kwarg.
    assert all(m.use_dora for m in nets[0].unet_loras)
    pred = _forward(model, cfg)
    assert torch.isfinite(pred).all()
    assert not torch.allclose(pred, base_pred)


def test_lokr_checkpoint_applies(tmp_path):
    path = _save_lokr_checkpoint(tmp_path)

    base_model, cfg = _tiny_model()
    base_pred = _forward(base_model, cfg)

    model, _ = _tiny_model()
    nets = apply_lora_weights_runtime(model, [path], [1.0], DEVICE, DTYPE)
    assert len(nets) == 1
    pred = _forward(model, cfg)
    assert torch.isfinite(pred).all()
    assert not torch.allclose(pred, base_pred)


def _save_loha_checkpoint(tmp_path, name="loha.safetensors"):
    from musubi_tuner.networks import loha

    donor, _ = _tiny_model()
    net = loha.create_arch_network(1.0, 4, 4.0, None, None, donor, architecture="i4")
    net.apply_to(None, donor, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0.0, 0.05)
    sd = {k: v.detach().clone().contiguous() for k, v in net.state_dict().items()}
    path = tmp_path / name
    save_file(sd, str(path))
    return str(path)


def test_multiplier_zero_is_noop_for_all_families(tmp_path):
    # Locks the IEEE-754-exact zero-delta invariant per family (review note on PR #23): LoRA
    # short-circuits, LoHa scales the delta term by multiplier, LoKr folds multiplier into
    # get_weight — all three must reproduce the base forward bitwise at multiplier 0.
    base_model, cfg = _tiny_model()
    base_pred = _forward(base_model, cfg)

    checkpoints = {
        "lora": _save_lora_checkpoint(tmp_path),
        "loha": _save_loha_checkpoint(tmp_path),
        "lokr": _save_lokr_checkpoint(tmp_path),
    }
    for family, path in checkpoints.items():
        model, _ = _tiny_model()
        apply_lora_weights_runtime(model, [path], [0.0], DEVICE, DTYPE)
        pred = _forward(model, cfg)
        assert torch.equal(pred, base_pred), f"{family}: multiplier 0 must be a bitwise no-op"


def test_unknown_checkpoint_rejected(tmp_path):
    path = tmp_path / "garbage.safetensors"
    save_file({"something.weird_key": torch.zeros(2, 2)}, str(path))
    model, _ = _tiny_model()
    with pytest.raises(ValueError, match="unknown"):
        apply_lora_weights_runtime(model, [str(path)], None, DEVICE, DTYPE)


def test_production_parser_keeps_sample_guidance_exclusion():
    # Companion to PR #15's guard: adding runtime-LoRA flags must not have smuggled the
    # training-only sample-guidance knob into the production parser.
    parser = setup_parser()
    options = [opt for action in parser._actions for opt in action.option_strings]
    assert "--ideogram4_sample_guidance" not in options


def test_namespace_main_args_compatible():
    # The helper consumes args.lora_weight/args.lora_multiplier exactly as argparse produces them.
    parser = setup_parser()
    ns = parser.parse_args(
        [
            "--dit",
            "x",
            "--unconditional_dit",
            "x",
            "--vae",
            "x",
            "--text_encoder",
            "x",
            "--text_encoder_config",
            "x",
            "--tokenizer",
            "x",
            "--prompt",
            "p",
            "--save_path",
            "o.png",
            "--lora_weight",
            "a.safetensors",
            "b.safetensors",
            "--lora_multiplier",
            "1.0",
            "0.7",
        ]
    )
    assert ns.lora_weight == ["a.safetensors", "b.safetensors"]
    assert ns.lora_multiplier == [1.0, 0.7]
    assert isinstance(ns, SimpleNamespace) or hasattr(ns, "lora_weight")
