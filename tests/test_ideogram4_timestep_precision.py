"""Tier 2: Ideogram 4 fp32 timestep conditioning (corrected) vs legacy bf16.

The DiT forward cast ``t = t.to(param_dtype)`` is a lossy round-trip (Ideogram4EmbedScalar re-upcasts
to fp32 and a 1e4 sinusoidal scale amplifies the rounding; standalone-probe embedding cosine ~0.79 vs
fp32). ``--ideogram4_fp32_timestep`` keeps t in fp32; the regime is recorded in LoRA metadata so
generation reproduces it. Default stays legacy bf16 (behavior-preserving; the A/B gates the flip).
"""

import argparse
import logging
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import musubi_tuner.ideogram4_generate_image as gen
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer, ideogram4_setup_parser
from musubi_tuner.training import trainer_base


# ----------------------------------------------------------------------------------------------------
# Model wiring: the flag controls ONLY the dtype of t reaching t_embedding (via a forward_pre_hook spy).
# The model is bf16 so param_dtype != fp32 and the two regimes are distinguishable (on an fp32 model
# both would feed fp32). A dtype-spy is used rather than output-differs: the embedding output is cast
# back to bf16 regardless, so a tiny-model output can round equal and silently pass for the wrong reason.
# ----------------------------------------------------------------------------------------------------


def _tiny_bf16_model():
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
    return cfg, Ideogram4Transformer(cfg).to(torch.bfloat16)


def _t_embedding_input_dtype(model, cfg):
    captured = {}

    def hook(_module, inputs):
        captured["dtype"] = inputs[0].dtype

    handle = model.t_embedding.register_forward_pre_hook(hook)
    try:
        latents = torch.randn(1, cfg.in_channels, 2, 2)
        noise = torch.randn(1, cfg.in_channels, 2, 2)
        text_features = [torch.randn(3, cfg.llm_features_dim)]
        ideogram4_flow_matching_target(
            model, latents, text_features, noise, torch.tensor([0.5]), network_dtype=torch.bfloat16, device="cpu"
        )
    finally:
        handle.remove()
    return captured["dtype"]


def test_default_is_legacy_bf16():
    _, model = _tiny_bf16_model()
    assert model.fp32_timestep is False  # behavior-preserving default


def test_legacy_regime_feeds_bf16_to_time_embedding():
    cfg, model = _tiny_bf16_model()
    model.fp32_timestep = False
    assert _t_embedding_input_dtype(model, cfg) == torch.bfloat16


def test_fp32_regime_feeds_fp32_to_time_embedding():
    cfg, model = _tiny_bf16_model()
    model.fp32_timestep = True
    assert _t_embedding_input_dtype(model, cfg) == torch.float32


# ----------------------------------------------------------------------------------------------------
# Trainer: parser flag + metadata stamp (so generation can reproduce the trained regime).
# ----------------------------------------------------------------------------------------------------


def test_setup_parser_adds_fp32_timestep_flag():
    parser = ideogram4_setup_parser(argparse.ArgumentParser())
    assert parser.parse_args([]).ideogram4_fp32_timestep is False  # store_true default = legacy
    assert parser.parse_args(["--ideogram4_fp32_timestep"]).ideogram4_fp32_timestep is True


@pytest.mark.parametrize("flag", [True, False])
def test_extra_metadata_records_timestep_regime(monkeypatch, flag):
    monkeypatch.setattr(trainer_base.NetworkTrainer, "extra_metadata", lambda self, args: {})
    trainer = Ideogram4NetworkTrainer()
    md = trainer.extra_metadata(SimpleNamespace(dit="/m/x.safetensors", ideogram4_fp32_timestep=flag))
    assert md["ss_ideogram4_fp32_timestep"] is flag


def test_extra_metadata_defaults_timestep_regime_to_false(monkeypatch):
    # Defensive getattr: an args namespace missing the attr stamps legacy bf16, not a crash.
    monkeypatch.setattr(trainer_base.NetworkTrainer, "extra_metadata", lambda self, args: {})
    trainer = Ideogram4NetworkTrainer()
    md = trainer.extra_metadata(SimpleNamespace(dit="/m/x.safetensors"))
    assert md["ss_ideogram4_fp32_timestep"] is False


# ----------------------------------------------------------------------------------------------------
# Generator: regime resolution. Explicit flag wins; else inherit from LoRA metadata; else legacy bf16;
# disagreeing stacked LoRAs warn and fall back conservatively.
# ----------------------------------------------------------------------------------------------------


def _write_lora_with_regime(path, regime_str):
    metadata = {} if regime_str is None else {"ss_ideogram4_fp32_timestep": regime_str}
    save_file({"dummy": torch.zeros(1)}, path, metadata=metadata)


def test_resolver_explicit_flag_wins():
    assert gen._resolve_generation_timestep_regime(True, None) is True
    assert gen._resolve_generation_timestep_regime(False, ["ignored.safetensors"]) is False


def test_resolver_defaults_legacy_with_no_flag_no_metadata():
    assert gen._resolve_generation_timestep_regime(None, None) is False
    assert gen._resolve_generation_timestep_regime(None, []) is False


@pytest.mark.parametrize("stamp,expected", [("True", True), ("False", False)])
def test_resolver_inherits_lora_metadata(tmp_path, stamp, expected):
    path = str(tmp_path / "lora.safetensors")
    _write_lora_with_regime(path, stamp)
    assert gen._resolve_generation_timestep_regime(None, [path]) is expected


def test_resolver_warns_and_falls_back_on_disagreement(tmp_path, caplog):
    a = str(tmp_path / "a.safetensors")
    b = str(tmp_path / "b.safetensors")
    _write_lora_with_regime(a, "True")
    _write_lora_with_regime(b, "False")
    with caplog.at_level(logging.WARNING):
        result = gen._resolve_generation_timestep_regime(None, [a, b])
    assert result is False  # conservative fallback
    assert any("disagree" in r.message.lower() for r in caplog.records)
