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
import musubi_tuner.ideogram4_train_network as itn
from musubi_tuner.ideogram4.constants import IDEOGRAM4_FP32_TIMESTEP_METADATA_KEY
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer, ideogram4_setup_parser
from musubi_tuner.training import trainer_base

# A timestep that is NOT exactly representable in bf16, so the legacy round-trip is observable (0.5 is
# bf16-exact and would hide it).
_T = 0.1234567


# ----------------------------------------------------------------------------------------------------
# Model wiring: the flag controls the dtype AND the bit-exactness of t reaching t_embedding. A value-spy
# (not just dtype) is used so a `t.to(bf16).to(fp32)` regression -- which would still present fp32 dtype
# -- cannot pass. The model is bf16 so param_dtype != fp32 and the regimes are distinguishable.
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


def _t_reaching_embedding(model, cfg):
    captured = {}

    def hook(_module, inputs):
        captured["t"] = inputs[0].detach().clone()

    handle = model.t_embedding.register_forward_pre_hook(hook)
    try:
        latents = torch.randn(1, cfg.in_channels, 2, 2)
        noise = torch.randn(1, cfg.in_channels, 2, 2)
        text_features = [torch.randn(3, cfg.llm_features_dim)]
        ideogram4_flow_matching_target(
            model, latents, text_features, noise, torch.tensor([_T]), network_dtype=torch.bfloat16, device="cpu"
        )
    finally:
        handle.remove()
    return captured["t"]


def test_default_is_legacy_bf16():
    _, model = _tiny_bf16_model()
    assert model.fp32_timestep is False  # behavior-preserving default


def test_fp32_regime_preserves_timestep_bits():
    cfg, model = _tiny_bf16_model()
    model.fp32_timestep = True
    captured = _t_reaching_embedding(model, cfg)
    assert captured.dtype == torch.float32
    assert torch.equal(captured, torch.tensor([_T], dtype=torch.float32))  # bit-exact, no quantization


def test_legacy_regime_rounds_timestep_to_bf16():
    cfg, model = _tiny_bf16_model()
    model.fp32_timestep = False
    captured = _t_reaching_embedding(model, cfg)
    assert captured.dtype == torch.bfloat16
    assert torch.equal(captured, torch.tensor([_T], dtype=torch.float32).to(torch.bfloat16))
    # and the rounded value really differs from the fp32 original (proves a lossy round-trip happens)
    assert not torch.equal(captured.to(torch.float32), torch.tensor([_T], dtype=torch.float32))


# ----------------------------------------------------------------------------------------------------
# Trainer: parser flag (BooleanOptionalAction), metadata stamp, --no_metadata retention, load seam.
# ----------------------------------------------------------------------------------------------------


def test_setup_parser_fp32_timestep_flag_tri_state():
    parser = ideogram4_setup_parser(argparse.ArgumentParser())
    assert parser.parse_args([]).ideogram4_fp32_timestep is False  # default = legacy
    assert parser.parse_args(["--ideogram4_fp32_timestep"]).ideogram4_fp32_timestep is True
    assert parser.parse_args(["--no-ideogram4_fp32_timestep"]).ideogram4_fp32_timestep is False


@pytest.mark.parametrize("flag", [True, False])
def test_extra_metadata_records_timestep_regime(monkeypatch, flag):
    monkeypatch.setattr(trainer_base.NetworkTrainer, "extra_metadata", lambda self, args: {})
    trainer = Ideogram4NetworkTrainer()
    md = trainer.extra_metadata(SimpleNamespace(dit="/m/x.safetensors", ideogram4_fp32_timestep=flag))
    assert md[IDEOGRAM4_FP32_TIMESTEP_METADATA_KEY] is flag


def test_extra_metadata_defaults_timestep_regime_to_false(monkeypatch):
    monkeypatch.setattr(trainer_base.NetworkTrainer, "extra_metadata", lambda self, args: {})
    trainer = Ideogram4NetworkTrainer()
    md = trainer.extra_metadata(SimpleNamespace(dit="/m/x.safetensors"))
    assert md[IDEOGRAM4_FP32_TIMESTEP_METADATA_KEY] is False


def test_timestep_regime_key_retained_under_no_metadata():
    # The regime is execution-contract, so --no_metadata (minimum set) must NOT drop it; ordinary
    # provenance still drops. Tests the hook + the base _build_minimum_metadata filter together.
    trainer = Ideogram4NetworkTrainer()
    assert IDEOGRAM4_FP32_TIMESTEP_METADATA_KEY in trainer.extra_minimum_metadata_keys()
    full = {
        "ss_network_module": "networks.lora_ideogram4",
        IDEOGRAM4_FP32_TIMESTEP_METADATA_KEY: "True",
        "ss_some_provenance": "dropped",
    }
    minimum = trainer._build_minimum_metadata(full)
    assert minimum[IDEOGRAM4_FP32_TIMESTEP_METADATA_KEY] == "True"  # retained
    assert "ss_network_module" in minimum  # base network key retained
    assert "ss_some_provenance" not in minimum  # ordinary provenance dropped


def test_base_trainer_does_not_retain_arch_key():
    # Proves the retention is the Ideogram subclass hook, not a global base addition.
    assert trainer_base.NetworkTrainer.extra_minimum_metadata_keys(object.__new__(trainer_base.NetworkTrainer)) == []


@pytest.mark.parametrize("flag", [True, False])
def test_load_transformer_applies_regime(monkeypatch, flag):
    monkeypatch.setattr(itn.ideogram4_utils, "load_ideogram4_transformer", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(itn.ideogram4_utils, "detect_fp8_scale_layout", lambda t: "per_row")
    trainer = Ideogram4NetworkTrainer()
    trainer.dit_dtype = torch.bfloat16
    args = SimpleNamespace(ideogram4_fp32_timestep=flag, disable_numpy_memmap=False)
    transformer = trainer.load_transformer(None, args, "dit.safetensors", "sdpa", False, "cpu", None)
    assert transformer.fp32_timestep is flag


# ----------------------------------------------------------------------------------------------------
# Generator: regime resolution (explicit wins; inherit-from-metadata; unstamped participates; ambiguity
# raises; malformed raises) and the two-DiT apply seam.
# ----------------------------------------------------------------------------------------------------


def _write_lora(path, regime_str):
    metadata = {} if regime_str is None else {IDEOGRAM4_FP32_TIMESTEP_METADATA_KEY: regime_str}
    save_file({"dummy": torch.zeros(1)}, path, metadata=metadata)


def test_resolver_explicit_flag_wins_over_metadata(tmp_path):
    p = str(tmp_path / "conflict.safetensors")
    _write_lora(p, "True")
    assert gen._resolve_generation_timestep_regime(True, [p]) is True
    assert gen._resolve_generation_timestep_regime(False, [p]) is False  # explicit False overrides a True stamp


def test_resolver_no_lora_is_legacy():
    assert gen._resolve_generation_timestep_regime(None, None) is False
    assert gen._resolve_generation_timestep_regime(None, []) is False


@pytest.mark.parametrize("stamp,expected", [("True", True), ("true", True), ("1", True), ("False", False), ("0", False)])
def test_resolver_inherits_lora_metadata(tmp_path, stamp, expected):
    p = str(tmp_path / "lora.safetensors")
    _write_lora(p, stamp)
    assert gen._resolve_generation_timestep_regime(None, [p]) is expected


def test_resolver_raises_on_true_false_conflict(tmp_path):
    a, b = str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")
    _write_lora(a, "True")
    _write_lora(b, "False")
    with pytest.raises(ValueError, match="disagree"):
        gen._resolve_generation_timestep_regime(None, [a, b])


def test_resolver_raises_on_fp32_mixed_with_unstamped(tmp_path):
    known, unknown = str(tmp_path / "fp32.safetensors"), str(tmp_path / "legacy.safetensors")
    _write_lora(known, "True")
    _write_lora(unknown, None)  # unstamped = unknown regime (old blissful OR upstream-native)
    with pytest.raises(ValueError, match="ambiguous"):
        gen._resolve_generation_timestep_regime(None, [known, unknown])


def test_resolver_false_plus_unstamped_is_legacy_with_warning(tmp_path, caplog):
    a, b = str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")
    _write_lora(a, "False")
    _write_lora(b, None)
    with caplog.at_level(logging.WARNING):
        assert gen._resolve_generation_timestep_regime(None, [a, b]) is False
    assert any("no timestep regime stamp" in r.message for r in caplog.records)


def test_resolver_all_unstamped_is_legacy_with_warning(tmp_path, caplog):
    a, b = str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")
    _write_lora(a, None)
    _write_lora(b, None)
    with caplog.at_level(logging.WARNING):
        assert gen._resolve_generation_timestep_regime(None, [a, b]) is False
    assert any("no timestep regime stamp" in r.message for r in caplog.records)


def test_resolver_unstamped_with_explicit_flag_does_not_raise(tmp_path):
    # An upstream-native (unstamped) adapter run under an explicit flag must NOT raise -- the flag
    # short-circuits before any metadata read.
    p = str(tmp_path / "upstream.safetensors")
    _write_lora(p, None)
    assert gen._resolve_generation_timestep_regime(True, [p]) is True


def test_resolver_raises_on_malformed_stamp(tmp_path):
    p = str(tmp_path / "bad.safetensors")
    _write_lora(p, "definitely-not-a-bool")
    with pytest.raises(ValueError, match="invalid"):
        gen._resolve_generation_timestep_regime(None, [p])


def test_read_regime_returns_none_on_unreadable_file(tmp_path):
    bogus = tmp_path / "not_a_safetensors.txt"
    bogus.write_text("nope")
    assert gen._read_lora_timestep_regime(str(bogus)) is None  # unreadable metadata -> unknown, not a crash


def test_apply_timestep_regime_sets_both_dits():
    cond, uncond = SimpleNamespace(), SimpleNamespace()
    gen._apply_timestep_regime(cond, uncond, True)
    assert cond.fp32_timestep is True and uncond.fp32_timestep is True
