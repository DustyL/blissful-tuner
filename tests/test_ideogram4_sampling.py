"""Ideogram-4 sampling-during-training (the process_sample_prompts / do_inference bridge).

Tests exercise the REAL wiring (dedup, i4_llm_features extraction, vae.to(device), unsqueeze(2) shape, the
unconditional-DiT lifecycle, the no-op block-swap switches) rather than asserting a mock returns a mock.
"""

from types import SimpleNamespace

import pytest
import torch

import musubi_tuner.ideogram4_generate_image as gen_mod
import musubi_tuner.ideogram4_train_network as m
from musubi_tuner.ideogram4.generation import denoise_ideogram4_to_tokens
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer


def _cpu_accel():
    # unwrap_model returns the model unchanged: matches Accelerator's behavior on a non-DDP single-
    # device setup, and lets disable_accelerate_forward_autocast operate on the same model the
    # caller passed in (it then no-ops because no _original_forward attribute exists on test mocks).
    return SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model)


def test_process_sample_prompts_dedupes_attaches_and_frees(monkeypatch):
    calls = {"encode": 0, "clean": 0}
    monkeypatch.setattr(m, "load_prompts", lambda f: [{"prompt": "A"}, {"prompt": "B"}, {"prompt": "A"}])
    monkeypatch.setattr(m, "load_ideogram4_text_encoder", lambda *a, **k: object())
    monkeypatch.setattr(m, "load_ideogram4_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(m, "verify_caption", lambda *a, **k: None)

    def fake_encode(tok, te, prompt, device):
        calls["encode"] += 1
        return torch.randn(5, 53248)

    monkeypatch.setattr(m, "encode_prompt_to_features", fake_encode)
    monkeypatch.setattr(m, "clean_memory_on_device", lambda d: calls.__setitem__("clean", calls["clean"] + 1))

    trainer = Ideogram4NetworkTrainer()
    trainer.dit_dtype = torch.bfloat16
    args = SimpleNamespace(
        text_encoder="te", text_encoder_config="cfg", tokenizer="tok", text_encoder_format="hf_full", strict_caption_verifier=False
    )
    prompts = trainer.process_sample_prompts(args, _cpu_accel(), "prompts.txt")

    assert calls["encode"] == 2, "two UNIQUE prompts (A, B) — duplicate A must not re-encode"
    assert all("i4_llm_features" in p for p in prompts)
    assert prompts[0]["i4_llm_features"] is prompts[2]["i4_llm_features"], "identical prompt -> shared features"
    f = prompts[0]["i4_llm_features"]
    assert f.device.type == "cpu" and f.dtype == torch.float32, "features stashed CPU float32"
    assert calls["clean"] >= 1, "the ~8.78 GB TE must be freed before the training loop"


def test_sampling_args_required_only_when_sample_prompts_set():
    trainer = Ideogram4NetworkTrainer()
    # --sample_prompts set but resources missing -> fail fast at setup.
    with pytest.raises(ValueError, match="sampling"):
        trainer.handle_model_specific_args(SimpleNamespace(sample_prompts="p.txt", mixed_precision="bf16"))
    # No --sample_prompts -> normal training is unaffected (no raise).
    trainer.handle_model_specific_args(SimpleNamespace(mixed_precision="bf16"))
    assert trainer.dit_dtype == torch.bfloat16


def test_do_inference_frees_unconditional_before_decode_and_shape(monkeypatch):
    captured = {}

    def fake_denoise(cond, uncond, tf, *, height, width, preset, device, compute_dtype, generator):
        captured.update(cond=cond, uncond=uncond, tf=tf)
        captured["denoise_autocast"] = torch.is_autocast_enabled(device.type)
        return torch.zeros(1, (height // 16) * (width // 16), 128), height // 16, width // 16

    def fake_decode(ae, z, *, grid_h, grid_w):
        captured["decoded"] = True
        captured["decode_autocast"] = torch.is_autocast_enabled(z.device.type)
        return torch.rand(1, 3, grid_h * 16, grid_w * 16)  # [0, 1]

    monkeypatch.setattr(m, "denoise_ideogram4_to_tokens", fake_denoise)
    monkeypatch.setattr(m.ideogram4_utils, "decode_dit_tokens_to_pixels", fake_decode)
    monkeypatch.setattr(m, "clean_memory_on_device", lambda d: None)
    warns = []
    monkeypatch.setattr(m.logger, "warning", lambda msg, *a, **k: warns.append(str(msg)))

    trainer = Ideogram4NetworkTrainer()
    trainer.dit_dtype = torch.bfloat16
    trainer._sample_unconditional_dit = "UNCOND"  # loaded by on_before; not None -> no reload
    vae_dev = {}
    vae = SimpleNamespace(to=lambda d: vae_dev.__setitem__("d", d))
    sample_parameter = {"i4_llm_features": torch.randn(7, 53248), "negative_prompt": "blurry"}
    args = SimpleNamespace(sampler_preset="V4_TURBO_12", unconditional_dit="u.safetensors")

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = trainer.do_inference(
            _cpu_accel(),
            args,
            sample_parameter,
            vae,
            torch.bfloat16,
            "COND",
            None,
            None,
            64,
            64,
            1,
            None,
            False,
            None,
            None,
        )

    assert out.shape == (1, 3, 1, 64, 64), "pixels.unsqueeze(2) -> (B,C,1,H,W) for the base saver"
    assert out.device.type == "cpu" and 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert captured["cond"] == "COND" and captured["uncond"] == "UNCOND", "LoRA-active cond + separate uncond DiT"
    assert captured["tf"].shape == (7, 53248), "i4_llm_features extracted from sample_parameter"
    assert captured["denoise_autocast"] is False, "sampling denoise must bypass accelerator autocast"
    assert captured["decode_autocast"] is False, "sample decode must bypass accelerator autocast"
    assert vae_dev["d"] == torch.device("cpu"), "vae.to(device) before decode (the base leaves it on CPU)"
    assert trainer._sample_unconditional_dit is None, "unconditional DiT freed BEFORE decode (the 1024 memory fix)"
    assert captured.get("decoded"), "decode ran (after the free)"
    assert any("negative_prompt" in w for w in warns), "negative_prompt warned, not silently honored"


def test_do_inference_reloads_unconditional_when_freed(monkeypatch):
    # Across multiple prompts in one interval the uncond is freed per prompt; the next prompt must reload it.
    loads = {"n": 0}

    def fake_load(*a, **k):
        loads["n"] += 1
        return "RELOADED"

    monkeypatch.setattr(m.ideogram4_utils, "load_ideogram4_transformer", fake_load)
    monkeypatch.setattr(m, "denoise_ideogram4_to_tokens", lambda cond, uncond, tf, **k: (torch.zeros(1, 16, 128), 4, 4))
    monkeypatch.setattr(m.ideogram4_utils, "decode_dit_tokens_to_pixels", lambda ae, z, **k: torch.rand(1, 3, 64, 64))
    monkeypatch.setattr(m, "clean_memory_on_device", lambda d: None)

    trainer = Ideogram4NetworkTrainer()
    trainer.dit_dtype = torch.bfloat16
    trainer._sample_unconditional_dit = None  # a prior prompt freed it
    vae = SimpleNamespace(to=lambda d: None)
    args = SimpleNamespace(sampler_preset="V4_TURBO_12", unconditional_dit="u.safetensors")
    sp = {"i4_llm_features": torch.randn(3, 53248)}

    trainer.do_inference(_cpu_accel(), args, sp, vae, torch.bfloat16, "COND", None, None, 64, 64, 1, None, False, None, None)
    assert loads["n"] == 1, "reloaded the unconditional DiT when a prior prompt had freed it"
    assert trainer._sample_unconditional_dit is None, "freed again before its own decode"


def test_block_swap_switches_are_noops_so_sampler_does_not_crash():
    # The base sampler calls these unconditionally (trainer_base.py:868/911); they must exist + no-op or
    # sample_images AttributeErrors on the Ideogram transformer.
    model = Ideogram4Transformer(
        Ideogram4Config(
            emb_dim=16,
            num_layers=1,
            num_heads=2,
            intermediate_size=24,
            adanln_dim=8,
            in_channels=128,
            llm_features_dim=8,
            mrope_section=(2, 2, 2),
        )
    )
    assert model.switch_block_swap_for_inference() is None
    assert model.switch_block_swap_for_training() is None  # round-trip, no crash


def test_on_after_unloads_unconditional_dit_moves_vae_to_cpu_and_is_idempotent(monkeypatch):
    cleaned = []
    monkeypatch.setattr(m, "clean_memory_on_device", lambda d: cleaned.append(d))
    trainer = Ideogram4NetworkTrainer()
    trainer._sample_unconditional_dit = "BIG_MODEL"
    vae_dev = {}
    vae = SimpleNamespace(to=lambda d: vae_dev.__setitem__("d", d))

    # on_after signature: (accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype)
    trainer.on_after_sample_images(_cpu_accel(), None, None, None, vae, None, None, None, None)
    assert trainer._sample_unconditional_dit is None, "unconditional DiT freed after sampling"
    assert vae_dev["d"] == "cpu", "VAE returned to CPU defensively (covers a do_inference raise the base misses)"
    assert cleaned, "CUDA cache cleaned so VRAM doesn't ratchet across intervals"

    # Safe when nothing was loaded (a partial/OOM'd lazy load) and vae is None — no AttributeError masks the failure.
    trainer.on_after_sample_images(_cpu_accel(), None, None, None, None, None, None, None, None)


def test_standalone_and_training_share_one_denoise_core():
    # The validated B5 t-convention lives ONLY in denoise_ideogram4_to_tokens; both the standalone and the
    # trainer call it (then decode separately), so neither can fork the convention.
    assert m.denoise_ideogram4_to_tokens is denoise_ideogram4_to_tokens
    assert gen_mod.denoise_ideogram4_to_tokens is denoise_ideogram4_to_tokens


def test_setup_parser_adds_sampling_args():
    import argparse

    parser = m.ideogram4_setup_parser(argparse.ArgumentParser())
    args = parser.parse_args(
        ["--unconditional_dit", "u", "--text_encoder", "t", "--tokenizer", "tok", "--sampler_preset", "V4_QUALITY_48"]
    )
    assert args.unconditional_dit == "u" and args.text_encoder == "t" and args.sampler_preset == "V4_QUALITY_48"
    assert args.text_encoder_format == "hf_full"  # default
