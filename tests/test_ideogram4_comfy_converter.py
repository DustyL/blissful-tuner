# Tests for the Ideogram 4 blissful <-> ComfyUI LoRA converter (backlog P1-4).
#
# Ideogram 4 needs no key renames (blissful's lora_unet_layers_* names already match ComfyUI's
# generic lora_unet_{path} mapping; qkv stays fused on both sides). The conversion is semantic:
# strip blissful's flag buffers, bake rsLoRA's alpha/sqrt(r) scaling into alpha for ComfyUI's
# alpha/r convention, and rename+reshape DoRA magnitudes ((out,) dora_layer.weight -> (out,1)
# dora_scale — the reshape prevents a (out,out) broadcast in ComfyUI's weight_decompose).

import math
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.networks.convert_ideogram4_lora_to_comfy import convert_state_dict, main

V7_CHECKPOINT = "/home/dustin/output/dlay_ideogram4_lora_v7_prodigy_tuned/dlay_ideogram4_lora_v7.safetensors"


def _make_module(name: str, out_dim: int = 12, in_dim: int = 6, rank: int = 4, alpha: float = 4.0, dora: bool = False):
    sd = {
        f"{name}.lora_down.weight": torch.randn(rank, in_dim),
        f"{name}.lora_up.weight": torch.randn(out_dim, rank),
        f"{name}.alpha": torch.tensor(alpha),
    }
    if dora:
        sd[f"{name}.dora_layer.weight"] = torch.rand(out_dim)
    return sd


def _make_checkpoint(use_dora: bool = False, use_rslora: bool = False, n_modules: int = 2):
    sd = {}
    for i in range(n_modules):
        sd.update(_make_module(f"lora_unet_layers_{i}_attention_qkv", out_dim=18, in_dim=6, dora=use_dora))
        sd.update(_make_module(f"lora_unet_layers_{i}_attention_o", out_dim=6, in_dim=6, dora=use_dora))
    sd["use_dora_flag"] = torch.tensor(use_dora, dtype=torch.bool)
    sd["use_rslora_flag"] = torch.tensor(use_rslora, dtype=torch.bool)
    return sd


def test_plain_lora_strips_flags_and_leaves_weights_untouched():
    src = _make_checkpoint()
    out, stats = convert_state_dict(src)
    assert stats["flags_stripped"] == 2
    assert stats["alpha_rescaled"] == 0
    assert stats["dora_converted"] == 0
    assert "use_dora_flag" not in out and "use_rslora_flag" not in out
    for key, tensor in src.items():
        if key in ("use_dora_flag", "use_rslora_flag"):
            continue
        assert torch.equal(out[key], tensor), key
    # Source dict is not mutated.
    assert "use_dora_flag" in src


def test_missing_flags_tolerated():
    # A checkpoint that already lost its flags (or a foreign sd-scripts export) converts cleanly.
    src = _make_module("lora_unet_layers_0_attention_qkv")
    out, stats = convert_state_dict(src)
    assert stats["flags_stripped"] == 0
    assert set(out) == set(src)


def test_rslora_alpha_rescaled_for_alpha_over_rank_semantics():
    rank, alpha = 4, 4.0
    src = _make_checkpoint(use_rslora=True)
    out, stats = convert_state_dict(src)
    assert stats["alpha_rescaled"] == 4
    expected = alpha * math.sqrt(rank)
    for key, tensor in out.items():
        if key.endswith(".alpha"):
            assert tensor.item() == pytest.approx(expected)
            # Effective scale parity: blissful alpha/sqrt(r) == converted alpha/r.
            assert tensor.item() / rank == pytest.approx(alpha / math.sqrt(rank))


def test_rslora_alpha_without_down_weight_raises():
    src = {
        "lora_unet_layers_0_attention_o.alpha": torch.tensor(4.0),
        "use_rslora_flag": torch.tensor(True),
    }
    with pytest.raises(ValueError, match="lora_down.weight"):
        convert_state_dict(src)


def test_dora_magnitude_renamed_and_reshaped_column():
    src = _make_checkpoint(use_dora=True)
    out, stats = convert_state_dict(src)
    assert stats["dora_converted"] == 4
    assert not any(k.endswith(".dora_layer.weight") for k in out)
    qkv_scale = out["lora_unet_layers_0_attention_qkv.dora_scale"]
    assert qkv_scale.shape == (18, 1)  # (out, 1) column — NOT (out,) — see module docstring
    assert torch.equal(qkv_scale.reshape(-1), src["lora_unet_layers_0_attention_qkv.dora_layer.weight"])


def test_dora_flag_without_magnitudes_raises():
    src = _make_checkpoint()
    src["use_dora_flag"] = torch.tensor(True)
    with pytest.raises(ValueError, match="use_dora_flag"):
        convert_state_dict(src)


def test_reverse_roundtrip_restores_blissful_layout():
    src = _make_checkpoint(use_dora=True)
    forward, _ = convert_state_dict(src)
    back, stats = convert_state_dict(forward, reverse=True)
    assert stats["dora_converted"] == 4
    assert set(back) == set(src)
    for key, tensor in src.items():
        assert torch.equal(back[key], tensor), key
    assert back["use_dora_flag"].item() is True
    assert back["use_rslora_flag"].item() is False


def test_reverse_of_rslora_checkpoint_keeps_baked_alpha():
    # Forward bakes sqrt(rank) into alpha and drops the flag; reverse cannot (and must not) undo
    # that — the returned checkpoint is a valid blissful checkpoint under alpha/r semantics.
    src = _make_checkpoint(use_rslora=True)
    forward, _ = convert_state_dict(src)
    back, _ = convert_state_dict(forward, reverse=True)
    assert back["use_rslora_flag"].item() is False
    for key, tensor in back.items():
        if key.endswith(".alpha"):
            assert tensor.item() == pytest.approx(4.0 * math.sqrt(4))


def test_main_io_roundtrip(tmp_path):
    src_path = tmp_path / "src.safetensors"
    dst_path = tmp_path / "dst.safetensors"
    save_file(_make_checkpoint(), str(src_path), metadata={"ss_network_module": "networks.lora_ideogram4"})

    main(SimpleNamespace(src_path=str(src_path), dst_path=str(dst_path), reverse=False))

    with safe_open(str(dst_path), framework="pt") as f:
        keys = set(f.keys())
        metadata = f.metadata()
    assert "use_dora_flag" not in keys and "use_rslora_flag" not in keys
    assert metadata["ss_network_module"] == "networks.lora_ideogram4"
    assert "sshs_model_hash" in metadata  # hashes recomputed for the converted tensors


def test_loha_lokr_checkpoints_rejected():
    for marker_key in (
        "lora_unet_layers_0_attention_qkv.hada_w1_a",
        "lora_unet_layers_0_attention_qkv.lokr_w1",
    ):
        src = {marker_key: torch.randn(4, 4)}
        with pytest.raises(NotImplementedError, match="P1-3"):
            convert_state_dict(src)


def test_reverse_warns_on_blissful_format_input(caplog):
    import logging

    src = _make_checkpoint(use_dora=True)  # blissful format, fed to --reverse by mistake
    with caplog.at_level(logging.WARNING):
        convert_state_dict(src, reverse=True)
    assert any("blissful-format keys" in r.message for r in caplog.records)


def test_output_tensors_do_not_share_storage_with_input():
    # Mutating a converted DoRA tensor must not corrupt the caller's input dict.
    src = _make_checkpoint(use_dora=True)
    original = src["lora_unet_layers_0_attention_qkv.dora_layer.weight"].clone()
    out, _ = convert_state_dict(src)
    out["lora_unet_layers_0_attention_qkv.dora_scale"].mul_(0.0)
    assert torch.equal(src["lora_unet_layers_0_attention_qkv.dora_layer.weight"], original)


def test_rslora_mixed_rank_modules_rescale_per_module():
    # rank_pattern allows per-module ranks; the converter must read rank per-tensor, not globally.
    src = {}
    src.update(_make_module("lora_unet_layers_0_attention_qkv", out_dim=18, in_dim=6, rank=4, alpha=4.0))
    src.update(_make_module("lora_unet_layers_0_feed_forward_w1", out_dim=24, in_dim=6, rank=16, alpha=8.0))
    src["use_rslora_flag"] = torch.tensor(True)
    out, stats = convert_state_dict(src)
    assert stats["alpha_rescaled"] == 2
    assert out["lora_unet_layers_0_attention_qkv.alpha"].item() == pytest.approx(4.0 * math.sqrt(4))
    assert out["lora_unet_layers_0_feed_forward_w1.alpha"].item() == pytest.approx(8.0 * math.sqrt(16))


@pytest.mark.skipif(not os.path.exists(V7_CHECKPOINT), reason="local v7 DLAY checkpoint not present")
def test_real_v7_checkpoint_converts():
    state_dict = {}
    with safe_open(V7_CHECKPOINT, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    out, stats = convert_state_dict(state_dict)
    assert stats["flags_stripped"] == 2
    assert stats["alpha_rescaled"] == 0  # v7 trained with use_rslora=False
    assert stats["dora_converted"] == 0  # v7 trained with use_dora=False
    # 34 layers x 5 modules x 3 tensors (down/up/alpha), flags gone.
    assert len(out) == 510
    assert all(k.startswith("lora_unet_layers_") for k in out)
