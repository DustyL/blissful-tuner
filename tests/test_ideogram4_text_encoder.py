import copy
import os

import pytest
import torch

from musubi_tuner.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from musubi_tuner.ideogram4.text_encoder import (
    IDEOGRAM4_TE_FEATURE_DIM,
    _assert_hf_full_format,
    _tap_qwen3_vl_text_layers,
    inspect_te_checkpoint,
    load_ideogram4_text_encoder,
)

# Building a tiny native Qwen3-VL needs the local config (shrunk). Format-gate tests below don't.
_TE_CONFIG_DIR = "/home/dustin/Ideogram-4/Huggingface-Model-Weights/text_encoder"
_HAS_TE_CONFIG = os.path.isdir(_TE_CONFIG_DIR)
_needs_config = pytest.mark.skipif(not _HAS_TE_CONFIG, reason="local Qwen3-VL config required")


def _tiny_language_model():
    from transformers import Qwen3VLConfig, Qwen3VLModel

    cfg = copy.deepcopy(Qwen3VLConfig.from_pretrained(_TE_CONFIG_DIR, local_files_only=True))
    # 36 tiny layers so QWEN3_VL_ACTIVATION_LAYERS (taps up to 35) are all valid (real model has 36).
    for k, v in (
        ("num_hidden_layers", 36),
        ("hidden_size", 64),
        ("intermediate_size", 128),
        ("num_attention_heads", 4),
        ("num_key_value_heads", 2),
        ("head_dim", 16),
    ):
        setattr(cfg.text_config, k, v)
    for k, v in (("depth", 2), ("hidden_size", 64), ("num_heads", 4), ("intermediate_size", 128), ("out_hidden_size", 64)):
        if hasattr(cfg.vision_config, k):
            setattr(cfg.vision_config, k, v)
    torch.manual_seed(0)
    return Qwen3VLModel(cfg).eval().language_model


def _write_st(path, tensors, metadata=None):
    from safetensors.torch import save_file

    save_file(tensors, path, metadata=metadata or {})


# --------------------------------------------------------------------------- format gate (no weights / config)


def test_feature_dim_is_53248():
    assert IDEOGRAM4_TE_FEATURE_DIM == 4096 * len(QWEN3_VL_ACTIVATION_LAYERS) == 53248


def test_inspect_and_gate_hf_full_vs_comfy(tmp_path):
    fp8 = getattr(torch, "float8_e4m3fn")
    # HF-full-like: language_model.* + visual.*, per-row scale, no comfy_quant.
    hf = str(tmp_path / "hf.safetensors")
    _write_st(
        hf,
        {
            "language_model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=fp8),
            "language_model.layers.0.self_attn.q_proj.weight_scale": torch.zeros(8, dtype=torch.float32),
            "visual.blocks.0.attn.qkv.weight": torch.zeros(8, 8, dtype=fp8),
            "visual.blocks.0.attn.qkv.weight_scale": torch.zeros(8, dtype=torch.float32),
        },
    )
    info = inspect_te_checkpoint(hf)
    assert info["has_language_model"] and info["has_visual"]
    assert info["n_per_row_scale"] == 2 and info["n_per_tensor_scale"] == 0 and info["n_comfy_quant"] == 0
    _assert_hf_full_format(hf)  # must not raise

    # Comfy-text-like: model.* + lm_head, per-tensor scalar scale + comfy_quant. Gate must REJECT for hf_full.
    comfy = str(tmp_path / "comfy.safetensors")
    _write_st(
        comfy,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=fp8),
            "model.layers.0.self_attn.q_proj.weight_scale": torch.zeros((), dtype=torch.float32),
            "model.layers.0.self_attn.q_proj.comfy_quant": torch.zeros(27, dtype=torch.uint8),
            "lm_head.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
        },
    )
    cinfo = inspect_te_checkpoint(comfy)
    assert cinfo["has_model_prefix"] and not cinfo["has_visual"]
    assert cinfo["n_per_tensor_scale"] == 1 and cinfo["n_comfy_quant"] == 1
    with pytest.raises(ValueError, match="comfy_quant|per-row|language_model"):
        _assert_hf_full_format(comfy)


def test_comfy_text_format_not_implemented():
    with pytest.raises(NotImplementedError, match="comfy_text"):
        load_ideogram4_text_encoder("x", "y", text_encoder_format="comfy_text")


def test_invalid_format_rejected():
    with pytest.raises(ValueError, match="text_encoder_format"):
        load_ideogram4_text_encoder("x", "y", text_encoder_format="bogus")


def test_tokenizer_loads_local_only(monkeypatch):
    from musubi_tuner.ideogram4 import text_encoder as te

    captured = {}

    class _FakeTok:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured.update(path=path, **kwargs)
            return "tok"

    monkeypatch.setattr(te, "AutoTokenizer", _FakeTok)
    assert te.load_ideogram4_tokenizer("/local/tok") == "tok"
    assert captured["local_files_only"] is True and captured["trust_remote_code"] is False


# --------------------------------------------------------------------------- native tap (tiny real Qwen3-VL)


@_needs_config
def test_tap_returns_13_layers_and_feature_shape():
    lm = _tiny_language_model()
    token_ids = torch.randint(1, 100, (1, 7))
    selected = _tap_qwen3_vl_text_layers(lm, token_ids)
    assert len(selected) == len(QWEN3_VL_ACTIVATION_LAYERS)
    assert all(t.shape == (1, 7, lm.config.hidden_size) for t in selected)
    stacked = torch.stack(selected, 0).permute(1, 2, 3, 0).reshape(1, 7, -1)
    assert stacked.shape == (1, 7, lm.config.hidden_size * len(QWEN3_VL_ACTIVATION_LAYERS))
    assert torch.isfinite(stacked).all()


@_needs_config
def test_text_only_equals_packed_text_slice():
    """GATE 8 (foundational): text-only features == the text slice of the canonical packed [text][image]
    sequence, justifying per-prompt (image-independent) caching. Image positions are masked + causal, text
    precedes image, text positions are local arange -> the text features cannot depend on the image."""
    from transformers.masking_utils import create_causal_mask

    lm = _tiny_language_model()
    num_text = 5
    token_ids = torch.randint(1, 100, (1, num_text))

    # text-only (our production path)
    text_only = _tap_qwen3_vl_text_layers(lm, token_ids)

    # packed [text][image]: image tokens = 0, masked via attention_mask; text positions = arange, image = 0.
    num_image = 4
    total = num_text + num_image
    packed = torch.zeros(1, total, dtype=torch.long)
    packed[0, :num_text] = token_ids[0]
    pos_2d = torch.cat([torch.arange(num_text), torch.zeros(num_image, dtype=torch.long)]).view(1, total)
    attn = torch.cat([torch.ones(num_text), torch.zeros(num_image)]).view(1, total).long()
    pos_4d = pos_2d[None, ...].expand(4, 1, total)
    text_pos, mrope = pos_4d[0], pos_4d[1:]

    emb = lm.embed_tokens(packed)
    cm = create_causal_mask(config=lm.config, inputs_embeds=emb, attention_mask=attn, past_key_values=None, position_ids=text_pos)
    pe = lm.rotary_emb(emb, mrope)
    h = emb
    captured = {}
    tap = set(QWEN3_VL_ACTIVATION_LAYERS)
    for i, layer in enumerate(lm.layers):
        o = layer(h, attention_mask=cm, position_ids=text_pos, past_key_values=None, position_embeddings=pe)
        h = o[0] if isinstance(o, (tuple, list)) else o
        if i in tap:
            captured[i] = h
    packed_slice = [captured[i][:, :num_text] for i in QWEN3_VL_ACTIVATION_LAYERS]

    for a, b in zip(text_only, packed_slice):
        assert torch.allclose(a, b, atol=1e-4), "text-only features diverge from packed text slice"
