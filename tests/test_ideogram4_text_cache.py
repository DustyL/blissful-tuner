from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_ideogram4
from musubi_tuner.utils.model_utils import strip_dtype_suffix


def _item(tmp_path, name):
    return SimpleNamespace(text_encoder_output_cache_path=str(tmp_path / name), caption="a cat", item_key="x")


def test_text_cache_key_roundtrip_and_reader_strip(tmp_path):
    item = _item(tmp_path, "te.safetensors")
    feats = torch.randn(5, 53248, dtype=torch.bfloat16)
    save_text_encoder_output_cache_ideogram4(item, feats)

    sd = load_file(item.text_encoder_output_cache_path)
    assert list(sd) == ["varlen_i4_llm_features_bfloat16"]
    reloaded = sd["varlen_i4_llm_features_bfloat16"]
    assert reloaded.shape == (5, 53248) and reloaded.dtype == torch.bfloat16

    # Reader contract (bucket.py): drop varlen_, then dtype-aware strip -> i4_llm_features (varlen, unstacked).
    content_key = "varlen_i4_llm_features_bfloat16".replace("varlen_", "")
    assert strip_dtype_suffix(content_key) == "i4_llm_features"


def test_text_cache_rejects_wrong_rank(tmp_path):
    item = _item(tmp_path, "bad.safetensors")
    with pytest.raises(AssertionError, match="53248"):
        save_text_encoder_output_cache_ideogram4(item, torch.randn(1, 5, 53248, dtype=torch.bfloat16))


def test_text_cache_rejects_wrong_feature_width(tmp_path):
    # Right rank (2D) but wrong feature width must be rejected before it reaches the trainer as a varlen item.
    item = _item(tmp_path, "badw.safetensors")
    with pytest.raises(AssertionError, match="53248"):
        save_text_encoder_output_cache_ideogram4(item, torch.randn(3, 7, dtype=torch.bfloat16))


def test_text_cache_fp8_key_strips_to_i4_llm_features(tmp_path):
    # H5 integration: float8_e4m3fn has an underscore; without the dtype-aware stripper the reader would
    # normalize to "i4_llm_features_float8" and KeyError. Also exercises the common NaN guard on float8.
    item = _item(tmp_path, "te8.safetensors")
    feats = torch.randn(5, 53248, dtype=torch.float32).to(torch.float8_e4m3fn)
    save_text_encoder_output_cache_ideogram4(item, feats)

    sd = load_file(item.text_encoder_output_cache_path)
    key = next(iter(sd))
    assert key == "varlen_i4_llm_features_float8_e4m3fn"
    assert strip_dtype_suffix(key.replace("varlen_", "")) == "i4_llm_features"
