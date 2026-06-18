"""Regression coverage for the Ideogram 4 "Tier 1" hardening ported from upstream musubi-tuner #975.

Four independent, low-risk correctness/robustness items:
  1. Adaptive ``create_causal_mask`` wrapper (transformers signature-robustness).
  2. MRoPE fp32 autocast guard (image-position collapse under autocast).
  3. Non-float8 dtype rejection in the pre-quantized fp8 loader.
  4. Ideogram 4 default ModelSpec resolution (1024x1024).
"""

import types

import pytest
import torch

from musubi_tuner.ideogram4 import text_encoder as te
from musubi_tuner.ideogram4.ideogram4_utils import convert_prequantized_fp8_tensor
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4MRoPE


# ----------------------------------------------------------------------------------------------------
# Item 2: MRoPE fp32 autocast guard
# ----------------------------------------------------------------------------------------------------


def _build_mrope() -> Ideogram4MRoPE:
    # head_dim=16 -> 8 inv_freq entries; mrope_section indices [1],[2] stay in range.
    return Ideogram4MRoPE(head_dim=16, base=10000, mrope_section=(2, 2, 2)).eval()


def _image_position_ids(length: int) -> torch.Tensor:
    # (B, L, 3) = (t, h, w). Image position ids start at 2**16, where bf16 spacing is 512: without the
    # guard, an active autocast collapses every position in a 512-wide band to one value.
    offset = 2**16
    t = torch.zeros(1, length, dtype=torch.long)
    h = (offset + torch.arange(length)).view(1, length)
    w = (offset + torch.arange(length)).view(1, length)
    return torch.stack([t, h, w], dim=-1)


def test_mrope_autocast_is_a_noop_under_the_guard():
    mrope = _build_mrope()
    position_ids = _image_position_ids(8)

    cos_ref, sin_ref = mrope(position_ids)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cos_ac, sin_ac = mrope(position_ids)

    # The fp32 guard forces autocast off for the frequency matmul, so the result is bit-identical
    # whether or not an outer autocast is active. Removing the guard breaks this (matmul -> bf16).
    assert torch.equal(cos_ref, cos_ac)
    assert torch.equal(sin_ref, sin_ac)


def test_mrope_adjacent_image_positions_do_not_collapse_under_autocast():
    mrope = _build_mrope()
    position_ids = _image_position_ids(8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cos_ac, sin_ac = mrope(position_ids)
    # Adjacent image positions (h/w differ by 1, both ~2**16) must remain distinct. A bf16 collapse
    # would make these identical -- the flat-checkerboard failure mode.
    assert not torch.allclose(cos_ac[0, 0], cos_ac[0, 1])


# ----------------------------------------------------------------------------------------------------
# Item 3: non-float8 dtype rejection in the pre-quantized fp8 loader
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_prequant_loader_rejects_one_byte_integer_weights(dtype):
    weight = torch.zeros(4, 4, dtype=dtype)
    with pytest.raises(ValueError, match="not float8"):
        convert_prequantized_fp8_tensor("blocks.0.attn.qkv.weight", weight, torch.bfloat16)


def test_prequant_loader_keeps_fp8_weight_as_is():
    weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    key, value = convert_prequantized_fp8_tensor("blocks.0.attn.qkv.weight", weight, torch.bfloat16)
    assert key == "blocks.0.attn.qkv.weight"
    assert value.dtype == torch.float8_e4m3fn  # fp8 native weight is preserved, not rejected


def test_prequant_loader_casts_plain_float_weight_to_compute_dtype():
    weight = torch.zeros(4, 4, dtype=torch.float32)
    key, value = convert_prequantized_fp8_tensor("blocks.0.norm.weight", weight, torch.bfloat16)
    assert key == "blocks.0.norm.weight"
    assert value.dtype == torch.bfloat16  # ordinary float weights still flow through unchanged


# ----------------------------------------------------------------------------------------------------
# Item 1: adaptive create_causal_mask wrapper
# ----------------------------------------------------------------------------------------------------


def test_causal_mask_helper_uses_inputs_embeds_name(monkeypatch):
    captured = {}

    def fake_create_causal_mask(config, attention_mask, past_key_values, position_ids, inputs_embeds):
        captured.update(inputs_embeds=inputs_embeds, position_ids=position_ids, config=config)
        return "MASK"

    monkeypatch.setattr(te, "create_causal_mask", fake_create_causal_mask)
    lm = types.SimpleNamespace(config="CFG")
    out = te._create_qwen_causal_mask(lm, "EMB", "AM", "PID", "CACHE")

    assert out == "MASK"
    assert captured["inputs_embeds"] == "EMB"  # modern signature: inputs_embeds (no cache_position)


def test_causal_mask_helper_adapts_to_input_embeds_rename_and_cache_position(monkeypatch):
    captured = {}

    def fake_create_causal_mask(config, attention_mask, past_key_values, position_ids, input_embeds, cache_position):
        captured.update(input_embeds=input_embeds, cache_position=cache_position)
        return "MASK"

    monkeypatch.setattr(te, "create_causal_mask", fake_create_causal_mask)
    lm = types.SimpleNamespace(config="CFG")
    out = te._create_qwen_causal_mask(lm, "EMB", "AM", "PID", "CACHE")

    assert out == "MASK"
    assert captured["input_embeds"] == "EMB"  # adapted to the legacy/renamed parameter
    assert captured["cache_position"] == "CACHE"  # supplied only when the signature accepts it


# ----------------------------------------------------------------------------------------------------
# Item 4: Ideogram 4 default ModelSpec resolution
# ----------------------------------------------------------------------------------------------------


def test_sai_model_spec_ideogram4_defaults_to_1024():
    from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_IDEOGRAM4
    from musubi_tuner.utils import sai_model_spec

    metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_IDEOGRAM4, 0.0)
    assert metadata["modelspec.resolution"] == "1024x1024"
