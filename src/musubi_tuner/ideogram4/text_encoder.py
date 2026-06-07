"""Offline Qwen3-VL text encoder for Ideogram 4 (H6: native transformers classes, no remote code).

Loads the Qwen3-VL text encoder from LOCAL files only and extracts the 53,248-dim conditioning features
(13 activation-layer hidden states concatenated, 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)).

Two checkpoint formats (--text_encoder_format):
- hf_full (DEFAULT, canonical parity): native Qwen3VLModel(cfg) loaded from the HF text_encoder/
  model.safetensors (language_model.* + visual.*, per-row fp8 scales, no comfy_quant).
- comfy_text (opt-in optimization): text-only Comfy checkpoint (model.* + lm_head, per-tensor scales +
  comfy_quant). Built from cfg.text_config. Deferred — see load_ideogram4_text_encoder.

Feature extraction is a MANUAL decoder-layer tap (not output_hidden_states): the standard forward overwrites
the last layer's pre-norm output with the final norm, but canonical taps the raw (pre-norm) layer outputs at
ALL activation layers including the last. Verified equivalent to the native forward for intermediate layers.
"""

from __future__ import annotations

import json
import struct
from typing import Union

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel
from transformers.masking_utils import create_causal_mask

from blissful_tuner.blissful_logger import BlissfulLogger
from musubi_tuner.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from musubi_tuner.ideogram4.ideogram4_utils import (
    _validate_fp8_linears_patched,
    load_prequantized_fp8_state_dict,
    set_linear_compute_dtype,
)
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch

logger = BlissfulLogger(__name__, "green")

IDEOGRAM4_TE_HIDDEN_SIZE = 4096
IDEOGRAM4_TE_FEATURE_DIM = IDEOGRAM4_TE_HIDDEN_SIZE * len(QWEN3_VL_ACTIVATION_LAYERS)  # 4096 * 13 = 53248
TEXT_ENCODER_FORMATS = ("hf_full", "comfy_text")


def load_ideogram4_tokenizer(tokenizer_dir: str):
    """Load the Qwen3-VL tokenizer from a LOCAL directory only (H6: no network, no trust_remote_code)."""
    return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def inspect_te_checkpoint(path: str) -> dict:
    """Classify a TE checkpoint from its header alone (no tensor load) for the format gate."""
    header = _read_safetensors_header(path)
    fp8 = [k for k in header if header[k]["dtype"] in ("F8_E4M3", "F8_E5M2")]
    scale_keys = [k for k in header if k.endswith(".weight_scale")]
    per_tensor = sum(len(header[k]["shape"]) == 0 for k in scale_keys)
    per_row = sum(len(header[k]["shape"]) == 1 for k in scale_keys)
    return {
        "n_fp8": len(fp8),
        "n_scale": len(scale_keys),
        "n_per_tensor_scale": per_tensor,
        "n_per_row_scale": per_row,
        "n_comfy_quant": sum("comfy_quant" in k for k in header),
        "has_language_model": any(k.startswith("language_model.") for k in header),
        "has_visual": any(k.startswith("visual.") for k in header),
        "has_model_prefix": any(k.startswith("model.") for k in header),
    }


def _assert_hf_full_format(path: str) -> dict:
    """Gate (exit-gate 2): assert the file is the HF full multimodal fp8 checkpoint BEFORE loading tensors."""
    info = inspect_te_checkpoint(path)
    problems = []
    if not (info["has_language_model"] and info["has_visual"]):
        problems.append(f"expected language_model.* + visual.* keys, got {info}")
    if info["n_comfy_quant"] != 0:
        problems.append(f"expected 0 comfy_quant (HF full), got {info['n_comfy_quant']} — is this a Comfy checkpoint?")
    if info["n_per_row_scale"] == 0 or info["n_per_tensor_scale"] != 0:
        problems.append(
            f"expected per-row weight_scale, got per_row={info['n_per_row_scale']} per_tensor={info['n_per_tensor_scale']}"
        )
    if problems:
        raise ValueError(f"Ideogram 4 hf_full TE format gate failed for {path}:\n  " + "\n  ".join(problems))
    return info


def load_ideogram4_text_encoder(
    te_path: str,
    config_dir: str,
    *,
    text_encoder_format: str = "hf_full",
    dtype: torch.dtype = torch.bfloat16,
    loading_device: Union[str, torch.device] = "cpu",
    disable_numpy_memmap: bool = False,
) -> Qwen3VLModel:
    """Load the Ideogram 4 Qwen3-VL text encoder OFFLINE from local config + fp8 weights.

    H6: native Qwen3VLConfig/Qwen3VLModel only — no AutoModel, no AutoConfig, no trust_remote_code, no Hub.
    """
    if text_encoder_format not in TEXT_ENCODER_FORMATS:
        raise ValueError(f"text_encoder_format must be one of {TEXT_ENCODER_FORMATS}, got {text_encoder_format!r}")
    if text_encoder_format == "comfy_text":
        # Opt-in optimization (text-only, per-tensor scales, model.-prefix strip, lm_head drop). Deferred until
        # it passes structural parity vs the HF language tower; default stays on the canonical hf_full path.
        raise NotImplementedError("comfy_text TE format is not implemented yet; use --text_encoder_format hf_full")

    loading_device = torch.device(loading_device)
    config = Qwen3VLConfig.from_pretrained(config_dir, local_files_only=True, trust_remote_code=False)

    # Gate the file format from its header BEFORE materializing tensors.
    info = _assert_hf_full_format(te_path)

    with init_empty_weights():
        model = Qwen3VLModel(config)
        model.to(dtype)
        set_linear_compute_dtype(model, dtype)

    state_dict, stats = load_prequantized_fp8_state_dict(
        te_path, device=loading_device, compute_dtype=dtype, disable_numpy_memmap=disable_numpy_memmap
    )
    if stats.dropped_comfy_quant != 0:
        raise ValueError(f"hf_full TE expected 0 comfy_quant, dropped {stats.dropped_comfy_quant}")

    apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
    load_info = model.load_state_dict(state_dict, strict=True, assign=True)
    model.to(device=loading_device)
    model.eval()

    patched, fp8_total = _validate_fp8_linears_patched(model)
    logger.info(
        f"Loaded Ideogram 4 text encoder (hf_full) from {te_path}: info={load_info}, "
        f"fp8_linears_patched={patched}/{fp8_total}, header={info}"
    )
    return model


def _tap_qwen3_vl_text_layers(language_model, token_ids: torch.Tensor) -> list[torch.Tensor]:
    """Manual decoder-layer tap (text-only): return the pre-norm hidden states at QWEN3_VL_ACTIVATION_LAYERS.

    Mirrors transformers' Qwen3VLTextModel.forward (validated equivalent for intermediate layers) but captures
    raw layer outputs — including the LAST layer pre-final-norm, which output_hidden_states does not expose.
    """
    device = token_ids.device
    seq_len = token_ids.shape[1]
    # (4, B, L): row 0 = text positions (causal mask + decoder layers), rows 1:4 = MRoPE (t,h,w).
    # Text-only: every row is arange (no image tokens, positions are local).
    position_ids = torch.arange(seq_len, device=device).view(1, 1, seq_len).expand(4, token_ids.shape[0], seq_len).contiguous()
    text_position_ids = position_ids[0]
    mrope_position_ids = position_ids[1:]

    inputs_embeds = language_model.embed_tokens(token_ids)
    attention_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(token_ids.shape, dtype=torch.long, device=device),
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states
    return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]


def tokenize_ideogram4_prompt(tokenizer, prompt: str) -> torch.Tensor:
    """Canonical tokenization (exit-gate 6): chat template + add_special_tokens=False. Returns (1, L) ids."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"]


@torch.no_grad()
def encode_prompt_to_features(tokenizer, text_encoder: Qwen3VLModel, prompt: str, device: Union[str, torch.device]) -> torch.Tensor:
    """Encode one prompt to its (L, 53248) float32 Qwen3-VL conditioning features (per-prompt, image-independent)."""
    device = torch.device(device)
    token_ids = tokenize_ideogram4_prompt(tokenizer, prompt).to(device)
    selected = _tap_qwen3_vl_text_layers(text_encoder.language_model, token_ids)  # 13 x (1, L, 4096)
    stacked = torch.stack(selected, dim=0)  # (taps, 1, L, 4096)
    stacked = stacked.permute(1, 2, 3, 0).reshape(token_ids.shape[0], token_ids.shape[1], -1)  # (1, L, 53248)
    return stacked[0].to(torch.float32)
