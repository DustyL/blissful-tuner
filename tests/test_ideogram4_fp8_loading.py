import pytest
import torch
from torch import nn

from musubi_tuner.ideogram4.ideogram4_utils import (
    _validate_fp8_linears_patched,
    convert_prequantized_fp8_tensor,
    create_ideogram4_transformer,
)
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch


def _convert_state_dict(state_dict: dict[str, torch.Tensor], compute_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        new_key, new_value = convert_prequantized_fp8_tensor(key, value, compute_dtype)
        if new_key is not None and new_value is not None:
            converted[new_key] = new_value
    return converted


def test_prequantized_fp8_scale_keys_are_renamed_reshaped_and_cast():
    scalar_key, scalar = convert_prequantized_fp8_tensor(
        "proj.weight_scale",
        torch.tensor(0.5, dtype=torch.float32),
        torch.bfloat16,
    )
    vector_key, vector = convert_prequantized_fp8_tensor(
        "block.proj.weight_scale",
        torch.ones(4, dtype=torch.float32),
        torch.float16,
    )

    assert scalar_key == "proj.scale_weight"
    assert scalar.shape == (1,)
    assert scalar.dtype == torch.bfloat16

    assert vector_key == "block.proj.scale_weight"
    assert vector.shape == (4, 1)
    assert vector.dtype == torch.float16


def test_prequantized_fp8_comfy_quant_is_dropped():
    key, value = convert_prequantized_fp8_tensor("proj.comfy_quant", torch.ones(27, dtype=torch.uint8), torch.bfloat16)

    assert key is None
    assert value is None


def test_ideogram4_loader_sets_linear_compute_dtype():
    config = Ideogram4Config(
        emb_dim=12,
        num_layers=1,
        num_heads=3,
        intermediate_size=24,
        adanln_dim=6,
        in_channels=4,
        llm_features_dim=8,
        mrope_section=(1, 1, 0),
    )
    model = create_ideogram4_transformer(config=config, dtype=torch.float16)

    assert model.input_proj.compute_dtype == torch.float16
    assert model.t_embedding.mlp_in.compute_dtype == torch.float16


def test_ideogram4_loader_keeps_rope_inv_freq_float32():
    """RoPE inv_freq must stay float32 despite the bf16 model cast (the forward upcasts it, and a
    bf16-stored inv_freq loses precision the upcast can't recover). Matters for text rendering."""
    config = Ideogram4Config(
        emb_dim=12,
        num_layers=1,
        num_heads=3,
        intermediate_size=24,
        adanln_dim=6,
        in_channels=4,
        llm_features_dim=8,
        mrope_section=(1, 1, 0),
    )
    model = create_ideogram4_transformer(config=config, dtype=torch.bfloat16)

    assert model.rotary_emb.inv_freq.dtype == torch.float32

    # Durability: a LATER model-wide dtype cast must not reintroduce the drift. The invariant lives in
    # Ideogram4MRoPE._apply (self-healing), not in the loader, so these casts must keep inv_freq float32.
    model.to(torch.bfloat16)
    assert model.rotary_emb.inv_freq.dtype == torch.float32
    model.to(torch.float16)
    assert model.rotary_emb.inv_freq.dtype == torch.float32
    model.bfloat16()
    assert model.rotary_emb.inv_freq.dtype == torch.float32


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 is required for fp8 loading")
def test_prequantized_fp8_linear_forward_uses_compute_dtype_scale():
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 2, bias=False)

    model = TinyModel().to(torch.bfloat16)
    state_dict = _convert_state_dict(
        {
            "proj.weight": torch.randn(2, 3, dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj.weight_scale": torch.ones(2, dtype=torch.float32),
        },
        torch.bfloat16,
    )

    apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
    info = model.load_state_dict(state_dict, strict=True, assign=True)

    assert not info.missing_keys
    assert not info.unexpected_keys
    assert model.proj.scale_weight.shape == (2, 1)
    assert model.proj.scale_weight.dtype == torch.bfloat16

    output = model.proj(torch.randn(1, 3, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 is required for fp8 loading")
def test_prequantized_fp8_linear_forward_actually_applies_scale():
    """A non-unit per-row scale must change the output — guards against silently dropping the scale.

    The unit-scale test above cannot distinguish "scale applied" from "scale ignored" (1.0 is a
    no-op). Here scale != 1, so the patched forward must match the reference dequant AND differ from
    the un-scaled dequant.
    """

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 2, bias=False)

    model = TinyModel().to(torch.bfloat16)
    # Fixed literals (all exactly representable in fp8 e4m3 / bf16) so the assertions are deterministic.
    weight_fp8 = torch.tensor([[0.5, -0.25, 1.0], [2.0, 0.0, -1.5]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([2.0, 0.5], dtype=torch.float32)  # non-unit, per-row (out,)
    state_dict = _convert_state_dict(
        {"proj.weight": weight_fp8, "proj.weight_scale": scale},
        torch.bfloat16,
    )

    apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
    model.load_state_dict(state_dict, strict=True, assign=True)

    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.bfloat16)
    out = model.proj(x)

    dequant = weight_fp8.to(torch.bfloat16) * scale.reshape(2, 1).to(torch.bfloat16)
    expected = torch.nn.functional.linear(x, dequant)
    unscaled = torch.nn.functional.linear(x, weight_fp8.to(torch.bfloat16))

    assert torch.allclose(out, expected, atol=1e-3)
    # The scale must matter: dropping it would silently produce `unscaled`.
    assert not torch.allclose(out, unscaled, atol=1e-3)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 is required for fp8 loading")
def test_validate_fp8_linears_patched_raises_when_scale_missing():
    """An fp8 Linear with no scale_weight buffer must fail loudly, not silently load un-patched."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 2, bias=False)

    m = M()
    m.proj.weight.data = m.proj.weight.data.to(torch.float8_e4m3fn)  # fp8 weight, but no scale_weight

    with pytest.raises(RuntimeError, match="scale_weight"):
        _validate_fp8_linears_patched(m)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 is required for fp8 loading")
def test_validate_fp8_linears_patched_ok_when_all_patched():
    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 2, bias=False)

    m = M()
    m.proj.weight.data = m.proj.weight.data.to(torch.float8_e4m3fn)
    m.proj.register_buffer("scale_weight", torch.ones(2, 1))

    assert _validate_fp8_linears_patched(m) == (1, 1)
