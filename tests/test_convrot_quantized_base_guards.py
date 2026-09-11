"""Guards protecting true-weight-space operations from ConvRot's opaquely-quantized base weights.

Under ``--convrot_int8`` a target ``.weight`` holds Hadamard-ROTATED int8 codes plus a per-channel
``scale_weight``. Recovering true weights needs the inverse rotation, not just the scale — so any
consumer that does float math directly on ``.weight`` (DoRA's weight norm, PiSSA's SVD, a destructive
merge, ``pre_calculation``) computes on lattice values. That failure mode raises nothing and produces
no NaN, which is why it needs explicit guards and these tests.

Two layers are covered:
  1. ``dora_utils.raise_if_opaque_quantized`` wired into every true-space consumer (last resort,
     covers merge_lora.py / inference / any future entrypoint).
  2. ``Krea2NetworkTrainer._reject_convrot_incompatible_adapters`` (fail-fast, before a 26 GB load),
     which MUST parse with the same helpers the network factory uses — a private copy drifts.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from musubi_tuner.modules.convrot_int8_utils import (
    CONVROT_GROUPSIZE,
    apply_convrot_int8_monkey_patch,
    quantize_weight_convrot,
)
from musubi_tuner.networks.dora_utils import (
    dora_weight_norm_materialized,
    is_opaque_quantized_weight,
    raise_if_opaque_quantized,
)
from musubi_tuner.networks.lora import LoRAInfModule, parse_bool_arg, parse_init_lora_weights_arg

K, N, R = CONVROT_GROUPSIZE, 64, 8


def _convrot_linear():
    """An nn.Linear carrying ConvRot int8 weights, built exactly as load_krea2_dit builds them."""
    torch.manual_seed(0)
    w_true = torch.randn(N, K) * 0.02

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(K, N, bias=False)

        def forward(self, x):
            return self.l(x)

    m = M()
    wq, scale = quantize_weight_convrot("l.weight", w_true, CONVROT_GROUPSIZE)
    sd = {"l.weight": wq, "l.scale_weight": scale}
    apply_convrot_int8_monkey_patch(m, sd)
    m.requires_grad_(False)  # int8 params cannot require grad; the real loader does this too
    m.load_state_dict(sd, strict=True, assign=True)
    assert m.l.weight.dtype == torch.int8
    return m


# ---------------------------------------------------------------- layer 1: dtype guards


def test_opaque_detection_covers_int_but_not_float_or_fp8():
    assert is_opaque_quantized_weight(torch.zeros(2, 2, dtype=torch.int8))
    assert not is_opaque_quantized_weight(torch.zeros(2, 2, dtype=torch.bfloat16))
    assert not is_opaque_quantized_weight(torch.zeros(2, 2, dtype=torch.float32))
    if hasattr(torch, "float8_e4m3fn"):
        # fp8 is recoverable via scale_weight, so it must keep its own dedicated handling
        assert not is_opaque_quantized_weight(torch.zeros(2, 2, dtype=torch.float8_e4m3fn))


def test_raise_if_opaque_names_the_operation_and_passes_floats():
    raise_if_opaque_quantized(torch.zeros(2, 2, dtype=torch.bfloat16), "x")  # no raise
    with pytest.raises(ValueError, match="rotated int8"):
        raise_if_opaque_quantized(torch.zeros(2, 2, dtype=torch.int8), "DoRA weight norm")


def test_dora_weight_norm_rejects_int8_base_instead_of_computing_on_codes():
    wq, _ = quantize_weight_convrot("w", torch.randn(N, K) * 0.02, CONVROT_GROUPSIZE)
    lora_weight = torch.randn(N, K) * 0.01
    with pytest.raises(ValueError, match="DoRA weight norm"):
        dora_weight_norm_materialized(wq, lora_weight, scaling=1.0)


def test_efficient_dora_weight_norm_rejects_int8_base():
    """The memory-efficient norm path (no B@A materialization) needs the same guard."""
    from musubi_tuner.networks.lora import DoRALayer

    layer = DoRALayer.__new__(DoRALayer)  # bypass __init__: the method only reads its arguments
    wq, _ = quantize_weight_convrot("w", torch.randn(N, K) * 0.02, CONVROT_GROUPSIZE)
    with pytest.raises(ValueError, match="DoRA weight norm"):
        layer.get_weight_norm_efficient(wq, torch.randn(R, K), torch.randn(N, R), 1.0)


def test_destructive_merge_refuses_convrot_base_and_leaves_it_byte_identical():
    """The regression this PR fixes: merge_to used to silently corrupt the int8 codes.

    Before the guard this call returned normally, rewrote ~half the int8 codes, and left the
    adapter contribution off by more than an order of magnitude.
    """
    m = _convrot_linear()
    before = m.l.weight.data.clone()
    inf = LoRAInfModule("l", m.l, multiplier=1.0, lora_dim=R, alpha=R)
    sd = {"lora_down.weight": torch.randn(R, K) * 0.05, "lora_up.weight": torch.randn(N, R) * 0.05}

    with pytest.raises(ValueError, match="merge_to"):
        inf.merge_to(sd, None, "cpu")

    assert torch.equal(m.l.weight.data, before), "guard must refuse BEFORE mutating the base weight"


def test_convrot_forward_still_works_and_is_unaffected_by_the_guards():
    """The guards must not touch the supported path: runtime forward on a ConvRot base."""
    m = _convrot_linear()
    x = torch.randn(2, K)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, N)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- layer 2: trainer fail-fast


def _args(**overrides):
    base = dict(
        fp8_base=False,
        fp8_scaled=False,
        convrot_int8=True,
        convrot_int8_bwd="bf16",
        turbo_dit=None,
        turbo_dit_cache=False,
        blocks_to_swap=0,
        sample_prompts=None,
        network_args=None,
        network_module="networks.lora_krea2",
        base_weights=None,
        base_weights_multiplier=None,
        dim_from_weights=None,
        network_weights=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _handle(args):
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    Krea2NetworkTrainer().handle_model_specific_args(args)


# Every spelling the SHARED parser treats as true. A hand-rolled ("true", "1") check missed the last four.
@pytest.mark.parametrize("value", ["True", "true", "1", "yes", "on", "YES", "On"])
def test_dora_rejected_for_every_truthy_spelling_the_real_parser_accepts(value):
    assert parse_bool_arg(value) is True, "test premise: the factory would enable DoRA for this value"
    with pytest.raises(ValueError, match="DoRA"):
        _handle(_args(network_args=[f"use_dora={value}"]))


@pytest.mark.parametrize("value", ["false", "no", "off", "0"])
def test_dora_falsy_spellings_are_allowed(value):
    assert parse_bool_arg(value) is False
    _handle(_args(network_args=[f"use_dora={value}"]))


# "pissa_niter_<N>" is PiSSA too; an `== "pissa"` check missed it.
@pytest.mark.parametrize("value", ["pissa", "PISSA", "pissa_niter_5", "pissa_niter_16"])
def test_pissa_rejected_including_the_niter_form(value):
    assert parse_init_lora_weights_arg(value).startswith("pissa"), "test premise: this is a PiSSA mode"
    with pytest.raises(ValueError, match="init_lora_weights"):
        _handle(_args(network_args=[f"init_lora_weights={value}"]))


@pytest.mark.parametrize("value", ["kaiming", "orthogonal", "true"])
def test_non_pissa_init_allowed(value):
    _handle(_args(network_args=[f"init_lora_weights={value}"]))


def test_lycoris_bridge_rejected():
    with pytest.raises(ValueError, match="LyCORIS"):
        _handle(_args(network_module="networks.lycoris"))


def test_base_weights_rejected_because_the_merge_runs_after_quantization():
    with pytest.raises(ValueError, match="base_weights"):
        _handle(_args(base_weights=["/tmp/some_adapter.safetensors"]))


@pytest.mark.parametrize("attr", ["dim_from_weights", "network_weights"])
def test_checkpoint_inferred_dora_rejected(tmp_path, attr):
    """DoRA can arrive with no use_dora network arg: it is inferred from checkpoint keys."""
    p = tmp_path / f"dora_{attr}.safetensors"
    save_file(
        {
            "lora_unet_l.lora_down.weight": torch.zeros(R, K),
            "lora_unet_l.lora_up.weight": torch.zeros(N, R),
            "lora_unet_l.dora_layer.weight": torch.ones(N, 1),
        },
        str(p),
    )
    with pytest.raises(ValueError, match="DoRA"):
        _handle(_args(**{attr: str(p)}))


@pytest.mark.parametrize("attr", ["dim_from_weights", "network_weights"])
def test_checkpoint_inferred_dora_honors_a_false_use_dora_flag(tmp_path, attr):
    p = tmp_path / f"plain_{attr}.safetensors"
    save_file(
        {
            "lora_unet_l.lora_down.weight": torch.zeros(R, K),
            "lora_unet_l.lora_up.weight": torch.zeros(N, R),
            "use_dora_flag": torch.tensor(False),
        },
        str(p),
    )
    _handle(_args(**{attr: str(p)}))


def test_plain_lora_checkpoint_allowed(tmp_path):
    p = tmp_path / "plain.safetensors"
    save_file({"lora_unet_l.lora_down.weight": torch.zeros(R, K)}, str(p))
    _handle(_args(network_weights=str(p)))


def test_unreadable_checkpoint_warns_but_does_not_block(tmp_path):
    """A bad path is the normal loader's error to report, not this guard's."""
    bad = tmp_path / "not_safetensors.bin"
    bad.write_bytes(b"garbage")
    _handle(_args(network_weights=str(bad)))


def test_guards_are_inert_without_convrot():
    """None of this may fire on the fp8 / bf16 paths, which support all of these adapters."""
    _handle(
        _args(
            convrot_int8=False,
            fp8_base=True,
            fp8_scaled=True,
            network_args=["use_dora=yes", "init_lora_weights=pissa_niter_5"],
            base_weights=["/tmp/whatever.safetensors"],
        )
    )


if __name__ == "__main__":
    pytest.main([__file__])


# --------------------------------- layer 1b: the OTHER destructive mergers
#
# LoRAInfModule.merge_to is not the only merger. LoHa and LoKr have wholly independent
# implementations, and merge_nonlora_to_model dispatches to three separate tensor-merging
# helpers. A review found all of these still corrupting ConvRot weights after the first
# round of guards, so each one is pinned here.
#
# These use the STRICTER raise_if_unmergeable_base, which also refuses fp8: a merge has to
# write its result back, and re-quantizing into fp8 needs inverse scaling these mergers do
# not implement. (LoHa/LoKr previously accepted an fp8 base and corrupted it too -- verified
# at 502/16384 and 13956/16384 codes changed before this guard.)


def _loha_sd():
    return {
        "hada_w1_a": torch.randn(N, R) * 0.05,
        "hada_w1_b": torch.randn(R, K) * 0.05,
        "hada_w2_a": torch.randn(N, R) * 0.05,
        "hada_w2_b": torch.randn(R, K) * 0.05,
        "alpha": torch.tensor(float(R)),
    }


def _lokr_sd():
    return {"lokr_w1": torch.randn(8, 8) * 0.05, "lokr_w2": torch.randn(N // 8, K // 8) * 0.05, "alpha": torch.tensor(float(R))}


def _lora_tensor_sd(prefix="lora_unet_l"):
    return {
        f"{prefix}.lora_down.weight": torch.randn(R, K) * 0.05,
        f"{prefix}.lora_up.weight": torch.randn(N, R) * 0.05,
        f"{prefix}.alpha": torch.tensor(float(R)),
    }


def test_loha_module_merge_refuses_convrot_base_without_mutating():
    from musubi_tuner.networks.loha import LoHaInfModule

    m = _convrot_linear()
    before = m.l.weight.data.clone()
    mod = LoHaInfModule("l", m.l, multiplier=1.0, lora_dim=R, alpha=R)
    with pytest.raises(ValueError, match="LoHa merge_to"):
        mod.merge_to(_loha_sd(), None, "cpu")
    assert torch.equal(m.l.weight.data, before)


def test_lokr_module_merge_refuses_convrot_base_without_mutating():
    from musubi_tuner.networks.lokr import LoKrInfModule

    m = _convrot_linear()
    before = m.l.weight.data.clone()
    mod = LoKrInfModule("l", m.l, multiplier=1.0, lora_dim=R, alpha=R)
    with pytest.raises(ValueError, match="LoKr merge_to"):
        mod.merge_to(_lokr_sd(), None, "cpu")
    assert torch.equal(m.l.weight.data, before)


def test_shared_dispatcher_refuses_convrot_base_even_with_safe_merge():
    """safe_merge only checks finiteness; arithmetic in the wrong representation is finite."""
    from musubi_tuner.utils.lora_utils import merge_nonlora_to_model

    m = _convrot_linear()
    before = m.l.weight.data.clone()
    with pytest.raises(ValueError, match="merge_weights_to_tensor"):
        merge_nonlora_to_model(m, _lora_tensor_sd(), multiplier=1.0, device="cpu", safe_merge=True)
    assert torch.equal(m.l.weight.data, before)


@pytest.mark.parametrize("which", ["loha", "lokr", "lora"])
def test_tensor_helpers_refuse_int8_when_keys_match(which):
    from musubi_tuner.networks.loha import merge_weights_to_tensor as loha_merge
    from musubi_tuner.networks.lokr import merge_weights_to_tensor as lokr_merge
    from musubi_tuner.utils.lora_utils import lora_merge_weights_to_tensor

    wq, _ = quantize_weight_convrot("w", torch.randn(N, K) * 0.02, CONVROT_GROUPSIZE)
    fn, sd = {
        "loha": (loha_merge, {f"lora_unet_l.{k}": v for k, v in _loha_sd().items()}),
        "lokr": (lokr_merge, {f"lora_unet_l.{k}": v for k, v in _lokr_sd().items()}),
        "lora": (lora_merge_weights_to_tensor, _lora_tensor_sd()),
    }[which]
    keys = set(sd.keys())
    n_before = len(keys)
    before = wq.clone()

    with pytest.raises(ValueError, match="merge_weights_to_tensor"):
        fn(wq, "lora_unet_l", sd, keys, 1.0, "cpu")

    # the reviewer's explicit requirements: refuse without touching weights OR key bookkeeping
    assert torch.equal(wq, before), "must not mutate the base weight"
    assert len(keys) == n_before, "must not consume adapter keys when refusing"


@pytest.mark.parametrize("which", ["loha", "lokr", "lora"])
def test_tensor_helpers_stay_noops_on_int8_when_no_keys_match(which):
    """merge_nonlora_to_model calls all three helpers for EVERY parameter, so a helper with no
    matching keys must stay a silent no-op even on a quantized weight -- otherwise merging a
    LoRA would fail with a misleading LoHa error."""
    from musubi_tuner.networks.loha import merge_weights_to_tensor as loha_merge
    from musubi_tuner.networks.lokr import merge_weights_to_tensor as lokr_merge
    from musubi_tuner.utils.lora_utils import lora_merge_weights_to_tensor

    fn = {"loha": loha_merge, "lokr": lokr_merge, "lora": lora_merge_weights_to_tensor}[which]
    wq, _ = quantize_weight_convrot("w", torch.randn(N, K) * 0.02, CONVROT_GROUPSIZE)
    keys: set = set()
    out = fn(wq, "lora_unet_nomatch", {}, keys, 1.0, "cpu")
    assert out is wq
    assert keys == set()


@pytest.mark.parametrize("which", ["loha", "lokr", "lora"])
def test_tensor_helpers_still_merge_a_float_base(which):
    """The guards must not block the supported path."""
    from musubi_tuner.networks.loha import merge_weights_to_tensor as loha_merge
    from musubi_tuner.networks.lokr import merge_weights_to_tensor as lokr_merge
    from musubi_tuner.utils.lora_utils import lora_merge_weights_to_tensor

    fn, sd = {
        "loha": (loha_merge, {f"lora_unet_l.{k}": v for k, v in _loha_sd().items()}),
        "lokr": (lokr_merge, {f"lora_unet_l.{k}": v for k, v in _lokr_sd().items()}),
        "lora": (lora_merge_weights_to_tensor, _lora_tensor_sd()),
    }[which]
    w = torch.randn(N, K) * 0.02
    keys = set(sd.keys())
    out = fn(w.clone(), "lora_unet_l", sd, keys, 1.0, "cpu")
    assert not torch.equal(out, w), "a matching adapter must actually be merged"
    assert torch.isfinite(out).all()
    assert len(keys) < len(sd), "consumed keys must be removed on the success path"


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="fp8 dtype unavailable")
def test_loha_and_lokr_merge_also_refuse_fp8_matching_lora_policy():
    """Pre-existing gap closed alongside: LoRA refused fp8 merges, LoHa/LoKr silently corrupted them."""
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8
    from musubi_tuner.networks.loha import LoHaInfModule
    from musubi_tuner.networks.lokr import LoKrInfModule

    for cls, sd_fn, pattern in ((LoHaInfModule, _loha_sd, "LoHa merge_to"), (LoKrInfModule, _lokr_sd, "LoKr merge_to")):

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(K, N, bias=False)

            def forward(self, x):
                return self.l(x)

        m = M()
        sd8 = optimize_state_dict_with_fp8({"l.weight": (torch.randn(N, K) * 0.02)}, None, ["l."], None)
        apply_fp8_monkey_patch(m, sd8, use_scaled_mm=False)
        m.load_state_dict(sd8, strict=True, assign=True)
        before = m.l.weight.data.clone()
        mod = cls("l", m.l, multiplier=1.0, lora_dim=R, alpha=R)
        with pytest.raises(ValueError, match=pattern):
            mod.merge_to(sd_fn(), None, "cpu")
        assert torch.equal(m.l.weight.data, before)
