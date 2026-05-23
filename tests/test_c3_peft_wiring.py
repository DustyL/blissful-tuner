"""Guard tests for the C3 PEFT / base-hash wiring onto the refactored NetworkTrainer base.

After the upstream v0.3.0 refactor, two PEFT call-sites that the pre-merge monolith owned
in its training loop were rewired onto the new ``trainer_base.NetworkTrainer`` seams:

1. ``validate_pissa_training_args(args)`` is called inside ``_validate_args_and_init`` (right
   after the ``sage_attn`` rejection) so unsafe PiSSA combinations fail-fast BEFORE any heavy
   model/dataset load. It's lazy-imported from ``hv_train_network`` to dodge the
   trainer_base <-> hv_train_network import cycle.
2. ``NetworkTrainer.extra_metadata(args)`` injects ``ss_base_sha256`` for the LoRA path (the
   base-provenance hash used by hotswap validation), computed once at metadata assembly.

These wirings call already-tested helpers, so the risk isn't the helpers — it's the *wiring*
silently regressing on a future upstream merge (exactly how the merge that prompted this lost
``add_mask_loss_args`` / ``validate_mask_loss_args``). Both tests mock the underlying helpers so
nothing touches multi-GB model files or launches training.
"""

import argparse

import pytest

from musubi_tuner.training.trainer_base import NetworkTrainer


class _MinimalTrainer(NetworkTrainer):
    """Concrete NetworkTrainer with only what these tests touch; no model/dataset machinery."""

    def handle_model_specific_args(self, args):  # reached only if validation proceeds past PiSSA
        self._i2v_training = False
        self._control_training = False


def _validation_args(**overrides) -> argparse.Namespace:
    """Args that pass every check in ``_validate_args_and_init`` up to the PiSSA call."""
    base = dict(
        cuda_allow_tf32=False,
        cuda_cudnn_benchmark=False,
        dataset_config="dummy.toml",
        dit="dummy.safetensors",
        fp8_scaled=False,
        fp8_base=False,
        sage_attn=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_validate_args_and_init_calls_validate_pissa_pre_load(monkeypatch):
    """PiSSA validation must fire inside _validate_args_and_init, before model/dataset load."""
    import musubi_tuner.hv_train_network as hv

    class _Sentinel(Exception):
        pass

    calls = []

    def spy(args):
        calls.append(args)
        # Raise so we prove the call happens in the early path and execution stops BEFORE
        # handle_model_specific_args / any load — no need to satisfy later validation steps.
        raise _Sentinel

    # validate_pissa is lazy-imported from hv_train_network inside the method; patch it there.
    monkeypatch.setattr(hv, "validate_pissa_training_args", spy)

    trainer = _MinimalTrainer()
    args = _validation_args()
    with pytest.raises(_Sentinel):
        trainer._validate_args_and_init(args)

    assert len(calls) == 1, "validate_pissa_training_args must be called once during _validate_args_and_init"
    assert calls[0] is args


def test_validate_pissa_runs_after_sage_attn_rejection(monkeypatch):
    """sage_attn is rejected before PiSSA validation is reached (ordering guard)."""
    import musubi_tuner.hv_train_network as hv

    called = []
    monkeypatch.setattr(hv, "validate_pissa_training_args", lambda args: called.append(args))

    trainer = _MinimalTrainer()
    # sage_attn=True must raise its own ValueError before PiSSA validation runs.
    with pytest.raises(ValueError, match="SageAttention"):
        trainer._validate_args_and_init(_validation_args(sage_attn=True))
    assert called == [], "PiSSA validation must not run when sage_attn already rejected the args"


def test_extra_metadata_injects_base_sha256(monkeypatch):
    """Base extra_metadata wires ss_base_sha256 from the (mocked) base-hash helper."""
    import musubi_tuner.utils.lora_utils as lu

    monkeypatch.setattr(lu, "compute_and_log_base_sha256", lambda args: "deadbeefcafe0000")

    md = _MinimalTrainer().extra_metadata(argparse.Namespace())
    assert md.get("ss_base_sha256") == "deadbeefcafe0000"


def test_extra_metadata_omits_base_sha256_when_deferred(monkeypatch):
    """When the hash is not computable (e.g. WAN dual-expert deferral → None), the key is absent."""
    import musubi_tuner.utils.lora_utils as lu

    monkeypatch.setattr(lu, "compute_and_log_base_sha256", lambda args: None)

    md = _MinimalTrainer().extra_metadata(argparse.Namespace())
    assert "ss_base_sha256" not in md
