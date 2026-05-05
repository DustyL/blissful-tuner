"""Fail-fast PiSSA + incompatible-flag rejects at training start.

Pins validate_pissa_training_args (in hv_train_network.py) — the
single function that closes three training-time PiSSA footguns
before any model/dataset loading happens:

  - WAN dual-expert (args.dit_high_noise) + PiSSA
  - --resume + PiSSA
  - use_dora=True in --network_args + PiSSA

Each reject fires on the user's INTENT (init_lora_weights="pissa*"
plus the conflicting flag), so even cases where downstream code would
have silently disabled one path get a clear contract message at
training start instead of a confused runtime failure 30 seconds in.

These tests do NOT exercise the trainer.train() loop — that would
require accelerator setup + dataset construction + model load. They
exercise the helper directly with synthetic argparse.Namespace
fixtures, mirroring the surface of TestInjectSsBaseSha256Metadata
in test_compute_training_base_hash.py.
"""

from __future__ import annotations

import argparse
import unittest

from musubi_tuner.hv_train_network import (
    _network_args_bool,
    _network_args_init_lora_weights,
    validate_pissa_training_args,
)


def _ns(**kwargs) -> argparse.Namespace:
    """argparse.Namespace with sensible defaults for the three flags the
    validator reads: network_args, dit_high_noise, resume. Tests override
    only what they need to assert."""
    base = dict(network_args=None, dit_high_noise=None, resume=None)
    base.update(kwargs)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Helper extraction (the parsing primitives)
# ---------------------------------------------------------------------------


class TestNetworkArgsInitLoraWeightsExtraction(unittest.TestCase):
    """The detection predicate must align with what LoRAModule does."""

    def test_pissa_value_extracted(self) -> None:
        self.assertEqual(_network_args_init_lora_weights(["init_lora_weights=pissa"]), "pissa")

    def test_pissa_niter_value_extracted(self) -> None:
        self.assertEqual(
            _network_args_init_lora_weights(["init_lora_weights=pissa_niter_5"]),
            "pissa_niter_5",
        )

    def test_kaiming_value_extracted(self) -> None:
        self.assertEqual(_network_args_init_lora_weights(["init_lora_weights=kaiming"]), "kaiming")

    def test_orthogonal_value_extracted(self) -> None:
        self.assertEqual(_network_args_init_lora_weights(["init_lora_weights=orthogonal"]), "orthogonal")

    def test_default_kaiming_when_unset(self) -> None:
        self.assertEqual(_network_args_init_lora_weights(None), "kaiming")
        self.assertEqual(_network_args_init_lora_weights([]), "kaiming")
        self.assertEqual(_network_args_init_lora_weights(["other_arg=value"]), "kaiming")

    def test_routes_through_canonical_parse_helper(self) -> None:
        """Garbage init values raise via parse_init_lora_weights_arg —
        same predicate path LoRAModule uses, so the validator can never
        accept a value LoRAModule would reject (or vice versa)."""
        with self.assertRaises(ValueError):
            _network_args_init_lora_weights(["init_lora_weights=garbage_init"])


class TestNetworkArgsBoolExtraction(unittest.TestCase):
    """The bool extractor handles the same true/false token vocabulary
    parse_bool_arg supports."""

    def test_use_dora_true(self) -> None:
        for value in ("true", "True", "1", "yes", "on"):
            with self.subTest(value=value):
                self.assertTrue(_network_args_bool([f"use_dora={value}"], "use_dora"))

    def test_use_dora_false(self) -> None:
        for value in ("false", "False", "0", "no", "off"):
            with self.subTest(value=value):
                self.assertFalse(_network_args_bool([f"use_dora={value}"], "use_dora"))

    def test_use_dora_absent_defaults_false(self) -> None:
        self.assertFalse(_network_args_bool(None, "use_dora"))
        self.assertFalse(_network_args_bool([], "use_dora"))
        self.assertFalse(_network_args_bool(["other_arg=value"], "use_dora"))


# ---------------------------------------------------------------------------
# validate_pissa_training_args — the contract surface
# ---------------------------------------------------------------------------


class TestPissaTrainingValidationNoOpForNonPissa(unittest.TestCase):
    """Non-PiSSA training is COMPLETELY untouched — no logging, no warning,
    no exception, no surprise. Pin this so a future refactor that "tightens"
    the validator can't silently break every existing kaiming/orthogonal
    training run."""

    def test_kaiming_with_dit_high_noise_no_op(self) -> None:
        args = _ns(network_args=["init_lora_weights=kaiming"], dit_high_noise="/path/to/high.safetensors")
        validate_pissa_training_args(args)  # must not raise

    def test_kaiming_with_resume_no_op(self) -> None:
        args = _ns(network_args=["init_lora_weights=kaiming"], resume="/path/to/checkpoint")
        validate_pissa_training_args(args)

    def test_kaiming_with_use_dora_no_op(self) -> None:
        args = _ns(network_args=["init_lora_weights=kaiming", "use_dora=true"])
        validate_pissa_training_args(args)

    def test_default_init_no_op(self) -> None:
        """No init_lora_weights specified at all = kaiming default. WAN dual-
        expert + DoRA is a perfectly legal combination today."""
        args = _ns(network_args=["use_dora=true"], dit_high_noise="/path/to/high.safetensors")
        validate_pissa_training_args(args)

    def test_orthogonal_with_all_conflicts_no_op(self) -> None:
        """Orthogonal init has its own constraints (Conv2d fallback, even-rank)
        but is not coupled to base hash, so none of the PiSSA rejects apply."""
        args = _ns(
            network_args=["init_lora_weights=orthogonal", "use_dora=true"],
            dit_high_noise="/path/to/high.safetensors",
            resume="/path/to/ckpt",
        )
        validate_pissa_training_args(args)

    def test_no_network_args_no_op(self) -> None:
        """args.network_args may be None entirely (default trainer construction)."""
        args = _ns(network_args=None, dit_high_noise="/path/to/high.safetensors", resume="/path/to/ckpt")
        validate_pissa_training_args(args)


class TestPissaTrainingRejectsWanDualExpert(unittest.TestCase):
    """The WAN dual-expert reject — the deferred-to-Tier-2-#6a-2 footgun."""

    def test_pissa_with_dit_high_noise_raises(self) -> None:
        args = _ns(
            network_args=["init_lora_weights=pissa"],
            dit_high_noise="/path/to/high.safetensors",
        )
        with self.assertRaises(ValueError) as ctx:
            validate_pissa_training_args(args)
        msg = str(ctx.exception)
        self.assertIn("WAN dual-expert", msg)
        self.assertIn("Tier 2 #6a-2", msg)
        self.assertIn("pissa", msg)

    def test_pissa_niter_with_dit_high_noise_raises(self) -> None:
        """pissa_niter_<N> spelling routes through the same reject."""
        args = _ns(
            network_args=["init_lora_weights=pissa_niter_5"],
            dit_high_noise="/path/to/high.safetensors",
        )
        with self.assertRaisesRegex(ValueError, "WAN dual-expert"):
            validate_pissa_training_args(args)

    def test_pissa_with_empty_dit_high_noise_no_op(self) -> None:
        """Empty-string dit_high_noise is the same as None per the
        Tier 2 #6a empty-string defense — pin the same predicate here."""
        args = _ns(network_args=["init_lora_weights=pissa"], dit_high_noise="")
        validate_pissa_training_args(args)  # must not raise


class TestPissaTrainingRejectsResume(unittest.TestCase):
    """The --resume + PiSSA reject — re-residualization-of-already-residualized
    base footgun."""

    def test_pissa_with_resume_raises(self) -> None:
        args = _ns(network_args=["init_lora_weights=pissa"], resume="/path/to/ckpt")
        with self.assertRaises(ValueError) as ctx:
            validate_pissa_training_args(args)
        msg = str(ctx.exception)
        self.assertIn("--resume", msg)
        self.assertIn("re-residualize", msg)
        self.assertIn("Tier 2 #6c", msg)  # pointer to future converter tool

    def test_pissa_niter_with_resume_raises(self) -> None:
        args = _ns(network_args=["init_lora_weights=pissa_niter_8"], resume="/some/state")
        with self.assertRaisesRegex(ValueError, "--resume"):
            validate_pissa_training_args(args)

    def test_pissa_with_empty_resume_no_op(self) -> None:
        """Empty-string resume should be treated as falsy (no resume)."""
        args = _ns(network_args=["init_lora_weights=pissa"], resume="")
        validate_pissa_training_args(args)


class TestPissaTrainingRejectsDora(unittest.TestCase):
    """The DoRA + PiSSA reject — defense in depth (LoRAModule.__init__ also
    catches this at module construction, but firing earlier surfaces the
    error before model loading)."""

    def test_pissa_with_use_dora_true_raises(self) -> None:
        args = _ns(network_args=["init_lora_weights=pissa", "use_dora=true"])
        with self.assertRaises(ValueError) as ctx:
            validate_pissa_training_args(args)
        msg = str(ctx.exception)
        self.assertIn("use_dora", msg)
        self.assertIn("pissa", msg)
        self.assertIn("Tier 2 #6d", msg)  # pointer to pissa_decompose_dora

    def test_pissa_with_use_dora_false_no_op(self) -> None:
        """use_dora=false means DoRA is explicitly off — no conflict."""
        args = _ns(network_args=["init_lora_weights=pissa", "use_dora=false"])
        validate_pissa_training_args(args)


class TestPissaTrainingRejectsCombined(unittest.TestCase):
    """Multiple conflicts at once: the validator fires on the first one
    encountered (WAN dual-expert), since the rejects are independent
    safety-contract failures."""

    def test_all_three_conflicts_raises_first(self) -> None:
        """Order: WAN check first, then resume, then use_dora. The first-
        caught determinism makes the error message predictable for users
        who hit multiple footguns at once."""
        args = _ns(
            network_args=["init_lora_weights=pissa", "use_dora=true"],
            dit_high_noise="/high.safetensors",
            resume="/ckpt",
        )
        with self.assertRaisesRegex(ValueError, "WAN dual-expert"):
            validate_pissa_training_args(args)

    def test_resume_and_use_dora_raises_resume_first(self) -> None:
        args = _ns(
            network_args=["init_lora_weights=pissa", "use_dora=true"],
            resume="/ckpt",
        )
        with self.assertRaisesRegex(ValueError, "--resume"):
            validate_pissa_training_args(args)


if __name__ == "__main__":
    unittest.main()
