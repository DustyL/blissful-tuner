"""Fail-fast guards for Krea 2 training/inference.

The fp8/fused-attention guard prevents a confusing mid-run flash/xformers kernel dtype
error (fp8 feeds fp32 to attention, which only torch SDPA accepts). The mask-loss
acceptance test is the inverse of the retired Phase-1 guard: as of the Phase 2 training
wiring (process_batch override + mask cache payload), --use_mask_loss is now SUPPORTED and
must NOT be rejected.
"""

import sys
from argparse import Namespace

import pytest


def test_krea2_training_accepts_use_mask_loss():
    """Phase 2: --use_mask_loss is now wired (process_batch override) and must NOT be rejected.

    Inverse of the retired Phase-1 fail-fast guard. handle_model_specific_args must accept it.
    """
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    trainer = Krea2NetworkTrainer()
    args = Namespace(
        use_mask_loss=True,
        fp8_base=False,
        fp8_scaled=False,
        turbo_dit=None,
        turbo_dit_cache=None,
        blocks_to_swap=0,
        sample_prompts=None,
    )
    # Must complete without raising — mask-weighted loss is supported in Phase 2.
    trainer.handle_model_specific_args(args)


def test_krea2_training_allows_no_mask_loss():
    """The guard must NOT fire when --use_mask_loss is absent/false (normal vanilla training)."""
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    trainer = Krea2NetworkTrainer()
    args = Namespace(
        use_mask_loss=False,
        fp8_base=False,
        fp8_scaled=False,
        turbo_dit=None,
        turbo_dit_cache=None,
        blocks_to_swap=0,
        sample_prompts=None,
    )
    # Should complete without raising (sets dtype/flags, no mask guard trip).
    trainer.handle_model_specific_args(args)


def _train_args(**over):
    base = dict(
        use_mask_loss=False,
        fp8_base=True,
        fp8_scaled=True,
        flash_attn=False,
        xformers=False,
        sage_attn=False,
        flash3=False,
        turbo_dit=None,
        turbo_dit_cache=None,
        blocks_to_swap=0,
        sample_prompts=None,
    )
    base.update(over)
    return Namespace(**base)


@pytest.mark.parametrize("flag", ["flash_attn", "xformers", "sage_attn", "flash3"])
def test_krea2_training_rejects_fp8_with_fused_attn(flag):
    """--fp8_scaled with any fused training backend must fail fast (verified: fp8 -> fp32 -> fused reject)."""
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    trainer = Krea2NetworkTrainer()
    with pytest.raises(ValueError, match="fp8_scaled"):
        trainer.handle_model_specific_args(_train_args(**{flag: True}))


def test_krea2_training_allows_fp8_with_sdpa():
    """--fp8_scaled with SDPA (no fused flag) is the validated training path and must be accepted."""
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    trainer = Krea2NetworkTrainer()
    trainer.handle_model_specific_args(_train_args())  # all fused flags False -> no raise


@pytest.mark.parametrize("backend", ["flash", "xformers", "sageattn"])
def test_krea2_generate_rejects_fp8_with_fused_attn(monkeypatch, backend):
    """--fp8_scaled with any fused backend must fail fast (fp8 feeds fp32 to attention)."""
    from musubi_tuner import krea2_generate_image as gen

    argv = [
        "krea2_generate_image",
        "a prompt",
        "--dit",
        "/dev/null",
        "--vae",
        "/dev/null",
        "--text_encoder",
        "/dev/null",
        "--save_path",
        "/dev/null",
        "--fp8_scaled",
        "--attn_mode",
        backend,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="fp8_scaled"):
        gen.parse_args()


def test_krea2_generate_allows_fp8_with_torch(monkeypatch):
    """--fp8_scaled --attn_mode torch is the validated path and must parse cleanly."""
    from musubi_tuner import krea2_generate_image as gen

    argv = [
        "krea2_generate_image",
        "a prompt",
        "--dit",
        "/dev/null",
        "--vae",
        "/dev/null",
        "--text_encoder",
        "/dev/null",
        "--save_path",
        "/dev/null",
        "--fp8_scaled",
        "--attn_mode",
        "torch",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = gen.parse_args()
    assert args.fp8_scaled and args.attn_mode == "torch"
