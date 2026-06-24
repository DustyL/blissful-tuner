"""Phase 1 fail-fast guards for Krea 2.

Krea 2's Phase 1 is inference-parity only: mask-weighted loss / prior preservation are not
yet wired into training, and fp8 is incompatible with the fused attention backends. These
guards prevent silently-wrong runs (deceptive vanilla "masked" training; a confusing
mid-run flash/xformers kernel dtype error). They must keep firing until the real support
lands.
"""

import sys
from argparse import Namespace

import pytest


def test_krea2_training_rejects_use_mask_loss():
    """--use_mask_loss on Krea 2 must fail fast (it is not wired, would silently run vanilla MSE)."""
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
    with pytest.raises(ValueError, match="use_mask_loss"):
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
