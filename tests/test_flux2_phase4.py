"""Tests for FLUX.2 Phase 4 fixes: W12, W11, W10.

W12: Dead --i (image_path) handler removed from parse_prompt_line
W11: VAE cached in interactive mode (loaded once, not per-iteration)
W10: Prompt wildcard support via --prompt_wildcards
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock

import torch

from musubi_tuner.flux_2_generate_image import parse_prompt_line


# ── W12: Dead --i handler → warning ────────────────────────────────────────


def test_w12_image_path_override_not_set():
    """W12: --i shorthand should warn and NOT set image_path in overrides."""
    overrides = parse_prompt_line("a dog --i /path/to/img.png")
    assert "image_path" not in overrides, "--i should not produce an image_path override"


def test_w12_i_shorthand_emits_warning_with_value(caplog):
    """W12: --i should emit a warning that includes the provided value."""
    with caplog.at_level(logging.WARNING):
        parse_prompt_line("a dog --i /path/to/img.png")
    warnings = [r.message for r in caplog.records if "image_path" in r.message]
    assert len(warnings) == 1, f"Expected exactly one image_path warning, got {warnings}"
    assert "/path/to/img.png" in warnings[0], f"Warning should include the value: {warnings[0]}"


def test_w12_other_shorthands_still_work():
    """W12: Removing --i should not affect other shorthands."""
    overrides = parse_prompt_line("a dog --w 512 --h 768 --d 42 --s 10 --g 3.5 --e 2.0 --n bad quality")
    assert overrides["prompt"] == "a dog"
    assert overrides["image_size_width"] == 512
    assert overrides["image_size_height"] == 768
    assert overrides["seed"] == 42
    assert overrides["infer_steps"] == 10
    assert overrides["guidance_scale"] == 3.5
    assert overrides["embedded_cfg_scale"] == 2.0
    assert overrides["negative_prompt"] == "bad quality"


def test_w12_control_image_path_still_works():
    """W12: --ci shorthand for control images should still work."""
    overrides = parse_prompt_line("a dog --ci /path/a.png --ci /path/b.png")
    assert overrides["control_image_path"] == ["/path/a.png", "/path/b.png"]


# ── W11: VAE cached in interactive mode ─────────────────────────────────────


def _make_fake_ae():
    """Create a fake AE that tracks .to() calls and can be returned by generate()."""
    ae = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    def ae_to(*args, **kwargs):
        if args:
            ae.device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
        return ae

    ae.to = ae_to
    return ae


def test_w11_interactive_loads_vae_once_not_per_iteration():
    """W11: In interactive mode, load_ae should be called exactly once (before loop),
    and generate() should NOT call load_ae again during iterations."""
    from musubi_tuner import flux_2_generate_image

    load_ae_calls = []
    fake_ae = _make_fake_ae()

    def fake_load_ae(*args, **kwargs):
        load_ae_calls.append(1)
        return fake_ae

    fake_model_info = SimpleNamespace(
        qwen_variant=None,
        guidance_distilled=True,
        defaults={"num_steps": 50, "guidance": 4.0},
        fixed_params=set(),
    )

    args = SimpleNamespace(
        device="cpu",
        fp8=False,
        fp8_scaled=False,
        fp8_text_encoder=False,
        blocks_to_swap=0,
        compile=False,
        model_version="dev",
        vae="/fake/vae",
        text_encoder="/fake/te",
        prompt_wildcards=None,
        # Needed by apply_model_defaults_and_enforce_fixed_params (called via apply_overrides)
        infer_steps=None,
        embedded_cfg_scale=None,
        guidance_scale=None,
        image_size=[64, 64],
        prompt=None,
        negative_prompt=None,
        seed=42,
    )

    # Fake generate that verifies shared_models has "ae" and does NOT call load_ae
    generate_calls = []

    def fake_generate(prompt_args, gen_settings, shared_models=None):
        generate_calls.append(1)
        assert shared_models is not None and "ae" in shared_models, "shared_models should contain 'ae'"
        assert shared_models["ae"] is fake_ae, "shared_models['ae'] should be the pre-loaded VAE"
        # Return (ae, fake_latent) matching real signature
        fake_latent = [torch.zeros(1, 128, 4, 4)]
        return fake_ae, fake_latent

    # Simulate two interactive iterations then exit
    user_inputs = iter(["a dog", "a cat", EOFError()])

    def fake_input(prompt):
        val = next(user_inputs)
        if isinstance(val, Exception):
            raise val
        return val

    import sys

    with (
        mock.patch.object(flux_2_generate_image.flux2_utils, "load_ae", side_effect=fake_load_ae),
        mock.patch.object(flux_2_generate_image.flux2_utils, "FLUX2_MODEL_INFO", {"dev": fake_model_info}),
        mock.patch.object(flux_2_generate_image, "load_shared_models", return_value={}),
        mock.patch.object(
            flux_2_generate_image,
            "get_generation_settings",
            return_value=SimpleNamespace(device=torch.device("cpu"), dit_weight_dtype=None),
        ),
        mock.patch.object(flux_2_generate_image, "generate", side_effect=fake_generate),
        mock.patch.object(flux_2_generate_image, "save_output"),
        mock.patch("builtins.input", side_effect=fake_input),
        # Force prompt_toolkit import to fail so process_interactive uses builtins.input
        mock.patch.dict(sys.modules, {"prompt_toolkit": None}),
    ):
        flux_2_generate_image.process_interactive(args)

    # VAE loaded exactly once (before loop), not per iteration
    assert len(load_ae_calls) == 1, f"Expected load_ae called once, got {len(load_ae_calls)}"
    # generate() was called for both prompts
    assert len(generate_calls) == 2, f"Expected 2 generate calls, got {len(generate_calls)}"


def test_w11_get_or_load_ae_returns_cached():
    """W11: get_or_load_ae must return shared_models['ae'] without calling load_ae."""
    from musubi_tuner import flux_2_generate_image
    from musubi_tuner.flux_2_generate_image import get_or_load_ae

    fake_ae = _make_fake_ae()
    shared_models = {"ae": fake_ae}
    args = SimpleNamespace(vae="/fake/vae")

    with mock.patch.object(flux_2_generate_image.flux2_utils, "load_ae") as mock_load:
        result = get_or_load_ae(args, torch.device("cpu"), shared_models)

    assert result is fake_ae, "Should return the cached AE"
    mock_load.assert_not_called()


def test_w11_get_or_load_ae_loads_when_missing():
    """W11: get_or_load_ae must call load_ae with correct args when shared_models is None."""
    from musubi_tuner import flux_2_generate_image
    from musubi_tuner.flux_2_generate_image import get_or_load_ae

    fresh_ae = _make_fake_ae()
    args = SimpleNamespace(vae="/fake/vae")
    device = torch.device("cpu")

    with mock.patch.object(flux_2_generate_image.flux2_utils, "load_ae", return_value=fresh_ae) as mock_load:
        result = get_or_load_ae(args, device, shared_models=None)

    assert result is fresh_ae
    mock_load.assert_called_once_with("/fake/vae", dtype=torch.float32, device=device, disable_mmap=True)


def test_w11_get_or_load_ae_loads_when_empty_shared():
    """W11: get_or_load_ae must call load_ae when shared_models exists but has no 'ae' key."""
    from musubi_tuner import flux_2_generate_image
    from musubi_tuner.flux_2_generate_image import get_or_load_ae

    fresh_ae = _make_fake_ae()
    args = SimpleNamespace(vae="/models/ae.safetensors")
    device = torch.device("cpu")

    with mock.patch.object(flux_2_generate_image.flux2_utils, "load_ae", return_value=fresh_ae) as mock_load:
        result = get_or_load_ae(args, device, shared_models={"text_embedder": "something"})

    assert result is fresh_ae
    mock_load.assert_called_once_with("/models/ae.safetensors", dtype=torch.float32, device=device, disable_mmap=True)


# ── W10: Prompt wildcard support ────────────────────────────────────────────


def test_w10_wildcards_applied_in_parse_prompt_line(tmp_path):
    """W10: parse_prompt_line should resolve __keyword__ wildcards."""
    wildcard_dir = tmp_path / "wildcards"
    wildcard_dir.mkdir()
    (wildcard_dir / "color.txt").write_text("blue\n")  # Single option → deterministic

    overrides = parse_prompt_line("a __color__ sky", str(wildcard_dir))
    assert overrides["prompt"] == "a blue sky", f"Expected 'a blue sky', got '{overrides['prompt']}'"


def test_w10_wildcards_applied_to_negative_prompt(tmp_path):
    """W10: Wildcards should also resolve in negative prompts via --n shorthand."""
    wildcard_dir = tmp_path / "wildcards"
    wildcard_dir.mkdir()
    (wildcard_dir / "quality.txt").write_text("blurry\n")

    overrides = parse_prompt_line("a dog --n __quality__ image", str(wildcard_dir))
    assert overrides["negative_prompt"] == "blurry image"


def test_w10_wildcards_disabled_when_none():
    """W10: With prompt_wildcards=None, __keyword__ tokens should pass through unchanged."""
    overrides = parse_prompt_line("a __color__ sky", None)
    assert overrides["prompt"] == "a __color__ sky"


def test_w10_prompt_wildcards_arg_exists():
    """W10: --prompt_wildcards should be accepted by parse_args."""
    from musubi_tuner.flux_2_generate_image import parse_args
    import sys

    test_args = [
        "--dit",
        "/fake",
        "--text_encoder",
        "/fake",
        "--save_path",
        "/fake",
        "--prompt",
        "test",
        "--prompt_wildcards",
        "/fake/wildcards",
    ]

    with mock.patch.object(sys, "argv", ["prog"] + test_args):
        with mock.patch(
            "musubi_tuner.flux_2_generate_image.flux2_utils.FLUX2_MODEL_INFO",
            {
                "dev": SimpleNamespace(
                    qwen_variant=None,
                    guidance_distilled=True,
                    defaults={"num_steps": 50, "guidance": 4.0},
                    fixed_params=set(),
                ),
            },
        ):
            with mock.patch(
                "musubi_tuner.flux_2_generate_image.flux2_utils.add_model_version_args",
                lambda p: p.add_argument("--model_version", default="dev"),
            ):
                with mock.patch("musubi_tuner.flux_2_generate_image.validate_lycoris_arg"):
                    args = parse_args()
                    assert args.prompt_wildcards == "/fake/wildcards"
