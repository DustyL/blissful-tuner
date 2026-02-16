"""Tests for Z-Image sample inference defaults (shift, cfg_scale, --g compat shim)."""

import logging
from unittest.mock import MagicMock, patch

import torch

from musubi_tuner.zimage_train_network import ZImageNetworkTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer():
    """Create a ZImageNetworkTrainer with minimal state for testing sample defaults."""
    trainer = ZImageNetworkTrainer.__new__(ZImageNetworkTrainer)
    # Set attributes normally set by handle_model_specific_args and __init__
    trainer.default_guidance_scale = 0.0
    trainer.default_discrete_flow_shift = 3.0
    trainer._i2v_training = False
    trainer._control_training = False
    trainer.vae_frame_stride = 4
    return trainer


def _make_base_trainer():
    """Create a base NetworkTrainer-like object WITHOUT default_discrete_flow_shift."""
    trainer = ZImageNetworkTrainer.__new__(ZImageNetworkTrainer)
    trainer.default_guidance_scale = 6.0
    trainer._i2v_training = False
    trainer._control_training = False
    trainer.vae_frame_stride = 4
    # Intentionally do NOT set default_discrete_flow_shift
    return trainer


def _make_explicit_none_trainer():
    """Create a trainer with default_discrete_flow_shift explicitly set to None (FLUX.2 pattern)."""
    trainer = ZImageNetworkTrainer.__new__(ZImageNetworkTrainer)
    trainer.default_guidance_scale = 0.0
    trainer.default_discrete_flow_shift = None
    trainer._i2v_training = False
    trainer._control_training = False
    trainer.vae_frame_stride = 4
    return trainer


def _capture_do_inference_call(trainer, sample_parameter):
    """Call sample_image_inference with mocks, capture args passed to do_inference."""
    captured = {}

    def fake_do_inference(
        self,
        accelerator,
        args,
        sample_param,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_cfg,
        guidance_scale,
        cfg_scale,
        **kwargs,
    ):
        captured["discrete_flow_shift"] = discrete_flow_shift
        captured["guidance_scale"] = guidance_scale
        captured["cfg_scale"] = cfg_scale
        captured["do_classifier_free_guidance"] = do_cfg
        return None  # no video

    mock_accelerator = MagicMock()
    mock_accelerator.device = "cpu"
    mock_args = MagicMock()
    mock_args.output_name = None

    with patch.object(type(trainer), "do_inference", fake_do_inference):
        trainer.sample_image_inference(
            mock_accelerator,
            mock_args,
            MagicMock(),  # transformer
            "bf16",  # dit_dtype
            MagicMock(),  # vae
            "/tmp",  # save_dir
            sample_parameter,
            epoch=1,
            steps=100,
        )

    return captured


def _call_do_inference_with_mocks(trainer, sample_parameter, cfg_scale=None, guidance_scale=0.0):
    """Call do_inference on the real production code with mocked internals.

    Patches zimage_utils functions and the transformer forward so do_inference
    runs through the cfg_scale resolution logic without needing real models.
    Returns (resolved_cfg_scale, do_cfg) captured from local state via a patched
    transformer call.
    """
    captured = {}

    # Minimal mock model
    mock_model = MagicMock()
    mock_model.dtype = torch.bfloat16
    mock_model.in_channels = 4
    mock_model.all_patch_size = [2]

    mock_accelerator = MagicMock()
    mock_accelerator.device = torch.device("cpu")
    mock_accelerator.unwrap_model.return_value = mock_model

    # Make .to() calls on tensors return tensors
    dummy_embed = torch.randn(1, 8, 16)
    dummy_mask = torch.ones(1, 8, dtype=torch.bool)

    sample_parameter.setdefault("cap_feats", dummy_embed)
    sample_parameter.setdefault("cap_mask", dummy_mask)
    sample_parameter.setdefault("negative_cap_feats", dummy_embed.clone())
    sample_parameter.setdefault("negative_cap_mask", dummy_mask.clone())

    def fake_trim_pad(seq_len, embed, mask):
        return embed, mask

    def fake_get_timesteps(steps, shift):
        # Return empty timesteps so the loop body never runs
        return torch.tensor([]), torch.tensor([])

    # VAE needs a real dtype and decode method for the post-loop decode step
    mock_vae = MagicMock()
    mock_vae.dtype = torch.bfloat16
    mock_vae.decode.return_value = torch.randn(1, 3, 256, 256)

    with (
        patch("musubi_tuner.zimage_train_network.zimage_utils.trim_pad_embeds_and_mask", side_effect=fake_trim_pad),
        patch("musubi_tuner.zimage_train_network.zimage_utils.get_timesteps_sigmas", side_effect=fake_get_timesteps),
        patch("musubi_tuner.zimage_train_network.zimage_utils.shift_scale_latents_for_decode", side_effect=lambda x: x),
    ):
        result = trainer.do_inference(
            mock_accelerator,
            MagicMock(),  # args
            sample_parameter,
            mock_vae,
            "bf16",  # dit_dtype
            MagicMock(),  # transformer
            3.0,  # discrete_flow_shift
            20,  # sample_steps
            256,  # width
            256,  # height
            1,  # frame_count
            torch.Generator().manual_seed(42),  # generator
            False,  # do_classifier_free_guidance (base trainer flag, Z-Image re-derives)
            guidance_scale,
            cfg_scale,
        )

    return result


# ---------------------------------------------------------------------------
# H1: default_discrete_flow_shift
# ---------------------------------------------------------------------------


class TestDefaultDiscreteFlowShift:
    def test_zimage_uses_3_when_prompt_omits_fs(self):
        """Z-Image trainer should use 3.0 shift when prompt doesn't specify --fs."""
        trainer = _make_trainer()
        captured = _capture_do_inference_call(trainer, {"prompt": "test"})
        assert captured["discrete_flow_shift"] == 3.0

    def test_prompt_fs_overrides_default(self):
        """Explicit --fs in prompt should override the architecture default."""
        trainer = _make_trainer()
        captured = _capture_do_inference_call(trainer, {"prompt": "test", "discrete_flow_shift": 7.5})
        assert captured["discrete_flow_shift"] == 7.5

    def test_base_trainer_falls_back_to_14_5(self):
        """Trainer without default_discrete_flow_shift should fall back to 14.5."""
        trainer = _make_base_trainer()
        captured = _capture_do_inference_call(trainer, {"prompt": "test"})
        assert captured["discrete_flow_shift"] == 14.5

    def test_zero_shift_not_treated_as_falsy(self):
        """A default_discrete_flow_shift of 0.0 should be used, not treated as falsy."""
        trainer = _make_trainer()
        trainer.default_discrete_flow_shift = 0.0
        captured = _capture_do_inference_call(trainer, {"prompt": "test"})
        assert captured["discrete_flow_shift"] == 0.0

    def test_explicit_none_preserved_not_collapsed_to_14_5(self):
        """FLUX.2 sets default_discrete_flow_shift=None explicitly. This should pass None, not 14.5."""
        trainer = _make_explicit_none_trainer()
        captured = _capture_do_inference_call(trainer, {"prompt": "test"})
        assert captured["discrete_flow_shift"] is None


# ---------------------------------------------------------------------------
# M2 + L3: cfg_scale defaults and --g compat shim (via production do_inference)
# ---------------------------------------------------------------------------


class TestCfgScaleDefaults:
    def test_cfg_scale_passed_through(self):
        """Explicit --l value should be passed through to do_inference."""
        trainer = _make_trainer()
        captured = _capture_do_inference_call(trainer, {"prompt": "test", "cfg_scale": 7.0, "negative_prompt": "bad"})
        assert captured["cfg_scale"] == 7.0

    def test_cfg_scale_none_when_not_specified(self):
        """When no --l specified, cfg_scale should be None (architecture handles default)."""
        trainer = _make_trainer()
        captured = _capture_do_inference_call(trainer, {"prompt": "test"})
        assert captured["cfg_scale"] is None


class TestGuidanceScaleCompatShim:
    def test_g_without_l_warns_and_maps(self, caplog):
        """When --g is set but --l is not, Z-Image do_inference should warn and use it as cfg_scale."""
        trainer = _make_trainer()
        sample_param = {"guidance_scale": 5.0}  # explicit --g in prompt dict

        with caplog.at_level(logging.WARNING, logger="musubi_tuner.zimage_train_network"):
            _call_do_inference_with_mocks(trainer, sample_param, cfg_scale=None, guidance_scale=5.0)

        assert "use `--l` for cfg_scale" in caplog.text

    def test_g_ignored_when_l_present(self, caplog):
        """When both --g and --l are set, --l takes precedence (no compat warning)."""
        trainer = _make_trainer()
        sample_param = {"guidance_scale": 5.0}

        with caplog.at_level(logging.WARNING, logger="musubi_tuner.zimage_train_network"):
            _call_do_inference_with_mocks(trainer, sample_param, cfg_scale=7.0, guidance_scale=5.0)

        assert "use `--l` for cfg_scale" not in caplog.text

    def test_g_not_in_dict_no_warning(self, caplog):
        """When --g was NOT in the prompt, no compat warning even with default guidance_scale."""
        trainer = _make_trainer()
        sample_param = {}  # no explicit guidance_scale key

        with caplog.at_level(logging.WARNING, logger="musubi_tuner.zimage_train_network"):
            _call_do_inference_with_mocks(trainer, sample_param, cfg_scale=None, guidance_scale=0.0)

        assert "use `--l` for cfg_scale" not in caplog.text

    def test_default_cfg_logged_when_not_specified(self, caplog):
        """When cfg_scale is None and no --g compat, the 4.0 default should be logged."""
        trainer = _make_trainer()
        sample_param = {}

        with caplog.at_level(logging.INFO, logger="musubi_tuner.zimage_train_network"):
            _call_do_inference_with_mocks(trainer, sample_param, cfg_scale=None, guidance_scale=0.0)

        assert "cfg_scale not specified, using default: 4.0" in caplog.text


# ---------------------------------------------------------------------------
# CFG formula numerical verification (single denoise step)
# ---------------------------------------------------------------------------


class TestCfgFormula:
    def test_cfg_formula_applies_correctly(self):
        """Run one denoise step and verify: noise_pred = model_out + cfg_scale * (model_out - neg_out)."""
        trainer = _make_trainer()

        # Deterministic transformer outputs: positive returns 2.0, negative returns 1.0
        pos_out = torch.full((1, 4, 1, 4, 4), 2.0)
        neg_out = torch.full((1, 4, 1, 4, 4), 1.0)
        call_count = [0]

        def fake_transformer(*args, **kwargs):
            call_count[0] += 1
            return pos_out if call_count[0] % 2 == 1 else neg_out

        mock_model = MagicMock()
        mock_model.dtype = torch.bfloat16
        mock_model.in_channels = 4
        mock_model.all_patch_size = [2]

        mock_accelerator = MagicMock()
        mock_accelerator.device = torch.device("cpu")
        mock_accelerator.unwrap_model.return_value = mock_model
        mock_accelerator.autocast.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())

        mock_transformer = MagicMock(side_effect=fake_transformer)

        dummy_embed = torch.randn(1, 8, 16)
        dummy_mask = torch.ones(1, 8, dtype=torch.bool)

        sample_param = {
            "cap_feats": dummy_embed,
            "cap_mask": dummy_mask,
            "negative_cap_feats": dummy_embed.clone(),
            "negative_cap_mask": dummy_mask.clone(),
        }

        cfg_scale_val = 5.0

        # Expected: model_out + cfg * (model_out - neg_out) = 2 + 5*(2-1) = 7
        # Then: noise_pred = -7 (sign inversion in do_inference)
        expected_noise_pred = -7.0

        captured_noise_pred = []

        def fake_step(noise_pred, latents, sigmas, i):
            captured_noise_pred.append(noise_pred.clone())
            return latents  # return unchanged for simplicity

        mock_vae = MagicMock()
        mock_vae.dtype = torch.bfloat16
        mock_vae.decode.return_value = torch.randn(1, 3, 256, 256)

        with (
            patch("musubi_tuner.zimage_train_network.zimage_utils.trim_pad_embeds_and_mask", side_effect=lambda s, e, m: (e, m)),
            patch(
                "musubi_tuner.zimage_train_network.zimage_utils.get_timesteps_sigmas",
                side_effect=lambda steps, shift: (torch.tensor([500.0]), torch.tensor([0.5, 0.0])),
            ),
            patch("musubi_tuner.zimage_train_network.zimage_utils.step", side_effect=fake_step),
            patch("musubi_tuner.zimage_train_network.zimage_utils.shift_scale_latents_for_decode", side_effect=lambda x: x),
        ):
            trainer.do_inference(
                mock_accelerator,
                MagicMock(),
                sample_param,
                mock_vae,
                "bf16",
                mock_transformer,
                3.0,
                1,  # 1 step
                256,
                256,
                1,
                torch.Generator().manual_seed(42),
                False,
                0.0,
                cfg_scale_val,
            )

        assert len(captured_noise_pred) == 1
        assert torch.allclose(captured_noise_pred[0], torch.full((1, 4, 4, 4), expected_noise_pred))
