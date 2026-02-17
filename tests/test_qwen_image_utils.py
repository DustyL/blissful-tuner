import argparse
import unittest

import torch

from musubi_tuner.qwen_image import qwen_image_utils


class TestQwenImageUtils(unittest.TestCase):
    def test_pack_unpack_roundtrip_single_frame(self):
        bsz, channels, height, width = 2, 16, 16, 16
        latents = torch.randn(bsz, channels, 1, height, width, dtype=torch.float32)

        packed = qwen_image_utils.pack_latents(latents)
        unpacked = qwen_image_utils.unpack_latents(
            packed,
            height * qwen_image_utils.VAE_SCALE_FACTOR,
            width * qwen_image_utils.VAE_SCALE_FACTOR,
            vae_scale_factor=qwen_image_utils.VAE_SCALE_FACTOR,
            is_layered=False,
        )

        self.assertEqual(unpacked.shape, latents.shape)
        self.assertTrue(torch.equal(unpacked, latents))

    def test_pack_unpack_roundtrip_layered(self):
        bsz, num_layers, channels, height, width = 1, 3, 16, 16, 16
        latents = torch.randn(bsz, num_layers, channels, height, width, dtype=torch.float32)

        packed = qwen_image_utils.pack_latents(latents)
        unpacked = qwen_image_utils.unpack_latents(
            packed,
            height * qwen_image_utils.VAE_SCALE_FACTOR,
            width * qwen_image_utils.VAE_SCALE_FACTOR,
            vae_scale_factor=qwen_image_utils.VAE_SCALE_FACTOR,
            is_layered=True,
        )

        self.assertEqual(unpacked.shape, latents.shape)
        self.assertTrue(torch.equal(unpacked, latents))

    def test_calculate_shift_qwen_image_matches_endpoints(self):
        # By construction, mu(base_seq_len) == base_shift and mu(max_seq_len) == max_shift.
        base_mu = qwen_image_utils.calculate_shift_qwen_image(qwen_image_utils.SCHEDULER_BASE_IMAGE_SEQ_LEN)
        max_mu = qwen_image_utils.calculate_shift_qwen_image(qwen_image_utils.SCHEDULER_MAX_IMAGE_SEQ_LEN)

        self.assertAlmostEqual(base_mu, qwen_image_utils.SCHEDULER_BASE_SHIFT, places=7)
        self.assertAlmostEqual(max_mu, qwen_image_utils.SCHEDULER_MAX_SHIFT, places=7)

    def test_resolve_model_version_args_sets_flags(self):
        args = argparse.Namespace(model_version="EDIT-2511", edit=False, edit_plus=False)
        mv = qwen_image_utils.resolve_model_version_args(args)

        self.assertEqual(mv, "edit-2511")
        self.assertTrue(args.is_edit)
        self.assertFalse(args.is_layered)

    def test_resolve_model_version_args_edit_plus_shorthand(self):
        args = argparse.Namespace(model_version=None, edit=False, edit_plus=True)
        mv = qwen_image_utils.resolve_model_version_args(args)

        self.assertEqual(mv, "edit-2509")
        self.assertTrue(args.is_edit)
        self.assertFalse(args.is_layered)

    def test_resolve_model_version_args_rejects_invalid_value(self):
        args = argparse.Namespace(model_version="not-a-real-version", edit=False, edit_plus=False)
        with self.assertRaises(ValueError):
            qwen_image_utils.resolve_model_version_args(args)


if __name__ == "__main__":
    unittest.main()

