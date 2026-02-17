"""Tests for WAN 2.2 timestep default warnings (TP-13)."""

import argparse
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestWan22TimestepDefaultsWarning(unittest.TestCase):
    @patch("musubi_tuner.wan_train_network.detect_wan_sd_dtype", return_value=torch.bfloat16)
    @patch("musubi_tuner.wan_train_network.logger.warning")
    def test_warns_on_sigma_defaults_for_wan22(self, mock_warn: MagicMock, _mock_detect: MagicMock) -> None:
        from musubi_tuner.wan_train_network import WanNetworkTrainer

        trainer = WanNetworkTrainer.__new__(WanNetworkTrainer)
        args = argparse.Namespace(
            task="t2v-A14B",
            dit="some_path",
            mixed_precision="bf16",
            fp8_scaled=False,
            dit_high_noise=None,
            blocks_to_swap=None,
            offload_inactive_dit=False,
            num_timestep_buckets=None,
            timestep_boundary=None,
            timestep_sampling="sigma",
            discrete_flow_shift=1.0,
        )

        trainer.handle_model_specific_args(args)
        self.assertTrue(mock_warn.called)
        msg = mock_warn.call_args[0][0]
        self.assertIn("WAN 2.2", msg)
        self.assertIn("t2v-A14B", msg)

    @patch("musubi_tuner.wan_train_network.detect_wan_sd_dtype", return_value=torch.bfloat16)
    @patch("musubi_tuner.wan_train_network.logger.warning")
    def test_no_warn_on_shift_with_nondefault_flow_shift(self, mock_warn: MagicMock, _mock_detect: MagicMock) -> None:
        from musubi_tuner.wan_train_network import WanNetworkTrainer

        trainer = WanNetworkTrainer.__new__(WanNetworkTrainer)
        args = argparse.Namespace(
            task="t2v-A14B",
            dit="some_path",
            mixed_precision="bf16",
            fp8_scaled=False,
            dit_high_noise=None,
            blocks_to_swap=None,
            offload_inactive_dit=False,
            num_timestep_buckets=None,
            timestep_boundary=None,
            timestep_sampling="shift",
            discrete_flow_shift=12.0,
        )

        trainer.handle_model_specific_args(args)
        mock_warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
