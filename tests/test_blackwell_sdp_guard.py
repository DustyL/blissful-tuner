"""Tests for the Blackwell cuDNN-SDPA guard in modules/attention.py.

The guard disables PyTorch's cuDNN SDPA backend process-wide on Blackwell-family
GPUs, where cuDNN's fused-attention backward produces NaN / crashes during
training (observed on B300 sm_103; reported on consumer sm_120). It keeps
FLASH / mem-efficient / math, and is opt-out via BLISSFUL_ALLOW_CUDNN_SDP=1.

The decision logic is a pure function (``_should_disable_cudnn_sdp``) so these
run without a GPU. A final consistency check validates the actually-applied
process-global flag against the real device when CUDA is present.
"""

import os
import unittest

import torch

from musubi_tuner.modules.attention import (
    _BLACKWELL_SDP_MAJORS,
    _should_disable_cudnn_sdp,
)


class TestBlackwellSdpDecision(unittest.TestCase):
    def test_disable_on_datacenter_blackwell(self):
        # B200 = sm_100, B300 = sm_103 → major 10. The fork's `major >= 12` check
        # would have MISSED these — the exact cards where the crash was first seen.
        self.assertTrue(_should_disable_cudnn_sdp(10, allow_override=False))

    def test_disable_on_consumer_blackwell(self):
        # RTX 5090 / RTX Pro 6000 = sm_120 → major 12.
        self.assertTrue(_should_disable_cudnn_sdp(12, allow_override=False))

    def test_keep_on_hopper(self):
        # H100/H200 = sm_90 → major 9. Not Blackwell; cuDNN SDPA stays enabled.
        self.assertFalse(_should_disable_cudnn_sdp(9, allow_override=False))

    def test_keep_on_ada_and_ampere(self):
        self.assertFalse(_should_disable_cudnn_sdp(8, allow_override=False))  # Ampere/Ada
        self.assertFalse(_should_disable_cudnn_sdp(7, allow_override=False))  # Volta/Turing

    def test_keep_when_no_cuda(self):
        self.assertFalse(_should_disable_cudnn_sdp(None, allow_override=False))

    def test_override_keeps_cudnn_even_on_blackwell(self):
        self.assertFalse(_should_disable_cudnn_sdp(10, allow_override=True))
        self.assertFalse(_should_disable_cudnn_sdp(12, allow_override=True))

    def test_blackwell_majors_set(self):
        # Guard the family scope explicitly: Blackwell only, not Hopper.
        self.assertIn(10, _BLACKWELL_SDP_MAJORS)
        self.assertIn(12, _BLACKWELL_SDP_MAJORS)
        self.assertNotIn(9, _BLACKWELL_SDP_MAJORS)


class TestBlackwellSdpAppliedToDevice(unittest.TestCase):
    def test_global_flag_consistent_with_device(self):
        """On a real GPU, the process-global cuDNN-SDP flag must match the decision.

        Skips when CUDA / the cudnn-sdp toggle is unavailable so CI stays green.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
            self.skipTest("cudnn_sdp toggle not available in this torch")

        major = torch.cuda.get_device_capability()[0]
        allow_override = os.environ.get("BLISSFUL_ALLOW_CUDNN_SDP", "0") == "1"
        expect_disabled = _should_disable_cudnn_sdp(major, allow_override=allow_override)

        if expect_disabled:
            # Importing modules.attention applied the guard at import time.
            self.assertFalse(
                torch.backends.cuda.cudnn_sdp_enabled(),
                "cuDNN SDPA should be disabled on this Blackwell device but is still enabled",
            )


if __name__ == "__main__":
    unittest.main()
