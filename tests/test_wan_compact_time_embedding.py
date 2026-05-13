"""Lockdown tests for the WAN 2.2 compact-time-embedding + RoPE-cache-eviction optimization.

The compact time embedding is enabled by default for WAN 2.2 and changes intermediate
tensor shapes from full [B, L, ...] expansion to broadcastable [B, 1, ...]. Numerical
equivalence and the t.dim()==1 gate are load-bearing properties — these tests pin both
down. The RoPE freqs_fhw cache cap test pins down the FIFO eviction policy.
"""

import unittest

import torch

from musubi_tuner.wan.modules.model import WanModel


def _tiny_wan22_model() -> WanModel:
    """CPU-runnable WAN 2.2 model, just large enough to exercise the code paths.

    dim=16, num_heads=2 -> per-head dim 8 (satisfies the (dim/num_heads) % 2 == 0
    assertion in WanModel.__init__). patch_size (1, 2, 2) matches the production
    config; freq_dim 16 keeps the time-embedding linear layers cheap.
    """
    return WanModel(
        model_type="t2v",
        model_version="2.2",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        dim=16,
        ffn_dim=32,
        freq_dim=16,
        text_dim=4,
        out_dim=4,
        num_heads=2,
        num_layers=1,
    )


class TestWan22CompactTimeEmbedding(unittest.TestCase):
    """Compact time-embedding: shape contract + numerical equivalence + I2V gate."""

    def test_compact_path_returns_broadcast_shapes_when_t_dim_1(self):
        """Uniform timestep (t.dim()==1) with compact=True must keep e/e0 at [B, 1, ...]."""
        model = _tiny_wan22_model()
        B, seq_len = 2, 7
        t = torch.randn(B) * 1000.0  # uniform timestep per batch element
        model.compact_time_embedding = True

        e, e0 = model.get_time_embedding(t, seq_len, "cpu")

        self.assertEqual(e.shape, (B, 1, model.dim))
        self.assertEqual(e0.shape, (B, 1, 6, model.dim))

    def test_compact_and_full_paths_are_bit_identical_under_broadcast(self):
        """With uniform timesteps, the compact (broadcast) and full (expanded) paths must
        produce the same downstream values when compact is broadcast across seq_len."""
        model = _tiny_wan22_model()
        B, seq_len = 2, 7
        t = torch.randn(B) * 1000.0

        model.compact_time_embedding = True
        e_c, e0_c = model.get_time_embedding(t, seq_len, "cpu")
        model.compact_time_embedding = False
        e_f, e0_f = model.get_time_embedding(t, seq_len, "cpu")

        # Compact: [B, 1, dim] / [B, 1, 6, dim]; Full: [B, seq_len, dim] / [B, seq_len, 6, dim].
        # Broadcasting (or explicit expand) the compact path must reproduce the full path exactly.
        torch.testing.assert_close(e_c.expand(-1, seq_len, -1), e_f, rtol=0, atol=0)
        torch.testing.assert_close(e0_c.expand(-1, seq_len, -1, -1), e0_f, rtol=0, atol=0)

    def test_compact_is_bypassed_when_t_dim_2(self):
        """I2V/TI2V case: t.dim()==2 (per-token timesteps) must bypass the compact path
        and produce full-expansion outputs even with compact_time_embedding=True."""
        model = _tiny_wan22_model()
        B, seq_len = 2, 7
        t = torch.randn(B, seq_len) * 1000.0  # per-token, non-uniform
        model.compact_time_embedding = True

        e_c, e0_c = model.get_time_embedding(t, seq_len, "cpu")

        # The compact flag must be ignored: outputs should be fully expanded.
        self.assertEqual(e_c.shape, (B, seq_len, model.dim))
        self.assertEqual(e0_c.shape, (B, seq_len, 6, model.dim))

        # And the output must match running with compact disabled (i.e., the gate is correct).
        model.compact_time_embedding = False
        e_f, e0_f = model.get_time_embedding(t, seq_len, "cpu")
        torch.testing.assert_close(e_c, e_f, rtol=0, atol=0)
        torch.testing.assert_close(e0_c, e0_f, rtol=0, atol=0)


class TestWanRoPEFreqsCacheEviction(unittest.TestCase):
    """RoPE freqs_fhw cache FIFO eviction at the configured cap."""

    def test_cache_caps_at_threshold_and_evicts_oldest_first(self):
        model = _tiny_wan22_model()
        # Small cap so the test stays fast and the FIFO order is obvious.
        model._FREQS_CACHE_MAX_SIZE = 3

        # 4 distinct (F, H', W') keys by varying input width through patch_size (1, 2, 2):
        # input [C, 1, 2, W] -> grid (1, 1, W//2).
        widths = (2, 4, 6, 8)
        expected_keys = [(1, 1, w // 2) for w in widths]
        seq_len = 8  # >= largest patched seq (4)

        for w in widths:
            x = [torch.randn(model.in_dim, 1, 2, w)]
            model.get_patch_embedding(x, None, seq_len)

        self.assertEqual(len(model.freqs_fhw), 3, "cache must cap at _FREQS_CACHE_MAX_SIZE")
        self.assertNotIn(expected_keys[0], model.freqs_fhw, "first-inserted key must be evicted (FIFO)")
        for key in expected_keys[1:]:
            self.assertIn(key, model.freqs_fhw)
        self.assertTrue(getattr(model, "_freqs_eviction_warned", False), "_freqs_eviction_warned must be set after first eviction")


if __name__ == "__main__":
    unittest.main()
