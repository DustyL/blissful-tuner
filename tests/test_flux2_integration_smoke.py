"""Integration smoke tests for Flux.2 — closes two gaps surfaced by review.

The per-block anomaly tests in test_flux2_block_backward_anomaly.py and
the Liger tests in test_flux2_liger.py each cover narrow surfaces. This
file adds:

  1. A tiny end-to-end Flux2 forward + backward smoke (depth=2,
     depth_single=2, hidden_size=64, num_heads=4). Exercises the
     *composition* of all four perf commits (single_block_mod deferral,
     fresh-temporary modulation, per-tensor apply_rope, Liger fallback)
     through the model's real forward graph rather than synthetic
     per-block inputs.

  2. A cross-test that runs the DoubleStreamBlock anomaly probe with
     BLISSFUL_USE_LIGER_FLUX2=1 active. CUDA-only AND opt-in via
     BLISSFUL_RUN_CUDA_TESTS=1 — default pytest runs do NOT touch the
     GPU here.

The end-to-end test enforces "is this actually tiny?" assertions before
constructing the model. A regression that produces a full-sized Flux2
(e.g., the @dataclass-inheritance trap that broke an earlier version of
this file) will be caught BEFORE any model allocation rather than after
an 85 GB CPU RAM blowout.

CPU-only otherwise; the Liger cross-test self-skips without CUDA and
without the env-var opt-in.
"""

import importlib
import os
import unittest

import torch

from musubi_tuner.flux_2 import flux2_models
from musubi_tuner.flux_2.flux2_models import DoubleStreamBlock, Flux2, Flux2Params, rope
from musubi_tuner.modules.attention import AttentionParams


CUDA_AVAILABLE = torch.cuda.is_available()
LIGER_IMPORTABLE = flux2_models._LIGER_AVAILABLE
LIGER_CUDA_TESTS_ENABLED = os.environ.get("BLISSFUL_RUN_CUDA_TESTS", "0") == "1"


def _tiny_flux2_params() -> Flux2Params:
    """Construct Flux2Params directly with explicit tiny overrides.

    Subclassing Flux2Params without re-decorating with @dataclass is a
    silent footgun: the new class-level annotations become inert class
    attributes while the inherited __init__ still produces full-sized
    instances. Direct kwargs make the intent obvious and impossible to
    accidentally over-inherit.
    """
    return Flux2Params(
        in_channels=8,
        context_in_dim=32,
        hidden_size=64,
        num_heads=4,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[4, 4, 4, 4],  # must sum to hidden_size // num_heads = 16
        theta=2000,
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )


class TinyFlux2ForwardBackward(unittest.TestCase):
    """End-to-end Flux2 forward + backward at a tiny scale.

    Catches composition bugs that per-block tests can't see — most
    importantly, the single_block_mod deferral runs inside Flux2.forward,
    not the block, so without an end-to-end test that change is
    unverified in situ.
    """

    def test_tiny_flux2_forward_backward(self):
        params = _tiny_flux2_params()

        # Guard rail: assert this is in fact tiny BEFORE constructing the
        # model. A regression that quietly produces full-sized params
        # (the @dataclass inheritance trap) would otherwise allocate
        # 80+ GB of CPU RAM and OOM the host. These bounds are generous
        # enough to allow honest experimentation but tight enough to
        # catch full-sized accidents.
        self.assertLessEqual(params.hidden_size, 128, "params.hidden_size escaped 'tiny' bounds")
        self.assertLessEqual(params.depth, 2, "params.depth escaped 'tiny' bounds")
        self.assertLessEqual(params.depth_single_blocks, 2, "params.depth_single_blocks escaped 'tiny' bounds")
        self.assertFalse(params.use_guidance_embed, "tiny smoke does not exercise the guidance path")

        torch.manual_seed(0)
        model = Flux2(params, attn_mode="torch", split_attn=False)
        model.train()

        batch = 1
        img_seq = 4
        txt_seq = 3

        x = torch.randn(batch, img_seq, params.in_channels, requires_grad=True)
        # pe_embedder (EmbedND) consumes positional indices of shape
        # (B, L, num_axes) where num_axes == len(axes_dim).
        x_ids = torch.zeros(batch, img_seq, len(params.axes_dim), dtype=torch.float32)
        x_ids[..., 0] = torch.arange(img_seq, dtype=torch.float32)
        ctx = torch.randn(batch, txt_seq, params.context_in_dim, requires_grad=True)
        ctx_ids = torch.zeros(batch, txt_seq, len(params.axes_dim), dtype=torch.float32)
        ctx_ids[..., 0] = torch.arange(txt_seq, dtype=torch.float32)
        timesteps = torch.tensor([0.5], dtype=torch.float32)

        out = model(x, x_ids, timesteps, ctx, ctx_ids, guidance=None)
        self.assertEqual(out.shape, (batch, img_seq, params.in_channels))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ctx.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


def _reload_with_env(env_value: str):
    """Reload flux2_models with a specific BLISSFUL_USE_LIGER_FLUX2 value."""
    prior = os.environ.get("BLISSFUL_USE_LIGER_FLUX2")
    os.environ["BLISSFUL_USE_LIGER_FLUX2"] = env_value
    try:
        return importlib.reload(flux2_models)
    finally:
        if prior is None:
            os.environ.pop("BLISSFUL_USE_LIGER_FLUX2", None)
        else:
            os.environ["BLISSFUL_USE_LIGER_FLUX2"] = prior


@unittest.skipUnless(
    CUDA_AVAILABLE and LIGER_IMPORTABLE and LIGER_CUDA_TESTS_ENABLED,
    "Liger+anomaly cross-test requires CUDA + liger-kernel + BLISSFUL_RUN_CUDA_TESTS=1",
)
class LigerActivePlusInPlaceCrossTest(unittest.TestCase):
    """Liger-active QKNorm path × in-place modulation inside
    DoubleStreamBlock._forward, under autograd anomaly mode.

    Each surface is autograd-safe in isolation:
      - Liger uses in_place=True for RMSNorm's gradient computation
      - DoubleStreamBlock mutates img_modulated / img_temp in-place
        (fresh LayerNorm outputs, not residual streams)
    This test proves the composition stays clean.

    Opt-in via BLISSFUL_RUN_CUDA_TESTS=1 to keep default pytest runs
    free of GPU/Triton activity — see repo's local-safe policy.
    """

    def setUp(self):
        self.reloaded = _reload_with_env("1")
        self.assertTrue(self.reloaded._LIGER_ENABLED)

    def tearDown(self):
        _reload_with_env("0")

    def test_double_stream_block_backward_with_liger_active(self):
        torch.manual_seed(0)
        hidden_size = 16
        num_heads = 2
        head_dim = hidden_size // num_heads
        batch = 2
        img_seq = 6
        txt_seq = 4

        block = self.reloaded.DoubleStreamBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=2.0).cuda()
        block.train()

        img = torch.randn(batch, img_seq, hidden_size, device="cuda", requires_grad=True)
        txt = torch.randn(batch, txt_seq, hidden_size, device="cuda", requires_grad=True)
        pe_img = rope(
            torch.arange(img_seq, dtype=torch.float32, device="cuda").unsqueeze(0).expand(batch, -1),
            head_dim,
            theta=10_000,
        ).unsqueeze(1)
        pe_txt = rope(
            torch.arange(txt_seq, dtype=torch.float32, device="cuda").unsqueeze(0).expand(batch, -1),
            head_dim,
            theta=10_000,
        ).unsqueeze(1)

        def _mod_triple():
            return (
                torch.randn(batch, 1, hidden_size, device="cuda", requires_grad=True),
                torch.randn(batch, 1, hidden_size, device="cuda", requires_grad=True),
                torch.randn(batch, 1, hidden_size, device="cuda", requires_grad=True),
            )

        mod_img = (_mod_triple(), _mod_triple())
        mod_txt = (_mod_triple(), _mod_triple())
        attn_params = AttentionParams.create_attention_params("torch", split_attn=False)

        with torch.autograd.set_detect_anomaly(True):
            out_img, out_txt = block(img, txt, pe_img, pe_txt, mod_img, mod_txt, attn_params)
            loss = out_img.sum() + out_txt.sum()
            loss.backward()

        self.assertIsNotNone(img.grad)
        self.assertIsNotNone(txt.grad)


if __name__ == "__main__":
    unittest.main()
