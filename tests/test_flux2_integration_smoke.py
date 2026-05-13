"""End-to-end smoke for tiny Flux.2 forward + backward.

Per-block tests live in test_flux2_block_backward_anomaly.py; the
generic LoRA target-coverage invariant lives in test_lora_target_coverage.py;
the Flux.2-specific LoRA target-coverage lives in
test_flux2_lora_target_coverage.py. What this file adds is the
*composition* smoke — a real Flux2 forward + backward at a tiny scale.
Catches regressions that only surface when img_in, txt_in, pe_embedder,
the double-block loop, the single-block loop, and the final layer
participate in the same autograd graph.

Cheap to keep in the default suite (CPU only, ~3 seconds).

Important: this file enforces a "tiny" invariant before constructing the
model. Subclassing the Flux2Params dataclass without re-decorating with
@dataclass is a silent footgun (the parent's generated __init__ ignores
new fields), and that bug previously caused an 80+ GB CPU RAM OOM. The
guard catches the regression cheaply before any model allocation.
"""

import unittest

import torch

from musubi_tuner.flux_2.flux2_models import Flux2, Flux2Params


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
    """End-to-end Flux2 forward + backward at a tiny scale."""

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


if __name__ == "__main__":
    unittest.main()
