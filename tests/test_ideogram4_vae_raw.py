import torch
from torch import nn

from musubi_tuner.ideogram4.ideogram4_utils import (
    decode_raw_vae_tokens_to_pixels,
    encode_pixels_to_raw_vae_tokens,
    patchify_vae_latents,
    unpatchify_vae_latents,
)


class _FakeEncoder(nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixels.shape
        mean = torch.ones(batch_size, 32, height // 2, width // 2, device=pixels.device, dtype=pixels.dtype)
        logvar = torch.zeros_like(mean)
        return torch.cat([mean, logvar], dim=1)


class _FakeDecoder(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = z.shape
        return torch.zeros(batch_size, 3, height * 2, width * 2, device=z.device, dtype=z.dtype)


class _FakeAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder()
        self.bn = nn.BatchNorm2d(128)
        self._dtype = torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Ideogram raw VAE helpers must not call ae.encode()")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Ideogram raw VAE helpers must not call ae.decode()")


def test_patchify_unpatchify_vae_latents_roundtrip():
    latents = torch.arange(1 * 32 * 8 * 10, dtype=torch.float32).reshape(1, 32, 8, 10)
    tokens = patchify_vae_latents(latents)

    assert tokens.shape == (1, 128, 4, 5)
    assert torch.equal(unpatchify_vae_latents(tokens), latents)

    flattened = tokens.permute(0, 2, 3, 1).reshape(1, 20, 128)
    assert torch.equal(unpatchify_vae_latents(flattened, grid_h=4, grid_w=5), latents)


def _canonical_unpatch(tokens: torch.Tensor, grid_h: int, grid_w: int, patch: int = 2) -> torch.Tensor:
    """Byte-for-byte mirror of canonical pipeline_ideogram4.py:626-629 (token channel order = pi,pj,c)."""
    batch_size = tokens.shape[0]
    ae_channels = tokens.shape[-1] // (patch * patch)
    z = tokens.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    return z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)


def test_patchify_channel_order_pinned_to_canonical():
    """Pin the token channel layout to canonical's (pi, pj, c), not the fork's autoencoder (c, pi, pj).

    The round-trip test only proves unpatchify is patchify's inverse (self-consistent). It would NOT
    catch a "consistency edit" toward the fork's encode() order, which would scramble channels for
    the frozen DiT. This test fails loudly if the convention drifts. Ground truth:
    canonical pipeline_ideogram4.py:626-629.
    """
    latents = torch.arange(1 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 2, 2, 2)  # C=2, H=2, W=2
    tokens = patchify_vae_latents(latents)  # -> (1, 8, 1, 1), grid 1x1

    # Known-good (pi, pj, c) channel ordering for this labelled input:
    assert torch.equal(tokens.reshape(-1), torch.tensor([0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]))

    # Production unpatchify must equal the canonical pipeline formula on the flattened token grid.
    flat = tokens.permute(0, 2, 3, 1).reshape(1, 1, 8)
    assert torch.equal(unpatchify_vae_latents(flat, grid_h=1, grid_w=1), _canonical_unpatch(flat, 1, 1))


def test_raw_vae_helpers_bypass_convenience_methods_and_bn():
    ae = _FakeAutoencoder()
    running_mean = ae.bn.running_mean.clone()
    running_var = ae.bn.running_var.clone()

    pixels = torch.rand(1, 3, 16, 16)
    tokens = encode_pixels_to_raw_vae_tokens(ae, pixels)
    decoded = decode_raw_vae_tokens_to_pixels(ae, tokens)

    assert tokens.shape == (1, 128, 4, 4)
    assert decoded.shape == (1, 3, 16, 16)
    assert torch.all(decoded == 0.5)
    assert torch.equal(ae.bn.running_mean, running_mean)
    assert torch.equal(ae.bn.running_var, running_var)
