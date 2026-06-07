import pytest
import torch
from torch import nn

from musubi_tuner.ideogram4.ideogram4_utils import (
    LATENT_NORM_METADATA_KEY,
    assert_latent_norm_applied,
    decode_dit_tokens_to_pixels,
    encode_pixels_to_dit_tokens,
    encode_pixels_to_raw_vae_tokens,
)
from musubi_tuner.ideogram4.latent_norm import (
    LATENT_SCALE,
    LATENT_SHIFT,
    get_latent_norm,
    latent_denorm,
    latent_norm,
)


def test_latent_norm_constants_pinned_to_canonical():
    # Hardcoded literals — CI has no access to the canonical repo, so a "matches canonical" test must
    # carry the values, not read the source tree. Pins length + boundary values (catches a truncated or
    # reordered vendoring). Full (pi,pj,c) channel coupling is guarded by the patchify-pin test.
    assert len(LATENT_SHIFT) == 128
    assert len(LATENT_SCALE) == 128
    assert LATENT_SHIFT[:2] == (0.01984364, 0.10149707)
    assert LATENT_SHIFT[-2:] == (-0.34495114, -0.01760592)
    assert LATENT_SCALE[:2] == (1.63933691, 1.70204478)
    assert LATENT_SCALE[-2:] == (1.65655173, 1.68533454)


def test_latent_norm_direction_is_subtract_shift_then_divide_scale():
    # DIRECTION-PINNED (NOT a round-trip — a round-trip is direction-blind: a */÷ swap still round-trips).
    # Canonical decode is `latent = t*scale + shift`, so norm must be `(t - shift) / scale`.
    # Feed t = shift + scale: correct norm -> ((shift+scale) - shift)/scale = ones.
    # An inverted direction ((t-shift)*scale) would give scale**2 ~ 2.6, not 1.
    shift, scale = get_latent_norm(dtype=torch.float32)
    tokens = (shift + scale).reshape(1, 1, 128)
    assert torch.allclose(latent_norm(tokens), torch.ones(1, 1, 128), atol=1e-4)
    # denorm(ones) = ones*scale + shift = shift + scale.
    assert torch.allclose(latent_denorm(torch.ones(1, 1, 128)), (shift + scale).reshape(1, 1, 128), atol=1e-4)


def test_latent_norm_actually_transforms_and_roundtrips():
    x = torch.randn(2, 5, 128)
    normed = latent_norm(x)
    assert not torch.allclose(normed, x, atol=1e-2)  # scale != 1 / shift != 0, so it must change the data
    assert torch.allclose(latent_denorm(normed), x, atol=1e-4)  # round-trip (secondary to the direction test)


def test_latent_norm_preserves_input_dtype():
    # float32 math, but downstream must get the input dtype back (same lesson as inv_freq).
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    assert latent_norm(x).dtype == torch.bfloat16
    assert latent_denorm(x).dtype == torch.bfloat16


def test_latent_norm_rejects_wrong_channel_dim():
    with pytest.raises(ValueError, match="128"):
        latent_norm(torch.randn(1, 4, 64))
    with pytest.raises(ValueError, match="128"):
        latent_denorm(torch.randn(1, 4, 32))


class _FakeEncoder(nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixels.shape
        mean = torch.arange(batch_size * 32 * (height // 2) * (width // 2), dtype=pixels.dtype)
        mean = mean.reshape(batch_size, 32, height // 2, width // 2)
        return torch.cat([mean, torch.zeros_like(mean)], dim=1)


class _FakeDecoder(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = z.shape
        return torch.zeros(batch_size, 3, height * 2, width * 2, device=z.device, dtype=z.dtype)


class _FakeAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        raise AssertionError("dit-token helpers must not call ae.encode()")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise AssertionError("dit-token helpers must not call ae.decode()")


def test_encode_dit_tokens_is_raw_then_norm():
    ae = _FakeAutoencoder()
    pixels = torch.rand(1, 3, 16, 16)

    tokens, grid_h, grid_w = encode_pixels_to_dit_tokens(ae, pixels)

    assert (grid_h, grid_w) == (4, 4)  # 16 -> /2 encoder -> /2 patch
    assert tokens.shape == (1, grid_h * grid_w, 128)

    # encode_pixels_to_dit_tokens == raw encode (flattened) then latent_norm; denorm must recover the raw.
    raw_grid = encode_pixels_to_raw_vae_tokens(ae, pixels)  # (1, 128, 4, 4)
    raw_flat = raw_grid.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, 128)
    assert torch.allclose(latent_denorm(tokens), raw_flat, atol=1e-3)


def test_decode_dit_tokens_denorms_then_raw_decodes():
    ae = _FakeAutoencoder()
    pixels = torch.rand(1, 3, 16, 16)
    tokens, grid_h, grid_w = encode_pixels_to_dit_tokens(ae, pixels)

    out = decode_dit_tokens_to_pixels(ae, tokens, grid_h=grid_h, grid_w=grid_w)

    assert out.shape == (1, 3, 16, 16)  # fake decoder zeros -> 0.5 after [-1,1]->[0,1]
    assert torch.all(out == 0.5)


class _MeanLogvarEncoder(nn.Module):
    """Returns a constant mean (chunk[0]) and a LARGE logvar (chunk[1]); a sampler would be very noisy."""

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixels.shape
        mean = torch.full((batch_size, 32, height // 2, width // 2), 0.3, dtype=pixels.dtype)
        logvar = torch.full_like(mean, 5.0)
        return torch.cat([mean, logvar], dim=1)


def test_encode_dit_tokens_uses_posterior_mean_not_sample():
    # Training convention (fork training cache path: ideogram4_autoencoder.py:337 takes chunk(moments,2)[0]):
    # use the posterior MEAN, never a reparameterized sample. So encode is DETERMINISTIC and ignores logvar.
    ae = _FakeAutoencoder()
    ae.encoder = _MeanLogvarEncoder()
    pixels = torch.rand(1, 3, 16, 16)

    tokens_a, _, _ = encode_pixels_to_dit_tokens(ae, pixels)
    tokens_b, _, _ = encode_pixels_to_dit_tokens(ae, pixels)

    assert torch.equal(tokens_a, tokens_b)  # no sampling -> deterministic
    # denorm recovers the constant mean (0.3), confirming chunk[0] was used (a sample would be ~0.3 + huge noise)
    assert torch.allclose(latent_denorm(tokens_a), torch.full_like(latent_denorm(tokens_a), 0.3), atol=1e-2)


def test_cache_guard_requires_latent_norm_applied_flag():
    with pytest.raises(ValueError, match=LATENT_NORM_METADATA_KEY):
        assert_latent_norm_applied({})
    with pytest.raises(ValueError, match=LATENT_NORM_METADATA_KEY):
        assert_latent_norm_applied({LATENT_NORM_METADATA_KEY: "false"})
    assert_latent_norm_applied({LATENT_NORM_METADATA_KEY: "true"})  # sanctioned -> no raise
