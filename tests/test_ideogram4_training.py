import torch

from musubi_tuner.ideogram4.ideogram4_utils import grid_to_dit_tokens
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target


class _EchoDiT:
    """Returns x unchanged, so the image-slice of the output equals the noised image tokens."""

    def __call__(self, *, x, t, llm_features, position_ids, segment_ids, indicator):
        self.last_t = t
        return x


def test_flow_matching_convention_is_canonical():
    # noisy = (1-t)*noise + t*clean  (noise@t=0, clean@t=1);  target = clean - noise.
    batch, gh, gw = 2, 2, 2
    latents = torch.randn(batch, 128, gh, gw)
    noise = torch.randn(batch, 128, gh, gw)
    text_features = [torch.randn(3, 8), torch.randn(5, 8)]
    t = torch.tensor([0.25, 0.75])

    dit = _EchoDiT()
    pred, target = ideogram4_flow_matching_target(dit, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu")

    clean_tok = grid_to_dit_tokens(latents)
    noise_tok = grid_to_dit_tokens(noise)
    assert pred.shape == (batch, gh * gw, 128)
    # target is the velocity clean - noise.
    assert torch.allclose(target, clean_tok - noise_tok, atol=1e-5)
    # echo DiT -> pred is exactly the noised image tokens (1-t)*noise + t*clean.
    tb = t.view(-1, 1, 1)
    assert torch.allclose(pred, (1.0 - tb) * noise_tok + tb * clean_tok, atol=1e-4)
    # the model was conditioned at the cleanness coefficient t (not a mirrored 1-t).
    assert torch.allclose(dit.last_t, t)


def test_perfect_velocity_gives_zero_loss():
    latents = torch.randn(1, 128, 2, 2)
    noise = torch.randn(1, 128, 2, 2)
    text_features = [torch.randn(3, 8)]
    target_tokens = grid_to_dit_tokens(latents) - grid_to_dit_tokens(noise)  # the true velocity

    class _PerfectDiT:
        def __call__(self, *, x, t, llm_features, position_ids, segment_ids, indicator):
            out = torch.zeros_like(x)
            out[:, -target_tokens.shape[1] :] = target_tokens  # image is the trailing slice
            return out

    pred, target = ideogram4_flow_matching_target(
        _PerfectDiT(), latents, text_features, noise, torch.tensor([0.5]), network_dtype=torch.float32, device="cpu"
    )
    assert torch.nn.functional.mse_loss(pred, target).item() < 1e-6


def test_training_step_backward_on_real_tiny_model():
    # End-to-end training step on a REAL (tiny) Ideogram4Transformer: the joint-sequence forward (mRoPE, SDPA,
    # AdaLN, llm_features projection) must be differentiable and the MSE loss must backprop to the weights.
    # mrope_section[h/w] must be > 0 and *3 <= inv_freq_size (head_dim/2 = 8 here); (2,2,2) is valid.
    cfg = Ideogram4Config(
        emb_dim=32, num_layers=2, num_heads=2, intermediate_size=48,
        adanln_dim=16, in_channels=128, llm_features_dim=16, mrope_section=(2, 2, 2),
    )
    model = Ideogram4Transformer(cfg)  # real (random) weights on CPU
    latents = torch.randn(1, 128, 2, 2)
    noise = torch.randn(1, 128, 2, 2)
    text_features = [torch.randn(3, cfg.llm_features_dim)]

    pred, target = ideogram4_flow_matching_target(
        model, latents, text_features, noise, torch.tensor([0.5]), network_dtype=torch.float32, device="cpu"
    )
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "loss must backprop finite grads to the DiT"
