import torch

from musubi_tuner.ideogram4.generation import denoise_ideogram4_tokens
from musubi_tuner.ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals
from musubi_tuner.ideogram4.sequence import build_ideogram4_conditioning


class _FakeDiT:
    """Returns a constant velocity at every position and records the timesteps it was called with."""

    def __init__(self, out_dim: int, value: float):
        self.out_dim = out_dim
        self.value = value
        self.ts: list[float] = []

    def __call__(self, *, x, t, llm_features, position_ids, segment_ids, indicator):
        self.ts.append(float(t[0]))
        batch, length, _ = x.shape
        return torch.full((batch, length, self.out_dim), self.value, dtype=torch.float32)


def test_denoise_t_direction_is_canonical():
    # The model's first call must be at SMALL t (noise) and last at LARGE t (clean) — the B5-correct direction.
    seq = build_ideogram4_conditioning([torch.randn(3, 8)], 2, 2)
    cond, uncond = _FakeDiT(128, 1.0), _FakeDiT(128, 0.0)
    denoise_ideogram4_tokens(
        cond,
        uncond,
        seq,
        num_steps=6,
        guidance_schedule=(3.0, 3.0, 7.0, 7.0, 7.0, 7.0),
        schedule=get_schedule_for_resolution((512, 512), known_mean=0.0, std=1.75),
        device="cpu",
        compute_dtype=torch.float32,
        generator=torch.Generator().manual_seed(0),
    )
    # B5 discriminator: noise is presented at SMALL t (the fork's mirror would put it at t~0.999).
    assert cond.ts[0] < 0.01, f"first step should present noise at t~0, got {cond.ts[0]}"
    assert cond.ts == sorted(cond.ts), "t must increase monotonically (noise -> clean direction)"
    # The model is called at each interval's start t_val, so the last forward is < 1 (the final s_val~0.999
    # is the integration target, not a forward timestep).
    assert cond.ts[0] < cond.ts[-1] < 1.0


def test_denoise_cfg_mix_and_integration():
    # With cond velocity = 1 and uncond = 0, v = gw*1 + (1-gw)*0 = gw, so z advances by sum_i gw_i*(s_i - t_i).
    seq = build_ideogram4_conditioning([torch.randn(3, 8)], 2, 2)
    num_steps = 4
    gw = (2.0, 3.0, 5.0, 7.0)
    sch = get_schedule_for_resolution((512, 512), known_mean=0.0, std=1.75)
    cond, uncond = _FakeDiT(128, 1.0), _FakeDiT(128, 0.0)

    z = denoise_ideogram4_tokens(
        cond,
        uncond,
        seq,
        num_steps=num_steps,
        guidance_schedule=gw,
        schedule=sch,
        device="cpu",
        compute_dtype=torch.float32,
        generator=torch.Generator().manual_seed(0),
    )
    assert z.shape == (1, 4, 128)

    si = make_step_intervals(num_steps)
    expected = sum(gw[i] * (float(sch(si[i].unsqueeze(0))) - float(sch(si[i + 1].unsqueeze(0)))) for i in range(num_steps))
    z_init = torch.randn(1, 4, 128, dtype=torch.float32, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(z - z_init, torch.full_like(z, expected), atol=1e-3)


def test_denoise_rejects_bad_guidance_length():
    seq = build_ideogram4_conditioning([torch.randn(2, 8)], 2, 2)
    cond, uncond = _FakeDiT(128, 1.0), _FakeDiT(128, 0.0)
    try:
        denoise_ideogram4_tokens(
            cond,
            uncond,
            seq,
            num_steps=4,
            guidance_schedule=(7.0, 7.0),  # wrong length
            schedule=get_schedule_for_resolution((512, 512), known_mean=0.0, std=1.0),
            device="cpu",
        )
        raise AssertionError("expected ValueError for mismatched guidance_schedule length")
    except ValueError as e:
        assert "guidance_schedule" in str(e)
