"""Ideogram 4 single-sample SDPA mask elision.

The block-diagonal attn_mask is built ONCE in Ideogram4Transformer.forward and threaded through the
blocks; for a single sample (batch_size == 1) there is no padding (one-sample-per-row contract), so the
mask is all-True and is replaced by None, letting SDPA pick the flash backend (flash rejects a non-null
mask). batch>1 keeps the exact mask. The guard is the shape-static batch_size==1 (a free Python int),
NOT segment_ids.unique(): it runs in the eager root (no fullgraph boundary), so the reason to avoid
unique() is the per-forward GPU->CPU sync its .numel() would force, not a compile break.
"""

import torch

import musubi_tuner.ideogram4.modeling_ideogram4 as mod
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer, Ideogram4TransformerBlock
from musubi_tuner.ideogram4.sequence import build_ideogram4_conditioning
from musubi_tuner.ideogram4.training import ideogram4_flow_matching_target


def _cfg():
    return Ideogram4Config(
        emb_dim=32,
        num_layers=2,
        num_heads=2,
        intermediate_size=48,
        adanln_dim=16,
        in_channels=128,
        llm_features_dim=16,
        mrope_section=(2, 2, 2),
    )


def _spy_sdpa(monkeypatch):
    """Capture the attn_mask passed to every SDPA call inside the model forward."""
    captured = []
    orig = mod.F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kw):
        captured.append(attn_mask)
        return orig(q, k, v, attn_mask=attn_mask, **kw)

    monkeypatch.setattr(mod.F, "scaled_dot_product_attention", spy)
    return captured


def _run(model, cfg, batch, text_lens):
    latents = torch.randn(batch, cfg.in_channels, 2, 2)
    noise = torch.randn(batch, cfg.in_channels, 2, 2)
    text_features = [torch.randn(n, cfg.llm_features_dim) for n in text_lens]
    t = torch.full((batch,), 0.5)
    return ideogram4_flow_matching_target(model, latents, text_features, noise, t, network_dtype=torch.float32, device="cpu")


def test_single_sample_elides_mask(monkeypatch):
    cap = _spy_sdpa(monkeypatch)
    cfg = _cfg()
    model = Ideogram4Transformer(cfg).eval()
    _run(model, cfg, batch=1, text_lens=[3])
    assert cap, "expected SDPA calls"
    assert all(m is None for m in cap), f"batch=1 must elide the mask (None); got {[type(m).__name__ for m in cap]}"


def test_ragged_batch_keeps_exact_mask(monkeypatch):
    # batch>1 ragged must build the EXACT old block-diagonal mask -- a regression contract, not merely
    # "some masking present".
    cap = _spy_sdpa(monkeypatch)
    cfg = _cfg()
    model = Ideogram4Transformer(cfg).eval()
    text_features = [torch.randn(3, cfg.llm_features_dim), torch.randn(5, cfg.llm_features_dim)]  # ragged -> padding
    latents = torch.randn(2, cfg.in_channels, 2, 2)
    noise = torch.randn(2, cfg.in_channels, 2, 2)
    ideogram4_flow_matching_target(
        model, latents, text_features, noise, torch.full((2,), 0.5), network_dtype=torch.float32, device="cpu"
    )
    # Reconstruct the exact mask the model should have built (segment_ids depends only on text lengths +
    # the 2x2 image grid, not the random feature values).
    seq = build_ideogram4_conditioning(text_features, 2, 2, device="cpu", dtype=torch.float32)
    expected = (seq.segment_ids.unsqueeze(2) == seq.segment_ids.unsqueeze(1)).unsqueeze(1)
    assert (~expected).any(), "sanity: ragged mask must contain masked-out (False) padding positions"
    assert cap and all(m is not None for m in cap), "batch>1 ragged must retain the mask"
    for m in cap:
        assert torch.equal(m, expected), "batch>1 must build the exact block-diagonal mask"


def test_batch1_block_compiles_fullgraph_maskless():
    # The batch-1 path passes attn_mask=None into the (compiled) block. Pin that the block fullgraph-
    # compiles with a None mask -- the permanent guard against reintroducing a data-dependent predicate.
    cfg = _cfg()
    block = Ideogram4TransformerBlock(
        hidden_size=cfg.emb_dim,
        intermediate_size=cfg.intermediate_size,
        num_heads=cfg.num_heads,
        norm_eps=cfg.norm_eps,
        adanln_dim=cfg.adanln_dim,
    ).eval()
    B, L = 1, 8
    x = torch.randn(B, L, cfg.emb_dim)
    head_dim = cfg.emb_dim // cfg.num_heads
    cos = torch.randn(B, L, head_dim)
    sin = torch.randn(B, L, head_dim)
    adaln = torch.randn(B, 1, cfg.adanln_dim)  # adaln_modulation is Linear(adanln_dim, ...)
    compiled = torch.compile(block, fullgraph=True)
    out = compiled(x, None, cos, sin, adaln)  # attn_mask=None must not introduce a graph break
    assert out.shape == x.shape


# NOTE: we deliberately do NOT pin a negative "segment_ids.unique() must raise under fullgraph" test.
# It would be brittle (pytest.raises(Exception) catches anything; a future torch could support the
# data-dependent guard, turning a torch improvement into a repo failure) and it tests the wrong
# topology: the guard lives in the EAGER root forward, not a compiled block, so unique() would not break
# a fullgraph there at all -- the real reason to prefer the shape-static batch_size==1 is avoiding a
# per-forward GPU->CPU sync (see the source comment in modeling_ideogram4.py). The positive contract
# (a compiled block accepts attn_mask=None under fullgraph) is pinned by the test above.


def test_elision_semantic_parity_cpu():
    # CPU SEMANTIC parity: attn_mask=None must equal an all-True mask (same sample inside an
    # equal-length batch>1) for the image-token velocity predictions. This proves None is the correct
    # full-attention computation; it does NOT exercise the CUDA flash-vs-mem-efficient kernel split
    # (that backend speedup is established by the production A/B gate + the op-level probe, not here).
    cfg = _cfg()
    torch.manual_seed(0)
    model = Ideogram4Transformer(cfg).eval()
    latents = torch.randn(1, cfg.in_channels, 2, 2)
    noise = torch.randn(1, cfg.in_channels, 2, 2)
    feat = torch.randn(4, cfg.llm_features_dim)
    t = torch.tensor([0.5])

    with torch.no_grad():
        pred1, _ = ideogram4_flow_matching_target(model, latents, [feat], noise, t, network_dtype=torch.float32, device="cpu")
        pred2, _ = ideogram4_flow_matching_target(
            model,
            latents.repeat(2, 1, 1, 1),
            [feat, feat],
            noise.repeat(2, 1, 1, 1),
            t.repeat(2),
            network_dtype=torch.float32,
            device="cpu",
        )
    # pred2 holds both duplicated samples' image tokens; the first sample's slice must match the
    # batch-1 (maskless) result within tolerance.
    n = pred1.shape[0]
    torch.testing.assert_close(pred1, pred2[:n], rtol=0, atol=1e-4)


def test_single_sample_backward_finite(monkeypatch):
    # batch=1 maskless path must remain differentiable with finite grads (the optimization touches the
    # attention op, which is on the backward path).
    cap = _spy_sdpa(monkeypatch)
    cfg = _cfg()
    model = Ideogram4Transformer(cfg).eval()
    # give the adapter-free base a trainable parameter to check grads flow through the maskless attention
    pred, target = _run(model, cfg, batch=1, text_lens=[3])
    assert all(m is None for m in cap)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    g = model.layers[0].attention.qkv.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_single_item_conditioning_is_unpadded():
    # CONTRACT pinned for the batch-1 mask elision: build_ideogram4_conditioning with ONE sample yields a
    # uniform, all-valid segment_ids (no padding), so the batch_size==1 None-mask is correct. If
    # single-row packing or fixed-length padding is ever added, this fails first, flagging that the
    # modeling batch_size==1 branch must be revisited.
    cfg = _cfg()
    seq = build_ideogram4_conditioning([torch.randn(7, cfg.llm_features_dim)], 2, 2, device="cpu", dtype=torch.float32)
    assert seq.segment_ids.shape[0] == 1
    assert (seq.segment_ids == 1).all(), "batch-1 conditioning must be unpadded (all segment_ids == 1)"
