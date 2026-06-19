"""Ideogram 4 single-sample SDPA mask elision.

The block-diagonal attn_mask is built ONCE in Ideogram4Transformer.forward and threaded through the
blocks; for a single sample (batch_size == 1) there is no padding, so the mask is all-True and is
replaced by None, letting SDPA pick the flash backend (flash rejects a non-null mask). batch>1 keeps
the exact mask. The guard is the SHAPE-static batch_size==1 (compile-safe) -- a data-dependent
segment_ids.unique() guard would break torch.compile(fullgraph=True), which is the regression these
tests pin.
"""

import pytest
import torch

import musubi_tuner.ideogram4.modeling_ideogram4 as mod
from musubi_tuner.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer, Ideogram4TransformerBlock
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


def test_ragged_batch_keeps_mask(monkeypatch):
    cap = _spy_sdpa(monkeypatch)
    cfg = _cfg()
    model = Ideogram4Transformer(cfg).eval()
    _run(model, cfg, batch=2, text_lens=[3, 5])  # ragged -> left padding -> mask required
    assert cap and all(m is not None for m in cap), "batch>1 ragged must retain the block-diagonal mask"
    # The retained mask must be a real block-diagonal boolean mask with some False entries (padding).
    m0 = cap[0]
    assert m0.dtype == torch.bool and m0.dim() == 4
    assert (~m0).any(), "ragged mask must have masked-out (False) padding positions"


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


def test_shape_static_guard_compiles_but_unique_guard_does_not():
    # The lesson, pinned permanently: the chosen shape-static guard is fullgraph-safe; the rejected
    # data-dependent segment_ids.unique() guard is not.
    import torch.nn.functional as F

    def shape_guard(seg, q, k, v):
        m = None if q.shape[0] == 1 else (seg.unsqueeze(2) == seg.unsqueeze(1)).unsqueeze(1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

    def unique_guard(seg, q, k, v):
        m = None if seg.unique().numel() == 1 else (seg.unsqueeze(2) == seg.unsqueeze(1)).unsqueeze(1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

    seg = torch.zeros(1, 8, dtype=torch.long)
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    torch.compile(shape_guard, fullgraph=True)(seg, q, k, v)  # must succeed
    with pytest.raises(Exception):
        torch.compile(unique_guard, fullgraph=True)(seg, q, k, v)  # data-dependent -> fullgraph error


def test_elision_numerically_matches_all_true_mask():
    # Eliding an all-True mask (batch=1, flash) must match keeping it (the same sample inside an
    # equal-length batch>1, which carries an all-True per-row mask) within backend-switch tolerance --
    # NOT bitwise (different SDPA backend). Image-token velocity predictions are compared.
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
