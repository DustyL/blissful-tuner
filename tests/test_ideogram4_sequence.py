import pytest
import torch

from musubi_tuner.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)
from musubi_tuner.ideogram4.sequence import (
    build_ideogram4_conditioning,
    build_image_input,
    extract_image_tokens,
    image_grid_dims,
)


def _canonical_packed(text_features, grid_h, grid_w):
    """Faithful replica of canonical _build_packed_sequence (feature variant) for parity checking."""
    num_image = grid_h * grid_w
    max_text = max(f.shape[0] for f in text_features)
    batch_size = len(text_features)
    feat = text_features[0].shape[-1]
    total = max_text + num_image

    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    image_pos = torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    llm = torch.zeros(batch_size, total, feat)
    pos = torch.zeros(batch_size, total, 3, dtype=torch.long)
    seg = torch.full((batch_size, total), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    ind = torch.zeros(batch_size, total, dtype=torch.long)
    for b, f in enumerate(text_features):
        num_text = f.shape[0]
        offset = max_text - num_text
        text_pos = torch.arange(num_text)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
        llm[b, offset : offset + num_text] = f
        pos[b, offset : offset + num_text] = text_pos_3d
        pos[b, offset + num_text :] = image_pos
        ind[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
        ind[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
        seg[b, offset : offset + num_text + num_image] = 1
    return llm, pos, seg, ind


def test_sequence_matches_canonical_packed():
    feats = [torch.randn(3, 16), torch.randn(5, 16)]  # varlen batch
    seq = build_ideogram4_conditioning(feats, 2, 2)
    llm, pos, seg, ind = _canonical_packed(feats, 2, 2)

    assert torch.equal(seq.llm_features, llm)
    assert torch.equal(seq.position_ids, pos)
    assert torch.equal(seq.segment_ids, seg)
    assert torch.equal(seq.indicator, ind)
    # image is a fixed trailing slice [max_text : max_text+num_image] for every item.
    assert seq.image_start == 5 and seq.num_image_tokens == 4 and seq.total_seq_len == 9


def test_explicit_varlen_layout():
    feats = [torch.ones(3, 8), torch.ones(5, 8)]  # offsets 2 and 0
    seq = build_ideogram4_conditioning(feats, 2, 2)
    # item 0: [pad pad | LLM LLM LLM | IMG IMG IMG IMG]; item 1: [LLM x5 | IMG x4]
    img = OUTPUT_IMAGE_INDICATOR
    llm = LLM_TOKEN_INDICATOR
    assert seq.indicator[0].tolist() == [0, 0, llm, llm, llm, img, img, img, img]
    assert seq.indicator[1].tolist() == [llm, llm, llm, llm, llm, img, img, img, img]
    assert seq.segment_ids[0].tolist() == [SEQUENCE_PADDING_INDICATOR, SEQUENCE_PADDING_INDICATOR, 1, 1, 1, 1, 1, 1, 1]
    assert seq.segment_ids[1].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # padded text positions on item 0 are zero (not yet text); text features land at [2:5].
    assert torch.equal(seq.llm_features[0, :2], torch.zeros(2, 8))
    assert torch.equal(seq.llm_features[0, 2:5], torch.ones(3, 8))


def test_image_positions_match_grid():
    seq = build_ideogram4_conditioning([torch.randn(2, 8)], 2, 3)  # grid 2x3 -> 6 image tokens
    img_pos = seq.position_ids[0, seq.image_start :]
    expected = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]]) + IMAGE_POSITION_OFFSET
    assert torch.equal(img_pos, expected)


def test_image_scatter_gather_roundtrip():
    seq = build_ideogram4_conditioning([torch.randn(3, 16), torch.randn(5, 16)], 2, 2)
    image_tokens = torch.randn(2, seq.num_image_tokens, 128)
    x = build_image_input(image_tokens, seq.total_seq_len, seq.image_start)

    assert x.shape == (2, 9, 128)
    assert torch.equal(x[:, : seq.image_start], torch.zeros(2, seq.image_start, 128))  # text/pad slots are zero
    assert torch.equal(extract_image_tokens(x, seq.image_start, seq.num_image_tokens), image_tokens)


def test_image_grid_dims():
    assert image_grid_dims(512, 768) == (32, 48)  # / (8 VAE * 2 patch)
    with pytest.raises(ValueError, match="divisible by 16"):
        image_grid_dims(500, 512)


def test_build_image_input_rejects_misaligned_slice():
    with pytest.raises(ValueError, match="does not end at total"):
        build_image_input(torch.randn(1, 4, 128), total_seq_len=10, image_start=5)  # 5+4 != 10
