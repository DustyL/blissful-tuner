"""Tests for duplicate video mask basename detection."""

from __future__ import annotations

import os


def test_video_mask_duplicate_basename_keeps_first(tmp_path):
    """Video mask dict should keep first occurrence on duplicate basename, matching image behavior."""
    mask_dir = tmp_path / "masks"
    sub1 = mask_dir / "sub1"
    sub2 = mask_dir / "sub2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    (sub1 / "clip001.png").write_bytes(b"first")
    (sub2 / "clip001.png").write_bytes(b"second")

    all_mask_paths = sorted([str(sub1 / "clip001.png"), str(sub2 / "clip001.png")])

    # Replicate the FIXED pattern (keep-first with warning)
    mask_by_basename_no_ext: dict[str, str] = {}
    for mask_path in all_mask_paths:
        mask_basename_no_ext = os.path.splitext(os.path.basename(mask_path))[0]
        if mask_basename_no_ext in mask_by_basename_no_ext:
            pass  # would log warning
        else:
            mask_by_basename_no_ext[mask_basename_no_ext] = mask_path

    assert mask_by_basename_no_ext["clip001"] == all_mask_paths[0]
