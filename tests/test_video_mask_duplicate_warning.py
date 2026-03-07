"""Tests for duplicate video mask basename detection and deterministic ordering."""

from __future__ import annotations

import os
import struct
import zlib


def _minimal_png() -> bytes:
    """Create a minimal valid 1x1 white PNG."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc
    raw = b"\x00\xff\xff\xff"
    idat_data = zlib.compress(raw)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF)
    idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + idat_crc
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return signature + ihdr + idat + iend


def _minimal_jpeg() -> bytes:
    """Create a minimal valid JPEG (SOI + EOI markers only)."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def test_video_mask_duplicate_basename_keeps_first_sorted(tmp_path):
    """Video mask dict keeps first sorted occurrence on duplicate basename.

    glob_images() returns sorted paths. Two files with the same stem but
    different extensions (clip001.jpg and clip001.png) produce duplicate
    basenames. The keep-first logic must select deterministically.
    """
    from musubi_tuner.dataset.image_video_dataset import glob_images

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    # Same stem, different extensions — glob_images will find both
    (mask_dir / "clip001.jpg").write_bytes(_minimal_jpeg())
    (mask_dir / "clip001.png").write_bytes(_minimal_png())

    all_mask_paths = glob_images(str(mask_dir))
    assert len(all_mask_paths) == 2, f"expected 2 mask files, got {len(all_mask_paths)}"

    # glob_images returns sorted — .jpg sorts before .png
    assert all_mask_paths[0].endswith(".jpg")
    assert all_mask_paths[1].endswith(".png")

    # Production duplicate-detection loop (keep first)
    mask_by_basename_no_ext: dict[str, str] = {}
    for mask_path in all_mask_paths:
        mask_basename_no_ext = os.path.splitext(os.path.basename(mask_path))[0]
        if mask_basename_no_ext in mask_by_basename_no_ext:
            pass  # production code logs warning
        else:
            mask_by_basename_no_ext[mask_basename_no_ext] = mask_path

    # Must keep the .jpg (first sorted), not the .png
    assert mask_by_basename_no_ext["clip001"].endswith(".jpg")


def test_glob_images_returns_sorted_list(tmp_path):
    """Verify glob_images returns a sorted list, not a set.

    If the caller wraps glob_images() in set(), the sorted order is destroyed
    and duplicate-basename 'keep first' becomes nondeterministic across runs.
    """
    from musubi_tuner.dataset.image_video_dataset import glob_images

    result = glob_images(str(tmp_path))
    assert isinstance(result, list), f"glob_images should return list, got {type(result)}"


def test_production_code_does_not_wrap_in_set():
    """Verify the production video mask code uses glob_images() directly, not set(glob_images()).

    This is a source-level assertion to prevent regression of the set() wrapper
    that caused nondeterministic mask selection.
    """
    import inspect
    from musubi_tuner.dataset import image_video_dataset

    source = inspect.getsource(image_video_dataset)
    # The video mask section should NOT have set(glob_images(...))
    # Count occurrences — image_video_dataset uses set(glob_images(...)) in other places
    # (control images at line 1569, multi-target at line 1511) but NOT for mask_directory
    lines = source.split("\n")
    mask_section_lines = []
    in_mask_section = False
    for line in lines:
        if "glob mask images in" in line:
            in_mask_section = True
        if in_mask_section:
            mask_section_lines.append(line)
            if "mask_by_basename" in line:
                break

    mask_section = "\n".join(mask_section_lines)
    assert "set(glob_images" not in mask_section, "Video mask glob should NOT be wrapped in set() — it destroys sorted order"
