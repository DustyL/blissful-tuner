"""Blissful Config — registry-driven config compiler for blissful-tuner."""

from __future__ import annotations

from pathlib import Path


def find_meta_dir() -> Path:
    """Find configs/meta/ relative to the repo root.

    Walks up from this file's location to find the project root
    (identified by configs/meta/ existing).
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        meta = parent / "configs" / "meta"
        if meta.is_dir():
            return meta
    raise FileNotFoundError("Could not find configs/meta/ directory")


def find_compiled_dir() -> Path:
    """Find configs/compiled/ relative to the repo root.

    Creates the directory if it does not exist but configs/ does.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        configs = parent / "configs"
        if configs.is_dir():
            compiled = configs / "compiled"
            compiled.mkdir(parents=True, exist_ok=True)
            return compiled
    raise FileNotFoundError("Could not find configs/ directory")
