"""Shared CLI compatibility helpers for deprecated argument handling.

Centralizes argparse registration and deprecation warnings for flags
that are duplicated across multiple generation scripts.
"""

import argparse
import sys
from importlib.util import find_spec

from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")

lycoris_available: bool = find_spec("lycoris") is not None


def add_lycoris_arg(parser: argparse.ArgumentParser) -> None:
    """Add --prefer_lycoris / --lycoris (deprecated alias) to the parser."""
    parser.add_argument(
        "--prefer_lycoris",
        "--lycoris",
        dest="prefer_lycoris",
        action="store_true",
        help="Force LyCORIS backend for all LoRA weight merging (requires lycoris installed). "
        "(--lycoris is a deprecated alias for --prefer_lycoris)",
    )


def validate_lycoris_arg(args: argparse.Namespace, argv=None) -> None:
    """Emit deprecation warning for --lycoris and error if lycoris is not installed."""
    if argv is None:
        argv = sys.argv
    if "--lycoris" in argv:
        logger.warning("--lycoris is deprecated; use --prefer_lycoris instead. --lycoris will be removed in v0.14.0.")
    if getattr(args, "prefer_lycoris", False) and not lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")
