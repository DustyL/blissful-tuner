"""Config compiler: layer loading, deep merge, interpolation, TOML emission."""

from __future__ import annotations

import copy
from typing import Any


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dicts. Dicts merge recursively; arrays and scalars replace.

    Neither input is mutated — returns a new dict.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
