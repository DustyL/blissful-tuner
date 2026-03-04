"""Config compiler: layer loading, deep merge, interpolation, TOML emission."""

from __future__ import annotations

import copy
import re
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


# ---------------------------------------------------------------------------
# Variable interpolation: ${scope.key} resolution
# ---------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")
_MAX_INTERPOLATION_DEPTH = 10


def _resolve_var(
    var_path: str,
    context: dict[str, Any],
    _resolving: frozenset[str] = frozenset(),
    _depth: int = 0,
) -> Any:
    """Resolve a dotted variable path like 'machine.models_dir' from context.

    Tracks a resolving set to detect cycles and enforces max depth.
    Returns the resolved value preserving its original type.
    """
    if _depth > _MAX_INTERPOLATION_DEPTH:
        raise RecursionError(f"Interpolation depth exceeded {_MAX_INTERPOLATION_DEPTH} resolving '${{{var_path}}}'")
    if var_path in _resolving:
        cycle = " → ".join([*_resolving, var_path])
        raise RecursionError(f"Cycle detected in variable interpolation: {cycle}")

    parts = var_path.split(".")
    current: Any = context
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Variable '${{{var_path}}}': key '{part}' not found in context")
        current = current[part]

    # If the resolved value itself contains variables, resolve them too
    if isinstance(current, str) and _VAR_PATTERN.search(current):
        new_resolving = _resolving | {var_path}
        current = _VAR_PATTERN.sub(lambda m: str(_resolve_var(m.group(1), context, new_resolving, _depth + 1)), current)
    return current  # Preserve type: int, float, bool, list, or str


def _interpolate_value(value: Any, context: dict[str, Any]) -> Any:
    """Interpolate ${scope.key} variables in a single value."""
    if isinstance(value, str):
        # Whole-value substitution: "${scope.key}" with nothing else → preserve resolved type
        m = _VAR_PATTERN.fullmatch(value)
        if m:
            return _resolve_var(m.group(1), context)
        # Embedded template: "prefix_${scope.key}_suffix" → always string
        return _VAR_PATTERN.sub(lambda m: str(_resolve_var(m.group(1), context)), value)
    if isinstance(value, dict):
        return {k: _interpolate_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(item, context) for item in value]
    return value


def interpolate(data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Resolve all ${scope.key} variables in a config dict. Returns new dict.

    Detects cycles (self-reference, transitive) and enforces max depth of 10.
    """
    return _interpolate_value(copy.deepcopy(data), context)
