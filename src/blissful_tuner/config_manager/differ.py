"""Semantic TOML differ -- compares two TOML configs ignoring cosmetic differences."""

from __future__ import annotations

from typing import Any


def flatten_toml(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten a multi-section TOML into a single dict (simulates read_config_from_file).

    All section keys are merged into one namespace.  Last-write-wins for
    collisions.  Only flattens one level of sections (not deeply nested).
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Section -- merge its contents
            result.update(value)
        else:
            # Top-level scalar (unusual but possible)
            result[key] = value
    return result


def normalize_toml(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a TOML dict for comparison: sort keys recursively."""
    result: dict[str, Any] = {}
    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, dict):
            result[key] = normalize_toml(value)
        else:
            result[key] = value
    return result


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two values with float/int equivalence.

    TOML parsers may read ``12.0`` as float and ``12`` as int.  For training
    configs these are semantically identical, so we compare them as floats.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def semantic_diff(a: dict[str, Any], b: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    """Compute semantic differences between two dicts.

    Both inputs are normalized (keys sorted) before comparison so key
    ordering never produces spurious diffs.

    Returns a list of diff entries, each with:

    - ``key``  -- dotted path (e.g. ``"model.dit"``)
    - ``type`` -- ``"changed"``, ``"added"``, or ``"removed"``
    - ``old_value`` / ``new_value`` as appropriate
    """
    diffs: list[dict[str, Any]] = []
    a_norm = normalize_toml(a)
    b_norm = normalize_toml(b)

    all_keys = sorted(set(a_norm.keys()) | set(b_norm.keys()))

    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else key

        if key not in a_norm:
            diffs.append({"key": full_key, "type": "added", "new_value": b_norm[key]})
        elif key not in b_norm:
            diffs.append({"key": full_key, "type": "removed", "old_value": a_norm[key]})
        elif isinstance(a_norm[key], dict) and isinstance(b_norm[key], dict):
            diffs.extend(semantic_diff(a_norm[key], b_norm[key], full_key))
        elif not _values_equal(a_norm[key], b_norm[key]):
            diffs.append({"key": full_key, "type": "changed", "old_value": a_norm[key], "new_value": b_norm[key]})

    return diffs


def format_diff_report(diffs: list[dict[str, Any]]) -> str:
    """Format diffs as a human-readable report."""
    if not diffs:
        return "No differences"

    lines: list[str] = []
    for d in diffs:
        if d["type"] == "changed":
            lines.append(f"  ~ {d['key']}: {d['old_value']!r} \u2192 {d['new_value']!r}")
        elif d["type"] == "added":
            lines.append(f"  + {d['key']}: {d['new_value']!r}")
        elif d["type"] == "removed":
            lines.append(f"  - {d['key']}: {d['old_value']!r}")

    return f"{len(diffs)} difference(s):\n" + "\n".join(lines)
