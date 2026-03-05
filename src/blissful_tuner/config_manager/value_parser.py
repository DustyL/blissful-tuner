"""Type-safe value parsing for CLI --set and TUI edits.

Uses TOML value semantics: wraps input in a dummy TOML doc and parses.
Falls back to bare string if TOML parse fails.
"""

from __future__ import annotations

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]
from typing import Any


def parse_value(raw: str) -> Any:
    """Parse a string value using TOML semantics.

    - "42" -> int(42)
    - "5e-5" -> float(5e-5)
    - "true" -> bool(True)
    - '"hello"' -> str("hello")
    - '["a", "b"]' -> list(["a", "b"])
    - "bare_string" -> str("bare_string") (fallback)
    """
    # Try parsing as TOML value by wrapping in a key assignment
    try:
        parsed = tomllib.loads(f"_v = {raw}")
        return parsed["_v"]
    except Exception:
        # Fallback: treat as bare string
        return raw


def parse_and_typecheck(raw: str, existing_value: Any, key: str) -> Any:
    """Parse a string value and verify it matches the expected type.

    If existing_value is None (unknown key), no type checking is done.
    int->float coercion is allowed (12 where float expected -> 12.0).
    """
    parsed = parse_value(raw)

    if existing_value is None:
        return parsed

    expected_type = type(existing_value)
    actual_type = type(parsed)

    # Allow int->float coercion
    if expected_type is float and actual_type is int:
        return float(parsed)

    if actual_type is not expected_type:
        raise TypeError(
            f"Type mismatch for '{key}': expected {expected_type.__name__}, got {actual_type.__name__} (value: {raw!r})"
        )

    return parsed
