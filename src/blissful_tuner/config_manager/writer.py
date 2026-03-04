"""TOML writer with provenance headers and canonical section ordering.

Uses tomlkit to produce human-readable TOML output with comment blocks
documenting the compilation provenance (machine, arch, persona, preset).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import tomlkit
from tomlkit.items import AoT, Table

# Canonical section ordering for training TOML.  Sections not present
# in the input dict are silently skipped.
_TRAINING_SECTION_ORDER: list[str] = [
    "model",
    "dataset",
    "network",
    "optimizer",
    "training",
    "mask_loss",
    "prior",
    "sampling",
    "output",
    "advanced",
]


def _build_provenance_comment(provenance: dict[str, str]) -> str:
    """Build a provenance comment block for the top of a TOML file.

    Mirrors the style of existing hand-written configs in configs/OLVA/.
    """
    sep = "# " + "=" * 77
    lines = [sep]
    lines.append("# Compiled by blissful-config")
    lines.append(f"# Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("#")

    field_order = ["machine", "arch", "persona", "preset", "override"]
    for field in field_order:
        if field in provenance:
            label = field.capitalize() if field != "arch" else "Architecture"
            lines.append(f"#   {label}: {provenance[field]}")

    lines.append(sep)
    return "\n".join(lines)


def _add_value(table: Table, key: str, value: Any) -> None:
    """Add a single key-value pair to a tomlkit Table, handling type coercion."""
    if isinstance(value, list):
        arr = tomlkit.array()
        for item in value:
            arr.append(item)
        # Use multiline for non-empty arrays with string elements (like optimizer_args)
        if value and isinstance(value[0], str) and len(value) > 1:
            arr.multiline(True)
        table.add(key, arr)
    elif isinstance(value, dict):
        # Nested inline table (rare in training configs, but handle gracefully)
        inline = tomlkit.inline_table()
        for k, v in value.items():
            inline.add(k, v)
        table.add(key, inline)
    else:
        table.add(key, value)


def _dict_to_table(data: dict[str, Any]) -> Table:
    """Convert a plain dict to a tomlkit Table preserving insertion order."""
    table = tomlkit.table()
    for key, value in data.items():
        _add_value(table, key, value)
    return table


def render_training_toml(training_dict: dict[str, Any], provenance: dict[str, str]) -> str:
    """Render a training config dict to a TOML string with provenance header.

    Args:
        training_dict: Dict with section names as keys (model, optimizer, etc.).
        provenance: Dict with compilation provenance metadata.

    Returns:
        Complete TOML string ready to write to disk.
    """
    doc = tomlkit.document()

    # Provenance comment header
    if provenance:
        comment_block = _build_provenance_comment(provenance)
        for line in comment_block.split("\n"):
            # tomlkit.comment() adds "# " prefix, but our lines already have "#"
            # Use add(tomlkit.comment(...)) for bare comment lines
            doc.add(tomlkit.comment(line.lstrip("# ").rstrip()))
        doc.add(tomlkit.nl())

    # Add sections in canonical order
    first_section = True
    for section_name in _TRAINING_SECTION_ORDER:
        if section_name not in training_dict:
            continue
        section_data = training_dict[section_name]
        if not first_section:
            doc.add(tomlkit.nl())
        table = _dict_to_table(section_data)
        doc.add(section_name, table)
        first_section = False

    # Handle any extra sections not in the canonical order (future-proof)
    for section_name, section_data in training_dict.items():
        if section_name not in _TRAINING_SECTION_ORDER and isinstance(section_data, dict):
            doc.add(tomlkit.nl())
            table = _dict_to_table(section_data)
            doc.add(section_name, table)

    return tomlkit.dumps(doc)


def render_dataset_toml(dataset_dict: dict[str, Any], provenance: dict[str, str]) -> str:
    """Render a dataset config dict to a TOML string.

    Args:
        dataset_dict: Dict with "general" and "datasets" keys.
        provenance: Dict with compilation provenance metadata (can be empty).

    Returns:
        Complete TOML string ready to write to disk.
    """
    doc = tomlkit.document()

    # Optional provenance header
    if provenance:
        comment_block = _build_provenance_comment(provenance)
        for line in comment_block.split("\n"):
            doc.add(tomlkit.comment(line.lstrip("# ").rstrip()))
        doc.add(tomlkit.nl())

    # [general] section
    if "general" in dataset_dict:
        general_table = _dict_to_table(dataset_dict["general"])
        doc.add("general", general_table)

    # [[datasets]] array of tables
    if "datasets" in dataset_dict:
        aot = AoT([])
        for entry in dataset_dict["datasets"]:
            table = _dict_to_table(entry)
            aot.append(table)
        doc.add(tomlkit.nl())
        doc.add("datasets", aot)

    return tomlkit.dumps(doc)
