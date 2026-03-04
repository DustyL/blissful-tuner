"""Config validation -- fail-closed feature gating, constraints, and warnings."""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when a config fails validation in strict mode."""

    pass


@dataclass
class StrippedKeysReport:
    """Report of keys stripped during coerce mode."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, key: str, value: Any, reason: str) -> None:
        self.entries.append({"key": key, "value": value, "reason": reason})

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def format(self) -> str:
        if self.is_empty():
            return "No keys stripped"
        lines = [f"Stripped {len(self.entries)} key(s):"]
        for entry in sorted(self.entries, key=lambda e: e["key"]):
            lines.append(f"  - {entry['key']} = {entry['value']!r} ({entry['reason']})")
        return "\n".join(lines)


# Mask loss related keys that get stripped when arch doesn't support it
_MASK_LOSS_KEYS = {
    "use_mask_loss",
    "mask_gamma",
    "mask_min_weight",
    "mask_blur_kernel_size",
    "mask_blur_radius",
    "mask_area_scale_beta",
}

# Prior preservation related keys
_PRIOR_KEYS = {
    "prior_preservation_weight",
    "prior_mask_threshold",
    "prior_teacher_mode",
    "prior_teacher_ema_decay",
    "prior_preservation_timestep_threshold",
    "prior_decay_schedule",
    "prior_decay_timestep_start",
    "prior_decay_warmup_ratio",
    "normalize_per_sample",
}


def _flatten_training_toml(training_toml: dict[str, Any]) -> dict[str, Any]:
    """Flatten all sections of a training TOML dict into a single key-value mapping.

    Scans top-level sections (dicts) and collects their key-value pairs.
    """
    flat: dict[str, Any] = {}
    for section_name, section in training_toml.items():
        if isinstance(section, dict):
            for k, v in section.items():
                flat[k] = v
    return flat


def _strip_key(training_toml: dict[str, Any], key: str) -> None:
    """Remove a key from whichever section of training_toml contains it (in-place)."""
    for section_name, section in training_toml.items():
        if isinstance(section, dict) and key in section:
            del section[key]


def _check_feature_gate(
    training_flat: dict[str, Any],
    training_toml: dict[str, Any],
    feature_name: str,
    trigger_key: str,
    related_keys: set[str],
    supports: dict[str, Any],
    strict: bool,
    report: StrippedKeysReport,
    check_nonzero: bool = False,
) -> None:
    """Check a feature gate and either raise or strip keys.

    Args:
        training_flat: Flattened training config (all sections merged).
        training_toml: Original nested training config (mutated in coerce mode).
        feature_name: Human-readable feature name (e.g., "mask_loss").
        trigger_key: The key that activates the feature (e.g., "use_mask_loss").
        related_keys: All keys to strip if the feature is unsupported.
        supports: The arch supports dict.
        strict: If True, raise on unsupported feature. If False, strip.
        report: StrippedKeysReport to populate in coerce mode.
        check_nonzero: If True, the trigger is active only when value > 0 (for numeric triggers
            like prior_preservation_weight). If False, trigger is active when truthy.
    """
    trigger_value = training_flat.get(trigger_key)

    # Determine whether the feature is actually enabled
    if check_nonzero:
        feature_enabled = trigger_value is not None and isinstance(trigger_value, (int, float)) and trigger_value > 0
    else:
        feature_enabled = bool(trigger_value)

    if not feature_enabled:
        return

    if supports.get(feature_name, False):
        return  # Arch supports this feature, nothing to do

    # Feature is enabled but arch doesn't support it
    if strict:
        raise ConfigValidationError(f"Architecture does not support {feature_name} but {trigger_key}={trigger_value!r} is set")

    # Coerce mode: strip all related keys
    reason = f"architecture does not support {feature_name}"
    for key in related_keys:
        value = training_flat.get(key)
        if value is not None:
            report.add(key, value, reason)
            _strip_key(training_toml, key)


def validate_config(
    training_toml: dict[str, Any],
    arch: dict[str, Any],
    strict: bool = True,
) -> StrippedKeysReport:
    """Validate a compiled training config against architecture capabilities.

    In strict mode (default), raises ConfigValidationError on unsupported features.
    In coerce mode (strict=False), strips unsupported features and returns report.

    Args:
        training_toml: The compiled training config dict (sections as top-level keys).
            Mutated in coerce mode -- unsupported keys are removed in-place.
        arch: Architecture registry entry with ``supports``, ``warnings``, ``constraints``.
        strict: If True (default), unsupported features raise. If False, they are stripped.

    Returns:
        StrippedKeysReport listing any keys that were stripped (empty in strict mode
        or when all features are supported).

    Raises:
        ConfigValidationError: In strict mode when unsupported features are found,
            or in either mode when a hard constraint is violated.
    """
    report = StrippedKeysReport()
    supports = arch.get("supports", {})
    warnings_dict = arch.get("warnings", {})
    constraints = arch.get("constraints", {})

    # Flatten training sections to find all key-value pairs
    training_flat = _flatten_training_toml(training_toml)

    # --- Feature gating ---
    _check_feature_gate(training_flat, training_toml, "mask_loss", "use_mask_loss", _MASK_LOSS_KEYS, supports, strict, report)
    _check_feature_gate(
        training_flat,
        training_toml,
        "prior_preservation",
        "prior_preservation_weight",
        _PRIOR_KEYS,
        supports,
        strict,
        report,
        check_nonzero=True,
    )

    # cute attention
    if training_flat.get("cute") and not supports.get("cute_attention", False):
        if strict:
            raise ConfigValidationError("Architecture does not support cute attention but cute=true is set")
        report.add("cute", True, "architecture does not support cute attention")
        _strip_key(training_toml, "cute")

    # --- Arch warnings ---
    for key, warning_info in warnings_dict.items():
        value = training_flat.get(key)
        if value is not None:
            max_recommended = warning_info.get("max_recommended")
            if max_recommended is not None and isinstance(value, (int, float)) and value > max_recommended:
                logger.warning("[blissful-config] %s", warning_info["message"])

    # --- Arch constraints ---
    for key, constraint_info in constraints.items():
        value = training_flat.get(key)
        if value is not None:
            # Constraints always raise, even in coerce mode
            raise ConfigValidationError(f"Constraint violation: {key}={value!r} -- {constraint_info.get('message', 'not allowed')}")

    return report


# ---------------------------------------------------------------------------
# Preset slug uniqueness
# ---------------------------------------------------------------------------


def check_preset_slug_uniqueness(preset_paths: list[Path]) -> None:
    """Verify no two preset files share the same slug.

    Reads each preset TOML, extracts ``[preset].slug``, and raises
    ConfigValidationError if any slug appears more than once.
    """
    slugs: dict[str, Path] = {}
    for path in preset_paths:
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)
        slug = data.get("preset", {}).get("slug")
        if slug is None:
            continue
        if slug in slugs:
            raise ConfigValidationError(f"Duplicate preset slug '{slug}' found in {slugs[slug]} and {path}")
        slugs[slug] = path


# ---------------------------------------------------------------------------
# Unknown --set keys
# ---------------------------------------------------------------------------


def check_unknown_keys(
    compiled_keys: set[str],
    known_keys: set[str],
    allow_unknown: bool = False,
) -> None:
    """Check for unknown keys in a compiled config.

    Args:
        compiled_keys: Set of keys present in the compiled config.
        known_keys: Set of keys recognized by the system.
        allow_unknown: If True, unknown keys are silently accepted.

    Raises:
        ConfigValidationError: If unknown keys are found and allow_unknown is False.
    """
    if allow_unknown:
        return

    unknown = sorted(compiled_keys - known_keys)
    if unknown:
        raise ConfigValidationError(f"Unknown config key(s): {', '.join(unknown)}")


# ---------------------------------------------------------------------------
# Output-path collision guard
# ---------------------------------------------------------------------------

# Pattern to extract provenance fields from a blissful-config provenance header.
_PROVENANCE_FIELD_RE = re.compile(r"^#\s+(Machine|Architecture|Persona|Preset|Override):\s+(.+)$", re.MULTILINE)

# Map from provenance comment label to provenance dict key
_LABEL_TO_KEY = {
    "Machine": "machine",
    "Architecture": "arch",
    "Persona": "persona",
    "Preset": "preset",
    "Override": "override",
}


def _extract_provenance_from_file(path: Path) -> dict[str, str] | None:
    """Extract provenance fields from a file's header comments.

    Returns None if the file doesn't contain a blissful-config provenance header.
    """
    text = path.read_text()

    # Quick check: does this look like a blissful-config output?
    if "Compiled by blissful-config" not in text:
        return None

    provenance: dict[str, str] = {}
    for match in _PROVENANCE_FIELD_RE.finditer(text):
        label = match.group(1)
        value = match.group(2).strip()
        key = _LABEL_TO_KEY.get(label)
        if key:
            provenance[key] = value

    return provenance if provenance else None


def check_output_path_collision(path: Path, expected_provenance: dict[str, str]) -> None:
    """If path exists and has a different provenance header, raise ConfigValidationError.

    This prevents accidentally overwriting a config compiled for a different
    machine/arch/persona/preset combination.

    Args:
        path: The output file path to check.
        expected_provenance: The provenance dict for the config about to be written.

    Raises:
        ConfigValidationError: If the file exists with a different provenance header.
    """
    path = Path(path)
    if not path.exists():
        return

    existing = _extract_provenance_from_file(path)
    if existing is None:
        # File exists but has no provenance header (manual file) -- allow overwrite
        return

    # Compare provenance fields that exist in both
    for key in ("machine", "arch", "persona", "preset", "override"):
        existing_val = existing.get(key)
        expected_val = expected_provenance.get(key)
        if existing_val is not None and expected_val is not None and existing_val != expected_val:
            raise ConfigValidationError(
                f"Output path '{path}' has different provenance: existing {key}='{existing_val}' vs expected {key}='{expected_val}'"
            )
