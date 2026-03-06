"""Config compiler: layer loading, deep merge, interpolation, TOML emission."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import os
import re
import stat
import tempfile

import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomlkit

from blissful_tuner.config_manager.registry import resolve_arch, resolve_arch_key

logger = logging.getLogger(__name__)

_COMPILER_VERSION = "0.1.0"
_MANIFEST_SCHEMA_VERSION = 1


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


# ---------------------------------------------------------------------------
# Layer loading
# ---------------------------------------------------------------------------


def load_layer(path: Path) -> dict[str, Any]:
    """Load a TOML layer file. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config layer not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Compile pipeline internals
# ---------------------------------------------------------------------------


def _build_context(machine_data: dict[str, Any], persona_data: dict[str, Any]) -> dict[str, Any]:
    """Build the interpolation context from machine and persona layers.

    The context exposes machine.paths.* as machine.* shortcuts (flattened)
    so that ${machine.models_dir} works in arch registry model_files.
    """
    machine = machine_data.get("machine", {})
    paths = machine.get("paths", {})
    # Flatten machine.paths into machine namespace for convenience
    context: dict[str, Any] = {
        "machine": {**paths, **{k: v for k, v in machine.items() if k != "paths"}},
        "persona": persona_data.get("persona", {}),
        "dataset": persona_data.get("dataset", {}),
    }
    return context


def _build_training_toml(
    arch: dict[str, Any],
    arch_key: str,
    persona_data: dict[str, Any],
    preset_data: dict[str, Any],
    machine_data: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build the training TOML dict from merged layers.

    Sections: model, dataset, network, optimizer, training, sampling, output, advanced.
    """
    from blissful_tuner.config_manager.validator import ConfigValidationError

    persona_name = persona_data.get("persona", {}).get("name", "unknown")
    if not isinstance(persona_name, str):
        raise ConfigValidationError(f"Persona name must be a string, got {type(persona_name).__name__}: {persona_name!r}")
    persona_lower = persona_name.lower()
    preset_slug = preset_data.get("preset", {}).get("slug", "default")
    if not isinstance(preset_slug, str):
        raise ConfigValidationError(f"Preset slug must be a string, got {type(preset_slug).__name__}: {preset_slug!r}")
    machine_hw = machine_data.get("machine", {}).get("hardware", {})

    run_name = f"{persona_lower}_{arch_key}_{preset_slug}"
    machine_output_dir = context["machine"].get("output_dir", "/tmp/output")

    # [model] — arch registry model_files, interpolated
    model_files = interpolate(copy.deepcopy(arch.get("model_files", {})), context)
    model_section = dict(model_files)

    # [network] — from preset + arch network_module
    preset_network = copy.deepcopy(preset_data.get("network", {}))
    preset_network["network_module"] = arch["network_module"]
    network_section = preset_network

    # [optimizer] — from preset
    optimizer_section = copy.deepcopy(preset_data.get("optimizer", {}))

    # [training] — arch defaults + required_variant_args + preset training
    arch_defaults = copy.deepcopy(arch.get("defaults", {}))
    required_args = copy.deepcopy(arch.get("required_variant_args", {}))
    preset_training = copy.deepcopy(preset_data.get("training", {}))
    training_section = deep_merge(deep_merge(arch_defaults, required_args), preset_training)

    # Flatten [training.extra] passthrough: promote nested keys into [training]
    extra = training_section.get("extra")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if key in training_section:
                logger.warning(
                    "[training.extra] key '%s' collides with existing [training] key — extra value wins",
                    key,
                )
            training_section[key] = value
        del training_section["extra"]

    # [output] — computed paths
    output_dir = f"{machine_output_dir}/{run_name}"
    output_section = {
        "output_dir": output_dir,
        "output_name": run_name,
        "logging_dir": f"{output_dir}/logs",
        "log_prefix": f"{run_name}_",
    }

    # [advanced] — machine hardware flags
    advanced_section: dict[str, Any] = {}
    for key in ("gradient_checkpointing", "compile"):
        if key in machine_hw:
            advanced_section[key] = machine_hw[key]

    # [dataset] — placeholder for dataset_config path (set at compile-to-disk time)
    dataset_section = {"dataset_config": "__PLACEHOLDER__"}

    # [sampling] — empty for now, filled by preset if present
    sampling_section = copy.deepcopy(preset_data.get("sampling", {}))

    return {
        "model": model_section,
        "dataset": dataset_section,
        "network": network_section,
        "optimizer": optimizer_section,
        "training": training_section,
        "sampling": sampling_section,
        "output": output_section,
        "advanced": advanced_section,
    }


def _build_dataset_toml(
    arch: dict[str, Any],
    arch_key: str,
    persona_data: dict[str, Any],
    machine_data: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build the dataset TOML dict with multi-resolution expansion.

    Structure: [general] + [[datasets]] entries (one per resolution).
    """
    dataset_cfg = persona_data.get("dataset", {})
    machine_hw = machine_data.get("machine", {}).get("hardware", {})
    cache_suffix = arch.get("cache_suffix", arch_key)
    batch_size = machine_hw.get("default_batch_size", 1)

    # Interpolate dataset paths
    dataset_interpolated = interpolate(copy.deepcopy(dataset_cfg), context)

    image_directory = dataset_interpolated.get("image_directory", "")
    mask_directory = dataset_interpolated.get("mask_directory", "")
    use_masks = dataset_interpolated.get("use_masks", False)
    resolutions = dataset_interpolated.get(
        "resolutions", arch.get("dataset_defaults", {}).get("default_resolutions", [[1024, 1024]])
    )

    # Parent of image directory for cache sibling placement
    image_parent = str(Path(image_directory).parent) if image_directory else ""

    # [general]
    general = {
        "caption_extension": ".txt",
        "batch_size": batch_size,
    }

    # [[datasets]] — one per resolution
    datasets: list[dict[str, Any]] = []
    for res in resolutions:
        w, h = res[0], res[1]
        entry: dict[str, Any] = {
            "resolution": list(res),
            "image_directory": image_directory,
            "cache_directory": f"{image_parent}/{cache_suffix}_cache_{w}x{h}",
            "batch_size": batch_size,
        }
        if use_masks and mask_directory:
            entry["mask_directory"] = mask_directory
        datasets.append(entry)

    return {
        "general": general,
        "datasets": datasets,
    }


# ---------------------------------------------------------------------------
# Public API: compile_config
# ---------------------------------------------------------------------------


def compile_config(
    machine_path: Path,
    arch_key: str,
    persona_path: Path,
    preset_path: Path,
    override_path: Path | None = None,
    override_data: dict[str, Any] | None = None,
    file_override_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a full training config from TOML layer files.

    Args:
        machine_path: Path to machine TOML.
        arch_key: Architecture key or alias (resolved via registry).
        persona_path: Path to persona TOML.
        preset_path: Path to preset TOML.
        override_path: Optional path to override TOML (applied last).
            Ignored if file_override_data is provided.
        override_data: Optional override dict (applied after override_path).
            Used by the TUI for ephemeral overrides without writing a file.
        file_override_data: Pre-loaded override file data. When provided,
            override_path is not read (avoids TOCTOU re-read).

    Returns:
        Dict with keys: training_toml, dataset_toml, provenance.
    """
    # 1. Load layers
    machine_data = load_layer(machine_path)
    persona_data = load_layer(persona_path)
    preset_data = load_layer(preset_path)
    if file_override_data is None:
        file_override_data = load_layer(override_path) if override_path else {}

    # 2. Resolve arch
    canonical_key = resolve_arch_key(arch_key)
    arch = resolve_arch(arch_key)

    # 3. Build interpolation context
    context = _build_context(machine_data, persona_data)

    # 4. Apply overrides (if any) to preset — file overrides first, then dict overrides
    #    Only preset-layer sections (network, optimizer, training, sampling) are
    #    meaningful here. Non-preset sections (model, output, advanced, dataset) are
    #    computed from machine/persona/arch and would be silently ignored.
    if file_override_data:
        filtered_file = {k: v for k, v in file_override_data.items() if k in _OVERRIDABLE_SECTIONS or k == "preset"}
        if filtered_file:
            preset_data = deep_merge(preset_data, filtered_file)
    if override_data:
        filtered_data = {k: v for k, v in override_data.items() if k in _OVERRIDABLE_SECTIONS}
        if filtered_data:
            preset_data = deep_merge(preset_data, filtered_data)

    # 5. Build training TOML
    training_toml = _build_training_toml(
        arch=arch,
        arch_key=canonical_key,
        persona_data=persona_data,
        preset_data=preset_data,
        machine_data=machine_data,
        context=context,
    )

    # 6. Build dataset TOML
    dataset_toml = _build_dataset_toml(
        arch=arch,
        arch_key=canonical_key,
        persona_data=persona_data,
        machine_data=machine_data,
        context=context,
    )

    # 7. Provenance metadata (type validation already done in _build_training_toml)
    persona_name = persona_data.get("persona", {}).get("name", "unknown")
    preset_slug = preset_data.get("preset", {}).get("slug", "default")
    provenance = {
        "machine": Path(machine_path).name,
        "arch": canonical_key,
        "persona": Path(persona_path).name,
        "preset": Path(preset_path).name,
        "persona_name": persona_name,
        "preset_slug": preset_slug,
    }
    if override_path:
        provenance["override"] = Path(override_path).name
    if override_data:
        n_overrides = sum(len(v) if isinstance(v, dict) else 1 for v in override_data.values())
        provenance["ephemeral_overrides"] = n_overrides

    return {
        "training_toml": training_toml,
        "dataset_toml": dataset_toml,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# arch_display_dir: human-readable directory name from display_name
# ---------------------------------------------------------------------------


def _arch_display_dir(arch: dict[str, Any]) -> str:
    """Convert an arch display_name to a directory-safe uppercase form.

    Examples:
        "WAN 2.2 T2V"           -> "WAN-2.2-T2V"
        "Qwen-Image"            -> "QWEN-IMAGE"
        "FLUX.2 Klein-base-9B"  -> "FLUX.2-KLEIN-BASE-9B"
    """
    name = arch["display_name"]
    # Replace spaces with hyphens, then uppercase
    return name.replace(" ", "-").upper()


# ---------------------------------------------------------------------------
# env.sh emitter
# ---------------------------------------------------------------------------


def _render_env_sh(
    env_vars: dict[str, str],
    persona_name: str,
    machine_name: str,
    blissful_dir: str,
) -> str:
    """Render the env.sh script content.

    Produces a bash script that exports machine env vars and sources .env.local.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "#!/usr/bin/env bash",
        f"# Environment variables for {persona_name} on {machine_name}",
        f"# Compiled by blissful-config at {timestamp}",
        "",
    ]

    for key, value in sorted(env_vars.items()):
        lines.append(f'export {key}="{value}"')

    lines.append("")
    lines.append("# Source local secrets if present (API keys, tokens, etc.)")
    lines.append("# Create .env.local in the project root to set private variables.")
    lines.append(f'if [ -f "{blissful_dir}/.env.local" ]; then')
    lines.append(f'    source "{blissful_dir}/.env.local"')
    lines.append("fi")
    lines.append("")  # trailing newline

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifest management: tracking compiled artifacts
# ---------------------------------------------------------------------------


_TMP_MAX_AGE_SECONDS = 3600  # 1 hour


def _cleanup_tmp_files(output_dir: Path) -> None:
    """Remove leftover .tmp_* files from crashed compiles (older than 1 hour).

    Only deletes temp files older than _TMP_MAX_AGE_SECONDS to avoid
    removing in-flight temp files from concurrent compiles.
    """
    import time

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    cutoff = time.time() - _TMP_MAX_AGE_SECONDS
    for tmp_file in output_dir.rglob(".tmp_*"):
        try:
            if tmp_file.stat().st_mtime < cutoff:
                tmp_file.unlink()
                logger.debug("Cleaned up leftover temp file: %s", tmp_file)
        except OSError as e:
            logger.warning("Failed to clean up temp file %s: %s", tmp_file, e)


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read and return the manifest dict. Returns empty manifest if not found.

    Returns:
        Dict with schema_version, compiler_version, and entries list.
    """
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        with open(manifest_path, "rb") as f:
            return tomllib.load(f)
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "compiler_version": _COMPILER_VERSION,
        "entries": [],
    }


# Sections that can be overridden via --set (preset-layer sections)
_OVERRIDABLE_SECTIONS = {"network", "optimizer", "training", "sampling"}


def _parse_override_sets(override_sets: list[str]) -> dict[str, Any]:
    """Parse --set KEY=VALUE pairs into a section-keyed override dict.

    Keys use dot notation: "section.key=value" → {"section": {"key": parsed_value}}.
    Keys without a dot are placed in the "training" section by default.

    Only preset-layer sections (network, optimizer, training, sampling) can be
    overridden. Sections like model, output, advanced, and dataset are computed
    from machine/persona/arch and would be silently ignored — these are rejected.

    Uses the shared value parser for TOML-semantic type-safe parsing.
    """
    from blissful_tuner.config_manager.value_parser import parse_value

    result: dict[str, Any] = {}
    for item in override_sets:
        if "=" not in item:
            raise ValueError(f"Invalid --set format: '{item}' (expected KEY=VALUE)")
        key, raw_value = item.split("=", 1)
        parsed_value = parse_value(raw_value)

        if "." in key:
            section, field = key.split(".", 1)
        else:
            section, field = "training", key

        if section not in _OVERRIDABLE_SECTIONS:
            raise ValueError(
                f"Cannot override section '{section}' (key: {key}). "
                f"Only {sorted(_OVERRIDABLE_SECTIONS)} sections are overridable. "
                f"Sections like model, output, advanced are computed from machine/persona/arch."
            )

        if section not in result:
            result[section] = {}
        result[section][field] = parsed_value

    return result


def _compute_override_hash(
    override_sets: list[str] | None = None,
    override_data: dict[str, Any] | None = None,
    file_override_data: dict[str, Any] | None = None,
) -> str:
    """Compute SHA256[:16] hash of effective overrides from all three sources.

    Hashes the canonical JSON representation of the combined override dict.
    Returns empty string for base compiles (no overrides from any source).
    All three override sources contribute to artifact identity.

    Args:
        override_sets: --set KEY=VALUE pairs.
        override_data: Override dict from TUI/API.
        file_override_data: Pre-loaded override file data (already filtered
            to overridable sections + preset).
    """
    import json

    # Build combined override dict from all sources
    combined: dict[str, Any] = {}
    if file_override_data:
        filtered = {k: v for k, v in file_override_data.items() if k in _OVERRIDABLE_SECTIONS or k == "preset"}
        if filtered:
            combined = filtered
    if override_sets:
        parsed = _parse_override_sets(override_sets)
        combined = deep_merge(combined, parsed) if combined else parsed
    if override_data:
        combined = deep_merge(combined, override_data) if combined else override_data

    if not combined:
        return ""

    # Canonical JSON for deterministic hashing (sorted keys)
    serialized = json.dumps(combined, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


@contextlib.contextmanager
def _manifest_lock(manifest_path: Path):
    """Context manager for manifest file locking.

    Uses fcntl.flock on Unix for concurrent access safety.
    Falls back to no-lock on platforms without fcntl (Windows/Mac dev).
    """
    lock_path = manifest_path.parent / ".manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl

        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
    except ImportError:
        # fcntl not available (Windows) -- proceed without locking
        logger.debug("fcntl not available, proceeding without manifest lock")
        yield


def _upsert_manifest(manifest_path: Path, entry: dict[str, Any], *, _locked: bool = False) -> None:
    """Upsert a manifest entry using the 5-tuple key.

    The 5-tuple key is (persona, arch, preset, machine, override_hash).
    If an entry with the same key exists, it is replaced. Otherwise, a new
    entry is appended.

    The manifest is written atomically via temp file + os.replace.

    Args:
        _locked: If True, caller already holds the manifest lock. Skips
            re-acquisition to avoid deadlock.
    """
    manifest_path = Path(manifest_path)

    lock_cm = contextlib.nullcontext() if _locked else _manifest_lock(manifest_path)
    with lock_cm:
        manifest = _read_manifest(manifest_path)

        # Update compiler_version to current
        manifest["compiler_version"] = _COMPILER_VERSION
        manifest["schema_version"] = _MANIFEST_SCHEMA_VERSION

        # 5-tuple upsert key
        key_fields = ("persona", "arch", "preset", "machine", "override_hash")
        entry_key = tuple(entry[k] for k in key_fields)

        # Find existing entry with same key
        updated = False
        for i, existing in enumerate(manifest["entries"]):
            existing_key = tuple(existing.get(k, "") for k in key_fields)
            if existing_key == entry_key:
                manifest["entries"][i] = entry
                updated = True
                break

        if not updated:
            manifest["entries"].append(entry)

        # Render manifest as TOML using tomlkit for clean output
        doc = tomlkit.document()
        doc.add("schema_version", manifest["schema_version"])
        doc.add("compiler_version", manifest["compiler_version"])

        # Build entries as array of tables
        aot = tomlkit.aot()
        for e in manifest["entries"]:
            table = tomlkit.table()
            for k, v in e.items():
                table.add(k, v)
            aot.append(table)
        doc.add(tomlkit.nl())
        doc.add("entries", aot)

        manifest_str = tomlkit.dumps(doc)

        # Atomic write: temp file + os.replace
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(manifest_path.parent),
            prefix=".tmp_manifest_",
            suffix=".toml",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(manifest_str)
            os.replace(tmp_path, str(manifest_path))
        except BaseException:
            # Clean up temp file on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


# ---------------------------------------------------------------------------
# Public API: compile_to_disk
# ---------------------------------------------------------------------------


def compile_to_disk(
    machine_path: Path,
    arch_key: str,
    persona_path: Path,
    preset_path: Path,
    output_dir: Path,
    override_path: Path | None = None,
    override_sets: list[str] | None = None,
    override_data: dict[str, Any] | None = None,
    coerce: bool = False,
    allow_unknown: bool = False,
) -> dict[str, Any]:
    """Compile config, validate, and write training TOML, dataset TOML, and env.sh to disk.

    Pipeline order: compile → validate/coerce → render → write.

    Validation is fail-closed by default (strict=True). Pass coerce=True to
    strip unsupported features instead of erroring. A _stripped.txt sidecar
    report is written alongside the training TOML when keys are stripped.

    Uses transactional write ordering for crash safety:
    1. Compile in-memory
    2. Validate (strict or coerce) — BEFORE any disk write
    3. Write artifacts to unique .tmp_<uuid>_* temp files
    4. Rename to final paths (atomic on same filesystem)
    5. Upsert manifest entry

    Args:
        machine_path: Path to machine TOML.
        arch_key: Architecture key or alias (resolved via registry).
        persona_path: Path to persona TOML.
        preset_path: Path to preset TOML.
        output_dir: Root output directory for compiled configs.
        override_path: Optional path to override TOML (applied last).
        override_sets: Optional list of key=value override strings.
        override_data: Optional override dict (applied after override_sets).
            Used by TUI for type-safe ephemeral overrides without repr round-trip.
        coerce: If True, strip unsupported features instead of erroring.
        allow_unknown: If True, allow unknown --set keys.

    Returns:
        Dict with compile result (training_toml, dataset_toml, provenance)
        plus file paths: training_toml_path, dataset_toml_path, env_sh_path.
    """
    from blissful_tuner.config_manager.validator import validate_config
    from blissful_tuner.config_manager.writer import render_dataset_toml, render_training_toml

    output_dir = Path(output_dir)

    # 0. Best-effort cleanup of stale temp files from previous crashed compiles
    _cleanup_tmp_files(output_dir)

    from blissful_tuner.config_manager.validator import ConfigValidationError

    # 0b. Load and validate override_path sections (fail-closed)
    #     Load ONCE here and pass to compile_config + _compute_override_hash
    #     to avoid TOCTOU re-reads where the file could change between loads.
    file_override_data: dict[str, Any] = {}
    if override_path:
        file_override_data = load_layer(override_path)
        bad_file_sections = sorted(set(file_override_data.keys()) - _OVERRIDABLE_SECTIONS - {"preset"})
        if bad_file_sections:
            raise ConfigValidationError(
                f"Override file contains non-overridable section(s) {bad_file_sections}. "
                f"Only {sorted(_OVERRIDABLE_SECTIONS)} (and [preset] metadata) are allowed."
            )
        # Validate that every section value (including [preset]) is a dict
        for section, value in file_override_data.items():
            if not isinstance(value, dict):
                raise ConfigValidationError(f"Override file section '{section}' must be a table/dict, got {type(value).__name__}.")

    # 1. Build combined override dict from --set strings and/or direct override_data
    #    _parse_override_sets already rejects non-overridable sections for --set.
    #    Validate override_data sections here (TUI/API path).
    set_override_data = _parse_override_sets(override_sets) if override_sets else None
    if override_data:
        bad_sections = sorted(set(override_data.keys()) - _OVERRIDABLE_SECTIONS)
        if bad_sections:
            raise ConfigValidationError(
                f"Cannot override section(s) {bad_sections}. "
                f"Only {sorted(_OVERRIDABLE_SECTIONS)} are overridable. "
                f"Sections like model, output, advanced are computed from machine/persona/arch."
            )
        for section, value in override_data.items():
            if not isinstance(value, dict):
                raise ConfigValidationError(f"Override section '{section}' must be a dict, got {type(value).__name__}.")

    combined_override_data = set_override_data
    if override_data:
        if combined_override_data:
            combined_override_data = deep_merge(combined_override_data, override_data)
        else:
            combined_override_data = override_data

    # 2. Compile config (in-memory) — pass pre-loaded file data to avoid re-read.
    #    override_path is still passed for provenance (filename only, not re-read).
    result = compile_config(
        machine_path=machine_path,
        arch_key=arch_key,
        persona_path=persona_path,
        preset_path=preset_path,
        override_path=override_path,
        file_override_data=file_override_data or None,
        override_data=combined_override_data,
    )

    training_toml = result["training_toml"]
    dataset_toml = result["dataset_toml"]
    provenance = result["provenance"]

    # 3. Validate BEFORE writing — fail-closed by default
    arch = resolve_arch(arch_key)
    canonical_key = provenance["arch"]
    stripped_report = validate_config(training_toml, arch, strict=not coerce)

    # 3b. Validate override keys from ALL sources against a baseline compile
    # (without any overrides). Section-aware: each key must exist in the
    # same section of the baseline, preventing cross-section injection
    # (e.g. 'training.dit' passing because 'dit' exists in [model]).
    all_override_keys: dict[str, set[str]] = {}
    if file_override_data:
        for section, fields in file_override_data.items():
            if section in _OVERRIDABLE_SECTIONS and isinstance(fields, dict):
                all_override_keys.setdefault(section, set()).update(fields.keys())
    if combined_override_data:
        for section, fields in combined_override_data.items():
            if isinstance(fields, dict):
                all_override_keys.setdefault(section, set()).update(fields.keys())

    if not allow_unknown and all_override_keys:
        # Compile a baseline without any overrides to get the true known key set
        baseline = compile_config(
            machine_path=machine_path,
            arch_key=arch_key,
            persona_path=persona_path,
            preset_path=preset_path,
        )
        baseline_training = baseline["training_toml"]

        # Check each override field against the SAME section in the baseline
        for section_name, fields in all_override_keys.items():
            baseline_section = baseline_training.get(section_name, {})
            if not isinstance(baseline_section, dict):
                baseline_section = {}
            for field in fields:
                if field not in baseline_section:
                    raise ConfigValidationError(f"Unknown override key '{section_name}.{field}'. Use --allow-unknown to force.")

    # 4. Load machine data for env vars and naming
    machine_data = load_layer(machine_path)
    machine = machine_data.get("machine", {})
    machine_name = machine.get("name", "unknown")
    machine_paths_cfg = machine.get("paths", {})
    blissful_dir = machine_paths_cfg.get("blissful_dir", "/opt/blissful-tuner")
    env_vars = machine.get("env", {})

    # 5. Extract effective persona name and preset slug from provenance
    #    These come from the compiled result (post-merge), not raw source files,
    #    so that override_path [preset] changes are reflected in naming.
    persona_name = provenance["persona_name"]
    preset_slug = provenance["preset_slug"]

    # 5b. Validate persona_name and preset_slug are filesystem-safe.
    #     Type validation (must be string) already done in _build_training_toml.
    _SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    for label, value in [("persona name", persona_name), ("preset slug", preset_slug)]:
        if not _SAFE_NAME_RE.match(value):
            raise ConfigValidationError(
                f"{label.capitalize()} '{value}' contains filesystem-unsafe characters. "
                f"Only alphanumeric, hyphens, underscores, and dots are allowed."
            )

    # 6. Compute output paths — include _ov_{hash8} suffix when overrides exist
    arch_dir = _arch_display_dir(arch)
    out_base = output_dir / machine_name / persona_name / arch_dir
    out_base.mkdir(parents=True, exist_ok=True)

    persona_lower = persona_name.lower()
    override_hash = _compute_override_hash(
        override_sets=override_sets, override_data=override_data, file_override_data=file_override_data or None
    )
    override_suffix = f"_ov_{override_hash}" if override_hash else ""
    run_name = f"{persona_lower}_{canonical_key}_{preset_slug}{override_suffix}"

    training_filename = f"{run_name}.toml"
    dataset_filename = f"{run_name}_dataset.toml"
    env_filename = "env.sh"

    training_path = out_base / training_filename
    dataset_path = out_base / dataset_filename
    env_path = out_base / env_filename

    # 6b. Output path collision guard — detect provenance mismatch before writing.
    #     Uses the same provenance dict that render_training_toml writes to the header,
    #     so field values match exactly during comparison.
    from blissful_tuner.config_manager.validator import check_output_path_collision

    for guarded_path in (training_path, dataset_path):
        check_output_path_collision(guarded_path, provenance)

    # 7. Set dataset_config pointer to the absolute path of the dataset TOML
    training_toml["dataset"]["dataset_config"] = str(dataset_path.resolve())

    # 8. Render TOML strings (after validation/coerce, so stripped keys are gone)
    training_str = render_training_toml(training_toml, provenance)
    dataset_str = render_dataset_toml(dataset_toml, provenance)

    # 9. Render env.sh
    env_str = _render_env_sh(env_vars, persona_name, machine_name, blissful_dir)

    # 10. Transactional write: unique temp files first, then rename
    tmp_id = uuid.uuid4().hex[:8]
    tmp_training = out_base / f".tmp_{tmp_id}_{training_filename}"
    tmp_dataset = out_base / f".tmp_{tmp_id}_{dataset_filename}"
    tmp_env = out_base / f".tmp_{tmp_id}_{env_filename}"

    # Write to temp files
    tmp_training.write_text(training_str)
    tmp_dataset.write_text(dataset_str)
    tmp_env.write_text(env_str)

    # Make temp env.sh executable before rename
    current_mode = os.stat(tmp_env).st_mode
    os.chmod(tmp_env, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # 10b. Prepare manifest entry BEFORE publishing (for atomic publish+upsert)
    run_id = uuid.uuid4().hex[:12]
    training_rel = str(training_path.relative_to(output_dir))
    dataset_rel = str(dataset_path.relative_to(output_dir))

    env_rel = str(env_path.relative_to(output_dir))

    manifest_entry = {
        "persona": persona_name,
        "arch": canonical_key,
        "preset": preset_slug,
        "machine": machine_name,
        "override_hash": override_hash,
        "persona_source": Path(persona_path).stem,
        "preset_source": Path(preset_path).stem,
        "compiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "status": "ok",
        "training_config": training_rel,
        "dataset_config": dataset_rel,
        "env_config": env_rel,
    }

    manifest_path = output_dir / "manifest.toml"

    # Rename + manifest upsert under a single lock so prune cannot
    # observe files without a corresponding manifest entry.
    # On failure, restore previous artifacts from backups so recompile
    # failures never destroy last-known-good outputs.
    with _manifest_lock(manifest_path):
        backup_id = uuid.uuid4().hex[:8]
        finals = [str(training_path), str(dataset_path), str(env_path)]
        backups: dict[str, str] = {}  # final_path -> backup_path
        published: set[str] = set()  # files successfully renamed from tmp

        try:
            # Phase 1: Back up existing artifacts (inside try so partial
            # backups are restored on failure at any point)
            for final in finals:
                if os.path.exists(final):
                    backup = f"{final}.bak_{backup_id}"
                    os.replace(final, backup)
                    backups[final] = backup

            # Phase 2: Publish new artifacts (track each successful rename)
            os.replace(str(tmp_training), str(training_path))
            published.add(str(training_path))
            os.replace(str(tmp_dataset), str(dataset_path))
            published.add(str(dataset_path))
            os.replace(str(tmp_env), str(env_path))
            published.add(str(env_path))

            # Phase 3: Upsert manifest
            _upsert_manifest(manifest_path, manifest_entry, _locked=True)
        except BaseException:
            # Restore previous good artifacts from backups
            for final, backup in backups.items():
                with contextlib.suppress(OSError):
                    os.replace(backup, final)
            # Remove only files that were published (renamed from tmp) during
            # this compile and have no backup to restore from. This avoids
            # accidentally deleting pre-existing files that weren't backed up
            # because the backup phase failed before reaching them.
            for final in published:
                if final not in backups:
                    with contextlib.suppress(OSError):
                        os.unlink(final)
            raise
        else:
            # Success — remove backups
            for backup in backups.values():
                with contextlib.suppress(OSError):
                    os.unlink(backup)

            # Sidecar management (best-effort after successful publish+manifest)
            stripped_path = training_path.with_name(training_path.stem + "_stripped.txt")
            try:
                if not stripped_report.is_empty():
                    stripped_path.write_text(stripped_report.format())
                    result["stripped_report_path"] = str(stripped_path)
                elif stripped_path.exists():
                    stripped_path.unlink()
            except OSError as e:
                logger.warning("Failed to manage _stripped.txt sidecar: %s", e)

    # 12. Return result with file paths
    result["training_toml_path"] = str(training_path)
    result["dataset_toml_path"] = str(dataset_path)
    result["env_sh_path"] = str(env_path)

    return result
