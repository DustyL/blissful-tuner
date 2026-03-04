"""Config compiler: layer loading, deep merge, interpolation, TOML emission."""

from __future__ import annotations

import copy
import re
import tomllib
from pathlib import Path
from typing import Any

from blissful_tuner.config_manager.registry import resolve_arch, resolve_arch_key


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
    persona_name = persona_data.get("persona", {}).get("name", "unknown")
    persona_lower = persona_name.lower()
    preset_slug = preset_data.get("preset", {}).get("slug", "default")
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
) -> dict[str, Any]:
    """Compile a full training config from TOML layer files.

    Args:
        machine_path: Path to machine TOML.
        arch_key: Architecture key or alias (resolved via registry).
        persona_path: Path to persona TOML.
        preset_path: Path to preset TOML.
        override_path: Optional path to override TOML (applied last).

    Returns:
        Dict with keys: training_toml, dataset_toml, provenance.
    """
    # 1. Load layers
    machine_data = load_layer(machine_path)
    persona_data = load_layer(persona_path)
    preset_data = load_layer(preset_path)
    override_data = load_layer(override_path) if override_path else {}

    # 2. Resolve arch
    canonical_key = resolve_arch_key(arch_key)
    arch = resolve_arch(arch_key)

    # 3. Build interpolation context
    context = _build_context(machine_data, persona_data)

    # 4. Apply overrides (if any) to preset
    if override_data:
        preset_data = deep_merge(preset_data, override_data)

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

    # 7. Provenance metadata
    provenance = {
        "machine": Path(machine_path).name,
        "arch": canonical_key,
        "persona": Path(persona_path).name,
        "preset": Path(preset_path).name,
    }
    if override_path:
        provenance["override"] = Path(override_path).name

    return {
        "training_toml": training_toml,
        "dataset_toml": dataset_toml,
        "provenance": provenance,
    }
