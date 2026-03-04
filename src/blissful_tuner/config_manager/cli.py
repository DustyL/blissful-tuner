"""CLI entry points for blissful-config tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _find_meta_dir() -> Path:
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


def _find_compiled_dir() -> Path:
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


def _list_personas(meta_dir: Path) -> list[str]:
    """List available persona names from a personas/ subdirectory."""
    personas_dir = meta_dir / "personas"
    if not personas_dir.is_dir():
        return []
    return sorted(p.stem for p in personas_dir.glob("*.toml"))


def _list_presets(meta_dir: Path) -> list[str]:
    """List available preset slugs from a presets/ subdirectory."""
    presets_dir = meta_dir / "presets"
    if not presets_dir.is_dir():
        return []
    return sorted(p.stem for p in presets_dir.glob("*.toml"))


def _parse_set_overrides(set_overrides: list[str] | None) -> dict[str, str]:
    """Parse --set KEY=VALUE pairs into a dict.

    Splits on the first '=' only, so values containing '=' are preserved.

    Raises:
        ValueError: If a --set argument does not contain '='.
    """
    if not set_overrides:
        return {}
    result: dict[str, str] = {}
    for item in set_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set format: '{item}' (expected KEY=VALUE)")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def run_compile(
    persona: str,
    arch: str,
    preset: str,
    machine_path: Path,
    persona_path: Path,
    preset_path: Path,
    output_dir: Path,
    override_sets: list[str] | None = None,
    coerce: bool = False,
    allow_unknown: bool = False,
) -> dict[str, Any]:
    """Run the compile pipeline. Returns result dict with file paths.

    Args:
        persona: Persona name (informational, used for messaging).
        arch: Architecture key (resolved via registry).
        preset: Preset name (informational, used for messaging).
        machine_path: Path to machine TOML.
        persona_path: Path to persona TOML.
        preset_path: Path to preset TOML.
        output_dir: Root output directory for compiled configs.
        override_sets: List of KEY=VALUE strings to apply as overrides.
        coerce: If True, strip unsupported features instead of erroring.
        allow_unknown: If True, allow unknown keys in --set overrides.

    Returns:
        Dict with compile result including training_toml_path, dataset_toml_path, etc.
    """
    from blissful_tuner.config_manager.compiler import compile_to_disk

    result = compile_to_disk(
        machine_path=machine_path,
        arch_key=arch,
        persona_path=persona_path,
        preset_path=preset_path,
        output_dir=output_dir,
        override_sets=override_sets,
    )

    # Optionally validate with coerce mode
    if coerce:
        from blissful_tuner.config_manager.registry import resolve_arch
        from blissful_tuner.config_manager.validator import validate_config

        arch_entry = resolve_arch(arch)
        validate_config(result["training_toml"], arch_entry, strict=False)

    return result


def main_compile(args: list[str] | None = None) -> int:
    """Entry point for bt-compile CLI.

    Returns:
        Exit code (0 for success, 1 for usage/input error, 2 for compile error).
    """
    parser = argparse.ArgumentParser(
        prog="bt-compile",
        description="Compile layered TOML configs into standalone training configs.",
    )

    # List commands
    list_group = parser.add_argument_group("listing")
    list_group.add_argument("--list-archs", action="store_true", help="List available architectures")
    list_group.add_argument("--list-personas", action="store_true", help="List available personas")
    list_group.add_argument("--list-presets", action="store_true", help="List available presets")

    # Compile positional args
    parser.add_argument("persona", nargs="?", help="Persona name (or use --all)")
    parser.add_argument("arch", nargs="?", help="Architecture key (e.g. wan22_t2v, qwen_image)")
    parser.add_argument("preset", nargs="?", help="Preset name (e.g. adamw_cosine)")

    # Compile options
    parser.add_argument("--machine", default="default", help="Machine name (default: default)")
    parser.add_argument("--all", action="store_true", dest="compile_all", help="Compile for all personas")
    parser.add_argument("--set", action="append", dest="set_overrides", metavar="KEY=VALUE", help="Override a key (repeatable)")
    parser.add_argument("--coerce", action="store_true", help="Strip unsupported features instead of erroring")
    parser.add_argument("--allow-unknown", action="store_true", help="Allow unknown keys in --set overrides")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: configs/compiled/)")

    # Explicit path overrides (for testing and scripted usage)
    parser.add_argument("--machine-path", type=Path, help="Explicit path to machine TOML (overrides --machine)")
    parser.add_argument("--persona-path", type=Path, help="Explicit path to persona TOML (overrides PERSONA lookup)")
    parser.add_argument("--preset-path", type=Path, help="Explicit path to preset TOML (overrides PRESET lookup)")

    parsed = parser.parse_args(args)

    # --- Handle list commands ---
    if parsed.list_archs:
        from blissful_tuner.config_manager.registry import ARCH_REGISTRY

        for key, arch in sorted(ARCH_REGISTRY.items()):
            print(f"  {key:30s} {arch['display_name']}")
        return 0

    if parsed.list_personas:
        try:
            meta_dir = _find_meta_dir()
        except FileNotFoundError:
            print("No configs/meta/ directory found", file=sys.stderr)
            return 0
        for name in _list_personas(meta_dir):
            print(f"  {name}")
        return 0

    if parsed.list_presets:
        try:
            meta_dir = _find_meta_dir()
        except FileNotFoundError:
            print("No configs/meta/ directory found", file=sys.stderr)
            return 0
        for name in _list_presets(meta_dir):
            print(f"  {name}")
        return 0

    # --- Handle --all positional shift ---
    # With --all, PERSONA is replaced: `bt-compile --all ARCH PRESET`
    # argparse assigns ARCH to the persona slot and PRESET to the arch slot.
    # Shift them into the correct positions.
    if parsed.compile_all and parsed.persona and not parsed.arch:
        # Only persona was filled: `--all ARCH` -> shift persona->arch
        parsed.arch = parsed.persona
        parsed.persona = None
    elif parsed.compile_all and parsed.persona and parsed.arch and not parsed.preset:
        # persona and arch filled: `--all ARCH PRESET` -> shift both
        parsed.preset = parsed.arch
        parsed.arch = parsed.persona
        parsed.persona = None

    # --- Validate required compile args ---
    if not parsed.arch or not parsed.preset:
        if not parsed.persona and not parsed.compile_all:
            # No arguments at all -- show usage
            parser.print_usage(sys.stderr)
            return 1
        # Partial arguments
        parser.print_usage(sys.stderr)
        print("error: ARCH and PRESET are required for compilation", file=sys.stderr)
        return 1

    if not parsed.persona and not parsed.compile_all:
        parser.print_usage(sys.stderr)
        print("error: either PERSONA or --all is required", file=sys.stderr)
        return 1

    # --- Parse --set overrides ---
    try:
        set_overrides_dict = _parse_set_overrides(parsed.set_overrides)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # --- Resolve output directory ---
    if parsed.output_dir:
        output_dir = parsed.output_dir
    else:
        try:
            output_dir = _find_compiled_dir()
        except FileNotFoundError:
            print("error: could not find configs/ directory; use --output-dir", file=sys.stderr)
            return 1

    # --- Resolve paths ---
    try:
        meta_dir = _find_meta_dir()
    except FileNotFoundError:
        meta_dir = None

    # Machine path
    if parsed.machine_path:
        machine_path = parsed.machine_path
    elif meta_dir:
        if parsed.machine == "default":
            machine_path = meta_dir / "machines" / "default.toml"
        else:
            machine_path = meta_dir / "machines" / f"{parsed.machine}.toml"
    else:
        print("error: could not find configs/meta/ directory; use --machine-path", file=sys.stderr)
        return 1

    # Preset path
    if parsed.preset_path:
        preset_path = parsed.preset_path
    elif meta_dir:
        preset_path = meta_dir / "presets" / f"{parsed.preset}.toml"
    else:
        print("error: could not find configs/meta/ directory; use --preset-path", file=sys.stderr)
        return 1

    # --- Build persona list ---
    if parsed.compile_all:
        if meta_dir is None:
            print("error: --all requires configs/meta/ directory", file=sys.stderr)
            return 1
        persona_names = _list_personas(meta_dir)
        if not persona_names:
            print("error: no persona TOML files found in configs/meta/personas/", file=sys.stderr)
            return 1
        personas = [(name, meta_dir / "personas" / f"{name}.toml") for name in persona_names]
    else:
        persona_name = parsed.persona
        if parsed.persona_path:
            persona_path = parsed.persona_path
        elif meta_dir:
            persona_path = meta_dir / "personas" / f"{persona_name}.toml"
        else:
            print("error: could not find configs/meta/ directory; use --persona-path", file=sys.stderr)
            return 1
        personas = [(persona_name, persona_path)]

    # --- Run compile for each persona ---
    failed = False
    for persona_name, persona_path in personas:
        try:
            result = run_compile(
                persona=persona_name,
                arch=parsed.arch,
                preset=parsed.preset,
                machine_path=machine_path,
                persona_path=persona_path,
                preset_path=preset_path,
                output_dir=output_dir,
                override_sets=parsed.set_overrides,
                coerce=parsed.coerce,
                allow_unknown=parsed.allow_unknown,
            )
            print(f"  Compiled: {persona_name}/{parsed.arch}/{parsed.preset}")
            print(f"    Training: {result['training_toml_path']}")
            print(f"    Dataset:  {result['dataset_toml_path']}")
        except Exception as e:
            print(f"  FAILED: {persona_name}/{parsed.arch}/{parsed.preset}: {e}", file=sys.stderr)
            failed = True

    return 2 if failed else 0
