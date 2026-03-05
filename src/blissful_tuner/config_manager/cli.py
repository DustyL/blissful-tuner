"""CLI entry points for blissful-config tools."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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

    # --- Validate --set overrides (format check: KEY=VALUE) ---
    try:
        _parse_set_overrides(parsed.set_overrides)
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


def main_diff(args: list[str] | None = None) -> int:
    """Entry point for bt-diff CLI.

    Compares two TOML config files semantically and reports differences.

    Returns:
        Exit code: 0 if no differences, 1 if differences found, 2 on error.
    """
    parser = argparse.ArgumentParser(
        prog="bt-diff",
        description="Compare two TOML config files semantically.",
    )
    parser.add_argument("file_a", type=Path, help="First TOML file")
    parser.add_argument("file_b", type=Path, help="Second TOML file")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten all sections before comparing (simulates runtime loading)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output exit code, no report",
    )

    parsed = parser.parse_args(args)

    # Validate files exist
    for path in (parsed.file_a, parsed.file_b):
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2

    # Load both files
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(parsed.file_a, "rb") as f:
        data_a = tomllib.load(f)
    with open(parsed.file_b, "rb") as f:
        data_b = tomllib.load(f)

    # Optionally flatten
    if parsed.flatten:
        from blissful_tuner.config_manager.differ import flatten_toml

        data_a = flatten_toml(data_a)
        data_b = flatten_toml(data_b)

    # Compute diff
    from blissful_tuner.config_manager.differ import format_diff_report, semantic_diff

    diffs = semantic_diff(data_a, data_b)

    if not parsed.quiet:
        report = format_diff_report(diffs)
        print(report)

    return 1 if diffs else 0


# ---------------------------------------------------------------------------
# bt-prune: stale compiled artifact pruning
# ---------------------------------------------------------------------------

# File extensions that bt-prune is allowed to delete
_PRUNABLE_EXTENSIONS = {".toml", ".sh", ".txt"}

# Files that should never be treated as orphans
_INTERNAL_FILES = {"manifest.toml", ".manifest.lock"}


def find_stale_entries(compiled_dir: Path, meta_dir: Path | None = None) -> list[dict[str, Any]]:
    """Phase 1: check each manifest entry against source files.

    For each manifest entry, verify:
    - Persona TOML still exists in meta_dir/personas/
    - Arch key still exists in ARCH_REGISTRY
    - Preset TOML still exists in meta_dir/presets/

    Args:
        compiled_dir: Path to configs/compiled/ containing manifest.toml.
        meta_dir: Path to configs/meta/ containing personas/ and presets/.

    Returns:
        List of dicts with keys: entry (the manifest entry), reason (why it's stale).
    """
    from blissful_tuner.config_manager.registry import ARCH_REGISTRY

    manifest_path = compiled_dir / "manifest.toml"
    if not manifest_path.exists():
        return []

    with open(manifest_path, "rb") as f:
        manifest = tomllib.load(f)

    stale: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        persona = entry.get("persona", "")
        arch = entry.get("arch", "")
        preset = entry.get("preset", "")

        # Check persona TOML exists
        if meta_dir is not None:
            persona_path = meta_dir / "personas" / f"{persona}.toml"
            if not persona_path.exists():
                stale.append(
                    {
                        "entry": entry,
                        "reason": f"Persona TOML not found: {persona_path}",
                    }
                )
                continue

        # Check arch key exists in registry
        if arch not in ARCH_REGISTRY:
            stale.append(
                {
                    "entry": entry,
                    "reason": f"Arch key '{arch}' not found in ARCH_REGISTRY",
                }
            )
            continue

        # Check preset TOML exists
        if meta_dir is not None:
            preset_path = meta_dir / "presets" / f"{preset}.toml"
            if not preset_path.exists():
                stale.append(
                    {
                        "entry": entry,
                        "reason": f"Preset TOML not found: {preset_path}",
                    }
                )
                continue

    return stale


def find_orphaned_files(compiled_dir: Path) -> list[Path]:
    """Phase 2: find files on disk not referenced by any manifest entry.

    Scans compiled_dir recursively for .toml, .sh, .txt files.
    Files referenced by manifest entries (training_config, dataset_config)
    are not orphaned. Internal files (manifest.toml, .manifest.lock) and
    hidden files (starting with '.') are excluded.

    Safety: symlinks whose realpath resolves outside compiled_dir are skipped.

    Args:
        compiled_dir: Path to configs/compiled/.

    Returns:
        List of Paths to orphaned files (absolute).
    """
    compiled_dir = Path(compiled_dir).resolve()

    # Read manifest to get referenced files
    manifest_path = compiled_dir / "manifest.toml"
    referenced: set[Path] = set()
    if manifest_path.exists():
        with open(manifest_path, "rb") as f:
            manifest = tomllib.load(f)
        for entry in manifest.get("entries", []):
            for key in ("training_config", "dataset_config"):
                rel = entry.get(key, "")
                if rel:
                    referenced.add((compiled_dir / rel).resolve())

    # Scan disk for prunable files
    orphans: list[Path] = []
    for path in compiled_dir.rglob("*"):
        if not path.is_file():
            continue

        # Skip internal files
        if path.name in _INTERNAL_FILES:
            continue

        # Skip hidden files (dotfiles like .manifest.lock, .tmp_*)
        if path.name.startswith("."):
            continue

        # Only consider prunable extensions
        if path.suffix not in _PRUNABLE_EXTENSIONS:
            continue

        # Safety: verify realpath is inside compiled tree
        real = Path(os.path.realpath(path))
        try:
            real.relative_to(compiled_dir)
        except ValueError:
            # Symlink points outside compiled dir -- skip
            logger.warning("Skipping symlink that resolves outside compiled tree: %s -> %s", path, real)
            continue

        # Check if referenced by manifest
        if real not in referenced:
            orphans.append(path)

    return orphans


def prune_stale_artifacts(
    compiled_dir: Path,
    meta_dir: Path | None = None,
    dry_run: bool = True,
    machine: str | None = None,
    persona: str | None = None,
) -> dict[str, Any]:
    """Two-phase source-aware reconciliation for compiled artifacts.

    Phase 1: Find stale manifest entries (missing persona/arch/preset sources).
    Phase 2: Find orphaned files on disk not referenced by surviving entries.

    Args:
        compiled_dir: Path to configs/compiled/.
        meta_dir: Path to configs/meta/ (for persona/preset existence checks).
        dry_run: If True, report only (no deletions). Default True.
        machine: If set, only prune entries for this machine name.
        persona: If set, only prune entries for this persona name.

    Returns:
        Dict with stale_count, orphan_count, stale_entries, orphaned_files.
    """
    import tomlkit

    from blissful_tuner.config_manager.compiler import _read_manifest

    compiled_dir = Path(compiled_dir).resolve()
    manifest_path = compiled_dir / "manifest.toml"

    # Phase 1: find stale entries
    all_stale = find_stale_entries(compiled_dir, meta_dir=meta_dir)

    # Apply scoping filters
    scoped_stale = []
    for item in all_stale:
        entry = item["entry"]
        if machine is not None and entry.get("machine") != machine:
            continue
        if persona is not None and entry.get("persona") != persona:
            continue
        scoped_stale.append(item)

    # Report stale entries
    for item in scoped_stale:
        entry = item["entry"]
        logger.info(
            "Stale entry: %s/%s/%s (machine=%s) -- %s",
            entry.get("persona"),
            entry.get("arch"),
            entry.get("preset"),
            entry.get("machine"),
            item["reason"],
        )
        print(
            f"  STALE: {entry.get('persona')}/{entry.get('arch')}/{entry.get('preset')}"
            f" (machine={entry.get('machine')}) -- {item['reason']}"
        )

    # Remove stale entries from manifest (if executing)
    if not dry_run and scoped_stale and manifest_path.exists():
        manifest = _read_manifest(manifest_path)

        # Build set of stale 5-tuple keys
        key_fields = ("persona", "arch", "preset", "machine", "override_hash")
        stale_keys = {tuple(item["entry"].get(k, "") for k in key_fields) for item in scoped_stale}

        # Filter entries
        surviving_entries = [e for e in manifest.get("entries", []) if tuple(e.get(k, "") for k in key_fields) not in stale_keys]
        manifest["entries"] = surviving_entries

        # Rewrite manifest
        doc = tomlkit.document()
        doc.add("schema_version", manifest.get("schema_version", 1))
        doc.add("compiler_version", manifest.get("compiler_version", "0.1.0"))
        doc.add(tomlkit.nl())
        if surviving_entries:
            aot = tomlkit.aot()
            for e in surviving_entries:
                table = tomlkit.table()
                for k, v in e.items():
                    table.add(k, v)
                aot.append(table)
            doc.add("entries", aot)
        else:
            # Empty entries: use inline array to produce `entries = []`
            doc.add("entries", tomlkit.array())
        manifest_path.write_text(tomlkit.dumps(doc))

    # Phase 2: find orphaned files (after manifest cleanup)
    orphans = find_orphaned_files(compiled_dir)

    # Apply machine scoping to orphans: only delete files under machine/ subdirectory
    if machine is not None:
        machine_prefix = compiled_dir / machine
        orphans = [p for p in orphans if p.resolve().is_relative_to(machine_prefix.resolve())]

    # Apply persona scoping: only delete files under */persona/ subdirectory
    if persona is not None:
        orphans = [p for p in orphans if _file_under_persona(p, compiled_dir, persona)]

    # Collect _stripped.txt sidecars for orphaned .toml files
    sidecars: list[Path] = []
    for orphan in orphans:
        if orphan.suffix == ".toml":
            stripped = orphan.with_name(orphan.stem + "_stripped.txt")
            if stripped.exists() and stripped not in orphans:
                sidecars.append(stripped)
    orphans.extend(sidecars)

    # Report orphans
    for orphan in orphans:
        print(f"  ORPHAN: {orphan}")

    # Delete orphaned files (if executing)
    deleted_count = 0
    if not dry_run:
        for orphan in orphans:
            # Final safety check: realpath inside compiled tree
            real = Path(os.path.realpath(orphan))
            try:
                real.relative_to(compiled_dir)
            except ValueError:
                logger.warning("Refusing to delete file outside compiled tree: %s -> %s", orphan, real)
                continue

            # Only delete prunable extensions
            if orphan.suffix not in _PRUNABLE_EXTENSIONS:
                logger.warning("Refusing to delete non-prunable extension: %s", orphan)
                continue

            try:
                orphan.unlink()
                logger.info("Deleted orphaned file: %s", orphan)
                deleted_count += 1
            except OSError as e:
                logger.warning("Failed to delete orphaned file %s: %s", orphan, e)

    result = {
        "stale_count": len(scoped_stale),
        "orphan_count": deleted_count if not dry_run else len(orphans),
        "stale_entries": scoped_stale,
        "orphaned_files": orphans,
    }

    if dry_run:
        print(f"\n  Dry run: {result['stale_count']} stale entries, {result['orphan_count']} orphaned files")
        print("  Use --execute to apply changes.")
    else:
        print(f"\n  Pruned: {result['stale_count']} stale entries removed, {deleted_count} orphaned files deleted")

    return result


def _file_under_persona(file_path: Path, compiled_dir: Path, persona: str) -> bool:
    """Check if a file is under a persona subdirectory.

    Expected structure: compiled_dir / machine / persona / arch / ...
    """
    try:
        rel = file_path.resolve().relative_to(compiled_dir.resolve())
        parts = rel.parts
        # machine / persona / arch / file
        return len(parts) >= 2 and parts[1] == persona
    except ValueError:
        return False


def main_prune(args: list[str] | None = None) -> int:
    """Entry point for bt-prune CLI.

    Cleans up stale compiled artifacts using two-phase source-aware reconciliation.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    parser = argparse.ArgumentParser(
        prog="bt-prune",
        description="Clean up stale compiled training config artifacts.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete orphaned files and remove stale manifest entries (default: dry-run)",
    )
    parser.add_argument("--machine", help="Only prune artifacts for this machine name")
    parser.add_argument("--persona", help="Only prune artifacts for this persona name")
    parser.add_argument("--compiled-dir", type=Path, help="Path to configs/compiled/ (default: auto-detect)")
    parser.add_argument("--meta-dir", type=Path, help="Path to configs/meta/ (default: auto-detect)")

    parsed = parser.parse_args(args)

    # Resolve compiled dir
    if parsed.compiled_dir:
        compiled_dir = parsed.compiled_dir
    else:
        try:
            compiled_dir = _find_compiled_dir()
        except FileNotFoundError:
            print("error: could not find configs/compiled/ directory; use --compiled-dir", file=sys.stderr)
            return 1

    if not compiled_dir.is_dir():
        print(f"error: compiled directory does not exist: {compiled_dir}", file=sys.stderr)
        return 1

    # Resolve meta dir
    if parsed.meta_dir:
        meta_dir = parsed.meta_dir
    else:
        try:
            meta_dir = _find_meta_dir()
        except FileNotFoundError:
            meta_dir = None

    dry_run = not parsed.execute

    try:
        prune_stale_artifacts(
            compiled_dir=compiled_dir,
            meta_dir=meta_dir,
            dry_run=dry_run,
            machine=parsed.machine,
            persona=parsed.persona,
        )
    except Exception as e:
        print(f"error: prune failed: {e}", file=sys.stderr)
        return 1

    return 0
