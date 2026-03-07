"""Tests for bt-prune: stale compiled artifact pruning."""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomlkit

from blissful_tuner.config_manager.cli import (
    find_orphaned_files,
    find_stale_entries,
    main_prune,
    prune_stale_artifacts,
)


def _write_manifest(compiled_dir: Path, entries: list[dict]) -> None:
    """Write a manifest.toml with the given entries."""
    doc = tomlkit.document()
    doc.add("schema_version", 1)
    doc.add("compiler_version", "0.1.0")
    aot = tomlkit.aot()
    for e in entries:
        table = tomlkit.table()
        for k, v in e.items():
            table.add(k, v)
        aot.append(table)
    doc.add(tomlkit.nl())
    doc.add("entries", aot)
    manifest_path = compiled_dir / "manifest.toml"
    manifest_path.write_text(tomlkit.dumps(doc))


def _make_entry(
    persona: str = "Alice",
    arch: str = "qwen_image",
    preset: str = "test_adamw",
    machine: str = "test",
    override_hash: str = "",
    training_config: str | None = None,
    dataset_config: str | None = None,
) -> dict:
    """Create a manifest entry dict with defaults."""
    persona_lower = persona.lower()
    if training_config is None:
        training_config = f"{machine}/{persona}/{arch.upper()}/{persona_lower}_{arch}_{preset}.toml"
    if dataset_config is None:
        dataset_config = f"{machine}/{persona}/{arch.upper()}/{persona_lower}_{arch}_{preset}_dataset.toml"
    return {
        "persona": persona,
        "arch": arch,
        "preset": preset,
        "machine": machine,
        "override_hash": override_hash,
        "compiled_at": "2026-03-04T00:00:00+00:00",
        "run_id": "abc123",
        "status": "ok",
        "training_config": training_config,
        "dataset_config": dataset_config,
    }


def _create_compiled_file(compiled_dir: Path, rel_path: str, content: str = "# test") -> Path:
    """Create a file at compiled_dir / rel_path."""
    full = compiled_dir / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    return full


def _create_persona(meta_dir: Path, name: str) -> Path:
    """Create a persona TOML in meta_dir/personas/."""
    personas_dir = meta_dir / "personas"
    personas_dir.mkdir(parents=True, exist_ok=True)
    path = personas_dir / f"{name}.toml"
    path.write_text(f'[persona]\nname = "{name}"\n')
    return path


def _create_preset(meta_dir: Path, slug: str) -> Path:
    """Create a preset TOML in meta_dir/presets/."""
    presets_dir = meta_dir / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    path = presets_dir / f"{slug}.toml"
    path.write_text(f'[preset]\nslug = "{slug}"\n')
    return path


class TestFindStaleEntries:
    """Phase 1: manifest -> source check."""

    def test_no_stale_when_sources_exist(self, tmp_path):
        """All sources exist -> no stale entries."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"

        _create_persona(meta_dir, "Alice")
        _create_preset(meta_dir, "test_adamw")

        entry = _make_entry(persona="Alice", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])

        stale = find_stale_entries(compiled_dir, meta_dir=meta_dir)
        assert len(stale) == 0

    def test_deleted_persona_makes_entry_stale(self, tmp_path):
        """Persona TOML missing -> entry is stale."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()

        _create_preset(meta_dir, "test_adamw")
        # No persona TOML created for "Alice"

        entry = _make_entry(persona="Alice", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])

        stale = find_stale_entries(compiled_dir, meta_dir=meta_dir)
        assert len(stale) == 1
        assert stale[0]["entry"]["persona"] == "Alice"
        assert "persona" in stale[0]["reason"].lower()

    def test_deleted_preset_makes_entry_stale(self, tmp_path):
        """Preset TOML missing -> entry is stale."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "presets").mkdir()

        _create_persona(meta_dir, "Alice")
        # No preset TOML created for "test_adamw"

        entry = _make_entry(persona="Alice", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])

        stale = find_stale_entries(compiled_dir, meta_dir=meta_dir)
        assert len(stale) == 1
        assert stale[0]["entry"]["preset"] == "test_adamw"
        assert "preset" in stale[0]["reason"].lower()

    def test_unknown_arch_makes_entry_stale(self, tmp_path):
        """Arch key not in ARCH_REGISTRY -> entry is stale."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"

        _create_persona(meta_dir, "Alice")
        _create_preset(meta_dir, "test_adamw")

        entry = _make_entry(persona="Alice", arch="nonexistent_arch_xyz", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])

        stale = find_stale_entries(compiled_dir, meta_dir=meta_dir)
        assert len(stale) == 1
        assert "arch" in stale[0]["reason"].lower()

    def test_non_dict_entry_skipped(self, tmp_path):
        """Non-dict manifest entries are skipped gracefully."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"

        _create_persona(meta_dir, "Alice")
        _create_preset(meta_dir, "test_adamw")

        # Write manifest with a malformed entry (string instead of dict)
        manifest_path = compiled_dir / "manifest.toml"
        manifest_path.write_text('schema_version = 1\ncompiler_version = "0.1.0"\nentries = ["not_a_dict"]\n')

        # Should not crash
        stale = find_stale_entries(compiled_dir, meta_dir=meta_dir)
        assert isinstance(stale, list)

    def test_multiple_reasons_all_reported(self, tmp_path):
        """Multiple stale entries with different reasons are all found."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        # Entry 1: missing persona
        e1 = _make_entry(persona="Missing", arch="qwen_image", preset="test_adamw")
        # Entry 2: missing preset
        _create_persona(meta_dir, "Bob")
        e2 = _make_entry(persona="Bob", arch="qwen_image", preset="gone_preset")
        # Entry 3: valid
        _create_persona(meta_dir, "Carol")
        _create_preset(meta_dir, "test_adamw")
        e3 = _make_entry(persona="Carol", arch="qwen_image", preset="test_adamw")

        _write_manifest(compiled_dir, [e1, e2, e3])

        stale = find_stale_entries(compiled_dir, meta_dir=meta_dir)
        assert len(stale) == 2
        stale_personas = {s["entry"]["persona"] for s in stale}
        assert "Missing" in stale_personas
        assert "Bob" in stale_personas


class TestFindOrphanedFiles:
    """Phase 2: disk -> manifest check."""

    def test_no_orphans_when_manifest_matches_disk(self, tmp_path):
        """All disk files are referenced by manifest -> no orphans."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()

        entry = _make_entry(persona="Alice", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])

        # Create the files the manifest references
        _create_compiled_file(compiled_dir, entry["training_config"])
        _create_compiled_file(compiled_dir, entry["dataset_config"])

        orphans = find_orphaned_files(compiled_dir)
        assert len(orphans) == 0

    def test_unreferenced_files_are_orphaned(self, tmp_path):
        """Files in compiled/ not referenced by manifest are orphaned."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()

        entry = _make_entry(persona="Alice", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])

        # Create the files the manifest references
        _create_compiled_file(compiled_dir, entry["training_config"])
        _create_compiled_file(compiled_dir, entry["dataset_config"])

        # Create extra unreferenced files
        _create_compiled_file(compiled_dir, "test/Alice/QWEN-IMAGE/stale_old.toml")
        _create_compiled_file(compiled_dir, "test/Alice/QWEN-IMAGE/stale_old.sh")

        orphans = find_orphaned_files(compiled_dir)
        orphan_names = {p.name for p in orphans}
        assert "stale_old.toml" in orphan_names
        assert "stale_old.sh" in orphan_names

    def test_manifest_toml_is_not_orphaned(self, tmp_path):
        """manifest.toml itself should not be reported as orphaned."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()

        _write_manifest(compiled_dir, [])

        orphans = find_orphaned_files(compiled_dir)
        assert not any(p.name == "manifest.toml" for p in orphans)

    def test_only_prunable_extensions(self, tmp_path):
        """Only .toml, .sh, .txt files are reported as orphans, not .png etc."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()

        _write_manifest(compiled_dir, [])

        _create_compiled_file(compiled_dir, "sub/orphan.toml")
        _create_compiled_file(compiled_dir, "sub/orphan.sh")
        _create_compiled_file(compiled_dir, "sub/orphan.txt")
        _create_compiled_file(compiled_dir, "sub/photo.png")

        orphans = find_orphaned_files(compiled_dir)
        orphan_names = {p.name for p in orphans}
        assert "orphan.toml" in orphan_names
        assert "orphan.sh" in orphan_names
        assert "orphan.txt" in orphan_names
        assert "photo.png" not in orphan_names


class TestPruneStaleArtifacts:
    """Integration: full prune_stale_artifacts function."""

    def test_dry_run_does_not_delete(self, tmp_path):
        """Dry-run reports stale entries and orphans but deletes nothing."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        # Entry with missing persona -> stale
        entry = _make_entry(persona="Gone", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])

        # Create files on disk (will become orphans after stale entry removal)
        _create_compiled_file(compiled_dir, entry["training_config"])
        _create_compiled_file(compiled_dir, entry["dataset_config"])
        # Extra orphan
        _create_compiled_file(compiled_dir, "test/Gone/QWEN-IMAGE/extra.toml")

        result = prune_stale_artifacts(compiled_dir, meta_dir=meta_dir, dry_run=True)
        assert result["stale_count"] >= 1
        # Files should still exist
        assert (compiled_dir / entry["training_config"]).exists()
        assert (compiled_dir / entry["dataset_config"]).exists()
        assert (compiled_dir / "test/Gone/QWEN-IMAGE/extra.toml").exists()
        # Manifest should still have the entry
        with open(compiled_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert len(manifest["entries"]) == 1

    def test_execute_removes_stale_entries_and_orphans(self, tmp_path):
        """--execute removes stale manifest entries and orphaned files."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"

        # Create a valid persona but no preset -> entry is stale
        _create_persona(meta_dir, "Alice")
        (meta_dir / "presets").mkdir(parents=True, exist_ok=True)
        # No preset file -> stale

        entry = _make_entry(persona="Alice", arch="qwen_image", preset="gone_preset")
        _write_manifest(compiled_dir, [entry])

        # Create files on disk
        _create_compiled_file(compiled_dir, entry["training_config"])
        _create_compiled_file(compiled_dir, entry["dataset_config"])

        result = prune_stale_artifacts(compiled_dir, meta_dir=meta_dir, dry_run=False)
        assert result["stale_count"] == 1
        assert result["orphan_count"] >= 2  # training + dataset files

        # Manifest entry should be gone
        with open(compiled_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert len(manifest["entries"]) == 0

        # Files should be deleted
        assert not (compiled_dir / entry["training_config"]).exists()
        assert not (compiled_dir / entry["dataset_config"]).exists()

    def test_machine_scoping(self, tmp_path):
        """--machine only prunes entries for the specified machine."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        # Two entries: machine "alpha" (stale) and machine "beta" (stale)
        e_alpha = _make_entry(persona="Gone1", arch="qwen_image", preset="test_adamw", machine="alpha")
        e_beta = _make_entry(persona="Gone2", arch="qwen_image", preset="test_adamw", machine="beta")

        _write_manifest(compiled_dir, [e_alpha, e_beta])
        _create_compiled_file(compiled_dir, e_alpha["training_config"])
        _create_compiled_file(compiled_dir, e_alpha["dataset_config"])
        _create_compiled_file(compiled_dir, e_beta["training_config"])
        _create_compiled_file(compiled_dir, e_beta["dataset_config"])

        # Only prune machine "alpha"
        result = prune_stale_artifacts(compiled_dir, meta_dir=meta_dir, dry_run=False, machine="alpha")
        assert result["stale_count"] == 1

        # "beta" entry should survive
        with open(compiled_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert len(manifest["entries"]) == 1
        assert manifest["entries"][0]["machine"] == "beta"

    def test_persona_scoping(self, tmp_path):
        """--persona only prunes entries for the specified persona."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"

        _create_persona(meta_dir, "Valid")
        (meta_dir / "presets").mkdir(parents=True, exist_ok=True)
        # No preset -> both entries stale

        e1 = _make_entry(persona="Valid", arch="qwen_image", preset="gone1")
        e2 = _make_entry(persona="Valid", arch="wan22_t2v", preset="gone2")

        _write_manifest(compiled_dir, [e1, e2])
        _create_compiled_file(compiled_dir, e1["training_config"])
        _create_compiled_file(compiled_dir, e1["dataset_config"])
        _create_compiled_file(compiled_dir, e2["training_config"])
        _create_compiled_file(compiled_dir, e2["dataset_config"])

        # Scope to just persona "Valid" -- both entries match
        result = prune_stale_artifacts(compiled_dir, meta_dir=meta_dir, dry_run=False, persona="Valid")
        assert result["stale_count"] == 2

    def test_stripped_txt_sidecars_cleaned(self, tmp_path):
        """_stripped.txt sidecars are cleaned alongside their TOML files."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        entry = _make_entry(persona="Gone", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])

        # Create training toml + its stripped sidecar
        training_path = _create_compiled_file(compiled_dir, entry["training_config"])
        stripped_path = training_path.with_name(training_path.stem + "_stripped.txt")
        stripped_path.write_text("# stripped args")
        _create_compiled_file(compiled_dir, entry["dataset_config"])

        result = prune_stale_artifacts(compiled_dir, meta_dir=meta_dir, dry_run=False)
        assert result["stale_count"] == 1
        # stripped.txt should be deleted too
        assert not stripped_path.exists()

    def test_symlink_outside_compiled_refused(self, tmp_path):
        """Symlinks pointing outside compiled tree are refused."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        _write_manifest(compiled_dir, [])

        # Create a symlink pointing outside compiled dir
        outside_file = tmp_path / "outside_secret.toml"
        outside_file.write_text("secret data")
        sub = compiled_dir / "sub"
        sub.mkdir()
        symlink = sub / "evil_link.toml"
        symlink.symlink_to(outside_file)

        orphans = find_orphaned_files(compiled_dir)
        # The symlink should NOT appear in orphans because it resolves outside
        assert not any(p.name == "evil_link.toml" for p in orphans)
        # The outside file should still exist
        assert outside_file.exists()

    def test_png_file_survives(self, tmp_path):
        """Non-prunable file types like .png survive pruning."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        _write_manifest(compiled_dir, [])

        png_file = _create_compiled_file(compiled_dir, "sub/photo.png")

        result = prune_stale_artifacts(compiled_dir, meta_dir=meta_dir, dry_run=False)
        assert png_file.exists()
        assert result["orphan_count"] == 0


class TestMainPrune:
    """CLI entry point tests."""

    def test_dry_run_is_default(self, tmp_path):
        """No --execute means dry-run (no changes)."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        entry = _make_entry(persona="Missing", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])
        _create_compiled_file(compiled_dir, entry["training_config"])
        _create_compiled_file(compiled_dir, entry["dataset_config"])

        exit_code = main_prune(
            [
                "--compiled-dir",
                str(compiled_dir),
                "--meta-dir",
                str(meta_dir),
            ]
        )
        assert exit_code == 0

        # Files should still exist (dry-run)
        assert (compiled_dir / entry["training_config"]).exists()
        with open(compiled_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert len(manifest["entries"]) == 1

    def test_execute_flag(self, tmp_path):
        """--execute actually removes stale entries and orphans."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        entry = _make_entry(persona="Missing", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])
        _create_compiled_file(compiled_dir, entry["training_config"])
        _create_compiled_file(compiled_dir, entry["dataset_config"])

        exit_code = main_prune(
            [
                "--compiled-dir",
                str(compiled_dir),
                "--meta-dir",
                str(meta_dir),
                "--execute",
            ]
        )
        assert exit_code == 0

        # Files should be deleted
        assert not (compiled_dir / entry["training_config"]).exists()
        with open(compiled_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert len(manifest["entries"]) == 0

    def test_nothing_to_prune(self, tmp_path):
        """When nothing is stale or orphaned, exit 0 gracefully."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"

        _create_persona(meta_dir, "Alice")
        _create_preset(meta_dir, "test_adamw")

        entry = _make_entry(persona="Alice", arch="qwen_image", preset="test_adamw")
        _write_manifest(compiled_dir, [entry])
        _create_compiled_file(compiled_dir, entry["training_config"])
        _create_compiled_file(compiled_dir, entry["dataset_config"])

        exit_code = main_prune(
            [
                "--compiled-dir",
                str(compiled_dir),
                "--meta-dir",
                str(meta_dir),
            ]
        )
        assert exit_code == 0

    def test_machine_filter(self, tmp_path):
        """--machine scopes to specific machine."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        e1 = _make_entry(persona="Gone", arch="qwen_image", preset="test_adamw", machine="keep")
        e2 = _make_entry(persona="Gone", arch="qwen_image", preset="test_adamw", machine="prune")

        _write_manifest(compiled_dir, [e1, e2])
        _create_compiled_file(compiled_dir, e1["training_config"])
        _create_compiled_file(compiled_dir, e1["dataset_config"])
        _create_compiled_file(compiled_dir, e2["training_config"])
        _create_compiled_file(compiled_dir, e2["dataset_config"])

        exit_code = main_prune(
            [
                "--compiled-dir",
                str(compiled_dir),
                "--meta-dir",
                str(meta_dir),
                "--execute",
                "--machine",
                "prune",
            ]
        )
        assert exit_code == 0

        with open(compiled_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert len(manifest["entries"]) == 1
        assert manifest["entries"][0]["machine"] == "keep"

    def test_persona_filter(self, tmp_path):
        """--persona scopes to specific persona."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        e1 = _make_entry(persona="KeepMe", arch="qwen_image", preset="test_adamw")
        e2 = _make_entry(persona="PruneMe", arch="qwen_image", preset="test_adamw")

        _write_manifest(compiled_dir, [e1, e2])
        _create_compiled_file(compiled_dir, e1["training_config"])
        _create_compiled_file(compiled_dir, e1["dataset_config"])
        _create_compiled_file(compiled_dir, e2["training_config"])
        _create_compiled_file(compiled_dir, e2["dataset_config"])

        exit_code = main_prune(
            [
                "--compiled-dir",
                str(compiled_dir),
                "--meta-dir",
                str(meta_dir),
                "--execute",
                "--persona",
                "PruneMe",
            ]
        )
        assert exit_code == 0

        with open(compiled_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert len(manifest["entries"]) == 1
        assert manifest["entries"][0]["persona"] == "KeepMe"

    def test_env_sh_orphan_cleaned(self, tmp_path):
        """env.sh files that are orphaned are also cleaned."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "personas").mkdir()
        (meta_dir / "presets").mkdir()

        _write_manifest(compiled_dir, [])

        env_sh = _create_compiled_file(compiled_dir, "test/Gone/QWEN-IMAGE/env.sh")

        result = prune_stale_artifacts(compiled_dir, meta_dir=meta_dir, dry_run=False)
        assert not env_sh.exists()
        assert result["orphan_count"] >= 1

    def test_lock_file_and_manifest_lock_not_orphaned(self, tmp_path):
        """Internal files like .manifest.lock should not be reported as orphans."""
        compiled_dir = tmp_path / "compiled"
        compiled_dir.mkdir()

        _write_manifest(compiled_dir, [])

        # Create a .manifest.lock file (created by concurrent access)
        lock_file = compiled_dir / ".manifest.lock"
        lock_file.write_text("")

        orphans = find_orphaned_files(compiled_dir)
        assert not any(p.name == ".manifest.lock" for p in orphans)
