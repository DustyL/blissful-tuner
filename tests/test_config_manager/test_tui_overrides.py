"""Tests for TUI ephemeral override helpers and compiler override_data parameter."""

from __future__ import annotations

import time
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from blissful_tuner.config_manager.tui.app import (
    _apply_post_compile_overrides,
    _build_override_data,
    _cleanup_old_drafts,
    _list_draft_files,
    _load_draft,
    _write_override_toml,
)


# ---------------------------------------------------------------------------
# _build_override_data tests
# ---------------------------------------------------------------------------


class TestBuildOverrideData:
    """Test _build_override_data separates preset-layer sections from others."""

    def test_empty_overrides(self):
        assert _build_override_data({}) == {}

    def test_preset_sections_included(self):
        ephemeral = {
            "training": {"max_train_steps": 2000},
            "optimizer": {"learning_rate": 1e-4},
            "network": {"network_dim": 32},
        }
        result = _build_override_data(ephemeral)
        assert "training" in result
        assert "optimizer" in result
        assert "network" in result
        assert result["training"]["max_train_steps"] == 2000

    def test_non_preset_sections_excluded(self):
        ephemeral = {
            "model": {"dit": "/new/path"},
            "output": {"output_dir": "/tmp/new"},
            "advanced": {"compile": True},
        }
        result = _build_override_data(ephemeral)
        assert result == {}

    def test_mixed_sections(self):
        ephemeral = {
            "training": {"seed": 123},
            "model": {"dit": "/path"},
        }
        result = _build_override_data(ephemeral)
        assert "training" in result
        assert "model" not in result

    def test_empty_section_values_excluded(self):
        ephemeral = {
            "training": {},
            "optimizer": {"learning_rate": 5e-5},
        }
        result = _build_override_data(ephemeral)
        assert "training" not in result
        assert "optimizer" in result

    def test_sampling_section_included(self):
        """Sampling is a preset-layer section."""
        ephemeral = {"sampling": {"prompt": "test"}}
        result = _build_override_data(ephemeral)
        assert "sampling" in result


class TestApplyPostCompileOverrides:
    """Test _apply_post_compile_overrides for non-preset sections."""

    def test_model_override_applied(self):
        training_toml: dict[str, Any] = {"model": {"dit": "/old/path"}}
        ephemeral = {"model": {"dit": "/new/path"}}
        result = _apply_post_compile_overrides(training_toml, ephemeral)
        assert result["model"]["dit"] == "/new/path"

    def test_output_override_applied(self):
        training_toml: dict[str, Any] = {"output": {"output_dir": "/old"}}
        ephemeral = {"output": {"output_dir": "/new"}}
        result = _apply_post_compile_overrides(training_toml, ephemeral)
        assert result["output"]["output_dir"] == "/new"

    def test_preset_sections_not_applied(self):
        """Preset sections should NOT be applied by post-compile (they go via override_data)."""
        training_toml: dict[str, Any] = {"training": {"seed": 42}}
        ephemeral = {"training": {"seed": 99}}
        result = _apply_post_compile_overrides(training_toml, ephemeral)
        # training is a preset section, so it should not be touched by post-compile
        assert result["training"]["seed"] == 42

    def test_new_section_created(self):
        training_toml: dict[str, Any] = {}
        ephemeral = {"advanced": {"compile": True}}
        result = _apply_post_compile_overrides(training_toml, ephemeral)
        assert result["advanced"]["compile"] is True

    def test_modifies_in_place(self):
        training_toml: dict[str, Any] = {"model": {"dit": "/old"}}
        ephemeral = {"model": {"dit": "/new"}}
        _apply_post_compile_overrides(training_toml, ephemeral)
        assert training_toml["model"]["dit"] == "/new"


# ---------------------------------------------------------------------------
# Override TOML write/read round-trip
# ---------------------------------------------------------------------------


class TestWriteOverrideToml:
    """Test _write_override_toml writes valid TOML."""

    def test_round_trip(self, tmp_path: Path):
        overrides = {
            "training": {"max_train_steps": 2000, "seed": 123},
            "optimizer": {"learning_rate": 5e-5},
        }
        path = tmp_path / "test_override.toml"
        _write_override_toml(path, overrides)

        with open(path, "rb") as f:
            loaded = tomllib.load(f)

        assert loaded["training"]["max_train_steps"] == 2000
        assert loaded["training"]["seed"] == 123
        assert loaded["optimizer"]["learning_rate"] == 5e-5

    def test_empty_sections_skipped(self, tmp_path: Path):
        overrides = {
            "training": {"seed": 42},
            "optimizer": {},
        }
        path = tmp_path / "test.toml"
        _write_override_toml(path, overrides)

        with open(path, "rb") as f:
            loaded = tomllib.load(f)

        assert "training" in loaded
        assert "optimizer" not in loaded

    def test_bool_and_list_values(self, tmp_path: Path):
        overrides = {
            "training": {"flag": True, "items": [1, 2, 3]},
        }
        path = tmp_path / "test.toml"
        _write_override_toml(path, overrides)

        with open(path, "rb") as f:
            loaded = tomllib.load(f)

        assert loaded["training"]["flag"] is True
        assert loaded["training"]["items"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Draft file management
# ---------------------------------------------------------------------------


class TestDraftFileManagement:
    """Test draft listing, cleanup, and loading."""

    def test_list_draft_files(self, tmp_path: Path):
        overrides_dir = tmp_path / "overrides"
        overrides_dir.mkdir()
        (overrides_dir / ".draft_20260101_120000.toml").write_text("[training]\nseed = 42\n")
        (overrides_dir / ".draft_20260102_120000.toml").write_text("[training]\nseed = 99\n")
        (overrides_dir / "not_a_draft.toml").write_text("[training]\nseed = 1\n")

        # Patch _overrides_dir to return our test dir
        with patch("blissful_tuner.config_manager.tui.app._overrides_dir", return_value=overrides_dir):
            drafts = _list_draft_files(tmp_path)

        assert len(drafts) == 2
        # Most recent first (by mtime)
        assert all(d.name.startswith(".draft_") for d in drafts)

    def test_cleanup_old_drafts(self, tmp_path: Path):
        overrides_dir = tmp_path / "overrides"
        overrides_dir.mkdir()
        old_draft = overrides_dir / ".draft_old.toml"
        old_draft.write_text("[training]\nseed = 42\n")
        # Make it appear old by setting mtime to 30 days ago
        import os

        old_time = time.time() - (30 * 86400)
        os.utime(old_draft, (old_time, old_time))

        new_draft = overrides_dir / ".draft_new.toml"
        new_draft.write_text("[training]\nseed = 99\n")

        with patch("blissful_tuner.config_manager.tui.app._overrides_dir", return_value=overrides_dir):
            removed = _cleanup_old_drafts(tmp_path)

        assert removed == 1
        assert not old_draft.exists()
        assert new_draft.exists()

    def test_load_draft(self, tmp_path: Path):
        draft_path = tmp_path / ".draft_test.toml"
        draft_path.write_text("[training]\nmax_train_steps = 2000\nseed = 42\n\n[optimizer]\nlearning_rate = 5e-5\n")

        result = _load_draft(draft_path)
        assert result["training"]["max_train_steps"] == 2000
        assert result["training"]["seed"] == 42
        assert result["optimizer"]["learning_rate"] == 5e-5

    def test_load_draft_ignores_non_dict(self, tmp_path: Path):
        """Top-level non-dict values are filtered out."""
        draft_path = tmp_path / ".draft_test.toml"
        draft_path.write_text("version = 1\n\n[training]\nseed = 42\n")

        result = _load_draft(draft_path)
        assert "version" not in result
        assert "training" in result


# ---------------------------------------------------------------------------
# compile_config with override_data
# ---------------------------------------------------------------------------


class TestCompileConfigOverrideData:
    """Test compile_config() with the new override_data parameter."""

    @pytest.fixture()
    def meta_dir(self, tmp_path: Path) -> Path:
        """Create a minimal meta directory for testing."""
        meta = tmp_path / "configs" / "meta"

        machines = meta / "machines"
        machines.mkdir(parents=True)
        (machines / "test_machine.toml").write_text(
            '[machine]\nname = "test_machine"\n\n[machine.paths]\nmodels_dir = "/models"\n'
            'output_dir = "/output"\n\n[machine.hardware]\n\n[machine.env]\n'
        )

        personas = meta / "personas"
        personas.mkdir(parents=True)
        (personas / "test_persona.toml").write_text(
            '[persona]\nname = "TestPersona"\n\n[dataset]\nimage_directory = "/images"\nresolutions = [[1024, 1024]]\n'
        )

        presets = meta / "presets"
        presets.mkdir(parents=True)
        (presets / "test_preset.toml").write_text(
            '[preset]\nname = "Test Preset"\nslug = "test_preset"\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 5e-5\n\n'
            "[network]\nnetwork_dim = 64\nnetwork_alpha = 32\n\n"
            "[training]\nmax_train_steps = 4000\nseed = 42\n"
        )

        return meta

    def test_override_data_merges_into_preset(self, meta_dir: Path):
        from blissful_tuner.config_manager.compiler import compile_config

        result = compile_config(
            machine_path=meta_dir / "machines" / "test_machine.toml",
            arch_key="wan22_t2v",
            persona_path=meta_dir / "personas" / "test_persona.toml",
            preset_path=meta_dir / "presets" / "test_preset.toml",
            override_data={"training": {"max_train_steps": 2000}},
        )

        training = result["training_toml"]
        assert training["training"]["max_train_steps"] == 2000

    def test_override_data_without_override_path(self, meta_dir: Path):
        from blissful_tuner.config_manager.compiler import compile_config

        result = compile_config(
            machine_path=meta_dir / "machines" / "test_machine.toml",
            arch_key="wan22_t2v",
            persona_path=meta_dir / "personas" / "test_persona.toml",
            preset_path=meta_dir / "presets" / "test_preset.toml",
            override_data={"optimizer": {"learning_rate": 1e-4}},
        )

        training = result["training_toml"]
        assert training["optimizer"]["learning_rate"] == 1e-4

    def test_provenance_includes_ephemeral_count(self, meta_dir: Path):
        from blissful_tuner.config_manager.compiler import compile_config

        result = compile_config(
            machine_path=meta_dir / "machines" / "test_machine.toml",
            arch_key="wan22_t2v",
            persona_path=meta_dir / "personas" / "test_persona.toml",
            preset_path=meta_dir / "presets" / "test_preset.toml",
            override_data={"training": {"seed": 99, "max_train_steps": 1000}},
        )

        prov = result["provenance"]
        assert "ephemeral_overrides" in prov
        assert prov["ephemeral_overrides"] == 2

    def test_no_override_data_no_provenance_key(self, meta_dir: Path):
        from blissful_tuner.config_manager.compiler import compile_config

        result = compile_config(
            machine_path=meta_dir / "machines" / "test_machine.toml",
            arch_key="wan22_t2v",
            persona_path=meta_dir / "personas" / "test_persona.toml",
            preset_path=meta_dir / "presets" / "test_preset.toml",
        )

        prov = result["provenance"]
        assert "ephemeral_overrides" not in prov
