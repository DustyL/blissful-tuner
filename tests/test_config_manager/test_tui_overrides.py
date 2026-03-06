"""Tests for TUI ephemeral override helpers and compiler override_data parameter."""

from __future__ import annotations

import time
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from blissful_tuner.config_manager.tui.app import (
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


class TestNonPresetSectionRejection:
    """Non-preset sections (model, output, advanced) cannot be overridden."""

    def test_override_data_model_section_filtered_in_compile(self):
        """Model section in override_data is filtered out during compile_config merge."""
        from blissful_tuner.config_manager.compiler import compile_config

        fixtures = Path(__file__).parent / "fixtures"
        # Compile with a model override — it should be filtered out
        base = compile_config(
            machine_path=fixtures / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=fixtures / "personas" / "TESTPERSONA.toml",
            preset_path=fixtures / "presets" / "test_adamw.toml",
        )
        with_override = compile_config(
            machine_path=fixtures / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=fixtures / "personas" / "TESTPERSONA.toml",
            preset_path=fixtures / "presets" / "test_adamw.toml",
            override_data={"model": {"dit": "/overridden/path"}},
        )
        # Model section should be identical — override_data model keys are filtered
        assert base["training_toml"]["model"] == with_override["training_toml"]["model"]

    def test_override_data_output_section_filtered(self):
        """Output section in override_data is filtered out during compile_config merge."""
        from blissful_tuner.config_manager.compiler import compile_config

        fixtures = Path(__file__).parent / "fixtures"
        base = compile_config(
            machine_path=fixtures / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=fixtures / "personas" / "TESTPERSONA.toml",
            preset_path=fixtures / "presets" / "test_adamw.toml",
        )
        with_override = compile_config(
            machine_path=fixtures / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=fixtures / "personas" / "TESTPERSONA.toml",
            preset_path=fixtures / "presets" / "test_adamw.toml",
            override_data={"output": {"output_dir": "/overridden"}},
        )
        assert base["training_toml"]["output"] == with_override["training_toml"]["output"]

    def test_preset_section_override_still_works(self):
        """Preset sections (training, network, etc.) still apply correctly."""
        from blissful_tuner.config_manager.compiler import compile_config

        fixtures = Path(__file__).parent / "fixtures"
        result = compile_config(
            machine_path=fixtures / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=fixtures / "personas" / "TESTPERSONA.toml",
            preset_path=fixtures / "presets" / "test_adamw.toml",
            override_data={"training": {"seed": 12345}},
        )
        assert result["training_toml"]["training"]["seed"] == 12345

    def test_build_override_data_filters_non_preset(self):
        """_build_override_data only includes preset-layer sections."""
        ephemeral: dict[str, dict[str, Any]] = {
            "training": {"seed": 99},
            "model": {"dit": "/bad"},
            "output": {"output_dir": "/bad"},
        }
        result = _build_override_data(ephemeral)
        assert "training" in result
        assert "model" not in result
        assert "output" not in result


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


class TestComputeOverrideHash:
    """Test _compute_override_hash() covers both override_sets and override_data."""

    def test_empty_returns_empty(self):
        from blissful_tuner.config_manager.compiler import _compute_override_hash

        assert _compute_override_hash() == ""

    def test_override_sets_only(self):
        from blissful_tuner.config_manager.compiler import _compute_override_hash

        h = _compute_override_hash(override_sets=["training.seed=42"])
        assert len(h) == 16
        assert h != ""

    def test_override_data_only(self):
        from blissful_tuner.config_manager.compiler import _compute_override_hash

        h = _compute_override_hash(override_data={"training": {"seed": 42}})
        assert len(h) == 16
        assert h != ""

    def test_same_content_same_hash(self):
        from blissful_tuner.config_manager.compiler import _compute_override_hash

        h1 = _compute_override_hash(override_sets=["training.seed=42"])
        h2 = _compute_override_hash(override_data={"training": {"seed": 42}})
        assert h1 == h2

    def test_different_content_different_hash(self):
        from blissful_tuner.config_manager.compiler import _compute_override_hash

        h1 = _compute_override_hash(override_data={"training": {"seed": 42}})
        h2 = _compute_override_hash(override_data={"training": {"seed": 99}})
        assert h1 != h2


class TestParseOverrideSets:
    """Test _parse_override_sets() in compiler.py for --set KEY=VALUE wiring."""

    def test_dot_notation_parses_to_section(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        result = _parse_override_sets(["training.learning_rate=1e-4"])
        assert result == {"training": {"learning_rate": 1e-4}}

    def test_no_dot_defaults_to_training(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        result = _parse_override_sets(["seed=99"])
        assert result == {"training": {"seed": 99}}

    def test_multiple_overrides_across_sections(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        result = _parse_override_sets(
            [
                "training.max_train_steps=2000",
                "optimizer.learning_rate=1e-4",
                "network.network_dim=32",
            ]
        )
        assert result == {
            "training": {"max_train_steps": 2000},
            "optimizer": {"learning_rate": 1e-4},
            "network": {"network_dim": 32},
        }

    def test_value_with_equals_sign(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        result = _parse_override_sets(["training.some_arg='key=value'"])
        assert result == {"training": {"some_arg": "key=value"}}

    def test_invalid_format_raises(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        with pytest.raises(ValueError, match="Invalid --set format"):
            _parse_override_sets(["no_equals_here"])

    def test_boolean_value(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        result = _parse_override_sets(["training.use_mask_loss=true"])
        assert result == {"training": {"use_mask_loss": True}}

    def test_array_value(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        result = _parse_override_sets(['network.network_args=["loraplus_lr_ratio=8"]'])
        assert result == {"network": {"network_args": ["loraplus_lr_ratio=8"]}}

    def test_non_overridable_section_rejected(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        with pytest.raises(ValueError, match="Cannot override section 'output'"):
            _parse_override_sets(['output.output_dir="/tmp/custom"'])

    def test_model_section_rejected(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        with pytest.raises(ValueError, match="Cannot override section 'model'"):
            _parse_override_sets(['model.dit="/some/path"'])

    def test_sampling_section_allowed(self):
        from blissful_tuner.config_manager.compiler import _parse_override_sets

        result = _parse_override_sets(["sampling.cfg_scale=7.0"])
        assert result == {"sampling": {"cfg_scale": 7.0}}
