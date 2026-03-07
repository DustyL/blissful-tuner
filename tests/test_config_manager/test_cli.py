"""Tests for bt-compile CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from blissful_tuner.config_manager.cli import (
    _list_personas,
    _list_presets,
    _validate_set_format,
    main_compile,
    run_compile,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# --list-archs
# ---------------------------------------------------------------------------


class TestListArchs:
    """--list-archs prints registry keys."""

    def test_list_archs_returns_zero(self):
        assert main_compile(["--list-archs"]) == 0

    def test_list_archs_prints_registry_keys(self, capsys):
        main_compile(["--list-archs"])
        out = capsys.readouterr().out
        assert "wan22_t2v" in out
        assert "qwen_image" in out
        assert "flux2_klein9b" in out

    def test_list_archs_prints_display_names(self, capsys):
        main_compile(["--list-archs"])
        out = capsys.readouterr().out
        assert "WAN 2.2 T2V" in out
        assert "Qwen-Image" in out


# ---------------------------------------------------------------------------
# --list-personas / --list-presets (using a custom meta dir)
# ---------------------------------------------------------------------------


class TestListPersonas:
    """--list-personas lists TOML stems from personas directory."""

    def test_list_personas_returns_zero(self):
        with patch("blissful_tuner.config_manager.cli.find_meta_dir", return_value=FIXTURES):
            assert main_compile(["--list-personas"]) == 0

    def test_list_personas_prints_names(self, capsys):
        with patch("blissful_tuner.config_manager.cli.find_meta_dir", return_value=FIXTURES):
            main_compile(["--list-personas"])
        out = capsys.readouterr().out
        assert "TESTPERSONA" in out

    def test_list_personas_missing_dir(self, capsys, tmp_path):
        """Prints error when personas dir doesn't exist."""
        with patch("blissful_tuner.config_manager.cli.find_meta_dir", return_value=tmp_path):
            rc = main_compile(["--list-personas"])
        assert rc == 0  # Still returns 0, just prints nothing


class TestListPresets:
    """--list-presets lists TOML stems from presets directory."""

    def test_list_presets_returns_zero(self):
        with patch("blissful_tuner.config_manager.cli.find_meta_dir", return_value=FIXTURES):
            assert main_compile(["--list-presets"]) == 0

    def test_list_presets_prints_names(self, capsys):
        with patch("blissful_tuner.config_manager.cli.find_meta_dir", return_value=FIXTURES):
            main_compile(["--list-presets"])
        out = capsys.readouterr().out
        assert "test_adamw" in out


# ---------------------------------------------------------------------------
# --set KEY=VALUE parsing
# ---------------------------------------------------------------------------


class TestSetOverrideParsing:
    """--set key=value parsing into a dict."""

    def test_parse_single_override(self):
        result = _validate_set_format(["learning_rate=1e-4"])
        assert result == {"learning_rate": "1e-4"}

    def test_parse_multiple_overrides(self):
        result = _validate_set_format(["learning_rate=1e-4", "seed=123"])
        assert result == {"learning_rate": "1e-4", "seed": "123"}

    def test_parse_dotted_key(self):
        result = _validate_set_format(["training.seed=42"])
        assert result == {"training.seed": "42"}

    def test_parse_value_with_equals(self):
        """Value containing '=' should work (split on first '=' only)."""
        result = _validate_set_format(["optimizer_args=betas=(0.9, 0.99)"])
        assert result == {"optimizer_args": "betas=(0.9, 0.99)"}

    def test_parse_empty_list(self):
        result = _validate_set_format([])
        assert result == {}

    def test_parse_none(self):
        result = _validate_set_format(None)
        assert result == {}

    def test_parse_missing_equals_raises(self):
        with pytest.raises(ValueError, match="Invalid --set format"):
            _validate_set_format(["no_equals_here"])


# ---------------------------------------------------------------------------
# run_compile() -- actual compilation
# ---------------------------------------------------------------------------


class TestRunCompile:
    """run_compile produces output files from fixture inputs."""

    def test_compile_produces_output(self, tmp_path):
        result = run_compile(
            persona="TESTPERSONA",
            arch="qwen_image",
            preset="test_adamw",
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=tmp_path,
        )
        assert Path(result["training_toml_path"]).exists()
        assert Path(result["dataset_toml_path"]).exists()

    def test_compile_returns_provenance(self, tmp_path):
        result = run_compile(
            persona="TESTPERSONA",
            arch="qwen_image",
            preset="test_adamw",
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=tmp_path,
        )
        assert result["provenance"]["arch"] == "qwen_image"

    def test_compile_with_coerce(self, tmp_path):
        """coerce=True does not raise on unsupported features (pass-through to validator)."""
        result = run_compile(
            persona="TESTPERSONA",
            arch="qwen_image",
            preset="test_adamw",
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=tmp_path,
            coerce=True,
        )
        assert Path(result["training_toml_path"]).exists()


# ---------------------------------------------------------------------------
# Error handling in main_compile
# ---------------------------------------------------------------------------


class TestMainCompileErrors:
    """Error handling in main_compile."""

    def test_no_args_returns_nonzero(self):
        """No arguments should show usage and return 1."""
        rc = main_compile([])
        assert rc == 1

    def test_missing_arch_returns_nonzero(self):
        """PERSONA without ARCH and PRESET should fail."""
        rc = main_compile(["TESTPERSONA"])
        assert rc == 1

    def test_missing_preset_returns_nonzero(self):
        """PERSONA and ARCH without PRESET should fail."""
        rc = main_compile(["TESTPERSONA", "qwen_image"])
        assert rc == 1

    def test_all_without_arch_preset_returns_nonzero(self):
        """--all without ARCH and PRESET should fail."""
        rc = main_compile(["--all"])
        assert rc == 1


# ---------------------------------------------------------------------------
# main_compile with full compile (integration-style)
# ---------------------------------------------------------------------------


class TestMainCompileIntegration:
    """Integration tests calling main_compile with real fixture paths."""

    def test_full_compile_with_explicit_paths(self, tmp_path):
        """Compile with --machine-path, --persona-path, --preset-path overrides."""
        rc = main_compile(
            [
                "TESTPERSONA",
                "qwen_image",
                "test_adamw",
                "--machine-path",
                str(FIXTURES / "machines" / "test_machine.toml"),
                "--persona-path",
                str(FIXTURES / "personas" / "TESTPERSONA.toml"),
                "--preset-path",
                str(FIXTURES / "presets" / "test_adamw.toml"),
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        # Check output files were created
        output_files = list(tmp_path.rglob("*.toml"))
        assert len(output_files) >= 2  # training + dataset

    def test_full_compile_with_set_override(self, tmp_path):
        """--set key=value should not cause error."""
        rc = main_compile(
            [
                "TESTPERSONA",
                "qwen_image",
                "test_adamw",
                "--machine-path",
                str(FIXTURES / "machines" / "test_machine.toml"),
                "--persona-path",
                str(FIXTURES / "personas" / "TESTPERSONA.toml"),
                "--preset-path",
                str(FIXTURES / "presets" / "test_adamw.toml"),
                "--output-dir",
                str(tmp_path),
                "--set",
                "seed=99",
            ]
        )
        assert rc == 0

    def test_full_compile_with_coerce_flag(self, tmp_path):
        """--coerce flag passes through to compile."""
        rc = main_compile(
            [
                "TESTPERSONA",
                "qwen_image",
                "test_adamw",
                "--machine-path",
                str(FIXTURES / "machines" / "test_machine.toml"),
                "--persona-path",
                str(FIXTURES / "personas" / "TESTPERSONA.toml"),
                "--preset-path",
                str(FIXTURES / "presets" / "test_adamw.toml"),
                "--output-dir",
                str(tmp_path),
                "--coerce",
            ]
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# --all flag
# ---------------------------------------------------------------------------


class TestAllFlag:
    """--all compiles for all personas in the personas directory."""

    def test_all_flag_compiles_multiple_personas(self, tmp_path):
        """--all should iterate all .toml files in personas dir."""
        with patch("blissful_tuner.config_manager.cli.find_meta_dir", return_value=FIXTURES):
            rc = main_compile(
                [
                    "--all",
                    "qwen_image",
                    "test_adamw",
                    "--machine-path",
                    str(FIXTURES / "machines" / "test_machine.toml"),
                    "--preset-path",
                    str(FIXTURES / "presets" / "test_adamw.toml"),
                    "--output-dir",
                    str(tmp_path),
                ]
            )
        assert rc == 0
        # Should have compiled at least one persona (TESTPERSONA)
        output_files = list(tmp_path.rglob("*.toml"))
        assert len(output_files) >= 2


# ---------------------------------------------------------------------------
# find_meta_dir and listing helpers
# ---------------------------------------------------------------------------


class TestListHelpers:
    """Unit tests for _list_personas and _list_presets."""

    def test_list_personas_from_fixtures(self):
        names = _list_personas(FIXTURES)
        assert "TESTPERSONA" in names

    def test_list_presets_from_fixtures(self):
        slugs = _list_presets(FIXTURES)
        assert "test_adamw" in slugs
        assert "test_adamw_with_extra" in slugs

    def test_list_personas_empty_dir(self, tmp_path):
        names = _list_personas(tmp_path)
        assert names == []

    def test_list_presets_empty_dir(self, tmp_path):
        slugs = _list_presets(tmp_path)
        assert slugs == []
