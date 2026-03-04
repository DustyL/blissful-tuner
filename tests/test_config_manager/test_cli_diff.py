"""Tests for bt-diff CLI."""

from __future__ import annotations

from pathlib import Path

import tomlkit

from blissful_tuner.config_manager.cli import main_diff


class TestBtDiffCLI:
    def _write_toml(self, path: Path, data: dict) -> None:
        """Helper to write a TOML file."""
        doc = tomlkit.document()
        for k, v in data.items():
            doc.add(k, v)
        path.write_text(tomlkit.dumps(doc))

    def test_identical_files_exit_0(self, tmp_path):
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        self._write_toml(a, {"model": {"dit": "/path/dit"}, "training": {"lr": 5e-5}})
        self._write_toml(b, {"model": {"dit": "/path/dit"}, "training": {"lr": 5e-5}})
        result = main_diff(["--quiet", str(a), str(b)])
        assert result == 0

    def test_different_files_exit_1(self, tmp_path):
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        self._write_toml(a, {"model": {"dit": "/path/a"}})
        self._write_toml(b, {"model": {"dit": "/path/b"}})
        result = main_diff(["--quiet", str(a), str(b)])
        assert result == 1

    def test_flatten_mode_ignores_section_layout(self, tmp_path):
        """Two files with same keys in different sections -> no diff in flatten mode."""
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        # Same key 'lr' in different sections
        self._write_toml(a, {"optimizer": {"lr": 5e-5}, "training": {"seed": 42}})
        self._write_toml(b, {"training": {"lr": 5e-5, "seed": 42}})
        result = main_diff(["--flatten", "--quiet", str(a), str(b)])
        assert result == 0

    def test_flatten_mode_detects_value_diff(self, tmp_path):
        """Flatten mode still catches value differences."""
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        self._write_toml(a, {"training": {"lr": 5e-5}})
        self._write_toml(b, {"training": {"lr": 2e-5}})
        result = main_diff(["--flatten", "--quiet", str(a), str(b)])
        assert result == 1

    def test_output_format(self, tmp_path, capsys):
        """Diff output shows changes in readable format."""
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        self._write_toml(a, {"model": {"dit": "/path/a"}})
        self._write_toml(b, {"model": {"dit": "/path/b"}})
        main_diff([str(a), str(b)])
        captured = capsys.readouterr()
        assert "dit" in captured.out
        assert "difference" in captured.out.lower()

    def test_no_diff_message(self, tmp_path, capsys):
        """Identical files show 'No differences' message."""
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        self._write_toml(a, {"training": {"lr": 5e-5}})
        self._write_toml(b, {"training": {"lr": 5e-5}})
        main_diff([str(a), str(b)])
        captured = capsys.readouterr()
        assert "no differences" in captured.out.lower()

    def test_flatten_top_level_scalars_preserved(self, tmp_path):
        """Flatten mode preserves top-level scalars (not inside sections)."""
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        self._write_toml(a, {"seed": 42, "model": {"dit": "/path/dit"}})
        self._write_toml(b, {"seed": 42, "model": {"dit": "/path/dit"}})
        result = main_diff(["--flatten", "--quiet", str(a), str(b)])
        assert result == 0

    def test_flatten_collision_last_write_wins(self, tmp_path):
        """Flatten mode uses last-write-wins for key collisions across sections."""
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        # In file a, 'lr' appears in both sections -- second section wins
        self._write_toml(a, {"optimizer": {"lr": 1e-4}, "training": {"lr": 5e-5}})
        # In file b, 'lr' only in one section
        self._write_toml(b, {"training": {"lr": 5e-5}})
        result = main_diff(["--flatten", "--quiet", str(a), str(b)])
        assert result == 0

    def test_missing_file_exits_nonzero(self, tmp_path, capsys):
        """Referencing a nonexistent file should error."""
        a = tmp_path / "a.toml"
        self._write_toml(a, {"training": {"lr": 5e-5}})
        b = tmp_path / "nonexistent.toml"
        result = main_diff(["--quiet", str(a), str(b)])
        assert result != 0
