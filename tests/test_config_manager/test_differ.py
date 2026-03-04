"""Tests for semantic TOML differ."""

from blissful_tuner.config_manager.differ import normalize_toml, semantic_diff, format_diff_report


class TestNormalizeToml:
    def test_sorts_keys(self):
        data = {"z": 1, "a": 2, "m": 3}
        result = normalize_toml(data)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_normalizes_floats(self):
        """12.0 and 12 should be treated as equal after normalization."""
        data = {"shift": 12.0}
        result = normalize_toml(data)
        assert result["shift"] == 12.0  # Stays float, but repr is consistent

    def test_sorts_nested_keys(self):
        data = {"outer": {"z": 1, "a": 2}}
        result = normalize_toml(data)
        assert list(result["outer"].keys()) == ["a", "z"]


class TestSemanticDiff:
    def test_identical_dicts_no_diff(self):
        a = {"lr": 5e-5, "steps": 4000}
        b = {"lr": 5e-5, "steps": 4000}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 0

    def test_changed_value(self):
        a = {"lr": 5e-5}
        b = {"lr": 2e-5}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["key"] == "lr"
        assert diffs[0]["type"] == "changed"

    def test_added_key(self):
        a = {"lr": 5e-5}
        b = {"lr": 5e-5, "steps": 4000}
        diffs = semantic_diff(a, b)
        assert any(d["type"] == "added" and d["key"] == "steps" for d in diffs)

    def test_removed_key(self):
        a = {"lr": 5e-5, "steps": 4000}
        b = {"lr": 5e-5}
        diffs = semantic_diff(a, b)
        assert any(d["type"] == "removed" and d["key"] == "steps" for d in diffs)

    def test_nested_diff(self):
        a = {"model": {"dit": "/path/a", "vae": "/path/vae"}}
        b = {"model": {"dit": "/path/b", "vae": "/path/vae"}}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["key"] == "model.dit"

    def test_float_int_equivalence(self):
        """12.0 and 12 should not produce a diff."""
        a = {"shift": 12.0}
        b = {"shift": 12}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 0

    def test_array_comparison_whole_key(self):
        """Changed array produces one diff for the whole key."""
        a = {"resolution": [1024, 1024]}
        b = {"resolution": [1280, 720]}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["key"] == "resolution"
        assert diffs[0]["type"] == "changed"
        assert diffs[0]["old_value"] == [1024, 1024]
        assert diffs[0]["new_value"] == [1280, 720]

    def test_identical_arrays_no_diff(self):
        a = {"resolution": [1024, 1024]}
        b = {"resolution": [1024, 1024]}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 0

    def test_deeply_nested_diffs(self):
        """3+ levels deep nested diffs use dotted path."""
        a = {"level1": {"level2": {"level3": {"value": "old"}}}}
        b = {"level1": {"level2": {"level3": {"value": "new"}}}}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["key"] == "level1.level2.level3.value"
        assert diffs[0]["type"] == "changed"
        assert diffs[0]["old_value"] == "old"
        assert diffs[0]["new_value"] == "new"

    def test_changed_includes_old_and_new_value(self):
        a = {"lr": 5e-5}
        b = {"lr": 2e-5}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["old_value"] == 5e-5
        assert diffs[0]["new_value"] == 2e-5

    def test_added_includes_new_value(self):
        a = {}
        b = {"steps": 4000}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "added"
        assert diffs[0]["new_value"] == 4000
        assert "old_value" not in diffs[0]

    def test_removed_includes_old_value(self):
        a = {"steps": 4000}
        b = {}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "removed"
        assert diffs[0]["old_value"] == 4000
        assert "new_value" not in diffs[0]

    def test_multiple_diffs_sorted_by_key(self):
        a = {"z_key": 1, "a_key": 2}
        b = {"z_key": 10, "a_key": 20}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 2
        assert diffs[0]["key"] == "a_key"
        assert diffs[1]["key"] == "z_key"

    def test_empty_dicts_no_diff(self):
        diffs = semantic_diff({}, {})
        assert len(diffs) == 0

    def test_dict_becomes_scalar(self):
        """When a nested dict is replaced by a scalar, that's a changed value."""
        a = {"model": {"dit": "/path"}}
        b = {"model": "flat_value"}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["key"] == "model"
        assert diffs[0]["type"] == "changed"

    def test_scalar_becomes_dict(self):
        """When a scalar is replaced by a nested dict, that's a changed value."""
        a = {"model": "flat_value"}
        b = {"model": {"dit": "/path"}}
        diffs = semantic_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["key"] == "model"
        assert diffs[0]["type"] == "changed"


class TestFormatDiffReport:
    def test_no_diffs_message(self):
        report = format_diff_report([])
        assert report == "No differences"

    def test_changed_format(self):
        diffs = [{"key": "lr", "type": "changed", "old_value": 5e-5, "new_value": 2e-5}]
        report = format_diff_report(diffs)
        assert "1 difference(s):" in report
        assert "~" in report
        assert "lr" in report

    def test_added_format(self):
        diffs = [{"key": "steps", "type": "added", "new_value": 4000}]
        report = format_diff_report(diffs)
        assert "+" in report
        assert "steps" in report
        assert "4000" in report

    def test_removed_format(self):
        diffs = [{"key": "steps", "type": "removed", "old_value": 4000}]
        report = format_diff_report(diffs)
        assert "-" in report
        assert "steps" in report
        assert "4000" in report

    def test_multi_diff_report(self):
        diffs = [
            {"key": "lr", "type": "changed", "old_value": 5e-5, "new_value": 2e-5},
            {"key": "steps", "type": "added", "new_value": 4000},
            {"key": "old_param", "type": "removed", "old_value": True},
        ]
        report = format_diff_report(diffs)
        assert "3 difference(s):" in report
        lines = report.strip().split("\n")
        # Header + 3 diff lines
        assert len(lines) == 4

    def test_arrow_in_changed_line(self):
        """Changed values show old -> new with arrow."""
        diffs = [{"key": "lr", "type": "changed", "old_value": 5e-5, "new_value": 2e-5}]
        report = format_diff_report(diffs)
        assert "\u2192" in report  # Unicode arrow
