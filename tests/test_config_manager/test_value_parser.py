"""Tests for type-safe value parsing."""

import pytest

from blissful_tuner.config_manager.value_parser import parse_and_typecheck, parse_value


class TestParseValue:
    def test_float_scientific(self):
        assert parse_value("5e-5") == 5e-5
        assert isinstance(parse_value("5e-5"), float)

    def test_int(self):
        assert parse_value("42") == 42
        assert isinstance(parse_value("42"), int)

    def test_bool_true(self):
        assert parse_value("true") is True

    def test_bool_false(self):
        assert parse_value("false") is False

    def test_quoted_string(self):
        assert parse_value('"hello world"') == "hello world"

    def test_bare_string(self):
        assert parse_value("cosine_with_min_lr") == "cosine_with_min_lr"

    def test_array(self):
        assert parse_value('["a", "b"]') == ["a", "b"]

    def test_float_not_confused_with_int(self):
        assert parse_value("1.0") == 1.0
        assert isinstance(parse_value("1.0"), float)

    def test_negative_int(self):
        assert parse_value("-1") == -1
        assert isinstance(parse_value("-1"), int)

    def test_negative_float(self):
        assert parse_value("-0.5") == -0.5
        assert isinstance(parse_value("-0.5"), float)

    def test_array_of_numbers(self):
        assert parse_value("[1, 2, 3]") == [1, 2, 3]

    def test_empty_quoted_string(self):
        assert parse_value('""') == ""

    def test_path_like_string(self):
        assert parse_value("/path/to/model.safetensors") == "/path/to/model.safetensors"

    def test_zero(self):
        assert parse_value("0") == 0
        assert isinstance(parse_value("0"), int)

    def test_zero_float(self):
        assert parse_value("0.0") == 0.0
        assert isinstance(parse_value("0.0"), float)

    def test_large_int(self):
        assert parse_value("100000") == 100000

    def test_negative_scientific(self):
        assert parse_value("-1e-3") == -1e-3
        assert isinstance(parse_value("-1e-3"), float)

    def test_mixed_array(self):
        assert parse_value('[1, "two", 3.0]') == [1, "two", 3.0]

    def test_nested_array(self):
        assert parse_value("[[1, 2], [3, 4]]") == [[1, 2], [3, 4]]

    def test_non_toml_error_propagates(self):
        """Errors other than ValueError/TOMLDecodeError should propagate."""
        import unittest.mock

        with unittest.mock.patch(
            "blissful_tuner.config_manager.value_parser.tomllib.loads",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                parse_value("anything")


class TestParseAndTypecheck:
    def test_matching_type_passes(self):
        result = parse_and_typecheck("2e-5", existing_value=5e-5, key="learning_rate")
        assert result == 2e-5

    def test_type_mismatch_raises(self):
        with pytest.raises(TypeError, match="learning_rate.*float.*str"):
            parse_and_typecheck("fast", existing_value=5e-5, key="learning_rate")

    def test_int_float_coercion_allowed(self):
        """int value where float expected is OK (12 -> 12.0)."""
        result = parse_and_typecheck("12", existing_value=12.0, key="shift")
        assert isinstance(result, float)

    def test_unknown_key_no_typecheck(self):
        """Unknown keys (no existing value) are accepted as-is."""
        result = parse_and_typecheck("42", existing_value=None, key="new_flag")
        assert result == 42

    def test_float_to_int_coercion_rejected(self):
        """float -> int is NOT allowed (lossy)."""
        with pytest.raises(TypeError, match="steps.*int.*float"):
            parse_and_typecheck("1.5", existing_value=100, key="steps")

    def test_bool_existing_with_string_input_raises(self):
        """String input when bool expected must raise."""
        with pytest.raises(TypeError, match="flag.*bool.*str"):
            parse_and_typecheck("yes", existing_value=True, key="flag")

    def test_list_existing_with_int_input_raises(self):
        """int input when list expected must raise."""
        with pytest.raises(TypeError, match="items.*list.*int"):
            parse_and_typecheck("42", existing_value=[1, 2, 3], key="items")

    def test_bool_values_match(self):
        result = parse_and_typecheck("true", existing_value=False, key="enable")
        assert result is True

    def test_list_values_match(self):
        result = parse_and_typecheck("[4, 5, 6]", existing_value=[1, 2, 3], key="dims")
        assert result == [4, 5, 6]

    def test_string_values_match(self):
        result = parse_and_typecheck('"cosine"', existing_value="linear", key="scheduler")
        assert result == "cosine"

    def test_bare_string_matches_string_existing(self):
        """Bare string fallback should match an existing string value."""
        result = parse_and_typecheck("cosine_with_min_lr", existing_value="linear", key="scheduler")
        assert result == "cosine_with_min_lr"
