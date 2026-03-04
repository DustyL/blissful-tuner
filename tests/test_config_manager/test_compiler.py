"""Tests for config compiler core utilities."""

from blissful_tuner.config_manager.compiler import deep_merge


class TestDeepMerge:
    """Deep merge: dicts merge recursively, arrays/scalars replace."""

    def test_scalars_override(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 3}
        result = deep_merge(base, overlay)
        assert result == {"a": 1, "b": 3}

    def test_dicts_merge_recursively(self):
        base = {"outer": {"a": 1, "b": 2}}
        overlay = {"outer": {"b": 3, "c": 4}}
        result = deep_merge(base, overlay)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_arrays_replace_not_merge(self):
        """Arrays always replace (Helm/Kustomize semantics)."""
        base = {"args": ["betas=(0.9, 0.99)", "weight_decay=0.01"]}
        overlay = {"args": ["weight_decay=0.05"]}
        result = deep_merge(base, overlay)
        assert result == {"args": ["weight_decay=0.05"]}

    def test_new_keys_added(self):
        base = {"a": 1}
        overlay = {"b": 2}
        result = deep_merge(base, overlay)
        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self):
        base = {"outer": {"a": 1}}
        overlay = {"outer": {"b": 2}}
        deep_merge(base, overlay)
        assert base == {"outer": {"a": 1}}

    def test_three_layer_merge(self):
        """Simulates machine + arch + preset merge."""
        machine = {"batch_size": 4, "compile": True}
        arch = {"timestep_sampling": "shift", "batch_size": 1}
        preset = {"learning_rate": 5e-5, "timestep_sampling": "qwen_shift"}
        result = deep_merge(deep_merge(machine, arch), preset)
        assert result == {
            "batch_size": 1,
            "compile": True,
            "timestep_sampling": "qwen_shift",
            "learning_rate": 5e-5,
        }

    def test_none_overlay_value_replaces(self):
        base = {"a": 1}
        overlay = {"a": None}
        result = deep_merge(base, overlay)
        assert result == {"a": None}

    def test_empty_dicts(self):
        assert deep_merge({}, {"a": 1}) == {"a": 1}
        assert deep_merge({"a": 1}, {}) == {"a": 1}
        assert deep_merge({}, {}) == {}
