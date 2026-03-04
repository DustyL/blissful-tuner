"""Tests for config compiler core utilities."""

import pytest

from blissful_tuner.config_manager.compiler import deep_merge, interpolate


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


class TestInterpolate:
    """Variable interpolation: ${scope.key} resolved from context dict."""

    def test_simple_interpolation(self):
        data = {"path": "${machine.models_dir}/model.safetensors"}
        context = {"machine": {"models_dir": "/root/models"}}
        result = interpolate(data, context)
        assert result == {"path": "/root/models/model.safetensors"}

    def test_nested_dict_interpolation(self):
        data = {"model": {"dit": "${machine.models_dir}/dit.safetensors"}}
        context = {"machine": {"models_dir": "/root/models"}}
        result = interpolate(data, context)
        assert result == {"model": {"dit": "/root/models/dit.safetensors"}}

    def test_multiple_vars_in_one_string(self):
        data = {"output": "${machine.output_dir}/${persona.name_lower}_run"}
        context = {
            "machine": {"output_dir": "/root/output"},
            "persona": {"name_lower": "olva"},
        }
        result = interpolate(data, context)
        assert result == {"output": "/root/output/olva_run"}

    def test_no_interpolation_needed(self):
        data = {"plain": "no variables here", "num": 42}
        result = interpolate(data, {})
        assert result == {"plain": "no variables here", "num": 42}

    def test_array_values_interpolated(self):
        data = {"paths": ["${machine.dir}/a", "${machine.dir}/b"]}
        context = {"machine": {"dir": "/root"}}
        result = interpolate(data, context)
        assert result == {"paths": ["/root/a", "/root/b"]}

    def test_missing_var_raises(self):
        data = {"path": "${machine.missing_key}/foo"}
        context = {"machine": {"models_dir": "/root"}}
        with pytest.raises(KeyError, match="missing_key"):
            interpolate(data, context)

    def test_non_string_values_passthrough(self):
        data = {"lr": 5e-5, "flag": True, "steps": 4000}
        result = interpolate(data, {})
        assert result == {"lr": 5e-5, "flag": True, "steps": 4000}

    def test_data_not_mutated(self):
        data = {"path": "${machine.dir}/foo"}
        context = {"machine": {"dir": "/root"}}
        interpolate(data, context)
        assert data == {"path": "${machine.dir}/foo"}

    def test_self_reference_cycle_raises(self):
        """Direct self-reference must error, not infinite-loop."""
        data = {"path": "${machine.path}/sub"}
        context = {"machine": {"path": "${machine.path}/sub"}}
        with pytest.raises(RecursionError, match="[Cc]ycle"):
            interpolate(data, context)

    def test_two_key_cycle_raises(self):
        """Transitive cycle: a → b → a must error."""
        data = {"result": "${machine.a}"}
        context = {"machine": {"a": "${machine.b}/foo", "b": "${machine.a}/bar"}}
        with pytest.raises(RecursionError, match="[Cc]ycle"):
            interpolate(data, context)

    def test_max_depth_exceeded_raises(self):
        """Deeply nested (>10 levels) chain raises even without a true cycle."""
        # Build a chain: v0 → v1 → v2 → ... → v11
        context = {"machine": {f"v{i}": f"${{machine.v{i + 1}}}" for i in range(11)}}
        context["machine"]["v11"] = "terminal"
        data = {"out": "${machine.v0}"}
        with pytest.raises(RecursionError, match="depth"):
            interpolate(data, context)
