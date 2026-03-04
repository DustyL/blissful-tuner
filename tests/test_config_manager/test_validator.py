"""Tests for config validation -- fail-closed feature gating, constraints, and warnings."""

import logging

import pytest

from blissful_tuner.config_manager.validator import (
    ConfigValidationError,
    StrippedKeysReport,
    check_output_path_collision,
    check_preset_slug_uniqueness,
    check_unknown_keys,
    validate_config,
)


# ---------------------------------------------------------------------------
# Helpers: minimal arch dicts for testing
# ---------------------------------------------------------------------------


def _arch_no_mask_loss():
    """Arch that does NOT support mask_loss (e.g., FramePack)."""
    return {
        "supports": {
            "mask_loss": False,
            "prior_preservation": False,
            "cute_attention": False,
        },
        "warnings": {},
        "constraints": {},
    }


def _arch_full_support():
    """Arch that supports everything (e.g., WAN 2.2)."""
    return {
        "supports": {
            "mask_loss": True,
            "prior_preservation": True,
            "cute_attention": True,
        },
        "warnings": {
            "gradient_accumulation_steps": {
                "max_recommended": 1,
                "message": (
                    "Dual-expert training with GA>1 mixes experts within accumulation groups, "
                    "increasing noise. Consider batch_size>1 instead."
                ),
            },
        },
        "constraints": {},
    }


def _arch_with_constraint():
    """Arch with a hard constraint that can never be coerced."""
    return {
        "supports": {
            "mask_loss": True,
            "prior_preservation": True,
            "cute_attention": False,
        },
        "warnings": {},
        "constraints": {
            "blocks_to_swap": {
                "message": "blocks_to_swap is not supported for this architecture",
            },
        },
    }


def _training_toml_with_mask_loss():
    """Training TOML dict that enables mask loss."""
    return {
        "training": {
            "use_mask_loss": True,
            "mask_gamma": 1.2,
            "mask_min_weight": 0.05,
            "learning_rate": 5e-5,
        },
    }


def _training_toml_with_prior():
    """Training TOML dict that enables prior preservation."""
    return {
        "training": {
            "prior_preservation_weight": 0.5,
            "prior_teacher_mode": "ema",
            "prior_teacher_ema_decay": 0.999,
            "learning_rate": 5e-5,
        },
    }


def _training_toml_with_cute():
    """Training TOML dict that enables cute attention."""
    return {
        "training": {
            "cute": True,
            "learning_rate": 5e-5,
        },
    }


def _training_toml_clean():
    """Training TOML dict with no special features."""
    return {
        "training": {
            "learning_rate": 5e-5,
            "max_train_steps": 4000,
        },
    }


# ===========================================================================
# 1. Strict mode: mask_loss on unsupported arch -> ConfigValidationError
# ===========================================================================


class TestStrictModeMaskLoss:
    def test_mask_loss_unsupported_raises(self):
        """use_mask_loss=true on arch with supports.mask_loss=false -> error."""
        with pytest.raises(ConfigValidationError, match="mask.loss"):
            validate_config(
                _training_toml_with_mask_loss(),
                _arch_no_mask_loss(),
                strict=True,
            )

    def test_mask_loss_supported_passes(self):
        """use_mask_loss=true on arch with supports.mask_loss=true -> no error."""
        report = validate_config(
            _training_toml_with_mask_loss(),
            _arch_full_support(),
            strict=True,
        )
        assert report.is_empty()


# ===========================================================================
# 2. Strict mode: prior_preservation on unsupported arch -> error
# ===========================================================================


class TestStrictModePrior:
    def test_prior_unsupported_raises(self):
        """prior_preservation_weight>0 on unsupported arch -> error."""
        with pytest.raises(ConfigValidationError, match="prior"):
            validate_config(
                _training_toml_with_prior(),
                _arch_no_mask_loss(),
                strict=True,
            )

    def test_prior_zero_weight_passes(self):
        """prior_preservation_weight=0 (disabled) should NOT trigger gating."""
        toml_data = {
            "training": {
                "prior_preservation_weight": 0.0,
                "learning_rate": 5e-5,
            },
        }
        report = validate_config(toml_data, _arch_no_mask_loss(), strict=True)
        assert report.is_empty()

    def test_prior_key_absent_passes(self):
        """No prior keys at all should pass even on unsupported arch."""
        report = validate_config(_training_toml_clean(), _arch_no_mask_loss(), strict=True)
        assert report.is_empty()


# ===========================================================================
# 3. Strict mode: cute attention on unsupported arch -> error
# ===========================================================================


class TestStrictModeCute:
    def test_cute_unsupported_raises(self):
        """cute=true on arch with supports.cute_attention=false -> error."""
        with pytest.raises(ConfigValidationError, match="cute"):
            validate_config(
                _training_toml_with_cute(),
                _arch_no_mask_loss(),
                strict=True,
            )

    def test_cute_false_on_unsupported_passes(self):
        """cute=false on unsupported arch -> no error (feature not enabled)."""
        toml_data = {"training": {"cute": False, "learning_rate": 5e-5}}
        report = validate_config(toml_data, _arch_no_mask_loss(), strict=True)
        assert report.is_empty()

    def test_cute_supported_passes(self):
        """cute=true on arch with supports.cute_attention=true -> no error."""
        report = validate_config(
            _training_toml_with_cute(),
            _arch_full_support(),
            strict=True,
        )
        assert report.is_empty()


# ===========================================================================
# 4. Coerce mode: mask_loss stripped with report
# ===========================================================================


class TestCoerceModeMaskLoss:
    def test_mask_loss_stripped(self):
        """In coerce mode, mask_loss keys are stripped, not errored."""
        toml_data = _training_toml_with_mask_loss()
        report = validate_config(toml_data, _arch_no_mask_loss(), strict=False)
        assert not report.is_empty()

        # All mask keys should have been stripped from the training section
        assert "use_mask_loss" not in toml_data["training"]
        assert "mask_gamma" not in toml_data["training"]
        assert "mask_min_weight" not in toml_data["training"]

        # Non-mask keys should remain
        assert toml_data["training"]["learning_rate"] == 5e-5

    def test_mask_loss_report_contains_stripped_keys(self):
        """Report entries should list which keys were stripped."""
        toml_data = _training_toml_with_mask_loss()
        report = validate_config(toml_data, _arch_no_mask_loss(), strict=False)
        stripped_keys = {e["key"] for e in report.entries}
        assert "use_mask_loss" in stripped_keys
        assert "mask_gamma" in stripped_keys
        assert "mask_min_weight" in stripped_keys


# ===========================================================================
# 5. Coerce mode: prior stripped with report
# ===========================================================================


class TestCoerceModePrior:
    def test_prior_stripped(self):
        """In coerce mode, prior keys are stripped when unsupported."""
        toml_data = _training_toml_with_prior()
        report = validate_config(toml_data, _arch_no_mask_loss(), strict=False)
        assert not report.is_empty()

        assert "prior_preservation_weight" not in toml_data["training"]
        assert "prior_teacher_mode" not in toml_data["training"]
        assert "prior_teacher_ema_decay" not in toml_data["training"]
        assert toml_data["training"]["learning_rate"] == 5e-5

    def test_prior_zero_weight_not_stripped(self):
        """prior_preservation_weight=0.0 is disabled, should not be stripped."""
        toml_data = {
            "training": {
                "prior_preservation_weight": 0.0,
                "learning_rate": 5e-5,
            },
        }
        report = validate_config(toml_data, _arch_no_mask_loss(), strict=False)
        assert report.is_empty()


# ===========================================================================
# 6. Coerce mode: report is deterministic (same input -> same output)
# ===========================================================================


class TestCoerceDeterministic:
    def test_report_deterministic(self):
        """Same input always produces same report format string."""
        results = []
        for _ in range(3):
            toml_data = _training_toml_with_mask_loss()
            report = validate_config(toml_data, _arch_no_mask_loss(), strict=False)
            results.append(report.format())

        assert results[0] == results[1] == results[2]


# ===========================================================================
# 7. Warning: WAN GA>1 -> warning logged (not error)
# ===========================================================================


class TestArchWarnings:
    def test_ga_warning_logged(self, caplog):
        """gradient_accumulation_steps>1 on WAN should warn, not error."""
        toml_data = {
            "training": {
                "gradient_accumulation_steps": 4,
                "learning_rate": 5e-5,
            },
        }
        with caplog.at_level(logging.WARNING):
            report = validate_config(toml_data, _arch_full_support(), strict=True)

        # No error raised
        assert report.is_empty()
        # Warning should have been logged
        assert any("GA>1" in record.message for record in caplog.records)

    def test_ga_within_recommended_no_warning(self, caplog):
        """gradient_accumulation_steps=1 should not produce a warning."""
        toml_data = {
            "training": {
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-5,
            },
        }
        with caplog.at_level(logging.WARNING):
            report = validate_config(toml_data, _arch_full_support(), strict=True)

        assert report.is_empty()
        assert not any("GA>1" in record.message for record in caplog.records)

    def test_ga_absent_no_warning(self, caplog):
        """No gradient_accumulation_steps key -> no warning."""
        toml_data = {"training": {"learning_rate": 5e-5}}
        with caplog.at_level(logging.WARNING):
            validate_config(toml_data, _arch_full_support(), strict=True)

        assert not any("GA>1" in record.message for record in caplog.records)


# ===========================================================================
# 8. Constraint: hard error even in coerce mode
# ===========================================================================


class TestConstraints:
    def test_constraint_error_in_strict_mode(self):
        """Hard constraints raise in strict mode."""
        toml_data = {"training": {"blocks_to_swap": 10}}
        with pytest.raises(ConfigValidationError, match="blocks_to_swap"):
            validate_config(toml_data, _arch_with_constraint(), strict=True)

    def test_constraint_error_in_coerce_mode(self):
        """Hard constraints raise even in coerce mode -- can never be coerced."""
        toml_data = {"training": {"blocks_to_swap": 10}}
        with pytest.raises(ConfigValidationError, match="blocks_to_swap"):
            validate_config(toml_data, _arch_with_constraint(), strict=False)

    def test_constraint_key_absent_passes(self):
        """If the constrained key is absent, no error."""
        toml_data = {"training": {"learning_rate": 5e-5}}
        report = validate_config(toml_data, _arch_with_constraint(), strict=True)
        assert report.is_empty()


# ===========================================================================
# 9. Preset slug uniqueness: collision -> error
# ===========================================================================


class TestPresetSlugUniqueness:
    def test_collision_raises(self, tmp_path):
        """Two preset files with the same slug -> ConfigValidationError."""
        preset_a = tmp_path / "preset_a.toml"
        preset_b = tmp_path / "preset_b.toml"
        preset_a.write_text('[preset]\nslug = "my-preset"\n')
        preset_b.write_text('[preset]\nslug = "my-preset"\n')
        with pytest.raises(ConfigValidationError, match="slug.*my-preset"):
            check_preset_slug_uniqueness([preset_a, preset_b])

    def test_no_collision_passes(self, tmp_path):
        """Two preset files with different slugs -> passes."""
        preset_a = tmp_path / "preset_a.toml"
        preset_b = tmp_path / "preset_b.toml"
        preset_a.write_text('[preset]\nslug = "preset-a"\n')
        preset_b.write_text('[preset]\nslug = "preset-b"\n')
        # Should not raise
        check_preset_slug_uniqueness([preset_a, preset_b])

    def test_single_preset_passes(self, tmp_path):
        """Single preset file always passes."""
        preset = tmp_path / "preset.toml"
        preset.write_text('[preset]\nslug = "only-one"\n')
        check_preset_slug_uniqueness([preset])

    def test_empty_list_passes(self):
        """Empty list of presets -> passes."""
        check_preset_slug_uniqueness([])


# ===========================================================================
# 10. (Preset slug uniqueness no-collision is tested above)
# ===========================================================================

# ===========================================================================
# 11. Unknown --set keys: error by default
# ===========================================================================


class TestUnknownKeys:
    def test_unknown_key_errors(self):
        """Unknown keys raise ConfigValidationError by default."""
        compiled_keys = {"learning_rate", "max_train_steps", "some_totally_unknown_key"}
        known_keys = {"learning_rate", "max_train_steps", "batch_size", "timestep_sampling"}
        with pytest.raises(ConfigValidationError, match="some_totally_unknown_key"):
            check_unknown_keys(compiled_keys, known_keys, allow_unknown=False)

    def test_unknown_key_allowed(self):
        """With allow_unknown=True, unknown keys pass through."""
        compiled_keys = {"learning_rate", "some_totally_unknown_key"}
        known_keys = {"learning_rate", "max_train_steps"}
        # Should not raise
        check_unknown_keys(compiled_keys, known_keys, allow_unknown=True)

    def test_all_keys_known_passes(self):
        """When all keys are known, no error."""
        compiled_keys = {"learning_rate", "max_train_steps"}
        known_keys = {"learning_rate", "max_train_steps", "batch_size"}
        # Should not raise
        check_unknown_keys(compiled_keys, known_keys, allow_unknown=False)

    def test_multiple_unknown_keys_listed(self):
        """Error message lists all unknown keys."""
        compiled_keys = {"learning_rate", "unknown_a", "unknown_b"}
        known_keys = {"learning_rate"}
        with pytest.raises(ConfigValidationError) as exc_info:
            check_unknown_keys(compiled_keys, known_keys, allow_unknown=False)
        msg = str(exc_info.value)
        assert "unknown_a" in msg
        assert "unknown_b" in msg


# ===========================================================================
# 12. StrippedKeysReport formatting
# ===========================================================================


class TestStrippedKeysReport:
    def test_empty_report(self):
        report = StrippedKeysReport()
        assert report.is_empty()
        assert report.format() == "No keys stripped"

    def test_add_and_format(self):
        report = StrippedKeysReport()
        report.add("use_mask_loss", True, "architecture does not support mask loss")
        report.add("mask_gamma", 1.2, "architecture does not support mask loss")

        assert not report.is_empty()
        assert len(report.entries) == 2

        formatted = report.format()
        assert "Stripped 2 key(s):" in formatted
        assert "use_mask_loss" in formatted
        assert "mask_gamma" in formatted

    def test_format_sorted_by_key(self):
        """Report entries are sorted alphabetically by key in output."""
        report = StrippedKeysReport()
        report.add("z_key", 1, "reason")
        report.add("a_key", 2, "reason")

        formatted = report.format()
        lines = formatted.strip().split("\n")
        # First data line should be a_key, second z_key
        assert "a_key" in lines[1]
        assert "z_key" in lines[2]


# ===========================================================================
# Output-path collision guard
# ===========================================================================


class TestOutputPathCollision:
    def test_collision_raises(self, tmp_path):
        """Writing to a file with different provenance -> error."""
        path = tmp_path / "config.toml"
        # Write a file with some provenance header
        path.write_text(
            "# Compiled by blissful-config\n"
            "#   Machine: machine_a.toml\n"
            "#   Architecture: wan22_t2v\n"
            "#   Persona: alice.toml\n"
            "#   Preset: default.toml\n"
            "\n"
            "[training]\n"
            "learning_rate = 5e-5\n"
        )
        different_provenance = {
            "machine": "machine_b.toml",
            "arch": "wan22_t2v",
            "persona": "alice.toml",
            "preset": "default.toml",
        }
        with pytest.raises(ConfigValidationError, match="provenance"):
            check_output_path_collision(path, different_provenance)

    def test_same_provenance_passes(self, tmp_path):
        """Writing to a file with matching provenance -> no error."""
        path = tmp_path / "config.toml"
        path.write_text(
            "# Compiled by blissful-config\n"
            "#   Machine: machine_a.toml\n"
            "#   Architecture: wan22_t2v\n"
            "#   Persona: alice.toml\n"
            "#   Preset: default.toml\n"
            "\n"
            "[training]\n"
            "learning_rate = 5e-5\n"
        )
        same_provenance = {
            "machine": "machine_a.toml",
            "arch": "wan22_t2v",
            "persona": "alice.toml",
            "preset": "default.toml",
        }
        # Should not raise
        check_output_path_collision(path, same_provenance)

    def test_nonexistent_file_passes(self, tmp_path):
        """Writing to a path that doesn't exist -> no collision."""
        path = tmp_path / "does_not_exist.toml"
        provenance = {"machine": "m.toml", "arch": "wan22_t2v", "persona": "p.toml", "preset": "pr.toml"}
        # Should not raise
        check_output_path_collision(path, provenance)

    def test_file_without_provenance_passes(self, tmp_path):
        """Writing to a file that has no provenance header -> no collision."""
        path = tmp_path / "manual.toml"
        path.write_text("[training]\nlearning_rate = 5e-5\n")
        provenance = {"machine": "m.toml", "arch": "wan22_t2v", "persona": "p.toml", "preset": "pr.toml"}
        # Should not raise (no provenance to conflict with)
        check_output_path_collision(path, provenance)


# ===========================================================================
# Edge cases and integration
# ===========================================================================


class TestEdgeCases:
    def test_empty_training_toml_passes(self):
        """Empty training TOML should pass validation."""
        report = validate_config({}, _arch_no_mask_loss(), strict=True)
        assert report.is_empty()

    def test_multi_section_mask_keys(self):
        """Mask keys in any section should be detected."""
        toml_data = {
            "mask_loss": {
                "use_mask_loss": True,
                "mask_gamma": 1.5,
            },
            "training": {
                "learning_rate": 5e-5,
            },
        }
        with pytest.raises(ConfigValidationError, match="mask.loss"):
            validate_config(toml_data, _arch_no_mask_loss(), strict=True)

    def test_coerce_multi_section_strips_correctly(self):
        """Coerce mode strips keys from whatever section they appear in."""
        toml_data = {
            "mask_loss": {
                "use_mask_loss": True,
                "mask_gamma": 1.5,
            },
            "training": {
                "learning_rate": 5e-5,
            },
        }
        report = validate_config(toml_data, _arch_no_mask_loss(), strict=False)
        assert not report.is_empty()
        assert "use_mask_loss" not in toml_data.get("mask_loss", {})
        assert "mask_gamma" not in toml_data.get("mask_loss", {})

    def test_supported_arch_all_features_enabled(self):
        """All features enabled on a fully-supporting arch -> no errors/strips."""
        toml_data = {
            "training": {
                "use_mask_loss": True,
                "mask_gamma": 1.2,
                "prior_preservation_weight": 0.5,
                "prior_teacher_mode": "ema",
                "cute": True,
                "learning_rate": 5e-5,
            },
        }
        report = validate_config(toml_data, _arch_full_support(), strict=True)
        assert report.is_empty()

    def test_prior_without_mask_loss_on_partial_arch(self):
        """Arch supports prior but not mask_loss: prior passes, mask_loss errors."""
        arch = {
            "supports": {
                "mask_loss": False,
                "prior_preservation": True,
                "cute_attention": False,
            },
            "warnings": {},
            "constraints": {},
        }
        toml_data = {
            "training": {
                "prior_preservation_weight": 0.5,
                "learning_rate": 5e-5,
            },
        }
        report = validate_config(toml_data, arch, strict=True)
        assert report.is_empty()
