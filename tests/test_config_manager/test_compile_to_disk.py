"""Tests for compile_to_disk -- writes training TOML, dataset TOML, and env.sh to disk."""

import tomllib
import os
import stat
from pathlib import Path

import pytest

from blissful_tuner.config_manager.compiler import _ENV_KEY_RE, _SAFE_NAME_RE, _shell_escape, compile_to_disk

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def output_dir(tmp_path):
    """Temporary output directory for compile_to_disk."""
    return tmp_path / "compiled_output"


def _run_compile(output_dir: Path, arch_key: str = "qwen_image") -> dict:
    """Helper to invoke compile_to_disk with standard test fixtures."""
    return compile_to_disk(
        machine_path=FIXTURES / "machines" / "test_machine.toml",
        arch_key=arch_key,
        persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
        preset_path=FIXTURES / "presets" / "test_adamw.toml",
        output_dir=output_dir,
    )


class TestCreateThreeFiles:
    """compile_to_disk produces training TOML, dataset TOML, and env.sh."""

    def test_creates_three_files(self, output_dir):
        result = _run_compile(output_dir)
        assert Path(result["training_toml_path"]).exists()
        assert Path(result["dataset_toml_path"]).exists()
        assert Path(result["env_sh_path"]).exists()


class TestTrainingToml:
    """The emitted training TOML is valid and has expected content."""

    def test_training_toml_is_valid(self, output_dir):
        result = _run_compile(output_dir)
        path = Path(result["training_toml_path"])
        with open(path, "rb") as f:
            data = tomllib.load(f)
        # Should have core sections
        assert "model" in data
        assert "optimizer" in data
        assert "training" in data
        assert "network" in data
        assert "output" in data

    def test_training_toml_has_dataset_config_pointer(self, output_dir):
        result = _run_compile(output_dir)
        path = Path(result["training_toml_path"])
        with open(path, "rb") as f:
            data = tomllib.load(f)
        # dataset_config must point to the absolute path of the emitted dataset TOML
        expected_dataset_path = str(Path(result["dataset_toml_path"]).resolve())
        assert data["dataset"]["dataset_config"] == expected_dataset_path


class TestDatasetToml:
    """The emitted dataset TOML is valid and has expected structure."""

    def test_dataset_toml_is_valid(self, output_dir):
        result = _run_compile(output_dir)
        path = Path(result["dataset_toml_path"])
        with open(path, "rb") as f:
            data = tomllib.load(f)
        assert "general" in data
        assert "datasets" in data
        assert len(data["datasets"]) >= 1


class TestEnvSh:
    """The emitted env.sh has correct format, permissions, and content."""

    def test_env_sh_is_executable(self, output_dir):
        result = _run_compile(output_dir)
        path = Path(result["env_sh_path"])
        st = os.stat(path)
        # Check that at least the user execute bit is set
        assert st.st_mode & stat.S_IXUSR

    def test_env_sh_has_shebang(self, output_dir):
        result = _run_compile(output_dir)
        path = Path(result["env_sh_path"])
        content = path.read_text()
        assert content.startswith("#!/usr/bin/env bash\n")

    def test_env_sh_has_exports(self, output_dir):
        result = _run_compile(output_dir)
        path = Path(result["env_sh_path"])
        content = path.read_text()
        # Machine env has PYTHONUNBUFFERED and OMP_NUM_THREADS
        assert 'export PYTHONUNBUFFERED="1"' in content
        assert 'export OMP_NUM_THREADS="4"' in content

    def test_env_sh_has_env_local_sourcing(self, output_dir):
        result = _run_compile(output_dir)
        path = Path(result["env_sh_path"])
        content = path.read_text()
        # Must source .env.local for secrets
        assert ".env.local" in content
        assert "source" in content


class TestOutputDirectoryStructure:
    """Files are placed under {output_dir}/{machine_name}/{PERSONA_NAME}/{ARCH_DIR}/."""

    def test_output_directory_structure(self, output_dir):
        result = _run_compile(output_dir, arch_key="qwen_image")
        training_path = Path(result["training_toml_path"])
        # Structure: output_dir / machine_name / persona_name / arch_display_dir / file
        # Machine name: "test", persona: "TESTPERSONA", arch display: "QWEN-IMAGE"
        expected_parent = output_dir / "test" / "TESTPERSONA" / "QWEN-IMAGE"
        assert training_path.parent == expected_parent

    def test_output_directory_structure_wan(self, output_dir):
        result = _run_compile(output_dir, arch_key="wan22_t2v")
        training_path = Path(result["training_toml_path"])
        expected_parent = output_dir / "test" / "TESTPERSONA" / "WAN-2.2-T2V"
        assert training_path.parent == expected_parent

    def test_output_directory_structure_flux2(self, output_dir):
        result = _run_compile(output_dir, arch_key="flux2_klein9b")
        training_path = Path(result["training_toml_path"])
        expected_parent = output_dir / "test" / "TESTPERSONA" / "FLUX.2-KLEIN-BASE-9B"
        assert training_path.parent == expected_parent


class TestOutputFilenames:
    """Filenames use the {persona_lower}_{arch_key}_{preset_slug} pattern."""

    def test_output_filenames_use_preset_slug(self, output_dir):
        result = _run_compile(output_dir)
        training_path = Path(result["training_toml_path"])
        dataset_path = Path(result["dataset_toml_path"])
        env_path = Path(result["env_sh_path"])

        assert training_path.name == "testpersona_qwen_image_test_adamw.toml"
        assert dataset_path.name == "testpersona_qwen_image_test_adamw_dataset.toml"
        assert env_path.name == "env.sh"

    def test_output_filenames_wan(self, output_dir):
        result = _run_compile(output_dir, arch_key="wan22_t2v")
        training_path = Path(result["training_toml_path"])
        dataset_path = Path(result["dataset_toml_path"])

        assert training_path.name == "testpersona_wan22_t2v_test_adamw.toml"
        assert dataset_path.name == "testpersona_wan22_t2v_test_adamw_dataset.toml"


class TestCompileToDisReturnsCompileResult:
    """compile_to_disk returns the compile result plus file paths."""

    def test_returns_training_toml_dict(self, output_dir):
        result = _run_compile(output_dir)
        assert "training_toml" in result
        assert isinstance(result["training_toml"], dict)

    def test_returns_dataset_toml_dict(self, output_dir):
        result = _run_compile(output_dir)
        assert "dataset_toml" in result
        assert isinstance(result["dataset_toml"], dict)

    def test_returns_provenance(self, output_dir):
        result = _run_compile(output_dir)
        assert "provenance" in result
        assert result["provenance"]["arch"] == "qwen_image"


# ---------------------------------------------------------------------------
# Regression: override_data hash collision (issue #2)
# ---------------------------------------------------------------------------


class TestOverrideDataHashCollision:
    """override_data (TUI) must produce distinct filenames from base compile."""

    def test_override_data_produces_nonempty_hash(self, output_dir):
        """compile with override_data must have _ov_ suffix in filename."""
        result = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_data={"training": {"seed": 99}},
        )
        training_path = Path(result["training_toml_path"])
        assert "_ov_" in training_path.name

    def test_base_and_override_data_coexist(self, output_dir):
        """Base compile and override_data compile produce different files."""
        base = _run_compile(output_dir)
        override = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_data={"training": {"seed": 99}},
        )
        # Different filenames
        assert base["training_toml_path"] != override["training_toml_path"]
        # Both files exist
        assert Path(base["training_toml_path"]).exists()
        assert Path(override["training_toml_path"]).exists()

    def test_manifest_has_two_distinct_entries(self, output_dir):
        """Base and override_data compile produce 2 manifest entries."""
        _run_compile(output_dir)
        compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_data={"training": {"seed": 99}},
        )
        with open(output_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert len(manifest["entries"]) == 2
        hashes = {e["override_hash"] for e in manifest["entries"]}
        assert "" in hashes  # base
        assert len(hashes) == 2  # base + override

    def test_override_sets_and_override_data_same_content_same_hash(self, output_dir):
        """override_sets and override_data with same content produce same hash."""
        from blissful_tuner.config_manager.compiler import _compute_override_hash

        h1 = _compute_override_hash(override_sets=["training.seed=99"])
        h2 = _compute_override_hash(override_data={"training": {"seed": 99}})
        assert h1 == h2
        assert h1 != ""


# ---------------------------------------------------------------------------
# Regression: compile/prune atomicity (issue #1)
# ---------------------------------------------------------------------------


class TestCompilePruneAtomicity:
    """Compile publishes files and manifest under a single lock."""

    def test_compile_files_and_manifest_consistent(self, output_dir):
        """After compile, every manifest entry points to existing files."""
        _run_compile(output_dir)
        _run_compile(output_dir, arch_key="wan22_t2v")

        with open(output_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)

        for entry in manifest["entries"]:
            training_path = output_dir / entry["training_config"]
            dataset_path = output_dir / entry["dataset_config"]
            assert training_path.exists(), f"Missing: {training_path}"
            assert dataset_path.exists(), f"Missing: {dataset_path}"

    def test_prune_does_not_delete_freshly_compiled(self, output_dir):
        """Prune after compile must not delete any files that are in the manifest."""
        from blissful_tuner.config_manager.cli import find_orphaned_files

        _run_compile(output_dir)
        orphans = find_orphaned_files(output_dir)
        # All manifest-referenced files (training, dataset, env) must NOT be orphans.
        orphan_names = {p.name for p in orphans}
        with open(output_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        for entry in manifest["entries"]:
            for key in ("training_config", "dataset_config", "env_config"):
                name = Path(entry[key]).name
                assert name not in orphan_names, f"{key} ({name}) should not be an orphan"

    def test_env_sh_tracked_in_manifest(self, output_dir):
        """env.sh must be tracked in manifest so prune doesn't delete it."""
        _run_compile(output_dir)
        with open(output_dir / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)
        entry = manifest["entries"][0]
        assert "env_config" in entry
        env_path = output_dir / entry["env_config"]
        assert env_path.exists()
        assert env_path.name == "env.sh"

    def test_env_sh_not_orphaned(self, output_dir):
        """env.sh must not appear in orphan list."""
        from blissful_tuner.config_manager.cli import find_orphaned_files

        _run_compile(output_dir)
        orphans = find_orphaned_files(output_dir)
        orphan_names = {p.name for p in orphans}
        assert "env.sh" not in orphan_names


# ---------------------------------------------------------------------------
# Regression: override_data unknown-key enforcement
# ---------------------------------------------------------------------------


class TestOverrideDataUnknownKeyEnforcement:
    """override_data must be subject to the same unknown-key check as override_sets."""

    def test_override_data_unknown_key_rejected(self, output_dir):
        """Unknown key via override_data raises ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Unknown override key"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"training": {"completely_fake_key_xyz": 42}},
            )

    def test_override_data_unknown_key_allowed_with_flag(self, output_dir):
        """Unknown key via override_data succeeds with allow_unknown=True."""
        result = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_data={"training": {"completely_fake_key_xyz": 42}},
            allow_unknown=True,
        )
        assert Path(result["training_toml_path"]).exists()

    def test_override_sets_unknown_key_still_rejected(self, output_dir):
        """Unknown key via override_sets still raises (existing behavior preserved)."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Unknown override key"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_sets=["training.completely_fake_key_xyz=42"],
            )


# ---------------------------------------------------------------------------
# Regression: non-preset override_data section rejection
# ---------------------------------------------------------------------------


class TestNonPresetOverrideDataRejection:
    """override_data with non-overridable sections must be rejected by compile_to_disk."""

    def test_output_section_rejected(self, output_dir):
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Cannot override section"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"output": {"output_name": "HACKED"}},
            )

    def test_model_section_rejected(self, output_dir):
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Cannot override section"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"model": {"dit": "/hacked"}},
            )

    def test_advanced_section_rejected(self, output_dir):
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Cannot override section"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"advanced": {"compile": True}},
            )

    def test_mixed_sections_rejected_even_if_some_valid(self, output_dir):
        """If any section is non-overridable, the entire request is rejected."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Cannot override section"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"training": {"seed": 42}, "output": {"output_name": "X"}},
            )


# ---------------------------------------------------------------------------
# Regression: recompile failure preserves previous artifacts
# ---------------------------------------------------------------------------


class TestRecompileFailureRollback:
    """Recompile failure must preserve the previous good artifacts."""

    def test_previous_artifacts_survive_recompile_failure(self, output_dir):
        """If recompile fails during publish, previous training TOML is restored."""
        # Baseline compile
        result = _run_compile(output_dir)
        training_path = Path(result["training_toml_path"])
        original_content = training_path.read_text()
        assert training_path.exists()

        # Patch os.replace to fail after the first rename (training succeeds, dataset fails)
        original_replace = os.replace
        call_count = [0]

        def failing_replace(src, dst):
            call_count[0] += 1
            # In the publish phase (inside the lock), fail on the 2nd rename (dataset).
            # The backup phase uses replace too, so we need to only fail during publish.
            # Backups happen first (up to 3 calls), then publishes (3 more).
            # Fail on the 5th call = 2nd publish (dataset).
            if call_count[0] == 5:
                raise OSError("Injected failure on dataset rename")
            return original_replace(src, dst)

        from unittest.mock import patch

        with pytest.raises(OSError, match="Injected failure"):
            with patch("blissful_tuner.config_manager.compiler.os.replace", side_effect=failing_replace):
                _run_compile(output_dir)

        # Previous training TOML must be restored
        assert training_path.exists(), "Previous training TOML was lost"
        assert training_path.read_text() == original_content, "Training TOML content changed"


# ---------------------------------------------------------------------------
# Regression: override_path must contribute to artifact identity (finding #1)
# ---------------------------------------------------------------------------


class TestOverridePathIdentity:
    """override_path must produce distinct artifacts from base compile."""

    def test_override_path_produces_nonempty_hash(self, output_dir, tmp_path):
        """Compile with override_path must have _ov_ suffix in filename."""
        override_file = tmp_path / "overrides.toml"
        override_file.write_text("[training]\nseed = 99\n")
        result = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_path=override_file,
        )
        training_path = Path(result["training_toml_path"])
        assert "_ov_" in training_path.name

    def test_base_and_override_path_coexist(self, output_dir, tmp_path):
        """Base compile and override_path compile produce different files."""
        override_file = tmp_path / "overrides.toml"
        override_file.write_text("[training]\nseed = 99\n")
        base = _run_compile(output_dir)
        override = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_path=override_file,
        )
        assert base["training_toml_path"] != override["training_toml_path"]
        assert Path(base["training_toml_path"]).exists()
        assert Path(override["training_toml_path"]).exists()

    def test_override_path_and_override_data_same_content_same_hash(self, tmp_path):
        """Same overrides via pre-loaded file data vs dict produce the same hash."""
        from blissful_tuner.config_manager.compiler import _compute_override_hash, load_layer

        override_file = tmp_path / "overrides.toml"
        override_file.write_text("[training]\nseed = 99\n")
        file_data = load_layer(override_file)
        h_file = _compute_override_hash(file_override_data=file_data)
        h_data = _compute_override_hash(override_data={"training": {"seed": 99}})
        assert h_file == h_data
        assert h_file != ""


# ---------------------------------------------------------------------------
# Regression: override_path fail-closed validation (finding #2)
# ---------------------------------------------------------------------------


class TestOverridePathValidation:
    """override_path must be subject to section restrictions and unknown-key checks."""

    def test_override_path_model_section_rejected(self, output_dir, tmp_path):
        """Override file with non-overridable [model] section is rejected."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        override_file = tmp_path / "bad_override.toml"
        override_file.write_text('[model]\ndit = "/hacked"\n')
        with pytest.raises(ConfigValidationError, match="non-overridable section"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_path=override_file,
            )

    def test_override_path_unknown_key_rejected(self, output_dir, tmp_path):
        """Override file with unknown training key is rejected."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        override_file = tmp_path / "bad_override.toml"
        override_file.write_text("[training]\nfake_key_xyz = 42\n")
        with pytest.raises(ConfigValidationError, match="Unknown override key"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_path=override_file,
            )

    def test_override_path_preset_section_allowed(self, output_dir, tmp_path):
        """Override file with [preset] metadata + [training] overrides is valid."""
        override_file = tmp_path / "good_override.toml"
        override_file.write_text('[preset]\nslug = "custom"\n\n[training]\nseed = 99\n')
        result = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_path=override_file,
        )
        assert Path(result["training_toml_path"]).exists()


# ---------------------------------------------------------------------------
# Regression: section-aware key validation prevents cross-section injection (finding #3)
# ---------------------------------------------------------------------------


class TestCrossSectionInjection:
    """Override keys must be checked against their own section, not globally."""

    def test_training_dit_rejected_as_cross_section(self, output_dir):
        """'training.dit' is rejected even though 'dit' exists in [model]."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Unknown override key 'training.dit'"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"training": {"dit": "/sneaky/injection"}},
            )

    def test_optimizer_network_dim_rejected_as_cross_section(self, output_dir):
        """'optimizer.network_dim' is rejected — network_dim belongs to [network]."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Unknown override key 'optimizer.network_dim'"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"optimizer": {"network_dim": 32}},
            )

    def test_correct_section_accepted(self, output_dir):
        """'network.network_dim' is accepted — correct section."""
        result = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_data={"network": {"network_dim": 16}},
        )
        assert Path(result["training_toml_path"]).exists()


# ---------------------------------------------------------------------------
# Regression: first-time compile failure cleanup (finding #4)
# ---------------------------------------------------------------------------


class TestFirstTimeCompileFailureCleanup:
    """First-time compile failure must not leave partial artifacts on disk."""

    def test_first_compile_failure_cleans_up(self, output_dir):
        """If first compile fails during publish, no partial files remain."""
        from unittest.mock import patch

        original_replace = os.replace
        call_count = [0]

        def failing_replace(src, dst):
            call_count[0] += 1
            # First compile has no backups. Fail on the 2nd rename (dataset).
            if call_count[0] == 2:
                raise OSError("Injected failure on dataset rename")
            return original_replace(src, dst)

        with pytest.raises(OSError, match="Injected failure"):
            with patch("blissful_tuner.config_manager.compiler.os.replace", side_effect=failing_replace):
                _run_compile(output_dir)

        # No partial TOML artifacts should remain (manifest and hidden files excluded)
        leftover = [f for f in output_dir.rglob("*.toml") if f.name != "manifest.toml" and not f.name.startswith(".")]
        assert leftover == [], f"Partial artifacts left behind: {leftover}"


# ---------------------------------------------------------------------------
# Regression: stale _stripped.txt sidecar cleanup (finding #7)
# ---------------------------------------------------------------------------


class TestStaleSidecarCleanup:
    """Stale _stripped.txt sidecars from previous compiles must be cleaned up."""

    def test_stale_sidecar_removed_on_recompile(self, output_dir):
        """If a previous compile had stripping but the current one doesn't,
        the _stripped.txt sidecar must be removed."""
        # First compile (normal — no stripping)
        result = _run_compile(output_dir)
        training_path = Path(result["training_toml_path"])
        stripped_path = training_path.with_name(training_path.stem + "_stripped.txt")

        # Simulate a previous compile that left a sidecar
        stripped_path.write_text("Previously stripped keys\n")
        assert stripped_path.exists()

        # Recompile (no coerce, so no stripping)
        _run_compile(output_dir)

        # Stale sidecar must be gone
        assert not stripped_path.exists()


# ---------------------------------------------------------------------------
# Regression: override_path [preset] in artifact identity (v6 finding #1)
# ---------------------------------------------------------------------------


class TestOverridePathPresetIdentity:
    """override_path [preset] section must contribute to artifact identity."""

    def test_preset_slug_override_produces_distinct_hash(self, tmp_path, output_dir):
        """Override file changing [preset].slug must produce a non-empty hash."""
        override_file = tmp_path / "slug_override.toml"
        override_file.write_text('[preset]\nslug = "custom_slug"\n')
        result = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_path=override_file,
        )
        training_path = Path(result["training_toml_path"])
        # Must have _ov_ suffix (non-empty hash from [preset] content)
        assert "_ov_" in training_path.name

    def test_preset_slug_override_coexists_with_base(self, tmp_path, output_dir):
        """Override file with [preset].slug produces distinct artifact from base."""
        base = _run_compile(output_dir)
        override_file = tmp_path / "slug_override.toml"
        override_file.write_text('[preset]\nslug = "custom_slug"\n')
        override = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_path=override_file,
        )
        assert base["training_toml_path"] != override["training_toml_path"]
        assert Path(base["training_toml_path"]).exists()
        assert Path(override["training_toml_path"]).exists()

    def test_effective_slug_used_in_output_name(self, tmp_path, output_dir):
        """The compiled content's output_name must reflect the effective (merged) slug."""
        override_file = tmp_path / "slug_override.toml"
        override_file.write_text('[preset]\nslug = "custom_slug"\n')
        result = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
            output_dir=output_dir,
            override_path=override_file,
        )
        training_path = Path(result["training_toml_path"])
        with open(training_path, "rb") as f:
            data = tomllib.load(f)
        # The output_name in the file should use the effective slug
        assert "custom_slug" in data["output"]["output_name"]


# ---------------------------------------------------------------------------
# Regression: backup failure during publish (v6 finding #2)
# ---------------------------------------------------------------------------


class TestBackupPhaseFailure:
    """Failure during backup renames must restore already-backed-up files."""

    def test_backup_failure_restores_moved_files(self, output_dir):
        """If the 2nd backup rename fails, the 1st backed-up file is restored."""
        from unittest.mock import patch

        # First compile to create initial artifacts
        result = _run_compile(output_dir)
        training_path = Path(result["training_toml_path"])
        dataset_path = Path(result["dataset_toml_path"])
        env_path = Path(result["env_sh_path"])
        original_training = training_path.read_text()
        original_dataset = dataset_path.read_text()
        original_env = env_path.read_text()

        # Fail on the 2nd os.replace call (inside the lock, during backup phase)
        original_replace = os.replace
        call_count = [0]

        def failing_replace(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:  # 2nd backup rename
                raise OSError("Injected failure during backup")
            return original_replace(src, dst)

        with pytest.raises(OSError, match="Injected failure during backup"):
            with patch("blissful_tuner.config_manager.compiler.os.replace", side_effect=failing_replace):
                _run_compile(output_dir)

        # All three original files must still exist with original content
        assert training_path.exists(), "Training TOML lost during backup failure"
        assert dataset_path.exists(), "Dataset TOML lost during backup failure"
        assert env_path.exists(), "env.sh lost during backup failure"
        assert training_path.read_text() == original_training

        # No .bak_ files should remain
        bak_files = list(output_dir.rglob("*.bak_*"))
        assert bak_files == [], f"Leftover backup files: {bak_files}"


# ---------------------------------------------------------------------------
# Regression: malformed override_data section values (v6 finding #4)
# ---------------------------------------------------------------------------


class TestMalformedOverrideDataValues:
    """Malformed override_data values must raise ConfigValidationError, not crash."""

    def test_scalar_section_value_rejected(self, output_dir):
        """override_data={"training": 123} must raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="must be a dict"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"training": 123},
            )

    def test_list_section_value_rejected(self, output_dir):
        """override_data={"training": [1, 2]} must raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="must be a dict"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"training": [1, 2]},
            )

    def test_string_section_value_rejected(self, output_dir):
        """override_data={"training": "bad"} must raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="must be a dict"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_data={"training": "bad"},
            )


# ---------------------------------------------------------------------------
# Regression: legacy manifest entries without env_config (v6 finding #6)
# ---------------------------------------------------------------------------


class TestLegacyManifestEnvConfig:
    """bt-prune must not orphan env.sh for legacy manifest entries lacking env_config."""

    def test_env_sh_not_orphaned_for_legacy_entry(self, output_dir):
        """Legacy manifest entry (no env_config) must not flag env.sh as orphan."""
        from blissful_tuner.config_manager.cli import find_orphaned_files

        # First compile normally to create directory structure + env.sh
        result = _run_compile(output_dir)
        training_path = Path(result["training_toml_path"])
        env_path = Path(result["env_sh_path"])

        # Rewrite manifest to remove env_config (simulate legacy format)
        manifest_path = output_dir / "manifest.toml"
        with open(manifest_path, "rb") as f:
            manifest = tomllib.load(f)
        for entry in manifest["entries"]:
            entry.pop("env_config", None)

        import tomlkit

        doc = tomlkit.document()
        doc.add("schema_version", manifest["schema_version"])
        doc.add("compiler_version", manifest["compiler_version"])
        aot = tomlkit.aot()
        for e in manifest["entries"]:
            table = tomlkit.table()
            for k, v in e.items():
                table.add(k, v)
            aot.append(table)
        doc.add("entries", aot)
        manifest_path.write_text(tomlkit.dumps(doc))

        # env.sh must NOT be flagged as orphan
        orphans = find_orphaned_files(output_dir)
        orphan_names = {p.name for p in orphans}
        assert "env.sh" not in orphan_names, "env.sh flagged as orphan for legacy manifest"


# ---------------------------------------------------------------------------
# Regression: override_path TOCTOU -- single load (v7 finding #1)
# ---------------------------------------------------------------------------


class TestOverridePathSingleLoad:
    """override_path must be loaded exactly once to avoid TOCTOU divergence."""

    def test_file_override_data_passed_to_compile_config(self, output_dir, tmp_path):
        """compile_to_disk passes pre-loaded file data, not path, to compile_config."""
        from unittest.mock import patch

        override_file = tmp_path / "overrides.toml"
        override_file.write_text("[training]\nseed = 99\n")

        calls = []
        original_compile = compile_to_disk.__wrapped__ if hasattr(compile_to_disk, "__wrapped__") else None

        from blissful_tuner.config_manager import compiler

        original_fn = compiler.compile_config

        def spy_compile_config(**kwargs):
            calls.append(kwargs)
            return original_fn(**kwargs)

        with patch.object(compiler, "compile_config", side_effect=spy_compile_config):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_path=override_file,
            )

        assert len(calls) >= 1
        # compile_config should receive file_override_data (pre-loaded dict)
        assert calls[0].get("file_override_data") is not None
        assert isinstance(calls[0]["file_override_data"], dict)
        assert calls[0]["file_override_data"]["training"]["seed"] == 99


# ---------------------------------------------------------------------------
# Regression: override_path value-type validation (v7 finding #2)
# ---------------------------------------------------------------------------


class TestOverridePathValueType:
    """Override file with non-dict section values must raise ConfigValidationError."""

    def test_list_section_value_rejected(self, output_dir, tmp_path):
        """training = [1,2,3] in override file must raise, not crash with AttributeError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        override_file = tmp_path / "bad_override.toml"
        override_file.write_text("training = [1, 2, 3]\n")
        with pytest.raises(ConfigValidationError, match="must be a table/dict"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_path=override_file,
            )

    def test_scalar_section_value_rejected(self, output_dir, tmp_path):
        """training = 42 in override file must raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        override_file = tmp_path / "bad_override.toml"
        override_file.write_text("training = 42\n")
        with pytest.raises(ConfigValidationError, match="must be a table/dict"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_path=override_file,
            )

    def test_string_section_value_rejected(self, output_dir, tmp_path):
        """training = "bad" in override file must raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        override_file = tmp_path / "bad_override.toml"
        override_file.write_text('training = "bad"\n')
        with pytest.raises(ConfigValidationError, match="must be a table/dict"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_path=override_file,
            )


# ---------------------------------------------------------------------------
# Regression: unsafe preset slug sanitization (v7 finding #4)
# ---------------------------------------------------------------------------


class TestSlugSanitization:
    """Preset slugs with filesystem-unsafe characters must be rejected."""

    def test_slash_in_slug_rejected(self, output_dir, tmp_path):
        """Slug containing '/' must raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        preset_file = tmp_path / "bad_slug.toml"
        preset_file.write_text(
            '[preset]\nname = "Bad"\nslug = "x/y"\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 5e-5\n\n'
            "[network]\nnetwork_dim = 32\nnetwork_alpha = 16\n\n"
            "[training]\nmax_train_steps = 1000\n"
        )
        with pytest.raises(ConfigValidationError, match="filesystem-unsafe"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=preset_file,
                output_dir=output_dir,
            )

    def test_space_in_slug_rejected(self, output_dir, tmp_path):
        """Slug containing spaces must raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        preset_file = tmp_path / "bad_slug.toml"
        preset_file.write_text(
            '[preset]\nname = "Bad"\nslug = "has space"\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 5e-5\n\n'
            "[network]\nnetwork_dim = 32\nnetwork_alpha = 16\n\n"
            "[training]\nmax_train_steps = 1000\n"
        )
        with pytest.raises(ConfigValidationError, match="filesystem-unsafe"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=preset_file,
                output_dir=output_dir,
            )

    def test_safe_slug_accepted(self, output_dir, tmp_path):
        """Slug with alphanumeric, hyphens, underscores, and dots passes."""
        preset_file = tmp_path / "good_slug.toml"
        preset_file.write_text(
            '[preset]\nname = "Good"\nslug = "adamw-v2.1_beta"\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 5e-5\n\n'
            "[network]\nnetwork_dim = 32\nnetwork_alpha = 16\n\n"
            "[training]\nmax_train_steps = 1000\n"
        )
        result = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=preset_file,
            output_dir=output_dir,
        )
        assert Path(result["training_toml_path"]).exists()


# ---------------------------------------------------------------------------
# Regression: output path collision detection (v7 finding #3)
# ---------------------------------------------------------------------------


class TestOutputPathCollision:
    """Compiling with a different preset that produces the same slug must be detected."""

    def test_collision_detected_via_provenance(self, output_dir, tmp_path):
        """If target file exists with different preset source, raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        # Compile with preset A
        preset_a = tmp_path / "preset_a.toml"
        preset_a.write_text(
            '[preset]\nname = "Preset A"\nslug = "shared_slug"\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 5e-5\n\n'
            "[network]\nnetwork_dim = 32\nnetwork_alpha = 16\n\n"
            "[training]\nmax_train_steps = 1000\n"
        )
        compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=preset_a,
            output_dir=output_dir,
        )

        # Compile with preset B (same slug, different source file)
        preset_b = tmp_path / "preset_b.toml"
        preset_b.write_text(
            '[preset]\nname = "Preset B"\nslug = "shared_slug"\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 1e-4\n\n'
            "[network]\nnetwork_dim = 64\nnetwork_alpha = 32\n\n"
            "[training]\nmax_train_steps = 2000\n"
        )
        with pytest.raises(ConfigValidationError, match="different provenance"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=preset_b,
                output_dir=output_dir,
            )

    def test_recompile_same_preset_no_collision(self, output_dir):
        """Recompiling with the same preset must not trigger collision."""
        # First compile
        _run_compile(output_dir)
        # Second compile — same sources, no error
        result = _run_compile(output_dir)
        assert Path(result["training_toml_path"]).exists()

    def test_collision_detected_via_dataset_when_training_deleted(self, output_dir, tmp_path):
        """Collision guard must also check dataset TOML, not just training TOML."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        # Compile with preset A
        preset_a = tmp_path / "preset_a.toml"
        preset_a.write_text(
            '[preset]\nname = "Preset A"\nslug = "shared_slug"\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 5e-5\n\n'
            "[network]\nnetwork_dim = 32\nnetwork_alpha = 16\n\n"
            "[training]\nmax_train_steps = 1000\n"
        )
        result_a = compile_to_disk(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=preset_a,
            output_dir=output_dir,
        )
        # Delete training TOML but leave dataset
        Path(result_a["training_toml_path"]).unlink()

        # Compile with preset B (same slug, different source) — dataset guard catches it
        preset_b = tmp_path / "preset_b.toml"
        preset_b.write_text(
            '[preset]\nname = "Preset B"\nslug = "shared_slug"\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 1e-4\n\n'
            "[network]\nnetwork_dim = 64\nnetwork_alpha = 32\n\n"
            "[training]\nmax_train_steps = 2000\n"
        )
        with pytest.raises(ConfigValidationError, match="different provenance"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=preset_b,
                output_dir=output_dir,
            )


# ---------------------------------------------------------------------------
# Regression: override file [preset] non-table value (v8 finding #2)
# ---------------------------------------------------------------------------


class TestOverrideFilePresetValueType:
    """Override file with [preset] as non-table must raise ConfigValidationError."""

    def test_preset_scalar_rejected(self, output_dir, tmp_path):
        """preset = 42 in override file must raise, not crash."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        override_file = tmp_path / "bad_override.toml"
        override_file.write_text("preset = 42\n")
        with pytest.raises(ConfigValidationError, match="must be a table/dict"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_path=override_file,
            )

    def test_preset_string_rejected(self, output_dir, tmp_path):
        """preset = "bad" in override file must raise ConfigValidationError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        override_file = tmp_path / "bad_override.toml"
        override_file.write_text('preset = "bad"\n')
        with pytest.raises(ConfigValidationError, match="must be a table/dict"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
                override_path=override_file,
            )


# ---------------------------------------------------------------------------
# Regression: non-string slug/persona name TypeError (v8 finding #3)
# ---------------------------------------------------------------------------


class TestNonStringSlugType:
    """Non-string preset slug or persona name must raise ConfigValidationError."""

    def test_integer_slug_rejected(self, output_dir, tmp_path):
        """[preset] slug = 123 must raise ConfigValidationError, not TypeError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        preset_file = tmp_path / "int_slug.toml"
        preset_file.write_text(
            '[preset]\nname = "Test"\nslug = 123\n\n'
            '[optimizer]\noptimizer_type = "adamw"\nlearning_rate = 5e-5\n\n'
            "[network]\nnetwork_dim = 32\nnetwork_alpha = 16\n\n"
            "[training]\nmax_train_steps = 1000\n"
        )
        with pytest.raises(ConfigValidationError, match="must be a string"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=preset_file,
                output_dir=output_dir,
            )

    def test_integer_persona_name_rejected(self, output_dir, tmp_path):
        """[persona] name = 999 must raise ConfigValidationError, not TypeError."""
        from blissful_tuner.config_manager.validator import ConfigValidationError

        persona_file = tmp_path / "int_persona.toml"
        persona_file.write_text('[persona]\nname = 999\n\n[dataset]\nimage_directory = "/images"\nresolutions = [[1024, 1024]]\n')
        with pytest.raises(ConfigValidationError, match="must be a string"):
            compile_to_disk(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=persona_file,
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
                output_dir=output_dir,
            )


class TestShellEscape:
    """_shell_escape handles dangerous shell characters."""

    def test_dollar_escaped(self):
        assert _shell_escape("$(whoami)") == "\\$(whoami)"

    def test_backtick_escaped(self):
        assert _shell_escape("`id`") == "\\`id\\`"

    def test_double_quote_escaped(self):
        assert _shell_escape('say "hello"') == 'say \\"hello\\"'

    def test_backslash_escaped(self):
        assert _shell_escape("a\\b") == "a\\\\b"

    def test_exclamation_escaped(self):
        assert _shell_escape("hello!") == "hello\\!"

    def test_safe_value_unchanged(self):
        assert _shell_escape("/opt/blissful-tuner") == "/opt/blissful-tuner"

    def test_combined_dangerous(self):
        result = _shell_escape('$(rm -rf /)"`evil`"$HOME')
        assert "\\$" in result
        assert "\\`" in result


class TestEnvKeyValidation:
    """Invalid env var keys are rejected by _ENV_KEY_RE."""

    def test_valid_key(self):
        assert _ENV_KEY_RE.match("CUDA_VISIBLE_DEVICES")

    def test_valid_underscore_start(self):
        assert _ENV_KEY_RE.match("_PRIVATE")

    def test_invalid_key_with_spaces(self):
        assert not _ENV_KEY_RE.match("BAD KEY")

    def test_invalid_key_with_equals(self):
        assert not _ENV_KEY_RE.match("KEY=VALUE")

    def test_invalid_key_starts_with_number(self):
        assert not _ENV_KEY_RE.match("1BAD")

    def test_invalid_empty(self):
        assert not _ENV_KEY_RE.match("")


class TestMachineNameValidation:
    """Unsafe machine names are rejected by _SAFE_NAME_RE."""

    def test_valid_name(self):
        assert _SAFE_NAME_RE.match("my-machine")

    def test_path_traversal_rejected(self):
        assert not _SAFE_NAME_RE.match("../../etc")

    def test_slash_rejected(self):
        assert not _SAFE_NAME_RE.match("foo/bar")

    def test_space_rejected(self):
        assert not _SAFE_NAME_RE.match("bad name")

    def test_dot_start_rejected(self):
        assert not _SAFE_NAME_RE.match(".hidden")
