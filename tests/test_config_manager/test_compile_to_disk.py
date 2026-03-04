"""Tests for compile_to_disk -- writes training TOML, dataset TOML, and env.sh to disk."""

try:
    import tomllib
except ImportError:
    import tomli as tomllib
import os
import stat
from pathlib import Path

import pytest

from blissful_tuner.config_manager.compiler import compile_to_disk

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
