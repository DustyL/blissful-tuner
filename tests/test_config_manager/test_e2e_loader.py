"""E2E loader smoke tests -- verify compiled configs are compatible with runtime parsers.

These tests compile configs via compile_to_disk() using test fixtures, then load
the emitted TOML files through the ACTUAL runtime parser chain used by training
scripts.  This catches format mismatches, unknown keys, and schema violations
before they surface at training time.

Heavy ML dependencies (torch, accelerate, etc.) are required because the runtime
parsers import them at module level.  Tests skip gracefully when unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from blissful_tuner.config_manager.compiler import compile_to_disk

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def compiled_qwen(tmp_path):
    """Compile a qwen_image config from test fixtures and return the result dict."""
    output_dir = tmp_path / "compiled"
    return compile_to_disk(
        machine_path=FIXTURES / "machines" / "test_machine.toml",
        arch_key="qwen_image",
        persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
        preset_path=FIXTURES / "presets" / "test_adamw.toml",
        output_dir=output_dir,
    )


def _parser_known_dests(parser: argparse.ArgumentParser) -> set[str]:
    """Extract all known argument destinations from an ArgumentParser."""
    return {action.dest for action in parser._actions}


def _load_training_config(compiled_qwen: dict, monkeypatch):
    """Build parser, run read_config_from_file, return parsed namespace.

    read_config_from_file internally calls parser.parse_args(namespace=...)
    which re-reads sys.argv.  We set sys.argv to include --config_file so
    that the second parse_args picks it up (mirroring real CLI invocation).
    """
    from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
    from musubi_tuner.qwen_image_train_network import qwen_image_setup_parser

    parser = setup_parser_common()
    parser = qwen_image_setup_parser(parser)

    training_toml_path = compiled_qwen["training_toml_path"]
    # Simulate CLI invocation: the runtime parse_args() re-reads sys.argv
    monkeypatch.setattr(sys, "argv", ["test", "--config_file", training_toml_path])

    args = argparse.Namespace(config_file=training_toml_path)
    result = read_config_from_file(args, parser)
    return result, parser


# ---------------------------------------------------------------------------
# Test 1: Training TOML through read_config_from_file
# ---------------------------------------------------------------------------


class TestCompiledTrainingTomlLoadsViaReadConfigFromFile:
    """Compiled training TOML must survive the runtime parser chain.

    The chain is:
    1. setup_parser_common() -- base args shared across all architectures
    2. qwen_image_setup_parser() -- qwen-image-specific args
    3. read_config_from_file() -- TOML loader that flattens sections, then calls
       parser.parse_args(namespace=...) to merge with argparse defaults
    """

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        pytest.importorskip("torch", reason="Runtime parser tests require torch")

    def test_loads_without_error(self, compiled_qwen, monkeypatch):
        result, _ = _load_training_config(compiled_qwen, monkeypatch)
        assert result is not None

    def test_model_paths_resolved(self, compiled_qwen, monkeypatch):
        result, _ = _load_training_config(compiled_qwen, monkeypatch)

        # Model paths should be interpolated -- no ${...} variables remaining
        assert result.dit is not None
        assert "${" not in result.dit
        assert result.dit == "/test/models/qwen-image/Qwen_Image_2512_BF16.safetensors"

        assert result.text_encoder is not None
        assert "${" not in result.text_encoder

        assert result.vae is not None
        assert "${" not in result.vae

    def test_optimizer_type_set(self, compiled_qwen, monkeypatch):
        result, _ = _load_training_config(compiled_qwen, monkeypatch)
        assert result.optimizer_type == "adamw"

    def test_dataset_config_points_to_file(self, compiled_qwen, monkeypatch):
        result, _ = _load_training_config(compiled_qwen, monkeypatch)

        # dataset_config should be a Path pointing to the emitted dataset TOML
        dataset_config = result.dataset_config
        assert dataset_config is not None
        dataset_path = Path(str(dataset_config))
        assert dataset_path.exists(), f"dataset_config points to non-existent file: {dataset_path}"

    def test_no_unknown_keys_leak_through(self, compiled_qwen, monkeypatch):
        """Verify that every key in the compiled training TOML maps to a known
        parser destination.

        argparse.parse_args(namespace=...) silently preserves unknown attributes
        on the namespace, so we must explicitly check against the parser's known
        action destinations.
        """
        result, parser = _load_training_config(compiled_qwen, monkeypatch)

        known_dests = _parser_known_dests(parser)
        # Also include config_file itself (set programmatically, not from TOML)
        known_dests.add("config_file")

        result_keys = set(vars(result).keys())
        unknown_keys = result_keys - known_dests

        assert not unknown_keys, (
            f"Compiled training TOML contains keys unknown to the runtime parser: {sorted(unknown_keys)}. "
            "Either add these args to the parser or remove them from the compiled output."
        )

    def test_arch_defaults_applied(self, compiled_qwen, monkeypatch):
        """Verify architecture-specific defaults from the registry made it through."""
        result, _ = _load_training_config(compiled_qwen, monkeypatch)

        # qwen_image registry defaults
        assert result.timestep_sampling == "qwen_shift"
        assert result.discrete_flow_shift == 2.2
        assert result.model_version == "original"

    def test_network_module_set(self, compiled_qwen, monkeypatch):
        result, _ = _load_training_config(compiled_qwen, monkeypatch)
        assert result.network_module == "networks.lora_qwen_image"


# ---------------------------------------------------------------------------
# Test 2: Dataset TOML through ConfigSanitizer
# ---------------------------------------------------------------------------


class TestCompiledDatasetTomlPassesSanitizer:
    """Compiled dataset TOML must pass the voluptuous schema validation
    performed by ConfigSanitizer.sanitize_user_config()."""

    @pytest.fixture(autouse=True)
    def _skip_without_deps(self):
        pytest.importorskip("voluptuous", reason="ConfigSanitizer requires voluptuous")
        pytest.importorskip("torch", reason="config_utils imports torch transitively")

    def test_sanitize_does_not_raise(self, compiled_qwen):
        from musubi_tuner.dataset.config_utils import ConfigSanitizer, load_user_config

        dataset_path = compiled_qwen["dataset_toml_path"]
        # Load the TOML through the runtime loader (handles JSON/TOML dispatch)
        dataset_dict = load_user_config(dataset_path)

        sanitizer = ConfigSanitizer()
        # This must not raise MultipleInvalid
        sanitized = sanitizer.sanitize_user_config(dataset_dict)

        assert sanitized is not None
        assert "general" in sanitized
        assert "datasets" in sanitized

    def test_sanitized_resolutions_are_tuples(self, compiled_qwen):
        """The sanitizer converts resolution lists to tuples -- verify it works."""
        from musubi_tuner.dataset.config_utils import ConfigSanitizer, load_user_config

        dataset_path = compiled_qwen["dataset_toml_path"]
        dataset_dict = load_user_config(dataset_path)

        sanitizer = ConfigSanitizer()
        sanitized = sanitizer.sanitize_user_config(dataset_dict)

        for ds in sanitized["datasets"]:
            assert isinstance(ds["resolution"], tuple), f"Expected tuple, got {type(ds['resolution'])}"
            assert len(ds["resolution"]) == 2


# ---------------------------------------------------------------------------
# Test 3: Dataset TOML through load_user_config
# ---------------------------------------------------------------------------


class TestCompiledDatasetTomlLoadsViaLoadUserConfig:
    """Compiled dataset TOML must load through the runtime load_user_config()
    function and have the expected structure."""

    @pytest.fixture(autouse=True)
    def _skip_without_deps(self):
        pytest.importorskip("torch", reason="config_utils imports torch transitively")

    def test_loads_without_error(self, compiled_qwen):
        from musubi_tuner.dataset.config_utils import load_user_config

        dataset_path = compiled_qwen["dataset_toml_path"]
        config = load_user_config(dataset_path)

        assert config is not None
        assert isinstance(config, dict)

    def test_has_general_section(self, compiled_qwen):
        from musubi_tuner.dataset.config_utils import load_user_config

        config = load_user_config(compiled_qwen["dataset_toml_path"])
        assert "general" in config
        assert isinstance(config["general"], dict)
        assert "caption_extension" in config["general"]
        assert "batch_size" in config["general"]

    def test_has_datasets_section(self, compiled_qwen):
        from musubi_tuner.dataset.config_utils import load_user_config

        config = load_user_config(compiled_qwen["dataset_toml_path"])
        assert "datasets" in config
        assert isinstance(config["datasets"], list)
        assert len(config["datasets"]) >= 1

    def test_dataset_entries_have_required_fields(self, compiled_qwen):
        from musubi_tuner.dataset.config_utils import load_user_config

        config = load_user_config(compiled_qwen["dataset_toml_path"])

        for i, ds in enumerate(config["datasets"]):
            assert "resolution" in ds, f"datasets[{i}] missing 'resolution'"
            assert "image_directory" in ds, f"datasets[{i}] missing 'image_directory'"
            assert "cache_directory" in ds, f"datasets[{i}] missing 'cache_directory'"

    def test_mask_directory_present_when_configured(self, compiled_qwen):
        """Test persona has use_masks=true, so mask_directory should be in dataset entries."""
        from musubi_tuner.dataset.config_utils import load_user_config

        config = load_user_config(compiled_qwen["dataset_toml_path"])

        for ds in config["datasets"]:
            assert "mask_directory" in ds, "Expected mask_directory in dataset entry (persona has use_masks=true)"
            assert "${" not in ds["mask_directory"], "mask_directory still contains unresolved variables"

    def test_multi_resolution_expansion(self, compiled_qwen):
        """Test persona has two resolutions, so there should be two dataset entries."""
        from musubi_tuner.dataset.config_utils import load_user_config

        config = load_user_config(compiled_qwen["dataset_toml_path"])

        # TESTPERSONA.toml has resolutions = [[1024, 1024], [512, 512]]
        assert len(config["datasets"]) == 2

        resolutions = [tuple(ds["resolution"]) for ds in config["datasets"]]
        assert (1024, 1024) in resolutions
        assert (512, 512) in resolutions
