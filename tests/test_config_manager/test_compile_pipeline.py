"""Tests for the full compile pipeline."""

import pytest
from pathlib import Path

from blissful_tuner.config_manager.compiler import compile_config, load_layer

FIXTURES = Path(__file__).parent / "fixtures"


class TestLoadLayer:
    def test_load_machine(self):
        data = load_layer(FIXTURES / "machines" / "test_machine.toml")
        assert data["machine"]["name"] == "test"
        assert data["machine"]["paths"]["models_dir"] == "/test/models"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_layer(FIXTURES / "machines" / "nonexistent.toml")


class TestCompileConfig:
    def test_compile_produces_training_and_dataset(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        assert "training_toml" in result
        assert "dataset_toml" in result

    def test_training_toml_has_model_section(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        # Model paths should be interpolated
        assert training["model"]["dit"] == "/test/models/qwen-image/Qwen_Image_2512_BF16.safetensors"

    def test_training_toml_has_optimizer_from_preset(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["optimizer"]["optimizer_type"] == "adamw"
        assert training["optimizer"]["learning_rate"] == 5e-5

    def test_training_toml_has_arch_defaults(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["training"]["timestep_sampling"] == "qwen_shift"

    def test_dataset_toml_has_general_section(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        dataset = result["dataset_toml"]
        assert "general" in dataset
        assert dataset["general"]["caption_extension"] == ".txt"

    def test_dataset_toml_multi_resolution_expansion(self):
        """resolutions = [[1024,1024],[512,512]] -> two [[datasets]] entries."""
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        dataset = result["dataset_toml"]
        assert len(dataset["datasets"]) == 2
        assert dataset["datasets"][0]["resolution"] == [1024, 1024]
        assert dataset["datasets"][1]["resolution"] == [512, 512]

    def test_dataset_toml_cache_dirs_are_sibling(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        dataset = result["dataset_toml"]
        # Cache dirs should be sibling to image dir, not nested
        assert dataset["datasets"][0]["cache_directory"] == "/test/DATASETS/TESTPERSONA/qwen_image_cache_1024x1024"
        assert dataset["datasets"][1]["cache_directory"] == "/test/DATASETS/TESTPERSONA/qwen_image_cache_512x512"

    def test_dataset_toml_mask_directory_when_use_masks(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        dataset = result["dataset_toml"]
        for ds in dataset["datasets"]:
            assert ds["mask_directory"] == "/test/DATASETS/TESTPERSONA/masks"

    def test_output_paths_use_preset_slug(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["output"]["output_dir"] == "/test/output/testpersona_qwen_image_test_adamw"

    def test_provenance_metadata_present(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        provenance = result["provenance"]
        assert provenance["machine"] == "test_machine.toml"
        assert provenance["arch"] == "qwen_image"
        assert provenance["persona"] == "TESTPERSONA.toml"
        assert provenance["preset"] == "test_adamw.toml"

    def test_training_toml_has_network_module(self):
        """Network module should come from the arch registry."""
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["network"]["network_module"] == "networks.lora_qwen_image"

    def test_training_toml_has_network_dim_from_preset(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["network"]["network_dim"] == 64
        assert training["network"]["network_alpha"] == 32

    def test_training_toml_output_name(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["output"]["output_name"] == "testpersona_qwen_image_test_adamw"

    def test_training_toml_logging_dir(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["output"]["logging_dir"] == "/test/output/testpersona_qwen_image_test_adamw/logs"
        assert training["output"]["log_prefix"] == "testpersona_qwen_image_test_adamw_"

    def test_training_toml_advanced_from_machine(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["advanced"]["gradient_checkpointing"] is False
        assert training["advanced"]["compile"] is False

    def test_training_toml_required_variant_args_merged(self):
        """WAN requires task='t2v-A14B' in training section."""
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="wan22_t2v",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["training"]["task"] == "t2v-A14B"

    def test_dataset_toml_batch_size_from_machine(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        dataset = result["dataset_toml"]
        assert dataset["general"]["batch_size"] == 2

    def test_unknown_arch_raises(self):
        with pytest.raises(KeyError, match="Unknown architecture"):
            compile_config(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="nonexistent_arch",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw.toml",
            )

    def test_dataset_toml_image_directory_interpolated(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        dataset = result["dataset_toml"]
        for ds in dataset["datasets"]:
            assert ds["image_directory"] == "/test/DATASETS/TESTPERSONA/images"

    def test_training_toml_preset_training_settings_merged(self):
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["training"]["max_train_steps"] == 2000
        assert training["training"]["save_every_n_steps"] == 250
        assert training["training"]["seed"] == 42
        assert training["training"]["mixed_precision"] == "bf16"

    def test_no_nested_dicts_in_training_toml(self):
        """Hard invariant: no section value in emitted training TOML is a dict."""
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        for section_name, section in result["training_toml"].items():
            if isinstance(section, dict):
                for key, value in section.items():
                    assert not isinstance(value, dict), (
                        f"Nested dict in [{section_name}].{key} — read_config_from_file() would not flatten this correctly"
                    )

    def test_training_extra_flattened(self):
        """[training.extra] keys are promoted into [training], not nested."""
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw_with_extra.toml",
        )
        training = result["training_toml"]
        assert training["training"]["some_new_flag"] is True
        assert training["training"]["experimental_option"] == "fast"
        assert "extra" not in training["training"]

    def test_emitted_toml_has_no_nested_dicts(self):
        """Hard invariant: no section value in emitted training TOML is a dict (with extra)."""
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="qwen_image",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw_with_extra.toml",
        )
        for section_name, section in result["training_toml"].items():
            if isinstance(section, dict):
                for key, value in section.items():
                    assert not isinstance(value, dict), (
                        f"Nested dict in [{section_name}].{key} — read_config_from_file() would not flatten this correctly"
                    )

    def test_training_extra_collision_warns(self, caplog):
        """[training.extra] key colliding with [training] key logs warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            result = compile_config(
                machine_path=FIXTURES / "machines" / "test_machine.toml",
                arch_key="qwen_image",
                persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
                preset_path=FIXTURES / "presets" / "test_adamw_with_extra_collision.toml",
            )
        assert "collision" in caplog.text.lower() or "override" in caplog.text.lower() or "extra" in caplog.text.lower()
        # The extra value should win (last write wins)
        training = result["training_toml"]
        assert training["training"]["seed"] == 99

    def test_model_paths_interpolated_for_wan(self):
        """WAN model paths should use machine.models_dir."""
        result = compile_config(
            machine_path=FIXTURES / "machines" / "test_machine.toml",
            arch_key="wan22_t2v",
            persona_path=FIXTURES / "personas" / "TESTPERSONA.toml",
            preset_path=FIXTURES / "presets" / "test_adamw.toml",
        )
        training = result["training_toml"]
        assert training["model"]["dit"] == "/test/models/WAN/Wan-2.2-T2V-Low-Noise-BF16.safetensors"
        assert training["model"]["dit_high_noise"] == "/test/models/WAN/Wan-2.2-T2V-High-Noise-BF16.safetensors"
        assert training["model"]["vae"] == "/test/models/WAN/Wan2_1_VAE_bf16.safetensors"
        assert training["model"]["t5"] == "/test/models/WAN/umt5-xxl-enc-bf16.safetensors"
