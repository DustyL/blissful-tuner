"""Tests for TOML writer with provenance headers."""

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from blissful_tuner.config_manager.writer import render_training_toml, render_dataset_toml


class TestRenderTrainingToml:
    def test_output_starts_with_provenance(self):
        training = {"model": {"dit": "/path/to/dit.safetensors"}}
        provenance = {"machine": "b300.toml", "arch": "wan22_t2v", "persona": "DLAY.toml", "preset": "adamw.toml"}
        output = render_training_toml(training, provenance)
        assert output.startswith("# =======")
        assert "Compiled by blissful-config" in output
        assert "b300.toml" in output

    def test_output_is_valid_toml(self):
        training = {
            "model": {"dit": "/path/dit.safetensors"},
            "optimizer": {"optimizer_type": "adamw", "learning_rate": 5e-5},
            "training": {"max_train_steps": 4000, "seed": 42},
        }
        provenance = {"machine": "b300.toml", "arch": "wan22_t2v", "persona": "DLAY.toml", "preset": "adamw.toml"}
        output = render_training_toml(training, provenance)
        # Should parse without error
        parsed = tomllib.loads(output)
        assert parsed["model"]["dit"] == "/path/dit.safetensors"
        assert parsed["optimizer"]["learning_rate"] == 5e-5

    def test_sections_in_expected_order(self):
        training = {
            "training": {"seed": 42},
            "model": {"dit": "/path"},
            "optimizer": {"lr": 1e-4},
        }
        provenance = {"machine": "m", "arch": "a", "persona": "p", "preset": "pr"}
        output = render_training_toml(training, provenance)
        model_pos = output.index("[model]")
        optimizer_pos = output.index("[optimizer]")
        training_pos = output.index("[training]")
        assert model_pos < optimizer_pos < training_pos

    def test_missing_sections_are_skipped(self):
        training = {
            "model": {"dit": "/path"},
            "training": {"seed": 42},
        }
        provenance = {"machine": "m", "arch": "a", "persona": "p", "preset": "pr"}
        output = render_training_toml(training, provenance)
        assert "[model]" in output
        assert "[training]" in output
        assert "[optimizer]" not in output
        assert "[mask_loss]" not in output
        assert "[prior]" not in output

    def test_arrays_render_as_toml_arrays(self):
        training = {
            "optimizer": {
                "optimizer_args": ["betas=(0.9, 0.99)", "weight_decay=0.01", "eps=1e-8"],
                "lr_scheduler_args": [],
            },
        }
        provenance = {"machine": "m", "arch": "a", "persona": "p", "preset": "pr"}
        output = render_training_toml(training, provenance)
        parsed = tomllib.loads(output)
        assert parsed["optimizer"]["optimizer_args"] == ["betas=(0.9, 0.99)", "weight_decay=0.01", "eps=1e-8"]
        assert parsed["optimizer"]["lr_scheduler_args"] == []

    def test_boolean_values_render_correctly(self):
        training = {
            "training": {
                "gradient_checkpointing": False,
                "flash_attn": True,
                "use_mask_loss": False,
            },
        }
        provenance = {"machine": "m", "arch": "a", "persona": "p", "preset": "pr"}
        output = render_training_toml(training, provenance)
        # TOML booleans are lowercase
        assert "true" in output
        assert "false" in output
        assert "True" not in output
        assert "False" not in output
        parsed = tomllib.loads(output)
        assert parsed["training"]["flash_attn"] is True
        assert parsed["training"]["gradient_checkpointing"] is False

    def test_float_scientific_notation_roundtrips(self):
        training = {
            "optimizer": {"learning_rate": 5e-5},
        }
        provenance = {"machine": "m", "arch": "a", "persona": "p", "preset": "pr"}
        output = render_training_toml(training, provenance)
        parsed = tomllib.loads(output)
        assert parsed["optimizer"]["learning_rate"] == 5e-5

    def test_provenance_includes_override_when_present(self):
        training = {"model": {"dit": "/path"}}
        provenance = {
            "machine": "b300.toml",
            "arch": "wan22_t2v",
            "persona": "DLAY.toml",
            "preset": "adamw.toml",
            "override": "custom.toml",
        }
        output = render_training_toml(training, provenance)
        assert "custom.toml" in output

    def test_full_section_ordering(self):
        """All defined sections appear in the canonical order."""
        training = {
            "model": {"dit": "/path"},
            "dataset": {"dataset_config": "/ds.toml"},
            "network": {"network_module": "networks.lora_wan"},
            "optimizer": {"optimizer_type": "adamw"},
            "training": {"seed": 42},
            "mask_loss": {"use_mask_loss": True},
            "prior": {"prior_preservation_weight": 0.5},
            "sampling": {"sample_every_n_steps": 100},
            "output": {"output_dir": "/out"},
            "advanced": {"compile": True},
        }
        provenance = {"machine": "m", "arch": "a", "persona": "p", "preset": "pr"}
        output = render_training_toml(training, provenance)
        positions = []
        for section in [
            "[model]",
            "[dataset]",
            "[network]",
            "[optimizer]",
            "[training]",
            "[mask_loss]",
            "[prior]",
            "[sampling]",
            "[output]",
            "[advanced]",
        ]:
            pos = output.index(section)
            positions.append(pos)
        # Each section appears after the previous one
        assert positions == sorted(positions)


class TestRenderDatasetToml:
    def test_output_has_general_and_datasets(self):
        dataset = {
            "general": {"caption_extension": ".txt", "batch_size": 4},
            "datasets": [
                {"image_directory": "/path", "resolution": [1024, 1024], "cache_directory": "/cache"},
            ],
        }
        output = render_dataset_toml(dataset, {})
        assert "[general]" in output
        assert "[[datasets]]" in output

    def test_output_is_valid_toml(self):
        dataset = {
            "general": {"caption_extension": ".txt"},
            "datasets": [
                {"image_directory": "/path", "resolution": [1024, 1024]},
            ],
        }
        output = render_dataset_toml(dataset, {})
        parsed = tomllib.loads(output)
        assert parsed["general"]["caption_extension"] == ".txt"
        assert len(parsed["datasets"]) == 1

    def test_multiple_datasets_roundtrip(self):
        dataset = {
            "general": {"caption_extension": ".txt", "batch_size": 2},
            "datasets": [
                {"image_directory": "/imgs", "resolution": [1024, 1024], "cache_directory": "/cache_1024"},
                {"image_directory": "/imgs", "resolution": [1328, 1328], "cache_directory": "/cache_1328"},
                {"image_directory": "/imgs", "resolution": [768, 1344], "cache_directory": "/cache_768x1344"},
            ],
        }
        output = render_dataset_toml(dataset, {})
        parsed = tomllib.loads(output)
        assert len(parsed["datasets"]) == 3
        assert parsed["datasets"][0]["resolution"] == [1024, 1024]
        assert parsed["datasets"][1]["resolution"] == [1328, 1328]
        assert parsed["datasets"][2]["resolution"] == [768, 1344]

    def test_dataset_provenance_header(self):
        dataset = {
            "general": {"caption_extension": ".txt"},
            "datasets": [
                {"image_directory": "/path", "resolution": [1024, 1024]},
            ],
        }
        provenance = {"machine": "b300.toml", "arch": "wan22_t2v", "persona": "DLAY.toml", "preset": "adamw.toml"}
        output = render_dataset_toml(dataset, provenance)
        assert "Compiled by blissful-config" in output
        assert "b300.toml" in output

    def test_empty_provenance_no_header(self):
        dataset = {
            "general": {"caption_extension": ".txt"},
            "datasets": [
                {"image_directory": "/path", "resolution": [1024, 1024]},
            ],
        }
        output = render_dataset_toml(dataset, {})
        # With empty provenance, should still be valid TOML
        parsed = tomllib.loads(output)
        assert parsed["general"]["caption_extension"] == ".txt"

    def test_dataset_mask_fields_preserved(self):
        dataset = {
            "general": {"caption_extension": ".txt"},
            "datasets": [
                {
                    "image_directory": "/imgs",
                    "resolution": [1024, 1024],
                    "cache_directory": "/cache",
                    "mask_directory": "/masks",
                    "alpha_mask": True,
                    "require_mask": False,
                },
            ],
        }
        output = render_dataset_toml(dataset, {})
        parsed = tomllib.loads(output)
        ds = parsed["datasets"][0]
        assert ds["mask_directory"] == "/masks"
        assert ds["alpha_mask"] is True
        assert ds["require_mask"] is False

    def test_dataset_integer_values(self):
        dataset = {
            "general": {"caption_extension": ".txt", "batch_size": 4},
            "datasets": [
                {
                    "image_directory": "/path",
                    "resolution": [1024, 1024],
                    "batch_size": 2,
                    "video_frames": 81,
                    "target_frames": 81,
                },
            ],
        }
        output = render_dataset_toml(dataset, {})
        parsed = tomllib.loads(output)
        assert parsed["general"]["batch_size"] == 4
        assert parsed["datasets"][0]["batch_size"] == 2
        assert parsed["datasets"][0]["video_frames"] == 81
