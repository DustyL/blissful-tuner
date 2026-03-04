"""Tests for architecture registry."""

import pytest

from blissful_tuner.config_manager.registry import ARCH_REGISTRY, resolve_arch


class TestArchRegistry:
    def test_registry_has_expected_keys(self):
        # Phase 1: at least these 3 architectures
        for key in ("wan22_t2v", "qwen_image", "flux2_klein9b"):
            assert key in ARCH_REGISTRY, f"Missing arch: {key}"

    def test_each_arch_has_required_fields(self):
        required = {
            "display_name",
            "train_script",
            "cache_latents_script",
            "cache_te_script",
            "network_module",
            "model_files",
            "defaults",
            "supports",
            "cache_suffix",
            "runtime_arch_short",
            "runtime_arch_full",
        }
        for key, arch in ARCH_REGISTRY.items():
            missing = required - set(arch.keys())
            assert not missing, f"{key} missing fields: {missing}"

    def test_runtime_ids_are_non_empty_strings(self):
        """Every arch must have runtime_arch_short and runtime_arch_full as non-empty strings."""
        for key, arch in ARCH_REGISTRY.items():
            assert isinstance(arch["runtime_arch_short"], str) and arch["runtime_arch_short"], f"{key} bad runtime_arch_short"
            assert isinstance(arch["runtime_arch_full"], str) and arch["runtime_arch_full"], f"{key} bad runtime_arch_full"

    def test_supports_has_mask_loss_key(self):
        for key, arch in ARCH_REGISTRY.items():
            assert "mask_loss" in arch["supports"], f"{key} missing supports.mask_loss"

    def test_model_files_have_at_least_dit(self):
        for key, arch in ARCH_REGISTRY.items():
            assert "dit" in arch["model_files"], f"{key} missing model_files.dit"

    def test_resolve_by_key(self):
        arch = resolve_arch("wan22_t2v")
        assert arch["display_name"] == "WAN 2.2 T2V"

    def test_resolve_by_alias(self):
        arch = resolve_arch("wan-t2v-a14b")
        assert arch["display_name"] == "WAN 2.2 T2V"

    def test_resolve_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown architecture"):
            resolve_arch("nonexistent_arch")

    def test_wan22_t2v_has_dual_dit(self):
        """WAN 2.2 requires dit + dit_high_noise (dual-expert)."""
        wan = ARCH_REGISTRY["wan22_t2v"]
        assert "dit_high_noise" in wan["model_files"]

    def test_wan22_t2v_ga_warning(self):
        """WAN dual-expert has a GA>1 warning (not a hard constraint -- matches runtime behavior)."""
        wan = ARCH_REGISTRY["wan22_t2v"]
        assert "gradient_accumulation_steps" in wan.get("warnings", {})

    def test_wan22_t2v_task_in_required_variant_args(self):
        """CRITICAL: task must be in required_variant_args, not just a top-level field.
        argparse default is 't2v-14B' (WAN 2.1!) -- omitting task silently uses wrong model config."""
        wan = ARCH_REGISTRY["wan22_t2v"]
        assert wan["required_variant_args"]["task"] == "t2v-A14B"

    def test_wan22_i2v_task_in_required_variant_args(self):
        """I2V variant must emit task='i2v-A14B' to override argparse default."""
        wan_i2v = ARCH_REGISTRY["wan22_i2v"]
        assert wan_i2v["required_variant_args"]["task"] == "i2v-A14B"

    def test_wan22_i2v_has_clip(self):
        """WAN 2.2 I2V requires CLIP (unlike T2V which is T5-only)."""
        wan_i2v = ARCH_REGISTRY["wan22_i2v"]
        assert "clip" in wan_i2v["model_files"]

    def test_wan22_i2v_ga_warning(self):
        """WAN 2.2 I2V also has dual-expert GA>1 warning."""
        wan_i2v = ARCH_REGISTRY["wan22_i2v"]
        assert "gradient_accumulation_steps" in wan_i2v.get("warnings", {})

    def test_qwen_image_layered_no_prior(self):
        """Qwen-Image Layered does not support prior preservation (NotImplementedError for layout='layered')."""
        arch = ARCH_REGISTRY["qwen_image_layered"]
        assert arch["supports"]["prior_preservation"] is False

    def test_kandinsky5_dual_text_encoders(self):
        """Kandinsky 5 uses dual text encoders (Qwen2.5-VL + CLIP)."""
        arch = ARCH_REGISTRY["kandinsky5"]
        assert "text_encoder_qwen" in arch["model_files"]
        assert "text_encoder_clip" in arch["model_files"]

    def test_no_duplicate_aliases(self):
        """Every alias must map to exactly one architecture key."""
        seen: dict[str, str] = {}
        for key, arch in ARCH_REGISTRY.items():
            for alias in arch.get("aliases", []):
                assert alias not in seen, f"Duplicate alias '{alias}': used by both '{seen[alias]}' and '{key}'"
                seen[alias] = key


class TestRuntimeIdGoldenTests:
    """Verify every runtime_arch_short/full matches the ARCHITECTURE_* constants."""

    # Map registry keys to expected (short, full) from image_video_dataset.py
    EXPECTED_IDS = {
        "wan22_t2v": ("wan", "wan"),
        "wan22_i2v": ("wan", "wan"),
        "hv": ("hv", "hunyuan_video"),
        "hv_1_5": ("hv15", "hunyuan_video_1_5"),
        "framepack": ("fp", "framepack"),
        "flux_kontext": ("fk", "flux_kontext"),
        "flux2_dev": ("f2d", "flux_2_dev"),
        "flux2_klein9b": ("f2k9b", "flux_2_klein_9b"),
        "flux2_klein9b_distil": ("f2k9b", "flux_2_klein_9b"),
        "flux2_klein4b": ("f2k4b", "flux_2_klein_4b"),
        "flux2_klein4b_distil": ("f2k4b", "flux_2_klein_4b"),
        "qwen_image": ("qi", "qwen_image"),
        "qwen_image_edit": ("qie", "qwen_image_edit"),
        "qwen_image_layered": ("qil", "qwen_image_layered"),
        "zimage_base": ("zi", "z_image"),
        "zimage_turbo": ("zi", "z_image"),
        "kandinsky5": ("k5", "kandinsky5"),
    }

    @pytest.mark.parametrize("arch_key,expected", EXPECTED_IDS.items())
    def test_runtime_id_matches_upstream_constants(self, arch_key, expected):
        """Each runtime_arch_short/full MUST match ARCHITECTURE_* constants."""
        arch = resolve_arch(arch_key)
        expected_short, expected_full = expected
        assert arch["runtime_arch_short"] == expected_short, f"Short ID mismatch for {arch_key}"
        assert arch["runtime_arch_full"] == expected_full, f"Full ID mismatch for {arch_key}"


class TestFlux2ModelVersions:
    """Verify FLUX.2 variant model_version values match FLUX2_MODEL_INFO keys."""

    FLUX2_VARIANTS = {
        "flux2_dev": "dev",
        "flux2_klein9b": "klein-base-9b",
        "flux2_klein9b_distil": "klein-9b",
        "flux2_klein4b": "klein-base-4b",
        "flux2_klein4b_distil": "klein-4b",
    }

    @pytest.mark.parametrize("arch_key,model_version", FLUX2_VARIANTS.items())
    def test_flux2_model_version(self, arch_key, model_version):
        arch = resolve_arch(arch_key)
        # Model version should be in required_variant_args or defaults
        if "model_version" in arch.get("required_variant_args", {}):
            assert arch["required_variant_args"]["model_version"] == model_version
        else:
            assert arch.get("defaults", {}).get("model_version") == model_version


class TestQwenImageVariants:
    """Verify all 3 Qwen-Image variants have correct IDs."""

    QWEN_VARIANTS = {
        "qwen_image": ("qi", "qwen_image"),
        "qwen_image_edit": ("qie", "qwen_image_edit"),
        "qwen_image_layered": ("qil", "qwen_image_layered"),
    }

    @pytest.mark.parametrize("arch_key,expected", QWEN_VARIANTS.items())
    def test_qwen_variant_ids(self, arch_key, expected):
        arch = resolve_arch(arch_key)
        assert arch["runtime_arch_short"] == expected[0]
        assert arch["runtime_arch_full"] == expected[1]


class TestRegistryCompleteness:
    """Verify all expected architectures are registered."""

    def test_total_architecture_count(self):
        """We expect exactly 17 architectures."""
        assert len(ARCH_REGISTRY) == 17

    def test_all_expected_keys_present(self):
        expected = {
            "wan22_t2v",
            "wan22_i2v",
            "hv",
            "hv_1_5",
            "framepack",
            "flux_kontext",
            "flux2_dev",
            "flux2_klein9b",
            "flux2_klein9b_distil",
            "flux2_klein4b",
            "flux2_klein4b_distil",
            "qwen_image",
            "qwen_image_edit",
            "qwen_image_layered",
            "zimage_base",
            "zimage_turbo",
            "kandinsky5",
        }
        assert set(ARCH_REGISTRY.keys()) == expected
