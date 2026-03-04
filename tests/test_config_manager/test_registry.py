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
        # This test is added in Task 16 when wan22_i2v is implemented;
        # listed here as a reminder of the invariant.
        pass  # TODO: enable when wan22_i2v entry is added
