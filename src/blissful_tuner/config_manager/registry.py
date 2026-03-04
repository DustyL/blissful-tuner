"""Architecture registry -- single source of truth for architecture-specific config."""

from __future__ import annotations

from typing import Any

ARCH_REGISTRY: dict[str, dict[str, Any]] = {
    "wan22_t2v": {
        "display_name": "WAN 2.2 T2V",
        "aliases": ["wan22-t2v", "wan-t2v-a14b"],
        "runtime_arch_short": "wan",
        "runtime_arch_full": "wan",
        # CRITICAL: task MUST be in required_variant_args so it appears in compiled output.
        # argparse default is "t2v-14B" (WAN 2.1!) -- omitting silently uses wrong config.
        "required_variant_args": {"task": "t2v-A14B"},
        "train_script": "wan_train_network.py",
        "cache_latents_script": "wan_cache_latents.py",
        "cache_te_script": "wan_cache_text_encoder_outputs.py",
        "generate_script": "wan_generate_video.py",
        "network_module": "networks.lora_wan",
        "model_files": {
            "dit": "${machine.models_dir}/WAN/Wan-2.2-T2V-Low-Noise-BF16.safetensors",
            "dit_high_noise": "${machine.models_dir}/WAN/Wan-2.2-T2V-High-Noise-BF16.safetensors",
            "vae": "${machine.models_dir}/WAN/Wan2_1_VAE_bf16.safetensors",
            "t5": "${machine.models_dir}/WAN/umt5-xxl-enc-bf16.safetensors",
        },
        "defaults": {
            "timestep_sampling": "shift",
            "discrete_flow_shift": 12.0,
            "timestep_boundary": 0.875,
            "sigmoid_scale": 2.0,
            "flash_attn": True,
            "rope_func": "comfy",
        },
        "dataset_defaults": {
            "default_resolutions": [[1024, 1024], [512, 512], [256, 256]],
            "video_target_frames": [1, 17, 33, 49, 81],
            "frame_extraction": "head",
            "source_fps": 30.0,
        },
        "supports": {
            "mask_loss": True,
            "prior_preservation": True,
            "latent_preview": True,
            "cfg_schedule": True,
            "cfgzerostar": True,
            "nag": True,
            "riflex": True,
            "v2v": True,
            "i2v": True,
            "cute_attention": False,
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
        "cache_suffix": "wan22_t2v",
    },
    "qwen_image": {
        "display_name": "Qwen-Image",
        "aliases": ["qwen", "qwen-image", "qwen_image_2512"],
        "runtime_arch_short": "qi",
        "runtime_arch_full": "qwen_image",
        "required_variant_args": {},
        "train_script": "qwen_image_train_network.py",
        "cache_latents_script": "qwen_image_cache_latents.py",
        "cache_te_script": "qwen_image_cache_text_encoder_outputs.py",
        "generate_script": "qwen_image_generate_image.py",
        "network_module": "networks.lora_qwen_image",
        "model_files": {
            "dit": "${machine.models_dir}/qwen-image/Qwen_Image_2512_BF16.safetensors",
            "text_encoder": "${machine.models_dir}/qwen-image/qwen_2.5_vl_7b_bf16.safetensors",
            "vae": "${machine.models_dir}/qwen-image/qwen_train_vae.safetensors",
        },
        "defaults": {
            "timestep_sampling": "qwen_shift",
            "discrete_flow_shift": 2.2,
            "model_version": "original",
        },
        "dataset_defaults": {
            "default_resolutions": [[1328, 1328]],
        },
        "supports": {
            "mask_loss": True,
            "prior_preservation": True,
            "latent_preview": False,
            "cfg_schedule": False,
            "cfgzerostar": False,
            "nag": False,
            "riflex": False,
            "v2v": False,
            "i2v": True,
            "cute_attention": True,
        },
        "constraints": {},
        "cache_suffix": "qwen_image",
    },
    "flux2_klein9b": {
        "display_name": "FLUX.2 Klein-base-9B",
        "aliases": ["flux2-klein-9b", "flux2-klein9b", "klein-base-9b"],
        "runtime_arch_short": "f2k9b",
        "runtime_arch_full": "flux_2_klein_9b",
        # model_version must match FLUX2_MODEL_INFO key -- controls block counts, params, and fixed_params
        "required_variant_args": {"model_version": "klein-base-9b"},
        "train_script": "flux_2_train_network.py",
        "cache_latents_script": "flux_2_cache_latents.py",
        "cache_te_script": "flux_2_cache_text_encoder_outputs.py",
        "generate_script": "flux_2_generate_image.py",
        "network_module": "networks.lora_flux_2",
        "model_files": {
            "dit": "${machine.models_dir}/flux-2-klein-base-9b.safetensors",
            "text_encoder": "${machine.models_dir}/text_encoder/model-00001-of-00004.safetensors",
            "vae": "${machine.models_dir}/flux2/ae.safetensors",
        },
        "defaults": {
            "timestep_sampling": "flux2_shift",
            "weighting_scheme": "none",
            "model_version": "klein-base-9b",
            "flash_attn": True,
        },
        "dataset_defaults": {
            "default_resolutions": [[1024, 1024]],
        },
        "supports": {
            "mask_loss": True,
            "prior_preservation": True,
            "latent_preview": False,
            "cfg_schedule": False,
            "cfgzerostar": False,
            "nag": False,
            "riflex": False,
            "v2v": False,
            "i2v": True,
            "cute_attention": False,
        },
        "constraints": {},
        "cache_suffix": "flux2klein9b",
    },
}

# Build alias lookup at import time
_ALIAS_MAP: dict[str, str] = {}
for _key, _arch in ARCH_REGISTRY.items():
    _ALIAS_MAP[_key] = _key
    for _alias in _arch.get("aliases", []):
        _ALIAS_MAP[_alias] = _key


def resolve_arch(name: str) -> dict[str, Any]:
    """Resolve an architecture name or alias to its registry entry."""
    canonical = _ALIAS_MAP.get(name)
    if canonical is None:
        raise KeyError(f"Unknown architecture: '{name}'. Available: {sorted(ARCH_REGISTRY.keys())}")
    return ARCH_REGISTRY[canonical]


def resolve_arch_key(name: str) -> str:
    """Resolve an architecture name or alias to its canonical key."""
    canonical = _ALIAS_MAP.get(name)
    if canonical is None:
        raise KeyError(f"Unknown architecture: '{name}'. Available: {sorted(ARCH_REGISTRY.keys())}")
    return canonical
