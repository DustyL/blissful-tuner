import argparse
import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Union
import torch
from tqdm import tqdm

from musubi_tuner.utils.device_utils import synchronize_device
from musubi_tuner.utils.safetensors_utils import (
    MemoryEfficientSafeOpen,
    TensorWeightAdapter,
    WeightTransformHooks,
    get_split_weight_filenames,
)
from musubi_tuner.modules.fp8_optimization_utils import load_safetensors_with_fp8_optimization
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")
_LOGGED_FAST_MERGE_FLAGS: set[str] = set()


UNKNOWN_NETWORK_FORMAT_HINT = (
    "Some scripts support --prefer_lycoris for non-native formats (IA3, DyLoRA, etc.). Otherwise, convert to a supported format."
)


def format_unknown_network_type_error(lora_path: str) -> str:
    """Build a consistent error message for unsupported/unknown LoRA weight formats."""
    return f"Unrecognized weight format in {lora_path}. {UNKNOWN_NETWORK_FORMAT_HINT}"


def _log_fast_merge_flag_once(flag_name: str) -> None:
    if flag_name in _LOGGED_FAST_MERGE_FLAGS:
        return
    logger.info(f"Fast LoRA merge detected {flag_name} weights; using matching merge math.")
    _LOGGED_FAST_MERGE_FLAGS.add(flag_name)


_DIFFUSERS_PREFIXES = frozenset(("diffusion_model", "transformer"))


def convert_diffusers_if_needed(lora_sd: Dict[str, torch.Tensor], prefix: str = "lora_unet_") -> Dict[str, torch.Tensor]:
    """Convert Diffusers-format keys to default format, preserving non-Diffusers keys.

    Splits the state dict into Diffusers-prefixed keys and passthrough keys,
    converts only the Diffusers subset, and merges back. This avoids data loss
    when a state dict contains both Diffusers and already-normalized keys.
    """
    from musubi_tuner.convert_lora import convert_from_diffusers

    diffusers_sd = {}
    passthrough_sd = {}
    for key, value in lora_sd.items():
        if "." in key and key.split(".", 1)[0] in _DIFFUSERS_PREFIXES:
            diffusers_sd[key] = value
        else:
            passthrough_sd[key] = value

    if not diffusers_sd:
        return lora_sd  # nothing to convert

    logger.info("Converting LoRA from foreign key naming format")
    converted = convert_from_diffusers(prefix, diffusers_sd)
    # Merge: passthrough first, converted on top (converted keys take priority on collision)
    passthrough_sd.update(converted)
    return passthrough_sd


_MODEL_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def _make_lora_name_from_model_key(model_weight_key: str) -> str:
    """Convert a model state-dict key (e.g. 'model.diffusion_model.blocks.0.attn.q.weight')
    to the corresponding LoRA module name (e.g. 'lora_unet_blocks_0_attn_q').

    Strips the trailing '.weight' suffix and any model-level prefixes that don't
    appear in LoRA key naming (e.g. 'model.diffusion_model.' from WAN checkpoints).
    """
    lora_name = model_weight_key.rsplit(".", 1)[0]  # remove trailing ".weight"
    for pfx in _MODEL_KEY_PREFIXES:
        if lora_name.startswith(pfx):
            lora_name = lora_name[len(pfx) :]
            break
    return "lora_unet_" + lora_name.replace(".", "_")


def detect_network_type(lora_sd_or_keys: Union[Dict[str, torch.Tensor], Iterable[str]]) -> str:
    """Detect network type from state dict keys.

    Returns 'lora', 'loha', 'lokr', 'hybrid', or 'unknown'.
    'hybrid' means multiple key families coexist (e.g. after QKV conversion).
    Accepts a state dict or an iterable of key strings.
    """
    keys = lora_sd_or_keys.keys() if isinstance(lora_sd_or_keys, dict) else lora_sd_or_keys
    found_types = set()
    for key in keys:
        # Standard LoRA keys (lora_down/lora_up) AND Diffusers-format keys (lora_A/lora_B)
        if "lora_down" in key or "lora_up" in key or "lora_A" in key or "lora_B" in key:
            found_types.add("lora")
        elif "hada_w1_a" in key or "hada_w2_a" in key:
            found_types.add("loha")
        elif "lokr_w1" in key or "lokr_w2" in key or "lokr_w2_a" in key:
            found_types.add("lokr")
    if len(found_types) > 1:
        return "hybrid"
    if len(found_types) == 1:
        return found_types.pop()
    return "unknown"


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    # apply include/exclude patterns
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if "." not in k or regex_include.search(k)}
        logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if "." not in k or not regex_exclude.search(k)}
        logger.info(f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}")

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys() if "." in k]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning("No keys left after filtering.")

    return weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    quantization_mode: str = "block",
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model with fp8 optimization if needed.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths. If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): List of dictionaries of LoRA weight tensors to load.
        lora_multipliers (Optional[List[float]]): List of multipliers for LoRA weights.
        fp8_optimization (bool): Whether to apply FP8 optimization.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        target_keys (Optional[List[str]]): Keys to target for optimization.
        exclude_keys (Optional[List[str]]): Keys to exclude from optimization.
        disable_numpy_memmap (bool): Whether to disable numpy memmap when loading safetensors.
        weight_transform_hooks (Optional[WeightTransformHooks]): Hooks for transforming weights during loading.
    """

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    # load LoRA weights
    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        list_of_lora_weight_keys = []
        for lora_sd in lora_weights_list:
            lora_weight_keys = set(lora_sd.keys())
            list_of_lora_weight_keys.append(lora_weight_keys)

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        # Detect network types for summary logging (actual dispatch is per-key-family)
        lora_network_types = [detect_network_type(lora_sd) for lora_sd in lora_weights_list]
        logger.info(f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}, types: {lora_network_types}")

        # Import merge functions once (deferred to avoid circular imports at module level)
        from musubi_tuner.networks.loha import merge_weights_to_tensor as loha_merge
        from musubi_tuner.networks.lokr import merge_weights_to_tensor as lokr_merge

        # make hook for LoRA merging
        def weight_hook_func(model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            original_dtype = model_weight.dtype
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)  # to make calculation faster

            for lora_weight_keys, lora_sd, multiplier in zip(list_of_lora_weight_keys, lora_weights_list, lora_multipliers):
                lora_name = _make_lora_name_from_model_key(model_weight_key)

                # Per-key-family dispatch: try each family in deterministic order.
                # Each merge function is a no-op if no matching keys found.
                # This handles hybrid dicts (lokr_* + lora_* after QKV conversion).
                model_weight = loha_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)
                model_weight = lokr_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)

                # Standard LoRA path (delegates to shared merge function for dtype safety)
                model_weight = lora_merge_weights_to_tensor(
                    model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device
                )

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device, original_dtype)  # move back to original device

            return model_weight

        weight_hook = weight_hook_func

    state_dict = load_safetensors_with_fp8_optimization_and_hook(
        model_files,
        fp8_optimization,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        target_keys,
        exclude_keys,
        weight_hook=weight_hook,
        quantization_mode=quantization_mode,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        # Exclude non-dotted keys (network-level metadata like lokr_factor, use_rslora_flag)
        remaining = {k for k in lora_weight_keys if "." in k}
        if len(remaining) > 0:
            logger.warning(f"Warning: not all LoRA keys are used: {', '.join(sorted(remaining))}")

    return state_dict


def load_safetensors_with_fp8_optimization_and_hook(
    model_files: list[str],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    weight_hook: callable = None,
    quantization_mode: str = "block",
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Load state dict from safetensors files and merge LoRA weights into the state dict with fp8 optimization if needed.
    """
    if fp8_optimization:
        logger.info(
            f"Loading state dict with FP8 optimization. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        # dit_weight_dtype is not used because we use fp8 optimization
        state_dict = load_safetensors_with_fp8_optimization(
            model_files,
            calc_device,
            target_keys,
            exclude_keys,
            move_to_device=move_to_device,
            weight_hook=weight_hook,
            quantization_mode=quantization_mode,
            disable_numpy_memmap=disable_numpy_memmap,
            weight_transform_hooks=weight_transform_hooks,
        )
    else:
        logger.info(
            f"Loading state dict without FP8 optimization. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                    if weight_hook is None and move_to_device:
                        value = f.get_tensor(key, device=calc_device, dtype=dit_weight_dtype)
                    else:
                        value = f.get_tensor(key)  # we cannot directly load to device because get_tensor does non-blocking transfer
                        if weight_hook is not None:
                            value = weight_hook(key, value, keep_on_calc_device=move_to_device)
                        if move_to_device:
                            value = value.to(calc_device, dtype=dit_weight_dtype, non_blocking=True)
                        elif dit_weight_dtype is not None:
                            value = value.to(dit_weight_dtype)

                    state_dict[key] = value
        if move_to_device:
            synchronize_device(calc_device)

    return state_dict


def lora_merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
    safe_merge: bool = False,
) -> torch.Tensor:
    """Merge standard LoRA weights directly into a model weight tensor.

    Supports Linear and Conv2d (1x1 and 3x3). Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoRA keys found.
    """
    down_key = lora_name + ".lora_down.weight"
    up_key = lora_name + ".lora_up.weight"
    alpha_key = lora_name + ".alpha"
    prefixed_rslora_key = lora_name + ".use_rslora_flag"
    prefixed_dora_flag_key = lora_name + ".use_dora_flag"
    dora_magnitude_key = lora_name + ".dora_layer.weight"

    if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
        return model_weight

    def get_bool_flag(*keys: str) -> bool:
        for key in keys:
            if key in lora_sd:
                value = lora_sd[key]
                return bool(value.item()) if isinstance(value, torch.Tensor) else bool(value)
        return False

    # Current blissful-tuner checkpoints write network-level flag buffers.
    # Accept prefixed flags too, so partial or future per-module dicts still
    # use the same merge math.
    has_dora_magnitude = dora_magnitude_key in lora_sd
    use_rslora = get_bool_flag(prefixed_rslora_key, "use_rslora_flag")
    use_dora = get_bool_flag(prefixed_dora_flag_key, "use_dora_flag")
    if use_rslora:
        _log_fast_merge_flag_once("rsLoRA")
    if use_dora:
        _log_fast_merge_flag_once("DoRA")

    down_weight = lora_sd[down_key].to(calc_device)
    up_weight = lora_sd[up_key].to(calc_device)

    dim = down_weight.size()[0]
    alpha = lora_sd.get(alpha_key, dim)
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    scale = alpha / math.sqrt(dim) if use_rslora else alpha / dim

    org_device = model_weight.device
    original_dtype = model_weight.dtype
    compute_dtype = torch.float16 if original_dtype.itemsize == 1 else torch.float32
    model_weight = model_weight.to(calc_device, dtype=compute_dtype)
    down_weight = down_weight.to(compute_dtype)
    up_weight = up_weight.to(compute_dtype)

    if len(model_weight.size()) == 2:
        # linear
        if len(up_weight.size()) == 4:  # use linear projection mismatch
            up_weight = up_weight.squeeze(3).squeeze(2)
            down_weight = down_weight.squeeze(3).squeeze(2)
        delta_weight = multiplier * (up_weight @ down_weight) * scale
        if use_dora:
            if not has_dora_magnitude:
                raise ValueError(
                    f"DoRA enabled for {lora_name} but {dora_magnitude_key} is missing from weights. "
                    "This would silently produce incorrect results (uninitialized magnitudes)."
                )
            from musubi_tuner.networks.dora_utils import dora_weight_norm_materialized

            dora_magnitude = lora_sd[dora_magnitude_key].to(calc_device, dtype=compute_dtype)
            weight_norm = dora_weight_norm_materialized(model_weight, delta_weight, 1.0)
            dora_factor = dora_magnitude / weight_norm
            merged_weight = dora_factor.view(-1, 1) * (model_weight + delta_weight)
        else:
            merged_weight = model_weight + delta_weight
    elif down_weight.size()[2:4] == (1, 1):
        # conv2d 1x1
        if use_dora and has_dora_magnitude:
            raise NotImplementedError("DoRA fast merge is only supported for Linear weights, not Conv2d weights.")
        merged_weight = (
            model_weight
            + multiplier * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale
        )
    else:
        # conv2d 3x3
        if use_dora and has_dora_magnitude:
            raise NotImplementedError("DoRA fast merge is only supported for Linear weights, not Conv2d weights.")
        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
        merged_weight = model_weight + multiplier * conved * scale

    if safe_merge and not torch.isfinite(merged_weight).all():
        raise ValueError(
            f"Merge for {lora_name} produced non-finite values (rsLoRA={use_rslora}, dora={use_dora}). Refusing to commit."
        )

    model_weight = merged_weight.to(device=org_device, dtype=original_dtype)

    # Remove consumed keys
    for key in [
        down_key,
        up_key,
        alpha_key,
        prefixed_rslora_key,
        prefixed_dora_flag_key,
        "use_rslora_flag",
        "use_dora_flag",
    ]:
        lora_weight_keys.discard(key)
    if use_dora:
        lora_weight_keys.discard(dora_magnitude_key)

    return model_weight


def merge_nonlora_to_model(
    model: torch.nn.Module,
    weights_sd: Dict[str, torch.Tensor],
    multiplier: float,
    device: torch.device,
    safe_merge: bool = False,
) -> int:
    """Merge LoHa/LoKr/LoRA weights directly into model parameters via per-key-family dispatch.

    Iterates model named_parameters, constructs lora_name from each param name,
    and tries each merge family in order (each is a no-op if no matching keys).
    Handles hybrid dicts (e.g. lokr_* + lora_* after QKV conversion).
    Returns number of consumed keys.
    """
    from musubi_tuner.networks.loha import merge_weights_to_tensor as loha_merge
    from musubi_tuner.networks.lokr import merge_weights_to_tensor as lokr_merge

    lora_weight_keys = set(weights_sd.keys())
    initial_key_count = len(lora_weight_keys)

    for param_name, param in model.named_parameters():
        if not param_name.endswith(".weight"):
            continue

        lora_name = "lora_unet_" + param_name.rsplit(".", 1)[0].replace(".", "_")

        # Per-key-family dispatch: LoHa → LoKr → LoRA. Keep the merged tensor
        # local until safe_merge passes so a non-finite result is never committed
        # to the model parameter.
        merged_param = loha_merge(param.data, lora_name, weights_sd, lora_weight_keys, multiplier, device)
        merged_param = lokr_merge(merged_param, lora_name, weights_sd, lora_weight_keys, multiplier, device)
        merged_param = lora_merge_weights_to_tensor(
            merged_param, lora_name, weights_sd, lora_weight_keys, multiplier, device, safe_merge=safe_merge
        )
        if safe_merge and not torch.isfinite(merged_param).all():
            raise ValueError(f"safe_merge detected non-finite merged weight for {lora_name}")
        param.data = merged_param

    merged_count = initial_key_count - len(lora_weight_keys)

    # Warn about remaining unmerged keys (exclude non-dotted metadata like lokr_factor)
    remaining = {k for k in lora_weight_keys if "." in k}
    if remaining:
        logger.warning(f"{len(remaining)} LoHa/LoKr/LoRA keys were not matched to model parameters")

    return merged_count


# =====================================================================
# Hotswap: compile-friendly LoRA replacement
#
# Approach A (copy merged weights into compiled params). The empirical
# foundation: param.data.copy_() on a torch.compile'd module's parameter
# does not trigger Dynamo recompile, and the compiled graph correctly
# reflects the new values on next forward. Verified 2026-05-02.
#
# Phase 1 contract:
#   - Standard LoRA only (detect_network_type == "lora")
#   - --prefer_lycoris and --fp8_scaled rejected at arg-parse time
#   - WAN-only wiring (other architectures deferred to Phase 2)
#
# Critical correctness invariant: every hotswap must start from the
# un-merged base, never from the model's current parameters. See
# docs/plans/2026-05-02-peft-tier1-hotswap.md for the full design.
# =====================================================================


@dataclass
class HotswapState:
    """Per-model state for compile-friendly LoRA hotswap.

    One instance per loaded DiT (so WAN2.2's high+low expert path
    carries two states, one per model). Stored as `model.hotswap_state`.
    """

    base_dit_paths: List[str]
    base_weights_paths: Optional[List[str]] = None
    base_weights_multipliers: Optional[List[float]] = None
    cached_base_sd: Optional[Dict[str, torch.Tensor]] = None
    base_sha256: Optional[str] = None
    cache_in_ram: bool = False
    strict_base_hash: bool = True
    active_lora_paths: List[str] = field(default_factory=list)
    active_lora_multipliers: List[float] = field(default_factory=list)


def setup_parser_hotswap(parser: argparse.ArgumentParser) -> None:
    """Add --prepare_for_hotswap, --cache_unmerged_base, --hotswap_strict_base_hash flags.

    Called from `hv_generate_video.setup_parser_compile()` so every
    generation script that already wires compile flags inherits hotswap
    flags too.
    """
    parser.add_argument(
        "--prepare_for_hotswap",
        action="store_true",
        help=(
            "Enable compile-friendly LoRA hotswap. Lets sweep scripts swap LoRAs without recompiling. "
            "Phase 1: WAN only, standard LoRA only, incompatible with --prefer_lycoris and --fp8_scaled. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--cache_unmerged_base",
        action="store_true",
        help=(
            "Cache the un-merged DiT base in CPU RAM (~14-30 GB depending on architecture). "
            "Without this, hotswap re-loads base from disk on each swap (~3-5s on NVMe). "
            "Tighter RAM systems should leave this off."
        ),
    )
    parser.add_argument(
        "--hotswap_strict_base_hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refuse to hotswap onto a base whose ss_base_sha256 metadata does not match. "
            "Default: ON. Pass --no-hotswap_strict_base_hash to downgrade mismatch to a warning. "
            "LoRAs lacking the metadata always warn-only (back-compat for older checkpoints)."
        ),
    )


def compute_base_hash(dit_paths: List[str]) -> str:
    """SHA256 of the on-disk DiT file(s), in path-list order.

    Phase 1 uses a file-content hash (not a parameter-tensor hash) so it
    can be computed at training time without loading the model into RAM.
    Multi-file DiTs (e.g. WAN2.2's high+low) are hashed by concatenating
    each file's bytes into the same hasher in the order given.
    """
    h = hashlib.sha256()
    chunk_size = 8 * 1024 * 1024
    for path in _expand_weight_paths(dit_paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
    return h.hexdigest()


def _expand_weight_paths(paths: List[str]) -> List[str]:
    """Expand split safetensors paths to the concrete shard list."""
    expanded: List[str] = []
    for path in paths:
        split_paths = get_split_weight_filenames(path)
        if split_paths is None:
            expanded.append(path)
        else:
            expanded.extend(split_paths)
    return expanded


def compute_and_log_base_sha256(args: argparse.Namespace) -> Optional[str]:
    """Compute the training-time base hash and log the appropriate message.

    Returns the hash for the caller to cache. Use this directly when the
    metadata composition happens INSIDE a per-save closure (e.g.,
    hv_train.py's FineTuningTrainer.train.save_model) so the hash isn't
    recomputed from disk on every checkpoint write — a real cost for big
    bases (HV's ~26 GB → ~50s SHA256 per save).

    For the simpler "metadata dict built once, reused per save" pattern,
    use inject_ss_base_sha256_metadata(args, metadata) instead — it
    wraps this helper plus the dict mutation in a single one-line call.

    Behavior:
      - args.dit unset/empty: returns None silently.
      - args.dit_high_noise truthy: returns None and warns about WAN
        dual-expert deferral.
      - otherwise: returns hash and logs an info line with the prefix.
    """
    base_sha256 = compute_training_base_hash(args)
    if base_sha256 is not None:
        logger.info(f"recorded ss_base_sha256={base_sha256[:12]}... for base provenance validation")
    elif getattr(args, "dit_high_noise", None):
        logger.warning(
            "ss_base_sha256 omitted: WAN dual-expert (args.dit_high_noise set) deferred "
            "to Tier 2 #6a-2 follow-up. Hotswap of LoRAs from this run will be warn-only."
        )
    return base_sha256


def inject_ss_base_sha256_metadata(args: argparse.Namespace, metadata: Dict[str, str]) -> None:
    """Add ss_base_sha256 to metadata when computable; warn when deferred.

    One-line caller for the "metadata dict built once, reused per save"
    pattern (the shared LoRA NetworkTrainer path + the two image full-FT
    trainers). Wraps compute_and_log_base_sha256 plus the dict mutation.

    For the rebuilt-per-save closure pattern (hv_train.py SAI metadata),
    cache compute_and_log_base_sha256(args) once at training start and
    check-and-inject inside the closure to avoid recomputing the hash
    on every save.
    """
    base_sha256 = compute_and_log_base_sha256(args)
    if base_sha256 is not None:
        metadata["ss_base_sha256"] = base_sha256


def compute_training_base_hash(args: argparse.Namespace) -> Optional[str]:
    """Compute SHA256 of the DiT base file at training time.

    Single canonical helper for trainer metadata composers. Returns the
    same byte-for-byte hash that hotswap's prepare_for_hotswap computes
    from the same path, so a hotswap-time check can validate a saved
    LoRA against the live base.

    Returns None in two cases (caller should warn-and-omit the metadata
    key in both):

      1. args.dit is unset / empty. No DiT path to hash. Trainers that
         construct the model from a config rather than a checkpoint
         path land here.
      2. args.dit_high_noise is set (WAN 2.2 dual-expert training). The
         hotswap read-side hashes per-expert (single-element dit_paths
         list per HotswapState; verified at wan_generate_video.py:800
         and test_lora_hotswap.py:259), so a combined two-file hash
         would never match either expert's hotswap state. Symmetric
         per-expert metadata keys + read-side any-match validation are
         deferred to a Tier 2 #6a follow-up.

    Empty-string defense for dit_high_noise: == "" is treated as None,
    mirroring the if-truthy gate at wan_generate_video.py:613.
    """
    dit = getattr(args, "dit", None)
    if not dit:
        return None
    high = getattr(args, "dit_high_noise", None)
    if high:
        return None
    return compute_base_hash([dit])


def _assert_standard_lora_only(lora_sd: Dict[str, torch.Tensor], lora_path: str) -> None:
    """Phase 1 contract: hotswap accepts only standard LoRA networks.

    Raises ValueError with actionable message for loha / lokr / hybrid /
    unknown. The caller is expected to supply the LoRA path string for
    the error message.
    """
    net_type = detect_network_type(lora_sd)
    if net_type != "lora":
        raise ValueError(
            f"Hotswap rejected {lora_path}: detected network type {net_type!r}, "
            "but Phase 1 supports standard LoRA only (detect_network_type == 'lora'). "
            "Use the standard non-hotswap merge path (omit --prepare_for_hotswap) for "
            f"{net_type} adapters, or convert to standard LoRA first."
        )


def _check_lora_base_hash(
    lora_sd_metadata: Optional[Dict[str, str]],
    lora_path: str,
    expected_sha256: str,
    strict: bool,
) -> None:
    """Validate the LoRA's ss_base_sha256 metadata against expected.

    Missing on the LoRA side: warn-only (back-compat for old checkpoints).
    Mismatched + strict: raise.
    Mismatched + not strict: warn.
    """
    if not lora_sd_metadata:
        logger.warning(
            f"Hotswap: {lora_path} has no safetensors metadata; cannot validate base hash. "
            "Older blissful-tuner LoRAs predate ss_base_sha256 — proceeding."
        )
        return
    lora_base = lora_sd_metadata.get("ss_base_sha256")
    if lora_base is None:
        logger.warning(f"Hotswap: {lora_path} metadata lacks ss_base_sha256; cannot validate base hash. Proceeding.")
        return
    if lora_base == expected_sha256:
        return
    short_lora = lora_base[:12]
    short_expected = expected_sha256[:12]
    msg = (
        f"Hotswap base-hash mismatch for {lora_path}: "
        f"LoRA was trained against base {short_lora}..., but the loaded base is {short_expected}.... "
    )
    if strict:
        raise ValueError(
            msg + "Pass --no-hotswap_strict_base_hash to downgrade this to a warning, " + "or load the matching base DiT."
        )
    logger.warning(msg + "Proceeding because --no-hotswap_strict_base_hash is set.")


def _read_safetensors_metadata(path: str) -> Optional[Dict[str, str]]:
    """Read just the metadata header of a safetensors file, without loading tensors."""
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            return f.metadata() or {}
    except Exception as e:
        logger.warning(f"Failed to read metadata from {path}: {e}")
        return None


def _copy_state_dict_to_model_parameters(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], source: str) -> int:
    """Copy matching state-dict tensors into model parameters.

    WAN base checkpoints may include a model-level prefix that is stripped
    during the normal load path. Accept those prefixed keys here too so
    hotswap re-load mode resets to the same base the model was built from.
    """
    copied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            tensor = state_dict.get(name)
            if tensor is None:
                for prefix in _MODEL_KEY_PREFIXES:
                    tensor = state_dict.get(prefix + name)
                    if tensor is not None:
                        break
            if tensor is None:
                continue
            param.data.copy_(tensor.to(param.device, param.dtype))
            copied += 1
    if copied == 0:
        sample_keys = ", ".join(list(state_dict.keys())[:5])
        raise ValueError(
            f"Hotswap reset loaded {source} but matched 0 model parameters. "
            f"Check checkpoint architecture/key prefixes. Sample keys: {sample_keys}"
        )
    return copied


def prepare_for_hotswap(
    model: torch.nn.Module,
    dit_paths: List[str],
    base_weights_paths: Optional[List[str]] = None,
    base_weights_multipliers: Optional[List[float]] = None,
    cache_in_ram: bool = False,
    strict_base_hash: bool = True,
) -> HotswapState:
    """One-time setup. Snapshot the un-merged base for future hotswap calls.

    MUST be called AFTER the DiT is loaded WITHOUT initial LoRAs (the
    caller is responsible for suppressing the standard LoRA preload),
    and AFTER any --base_weights merge, but BEFORE the initial active
    LoRA merge.

    cache_in_ram=True snapshots the model's parameters into CPU RAM
    right now. The snapshot is FROZEN — never mutated by subsequent
    hotswaps (hotswap_lora always copies it before merging).

    cache_in_ram=False stores only the paths. Each hotswap re-reads
    dit_paths from disk and re-applies base_weights.
    """
    base_sha256 = compute_base_hash(dit_paths)
    cached_sd: Optional[Dict[str, torch.Tensor]] = None
    if cache_in_ram:
        cached_sd = {name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters()}
        total_bytes = sum(t.element_size() * t.numel() for t in cached_sd.values())
        logger.info(f"Hotswap: cached un-merged base in CPU RAM ({len(cached_sd)} parameters, ~{total_bytes / (1024**3):.1f} GB)")
    state = HotswapState(
        base_dit_paths=list(dit_paths),
        base_weights_paths=list(base_weights_paths) if base_weights_paths else None,
        base_weights_multipliers=list(base_weights_multipliers) if base_weights_multipliers else None,
        cached_base_sd=cached_sd,
        base_sha256=base_sha256,
        cache_in_ram=cache_in_ram,
        strict_base_hash=strict_base_hash,
    )
    logger.info(f"Hotswap prepared: cache_in_ram={cache_in_ram}, strict_hash={strict_base_hash}, base_sha256={base_sha256[:12]}...")
    return state


def _reset_model_to_unmerged_base(
    model: torch.nn.Module,
    state: HotswapState,
    calc_device: torch.device,
) -> None:
    """Step 1 of hotswap: restore model parameters to the un-merged base.

    Cache mode: copy from frozen snapshot (snapshot is never mutated).
    Re-load mode: load from disk + re-apply base_weights.

    Uses param.data.copy_() throughout so the compiled graph is unaffected.
    """
    if state.cache_in_ram:
        if state.cached_base_sd is None:
            raise RuntimeError("HotswapState in cache mode but cached_base_sd is None — bug")
        # COPY into model parameters; never mutate the snapshot in place
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in state.cached_base_sd:
                    src = state.cached_base_sd[name]
                    p.data.copy_(src.to(p.device, p.dtype))
    else:
        from safetensors.torch import load_file

        # Re-load DiT files from disk
        total_copied = 0
        for path in _expand_weight_paths(state.base_dit_paths):
            fresh_sd = load_file(path)
            total_copied += _copy_state_dict_to_model_parameters(model, fresh_sd, path)
            del fresh_sd
        logger.info(f"Hotswap: reset model from disk base ({total_copied} parameters copied)")

        # Re-apply --base_weights (permanent merge by design — see plan)
        if state.base_weights_paths:
            for i, bw_path in enumerate(state.base_weights_paths):
                bw_mult = (
                    state.base_weights_multipliers[i]
                    if state.base_weights_multipliers and i < len(state.base_weights_multipliers)
                    else 1.0
                )
                bw_sd = load_file(bw_path)
                _assert_standard_lora_only(bw_sd, bw_path)
                merge_nonlora_to_model(model, bw_sd, bw_mult, calc_device)
                del bw_sd


def hotswap_lora(
    model: torch.nn.Module,
    state: HotswapState,
    new_lora_paths: List[str],
    new_multipliers: Optional[List[float]] = None,
    calc_device: Optional[torch.device] = None,
) -> None:
    """Replace the currently-merged LoRAs with new ones, in place.

    Lifecycle (see plan for invariants):
      1. RESET    — fresh un-merged base into model.parameters()
      2. GUARD    — _assert_standard_lora_only on each new LoRA
      3. VALIDATE — ss_base_sha256 check vs state.base_sha256
      4. MERGE    — merge each new LoRA into model via merge_nonlora_to_model
      5. UPDATE   — bookkeeping in state.active_lora_*

    The compiled graph is unaffected because every write to model is
    via param.data.copy_() (directly in step 1, transitively in step 4
    through merge_nonlora_to_model's existing pattern).
    """
    from safetensors.torch import load_file

    if new_multipliers is None:
        new_multipliers = [1.0] * len(new_lora_paths)
    if len(new_multipliers) != len(new_lora_paths):
        raise ValueError(f"hotswap_lora: got {len(new_lora_paths)} LoRA paths but {len(new_multipliers)} multipliers")
    if calc_device is None:
        # Pick the device of the first parameter as a sensible default
        calc_device = next(model.parameters()).device

    # Step 2 (GUARD) and Step 3 (VALIDATE) — pre-load and check before mutating model
    new_sds: List[Dict[str, torch.Tensor]] = []
    for path in new_lora_paths:
        sd = load_file(path)
        _assert_standard_lora_only(sd, path)
        if state.base_sha256 is not None:
            metadata = _read_safetensors_metadata(path)
            _check_lora_base_hash(metadata, path, state.base_sha256, state.strict_base_hash)
        new_sds.append(sd)

    # Step 1 (RESET) — only proceed once all guards pass
    _reset_model_to_unmerged_base(model, state, calc_device)

    # Step 4 (MERGE) — apply new LoRAs in order
    for path, mult, sd in zip(new_lora_paths, new_multipliers, new_sds):
        logger.info(f"Hotswap: applying {path} with multiplier {mult}")
        if mult == 0:
            logger.info(f"Hotswap: skipping {path} because multiplier is 0")
            continue
        merge_nonlora_to_model(model, sd, mult, calc_device)

    # Step 5 (UPDATE) — bookkeeping
    state.active_lora_paths = list(new_lora_paths)
    state.active_lora_multipliers = list(new_multipliers)
    logger.info(f"Hotswap complete: {len(new_lora_paths)} LoRA(s) merged")
