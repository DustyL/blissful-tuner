#!/usr/bin/env python
"""Offline LoRA merge algebra CLI.

Combines standard LoRA safetensors files in materialized-delta space using
linear, TIES, DARE-linear, or DARE-TIES algebra, then writes a standard LoRA
adapter by SVD recompression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Callable, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musubi_tuner.utils.lora_utils import (
    _make_lora_name_from_model_key,
    convert_diffusers_if_needed,
    detect_network_type,
)
from musubi_tuner.utils.safetensors_utils import get_split_weight_filenames


METHODS = ("linear", "ties", "dare_linear", "dare_ties")
OUTPUT_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
SPECTRUM_RANKS = (8, 16, 32, 64, 128)
MATCH_SEMANTICS = "materialized_delta_v1"
RECOMPRESSION_SEMANTICS = "svd_v1"


# fp8 dtypes used by quantized base checkpoints. Detected via getattr so this stays
# robust if a future torch version drops one of the variants. Used by _assert_no_fp8_in_base
# to refuse fold-mode operations that would require de-quantize → add → re-quantize cycles.
_FP8_DTYPES: frozenset[torch.dtype] = frozenset(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
)

@dataclass(frozen=True)
class InputSpec:
    path: str
    weight: float


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    rank: int
    alpha: float
    down_shape: tuple[int, ...]
    up_shape: tuple[int, ...]
    use_rslora: bool = False


@dataclass
class AdapterInfo:
    spec: InputSpec
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, str]
    sha256: str
    modules: dict[str, ModuleInfo]
    use_rslora: bool = False


@dataclass(frozen=True)
class MergeConfig:
    method: str
    inputs: list[InputSpec]
    output: str | None
    output_rank: int | None
    output_alpha: float | None
    # output_dtype is None only when output_dtype_name == "base" (fold-mode sentinel).
    # LoRA-output mode always carries a concrete torch.dtype here.
    output_dtype: torch.dtype | None
    # One of {"fp32", "bf16", "fp16", "base"}. "base" is an internal sentinel for fold mode
    # only — preserves each base tensor's original dtype. It is NOT an argparse choice.
    output_dtype_name: str
    density: float | None
    drop_prob: float | None
    seed: int | None
    preview_spectrum: bool
    preview_per_module: bool
    prune_threshold: float
    output_use_rslora: bool
    # When set, switches from LoRA-output to fold-into-checkpoint mode. Mutually exclusive
    # with --output_rank / --output_alpha / --output_use_rslora and preview flags.
    fold_into: str | None

@dataclass
class MergeResult:
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, str]
    modules_processed: int
    modules_written: int
    spectrum_energy: dict[int, list[float]]
    per_module_energy: dict[str, dict[int, float]]

@dataclass
class MergedModuleDelta:
    """One module's merged delta in materialized-delta space (float32, CPU).

    Yielded by ``iter_merged_module_deltas`` for every module that has at least
    one contributing adapter. ``was_pruned=True`` indicates the merged delta is
    at or below ``--prune_threshold`` — still counted in ``modules_processed``
    accounting, but downstream consumers (SVD output, fold mode) must skip it.
    """

    module_name: str
    merged_delta: torch.Tensor
    was_pruned: bool = False

@dataclass(frozen=True)
class FoldTarget:
    """Resolved fold target: one LoRA module → one base tensor.

    Produced by :func:`resolve_fold_plan` after orphan / ambiguity / non-floating /
    shape checks pass. The fold pipeline reads ``base_key`` to look up the tensor in
    the loaded base state_dict, and uses ``base_dtype`` to decide whether to preserve
    per-tensor dtype (``output_dtype_name == "base"`` sentinel) or cast to a concrete
    output dtype.
    """

    lora_name: str
    base_key: str
    base_shape: tuple[int, ...]
    base_dtype: torch.dtype

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge standard LoRA safetensors using materialized-delta algebra and SVD recompression."
    )
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("PATH", "WEIGHT"),
        required=True,
        help="Repeat per input. Example: --input lora_a.safetensors 0.6",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output safetensors path. Required unless --preview_spectrum is set.",
    )
    parser.add_argument(
        "--output_rank",
        type=int,
        default=None,
        help="SVD recompression rank. Required when --output is set.",
    )
    parser.add_argument("--output_alpha", type=float, default=None, help="Output alpha. Defaults to --output_rank.")
    parser.add_argument(
        "--output_dtype",
        type=str,
        default=None,
        choices=tuple(OUTPUT_DTYPES),
        help=(
            "Output dtype. LoRA-output mode default: fp32. Fold mode (--fold_into) default: "
            "preserve per-tensor base dtype. Pass an explicit value to override either default."
        ),
    )
    parser.add_argument("--density", type=float, default=None, help="TIES trim density [0, 1]. Required for ties / dare_ties.")
    parser.add_argument("--drop_prob", type=float, default=None, help="DARE drop probability [0, 1). Required for dare_*.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed. Required for dare_*.")
    parser.add_argument(
        "--prune_threshold",
        type=float,
        default=0.0,
        help=(
            "Skip a merged module from the output when merged_delta.abs().max() <= prune_threshold. "
            "Default 0.0 preserves v1 exact-zero-only behavior. Prunes merged materialized deltas, "
            "not individual LoRA factors. Must be non-negative and finite."
        ),
    )
    parser.add_argument(
        "--output_use_rslora",
        action="store_true",
        help=(
            "Write output with rsLoRA scaling (alpha / sqrt(rank)) and use_rslora_flag=True. "
            "Use for downstream loaders that honor use_rslora_flag."
        ),
    )
    parser.add_argument(
        "--fold_into",
        type=str,
        default=None,
        help=(
            "Path to a base model safetensors checkpoint. Switches output mode from "
            "LoRA-adapter to full-checkpoint: the merged adapter delta is folded into the "
            "matching base tensors (full-rank, no SVD recompression) and written to --output. "
            "Mutually exclusive with --output_rank, --output_alpha, --output_use_rslora, and "
            "preview flags. Splits supported via the same shard-pattern rules as model loading."
        ),
    )
    parser.add_argument(
        "--preview_spectrum",
        action="store_true",
        help="Print aggregate singular-value energy at common ranks; do not write output.",
    )
    parser.add_argument(
        "--preview_per_module",
        action="store_true",
        help="With --preview_spectrum, print per-module spectrum rows.",
    )
    return parser

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> MergeConfig:
    fold_mode = bool(args.fold_into)

    # ----- Mode-specific output validation (must run before generic rules
    # so fold-mode rejections produce the more informative message) -----
    if fold_mode:
        if not args.output:
            raise ValueError("--fold_into requires --output (path for the folded checkpoint).")
        if args.output_rank is not None:
            raise ValueError(
                "--output_rank is not used with --fold_into; the folded checkpoint "
                "replaces base tensors in place at full rank (no SVD recompression)."
            )
        if args.output_alpha is not None:
            raise ValueError(
                "--output_alpha is not used with --fold_into; the folded checkpoint "
                "replaces base tensors in place at full rank (no SVD recompression)."
            )
        if args.output_use_rslora:
            raise ValueError(
                "--output_use_rslora is not used with --fold_into; rsLoRA scaling is a "
                "LoRA-output convention with no meaning for a folded checkpoint."
            )
        if args.preview_spectrum:
            raise ValueError(
                "--preview_spectrum is not used with --fold_into; SVD-energy previews "
                "report rank-selection statistics for LoRA recompression, which fold mode skips."
            )
        if args.preview_per_module:
            raise ValueError("--preview_per_module is not used with --fold_into.")
    else:
        if not args.output and not args.preview_spectrum:
            raise ValueError("--output is required unless --preview_spectrum is set.")
        if args.output and args.preview_spectrum:
            raise ValueError("--preview_spectrum is a dry-run mode; omit --output when previewing.")
        if args.output and args.output_rank is None:
            raise ValueError("--output_rank is required when --output is set.")
        if args.output_use_rslora and not args.output:
            raise ValueError("--output_use_rslora requires --output (no meaning in preview mode).")

    # ----- Generic numeric / dependency checks (apply to both modes;
    # in fold mode the relevant flags are already rejected above) -----
    if args.output_rank is not None and args.output_rank <= 0:
        raise ValueError("--output_rank must be a positive integer.")
    if args.output_alpha is not None and args.output_alpha <= 0:
        raise ValueError("--output_alpha must be positive.")
    if args.preview_per_module and not args.preview_spectrum:
        raise ValueError("--preview_per_module requires --preview_spectrum.")

    # ----- Method / DARE / TIES validation (unchanged) -----
    if args.method in {"ties", "dare_ties"} and args.density is None:
        raise ValueError(f"--density is required for --method {args.method}.")
    if args.method == "linear":
        for flag_name, value in (("--density", args.density), ("--drop_prob", args.drop_prob), ("--seed", args.seed)):
            if value is not None:
                raise ValueError(f"{flag_name} is not used with --method linear.")
    if args.method == "ties":
        for flag_name, value in (("--drop_prob", args.drop_prob), ("--seed", args.seed)):
            if value is not None:
                raise ValueError(f"{flag_name} is not used with --method ties.")
    if args.method == "dare_linear" and args.density is not None:
        raise ValueError("--density is not used with --method dare_linear.")
    if args.density is not None and not (0.0 <= args.density <= 1.0):
        raise ValueError("--density must be in [0, 1].")
    if args.method in {"dare_linear", "dare_ties"}:
        if args.drop_prob is None:
            raise ValueError(f"--drop_prob is required for --method {args.method}.")
        if args.seed is None:
            raise ValueError(f"--seed is required for --method {args.method}.")
    if args.drop_prob is not None and not (0.0 <= args.drop_prob < 1.0):
        raise ValueError("--drop_prob must be in [0, 1).")

    if not math.isfinite(args.prune_threshold) or args.prune_threshold < 0:
        raise ValueError("--prune_threshold must be non-negative and finite.")

    inputs: list[InputSpec] = []
    for path, raw_weight in args.input:
        weight = float(raw_weight)
        if not math.isfinite(weight):
            raise ValueError(f"Input weight for {os.path.basename(path)} must be finite, got {raw_weight!r}.")
        if weight < 0:
            warnings.warn(
                f"Input weight for {os.path.basename(path)} is negative ({weight}); proceeding with subtraction semantics.",
                stacklevel=2,
            )
        inputs.append(InputSpec(path=path, weight=weight))

    output_alpha = args.output_alpha
    if output_alpha is None and args.output_rank is not None:
        output_alpha = float(args.output_rank)

    # ----- Resolve output dtype with mode-aware defaults -----
    # LoRA mode: --output_dtype omitted → "fp32" (preserves v1 behavior).
    # Fold mode: --output_dtype omitted → "base" sentinel (preserve per-tensor base dtype).
    # Either mode: explicit --output_dtype fp32/bf16/fp16 → that concrete dtype.
    # "base" is never a user-facing argparse choice (would be accepted-but-ignored in LoRA mode).
    if args.output_dtype is None:
        output_dtype_name = "base" if fold_mode else "fp32"
    else:
        output_dtype_name = args.output_dtype
    output_dtype = None if output_dtype_name == "base" else OUTPUT_DTYPES[output_dtype_name]

    return MergeConfig(
        method=args.method,
        inputs=inputs,
        output=args.output,
        output_rank=args.output_rank,
        output_alpha=output_alpha,
        output_dtype=output_dtype,
        output_dtype_name=output_dtype_name,
        density=args.density,
        drop_prob=args.drop_prob,
        seed=args.seed,
        preview_spectrum=args.preview_spectrum,
        preview_per_module=args.preview_per_module,
        prune_threshold=args.prune_threshold,
        output_use_rslora=args.output_use_rslora,
        fold_into=args.fold_into,
    )

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Base-loading helpers for --fold_into mode (Tier 2 #5 v1.5 #3 step 3) ---

def _expand_fold_base_paths(path: str) -> list[str]:
    """Expand a base path into the concrete shard list (single file → ``[path]``).

    Wraps :func:`get_split_weight_filenames` to add fold-mode error context. The wrapped
    helper is shard-name agnostic — passing any shard of a split set (``...-00002-of-00004``)
    rebuilds the canonical 1..N sequence from the prefix and shard count. Missing shards
    raise :class:`FileNotFoundError` with a fold-specific hint.
    """
    try:
        split_paths = get_split_weight_filenames(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"--fold_into base shard missing: {e}. Pass any shard of the split set; "
            "the helper expands to all shards by index from the prefix."
        ) from e
    return [path] if split_paths is None else split_paths



def _load_base_as_stored(path: str) -> dict[str, torch.Tensor]:
    """Load a base safetensors checkpoint preserving on-disk dtypes.

    Returns a single merged state_dict on CPU. Tensors retain their stored dtype — no
    fp32 promotion. The fold writer is responsible for dtype handling per the
    ``output_dtype_name`` sentinel: ``"base"`` preserves per-tensor dtype, otherwise
    floating-point tensors are cast to the chosen concrete dtype.

    Hard-rejects malformed split bases where two shards declare the same tensor key
    (``dict.update`` would silently overwrite the earlier shard's tensor, losing the
    evidence before any downstream fold-plan check could see it).
    """
    state_dict: dict[str, torch.Tensor] = {}
    for shard in _expand_fold_base_paths(path):
        shard_sd = load_file(shard, device="cpu")
        overlap = set(state_dict).intersection(shard_sd)
        if overlap:
            sample = ", ".join(sorted(overlap)[:3])
            more = f" (+{len(overlap) - 3} more)" if len(overlap) > 3 else ""
            raise ValueError(
                f"--fold_into base has duplicate tensor keys across shards: {sample}{more}. "
                "Refusing to silently overwrite split checkpoint tensors. A well-formed split "
                "base must have disjoint key sets across shards."
            )
        state_dict.update(shard_sd)
    return state_dict



def _composite_base_hash(path: str) -> str:
    """Composite SHA-256 over base safetensors file bytes (single file or shards).

    For single-file bases this returns ``_file_sha256(path)`` directly so users can
    cross-check against ``sha256sum``. For multi-shard bases, returns

        sha256(hex_sha256(shard_1) || hex_sha256(shard_2) || ...)

    where ``hex_sha256(s)`` is the lowercase ASCII hex digest of shard ``s``'s file
    bytes and ``||`` is byte concatenation, in canonical numeric shard order. File-bytes
    semantics (not loaded tensors) means dtype/layout differences in deserialization do
    not change the provenance string. Using ASCII hex (not raw digest bytes) keeps the
    protocol trivially reproducible from a shell snippet.
    """
    shards = _expand_fold_base_paths(path)
    if len(shards) == 1:
        return _file_sha256(shards[0])
    h = hashlib.sha256()
    for shard in shards:
        h.update(_file_sha256(shard).encode("ascii"))
    return h.hexdigest()



def _assert_no_fp8_in_base(base_sd: dict[str, torch.Tensor], base_path: str) -> None:
    """Refuse fold-mode operations on fp8 base checkpoints.

    fp8 fold would require de-quantize → add delta → re-quantize, including calibration
    of new scale factors. That is out of scope for the merge CLI. The actionable
    resolution is to fold into the unquantized base, then re-quantize downstream.
    """
    fp8_keys = [k for k, t in base_sd.items() if t.dtype in _FP8_DTYPES]
    if not fp8_keys:
        return
    sample = ", ".join(f"{k} ({base_sd[k].dtype})" for k in fp8_keys[:3])
    more = f" (+{len(fp8_keys) - 3} more)" if len(fp8_keys) > 3 else ""
    raise ValueError(
        f"--fold_into base {os.path.basename(base_path)!r} contains fp8 tensors which fold "
        f"mode does not support: {sample}{more}. fp8 fold would require de-quantize → add → "
        "re-quantize with scale recalibration, which is out of the merge CLI's scope. Fold "
        "into the unquantized base, then re-quantize downstream."
    )

def build_base_lora_index(base_sd: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """Forward map: LoRA module name → list of base keys whose forward mapping yields it.

    Iterates only ``.weight`` keys (LoRA adapters target weight matrices). Uses the
    production helper :func:`_make_lora_name_from_model_key` for the forward mapping —
    NEVER inverts the underscore-separated LoRA name to guess a base key, because
    underscores can be either real characters in a base key or dot-replacements, and
    the inverse is genuinely ambiguous. The forward mapping is many-to-one, so values
    are ``list[str]`` (not ``str``) to preserve ambiguity evidence for
    :func:`resolve_fold_plan`.
    """
    index: dict[str, list[str]] = {}
    for key in base_sd:
        if not key.endswith(".weight"):
            continue
        lora_name = _make_lora_name_from_model_key(key)
        index.setdefault(lora_name, []).append(key)
    return index



def _delta_shape_from_module_info(info: ModuleInfo) -> tuple[int, ...]:
    """Compute the materialized merged-delta shape for a LoRA module from its factor shapes.

    Linear (down=2D, up=2D): ``(up.shape[0], down.shape[1])`` — i.e. ``(out_dim, in_dim)``.
    Conv2d (down=4D, up=4D): ``(up.shape[0], down.shape[1], down.shape[2], down.shape[3])``
    — i.e. ``(out_dim, in_dim, kernel_h, kernel_w)``. Requires ``up.shape[2:] == (1, 1)``,
    mirroring the production materializer's constraint (see :func:`materialize_module_delta`).

    Used by :func:`resolve_fold_plan` for preflight shape validation against the base
    tensor — before any tensor-heavy materialization runs.
    """
    if len(info.down_shape) == 2 and len(info.up_shape) == 2:
        return (info.up_shape[0], info.down_shape[1])
    if len(info.down_shape) == 4 and len(info.up_shape) == 4:
        if info.up_shape[2:] != (1, 1):
            # Mirror the materializer guard so preflight catches unsupported geometry rather
            # than letting it through to fail mid-pipeline. Same error wording as
            # materialize_module_delta for consistency.
            raise ValueError(
                f"{info.name} has unsupported Conv2d LoRA up kernel {tuple(info.up_shape[2:])}; expected 1x1."
            )
        return (info.up_shape[0], info.down_shape[1], info.down_shape[2], info.down_shape[3])
    raise ValueError(f"{info.name} mixes Linear and Conv2d LoRA tensor shapes, which fold mode cannot resolve.")



def resolve_fold_plan(
    adapters: list[AdapterInfo],
    base_sd: dict[str, torch.Tensor],
) -> dict[str, FoldTarget]:
    """Preflight: resolve every LoRA module in the adapter union to a unique base tensor.

    Hard-rejects (any one fires before any tensor-heavy materialization runs):

    * **Orphan** — LoRA module has no matching base key under the forward index.
    * **Ambiguous** — multiple base keys forward-map to the same LoRA name.
    * **Non-floating** — matched base tensor has integer / bool / quantized dtype.
    * **Shape mismatch** — would-be merged delta shape ≠ base tensor shape.

    Returns a dict keyed by LoRA module name in :func:`_module_union` order
    (deterministic). Cross-adapter shape consistency is the responsibility of
    :func:`_validate_module_shapes_and_output_rank`; this function uses the first
    contributing adapter's :class:`ModuleInfo` to derive the would-be delta shape.
    """
    index = build_base_lora_index(base_sd)
    plan: dict[str, FoldTarget] = {}

    for module_name in _module_union(adapters):
        candidates = index.get(module_name, [])
        if not candidates:
            raise ValueError(
                f"--fold_into orphan: LoRA module {module_name!r} has no matching base tensor. "
                "Verify the base checkpoint matches the architecture the adapter was trained against, "
                "or remove the unmatched module from the adapter."
            )
        if len(candidates) > 1:
            sample = ", ".join(sorted(candidates)[:3])
            more = f" (+{len(candidates) - 3} more)" if len(candidates) > 3 else ""
            raise ValueError(
                f"--fold_into ambiguity: LoRA module {module_name!r} forward-maps to multiple base "
                f"keys: {sample}{more}. Refusing to guess. Rename or remove the duplicate base keys "
                "before folding."
            )

        base_key = candidates[0]
        base_tensor = base_sd[base_key]

        if not base_tensor.is_floating_point():
            raise ValueError(
                f"--fold_into non-floating target: LoRA module {module_name!r} matched base tensor "
                f"{base_key!r} with non-floating dtype {base_tensor.dtype}. Fold mode requires a "
                "floating-point base for delta addition."
            )

        delta_shape: tuple[int, ...] | None = None
        for adapter in adapters:
            info = adapter.modules.get(module_name)
            if info is not None:
                delta_shape = _delta_shape_from_module_info(info)
                break
        # _module_union only emits names that have at least one contributor, so this assertion
        # is structurally guaranteed; documented here to make the invariant explicit.
        assert delta_shape is not None, "_module_union returned a name with no contributing adapter"

        if tuple(base_tensor.shape) != delta_shape:
            raise ValueError(
                f"--fold_into shape mismatch: LoRA module {module_name!r} would produce a delta of "
                f"shape {delta_shape}, but base tensor {base_key!r} has shape {tuple(base_tensor.shape)}. "
                "The adapter was trained against a different layer geometry than this base provides."
            )

        plan[module_name] = FoldTarget(
            lora_name=module_name,
            base_key=base_key,
            base_shape=tuple(base_tensor.shape),
            base_dtype=base_tensor.dtype,
        )

    return plan

def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _bool_tensor_value(value: torch.Tensor | bool | int | float) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def _has_true_flag(sd: dict[str, torch.Tensor], *keys: str) -> bool:
    for key in keys:
        if key in sd and _bool_tensor_value(sd[key]):
            return True
    return False


def _format_rejection_message(path: str, detected_type: str) -> str:
    basename = os.path.basename(path)
    if detected_type == "dora":
        return (
            "DoRA adapter merge algebra is not supported in v1 because DoRA delta materialization "
            "depends on the base model weights. Use standard LoRA inputs, or merge/apply this DoRA "
            f"adapter through the existing runtime merge path.\n\nRejected input: {basename}"
        )
    if detected_type == "split_dims":
        return (
            "Split-dims LoRA adapter merge algebra is not supported in v1. Use standard "
            f"lora_down.weight / lora_up.weight adapters.\n\nRejected input: {basename}"
        )
    return (
        f"Adapter merge algebra rejected {basename}: detected network type {detected_type!r}. "
        "v1 supports standard LoRA only. Use the existing runtime merge path for this adapter "
        "family, or convert to standard LoRA first."
    )


def _assert_supported_adapter(sd: dict[str, torch.Tensor], path: str) -> None:
    has_dora = _has_true_flag(sd, "use_dora_flag") or any(key.endswith(".dora_layer.weight") for key in sd)
    if has_dora:
        raise ValueError(_format_rejection_message(path, "dora"))

    net_type = detect_network_type(sd)
    if net_type != "lora":
        raise ValueError(_format_rejection_message(path, net_type))

    split_keys = [key for key in sd if re.search(r"\.lora_(down|up)\.\d+\.weight$", key)]
    if split_keys:
        raise ValueError(_format_rejection_message(path, "split_dims"))


def _alpha_value(sd: dict[str, torch.Tensor], module_name: str, rank: int) -> float:
    alpha = sd.get(f"{module_name}.alpha")
    if alpha is None:
        return float(rank)
    if isinstance(alpha, torch.Tensor):
        return float(alpha.item())
    return float(alpha)


def _collect_modules(sd: dict[str, torch.Tensor], path: str) -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    use_rslora = _has_true_flag(sd, "use_rslora_flag")

    for key, down in sd.items():
        if not key.endswith(".lora_down.weight"):
            continue
        module_name = key[: -len(".lora_down.weight")]
        up_key = f"{module_name}.lora_up.weight"
        if up_key not in sd:
            raise ValueError(f"{os.path.basename(path)} has {key} but missing {up_key}.")
        up = sd[up_key]
        if down.ndim not in (2, 4) or up.ndim not in (2, 4):
            raise ValueError(f"{os.path.basename(path)} module {module_name} has unsupported LoRA tensor rank.")
        rank = int(down.shape[0])
        if rank <= 0:
            raise ValueError(f"{os.path.basename(path)} module {module_name} has non-positive rank {rank}.")
        if int(up.shape[1]) != rank:
            raise ValueError(
                f"{os.path.basename(path)} module {module_name} has mismatched rank: "
                f"down shape {tuple(down.shape)}, up shape {tuple(up.shape)}."
            )
        module_use_rslora = use_rslora or _has_true_flag(sd, f"{module_name}.use_rslora_flag")
        modules[module_name] = ModuleInfo(
            name=module_name,
            rank=rank,
            alpha=_alpha_value(sd, module_name, rank),
            down_shape=tuple(down.shape),
            up_shape=tuple(up.shape),
            use_rslora=module_use_rslora,
        )

    if not modules:
        raise ValueError(f"{os.path.basename(path)} contains no standard LoRA modules.")
    return modules


def load_adapter(path: str, weight: float) -> AdapterInfo:
    with safe_open(path, framework="pt") as f:
        metadata = dict(f.metadata() or {})
    original_sd = load_file(path, device="cpu")
    sd = convert_diffusers_if_needed(original_sd)
    _assert_supported_adapter(sd, path)
    modules = _collect_modules(sd, path)
    return AdapterInfo(
        spec=InputSpec(path=path, weight=weight),
        state_dict=sd,
        metadata=metadata,
        sha256=_file_sha256(path),
        modules=modules,
        use_rslora=_has_true_flag(sd, "use_rslora_flag"),
    )


def load_adapters(inputs: Iterable[InputSpec]) -> list[AdapterInfo]:
    return [load_adapter(spec.path, spec.weight) for spec in inputs]


def materialize_module_delta(adapter: AdapterInfo, module_name: str) -> torch.Tensor | None:
    info = adapter.modules.get(module_name)
    if info is None:
        return None

    sd = adapter.state_dict
    down = sd[f"{module_name}.lora_down.weight"].to(device="cpu", dtype=torch.float32)
    up = sd[f"{module_name}.lora_up.weight"].to(device="cpu", dtype=torch.float32)
    scale = info.alpha / math.sqrt(info.rank) if info.use_rslora else info.alpha / info.rank

    if down.ndim == 2 and up.ndim == 2:
        delta = up @ down
    elif down.ndim == 4 and up.ndim == 4:
        if up.shape[2:] != (1, 1):
            raise ValueError(f"{module_name} has unsupported Conv2d LoRA up kernel {tuple(up.shape[2:])}; expected 1x1.")
        if down.shape[2:] == (1, 1):
            delta = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            delta = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
    else:
        raise ValueError(f"{module_name} mixes Linear and Conv2d LoRA tensor shapes, which v1 cannot merge.")

    return delta * scale


def _prune_magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    k = int(density * tensor.numel())
    if k <= 0:
        return torch.zeros_like(tensor)
    mask = torch.zeros(tensor.numel(), dtype=torch.bool, device=tensor.device)
    topk = torch.topk(tensor.abs().reshape(-1), k=k, largest=True).indices
    mask[topk] = True
    return tensor * mask.reshape_as(tensor)


def _prune_random(tensor: torch.Tensor, keep_prob: float, generator: torch.Generator) -> torch.Tensor:
    if keep_prob >= 1:
        return tensor
    if keep_prob <= 0:
        raise ValueError("DARE keep probability must be > 0.")
    probs = torch.full_like(tensor, keep_prob)
    mask = torch.bernoulli(probs, generator=generator)
    return tensor * mask / keep_prob


def _subseed_for_adapter_module(base_seed: int, adapter_sha256: str, module_name: str) -> int:
    payload = f"{base_seed}|{adapter_sha256}|{module_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _generator_from_seed(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _reshape_weights(task_tensors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return weights.view(weights.shape + (1,) * (task_tensors.dim() - weights.dim()))


def _majority_sign_mask(task_tensors: torch.Tensor) -> torch.Tensor:
    sign = task_tensors.sign()
    sign_magnitude = task_tensors.sum(dim=0)
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def _disjoint_merge(weighted_task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
    mixed = (weighted_task_tensors * majority_sign_mask).sum(dim=0)
    preserved = majority_sign_mask.sum(dim=0)
    return mixed / torch.clamp(preserved, min=1.0)


def _linear(task_tensors: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack(task_tensors, dim=0)
    return (stacked * _reshape_weights(stacked, weights)).sum(dim=0)


def _ties(task_tensors: list[torch.Tensor], weights: torch.Tensor, density: float) -> torch.Tensor:
    pruned = [_prune_magnitude(tensor, density) for tensor in task_tensors]
    stacked = torch.stack(pruned, dim=0)
    majority_sign_mask = _majority_sign_mask(stacked)
    weighted = stacked * _reshape_weights(stacked, weights)
    return _disjoint_merge(weighted, majority_sign_mask)


def combine_deltas(
    method: str,
    deltas: list[torch.Tensor],
    weights: list[float],
    *,
    density: float | None = None,
    drop_prob: float | None = None,
    generator: torch.Generator | None = None,
    random_generators: list[torch.Generator] | None = None,
) -> torch.Tensor:
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if method == "linear":
        return _linear(deltas, weight_tensor)
    if method == "ties":
        assert density is not None
        return _ties(deltas, weight_tensor, density)

    assert drop_prob is not None
    keep_prob = 1.0 - drop_prob
    if random_generators is None:
        assert generator is not None
        random_generators = [generator] * len(deltas)
    if len(random_generators) != len(deltas):
        raise ValueError("random_generators must match deltas length.")
    pruned = [_prune_random(delta, keep_prob, gen) for delta, gen in zip(deltas, random_generators)]
    if method == "dare_linear":
        return _linear(pruned, weight_tensor)
    if method == "dare_ties":
        assert density is not None
        return _ties(pruned, weight_tensor, density)
    raise ValueError(f"Unknown merge method {method!r}.")


def _flatten_delta_for_svd(delta: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    if delta.ndim == 2:
        return delta, None
    if delta.ndim == 4:
        return delta.flatten(start_dim=1), tuple(delta.shape)
    raise ValueError(f"Unsupported merged delta shape {tuple(delta.shape)}.")


def svd_recompress_delta(
    delta: torch.Tensor,
    module_name: str,
    output_rank: int,
    output_alpha: float,
    output_dtype: torch.dtype,
    output_use_rslora: bool = False,
) -> dict[str, torch.Tensor]:
    matrix, conv_shape = _flatten_delta_for_svd(delta.to(dtype=torch.float32, device="cpu"))
    if output_rank > min(matrix.shape):
        raise ValueError(
            f"--output_rank {output_rank} is too large for {module_name} with merged delta shape "
            f"{tuple(delta.shape)} (max useful rank {min(matrix.shape)})."
        )

    # rsLoRA convention: scale = alpha / sqrt(rank). Standard convention: scale = alpha / rank.
    # Either way, exact merged-delta reconstruction is preserved because the SVD factors
    # are reconstructed using the selected scale (root = sqrt(s / scale)).
    denom = math.sqrt(output_rank) if output_use_rslora else float(output_rank)
    scale = output_alpha / denom
    if scale <= 0:
        raise ValueError("Output LoRA scale must be positive.")

    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    u = u[:, :output_rank]
    s = s[:output_rank]
    vh = vh[:output_rank, :]
    root = torch.sqrt(s / scale)
    up_2d = u * root.unsqueeze(0)
    down_2d = root.unsqueeze(1) * vh

    if conv_shape is None:
        up = up_2d
        down = down_2d
    else:
        out_dim, in_dim, kernel_h, kernel_w = conv_shape
        up = up_2d.reshape(out_dim, output_rank, 1, 1)
        down = down_2d.reshape(output_rank, in_dim, kernel_h, kernel_w)

    return {
        f"{module_name}.lora_down.weight": down.to(output_dtype).contiguous(),
        f"{module_name}.lora_up.weight": up.to(output_dtype).contiguous(),
        f"{module_name}.alpha": torch.tensor(float(output_alpha), dtype=torch.float32),
    }


def _singular_energy(delta: torch.Tensor, ranks: Iterable[int] = SPECTRUM_RANKS) -> dict[int, float]:
    matrix, _ = _flatten_delta_for_svd(delta.to(dtype=torch.float32, device="cpu"))
    singular_values = torch.linalg.svdvals(matrix)
    total = torch.sum(singular_values.square())
    if total.item() == 0:
        return {rank: 1.0 for rank in ranks}
    return {
        rank: float(torch.sum(singular_values[: min(rank, singular_values.numel())].square()).item() / total.item())
        for rank in ranks
    }


def _matrix_shape_from_module_info(info: ModuleInfo) -> tuple[int, int]:
    if len(info.down_shape) == 2 and len(info.up_shape) == 2:
        return info.up_shape[0], info.down_shape[1]
    if len(info.down_shape) == 4 and len(info.up_shape) == 4:
        if info.up_shape[2:] != (1, 1):
            raise ValueError(f"{info.name} has unsupported Conv2d LoRA up kernel {info.up_shape[2:]}; expected 1x1.")
        return info.up_shape[0], info.down_shape[1] * info.down_shape[2] * info.down_shape[3]
    raise ValueError(f"{info.name} mixes Linear and Conv2d LoRA tensor shapes, which v1 cannot merge.")


def _validate_module_shapes_and_output_rank(config: MergeConfig, adapters: list[AdapterInfo]) -> None:
    for module_name in _module_union(adapters):
        ref_matrix_shape: tuple[int, int] | None = None
        ref_delta_shape: tuple[int, ...] | None = None
        for adapter in adapters:
            info = adapter.modules.get(module_name)
            if info is None:
                continue
            matrix_shape = _matrix_shape_from_module_info(info)
            delta_shape = (
                (matrix_shape[0], matrix_shape[1]) if len(info.down_shape) == 2 else (info.up_shape[0], *info.down_shape[1:])
            )
            if ref_matrix_shape is None:
                ref_matrix_shape = matrix_shape
                ref_delta_shape = delta_shape
            elif matrix_shape != ref_matrix_shape or delta_shape != ref_delta_shape:
                raise ValueError(
                    f"Shape mismatch for module {module_name}: expected delta shape {ref_delta_shape}, "
                    f"but {os.path.basename(adapter.spec.path)} would produce {delta_shape}."
                )
        if config.output_rank is not None and ref_matrix_shape is not None and config.output_rank > min(ref_matrix_shape):
            raise ValueError(
                f"--output_rank {config.output_rank} is too large for {module_name} with merged matrix shape "
                f"{ref_matrix_shape} (max useful rank {min(ref_matrix_shape)})."
            )


def _merge_inputs_metadata(adapters: list[AdapterInfo]) -> str:
    items = []
    for adapter in adapters:
        ranks = sorted({info.rank for info in adapter.modules.values()})
        alphas = sorted({float(info.alpha) for info in adapter.modules.values()})
        items.append(
            {
                "basename": os.path.basename(adapter.spec.path),
                "sha256": adapter.sha256,
                "weight": adapter.spec.weight,
                "rank": ranks[0] if len(ranks) == 1 else ranks,
                "alpha": alphas[0] if len(alphas) == 1 else alphas,
            }
        )
    return json.dumps(items, sort_keys=True, separators=(",", ":"))


def build_metadata(config: MergeConfig, adapters: list[AdapterInfo]) -> dict[str, str]:
    return {
        "ss_merge_tool": "blissful-tuner",
        "ss_merge_tool_version": _git_version(),
        "ss_merge_method": config.method,
        "ss_merge_output_format": "lora",
        "ss_merge_output_rank": "" if config.output_rank is None else str(config.output_rank),
        "ss_merge_output_alpha": "" if config.output_alpha is None else str(config.output_alpha),
        "ss_merge_output_dtype": config.output_dtype_name,
        "ss_merge_density": "" if config.density is None else str(config.density),
        "ss_merge_drop_prob": "" if config.drop_prob is None else str(config.drop_prob),
        "ss_merge_seed": "" if config.seed is None else str(config.seed),
        "ss_merge_prune_threshold": str(config.prune_threshold),
        "ss_merge_output_use_rslora": str(config.output_use_rslora).lower(),
        "ss_merge_inputs": _merge_inputs_metadata(adapters),
        "ss_merge_input_count": str(len(adapters)),
        "ss_merge_match_semantics": MATCH_SEMANTICS,
        "ss_merge_recompression": RECOMPRESSION_SEMANTICS,
        "ss_merge_rejects_dora": "true",
    }


def _module_union(adapters: list[AdapterInfo]) -> list[str]:
    return sorted({name for adapter in adapters for name in adapter.modules})


def iter_merged_module_deltas(
    config: MergeConfig,
    adapters: list[AdapterInfo],
    *,
    module_callback: Callable[[str], None] | None = None,
) -> Iterator[MergedModuleDelta]:
    """Yield merged-delta results per module in module-union order.

    For each module that has at least one contributing adapter, yields a
    :class:`MergedModuleDelta` carrying the materialized merged delta in float32
    on CPU. Modules below ``--prune_threshold`` are still yielded (with
    ``was_pruned=True``) so callers can preserve v1 ``modules_processed``
    accounting; consumers (SVD output, fold mode) must skip pruned items
    themselves. Modules with no contributing adapters are skipped entirely.

    Raises ``ValueError`` if any module produces a non-finite merged delta or a
    shape mismatch across adapters.
    """
    for module_name in _module_union(adapters):
        if module_callback is not None:
            module_callback(module_name)

        deltas: list[torch.Tensor] = []
        ref_shape: tuple[int, ...] | None = None
        for adapter in adapters:
            delta = materialize_module_delta(adapter, module_name)
            if delta is None:
                if ref_shape is None:
                    deltas.append(delta)  # type: ignore[arg-type]
                else:
                    deltas.append(torch.zeros(ref_shape, dtype=torch.float32))
                continue
            if ref_shape is None:
                ref_shape = tuple(delta.shape)
                deltas = [torch.zeros(ref_shape, dtype=torch.float32) if d is None else d for d in deltas]
            elif tuple(delta.shape) != ref_shape:
                raise ValueError(
                    f"Shape mismatch for module {module_name}: expected delta shape {ref_shape}, "
                    f"but {os.path.basename(adapter.spec.path)} produced {tuple(delta.shape)}."
                )
            deltas.append(delta)

        if ref_shape is None:
            continue

        merged_delta = combine_deltas(
            config.method,
            deltas,  # type: ignore[arg-type]
            [adapter.spec.weight for adapter in adapters],
            density=config.density,
            drop_prob=config.drop_prob,
            random_generators=[
                _generator_from_seed(_subseed_for_adapter_module(config.seed, adapter.sha256, module_name)) for adapter in adapters
            ]
            if config.seed is not None
            else None,
        )
        if not torch.isfinite(merged_delta).all():
            raise ValueError(
                f"Non-finite merged delta for module {module_name}; check input LoRA weights and merge method args."
            )
        # At default --prune_threshold 0.0, this is byte-equivalent to the v1
        # exact-zero check (since |x| <= 0 requires x == 0). Larger thresholds
        # skip near-zero modules — see Tier 2 #5 v1.5 #1 in the plan doc.
        was_pruned = merged_delta.abs().max().item() <= config.prune_threshold
        yield MergedModuleDelta(module_name=module_name, merged_delta=merged_delta, was_pruned=was_pruned)
        # Drop generator-local references so the previous iteration's per-adapter deltas
        # are freed before the next module's module_callback fires. The yielded
        # MergedModuleDelta still holds merged_delta via the consumer; the consumer's
        # next loop step releases it. See test_weakref_probe_confirms_previous_module_deltas_are_released.
        del deltas, delta, merged_delta

def merge_adapters(
    config: MergeConfig,
    adapters: list[AdapterInfo] | None = None,
    *,
    module_callback: Callable[[str], None] | None = None,
) -> MergeResult:
    adapters = load_adapters(config.inputs) if adapters is None else adapters
    _validate_module_shapes_and_output_rank(config, adapters)
    output_sd: dict[str, torch.Tensor] = {}
    spectrum_energy: dict[int, list[float]] = {rank: [] for rank in SPECTRUM_RANKS}
    per_module_energy: dict[str, dict[int, float]] = {}

    modules_written = 0
    modules_processed = 0
    for item in iter_merged_module_deltas(config, adapters, module_callback=module_callback):
        modules_processed += 1
        if item.was_pruned:
            continue

        energy = _singular_energy(item.merged_delta)
        per_module_energy[item.module_name] = energy
        for rank, value in energy.items():
            spectrum_energy[rank].append(value)

        if config.output is not None:
            assert config.output_rank is not None
            assert config.output_alpha is not None
            output_sd.update(
                svd_recompress_delta(
                    item.merged_delta,
                    item.module_name,
                    config.output_rank,
                    config.output_alpha,
                    config.output_dtype,
                    output_use_rslora=config.output_use_rslora,
                )
            )
            modules_written += 1

    # rsLoRA output: write the global use_rslora_flag tensor only when at least one module
    # was actually written. Empty/all-pruned outputs would otherwise contain just a lone
    # flag tensor with no module data, which is technically valid but misleading to inspect.
    # See Tier 2 #5 v1.5 #2 plan, locked decision #5.
    if config.output_use_rslora and modules_written > 0:
        output_sd["use_rslora_flag"] = torch.tensor(True, dtype=torch.bool)

    return MergeResult(
        state_dict=output_sd,
        metadata=build_metadata(config, adapters),
        modules_processed=modules_processed,
        modules_written=modules_written,
        spectrum_energy=spectrum_energy,
        per_module_energy=per_module_energy,
    )

def _format_percent(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def print_spectrum_preview(config: MergeConfig, result: MergeResult) -> None:
    inputs = ", ".join(f"{os.path.basename(spec.path)}:{spec.weight:g}" for spec in config.inputs)
    method_args = []
    if config.density is not None:
        method_args.append(f"density={config.density:g}")
    if config.drop_prob is not None:
        method_args.append(f"drop_prob={config.drop_prob:g}")
    if config.seed is not None:
        method_args.append(f"seed={config.seed}")
    # Pruning happens before the spectrum is computed, so the threshold shapes
    # the rank-selection stats. Echo it in the preview header so a pasted
    # transcript records the knob that produced the numbers.
    if config.prune_threshold > 0:
        method_args.append(f"prune_threshold={config.prune_threshold:g}")
    method_suffix = f" ({', '.join(method_args)})" if method_args else ""

    print("Tier 2 #5 spectrum preview")
    print(f"Method: {config.method}{method_suffix}")
    print(f"Inputs: {len(config.inputs)} ({inputs})")
    print(f"Modules processed: {result.modules_processed}")
    print()
    print("Aggregate energy captured (mean / median / p95) at candidate ranks:")
    for rank in SPECTRUM_RANKS:
        values = result.spectrum_energy[rank]
        if not values:
            mean = median = p95 = 0.0
        else:
            tensor = torch.tensor(values, dtype=torch.float32)
            mean = float(tensor.mean().item())
            median = float(torch.quantile(tensor, 0.5).item())
            p95 = float(torch.quantile(tensor, 0.95).item())
        print(f"  rank={rank:3d}   {_format_percent(mean)} / {_format_percent(median)} / {_format_percent(p95)}")

    if config.preview_per_module:
        print()
        print("Per-module energy captured:")
        for module_name, energy in result.per_module_energy.items():
            parts = " ".join(f"r{rank}={_format_percent(energy[rank])}" for rank in SPECTRUM_RANKS)
            print(f"  {module_name}: {parts}")

    print()
    print("Run again with --output_rank N --output PATH to write a LoRA at the chosen rank.")


def run(config: MergeConfig) -> MergeResult:
    if config.fold_into is not None:
        # Parser/config/validation surface for --fold_into is present; fold execution lands later.
        raise NotImplementedError("--fold_into execution is not yet implemented")
    result = merge_adapters(config)
    if config.preview_spectrum:
        print_spectrum_preview(config, result)
        return result
    assert config.output is not None
    save_file(result.state_dict, config.output, metadata=result.metadata)
    if result.modules_processed > 0 and result.modules_written == 0:
        if config.prune_threshold > 0:
            print(
                f"Warning: all merged modules were exact-zero or below --prune_threshold {config.prune_threshold}; "
                "output safetensors contains only metadata."
            )
        else:
            print("Warning: all merged modules were exact-zero; output safetensors contains only metadata.")
    print(
        f"Saved merged LoRA to {config.output} "
        f"({result.modules_written}/{result.modules_processed} modules written, dtype={config.output_dtype_name})."
    )
    return result

def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        config = validate_args(args)
        run(config)
    except ValueError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
