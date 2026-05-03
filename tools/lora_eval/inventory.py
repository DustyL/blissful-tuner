"""Inventory + lineage grouping for LoRA evaluation.

Walks a directory tree of `.safetensors` LoRA files, parses each filename to
extract its lineage (training run identity), step number, noise variant
(WAN2.2 high/low pair), EMA tag, and named tag. Pairs WAN2.2 high/low noise
files into single variants. Reads each safetensors header to surface training
metadata (`ss_*` keys) without loading tensor data. Emits a JSON manifest the
GPU eval harness consumes downstream.

Designed for WAN2.2 T2V LoRAs first (the dominant disk consumer) but tolerant
of single-noise architectures (Qwen-Image, FLUX.2) — single-file variants are
emitted as `is_paired=false`.

Usage:
    python tools/lora_eval/inventory.py \\
        --root ~/SwarmUI/Models/Lora/loras \\
        --output ~/eval_runs/wan22_inventory.json
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TRIGGERS: tuple[str, ...] = (
    "DLAY",
    "ELLE",
    "OLVA",
    "KYLA",
    "SIENNA",
    "SYDNY",
    "CRLN",
    "ALXXA",
    "CARLEN",
    "HASLEY",
)

# Subset of safetensors __metadata__ keys we surface in the manifest. These are
# what musubi-tuner / blissful-tuner / kohya_ss-style trainers typically write.
# Missing keys are tolerated — different trainers emit different subsets.
INTERESTING_METADATA_KEYS: tuple[str, ...] = (
    "ss_base_model_version",
    "ss_network_module",
    "ss_network_dim",
    "ss_network_alpha",
    "ss_network_args",
    "ss_steps",
    "ss_max_train_steps",
    "ss_epoch",
    "ss_learning_rate",
    "ss_optimizer",
    "ss_mixed_precision",
    "ss_training_started_at",
    "ss_training_finished_at",
    "ss_dataset_dirs",
    "ss_sd_scripts_commit_hash",
    "ss_lokr_factor",
    "ss_seed",
)


@dataclass
class ParsedName:
    """Result of parsing a single LoRA filename."""

    raw_filename: str
    lineage_basename: str
    step: int | None
    noise: str | None  # "high", "low", or None (single-file / not WAN dual)
    ema_tag: str | None  # everything after `_ema_` if present, else None; empty string for bare `_ema`
    named_tag: str | None  # trailing `-<tag>` like "-13K-steps" or "-4k-steps-old" or "-final"
    is_final: bool  # True if a non-numeric "final" marker was peeled
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class FileEntry:
    """One safetensors file plus parsed identity, size, and (optionally) header metadata."""

    path: str  # absolute path
    filename: str
    source_root: str  # absolute path of the --root this file was found under
    source_root_label: str  # short label (last path component of source_root)
    parent_dirname: str  # immediate parent directory name relative to source_root, or "" if at root
    size_bytes: int
    parsed: ParsedName
    trigger: str
    metadata: dict[str, Any]


@dataclass
class Variant:
    """One evaluatable unit within a lineage.

    For WAN2.2 dual-noise architectures, a variant pairs (high, low). For
    single-file architectures, only `single_file` is set. The `*_path` fields
    hold absolute paths and are the canonical handle the GPU eval runner uses.
    The `*_file` fields are filename-only for display and remain stable for
    downstream tools that key on basename.
    """

    step: int | None
    ema_tag: str | None
    named_tag: str | None
    is_final: bool
    is_paired: bool  # True if BOTH high and low present (WAN2.2)
    high_file: str | None  # filename only (display)
    low_file: str | None
    single_file: str | None
    high_path: str | None  # absolute path on disk (canonical handle)
    low_path: str | None
    single_path: str | None
    size_bytes: int  # sum of files in this variant
    metadata: dict[str, Any]  # taken from the high-noise file (or single)


@dataclass
class Lineage:
    """A training run identity. One row in the manifest = one training run."""

    name: str
    source_root: str  # absolute path of the --root this lineage's files were found under
    source_root_label: str  # short label (last path component of source_root)
    trigger: str
    total_size_bytes: int
    variant_count: int
    variants: list[Variant]


@dataclass
class Orphan:
    """A file that could not be cleanly grouped, with reason.

    For `unpaired_noise` orphans, `unpaired_info` carries the structured fields
    (lineage, root, step, present_channel, missing_channel) needed by the
    fallback-pairing post-pass. If a fallback partner is found, `fallback_partner`
    holds the chosen opposite-noise file from a nearby step; the GPU runner can
    use this to evaluate the orphan with a substitute pair.
    """

    file: str
    path: str
    size_bytes: int
    reason: str
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    unpaired_info: dict[str, Any] | None = None
    fallback_partner: dict[str, Any] | None = None


# ─── Filename parsing ───────────────────────────────────────────────────────
#
# Filenames in the wild (from /home/dustin/SwarmUI/Models/Lora/loras/):
#   crln_persona_lora.safetensors
#   crln_persona_lora_ema.safetensors
#   crln_persona_lora_ema_015.safetensors
#   ELLE_wan22_h200_character_high_noise-final.safetensors
#   crln_wan22_h200_character_high_noise-4k-steps-old.safetensors
#   WAN2.2-LoRA-V3_000006000_high_noise.safetensors
#   WAN2.2-LoRA-V3_high_noise.safetensors
#   dlay_wan22_persona_lokr_prodigy_sf_fullmatrix_factor16_ab-v1-step00007000.safetensors
#   dlay_man_HIGH_noise_lokr_v3-musubi-step00009750.safetensors    ← "HIGH_noise" is part of basename
#   crln_wan22_persona_lora_..._masked-experimental_ema_beta0.90_last999.safetensors
#
# Strategy: peel suffixes from the right in this order:
#   1. EMA suffix:        `_ema(_<tag>)?$`  → captures ema_tag (None if no `_ema`, empty str if bare `_ema`)
#   2. Final/named tag:   `-(final|<tag>)$` → captures named_tag, sets is_final if "final"
#   3. Noise suffix:      `_(high|low)_noise$`   → captures noise type
#   4. Step suffix:       `_<digits>$` (>= 4 digits) OR `[-_]step<digits>$` OR `[-_]Step-<digits>$`
#
# The order matters because some files have multiple suffixes (e.g.
# `..._high_noise-final` peels final first, then noise). We retry-step
# AFTER noise because the step number can be on either side of noise in
# practice (`_NNNNNNNNN_high_noise` vs. `_high_noise-step00006000` — only
# the first has been observed but both are tolerated).

_RE_EMA = re.compile(r"_ema(?:[_-](?P<tag>[A-Za-z0-9._-]+))?$")
# Noise suffix accepts both underscore (kohya: `_high_noise`) and dash
# (DiffusionPipe-style dirname: `wan22-high-noise`) separators. End-anchored so
# `dlay_man_HIGH_noise_lokr_v3-musubi` (where `HIGH_noise` is mid-string in the
# user's deliberate basename) is correctly NOT peeled.
_RE_NOISE = re.compile(r"[-_](?P<noise>high|low)[-_]noise$", re.IGNORECASE)
_RE_STEP_NUMERIC = re.compile(r"_(?P<step>\d{4,9})$")
_RE_STEP_PREFIXED = re.compile(r"[-_]step(?P<step>\d+)$", re.IGNORECASE)
_RE_STEP_COSMOS = re.compile(r"[-_]Step[-_](?P<step>\d+)$")
_RE_STEP_K = re.compile(r"-(?P<step>\d+)(?P<k>[Kk])-?steps?$")  # matches `-13K-steps`, `-4k-steps`
_RE_FINAL = re.compile(r"-final$", re.IGNORECASE)
# We deliberately do NOT peel arbitrary trailing `-<tag>` suffixes from the
# basename. That heuristic is too aggressive: `WAN2.2-LoRA-V3` would get
# mangled to lineage `WAN2.2` + tag `LoRA-V3`, splitting one training run into
# many singleton lineages. Instead, the only "tag" we peel is the canonical
# `-final` marker (above) and the K-style step shorthand (below). Anything
# else stays in the lineage basename, and the post-pass near-duplicate
# detector surfaces lineages that look related so the user can decide.


def parse_filename(filename: str) -> ParsedName:
    """Strip suffixes from the right to recover the lineage basename and variant identity.

    Returns a ParsedName with `parse_warnings` populated for any oddities
    encountered. Idempotent: passing the same filename twice yields the same
    result.
    """
    if not filename.endswith(".safetensors"):
        return ParsedName(
            raw_filename=filename,
            lineage_basename=filename,
            step=None,
            noise=None,
            ema_tag=None,
            named_tag=None,
            is_final=False,
            parse_warnings=["filename did not end with .safetensors"],
        )

    name = filename[: -len(".safetensors")]
    warnings: list[str] = []

    # Peel EMA suffix first — it's always at the very end if present.
    ema_tag: str | None = None
    if m := _RE_EMA.search(name):
        ema_tag = m.group("tag") or ""  # empty string => bare `_ema`
        name = name[: m.start()]

    # Peel `-final` marker.
    is_final = False
    if _RE_FINAL.search(name):
        is_final = True
        name = name[: -len("-final")]

    # Peel noise suffix. Note: this only matches `_high_noise`/`_low_noise`
    # at the END. The basename `dlay_man_HIGH_noise_lokr_v3-musubi` contains
    # `HIGH_noise` mid-string and is therefore correctly NOT peeled.
    noise: str | None = None
    if m := _RE_NOISE.search(name):
        noise = m.group("noise").lower()
        name = name[: m.start()]

    # Peel step suffix. Try numeric, then -step, then -Step-, then -<N>K-steps.
    step: int | None = None
    for pat in (_RE_STEP_NUMERIC, _RE_STEP_PREFIXED, _RE_STEP_COSMOS, _RE_STEP_K):
        if m := pat.search(name):
            try:
                raw = int(m.group("step"))
            except ValueError:
                warnings.append(f"could not parse step number from match: {m.group(0)!r}")
                step = None
            else:
                # K-shorthand multiplies by 1000 (`-13K-steps` => 13000).
                step = raw * 1000 if pat is _RE_STEP_K else raw
                name = name[: m.start()]
            break

    # If we didn't find a step yet, also check for a noise suffix that we may
    # have missed because the order was step-then-noise. (Belt and suspenders.)
    if noise is None:
        if m := _RE_NOISE.search(name):
            noise = m.group("noise").lower()
            name = name[: m.start()]

    # named_tag is intentionally always None in this version — see the comment
    # above the regex block for why we don't peel arbitrary `-<tag>` suffixes.
    # The field is retained in the data model for forward compatibility with
    # downstream consumers that may bring back tagged variants.
    named_tag: str | None = None

    if not name:
        warnings.append("nothing left after suffix peeling — entire name was suffix-like")
        name = filename[: -len(".safetensors")]

    return ParsedName(
        raw_filename=filename,
        lineage_basename=name,
        step=step,
        noise=noise,
        ema_tag=ema_tag,
        named_tag=named_tag,
        is_final=is_final,
        parse_warnings=warnings,
    )


# ─── DiffusionPipe path-based parsing ───────────────────────────────────────
#
# Some LoRAs come from the DiffusionPipe trainer, which stores its outputs as:
#   <lineage_dir>/[<timestamp>/]epoch<N>/adapter_model.safetensors
# The lineage identity is in the directory name; the step is the epoch number;
# the filename is always `adapter_model.safetensors`. We synthesize the
# equivalent kohya-style fields here so the downstream grouping/pairing logic
# (which lives in `parse_filename`) doesn't need to know about either format.
#
# Some DiffusionPipe outputs put the epoch in the lineage dir name itself
# (`wan22-low-noise-epoch300/adapter_model.safetensors`) — no nested epoch dir.
# We handle that case by peeling `-epoch<N>$` from the lineage dir name.

_DIFFUSION_PIPE_FILENAME = "adapter_model.safetensors"
_RE_EPOCH_DIR = re.compile(r"^epoch(?P<n>\d+)$", re.IGNORECASE)
_RE_TIMESTAMP_DIR = re.compile(r"^\d{8}_\d{2}-\d{2}-\d{2}$")
_RE_EPOCH_IN_DIRNAME = re.compile(r"-epoch(?P<n>\d+)$", re.IGNORECASE)


def parse_diffusion_pipe_path(path: Path, root: Path, *, merge_timestamps: bool = False) -> ParsedName | None:
    """Parse a DiffusionPipe-style path into a ParsedName, or return None if
    the path doesn't match the convention.

    Recognized layouts:
        <lineage_dir>/epoch<N>/adapter_model.safetensors
        <lineage_dir>/<timestamp>/epoch<N>/adapter_model.safetensors
        <lineage_dir>-epoch<N>/adapter_model.safetensors

    The synthesized lineage_basename is the dir name with any `_high_noise` /
    `_low_noise` suffix peeled (via the same noise regex used for kohya names).
    If a timestamp directory is present, the default behavior appends it as
    `@<timestamp>` so concurrent runs of the same config stay as separate
    lineages. Pass `merge_timestamps=True` to drop the suffix — this lets a
    high-noise run at one timestamp pair with a low-noise run at another (a
    common DiffusionPipe pattern where the user trains the two halves
    sequentially in separate jobs).

    Known future risk: outer "campaign" dirs above the lineage_dir are
    intentionally ignored. If two separate campaigns ever produce the same
    lineage_dir name (e.g., two different `wan22-high-noise-epoch150/` trees
    under different outer parents), they will silently merge. Currently the
    user has only one such campaign so this is fine, but if a second campaign
    is added later, the cross-root or parent-collision logic must be extended
    to walk further up the path. The unit tests pin the current behavior; a
    new fixture added at that point will catch the merge regression.
    """
    if path.name != _DIFFUSION_PIPE_FILENAME:
        return None

    try:
        rel_parts = path.relative_to(root).parts[:-1]  # exclude the filename
    except ValueError:
        return None
    if not rel_parts:
        return None  # adapter_model.safetensors at root has no lineage info

    parts = list(rel_parts)
    step: int | None = None
    timestamp: str | None = None

    if m := _RE_EPOCH_DIR.match(parts[-1]):
        step = int(m.group("n"))
        parts.pop()

    if step is not None and parts and _RE_TIMESTAMP_DIR.match(parts[-1]):
        timestamp = parts.pop()

    if not parts:
        return None  # path was just epoch/timestamp dirs with no lineage

    lineage_dir = parts[-1]

    # Handle the epoch-baked-into-dirname variant.
    if step is None:
        if m := _RE_EPOCH_IN_DIRNAME.search(lineage_dir):
            step = int(m.group("n"))
            lineage_dir = lineage_dir[: m.start()]

    if step is None:
        return None  # path looks DiffusionPipe-shaped but we couldn't extract a step

    # Apply the kohya-style suffix peeler to the lineage dir name so we pick up
    # any `_high_noise` / `_low_noise` / `_ema` / `-final` / etc. suffixes the
    # user may have added.
    sub_parsed = parse_filename(lineage_dir + ".safetensors")
    final_lineage = sub_parsed.lineage_basename
    if timestamp and not merge_timestamps:
        final_lineage = f"{final_lineage}@{timestamp}"

    warnings = ["diffusion_pipe_path"]
    if sub_parsed.parse_warnings:
        warnings.extend(sub_parsed.parse_warnings)

    return ParsedName(
        raw_filename=path.name,
        lineage_basename=final_lineage,
        step=step,
        noise=sub_parsed.noise,
        ema_tag=sub_parsed.ema_tag,
        named_tag=sub_parsed.named_tag,
        is_final=sub_parsed.is_final,
        parse_warnings=warnings,
    )


# ─── Trigger word detection ─────────────────────────────────────────────────


def detect_trigger(filename: str, triggers: tuple[str, ...], default: str) -> str:
    """Return the first matching trigger token from `triggers`, else `default`.

    Match is case-insensitive and word-boundaried (preceded by start, `_`, or
    `-`; followed by `_`, `-`, `.`, or end-of-string). Order in `triggers`
    determines priority on overlap.
    """
    upper = filename.upper()
    for token in triggers:
        # Word boundary: not surrounded by alphanumeric on either side.
        pattern = rf"(?:^|[_\-]){re.escape(token.upper())}(?:[_\-.]|$)"
        if re.search(pattern, upper):
            return token
    return default


# ─── Safetensors header reader ──────────────────────────────────────────────


def read_safetensors_metadata(path: Path) -> dict[str, Any]:
    """Read just the `__metadata__` dict from a safetensors file's JSON header.

    Does NOT load any tensor data. Reads ~kilobytes per file. Returns an empty
    dict if the file has no metadata or the header is unparseable.
    """
    try:
        with path.open("rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                return {}
            (header_len,) = struct.unpack("<Q", header_len_bytes)
            # Defensive cap — a legitimate metadata header is far smaller than
            # the file itself; if we see 100MB we're looking at corruption or
            # a non-safetensors file.
            if header_len > 100 * 1024 * 1024:
                return {}
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                return {}
            header = json.loads(header_bytes.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    raw_meta = header.get("__metadata__", {})
    return {k: raw_meta[k] for k in INTERESTING_METADATA_KEYS if k in raw_meta}


# ─── Lineage grouping + variant pairing ─────────────────────────────────────


def _variant_key(parsed: ParsedName) -> tuple[int | None, str | None, str | None, bool]:
    """Identity tuple for a variant within a lineage. Files with matching
    keys (and differing only in noise) get paired into one variant."""
    return (parsed.step, parsed.ema_tag, parsed.named_tag, parsed.is_final)


def _variant_sort_key(v: Variant) -> tuple[int, int, str, str, int]:
    """Order: numeric steps first (ascending), then EMA variants, then named
    tags, then final markers. Within each bucket, sort by step / tag string.
    The leading int discriminates buckets so e.g. step=100 sorts before
    ema_tag='beta0.90'."""
    if v.step is not None and v.ema_tag is None and v.named_tag is None and not v.is_final:
        bucket = 0
    elif v.ema_tag is not None:
        bucket = 1
    elif v.named_tag is not None:
        bucket = 2
    elif v.is_final:
        bucket = 3
    else:
        bucket = 4
    return (bucket, v.step if v.step is not None else 0, v.ema_tag or "", v.named_tag or "", int(v.is_final))


def _resolve_effective_lineage_names(entries: list[FileEntry]) -> dict[int, str]:
    """For non-DiffusionPipe entries whose parsed `lineage_basename` collides
    across distinct parent directories within the same source_root, return an
    override mapping `id(entry) -> effective_name`. The override uses the
    parent dir name as the lineage label, which is the user's authoritative
    organization unit. DiffusionPipe entries already encode parent context in
    their lineage_basename and are skipped here.

    Files at root in a collision (parent_dirname == "") keep the parsed
    basename as their effective name — the root-level files are the
    "canonical" copies and the subdir copies get the discriminating label.
    """
    # Build (basename, source_root) -> set of parent_dirnames
    parents_by_key: dict[tuple[str, str], set[str]] = {}
    for e in entries:
        if "diffusion_pipe_path" in e.parsed.parse_warnings:
            continue
        key = (e.parsed.lineage_basename, e.source_root)
        parents_by_key.setdefault(key, set()).add(e.parent_dirname)

    overrides: dict[int, str] = {}
    for (basename, source_root), parents in parents_by_key.items():
        if len(parents) <= 1:
            continue  # no collision
        for e in entries:
            if (
                e.parsed.lineage_basename == basename
                and e.source_root == source_root
                and "diffusion_pipe_path" not in e.parsed.parse_warnings
                and e.parent_dirname  # keep root-level files using the bare basename
            ):
                overrides[id(e)] = e.parent_dirname
    return overrides


def group_into_lineages(entries: list[FileEntry]) -> tuple[list[Lineage], list[Orphan]]:
    """Build lineages + orphans from parsed file entries.

    A lineage is keyed by `(effective_lineage_name, source_root)`. Files from
    different `--root`s are kept as separate lineages (cross-root duplicates
    are surfaced separately for review). Files in different subdirectories of
    the same root that parse to the same basename are split via parent-dir
    discrimination — see `_resolve_effective_lineage_names`. Within a lineage,
    files with matching `_variant_key` are paired by noise channel.
    Unpairable or duplicate files become orphans.
    """
    overrides = _resolve_effective_lineage_names(entries)

    def effective_name(e: FileEntry) -> str:
        return overrides.get(id(e), e.parsed.lineage_basename)

    by_lineage: dict[tuple[str, str], list[FileEntry]] = {}
    for entry in entries:
        key = (effective_name(entry), entry.source_root)
        by_lineage.setdefault(key, []).append(entry)

    lineages: list[Lineage] = []
    orphans: list[Orphan] = []

    for (lineage_name, source_root), lineage_entries in by_lineage.items():
        source_root_label = lineage_entries[0].source_root_label
        # First pass: bucket entries by variant key + noise.
        slots: dict[tuple[int | None, str | None, str | None, bool], dict[str, list[FileEntry]]] = {}
        for entry in lineage_entries:
            key = _variant_key(entry.parsed)
            slots.setdefault(key, {"high": [], "low": [], "single": []})
            channel = entry.parsed.noise if entry.parsed.noise else "single"
            slots[key][channel].append(entry)

        variants: list[Variant] = []
        for key, channels in slots.items():
            step, ema_tag, named_tag, is_final = key
            high_files = channels["high"]
            low_files = channels["low"]
            single_files = channels["single"]

            # Detect duplicates within the same channel — same lineage, same
            # variant key, same noise = duplicate file (probably a copy).
            for ch, files in (("high", high_files), ("low", low_files), ("single", single_files)):
                if len(files) > 1:
                    for dup in files[1:]:
                        orphans.append(
                            Orphan(
                                file=dup.filename,
                                path=dup.path,
                                size_bytes=dup.size_bytes,
                                reason="duplicate_in_lineage",
                                detail=f"lineage={lineage_name} root={source_root_label} key={key} channel={ch}",
                                metadata=dup.metadata,
                            )
                        )

            # Pick first file per channel for variant assembly.
            high = high_files[0] if high_files else None
            low = low_files[0] if low_files else None
            single = single_files[0] if single_files else None

            # Three legal cases:
            #   1. paired (high + low, no single)
            #   2. single (single, no high/low) — for non-WAN architectures
            #   3. unpaired (high without low or low without high) — orphan
            if single and not high and not low:
                variants.append(
                    Variant(
                        step=step,
                        ema_tag=ema_tag,
                        named_tag=named_tag,
                        is_final=is_final,
                        is_paired=False,
                        high_file=None,
                        low_file=None,
                        single_file=single.filename,
                        high_path=None,
                        low_path=None,
                        single_path=single.path,
                        size_bytes=single.size_bytes,
                        metadata=single.metadata,
                    )
                )
            elif high and low and not single:
                variants.append(
                    Variant(
                        step=step,
                        ema_tag=ema_tag,
                        named_tag=named_tag,
                        is_final=is_final,
                        is_paired=True,
                        high_file=high.filename,
                        low_file=low.filename,
                        single_file=None,
                        high_path=high.path,
                        low_path=low.path,
                        single_path=None,
                        size_bytes=high.size_bytes + low.size_bytes,
                        metadata=high.metadata,
                    )
                )
            else:
                # Mixed (single AND high/low) or unpaired noise — flag every file.
                missing = []
                if high and not low:
                    missing.append("low")
                if low and not high:
                    missing.append("high")
                reason = "unpaired_noise" if missing else "mixed_single_and_paired"
                detail = f"lineage={lineage_name} root={source_root_label} key={key}"
                if missing:
                    detail += f" missing={','.join(missing)}"
                for entry in (*high_files, *low_files, *single_files):
                    unpaired_info: dict[str, Any] | None = None
                    if reason == "unpaired_noise" and entry.parsed.noise:
                        present = entry.parsed.noise
                        missing_channel = "low" if present == "high" else "high"
                        unpaired_info = {
                            "lineage": lineage_name,
                            "source_root": source_root,
                            "source_root_label": source_root_label,
                            "step": step,
                            "ema_tag": ema_tag,
                            "named_tag": named_tag,
                            "is_final": is_final,
                            "present_channel": present,
                            "missing_channel": missing_channel,
                        }
                    orphans.append(
                        Orphan(
                            file=entry.filename,
                            path=entry.path,
                            size_bytes=entry.size_bytes,
                            reason=reason,
                            detail=detail,
                            metadata=entry.metadata,
                            unpaired_info=unpaired_info,
                        )
                    )

        if not variants:
            continue

        # Promote degenerate `adapter_model` lineages (bare DiffusionPipe files
        # at root with no path/name context) to orphans. These have no trigger
        # word, no step, no noise — nothing for the GPU eval pipeline to act
        # on. Surfacing them as a "lineage" produces a useless contact-sheet
        # row downstream.
        if (
            lineage_name == "adapter_model"
            and len(variants) == 1
            and variants[0].step is None
            and not variants[0].is_paired
            and variants[0].ema_tag is None
            and not variants[0].is_final
        ):
            v = variants[0]
            file = v.high_file or v.low_file or v.single_file or "<unknown>"
            full_path = v.high_path or v.low_path or v.single_path or file
            orphans.append(
                Orphan(
                    file=file,
                    path=full_path,
                    size_bytes=v.size_bytes,
                    reason="no_lineage_info",
                    detail=f"bare adapter_model.safetensors with no path or filename context (root={source_root_label})",
                    metadata=v.metadata,
                )
            )
            continue

        variants.sort(key=_variant_sort_key)
        total_size = sum(v.size_bytes for v in variants)

        # Trigger: pick the most common trigger across this lineage's entries
        # (in case some files were detected as different triggers — usually
        # they should agree).
        triggers_seen: dict[str, int] = {}
        for entry in lineage_entries:
            triggers_seen[entry.trigger] = triggers_seen.get(entry.trigger, 0) + 1
        most_common_trigger = max(triggers_seen, key=lambda t: triggers_seen[t])

        lineages.append(
            Lineage(
                name=lineage_name,
                source_root=source_root,
                source_root_label=source_root_label,
                trigger=most_common_trigger,
                total_size_bytes=total_size,
                variant_count=len(variants),
                variants=variants,
            )
        )

    lineages.sort(key=lambda lin: -lin.total_size_bytes)
    return lineages, orphans


# ─── Walking + entry construction ───────────────────────────────────────────


def collect_entries(
    roots: list[Path],
    *,
    triggers: tuple[str, ...],
    default_trigger: str,
    read_metadata: bool,
    merge_diffusion_pipe_timestamps: bool = False,
) -> list[FileEntry]:
    """Walk each root in `roots`, parse every `*.safetensors` file, optionally
    read header metadata. Each entry is tagged with its source_root and its
    immediate parent directory name (relative to that root) so downstream
    grouping can disambiguate collisions across subdirectories and roots."""
    entries: list[FileEntry] = []
    for root in roots:
        root_label = root.name or str(root)
        for path in sorted(root.rglob("*.safetensors")):
            try:
                size = path.stat().st_size
            except OSError as e:
                print(f"WARNING: could not stat {path}: {e}", file=sys.stderr)
                continue
            # Try DiffusionPipe path-based parsing first; fall back to filename parsing.
            parsed = parse_diffusion_pipe_path(path, root, merge_timestamps=merge_diffusion_pipe_timestamps) or parse_filename(
                path.name
            )
            # Determine the immediate parent dir relative to this root. Empty
            # string means the file is at root level.
            try:
                rel_parts = path.relative_to(root).parts
                parent_dirname = rel_parts[-2] if len(rel_parts) >= 2 else ""
            except ValueError:
                parent_dirname = ""
            # For trigger detection, use the lineage_basename (which captures the
            # dir name in the DiffusionPipe case and the filename otherwise) plus
            # the original filename as a fallback. This catches triggers embedded
            # in dir names that the filename alone wouldn't expose.
            trigger_input = parsed.lineage_basename + " " + path.name
            trigger = detect_trigger(trigger_input, triggers, default_trigger)
            metadata = read_safetensors_metadata(path) if read_metadata else {}
            entries.append(
                FileEntry(
                    path=str(path),
                    filename=path.name,
                    source_root=str(root),
                    source_root_label=root_label,
                    parent_dirname=parent_dirname,
                    size_bytes=size,
                    parsed=parsed,
                    trigger=trigger,
                    metadata=metadata,
                )
            )
    return entries


# ─── Output ─────────────────────────────────────────────────────────────────


def _humansize(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


def build_manifest(
    roots: list[Path],
    triggers: tuple[str, ...],
    default_trigger: str,
    lineages: list[Lineage],
    orphans: list[Orphan],
) -> dict[str, Any]:
    total_files = sum(len([f for f in (v.high_file, v.low_file, v.single_file) if f]) for lin in lineages for v in lin.variants)
    total_files += len(orphans)
    total_size = sum(lin.total_size_bytes for lin in lineages) + sum(o.size_bytes for o in orphans)
    return {
        "scope_roots": [{"path": str(r), "label": r.name or str(r)} for r in roots],
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "trigger_words": list(triggers),
        "default_trigger": default_trigger,
        "summary": {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_human": _humansize(total_size),
            "lineages_count": len(lineages),
            "variants_count": sum(lin.variant_count for lin in lineages),
            "orphans_count": len(orphans),
        },
        "lineages": [
            {
                **{k: v for k, v in asdict(lin).items() if k != "variants"},
                "total_size_human": _humansize(lin.total_size_bytes),
                "variants": [asdict(v) for v in lin.variants],
            }
            for lin in lineages
        ],
        "orphans": [asdict(o) for o in orphans],
    }


def attach_fallback_partners(
    lineages: list[Lineage],
    orphans: list[Orphan],
    *,
    max_step_distance: int | None = None,
) -> int:
    """For each `unpaired_noise` orphan, find the nearest opposite-noise file in
    the same lineage (by step distance) and attach it as `fallback_partner`.

    The GPU runner can use this to evaluate the orphan with a substitute pair
    instead of skipping it entirely. Useful when the user has partial cleanup
    history (e.g., a `_low_noise_3000` survives but its `_high_noise_3000`
    sibling was already deleted; we can pair it with a nearby paired step like
    `_high_noise_3500`). Pairing is cross-step but same-lineage, so it's still
    semantically meaningful.

    Pulls candidates from PAIRED variants of the same lineage (which have both
    high and low channels available) plus other unpaired orphans of the same
    lineage with the opposite channel. Returns the number of orphans for which
    a fallback partner was attached.

    `max_step_distance=None` means no distance limit. Set to a positive int to
    skip fallbacks that would jump too far from the orphan's step.
    """
    # Index lineages by (name, source_root) so we can look up a lineage by an
    # orphan's unpaired_info fields.
    lineage_index = {(lin.name, lin.source_root): lin for lin in lineages}
    # Group orphans by (lineage, source_root) so we can also use other orphans
    # as candidate partners (e.g., two unpaired orphans of opposite channels at
    # different steps can pair each other).
    orphans_by_lineage: dict[tuple[str, str], list[Orphan]] = {}
    for o in orphans:
        if o.unpaired_info:
            key = (o.unpaired_info["lineage"], o.unpaired_info["source_root"])
            orphans_by_lineage.setdefault(key, []).append(o)

    attached = 0
    for orphan in orphans:
        info = orphan.unpaired_info
        if not info or info.get("step") is None:
            continue
        target_channel = info["missing_channel"]
        target_step = info["step"]
        lineage = lineage_index.get((info["lineage"], info["source_root"]))

        candidates: list[tuple[int, str, str, str]] = []  # (step, path, source, file)

        # Candidates from paired variants (have both channels).
        if lineage:
            for v in lineage.variants:
                if v.step is None or not v.is_paired:
                    continue
                path = v.high_path if target_channel == "high" else v.low_path
                fname = v.high_file if target_channel == "high" else v.low_file
                if path and fname:
                    candidates.append((v.step, path, "paired_variant", fname))

        # Candidates from other unpaired orphans of opposite channel.
        for other in orphans_by_lineage.get((info["lineage"], info["source_root"]), []):
            if other is orphan:
                continue
            other_info = other.unpaired_info
            if not other_info or other_info.get("step") is None:
                continue
            if other_info["present_channel"] == target_channel:
                candidates.append((other_info["step"], other.path, "unpaired_orphan", other.file))

        if not candidates:
            continue

        # Pick nearest by step distance.
        nearest = min(candidates, key=lambda c: abs(c[0] - target_step))
        step, path, source, fname = nearest
        distance = abs(step - target_step)
        if max_step_distance is not None and distance > max_step_distance:
            continue

        orphan.fallback_partner = {
            "path": path,
            "file": fname,
            "step": step,
            "step_distance": distance,
            "channel": target_channel,
            "source": source,  # "paired_variant" or "unpaired_orphan"
        }
        attached += 1

    return attached


def find_cross_root_duplicates(lineages: list[Lineage]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Find lineages whose `name` appears under more than one source_root.

    Returns a list of (basename, [(label, source_root_path), ...]) tuples for
    each name that has more than one root. The user reviews these to decide if
    one copy can be deleted, or if the two roots are intentionally separate
    backups/forks.
    """
    by_name: dict[str, list[tuple[str, str]]] = {}
    for lin in lineages:
        by_name.setdefault(lin.name, []).append((lin.source_root_label, lin.source_root))
    return [(name, sorted(set(roots))) for name, roots in by_name.items() if len({r[1] for r in roots}) > 1]


def find_related_lineages(lineages: list[Lineage]) -> list[tuple[str, list[str]]]:
    """Find groups of lineages whose names share a common prefix.

    A "group" is a longer-named lineage whose first N characters match a
    shorter-named lineage's full name (with a `_` or `-` boundary at the join).
    Surfaces cases where the user has e.g. `crln_wan22_h200_character` (with
    many step variants) plus `crln_wan22_h200_character_high_noise-4k-steps-old`
    (a singleton with a tag the parser couldn't safely strip). The user reviews
    these and tells the tool whether to merge.
    """
    # Dedupe names across roots — `find_cross_root_duplicates` handles the
    # "same name in multiple roots" case separately, so this detector only
    # cares about NAME relationships and shouldn't echo each name once per root.
    names_by_length = sorted({lin.name for lin in lineages}, key=len)
    groups: dict[str, list[str]] = {}
    for i, short in enumerate(names_by_length):
        for longer in names_by_length[i + 1 :]:
            if longer == short:
                continue
            if longer.startswith(short) and longer[len(short)] in ("_", "-"):
                groups.setdefault(short, []).append(longer)
    # Convert to sorted list of (anchor, [related]) tuples.
    return sorted(((anchor, sorted(rest)) for anchor, rest in groups.items()), key=lambda t: -len(t[1]))


def find_dash_noise_siblings(lineages: list[Lineage]) -> list[tuple[str, list[str]]]:
    """Find lineages that differ only by a `high` ↔ `low` substring.

    Catches cases like `wan22-high-noise` and `wan22-low-noise` (separate
    lineages because the parser doesn't peel dash-style noise) that should
    obviously be A/B compared. The canonical form replaces the matched
    substring with `|N|` so siblings collapse to the same key.

    The prefix-based `find_related_lineages` misses these because neither name
    is a prefix of the other.
    """
    by_canonical: dict[str, set[str]] = {}
    for lin in lineages:
        canonical = re.sub(r"[-_](?:high|low)[-_]", "|N|", lin.name, flags=re.IGNORECASE)
        if canonical != lin.name:
            by_canonical.setdefault(canonical, set()).add(lin.name)
    return [(canonical, sorted(names)) for canonical, names in by_canonical.items() if len(names) > 1]


def print_human_summary(lineages: list[Lineage], orphans: list[Orphan]) -> None:
    print()
    print("─── Inventory summary ──────────────────────────────────────────────")
    total_size = sum(lin.total_size_bytes for lin in lineages) + sum(o.size_bytes for o in orphans)
    print(f"  Lineages: {len(lineages)}")
    print(f"  Variants: {sum(lin.variant_count for lin in lineages)}")
    print(f"  Orphans:  {len(orphans)}")
    print(f"  Total disk: {_humansize(total_size)}")
    print()

    if lineages:
        # Show source_root_label only when more than one root is in play.
        roots_in_play = {lin.source_root_label for lin in lineages}
        show_root = len(roots_in_play) > 1
        print("─── Top 10 lineages by disk size (deletion ROI) ────────────────────")
        for lin in lineages[:10]:
            root_col = f"  [{lin.source_root_label}]" if show_root else ""
            print(
                f"  {_humansize(lin.total_size_bytes):>10}  {lin.variant_count:>3} variants   {lin.trigger:>7}{root_col}   {lin.name}"
            )
        print()

    if orphans:
        print("─── Orphans (review manually) ──────────────────────────────────────")
        by_reason: dict[str, list[Orphan]] = {}
        for o in orphans:
            by_reason.setdefault(o.reason, []).append(o)
        for reason, group in sorted(by_reason.items()):
            print(f"  [{reason}] {len(group)} file(s)")
            for o in group[:5]:
                detail = f" — {o.detail}" if o.detail else ""
                print(f"      {o.file}{detail}")
            if len(group) > 5:
                print(f"      ... and {len(group) - 5} more")
        print()

    related = find_related_lineages(lineages)
    if related:
        print("─── Possibly-related lineages (review for manual merge) ────────────")
        print("  These lineages share a common prefix and may be the same training")
        print("  run with non-canonical name suffixes the parser couldn't strip.")
        print("  If they should be merged, rename the files OR tell the tool author.")
        print()
        for anchor, rest in related[:10]:
            print(f"  anchor:  {anchor}")
            for r in rest:
                print(f"    related: {r}")
            print()
        if len(related) > 10:
            print(f"  ... and {len(related) - 10} more anchor lineages with relatives")
            print()

    dash_siblings = find_dash_noise_siblings(lineages)
    if dash_siblings:
        print("─── Likely high/low noise siblings (dash-style) ────────────────────")
        print("  These lineages differ only in their `high`/`low` substring, which")
        print("  suggests they're the two halves of a WAN2.2 dual-noise training run")
        print("  using dash-style naming the parser doesn't auto-merge. Each pair")
        print("  should typically be evaluated together.")
        print()
        for canonical, names in dash_siblings[:10]:
            print(f"  canonical: {canonical}")
            for n in names:
                print(f"    sibling: {n}")
            print()

    cross_root = find_cross_root_duplicates(lineages)
    if cross_root:
        print("─── Cross-root duplicate lineages (review for redundant copies) ────")
        print("  The same lineage name appears under more than one --root. These")
        print("  may be intentional copies/backups OR redundant copies eligible")
        print("  for deletion. The eval pipeline runs each independently.")
        print()
        for name, root_pairs in cross_root[:15]:
            labels = ", ".join(label for label, _ in root_pairs)
            print(f"  {name}  →  {labels}")
        if len(cross_root) > 15:
            print(f"  ... and {len(cross_root) - 15} more cross-root duplicates")
        print()

    fallback_attached = [o for o in orphans if o.fallback_partner]
    if fallback_attached:
        print("─── Fallback-paired orphans (cross-step nearest-neighbor pairing) ──")
        print("  These unpaired-noise orphans have a substitute partner from a")
        print("  nearby step in the same lineage. The GPU runner can use the")
        print("  fallback to evaluate them; results will be noted as `[fallback]`")
        print("  in the contact sheet so you know it's not a same-step pair.")
        print()
        for o in fallback_attached[:20]:
            fp = o.fallback_partner
            assert fp is not None  # for type checker; we filtered above
            ui = o.unpaired_info or {}
            print(f"  {o.file}")
            print(
                f"    step={ui.get('step')} {ui.get('present_channel')} → "
                f"fallback step={fp['step']} {fp['channel']} (Δ={fp['step_distance']}, from {fp['source']})"
            )
        if len(fallback_attached) > 20:
            print(f"  ... and {len(fallback_attached) - 20} more")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory + group LoRA safetensors files into lineages for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories to scan recursively. Each --root contributes a separate "
        "namespace — same-named lineages found under different roots stay as distinct lineages "
        "and are surfaced as cross-root duplicates for review.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to write manifest.json.")
    parser.add_argument(
        "--triggers",
        type=str,
        default=",".join(DEFAULT_TRIGGERS),
        help="Comma-separated list of trigger word tokens to detect in filenames (case-insensitive match).",
    )
    parser.add_argument(
        "--default-trigger",
        type=str,
        default="DLAY",
        help="Trigger to assign when no token from --triggers is found in a filename.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip safetensors header parsing. Faster, but the manifest will not include training metadata.",
    )
    parser.add_argument(
        "--merge-diffusion-pipe-timestamps",
        action="store_true",
        help="When set, drop the `@<timestamp>` discriminator from DiffusionPipe lineage names so "
        "that high/low noise halves trained in separate jobs (different timestamps) merge into one "
        "lineage. Default: off (timestamps kept distinct).",
    )
    parser.add_argument(
        "--max-fallback-step-distance",
        type=int,
        default=None,
        help="For unpaired-noise orphans, attach a `fallback_partner` from the nearest opposite-noise "
        "file in the same lineage (within this step distance). Default: no limit. Set to 0 to disable.",
    )
    args = parser.parse_args()

    roots: list[Path] = []
    for r in args.root:
        resolved = r.expanduser().resolve()
        if not resolved.is_dir():
            print(f"ERROR: --root is not a directory: {resolved}", file=sys.stderr)
            return 2
        roots.append(resolved)

    output: Path = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    triggers = tuple(t.strip() for t in args.triggers.split(",") if t.strip())
    default_trigger = args.default_trigger.strip()
    if default_trigger not in triggers:
        triggers = (*triggers, default_trigger)

    for r in roots:
        print(f"Scanning {r} ...")
    entries = collect_entries(
        roots,
        triggers=triggers,
        default_trigger=default_trigger,
        read_metadata=not args.no_metadata,
        merge_diffusion_pipe_timestamps=args.merge_diffusion_pipe_timestamps,
    )
    print(f"  {len(entries)} safetensors files found")

    lineages, orphans = group_into_lineages(entries)

    # Attach fallback partners to unpaired-noise orphans (max_step_distance=0
    # disables the post-pass entirely; None means no distance limit).
    if args.max_fallback_step_distance != 0:
        attached = attach_fallback_partners(lineages, orphans, max_step_distance=args.max_fallback_step_distance)
        if attached:
            print(f"  Attached fallback partners to {attached} unpaired-noise orphan(s)")

    manifest = build_manifest(roots, triggers, default_trigger, lineages, orphans)

    output.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    print(f"  Manifest written to {output}")
    print_human_summary(lineages, orphans)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
