"""GPU eval runner for the WAN2.2 LoRA inventory.

Reads a manifest produced by tools/lora_eval/inventory.py and generates a
single still frame (Phase 1) for each variant by invoking wan_generate_video.py
as a subprocess. Designed for the deletion-triage workflow: fixed seed per
lineage so step variants of one lineage are visually comparable, skip-if-exists
so the run is resume-safe, and orphans with `fallback_partner` (cross-step
nearest-neighbor pairing) get evaluated too with a `_fallback_<step>` marker
in the output filename.

Subprocess-per-variant is intentional for v1: it's much simpler than
in-process LoRA swap, robust to per-variant failures (one bad LoRA doesn't
crash the whole run), and the 30–60 s model-load cost per variant is
acceptable for a one-off triage pass. If this proves too slow, the next
iteration can keep wan_generate_video's models warm across calls.

Usage (typical):
    python tools/lora_eval/run.py \\
        --manifest ~/eval_runs/wan22_inventory.json \\
        --output-dir ~/eval_runs/$(date +%Y%m%d_%H%M%S)/wan22 \\
        --dry-run             # plan only, don't generate

Then once the plan looks right, drop --dry-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Default base model + encoder + VAE paths for WAN2.2 T2V on this workstation.
# These were located by file search; override via CLI if your paths differ.
DEFAULT_DIT_LOW = Path("/home/dustin/SwarmUI/Models/diffusion_models/Wan-2.2-T2V-Low-Noise-BF16.safetensors")
DEFAULT_DIT_HIGH = Path("/home/dustin/SwarmUI/Models/diffusion_models/Wan-2.2-T2V-High-Noise-BF16.safetensors")
# Use the WAN 2.1 VAE for T2V-A14B / I2V-A14B — these 14B dual-noise models
# inherit WAN 2.1's VAE (3 input channels, 16-dim latent). The newer
# `wan2.2_vae.safetensors` is for the WAN 2.2 5B model (12 channels, 48-dim)
# and is NOT compatible with T2V-A14B; using it produces a state_dict shape
# mismatch on encoder.conv1 ([160,12,...] vs [96,3,...]) at VAE load time.
DEFAULT_VAE = Path("/home/dustin/SwarmUI/Models/VAE/Wan/wan_2.1_vae.safetensors")
DEFAULT_T5 = Path("/home/dustin/SwarmUI/Models/clip/umt5-xxl-enc-bf16.safetensors")

# Path to the wan_generate_video.py wrapper script in the project root.
DEFAULT_GENERATE_SCRIPT = Path("/home/dustin/blissful-tuner/wan_generate_video.py")

# Default class token appended after the trigger in the prompt (e.g. "CRLN woman").
# Without a class noun, ambiguous trigger tokens get ungendered/wrong-gendered by
# the base model's priors. Override per-trigger via --class-token-override
# (e.g. DLAY=man); the trigger-keyed map below covers known exceptions.
DEFAULT_CLASS_TOKEN = "woman"
DEFAULT_CLASS_TOKEN_OVERRIDES: dict[str, str] = {"DLAY": "man"}

# Phase 1 still-frame likeness prompt — naturalistic English in the
# Subject + Scene + Aesthetic Control form recommended by Alibaba's WAN
# prompting guide. {TRIGGER} is substituted per-lineage from the manifest;
# the substitution includes the class token (e.g. "CRLN woman", "DLAY man").
DEFAULT_PROMPT = (
    "Sunny lighting, medium close-up shot, soft lighting, low contrast, warm colors, "
    "side lighting, daytime, photorealistic. {TRIGGER} stands in a sunlit kitchen, "
    "looking directly at the camera with a soft, natural smile. {TRIGGER} has natural "
    "skin texture with visible pores and subtle imperfections, eyes catching the light. "
    "{TRIGGER} wears a plain cotton shirt. Background: a softly blurred kitchen window "
    "with morning light pouring in, warm bokeh highlights. Cinematic quality, sharp "
    "focus on the face, shallow depth of field."
)

# Default WAN2.2 generation parameters for a 960x960 single-frame eval. These
# match the values commonly recommended for WAN2.2 T2V from CLAUDE.md and the
# Alibaba prompting guide (flow_shift=12 for T2V, 30 inference steps, CFG 5).
DEFAULT_VIDEO_SIZE = (960, 960)
DEFAULT_VIDEO_LENGTH = 1
DEFAULT_INFER_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_FLOW_SHIFT = 12.0
DEFAULT_TASK = "t2v-A14B"


@dataclass
class PlannedRun:
    """One generation job in the eval plan.

    For paired WAN2.2 variants both `lora_high_path` and `lora_low_path` are
    set. For unpaired-with-fallback orphans, one of them comes from the
    fallback partner and `fallback_step` is recorded for transparency. For
    single-noise variants (rare for WAN2.2 but possible), the missing channel
    is None and wan_generate_video uses the base model for that half.
    """

    lineage_name: str
    source_root_label: str
    trigger: str
    step: int | None
    lora_high_path: str | None
    lora_low_path: str | None
    fallback_used: bool
    fallback_step: int | None
    fallback_distance: int | None
    seed: int
    prompt: str
    output_path: str
    notes: list[str] = field(default_factory=list)


# ─── Plan construction ──────────────────────────────────────────────────────


def safe_dirname(name: str) -> str:
    """Make a lineage name filesystem-safe. Replace `@` (used by the inventory
    for DiffusionPipe timestamps) with `_at_`, slashes with underscores, and
    collapse anything else weird."""
    return re.sub(r"[^A-Za-z0-9._+-]", "_", name.replace("@", "_at_"))


def lineage_seed(lineage_name: str, seed_base: int) -> int:
    """Derive a stable seed per lineage so step variants of one lineage all
    use the same initial noise — making cross-step quality comparison fair."""
    h = hashlib.sha256(lineage_name.encode("utf-8")).digest()
    raw = int.from_bytes(h[:4], "big")
    # Combine with seed_base via XOR so the user can shift the whole eval to a
    # different seed family without changing the per-lineage relativity.
    return (raw ^ seed_base) & 0x7FFFFFFF


def resolve_class_token(
    trigger: str,
    default_class_token: str,
    class_token_overrides: dict[str, str],
) -> str:
    """Look up the class noun for a given trigger (case-insensitive match on
    the override map). Returns the override if present, else the default."""
    upper = trigger.upper()
    for key, value in class_token_overrides.items():
        if key.upper() == upper:
            return value
    return default_class_token


def resolve_prompt(
    lineage_name: str,
    trigger: str,
    prompt_overrides: dict[str, str],
    default_prompt: str,
    class_token: str,
) -> str:
    """Pick a prompt for `lineage_name`: per-lineage override if present, else
    the default. {TRIGGER} substitution happens after lookup so override
    prompts can also use the placeholder. The substitution expands to
    `<trigger> <class_token>` (e.g. "CRLN woman") so the model knows the
    subject's class — without this, ambiguous triggers default to whatever
    gender the base model's priors lean toward."""
    template = prompt_overrides.get(lineage_name, default_prompt)
    substitution = f"{trigger} {class_token}".strip()
    return template.replace("{TRIGGER}", substitution)


def output_filename(step: int | None, suffix: str = "") -> str:
    """Stable filename: `step_NNNNNNNN[suffix].png`. Step=None becomes 'final'."""
    step_str = f"{step:08d}" if step is not None else "final"
    return f"step_{step_str}{suffix}.png"


def plan_runs(
    manifest: dict[str, Any],
    output_root: Path,
    prompt_overrides: dict[str, str],
    default_prompt: str,
    seed_base: int,
    default_class_token: str,
    class_token_overrides: dict[str, str],
    *,
    only_lineages: set[str] | None = None,
    skip_singles: bool = False,
    include_orphans: bool = True,
) -> list[PlannedRun]:
    """Walk the manifest and emit one PlannedRun per evaluatable variant.

    A variant is evaluatable if it has at least one of high_path / low_path /
    single_path set. Unpaired-noise orphans with `fallback_partner` are also
    emitted (the orphan + its fallback partner together form a synthetic
    paired job). Orphans without a fallback partner are skipped.
    """
    plans: list[PlannedRun] = []
    for lin in manifest["lineages"]:
        lineage_name = lin["name"]
        if only_lineages is not None and lineage_name not in only_lineages:
            continue
        trigger = lin["trigger"]
        source_root_label = lin["source_root_label"]
        seed = lineage_seed(lineage_name, seed_base)
        class_token = resolve_class_token(trigger, default_class_token, class_token_overrides)
        prompt = resolve_prompt(lineage_name, trigger, prompt_overrides, default_prompt, class_token)
        lineage_dir = output_root / source_root_label / safe_dirname(lineage_name)

        for v in lin["variants"]:
            high = v.get("high_path")
            low = v.get("low_path")
            single = v.get("single_path")
            step = v.get("step")
            ema_tag = v.get("ema_tag")
            named_tag = v.get("named_tag")

            # Build a suffix to disambiguate variants within the same step
            # bucket (EMA variants, named tags, finals).
            tag_parts: list[str] = []
            if ema_tag is not None:
                tag_parts.append(f"ema-{ema_tag}" if ema_tag else "ema")
            if named_tag:
                tag_parts.append(f"tag-{named_tag}")
            if v.get("is_final"):
                tag_parts.append("final")
            suffix = ("_" + "_".join(tag_parts)) if tag_parts else ""

            # Single-channel variant (no high/low pair). For WAN2.2 this is
            # rare but happens when only one noise model was trained against.
            # We can still generate, but the missing channel uses the base
            # model with no LoRA influence.
            if single and not (high or low):
                if skip_singles:
                    continue
                lora_high = None
                lora_low = single  # treat single as low-noise by convention
                notes = ["single-channel (no pair); using base for high-noise side"]
            elif high or low:
                lora_high = high
                lora_low = low
                notes = []
                if not high:
                    notes.append("no high-noise LoRA; using base for high")
                if not low:
                    notes.append("no low-noise LoRA; using base for low")
            else:
                # Nothing to load — skip.
                continue

            plans.append(
                PlannedRun(
                    lineage_name=lineage_name,
                    source_root_label=source_root_label,
                    trigger=trigger,
                    step=step,
                    lora_high_path=lora_high,
                    lora_low_path=lora_low,
                    fallback_used=False,
                    fallback_step=None,
                    fallback_distance=None,
                    seed=seed,
                    prompt=prompt,
                    output_path=str(lineage_dir / output_filename(step, suffix)),
                    notes=notes,
                )
            )

    if include_orphans:
        for o in manifest.get("orphans", []):
            fp = o.get("fallback_partner")
            ui = o.get("unpaired_info")
            if not fp or not ui:
                continue
            lineage_name = ui["lineage"]
            if only_lineages is not None and lineage_name not in only_lineages:
                continue
            source_root_label = ui["source_root_label"]
            present_channel = ui["present_channel"]
            present_path = o["path"]
            partner_path = fp["path"]
            partner_channel = fp["channel"]
            step = ui["step"]
            partner_step = fp["step"]

            # Need a trigger for the orphan; look it up from the lineage
            # (fall through to default if the lineage isn't in the manifest's
            # `lineages` list — happens if all the lineage's variants were
            # also unpaired and never made it past the orphan check).
            lineage_record = next(
                (
                    lin
                    for lin in manifest["lineages"]
                    if lin["name"] == lineage_name and lin["source_root_label"] == source_root_label
                ),
                None,
            )
            trigger = lineage_record["trigger"] if lineage_record else manifest.get("default_trigger", "DLAY")
            seed = lineage_seed(lineage_name, seed_base)
            class_token = resolve_class_token(trigger, default_class_token, class_token_overrides)
            prompt = resolve_prompt(lineage_name, trigger, prompt_overrides, default_prompt, class_token)
            lineage_dir = output_root / source_root_label / safe_dirname(lineage_name)
            suffix = f"_fallback_{partner_step:08d}"

            if present_channel == "high":
                lora_high = present_path
                lora_low = partner_path
            else:
                lora_low = present_path
                lora_high = partner_path

            plans.append(
                PlannedRun(
                    lineage_name=lineage_name,
                    source_root_label=source_root_label,
                    trigger=trigger,
                    step=step,
                    lora_high_path=lora_high,
                    lora_low_path=lora_low,
                    fallback_used=True,
                    fallback_step=partner_step,
                    fallback_distance=fp["step_distance"],
                    seed=seed,
                    prompt=prompt,
                    output_path=str(lineage_dir / output_filename(step, suffix)),
                    notes=[f"fallback {partner_channel} from step {partner_step} (Δ={fp['step_distance']})"],
                )
            )

    return plans


# ─── Subprocess execution ───────────────────────────────────────────────────


def build_command(
    plan: PlannedRun,
    *,
    generate_script: Path,
    python_exe: Path,
    task: str,
    dit_low: Path,
    dit_high: Path,
    vae: Path,
    t5: Path,
    video_size: tuple[int, int],
    video_length: int,
    infer_steps: int,
    guidance_scale: float,
    flow_shift: float,
    fp8: bool,
    fp8_scaled: bool,
    fp8_t5: bool,
    blocks_to_swap: int | None,
    offload_inactive_dit: bool,
    vae_cache_cpu: bool,
    use_pinned_memory_for_block_swap: bool,
    lazy_loading: bool,
) -> list[str]:
    """Assemble the wan_generate_video.py command for one PlannedRun."""
    cmd: list[str] = [
        str(python_exe),
        str(generate_script),
        "--task",
        task,
        "--dit",
        str(dit_low),
        "--dit_high_noise",
        str(dit_high),
        "--vae",
        str(vae),
        "--t5",
        str(t5),
        "--prompt",
        plan.prompt,
        "--video_size",
        str(video_size[0]),
        str(video_size[1]),
        "--video_length",
        str(video_length),
        "--infer_steps",
        str(infer_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--flow_shift",
        str(flow_shift),
        "--seed",
        str(plan.seed),
        "--save_path",
        str(Path(plan.output_path).parent),
        "--output_type",
        "images",
    ]
    # Blissful Tuner's metadata-prep code (common_extensions.py:prepare_metadata)
    # indexes args.lora_multiplier[i] without checking for None, so we must pass
    # `--lora_multiplier 1.0` whenever we pass `--lora_weight`. Same for the
    # high-noise pair. We always send exactly one LoRA per channel from the
    # manifest, so multiplier is always a single 1.0.
    if plan.lora_low_path:
        cmd.extend(["--lora_weight", plan.lora_low_path, "--lora_multiplier", "1.0"])
    if plan.lora_high_path:
        cmd.extend(["--lora_weight_high_noise", plan.lora_high_path, "--lora_multiplier_high_noise", "1.0"])
    if fp8:
        cmd.append("--fp8")
    if fp8_scaled:
        cmd.append("--fp8_scaled")
    if fp8_t5:
        cmd.append("--fp8_t5")
    if blocks_to_swap is not None:
        cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if offload_inactive_dit:
        cmd.append("--offload_inactive_dit")
    if vae_cache_cpu:
        cmd.append("--vae_cache_cpu")
    if use_pinned_memory_for_block_swap:
        cmd.append("--use_pinned_memory_for_block_swap")
    if lazy_loading:
        cmd.append("--lazy_loading")
    return cmd


def run_one(
    plan: PlannedRun,
    *,
    cmd: list[str],
    log_file: Path,
    timeout: int | None,
) -> tuple[int, float]:
    """Execute one wan_generate_video subprocess. Captures stdout+stderr to
    `log_file` (appending). Returns (exit_code, elapsed_seconds)."""
    Path(plan.output_path).parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        with log_file.open("a") as lf:
            lf.write(f"\n\n──── {datetime.now().isoformat()} ────\n")
            lf.write(f"lineage: {plan.lineage_name}  step: {plan.step}\n")
            lf.write(f"output: {plan.output_path}\n")
            lf.write(f"command: {' '.join(repr(a) if ' ' in a else a for a in cmd)}\n\n")
            lf.flush()
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return result.returncode, time.monotonic() - start
    except subprocess.TimeoutExpired:
        return 124, time.monotonic() - start
    except OSError as e:
        with log_file.open("a") as lf:
            lf.write(f"\nOSError: {e}\n")
        return 127, time.monotonic() - start


# ─── Output reconciliation ──────────────────────────────────────────────────
#
# wan_generate_video.py writes its output PNG to `--save_path` with an
# auto-generated filename (e.g., `wan_<seed>_<timestamp>.png`). We want a
# stable predictable name so skip-if-exists works. The robust contract is:
# capture the set of PNGs in the save_path BEFORE the subprocess, then the
# new file is the set difference after. This avoids any reliance on mtime
# resolution and is safe against earlier already-renamed files in the same
# directory (e.g., when generating step 2 after step 1 already produced
# `step_00006000.png` in the same lineage dir).


def snapshot_save_dir(plan: PlannedRun) -> set[Path]:
    """Capture the set of existing PNGs in the save dir before subprocess run."""
    save_dir = Path(plan.output_path).parent
    if not save_dir.exists():
        return set()
    return set(save_dir.glob("*.png"))


def reconcile_output(plan: PlannedRun, before: set[Path]) -> Path | None:
    """Find the new PNG produced by the subprocess (set difference vs. `before`)
    and rename it to the planned output path. Returns the final path on
    success, or None if no new PNG appeared (likely a generation failure)."""
    save_dir = Path(plan.output_path).parent
    target = Path(plan.output_path)
    after = set(save_dir.glob("*.png"))
    new = after - before - {target}
    if not new:
        return None
    # If the subprocess wrote multiple PNGs (unusual), take the most recent.
    chosen = max(new, key=lambda p: p.stat().st_mtime)
    chosen.rename(target)
    return target


# ─── CLI ────────────────────────────────────────────────────────────────────


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_prompt_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        print(f"WARNING: prompt-file not found: {path}", file=sys.stderr)
        return {}
    # YAML if available, else assume JSON. Avoid hard yaml dependency.
    text = path.read_text()
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    if not isinstance(data, dict):
        print("ERROR: prompt-file must contain a mapping of lineage_name -> prompt", file=sys.stderr)
        return {}
    return {str(k): str(v) for k, v in data.items()}


def print_plan_summary(plans: list[PlannedRun]) -> None:
    print()
    print("─── Eval plan ──────────────────────────────────────────────────────")
    print(f"  Total runs planned: {len(plans)}")
    fb = sum(1 for p in plans if p.fallback_used)
    if fb:
        print(f"    Fallback-paired (cross-step): {fb}")
    by_lineage: dict[str, int] = {}
    for p in plans:
        by_lineage[p.lineage_name] = by_lineage.get(p.lineage_name, 0) + 1
    print(f"  Distinct lineages: {len(by_lineage)}")
    print()
    print("  Sample (first 5):")
    for p in plans[:5]:
        fb_marker = f" [fallback Δ={p.fallback_distance}]" if p.fallback_used else ""
        print(f"    {p.lineage_name}  step={p.step}{fb_marker}  → {p.output_path}")
    if len(plans) > 5:
        print(f"    ... and {len(plans) - 5} more")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run WAN2.2 still-frame eval over an inventory manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to inventory manifest.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root directory for generated PNGs and logs.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Optional YAML/JSON mapping lineage_name -> prompt.")
    parser.add_argument(
        "--lineage",
        action="append",
        default=None,
        help="If set, only evaluate the given lineage name(s). Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of runs (for testing).")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed; per-lineage seed = hash(lineage) ^ seed_base.")
    parser.add_argument(
        "--class-token",
        default=DEFAULT_CLASS_TOKEN,
        help="Default class noun appended after the trigger in the prompt (e.g. 'CRLN woman'). "
        "Use --class-token-override to set per-trigger exceptions.",
    )
    parser.add_argument(
        "--class-token-override",
        action="append",
        default=[],
        metavar="TRIGGER=TOKEN",
        help="Per-trigger class-token override, repeatable. Match is case-insensitive on TRIGGER. "
        f"Built-in defaults: {', '.join(f'{k}={v}' for k, v in DEFAULT_CLASS_TOKEN_OVERRIDES.items())}. "
        "Pass an empty TOKEN (e.g. 'KIDX=') to suppress the class word for a specific trigger.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without running anything.")
    parser.add_argument("--no-orphans", action="store_true", help="Skip orphans with fallback_partner.")
    parser.add_argument("--skip-singles", action="store_true", help="Skip variants without high+low pair.")
    parser.add_argument("--force", action="store_true", help="Re-generate even if output PNG already exists.")

    grp = parser.add_argument_group("WAN2.2 base models (override defaults if your paths differ)")
    grp.add_argument("--task", default=DEFAULT_TASK, help="WAN task tag.")
    grp.add_argument("--dit-low", type=Path, default=DEFAULT_DIT_LOW)
    grp.add_argument("--dit-high", type=Path, default=DEFAULT_DIT_HIGH)
    grp.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    grp.add_argument("--t5", type=Path, default=DEFAULT_T5)
    grp.add_argument("--generate-script", type=Path, default=DEFAULT_GENERATE_SCRIPT)
    grp.add_argument("--python", type=Path, default=Path(sys.executable))

    grp = parser.add_argument_group("Generation parameters (Phase 1 still-frame defaults)")
    grp.add_argument("--video-size", type=int, nargs=2, default=list(DEFAULT_VIDEO_SIZE), metavar=("W", "H"))
    grp.add_argument("--video-length", type=int, default=DEFAULT_VIDEO_LENGTH)
    grp.add_argument("--infer-steps", type=int, default=DEFAULT_INFER_STEPS)
    grp.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    grp.add_argument("--flow-shift", type=float, default=DEFAULT_FLOW_SHIFT)
    grp.add_argument("--fp8", action="store_true", help="Use fp8 for DiT (saves VRAM, on-the-fly quantization at load).")
    grp.add_argument("--fp8-scaled", action="store_true", help="Scaled fp8 (only with --fp8); often more memory-efficient.")
    grp.add_argument("--fp8-t5", action="store_true", help="Use fp8 for T5 encoder.")
    grp.add_argument(
        "--blocks-to-swap",
        type=int,
        default=None,
        help="Swap N transformer blocks to CPU RAM (max 39 for 14B). Trades inference speed for VRAM. "
        "Recommend 20-30 for 32 GB GPU + BF16 base models.",
    )
    grp.add_argument(
        "--offload-inactive-dit",
        action="store_true",
        help="WAN2.2-only: keep the inactive (high or low) noise model in CPU RAM during inference. "
        "Roughly halves DiT VRAM. Required for 32 GB cards with BF16 base models.",
    )
    grp.add_argument("--vae-cache-cpu", action="store_true", help="Cache VAE features in CPU RAM.")
    grp.add_argument(
        "--use-pinned-memory-for-block-swap",
        action="store_true",
        help="Use pinned host memory for block swaps (faster CPU↔GPU transfer; needs free RAM).",
    )
    grp.add_argument(
        "--lazy-loading",
        action="store_true",
        help="Defer DiT model loading until needed.",
    )
    grp.add_argument(
        "--low-vram",
        action="store_true",
        help="Preset: enable a sane VRAM-saving combination "
        "(--fp8 --fp8-scaled --fp8-t5 --offload-inactive-dit --blocks-to-swap 25 --vae-cache-cpu --use-pinned-memory-for-block-swap). "
        "Recommended for 32 GB GPUs running BF16 WAN2.2 base models.",
    )
    grp.add_argument("--timeout", type=int, default=1200, help="Per-variant timeout in seconds (default 20 min).")

    args = parser.parse_args()

    # --low-vram preset turns on a sane VRAM-saving combination, but only for
    # flags the user hasn't explicitly set themselves (so individual flags can
    # still override).
    if args.low_vram:
        if not args.fp8:
            args.fp8 = True
        if not args.fp8_scaled:
            args.fp8_scaled = True
        if not args.fp8_t5:
            args.fp8_t5 = True
        if not args.offload_inactive_dit:
            args.offload_inactive_dit = True
        if args.blocks_to_swap is None:
            args.blocks_to_swap = 25
        if not args.vae_cache_cpu:
            args.vae_cache_cpu = True
        if not args.use_pinned_memory_for_block_swap:
            args.use_pinned_memory_for_block_swap = True

    manifest = load_manifest(args.manifest.expanduser().resolve())
    prompt_overrides = load_prompt_overrides(args.prompt_file.expanduser().resolve() if args.prompt_file else None)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Class-token map: start from built-in defaults, then layer CLI overrides
    # on top so the user can extend or replace specific triggers.
    class_token_overrides: dict[str, str] = dict(DEFAULT_CLASS_TOKEN_OVERRIDES)
    for spec in args.class_token_override:
        if "=" not in spec:
            print(f"ERROR: --class-token-override must be TRIGGER=TOKEN, got {spec!r}", file=sys.stderr)
            return 2
        key, _, value = spec.partition("=")
        class_token_overrides[key.strip()] = value.strip()

    only_lineages = set(args.lineage) if args.lineage else None
    plans = plan_runs(
        manifest,
        output_root,
        prompt_overrides,
        DEFAULT_PROMPT,
        args.seed_base,
        args.class_token,
        class_token_overrides,
        only_lineages=only_lineages,
        skip_singles=args.skip_singles,
        include_orphans=not args.no_orphans,
    )
    if args.limit is not None:
        plans = plans[: args.limit]

    # Persist the plan so failures are debuggable later.
    plan_path = output_root / "plan.json"
    plan_path.write_text(json.dumps([asdict(p) for p in plans], indent=2))

    print_plan_summary(plans)

    if args.dry_run:
        print()
        print(f"  --dry-run set; plan written to {plan_path}. Exiting without generation.")
        return 0

    # Sanity-check tooling before we start a long run.
    for name, path in (
        ("generate-script", args.generate_script),
        ("dit-low", args.dit_low),
        ("dit-high", args.dit_high),
        ("vae", args.vae),
        ("t5", args.t5),
    ):
        if not Path(path).exists():
            print(f"ERROR: {name} not found: {path}", file=sys.stderr)
            return 2

    log_file = output_root / "run.log"
    print()
    print(f"  Streaming subprocess output to {log_file}")
    print(f"  Plan persisted at {plan_path}")
    print()

    completed = 0
    skipped = 0
    failed = 0
    total = len(plans)
    overall_start = time.monotonic()
    for i, plan in enumerate(plans, 1):
        out = Path(plan.output_path)
        if out.exists() and not args.force:
            skipped += 1
            print(f"[{i:>4}/{total}] SKIP  {plan.lineage_name}  step={plan.step}  (already exists)")
            continue

        cmd = build_command(
            plan,
            generate_script=args.generate_script,
            python_exe=args.python,
            task=args.task,
            dit_low=args.dit_low,
            dit_high=args.dit_high,
            vae=args.vae,
            t5=args.t5,
            video_size=(args.video_size[0], args.video_size[1]),
            video_length=args.video_length,
            infer_steps=args.infer_steps,
            guidance_scale=args.guidance_scale,
            flow_shift=args.flow_shift,
            fp8=args.fp8,
            fp8_scaled=args.fp8_scaled,
            fp8_t5=args.fp8_t5,
            blocks_to_swap=args.blocks_to_swap,
            offload_inactive_dit=args.offload_inactive_dit,
            vae_cache_cpu=args.vae_cache_cpu,
            use_pinned_memory_for_block_swap=args.use_pinned_memory_for_block_swap,
            lazy_loading=args.lazy_loading,
        )
        fb = f" [fb Δ={plan.fallback_distance}]" if plan.fallback_used else ""
        print(f"[{i:>4}/{total}] RUN   {plan.lineage_name}  step={plan.step}{fb}  ", end="", flush=True)

        before = snapshot_save_dir(plan)
        rc, elapsed = run_one(plan, cmd=cmd, log_file=log_file, timeout=args.timeout)
        if rc == 0:
            renamed = reconcile_output(plan, before)
            if renamed is None:
                failed += 1
                print(f"FAILED (no PNG produced) ({elapsed:.1f}s)")
            else:
                completed += 1
                print(f"OK ({elapsed:.1f}s) → {renamed.name}")
        else:
            failed += 1
            print(f"FAILED rc={rc} ({elapsed:.1f}s) — see {log_file}")

    overall = time.monotonic() - overall_start
    print()
    print("─── Run summary ────────────────────────────────────────────────────")
    print(f"  Total: {total}  Completed: {completed}  Skipped: {skipped}  Failed: {failed}")
    print(f"  Wall time: {overall / 60:.1f} min ({overall:.0f} s)")
    if failed:
        print(f"  See {log_file} for failure details.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
