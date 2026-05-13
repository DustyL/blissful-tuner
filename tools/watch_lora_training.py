"""Live-watching companion for LoRA training runs.

Polls the latest TensorBoard event file in a run's logging_dir and prints a
compact decision-critical dashboard: training progress, Prodigy d-discovery
status (color-coded for the v8-validated d>1e-3 healthy threshold),
Schedule-Free effective LR, and prior-preservation gating.

Designed for the Flux2/Z-Image Prodigy+ScheduleFree kill/switch protocol: by
step ~500 the d-value should escape the 1e-5 collapse trap. This tool surfaces
that signal at a glance instead of clicking through TensorBoard tabs.

Usage:
  # one-shot snapshot
  python tools/watch_lora_training.py /home/dustin/output/dlay_zimagebase_advanced_lora_v3

  # follow mode: re-poll every 30s, clear screen between updates
  python tools/watch_lora_training.py /home/dustin/output/dlay_zimagebase_advanced_lora_v3 --watch 30

  # decision-mode: assert verdict at a specific step (exits 0 healthy / 1 collapsed)
  python tools/watch_lora_training.py /home/dustin/output/dlay_zimagebase_advanced_lora_v3 --verdict-at 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

D_HEALTHY_THRESHOLD = 1e-3
D_COLLAPSE_THRESHOLD = 1e-4

DECISION_TAGS = [
    "lr/d/unet",
    "lr/d/unet plus",
    "lr/effective_lr/unet",
    "lr/effective_lr/unet plus",
    "loss/current",
    "loss/average",
    "prior/gate_frac",
    "prior/teacher_ran",
]


def find_event_file(run_dir: Path) -> Path | None:
    """Find the most recently modified TB event file under run_dir/logs/."""
    logs = run_dir / "logs"
    if not logs.exists():
        return None
    candidates = list(logs.rglob("events.out.tfevents.*"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_latest_scalars(event_file: Path) -> dict[str, tuple[int, float]]:
    """Return {tag: (step, value)} for the most recent point of each tag."""
    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    out: dict[str, tuple[int, float]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        if events:
            last = events[-1]
            out[tag] = (int(last.step), float(last.value))
    return out


def classify_d(value: float) -> tuple[str, str]:
    """Return (color, label) for a Prodigy d value."""
    if value >= D_HEALTHY_THRESHOLD:
        return GREEN, "HEALTHY"
    if value >= D_COLLAPSE_THRESHOLD:
        return YELLOW, "BORDERLINE"
    return RED, "COLLAPSED"


def fmt(scalars: dict[str, tuple[int, float]], tag: str, fmt_spec: str = ".4g") -> tuple[int | None, str]:
    """Format a scalar value with safe fallback for missing tags."""
    if tag not in scalars:
        return None, f"{DIM}n/a{RESET}"
    step, value = scalars[tag]
    return step, f"{value:{fmt_spec}}"


def render(run_dir: Path, max_steps: int | None) -> tuple[str, int | None, dict[str, tuple[int, float]]]:
    """Build the dashboard string. Returns (text, current_step, scalars)."""
    event_file = find_event_file(run_dir)
    if event_file is None:
        return f"{RED}No TB event files found under {run_dir / 'logs'}{RESET}\n", None, {}

    scalars = load_latest_scalars(event_file)
    if not scalars:
        return f"{YELLOW}Event file exists but has no scalars yet: {event_file.name}{RESET}\n", None, {}

    current_step = max((s for s, _ in scalars.values()), default=0)
    progress = f"{current_step:,}"
    if max_steps:
        pct = 100.0 * current_step / max_steps
        progress += f" / {max_steps:,}  ({pct:.1f}%)"

    lines = [
        f"{BOLD}Run:{RESET} {run_dir.name}",
        f"{BOLD}Event file:{RESET} {DIM}{event_file.relative_to(run_dir)}{RESET}",
        f"{BOLD}Step:{RESET} {progress}",
        "",
    ]

    # Loss
    _, cur_loss = fmt(scalars, "loss/current", ".5f")
    _, avg_loss = fmt(scalars, "loss/average", ".5f")
    lines += [f"{BOLD}Loss:{RESET}  current={cur_loss}   smoothed_avg={avg_loss}", ""]

    # Prodigy d-discovery — only the main LoRA-A group ("unet") is the kill/switch
    # signal. The LoRA+ split group ("unet plus") has its d-value scaled DOWN by
    # the loraplus_lr_ratio (typically 12x), so a tiny "plus" d-value with a
    # healthy effective_lr is by design, not a collapse.
    lines.append(
        f"{BOLD}Prodigy d-discovery{RESET}  {DIM}(decision metric — must exceed {D_HEALTHY_THRESHOLD:.0e} by step 500){RESET}"
    )
    if "lr/d/unet" in scalars:
        _, value = scalars["lr/d/unet"]
        color, label = classify_d(value)
        lines.append(f"  lr/d/unet (LoRA-A, main)   = {color}{value:.3e}{RESET}  {color}{label}{RESET}")
    else:
        lines.append(f"  lr/d/unet (LoRA-A, main)   = {DIM}n/a{RESET}")
    if "lr/d/unet plus" in scalars:
        _, value = scalars["lr/d/unet plus"]
        lines.append(f"  lr/d/unet plus (LoRA+ B)   = {value:.3e}  {DIM}informational (scaled by lr_ratio){RESET}")
    lines.append("")

    # Schedule-Free effective LR (sanity check — should stabilize near 0.05)
    lines.append(f"{BOLD}Schedule-Free effective LR{RESET}  {DIM}(target: ~0.05){RESET}")
    for tag in ("lr/effective_lr/unet", "lr/effective_lr/unet plus"):
        _, val = fmt(scalars, tag, ".4e")
        lines.append(f"  {tag.ljust(30)} = {val}")
    lines.append("")

    # Prior preservation gating (mostly informational)
    if "prior/gate_frac" in scalars or "prior/teacher_ran" in scalars:
        lines.append(f"{BOLD}Prior preservation{RESET}")
        _, gate = fmt(scalars, "prior/gate_frac", ".3f")
        _, ran = fmt(scalars, "prior/teacher_ran", ".0f")
        lines.append(f"  gate_frac (teacher-active steps fraction)  = {gate}")
        lines.append(f"  teacher_ran (latest step)                  = {ran}")
        lines.append("")

    return "\n".join(lines) + "\n", current_step, scalars


def verdict(scalars: dict[str, tuple[int, float]], at_step: int) -> int:
    """Print kill/switch verdict and return shell exit code (0=keep, 1=switch)."""
    if "lr/d/unet" not in scalars:
        print(f"{RED}Cannot evaluate verdict: lr/d/unet not yet logged{RESET}")
        return 2
    step, d_unet = scalars["lr/d/unet"]
    if step < at_step:
        print(f"{YELLOW}Current step ({step}) < verdict step ({at_step}); too early to decide.{RESET}")
        return 2
    color, label = classify_d(d_unet)
    print(f"\n{BOLD}=== KILL/SWITCH VERDICT at step {step} ==={RESET}")
    print(f"  lr/d/unet = {color}{d_unet:.3e}{RESET}  {color}{label}{RESET}")
    if d_unet >= D_HEALTHY_THRESHOLD:
        print(f"{GREEN}{BOLD}DECISION: KEEP RUNNING{RESET} — Prodigy d-discovery is working.")
        return 0
    print(f"{RED}{BOLD}DECISION: KILL AND SWITCH TO FALLBACK{RESET}")
    print(f"  d-value is below the {D_HEALTHY_THRESHOLD:.0e} healthy threshold")
    print("  Likely Flux2-v11-style dataset-induced d-collapse pathology.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Training output_dir (contains logs/ subdirectory)")
    ap.add_argument("--watch", type=float, metavar="SEC", help="Re-poll every SEC seconds (clears screen).")
    ap.add_argument("--verdict-at", type=int, metavar="STEP", help="Exit with kill/switch verdict once at STEP.")
    ap.add_argument("--max-steps", type=int, help="Total steps in training (for %% progress display).")
    args = ap.parse_args()

    if not args.run_dir.exists():
        print(f"{RED}Run directory does not exist: {args.run_dir}{RESET}", file=sys.stderr)
        return 2

    if args.verdict_at is not None:
        # Decision mode: poll until verdict step is reached, then exit
        while True:
            text, step, scalars = render(args.run_dir, args.max_steps)
            print(text)
            if step is not None and step >= args.verdict_at:
                return verdict(scalars, args.verdict_at)
            if args.watch is None:
                print(f"{YELLOW}Step {step} < {args.verdict_at}; rerun later (or pass --watch for polling).{RESET}")
                return 2
            time.sleep(args.watch)

    if args.watch is not None:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                text, _, _ = render(args.run_dir, args.max_steps)
                print(text)
                print(f"{DIM}Polling every {args.watch}s — Ctrl+C to stop{RESET}")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0

    text, _, _ = render(args.run_dir, args.max_steps)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
