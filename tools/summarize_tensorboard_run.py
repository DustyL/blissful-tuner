from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator


@dataclass(frozen=True)
class ScalarSummary:
    count: int
    first_step: int
    last_step: int
    min_value: float
    max_value: float
    last_value: float


def _find_event_files(logdir: Path) -> list[Path]:
    return sorted(logdir.rglob("events.out.tfevents*"))


def _summarize_scalars(event_file: Path, *, prefix: str | None = None) -> dict[str, ScalarSummary]:
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    summaries: dict[str, ScalarSummary] = {}
    for tag in sorted(tags):
        if prefix is not None and not tag.startswith(prefix):
            continue
        events = ea.Scalars(tag)
        if not events:
            continue

        first = events[0]
        last = events[-1]
        min_v = float("inf")
        max_v = float("-inf")
        for e in events:
            v = float(e.value)
            if v < min_v:
                min_v = v
            if v > max_v:
                max_v = v

        summaries[tag] = ScalarSummary(
            count=len(events),
            first_step=int(first.step),
            last_step=int(last.step),
            min_value=min_v,
            max_value=max_v,
            last_value=float(last.value),
        )

    return summaries


def _as_jsonable(summaries: dict[str, ScalarSummary]) -> dict[str, dict[str, Any]]:
    return {
        k: {
            "count": v.count,
            "first_step": v.first_step,
            "last_step": v.last_step,
            "min": v.min_value,
            "max": v.max_value,
            "last": v.last_value,
        }
        for k, v in summaries.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TensorBoard scalar logs for a training run.")
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Root log directory containing events.out.tfevents* files. If omitted, the script will prompt.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional: only include tags that start with this prefix (e.g., 'masked_loss/' or 'prior/').",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON (machine-readable) instead of a human summary.")
    args = parser.parse_args()

    logdir_str = args.logdir
    if logdir_str is None:
        logdir_str = input("TensorBoard logdir: ").strip()
        if not logdir_str:
            raise SystemExit("logdir is required")

    logdir = Path(logdir_str).expanduser()
    if not logdir.exists():
        raise SystemExit(f"logdir not found: {logdir}")

    event_files = _find_event_files(logdir)
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found under: {logdir}")

    # Heuristic: prefer the largest event file (usually contains the actual run, not a stub).
    event_files = sorted(event_files, key=lambda p: p.stat().st_size, reverse=True)
    chosen = event_files[0]

    summaries = _summarize_scalars(chosen, prefix=args.prefix)

    if args.json:
        payload = {
            "event_file": str(chosen),
            "num_event_files_found": len(event_files),
            "num_scalar_tags": len(summaries),
            "scalars": _as_jsonable(summaries),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"TensorBoard event summary: {chosen}")
    print(f"  event_files_found: {len(event_files)}")
    print(f"  scalar_tags:       {len(summaries)}")

    # Print a short priority list first (if present).
    priority = [
        "loss/current",
        "loss/average",
        "masked_loss/target",
        "masked_loss/prior",
        "masked_loss/mask/raw_mean",
        "masked_loss/mask/processed_mean",
        "masked_loss/mask/prior_mean",
        "masked_loss/mask/overlap_mass",
        "prior/gate_frac",
        "prior/w_prior_mean",
        "prior/teacher_ran",
        "timestep/mean",
    ]
    for tag in priority:
        if tag not in summaries:
            continue
        s = summaries[tag]
        print(f"- {tag}: last={s.last_value:.6g} step={s.last_step} (min={s.min_value:.6g} max={s.max_value:.6g} n={s.count})")

    # Then show everything else.
    remaining = [t for t in summaries.keys() if t not in priority]
    if remaining:
        print("\nOther tags:")
        for tag in remaining:
            s = summaries[tag]
            print(f"- {tag}: last={s.last_value:.6g} step={s.last_step} (min={s.min_value:.6g} max={s.max_value:.6g} n={s.count})")


if __name__ == "__main__":
    main()
