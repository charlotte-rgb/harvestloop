#!/usr/bin/env python3
"""
Diagnose a route run from its recorded per-truss data.

Built to answer why the left row fails far more often than the right
one, so it reports outcome against reach, base pose and arm
configuration rather than just success counts.

Usage:
    python3 scripts/analyze_run.py [run_dir]

Defaults to the newest directory under results/runs/.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"

# From ur5e_scaled_06.urdf, already scaled to 0.6:
#   shoulder sits 0.0975 above base_link
#   upper arm 0.255 + forearm 0.235 + grasp tip 0.042
SHOULDER_HEIGHT = 0.0975
ARM_ENVELOPE = 0.255 + 0.23532 + 0.042011

JOINTS = ("pan", "lift", "elbow", "wrist1", "wrist2", "wrist3")


def latest_run_dir():
    candidates = []

    if not RUNS_ROOT.exists():
        raise SystemExit(f"No runs under {RUNS_ROOT}")

    for path in RUNS_ROOT.rglob("trusses.csv"):
        candidates.append(path.parent)

    if not candidates:
        raise SystemExit(f"No runs with trusses.csv under {RUNS_ROOT}")

    return max(
        candidates,
        key=lambda path: (
            path.stat().st_mtime,
            str(path),
        ),
    )


def number(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def vector(row, key, suffixes):
    values = [number(row, f"{key}_{suffix}") for suffix in suffixes]
    return None if any(v is None for v in values) else values


def shoulder_reach(row):
    """Target distance from the shoulder joint, not from base_link."""
    target = vector(row, "grasp_in_arm_base", ("x", "y", "z"))

    if target is None:
        return None

    return math.dist(target, (0.0, 0.0, SHOULDER_HEIGHT))


def succeeded(row):
    return not row["stage"]


def percent(part, whole):
    return f"{part}/{whole}" + (f" ({100.0 * part / whole:.0f}%)" if whole else "")


def main():
    run_dir = (
        Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_run_dir()
    )

    rows = list(csv.DictReader((run_dir / "trusses.csv").open()))

    print(f"Run: {run_dir.name}")
    print(f"Trusses: {len(rows)}")
    print()

    if not any("grasp_in_arm_base_x" in row for row in rows):
        print(
            "This run predates the reach instrumentation; rerun the route "
            "to get base pose, reach and joint columns."
        )
        return 0

    # ---------------------------------------------------------
    # PER TRUSS
    # ---------------------------------------------------------

    print(
        f"{'side':>5} {'stop':>4} {'tr':>3} {'z':>6} {'reach':>6} "
        f"{'margin':>7} {'base_y':>7} {'yaw':>7} "
        f"{'pre_err':>8} {'grasp_err':>9} {'pan':>7} {'lift':>7} "
        f"{'elbow':>7} {'stage':>8}"
    )

    for row in sorted(
        rows, key=lambda r: (r["side"], number(r, "grasp_world_z") or 0.0)
    ):
        reach = shoulder_reach(row)
        q = vector(row, "q_at_pregrasp", JOINTS)

        print(
            f"{row['side']:>5} {row['stop_index']:>4} {row['truss_index']:>3} "
            f"{number(row, 'grasp_world_z') or 0:6.3f} "
            f"{reach if reach is not None else float('nan'):6.3f} "
            f"{(ARM_ENVELOPE - reach) if reach is not None else float('nan'):+7.3f} "
            f"{number(row, 'base_y') or float('nan'):7.3f} "
            f"{number(row, 'base_yaw') or float('nan'):+7.3f} "
            f"{number(row, 'pregrasp_error_m') or float('nan'):8.4f} "
            f"{number(row, 'grasp_error_m') or float('nan'):9.4f} "
            f"{q[0] if q else float('nan'):+7.3f} "
            f"{q[1] if q else float('nan'):+7.3f} "
            f"{q[2] if q else float('nan'):+7.3f} "
            f"{(row['stage'] or 'OK'):>8}"
        )

    # ---------------------------------------------------------
    # LEFT VS RIGHT
    # ---------------------------------------------------------

    print()
    print("=== by side ===")
    print(f"Arm envelope from shoulder: {ARM_ENVELOPE:.3f} m")
    print()

    by_side = defaultdict(list)
    for row in rows:
        by_side[row["side"]].append(row)

    for side, side_rows in sorted(by_side.items()):
        wins = [r for r in side_rows if succeeded(r)]
        reaches = [shoulder_reach(r) for r in side_rows]
        reaches = [r for r in reaches if r is not None]
        base_ys = [number(r, "base_y") for r in side_rows]
        base_ys = [b for b in base_ys if b is not None]

        print(f"{side:>5}: success {percent(len(wins), len(side_rows))}")

        if reaches:
            over = sum(1 for r in reaches if r > ARM_ENVELOPE)
            print(
                f"       reach min {min(reaches):.3f} "
                f"median {sorted(reaches)[len(reaches) // 2]:.3f} "
                f"max {max(reaches):.3f} m | "
                f"beyond envelope: {over}/{len(reaches)}"
            )

        if base_ys:
            print(
                f"       base_y min {min(base_ys):.3f} "
                f"max {max(base_ys):.3f} m"
            )

    # ---------------------------------------------------------
    # OUTCOME VS REACH
    # ---------------------------------------------------------

    print()
    print("=== outcome vs reach from shoulder ===")

    buckets = defaultdict(lambda: [0, 0])

    for row in rows:
        reach = shoulder_reach(row)
        if reach is None:
            continue

        key = round(reach * 20) / 20.0  # 5 cm buckets
        buckets[key][1] += 1
        if succeeded(row):
            buckets[key][0] += 1

    for key in sorted(buckets):
        wins, total = buckets[key]
        marker = "  <-- beyond envelope" if key > ARM_ENVELOPE else ""
        print(f"  reach ~{key:.2f} m: {percent(wins, total)}{marker}")

    # ---------------------------------------------------------
    # WHERE THE TIP STALLED
    # ---------------------------------------------------------

    print()
    print("=== failed grasps: commanded vs achieved, in shoulder frame ===")

    for row in rows:
        if row["stage"] != "GRASP":
            continue

        target = vector(row, "grasp_in_arm_base", ("x", "y", "z"))
        tip = vector(row, "tip_in_arm_base_at_grasp", ("x", "y", "z"))

        if target is None or tip is None:
            continue

        residual = [t - a for t, a in zip(target, tip)]

        print(
            f"  {row['side']:>5} {row['truss']:>9} "
            f"target=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f}) "
            f"tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f}) "
            f"short_by=({residual[0]:+.3f},{residual[1]:+.3f},{residual[2]:+.3f}) "
            f"|{math.dist(target, tip):.3f}| m"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
