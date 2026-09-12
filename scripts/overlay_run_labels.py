#!/usr/bin/env python3
"""
Draw the recorded GraspPoint / CutPoint labels onto a run's images.

The pixel labels come from a projection whose aperture-fit convention
is an assumption about how Replicator renders the wrist camera. This
tool is how that assumption gets checked: if the crosses land on the
truss and its peduncle, the projection is right.

Usage:
    python3 scripts/overlay_run_labels.py [run_dir]

Defaults to the newest directory under results/runs/. Writes into
<run_dir>/labels_overlay/ and never touches the original images.
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"

MARKER_COLORS = {
    "grasp": (0, 255, 0),
    "cut": (255, 40, 40),
}

MARKER_RADIUS = 7


def latest_run_dir():
    candidates = []

    if not RUNS_ROOT.exists():
        raise SystemExit(
            f"No runs under {RUNS_ROOT}"
        )

    for path in RUNS_ROOT.rglob(
        "trusses.csv"
    ):
        candidates.append(
            path.parent
        )

    if not candidates:
        raise SystemExit(
            f"No runs with trusses.csv under {RUNS_ROOT}"
        )

    return max(
        candidates,
        key=lambda path: (
            path.stat().st_mtime,
            str(
                path
            ),
        ),
    )


def read_pixel(row, name):
    u = row.get(f"{name}_px_u", "")
    v = row.get(f"{name}_px_v", "")

    if not u or not v:
        return None

    return float(u), float(v)


def draw_marker(draw, position, color, label):
    u, v = position

    draw.line(
        [(u - MARKER_RADIUS, v), (u + MARKER_RADIUS, v)],
        fill=color,
        width=2,
    )

    draw.line(
        [(u, v - MARKER_RADIUS), (u, v + MARKER_RADIUS)],
        fill=color,
        width=2,
    )

    draw.text(
        (u + MARKER_RADIUS + 2, v - MARKER_RADIUS),
        label,
        fill=color,
    )


def main():
    run_dir = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else latest_run_dir()
    )

    rows = list(
        csv.DictReader(
            (run_dir / "trusses.csv").open()
        )
    )

    output_dir = run_dir / "labels_overlay"
    output_dir.mkdir(exist_ok=True)

    written = 0
    skipped = 0

    for row in rows:
        image_field = row.get("image", "")

        if not image_field:
            skipped += 1
            continue

        image_path = run_dir / image_field

        if not image_path.exists():
            print(f"[MISS] {image_path}")
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for name, color in MARKER_COLORS.items():
            pixel = read_pixel(row, name)

            if pixel is None:
                continue

            draw_marker(
                draw,
                pixel,
                color,
                f"{name} {row.get(f'{name}_depth_m', '')[:5]}m",
            )

        outcome = row.get("stage") or "OK"

        draw.text(
            (6, 6),
            f"{row.get('truss', '')}  [{outcome}]",
            fill=(255, 255, 0),
        )

        image.save(output_dir / image_path.name)
        written += 1

    print(f"Run:     {run_dir}")
    print(f"Written: {written} overlay images -> {output_dir}")

    if skipped:
        print(f"Skipped: {skipped} rows without a usable image")


if __name__ == "__main__":
    main()
