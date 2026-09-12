#!/usr/bin/env python3
"""
Offline score of the keypoint estimator against a finished run.

K1 = CutPoint, K2 = GraspPoint. Replays each saved RGB-D capture and
compares with hidden USD GT cut / grasp from trusses.csv.

Usage (Isaac python.sh):

  /home/charlotte/isaac-sim/python.sh \\
    /home/charlotte/harvestloop_sim/scripts/eval_keypoint_estimator.py \\
    [run_dir]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS = Path(
    __file__
).resolve().parent

if str(
    SCRIPTS
) not in sys.path:
    sys.path.insert(
        0,
        str(
            SCRIPTS
        ),
    )

from harvestloop.keypoint_estimate import (  # noqa: E402
    estimate_keypoint_targets,
)

PROJECT_ROOT = SCRIPTS.parent

DEFAULT_INTRINSICS = {
    "focal_length": 12.0,
    "horizontal_aperture": 20.955,
    "vertical_aperture": 15.2908,
    "width": 640,
    "height": 480,
}


def latest_run_with_csv():
    runs = [
        path
        for path in (
            PROJECT_ROOT / "results" / "runs"
        ).glob(
            "*"
        )
        if (
            path / "trusses.csv"
        ).exists()
    ]

    if not runs:
        raise SystemExit(
            "No runs with trusses.csv under results/runs/"
        )

    return max(
        runs,
        key=lambda path: path.name,
    )


def read_vector(
    row,
    prefix,
    components=("x", "y", "z"),
):
    values = []

    for component in components:
        raw = row.get(
            f"{prefix}_{component}",
            "",
        )

        if raw in (
            "",
            None,
        ):
            return None

        values.append(
            float(
                raw
            )
        )

    return np.array(
        values,
        dtype=np.float64,
    )


def camera_pose_from_row(
    row,
):
    position = read_vector(
        row,
        "camera_position",
    )

    orientation = read_vector(
        row,
        "camera_quat",
        components=(
            "w",
            "qx",
            "qy",
            "qz",
        ),
    )

    if (
        position is None
        or orientation is None
    ):
        return None

    return {
        "position": position,
        "orientation": orientation,
        "intrinsics": DEFAULT_INTRINSICS,
    }


def summarize(
    label,
    errors,
):
    if not errors:
        return f"  {label:<22} n=0"

    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    return (
        f"  {label:<22} n={errors.size:<3} "
        f"mean {errors.mean() * 100:5.1f} cm  "
        f"median {np.median(errors) * 100:5.1f} cm  "
        f"<3cm {int((errors < 0.03).sum())}  "
        f"<5cm {int((errors < 0.05).sum())}"
    )


def main(
    argv,
):
    run_dir = (
        Path(
            argv[1]
        ).resolve()
        if len(
            argv
        ) > 1
        else latest_run_with_csv()
    )

    rows = list(
        csv.DictReader(
            (
                run_dir / "trusses.csv"
            ).open()
        )
    )

    spacings = []

    for row in rows:
        grasp = read_vector(
            row,
            "gt_grasp_world",
        )

        cut = read_vector(
            row,
            "gt_cut_world",
        )

        if (
            grasp is not None
            and cut is not None
        ):
            spacings.append(
                float(
                    np.linalg.norm(
                        grasp - cut
                    )
                )
            )

    if not spacings:
        raise SystemExit(
            f"{run_dir}: no GT grasp/cut pairs to score against."
        )

    tool_spacing = float(
        np.median(
            spacings
        )
    )

    print(
        f"Run:          {run_dir.name}"
    )
    print(
        f"Trusses:      {len(rows)}"
    )
    print(
        f"Tool spacing: {tool_spacing:.4f} m"
    )
    print(
        ""
    )

    k1_errors = []
    k2_errors = []
    cut_errors = []
    grasp_errors = []
    detect_failures = {}

    for row in rows:
        depth_field = row.get(
            "depth_image",
            "",
        )

        if not depth_field:
            continue

        depth_path = run_dir / depth_field
        image_path = run_dir / row["image"]

        if not (
            depth_path.exists()
            and image_path.exists()
        ):
            continue

        camera_pose = camera_pose_from_row(
            row
        )

        gt_grasp = read_vector(
            row,
            "gt_grasp_world",
        )

        gt_cut = read_vector(
            row,
            "gt_cut_world",
        )

        if (
            camera_pose is None
            or gt_grasp is None
            or gt_cut is None
        ):
            continue

        estimate = estimate_keypoint_targets(
            np.asarray(
                Image.open(
                    image_path
                ).convert(
                    "RGB"
                )
            ),
            np.load(
                depth_path
            ),
            camera_pose,
            tool_spacing,
        )

        stem = row.get(
            "image_stem",
            image_path.stem,
        )

        if estimate[
            "cut_point_world"
        ] is None:
            reason = (
                estimate.get(
                    "reason"
                )
                or estimate.get(
                    "diagnostics",
                    {},
                ).get(
                    "reason",
                    "failed",
                )
            )

            detect_failures[reason] = (
                detect_failures.get(
                    reason,
                    0,
                )
                + 1
            )

            print(
                f"FAIL {stem}: {reason}"
            )
            continue

        k1 = estimate["keypoints"]["k1"]["world"]
        k2 = estimate["keypoints"]["k2"]["world"]

        k1_err = float(
            np.linalg.norm(
                k1 - gt_cut
            )
        )

        k2_err = float(
            np.linalg.norm(
                k2 - gt_grasp
            )
        )

        cut_err = float(
            np.linalg.norm(
                estimate["cut_point_world"]
                - gt_cut
            )
        )

        grasp_err = float(
            np.linalg.norm(
                estimate["grasp_point_world"]
                - gt_grasp
            )
        )

        k1_errors.append(
            k1_err
        )
        k2_errors.append(
            k2_err
        )
        cut_errors.append(
            cut_err
        )
        grasp_errors.append(
            grasp_err
        )

        print(
            f"{stem}: "
            f"conf "
            f"{estimate['keypoints']['k1']['confidence']:.2f}/"
            f"{estimate['keypoints']['k2']['confidence']:.2f}  "
            f"K1/cut {k1_err * 100:4.1f}cm  "
            f"K2/grasp {k2_err * 100:4.1f}cm"
        )

    print(
        ""
    )
    print(
        "Keypoints (= targets):"
    )
    print(
        summarize(
            "K1 / cut",
            k1_errors,
        )
    )
    print(
        summarize(
            "K2 / grasp",
            k2_errors,
        )
    )
    print(
        summarize(
            "cut (controller)",
            cut_errors,
        )
    )
    print(
        summarize(
            "grasp (controller)",
            grasp_errors,
        )
    )

    if detect_failures:
        print(
            "  failures: "
            + ", ".join(
                f"{reason} x{count}"
                for reason, count in sorted(
                    detect_failures.items()
                )
            )
        )


if __name__ == "__main__":
    main(
        sys.argv
    )
