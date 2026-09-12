#!/usr/bin/env python3
"""
Offline score of rgbd_peduncle against a finished run's RGB-D + GT.

Usage (Isaac python.sh):

  /home/charlotte/isaac-sim/python.sh \\
    /home/charlotte/harvestloop_sim/scripts/eval_rgbd_estimator.py \\
    [/path/to/results/runs/<run_id>]
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

from harvestloop.rgbd_estimate import (  # noqa: E402
    estimate_peduncle_targets,
)


DEFAULT_INTRINSICS = {
    "focal_length": 12.0,
    "horizontal_aperture": 20.955,
    "vertical_aperture": 15.2908,
    "width": 640,
    "height": 480,
}


def latest_run(
    root: Path,
) -> Path:
    runs = sorted(
        (
            root / "results" / "runs"
        ).glob(
            "*"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise SystemExit(
            "No runs found under results/runs/"
        )

    return runs[0]


def main(
    argv,
):
    project = Path(
        "/home/charlotte/harvestloop_sim"
    )

    run_dir = (
        Path(
            argv[1]
        )
        if len(
            argv
        ) > 1
        else latest_run(
            project
        )
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
        g = np.array(
            [
                float(
                    row[
                        "gt_grasp_world_x"
                    ]
                ),
                float(
                    row[
                        "gt_grasp_world_y"
                    ]
                ),
                float(
                    row[
                        "gt_grasp_world_z"
                    ]
                ),
            ]
        )
        c = np.array(
            [
                float(
                    row[
                        "gt_cut_world_x"
                    ]
                ),
                float(
                    row[
                        "gt_cut_world_y"
                    ]
                ),
                float(
                    row[
                        "gt_cut_world_z"
                    ]
                ),
            ]
        )
        spacings.append(
            float(
                np.linalg.norm(
                    g - c
                )
            )
        )

    spacing = float(
        np.median(
            spacings
        )
    )

    print(
        f"Run: {run_dir}"
    )
    print(
        f"Trusses: {len(rows)} | tool spacing {spacing:.4f} m"
    )

    grasp_errs = []
    cut_errs = []
    fails = 0

    for row in rows:
        rgb = np.asarray(
            Image.open(
                run_dir / row[
                    "image"
                ]
            )
        )
        depth = np.load(
            run_dir / row[
                "depth_image"
            ]
        )
        pose = {
            "position": np.array(
                [
                    float(
                        row[
                            "camera_position_x"
                        ]
                    ),
                    float(
                        row[
                            "camera_position_y"
                        ]
                    ),
                    float(
                        row[
                            "camera_position_z"
                        ]
                    ),
                ]
            ),
            "orientation": np.array(
                [
                    float(
                        row[
                            "camera_quat_w"
                        ]
                    ),
                    float(
                        row[
                            "camera_quat_qx"
                        ]
                    ),
                    float(
                        row[
                            "camera_quat_qy"
                        ]
                    ),
                    float(
                        row[
                            "camera_quat_qz"
                        ]
                    ),
                ]
            ),
            "intrinsics": DEFAULT_INTRINSICS,
        }

        estimate = estimate_peduncle_targets(
            rgb,
            depth,
            pose,
            spacing,
        )

        gt_g = np.array(
            [
                float(
                    row[
                        "gt_grasp_world_x"
                    ]
                ),
                float(
                    row[
                        "gt_grasp_world_y"
                    ]
                ),
                float(
                    row[
                        "gt_grasp_world_z"
                    ]
                ),
            ]
        )
        gt_c = np.array(
            [
                float(
                    row[
                        "gt_cut_world_x"
                    ]
                ),
                float(
                    row[
                        "gt_cut_world_y"
                    ]
                ),
                float(
                    row[
                        "gt_cut_world_z"
                    ]
                ),
            ]
        )

        if estimate[
            "cut_point_world"
        ] is None:
            fails += 1
            print(
                f"FAIL {row['image_stem']}: "
                f"{estimate.get('diagnostics', {}).get('reason')}"
            )
            continue

        ge = float(
            np.linalg.norm(
                estimate[
                    "grasp_point_world"
                ]
                - gt_g
            )
        )
        ce = float(
            np.linalg.norm(
                estimate[
                    "cut_point_world"
                ]
                - gt_c
            )
        )
        grasp_errs.append(
            ge
        )
        cut_errs.append(
            ce
        )

        print(
            f"{row['image_stem']}: "
            f"cut={ce * 100:5.1f}cm "
            f"grasp={ge * 100:5.1f}cm "
            f"conf={estimate['confidence']:.2f}"
        )

    print(
        "---"
    )
    print(
        f"fails {fails}/{len(rows)}"
    )

    if cut_errs:
        print(
            "mean/med cut cm:   "
            f"{np.mean(cut_errs) * 100:.2f} / "
            f"{np.median(cut_errs) * 100:.2f}"
        )
        print(
            "mean/med grasp cm: "
            f"{np.mean(grasp_errs) * 100:.2f} / "
            f"{np.median(grasp_errs) * 100:.2f}"
        )
        print(
            "cut <5cm / <3cm:   "
            f"{sum(e < 0.05 for e in cut_errs)} / "
            f"{sum(e < 0.03 for e in cut_errs)} "
            f"of {len(cut_errs)}"
        )


if __name__ == "__main__":
    main(
        sys.argv
    )
