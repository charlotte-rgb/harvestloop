#!/usr/bin/env python3
"""
Draw what the keypoint estimator saw, onto a run's wrist captures.

Per image:

    blue     peduncle segment supporting the keypoints
    cyan     K1 = CutPoint
    magenta  K2 = GraspPoint
    thin GT  hidden USD cut / grasp

Writes into <run_dir>/keypoints_overlay/.

Usage (Isaac python.sh):

  /home/charlotte/isaac-sim/python.sh \\
    /home/charlotte/harvestloop_sim/scripts/visualize_keypoints.py \\
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

from eval_keypoint_estimator import (  # noqa: E402
    camera_pose_from_row,
    latest_run_with_csv,
    read_vector,
)
from harvestloop.keypoint_estimate import (  # noqa: E402
    estimate_keypoint_targets,
    render_keypoint_overlay,
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

    spacings = [
        float(
            np.linalg.norm(
                read_vector(
                    row,
                    "gt_grasp_world",
                )
                - read_vector(
                    row,
                    "gt_cut_world",
                )
            )
        )
        for row in rows
        if read_vector(
            row,
            "gt_grasp_world",
        ) is not None
        and read_vector(
            row,
            "gt_cut_world",
        ) is not None
    ]

    tool_spacing = float(
        np.median(
            spacings
        )
    ) if spacings else 0.03

    output_dir = run_dir / "keypoints_overlay"
    output_dir.mkdir(
        exist_ok=True
    )

    written = 0

    for row in rows:
        if not row.get(
            "depth_image"
        ):
            continue

        image_path = run_dir / row["image"]
        depth_path = run_dir / row["depth_image"]

        if not (
            image_path.exists()
            and depth_path.exists()
        ):
            continue

        camera_pose = camera_pose_from_row(
            row
        )

        if camera_pose is None:
            continue

        gt_grasp = read_vector(
            row,
            "gt_grasp_world",
        )

        gt_cut = read_vector(
            row,
            "gt_cut_world",
        )

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

        render_keypoint_overlay(
            np.asarray(
                Image.open(
                    image_path
                ).convert(
                    "RGB"
                )
            ),
            estimate,
            camera_pose,
            label=row.get(
                "truss",
                "",
            ),
            gt_k1_world=gt_cut,
            gt_k2_world=gt_grasp,
            gt_grasp_world=gt_grasp,
            gt_cut_world=gt_cut,
        ).save(
            output_dir / image_path.name
        )

        written += 1

    print(
        f"Run:     {run_dir}"
    )
    print(
        f"Written: {written} overlays -> {output_dir}"
    )


if __name__ == "__main__":
    main(
        sys.argv
    )
