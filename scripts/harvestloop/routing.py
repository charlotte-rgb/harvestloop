"""Stem/truss discovery and aisle-1 harvest stop planning."""

import math
import re

import numpy as np

from .config import *  # noqa: F401,F403
from .mathutils import look_quaternion_from_points
from .usd_pose import pose_position


# ============================================================
# AISLE-1 STEM / STOP DISCOVERY + VIEWPOINT CAPTURE
# ============================================================

def _stem_index_from_name(
    name,
):
    match = re.match(
        r"Stem_(\d+)$",
        name,
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def _truss_index_from_name(
    name,
):
    match = re.match(
        r"Truss_(\d+)$",
        name,
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def discover_trusses_under_stem(
    stage,
    stem_path,
):
    """
    Return every Truss_* under one stem, with privileged GraspPoint.
    """
    trusses = []

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        path = str(
            prim.GetPath()
        )

        if not path.startswith(
            stem_path + "/"
        ):
            continue

        truss_index = _truss_index_from_name(
            prim.GetName()
        )

        if truss_index is None:
            continue

        # Exact child Truss_* of this stem (not nested elsewhere).
        parent_path = str(
            prim.GetPath().GetParentPath()
        )

        if parent_path != stem_path:
            continue

        grasp_path = f"{path}/GraspPoint"
        cut_path = f"{path}/CutPoint"
        peduncle_path = f"{path}/Peduncle"

        grasp_prim = stage.GetPrimAtPath(
            grasp_path
        )

        if not grasp_prim.IsValid():
            print(
                f"[WARN] Missing GraspPoint under {path}"
            )
            continue

        grasp_position = pose_position(
            stage,
            grasp_path,
        )

        cut_position = None

        if stage.GetPrimAtPath(
            cut_path
        ).IsValid():
            cut_position = pose_position(
                stage,
                cut_path,
            )

        look_target = grasp_position.copy()

        if cut_position is not None:
            look_target = 0.5 * (
                grasp_position
                + cut_position
            )

        trusses.append(
            {
                "truss_index": truss_index,
                "truss_name": prim.GetName(),
                "truss_path": path,
                "grasp_path": grasp_path,
                "cut_path": cut_path,
                "peduncle_path": peduncle_path,
                "grasp_position": grasp_position,
                "cut_position": cut_position,
                "look_target": look_target,
            }
        )

    trusses.sort(
        key=lambda item: item["truss_index"]
    )

    return trusses


def discover_stems_in_row(
    stage,
    row_root,
):
    """
    Discover Stem_* under one crop bed, with their trusses.
    """
    stems = []

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        path = str(
            prim.GetPath()
        )

        if not path.startswith(
            row_root + "/"
        ):
            continue

        stem_index = _stem_index_from_name(
            prim.GetName()
        )

        if stem_index is None:
            continue

        root_position = pose_position(
            stage,
            path,
        )

        trusses = discover_trusses_under_stem(
            stage,
            path,
        )

        stems.append(
            {
                "stem_index": stem_index,
                "stem_name": prim.GetName(),
                "stem_path": path,
                "root_position": root_position,
                "x": float(
                    root_position[0]
                ),
                "y": float(
                    root_position[1]
                ),
                "trusses": trusses,
            }
        )

    stems.sort(
        key=lambda item: item["x"]
    )

    return stems


def jackal_stop_x_before_stem(
    stem_workspace_x_value,
):
    """
    Convert a stem-workspace X into a Jackal park X that keeps the
    stem ahead of the robot while facing -X.
    """
    return (
        float(
            stem_workspace_x_value
        )
        + STOP_BEFORE_STEM_OFFSET
    )


def build_aisle1_harvest_stops(
    left_stems,
    right_stems,
):
    """
    One Jackal stop per stem, ordered by decreasing X
    (corridor-proximal first, then toward -X).

    At each stop the robot only images that stem's trusses.
    """
    stops = []

    for stem in left_stems:
        workspace_x = float(
            stem["x"]
        )

        stops.append(
            {
                "stem_workspace_x": workspace_x,
                "stop_x": jackal_stop_x_before_stem(
                    workspace_x
                ),
                "side_name": "left",
                "stem": stem,
                "approach_offset": np.array(
                    [0.0, -VIEW_OFFSET, 0.0],
                    dtype=np.float64,
                ),
            }
        )

    for stem in right_stems:
        workspace_x = float(
            stem["x"]
        )

        stops.append(
            {
                "stem_workspace_x": workspace_x,
                "stop_x": jackal_stop_x_before_stem(
                    workspace_x
                ),
                "side_name": "right",
                "stem": stem,
                "approach_offset": np.array(
                    [0.0, +VIEW_OFFSET, 0.0],
                    dtype=np.float64,
                ),
            }
        )

    # Sequence by stem X: larger X (near corridor) first.
    stops.sort(
        key=lambda item: (
            -item["stem_workspace_x"],
            item["side_name"],
        )
    )

    # Clamp to the LEFT aisle driveline only.
    x_min = LEFT_BED_OUTER_X + STOP_X_MARGIN
    x_max = CORRIDOR_X - STOP_X_MARGIN

    for index, stop in enumerate(
        stops
    ):
        stop["stop_index"] = index + 1

        raw_stop_x = float(
            stop["stop_x"]
        )

        stop["raw_stop_x"] = raw_stop_x
        stop["stop_x"] = float(
            np.clip(
                raw_stop_x,
                x_min,
                x_max,
            )
        )

        if abs(
            stop["stop_x"] - raw_stop_x
        ) > 1e-4:
            print(
                f"[STOP CLAMP] stop_{stop['stop_index']:02d}: "
                f"raw={raw_stop_x:+.3f} -> "
                f"clamped={stop['stop_x']:+.3f} "
                f"(limits [{x_min:+.3f}, {x_max:+.3f}])"
            )

    return stops


def build_target_manifest_from_stage(
    stage,
    *,
    scene_seed,
    occlusion_level,
    max_targets=None,
    suite_id=None,
):
    """
    Discover aisle-1 harvest targets on the current stage and return a
    lockable targets.json payload (same order as the route).
    """
    left_stems = discover_stems_in_row(
        stage,
        AISLE1_LEFT_ROW_ROOT,
    )
    right_stems = discover_stems_in_row(
        stage,
        AISLE1_RIGHT_ROW_ROOT,
    )
    harvest_stops = build_aisle1_harvest_stops(
        left_stems,
        right_stems,
    )
    planned = flatten_harvest_targets(
        harvest_stops
    )

    if max_targets is not None:
        planned = planned[
            : int(max_targets)
        ]

    return {
        "suite_id": suite_id,
        "scene_seed": int(scene_seed),
        "occlusion_level": str(
            occlusion_level
        ),
        "max_targets": max_targets,
        "discovered": len(
            flatten_harvest_targets(
                harvest_stops
            )
        ),
        "protocol": (
            "shared_view_rescue_matched_manifest"
        ),
        "gt_scoring_only": True,
        "matched_targets": True,
        "fixed_views": 1,
        "active_max_views": ACTIVE_PERCEPTION_MAX_VIEWS,
        "targets": planned,
        "truss_paths": [
            item["truss_path"]
            for item in planned
        ],
        "stop_indices": [
            item["stop_index"]
            for item in planned
        ],
    }


def flatten_harvest_targets(
    harvest_stops,
):
    """
    Stable ordered list of every truss the aisle-1 route would visit.

    Order matches inspect_and_harvest_stem_trusses_at_stop: stop
    order, then increasing look-target Z on that stem.
    """
    targets = []

    for stop in harvest_stops:
        stem = stop.get("stem") or {}

        trusses = sorted(
            list(stem.get("trusses") or []),
            key=lambda item: float(
                np.asarray(
                    item["look_target"],
                    dtype=np.float64,
                ).reshape(3)[2]
            ),
        )

        for truss in trusses:
            look = truss.get("look_target")

            targets.append(
                {
                    "stop_index": stop.get("stop_index"),
                    "stop_x": stop.get("stop_x"),
                    "side": stop.get("side_name"),
                    "stem_name": stem.get("stem_name"),
                    "stem_path": stem.get("stem_path"),
                    "truss_name": truss.get("truss_name"),
                    "truss_path": truss.get("truss_path"),
                    "truss_index": truss.get("truss_index"),
                    "look_target": (
                        [
                            float(look[0]),
                            float(look[1]),
                            float(look[2]),
                        ]
                        if look is not None
                        else None
                    ),
                }
            )

    return targets


def filter_stops_to_targets(
    harvest_stops,
    allowed_truss_paths,
):
    """Keep only stops/trusses that appear in the paired target list."""
    allowed = set(allowed_truss_paths)
    filtered = []

    for stop in harvest_stops:
        stem = dict(stop.get("stem") or {})
        trusses = [
            truss
            for truss in (stem.get("trusses") or [])
            if truss.get("truss_path") in allowed
        ]

        if not trusses:
            continue

        stem["trusses"] = trusses
        new_stop = dict(stop)
        new_stop["stem"] = stem
        filtered.append(new_stop)

    return filtered


def print_harvest_stop_plan(
    stops,
):
    print()
    print("========================================")
    print("[PLAN] AISLE-1 PER-STEM STOPS")
    print("========================================")

    print(
        f"View offset: {VIEW_OFFSET:.3f} m "
        f"(left approach -Y, right approach +Y)"
    )

    print(
        "Jackal stop offset before each stem (+X while facing -X): "
        f"{STOP_BEFORE_STEM_OFFSET:.3f} m"
    )

    print(
        "Order: decreasing stem X "
        "(corridor -> -X), one stem per stop"
    )

    print(
        f"Planned stops: {len(stops)}"
    )

    for stop in stops:
        stem = stop[
            "stem"
        ]

        print(
            f"  stop_{stop['stop_index']:02d} @ "
            f"x={stop['stop_x']:+.3f} "
            f"(stem_x={stop['stem_workspace_x']:+.3f}, "
            f"raw={stop.get('raw_stop_x', stop['stop_x']):+.3f}, "
            f"before=+{STOP_BEFORE_STEM_OFFSET:.3f} m) | "
            f"side={stop['side_name']} | "
            f"{stem['stem_name']} | "
            f"trusses={len(stem['trusses'])}"
        )


def compute_stem_camera_column_xy(
    stem,
    approach_offset,
    standoff=None,
):
    """
    Fixed aisle-side (x, y) for the wrist CAMERA on one stem.

    X stays on the stem root X, so all views on one stem share a
    vertical column. Y is pulled out to the camera standoff on the
    aisle side of the stem, because the viewpoint is defined by where
    the camera has to be to see the truss, not by where the tool tip
    ends up.
    """
    offset = np.asarray(
        approach_offset,
        dtype=np.float64,
    ).reshape(3)

    standoff = float(
        standoff
        if standoff is not None
        else VIEW_CAMERA_STANDOFF
    )

    # Keep the aisle side of the stem, replace the distance.
    aisle_sign = (
        1.0
        if float(offset[1]) >= 0.0
        else -1.0
    )

    return np.array(
        [
            float(
                stem["x"]
            ),
            float(
                stem["y"]
            )
            + aisle_sign * standoff,
        ],
        dtype=np.float64,
    )


def compute_camera_viewpoint_pose(
    stem_camera_xy,
    look_target,
):
    """
    Canonical camera pose for one truss: fixed stem column XY, truss Z,
    aimed at the truss. This is view 1 for both conditions.
    """
    look_target = np.asarray(
        look_target,
        dtype=np.float64,
    ).reshape(3)

    camera_position = np.array(
        [
            float(
                stem_camera_xy[0]
            ),
            float(
                stem_camera_xy[1]
            ),
            float(
                look_target[2]
            ),
        ],
        dtype=np.float64,
    )

    camera_orientation = (
        look_quaternion_from_points(
            camera_position,
            look_target,
        )
    )

    return (
        camera_position,
        camera_orientation,
    )


def compute_named_camera_viewpoint_pose(
    stem_camera_xy,
    look_target,
    viewpoint_name,
    lateral_offset_m=None,
):
    """
    Deterministic alternate wrist poses for active perception.

    canonical        — same as compute_camera_viewpoint_pose
    lateral_plus_x   — canonical + (offset, 0, 0), re-aimed
    lateral_minus_x  — canonical + (-offset, 0, 0), re-aimed
    """
    camera_position, _ = compute_camera_viewpoint_pose(
        stem_camera_xy,
        look_target,
    )

    offset = float(
        ACTIVE_VIEW_LATERAL_OFFSET_M
        if lateral_offset_m is None
        else lateral_offset_m
    )

    if viewpoint_name == "canonical":
        pass
    elif viewpoint_name == "lateral_plus_x":
        camera_position = camera_position + np.array(
            [offset, 0.0, 0.0],
            dtype=np.float64,
        )
    elif viewpoint_name == "lateral_minus_x":
        camera_position = camera_position + np.array(
            [-offset, 0.0, 0.0],
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"Unknown active viewpoint: {viewpoint_name!r}"
        )

    look_target = np.asarray(
        look_target,
        dtype=np.float64,
    ).reshape(3)

    camera_orientation = (
        look_quaternion_from_points(
            camera_position,
            look_target,
        )
    )

    return (
        camera_position,
        camera_orientation,
    )


def dense_aisle_waypoints(
    start_x,
    end_x,
    step=AISLE_WAYPOINT_STEP,
):
    """
    Dense pass-through waypoints along y=AISLE_Y from start_x to end_x.
    """
    start_x = float(
        start_x
    )
    end_x = float(
        end_x
    )

    if abs(
        end_x - start_x
    ) < 1e-6:
        return [
            (end_x, AISLE_Y)
        ]

    direction = math.copysign(
        1.0,
        end_x - start_x,
    )

    waypoints = []
    x = start_x

    # Advance until the final plane; always include end.
    while (
        direction * (end_x - x)
        > step + 1e-9
    ):
        x = x + direction * step
        waypoints.append(
            (x, AISLE_Y)
        )

    if (
        not waypoints
        or abs(
            waypoints[-1][0] - end_x
        ) > 1e-6
    ):
        waypoints.append(
            (end_x, AISLE_Y)
        )

    return waypoints
