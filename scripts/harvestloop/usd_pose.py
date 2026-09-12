"""USD pose queries and frame conversions."""

import numpy as np

from pxr import Gf, Usd, UsdGeom

from usd_utils import get_world_pose

from .config import *  # noqa: F401,F403
from .mathutils import (
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    quaternion_to_rpy,
)


def get_base_state(stage):
    position, orientation = get_world_pose(
        stage,
        JACKAL_BASE_PATH,
    )

    roll, pitch, yaw = quaternion_to_rpy(
        orientation
    )

    position = np.asarray(
        position,
        dtype=np.float64,
    ).reshape(3)

    return {
        "position": position,
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "tilt": max(
            abs(roll),
            abs(pitch),
        ),
    }


# ============================================================
# PRIM POSE QUERIES
# ============================================================

def pose_position(
    stage,
    prim_path,
):
    position, _ = get_world_pose(
        stage,
        prim_path,
    )

    return np.asarray(
        position,
        dtype=np.float64,
    ).reshape(3)


def pose_position_orientation(
    stage,
    prim_path,
):
    position, orientation = get_world_pose(
        stage,
        prim_path,
    )

    return (
        np.asarray(
            position,
            dtype=np.float64,
        ).reshape(3),
        quaternion_normalize(
            np.asarray(
                orientation,
                dtype=np.float64,
            )
        ),
    )


def peduncle_endpoints_world(
    stage,
    truss_path,
    near_point=None,
):
    """
    Hidden ground truth for the two task keypoints.

    Returns (K1, K2) in world space, where K1 is the stem-side end of
    the peduncle cylinder and K2 the tomato-side end. Scoring only —
    the estimator never reads this.

    `near_point` (the GT CutPoint, which sits a centimetre out from
    the stem end) decides which end is which, instead of relying on
    how the cylinder's local axis was authored.
    """
    prim = stage.GetPrimAtPath(
        f"{truss_path}/Peduncle"
    )

    if not prim.IsValid():
        return None, None

    cylinder = UsdGeom.Cylinder(
        prim
    )

    height = cylinder.GetHeightAttr().Get()

    if height is None:
        return None, None

    axis = (
        cylinder.GetAxisAttr().Get()
        or "Z"
    )

    half = float(
        height
    ) * 0.5

    offsets = {
        "X": (half, 0.0, 0.0),
        "Y": (0.0, half, 0.0),
        "Z": (0.0, 0.0, half),
    }[axis]

    to_world = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    ends = [
        np.asarray(
            to_world.Transform(
                Gf.Vec3d(
                    *[
                        sign * value
                        for value in offsets
                    ]
                )
            ),
            dtype=np.float64,
        )
        for sign in (-1.0, 1.0)
    ]

    if near_point is None:
        return ends[0], ends[1]

    near_point = np.asarray(
        near_point,
        dtype=np.float64,
    )

    if (
        np.linalg.norm(
            ends[1] - near_point
        )
        < np.linalg.norm(
            ends[0] - near_point
        )
    ):
        ends.reverse()

    return ends[0], ends[1]


def world_point_to_frame(
    stage,
    frame_path,
    world_point,
):
    """
    Express a world-space point in `frame_path` coordinates.

    This removes Jackal / UR5e-base world motion from validation.
    """
    frame_position, frame_orientation = (
        pose_position_orientation(
            stage,
            frame_path,
        )
    )

    world_point = np.asarray(
        world_point,
        dtype=np.float64,
    )

    rotation_world_from_frame = (
        quaternion_to_rotation_matrix(
            frame_orientation
        )
    )

    return (
        rotation_world_from_frame.T
        @ (
            world_point
            - frame_position
        )
    )


def grasp_tip_position_in_arm_base(
    stage,
):
    """
    Current physical GraspTip position expressed in UR5e base_link frame.
    """
    tip_world = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    return world_point_to_frame(
        stage,
        ARM_BASE_PATH,
        tip_world,
    )



def find_project_file(filename):
    preferred = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "config" / filename,
        PROJECT_ROOT / "configs" / filename,
        PROJECT_ROOT / "rmpflow" / filename,
        PROJECT_ROOT / "scripts" / filename,
        PROJECT_ROOT / "assets" / filename,
    ]

    for path in preferred:
        if path.is_file():
            return path

    matches = list(
        PROJECT_ROOT.rglob(
            filename
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} "
            f"under {PROJECT_ROOT}"
        )

    matches.sort(
        key=lambda p: (
            len(p.parts),
            len(str(p)),
        )
    )

    return matches[0]
