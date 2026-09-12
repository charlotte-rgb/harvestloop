"""Measure the fixed GraspTip <-> CutterTip spacing from the robot USD."""

from .params import *  # noqa: F401,F403
from .geometry import get_world_position


def find_tool_marker_path(
    stage,
    preferred_path,
    marker_name,
):
    """
    Use the known path first. If the USD hierarchy changes slightly,
    fall back to finding the marker by name under /World/HarvestBot.
    """
    prim = stage.GetPrimAtPath(
        preferred_path
    )

    if prim.IsValid():
        return preferred_path

    candidates = []

    for candidate in stage.Traverse():
        path = str(
            candidate.GetPath()
        )

        if not path.startswith(
            "/World/HarvestBot/"
        ):
            continue

        if (
            candidate.GetName()
            == marker_name
        ):
            candidates.append(
                path
            )

    if not candidates:
        raise RuntimeError(
            f"Could not find {marker_name} "
            "under /World/HarvestBot."
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple {marker_name} prims found: "
            f"{candidates}"
        )

    return candidates[0]


def measure_tool_marker_spacing(
    stage,
):
    """
    Measure the fixed physical distance between GraspTip and CutterTip
    directly from the referenced harvest_bot.usd.

    Both markers are children of the same rigid tool body, so this
    distance is invariant to UR5e joint motion.
    """
    grasp_path = find_tool_marker_path(
        stage,
        GRASP_TIP_PATH,
        "GraspTip",
    )

    cutter_path = find_tool_marker_path(
        stage,
        CUTTER_TIP_PATH,
        "CutterTip",
    )

    grasp_position = get_world_position(
        stage,
        grasp_path,
    )

    cutter_position = get_world_position(
        stage,
        cutter_path,
    )

    spacing = (
        cutter_position
        - grasp_position
    ).GetLength()

    if spacing < 1e-4:
        raise RuntimeError(
            "Measured GraspTip-CutterTip spacing "
            f"is invalid: {spacing:.6f} m"
        )

    print()
    print(
        "========================================"
    )
    print(
        "HARVEST TOOL GEOMETRY"
    )
    print(
        "========================================"
    )
    print(
        f"GraspTip:  {grasp_path}"
    )
    print(
        f"CutterTip: {cutter_path}"
    )
    print(
        "GraspTip <-> CutterTip spacing: "
        f"{spacing:.4f} m"
    )

    return spacing
