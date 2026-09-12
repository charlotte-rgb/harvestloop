"""Assemble the full narrow-aisle scene."""

import random

import omni.usd
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage

from .params import *  # noqa: F401,F403
from .materials import bind_scene_material_recursive, create_scene_materials
from .tooling import measure_tool_marker_spacing
from .layout import create_greenhouse


# =========================================================
# MAIN
# =========================================================

def build_scene():
    rng = random.Random(
        SEED
    )

    world = World(
        stage_units_in_meters=1.0
    )

    stage = (
        omni.usd
        .get_context()
        .get_stage()
    )

    # Create reusable deterministic materials before generating geometry.
    SCENE_MATERIALS.clear()
    SCENE_MATERIALS.update(
        create_scene_materials(
            stage
        )
    )

    GroundPlane(
        prim_path="/World/GroundPlane",
        name="ground_plane",
    )

    ground_prim = stage.GetPrimAtPath(
        "/World/GroundPlane"
    )

    bind_scene_material_recursive(
        ground_prim,
        "ground",
    )

    add_reference_to_stage(
        usd_path=HARVEST_BOT_USD,
        prim_path="/World/HarvestBot",
    )

    # Measure the REAL fixed GraspTip-CutterTip spacing from the
    # referenced robot/tool before generating any crop markers.
    tool_marker_spacing = (
        measure_tool_marker_spacing(
            stage
        )
    )

    create_greenhouse(
        stage,
        tool_marker_spacing=(
            tool_marker_spacing
        ),
        rng=rng,
    )

    world.reset()

    print(
        "HarvestLoop scene generated."
    )

    print(
        f"Scene seed: {SEED}"
    )

    print(
        f"Occlusion: {OCCLUSION_LEVEL} "
        f"(leaves {leaf_count_range()[0]}–"
        f"{leaf_count_range()[1]}, "
        f"size×{leaf_size_scale():.2f})"
    )

    print(
        f"Crop scale: {CROP_SCALE:.2f}"
    )

    print(
        "Bed dimensions: "
        f"{BED_LENGTH:.2f} x "
        f"{BED_WIDTH:.2f} x "
        f"{BED_HEIGHT:.2f} m"
    )

    print(
        "Driving corridor width: "
        f"{CORRIDOR_WIDTH:.2f} m"
    )

    print(
        "Cross-aisle gap (narrow): "
        f"{CROSS_AISLE_GAP:.2f} m"
    )

    aisle_y = (
        FIRST_ROW_Y
        + BED_WIDTH / 2.0
        + CROSS_AISLE_GAP / 2.0
    )

    print(
        "Aisle-1 centerline y: "
        f"{aisle_y:.3f} m"
    )

    print(
        "Matched GraspPoint-CutPoint spacing: "
        f"{tool_marker_spacing:.4f} m"
    )

    print(
        "Harvestable truss attach height: "
        f"<= {MAX_HARVESTABLE_ATTACH_HEIGHT:.3f} m "
        f"(shoulder reach <= {MAX_HARVEST_SHOULDER_REACH:.2f} m "
        "from aisle park)"
    )

    print(
        "CutPoint generation: "
        "CutPoint = GraspPoint - "
        "peduncle_direction * tool_marker_spacing"
    )

    print(
        "Visual materials: "
        "ground=gray, bed=brown, stems/branches=green, "
        "leaves=dark green, tomatoes=red"
    )

    print(
        "Ground-truth GraspPoint/CutPoint markers visible: "
        f"{SHOW_GROUND_TRUTH_MARKERS}"
    )

    return world


def rebuild_greenhouse(
    seed=None,
    occlusion=None,
):
    """
    Replace /World/Greenhouse with a fresh deterministic crop.

    Keeps /World/HarvestBot. Used between paired fixed/active runs
    so cut plants are restored without restarting Isaac.
    """
    global SEED
    global OCCLUSION_LEVEL

    if seed is not None:
        SEED = int(seed)

    if occlusion is not None:
        apply_occlusion_level(occlusion)

    stage = (
        omni.usd
        .get_context()
        .get_stage()
    )

    if stage is None:
        raise RuntimeError(
            "No USD stage — load / rebuild Isaac first."
        )

    greenhouse = "/World/Greenhouse"

    if stage.GetPrimAtPath(greenhouse).IsValid():
        stage.RemovePrim(greenhouse)
        print(f"[SCENE] Removed {greenhouse}")

    if not SCENE_MATERIALS:
        SCENE_MATERIALS.update(
            create_scene_materials(stage)
        )

    if not stage.GetPrimAtPath("/World/HarvestBot").IsValid():
        add_reference_to_stage(
            usd_path=HARVEST_BOT_USD,
            prim_path="/World/HarvestBot",
        )

    tool_marker_spacing = measure_tool_marker_spacing(stage)
    rng = random.Random(SEED)

    create_greenhouse(
        stage,
        tool_marker_spacing=tool_marker_spacing,
        rng=rng,
    )

    print(
        "[SCENE] Rebuilt greenhouse | "
        f"seed={SEED} | occlusion={OCCLUSION_LEVEL}"
    )

    return {
        "scene_seed": SEED,
        "occlusion_level": OCCLUSION_LEVEL,
        "tool_marker_spacing": tool_marker_spacing,
    }
