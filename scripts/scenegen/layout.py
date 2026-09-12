"""Crop beds, bed population and greenhouse layout."""

from truss_physics import add_collider

from .params import *  # noqa: F401,F403
from .materials import bind_scene_material
from .geometry import create_box
from .plants import create_main_stem


# =========================================================
# BED
# =========================================================

def populate_bed(
    stage,
    bed_path,
    center,
    bed_top_z,
    corridor_yaw,
    tool_marker_spacing,
    rng,
):
    stems_path = (
        f"{bed_path}/Stems"
    )

    stage.DefinePrim(
        stems_path,
        "Xform",
    )

    stem_count = rng.randint(
        4,
        7,
    )

    margin = 0.20 * CROP_SCALE

    usable_length = (
        BED_LENGTH
        - 2.0 * margin
    )

    spacing = (
        usable_length
        / max(
            stem_count - 1,
            1,
        )
    )

    for i in range(
        stem_count
    ):
        x = (
            center[0]
            - usable_length / 2
            + i * spacing
            + rng.uniform(
                -0.04 * CROP_SCALE,
                0.04 * CROP_SCALE,
            )
        )

        y = (
            center[1]
            + rng.uniform(
                -BED_WIDTH * 0.25,
                BED_WIDTH * 0.25,
            )
        )

        create_main_stem(
            stage=stage,
            stem_path=(
                f"{stems_path}/"
                f"Stem_{i+1:02d}"
            ),
            root_position=(
                x,
                y,
                bed_top_z,
            ),
            corridor_yaw=(
                corridor_yaw
            ),
            tool_marker_spacing=(
                tool_marker_spacing
            ),
            rng=rng,
        )


def create_crop_bed(
    stage,
    path,
    center,
    corridor_yaw,
    tool_marker_spacing,
    rng,
):
    stage.DefinePrim(
        path,
        "Xform",
    )

    bed_prim = create_box(
        stage,
        f"{path}/Geometry",
        center,
        (
            BED_LENGTH,
            BED_WIDTH,
            BED_HEIGHT,
        ),
    )

    # Bed itself is static.
    add_collider(
        bed_prim
    )

    bind_scene_material(
        bed_prim,
        "bed",
    )

    populate_bed(
        stage=stage,
        bed_path=path,
        center=center,
        bed_top_z=(
            center[2]
            + BED_HEIGHT / 2
        ),
        corridor_yaw=(
            corridor_yaw
        ),
        tool_marker_spacing=(
            tool_marker_spacing
        ),
        rng=rng,
    )


# =========================================================
# GREENHOUSE
# =========================================================

def create_greenhouse(
    stage,
    tool_marker_spacing,
    rng,
):
    root = "/World/Greenhouse"

    stage.DefinePrim(
        root,
        "Xform",
    )

    # Edge-to-edge free space.
    row_pitch = (
        BED_WIDTH
        + CROSS_AISLE_GAP
    )

    # Leaves empty central corridor.
    side_offset = (
        CORRIDOR_WIDTH / 2
        + BED_LENGTH / 2
    )

    for row in range(
        ROW_COUNT
    ):
        y = (
            FIRST_ROW_Y
            + row * row_pitch
        )

        # LEFT
        create_crop_bed(
            stage,
            f"{root}/Row_{row+1:02d}_Left",
            center=(
                -side_offset,
                y,
                BED_HEIGHT / 2,
            ),
            corridor_yaw=0.0,
            tool_marker_spacing=(
                tool_marker_spacing
            ),
            rng=rng,
        )

        # RIGHT
        create_crop_bed(
            stage,
            f"{root}/Row_{row+1:02d}_Right",
            center=(
                side_offset,
                y,
                BED_HEIGHT / 2,
            ),
            corridor_yaw=180.0,
            tool_marker_spacing=(
                tool_marker_spacing
            ),
            rng=rng,
        )
