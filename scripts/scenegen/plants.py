"""Procedural tomato plants: leaves, branches, trusses and main stems."""

import random

from pxr import Gf, UsdGeom

from truss_physics import add_collider, make_truss_attached

from .params import *  # noqa: F401,F403
from .materials import bind_scene_material, set_render_visibility
from .geometry import (
    create_cylinder_between,
    create_sphere,
    generate_direction,
    normalize,
    point_on_segment,
)


# =========================================================
# LEAF
# =========================================================

def create_leaf(
    stage,
    path,
    attachment_point,
    direction,
    length,
    width,
):
    direction = normalize(
        direction
    )

    center = (
        Gf.Vec3d(*attachment_point)
        + direction * length * 0.5
    )

    leaf = UsdGeom.Cube.Define(
        stage,
        path,
    )

    leaf.CreateSizeAttr(1.0)

    xform = UsdGeom.Xformable(
        leaf.GetPrim()
    )

    xform.AddTranslateOp().Set(
        center
    )

    rotation = Gf.Rotation(
        Gf.Vec3d(1, 0, 0),
        direction,
    )

    quat = rotation.GetQuat()
    imag = quat.GetImaginary()

    xform.AddOrientOp().Set(
        Gf.Quatf(
            float(quat.GetReal()),
            Gf.Vec3f(
                float(imag[0]),
                float(imag[1]),
                float(imag[2]),
            ),
        )
    )

    xform.AddScaleOp().Set(
        Gf.Vec3f(
            length,
            width,
            0.01 * CROP_SCALE,
        )
    )

    bind_scene_material(
        leaf.GetPrim(),
        "leaf",
    )


# =========================================================
# TOMATO BRANCH
# =========================================================

def create_tomato_branch(
    stage,
    truss_path,
    index,
    branch_start,
    branch_direction,
    branch_length,
    tomato_radius,
):
    branch_direction = normalize(
        branch_direction
    )

    tomato_center = (
        Gf.Vec3d(*branch_start)
        + branch_direction
        * branch_length
    )

    # Branch ends on tomato surface.
    branch_end = (
        tomato_center
        - branch_direction
        * tomato_radius
    )

    branch_prim = (
        create_cylinder_between(
            stage,
            f"{truss_path}/Branch_{index:02d}",
            branch_start,
            tuple(branch_end),
            radius=0.003 * CROP_SCALE,
        )
    )

    add_collider(
        branch_prim
    )

    bind_scene_material(
        branch_prim,
        "branch",
    )

    tomato_prim = create_sphere(
        stage,
        f"{truss_path}/Tomato_{index:02d}",
        tuple(tomato_center),
        tomato_radius,
    )

    add_collider(
        tomato_prim
    )

    bind_scene_material(
        tomato_prim,
        "tomato",
    )


# =========================================================
# TRUSS
# =========================================================

def create_truss(
    stage,
    stem_path,
    truss_index,
    stem_height,
    corridor_yaw,
    tool_marker_spacing,
    rng,
):
    truss_path = (
        f"{stem_path}/"
        f"Truss_{truss_index:02d}"
    )

    truss_prim = stage.DefinePrim(
        truss_path,
        "Xform",
    )

    # IMPORTANT:
    # kinematic = attached to plant.
    make_truss_attached(
        truss_prim
    )

    # Keep GraspPoint inside the aisle-parked UR5e workspace.
    # Without this clamp, attach heights up to ~0.92 m put targets
    # at ~0.7–0.8 m shoulder reach, which the 0.6-scale arm never
    # reaches from the aisle centerline.
    attach_lo = stem_height * 0.45
    attach_hi = min(
        stem_height * 0.90,
        MAX_HARVESTABLE_ATTACH_HEIGHT,
    )

    if attach_lo > MAX_HARVESTABLE_ATTACH_HEIGHT:
        print(
            f"[TRUSS SKIP] {truss_path}: lowest attach "
            f"{attach_lo:.3f} m already above harvestable "
            f"max {MAX_HARVESTABLE_ATTACH_HEIGHT:.3f} m"
        )
        stage.RemovePrim(
            truss_path
        )
        return False

    if attach_hi < stem_height * 0.90:
        print(
            f"[TRUSS REACH] {truss_path}: clamping attach "
            f"to <= {attach_hi:.3f} m "
            f"(was up to {stem_height * 0.90:.3f} m)"
        )

    attach_height = rng.uniform(
        attach_lo,
        attach_hi,
    )

    peduncle_start = (
        0.0,
        0.0,
        attach_height,
    )

    peduncle_yaw = (
        corridor_yaw
        + rng.uniform(
            -20.0,
            20.0,
        )
    )

    peduncle_pitch = rng.uniform(
        -10.0,
        10.0,
    )

    # -----------------------------------------------------
    # Peduncle length
    # -----------------------------------------------------
    #
    # The peduncle must contain:
    #
    #   start
    #     -> small cut offset
    #     -> CutPoint
    #     -> fixed tool spacing
    #     -> GraspPoint
    #     -> distal margin
    #
    # This guarantees that the matched GraspPoint/CutPoint pair
    # actually lies on the physical peduncle segment.
    original_min_length = (
        0.16 * CROP_SCALE
    )

    original_max_length = (
        0.27 * CROP_SCALE
    )

    required_length = (
        CUTPOINT_OFFSET_FROM_PEDUNCLE_START
        + tool_marker_spacing
        + GRASPPOINT_DISTAL_MARGIN
    )

    peduncle_min_length = max(
        original_min_length,
        required_length,
    )

    peduncle_max_length = max(
        original_max_length,
        peduncle_min_length
        + 0.05 * CROP_SCALE,
    )

    peduncle_length = rng.uniform(
        peduncle_min_length,
        peduncle_max_length,
    )

    peduncle_direction = (
        generate_direction(
            peduncle_yaw,
            peduncle_pitch,
        )
    )

    peduncle_end = tuple(
        Gf.Vec3d(
            *peduncle_start
        )
        + peduncle_direction
        * peduncle_length
    )

    peduncle_prim = (
        create_cylinder_between(
            stage,
            f"{truss_path}/Peduncle",
            peduncle_start,
            peduncle_end,
            radius=0.005 * CROP_SCALE,
        )
    )

    add_collider(
        peduncle_prim
    )

    bind_scene_material(
        peduncle_prim,
        "peduncle",
    )

    # =====================================================
    # MATCHED GraspPoint -> CutPoint GENERATION
    # =====================================================
    #
    # First choose GraspPoint on the peduncle.
    #
    # Then generate CutPoint from that GraspPoint using the
    # FIXED physical GraspTip-CutterTip spacing:
    #
    #   CutPoint
    #       =
    #   GraspPoint
    #       - peduncle_direction * tool_marker_spacing
    #
    # Therefore:
    #
    #   ||GraspPoint - CutPoint||
    #       ==
    #   ||GraspTip - CutterTip||
    #
    # exactly (up to floating-point precision).
    grasp_distance_from_start = (
        CUTPOINT_OFFSET_FROM_PEDUNCLE_START
        + tool_marker_spacing
    )

    grasp_point_vec = (
        Gf.Vec3d(
            *peduncle_start
        )
        + peduncle_direction
        * grasp_distance_from_start
    )

    grasp_point = tuple(
        grasp_point_vec
    )

    cut_point_vec = (
        grasp_point_vec
        - peduncle_direction
        * tool_marker_spacing
    )

    cut_point = tuple(
        cut_point_vec
    )

    # Sanity checks.
    generated_spacing = (
        grasp_point_vec
        - cut_point_vec
    ).GetLength()

    cut_distance_from_start = (
        cut_point_vec
        - Gf.Vec3d(
            *peduncle_start
        )
    ).GetLength()

    grasp_distance_to_end = (
        Gf.Vec3d(
            *peduncle_end
        )
        - grasp_point_vec
    ).GetLength()

    if abs(
        generated_spacing
        - tool_marker_spacing
    ) > 1e-6:
        raise RuntimeError(
            f"{truss_path}: matched point "
            "spacing generation failed."
        )

    if (
        grasp_distance_from_start
        > peduncle_length
    ):
        raise RuntimeError(
            f"{truss_path}: GraspPoint lies "
            "outside peduncle."
        )

    # -------------------------
    # GraspPoint
    # -------------------------

    grasp_marker_prim = create_sphere(
        stage,
        f"{truss_path}/GraspPoint",
        grasp_point,
        radius=0.008 * CROP_SCALE,
    )

    bind_scene_material(
        grasp_marker_prim,
        "grasp_marker",
    )

    set_render_visibility(
        grasp_marker_prim,
        SHOW_GROUND_TRUTH_MARKERS,
    )

    # -------------------------
    # CutPoint
    # -------------------------

    cut_marker_prim = create_sphere(
        stage,
        f"{truss_path}/CutPoint",
        cut_point,
        radius=0.008 * CROP_SCALE,
    )

    bind_scene_material(
        cut_marker_prim,
        "cut_marker",
    )

    set_render_visibility(
        cut_marker_prim,
        SHOW_GROUND_TRUTH_MARKERS,
    )

    print(
        f"[TRUSS MARKERS] {truss_path} | "
        f"tool={tool_marker_spacing:.4f} m | "
        f"pair={generated_spacing:.4f} m | "
        f"cut_from_start={cut_distance_from_start:.4f} m | "
        f"grasp_to_end={grasp_distance_to_end:.4f} m"
    )

    # -------------------------
    # Tomatoes
    # -------------------------

    tomato_count = rng.randint(
        2,
        5,
    )

    for i in range(
        tomato_count
    ):
        branch_start = (
            point_on_segment(
                peduncle_start,
                peduncle_end,
                rng.uniform(
                    0.35,
                    0.95,
                ),
            )
        )

        branch_direction = (
            generate_direction(
                peduncle_yaw
                + rng.uniform(
                    -45.0,
                    45.0,
                ),
                rng.uniform(
                    -65.0,
                    -30.0,
                ),
            )
        )

        create_tomato_branch(
            stage=stage,
            truss_path=truss_path,
            index=i + 1,
            branch_start=branch_start,
            branch_direction=branch_direction,
            branch_length=rng.uniform(
                0.08 * CROP_SCALE,
                0.15 * CROP_SCALE,
            ),
            tomato_radius=rng.uniform(
                0.035 * CROP_SCALE,
                0.05 * CROP_SCALE,
            ),
        )

    return True


# =========================================================
# MAIN STEM
# =========================================================

def create_main_stem(
    stage,
    stem_path,
    root_position,
    corridor_yaw,
    tool_marker_spacing,
    rng,
):
    stem_root = stage.DefinePrim(
        stem_path,
        "Xform",
    )

    UsdGeom.Xformable(
        stem_root
    ).AddTranslateOp().Set(
        Gf.Vec3d(
            *root_position
        )
    )

    stem_height = rng.uniform(
        1.3 * CROP_SCALE,
        1.7 * CROP_SCALE,
    )

    stem_prim = (
        create_cylinder_between(
            stage,
            f"{stem_path}/MainStem",
            (0, 0, 0),
            (
                0,
                0,
                stem_height,
            ),
            radius=rng.uniform(
                0.012 * CROP_SCALE,
                0.018 * CROP_SCALE,
            ),
        )
    )

    # Stem has collision but NO RigidBody.
    # Therefore it remains static.
    add_collider(
        stem_prim
    )

    bind_scene_material(
        stem_prim,
        "main_stem",
    )

    # Trusses — attach height is clamped to the aisle-parked arm
    # workspace so GraspPoints that can never be reached are not
    # authored.
    truss_count = rng.randint(
        1,
        4,
    )

    created = 0

    for _ in range(
        truss_count
    ):
        if create_truss(
            stage=stage,
            stem_path=stem_path,
            truss_index=created + 1,
            stem_height=stem_height,
            corridor_yaw=corridor_yaw,
            tool_marker_spacing=(
                tool_marker_spacing
            ),
            rng=rng,
        ):
            created += 1

    if created == 0:
        print(
            f"[STEM WARN] {stem_path}: no harvestable "
            f"truss fit under attach max "
            f"{MAX_HARVESTABLE_ATTACH_HEIGHT:.3f} m"
        )

    # Leaves — count/size follow OCCLUSION_LEVEL. medium keeps the
    # original 4–8 leaves so seed-42 layout stays comparable.
    #
    # CRITICAL: foliage must NOT share the structure RNG. Different
    # leaf counts would desync subsequent stem/truss draws and break
    # matched LOW/MEDIUM/HIGH target manifests for the same seed.
    leaf_lo, leaf_hi = leaf_count_range()
    size_scale = leaf_size_scale()
    leaf_rng = random.Random(
        f"{SEED}|{stem_path}|leaves"
    )

    leaf_count = leaf_rng.randint(
        leaf_lo,
        leaf_hi,
    )

    for i in range(
        leaf_count
    ):
        leaf_height = leaf_rng.uniform(
            stem_height * 0.25,
            stem_height * 0.95,
        )

        create_leaf(
            stage,
            f"{stem_path}/Leaf_{i+1:02d}",
            attachment_point=(
                0,
                0,
                leaf_height,
            ),
            direction=generate_direction(
                leaf_rng.uniform(
                    -180,
                    180,
                ),
                leaf_rng.uniform(
                    -20,
                    25,
                ),
            ),
            length=leaf_rng.uniform(
                0.15 * CROP_SCALE * size_scale,
                0.28 * CROP_SCALE * size_scale,
            ),
            width=leaf_rng.uniform(
                0.04 * CROP_SCALE * size_scale,
                0.08 * CROP_SCALE * size_scale,
            ),
        )
