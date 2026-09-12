"""Basket discovery, PhysX collider preparation and drop-task errors."""

import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api.objects import VisualCuboid

from .config import *  # noqa: F401,F403
from .usd_pose import pose_position, world_point_to_frame


def find_basket_drop_point_path(
    stage,
):
    """
    Find Basket/DropPoint without hard-coding the Basket parent path.
    """
    candidates = []

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        path = str(
            prim.GetPath()
        )

        if "/Basket/" not in path:
            continue

        if (
            str(
                prim.GetName()
            ).lower()
            == "droppoint"
        ):
            candidates.append(
                path
            )

    if not candidates:
        raise RuntimeError(
            "Could not find Basket/DropPoint."
        )

    candidates = sorted(
        set(
            candidates
        )
    )

    if len(candidates) > 1:
        print(
            f"[BASKET TASK] Multiple DropPoint candidates: "
            f"{candidates}"
        )

    return candidates[0]


def get_basket_root_path(
    stage,
):
    """
    Find the live Basket root via Basket/Bottom.
    """
    bottom_prim = (
        find_basket_component_geometry_prim(
            stage,
            "Bottom",
        )
    )

    basket_root = (
        find_basket_root_prim_from_prim(
            bottom_prim
        )
    )

    if basket_root is None:
        raise RuntimeError(
            "Could not find Basket root."
        )

    return str(
        basket_root.GetPath()
    )


def grasp_tip_drop_offset_in_basket_frame(
    stage,
):
    """
    Return:

        GraspTip_local - DropPoint_local

    in Basket coordinates.

    This is invariant to small Jackal world drift and directly describes
    the physical drop geometry we care about.
    """
    basket_root_path = (
        get_basket_root_path(
            stage
        )
    )

    drop_point_path = (
        find_basket_drop_point_path(
            stage
        )
    )

    grasp_world = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    drop_world = pose_position(
        stage,
        drop_point_path,
    )

    grasp_local = world_point_to_frame(
        stage,
        basket_root_path,
        grasp_world,
    )

    drop_local = world_point_to_frame(
        stage,
        basket_root_path,
        drop_world,
    )

    return (
        grasp_local
        - drop_local
    )


def basket_drop_task_errors(
    stage,
    reference_offset,
):
    """
    Compare current physical GraspTip->DropPoint relation with the
    known-good relation recorded while the arm was in the working
    basket transport pose.
    """
    current_offset = (
        grasp_tip_drop_offset_in_basket_frame(
            stage
        )
    )

    reference_offset = np.asarray(
        reference_offset,
        dtype=np.float64,
    )

    delta = (
        current_offset
        - reference_offset
    )

    xy_error = float(
        np.linalg.norm(
            delta[:2]
        )
    )

    z_error = float(
        abs(
            delta[2]
        )
    )

    error_3d = float(
        np.linalg.norm(
            delta
        )
    )

    task_ok = (
        xy_error
        <= BASKET_DROP_XY_TOLERANCE
        and z_error
        <= BASKET_DROP_Z_TOLERANCE
        and error_3d
        <= BASKET_DROP_3D_TOLERANCE
    )

    task_accept = (
        xy_error
        <= BASKET_DROP_ACCEPT_XY_TOLERANCE
        and z_error
        <= BASKET_DROP_ACCEPT_Z_TOLERANCE
        and error_3d
        <= BASKET_DROP_ACCEPT_3D_TOLERANCE
    )

    return {
        "ok": task_ok,
        "accept": task_accept,
        "current_offset": (
            current_offset
        ),
        "delta": delta,
        "xy_error": xy_error,
        "z_error": z_error,
        "error_3d": error_3d,
    }



def find_basket_cube_paths(stage):
    """
    Find the physical Cube prims belonging to:

        Basket/Bottom
        Basket/Front
        Basket/Rear
        Basket/Left
        Basket/Right

    This is deliberately path-based rather than hard-coding the
    exact Basket root, because harvest_bot.usd can be referenced
    under a different parent in the full greenhouse scene.
    """
    matches = []

    component_names = set(
        BASKET_COMPONENT_NAMES
    )

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        path = str(
            prim.GetPath()
        )

        # Only basket descendants.
        if "/Basket/" not in path:
            continue

        # RMPflow's cuboid wrapper expects a Cube prim.
        if prim.GetTypeName() != "Cube":
            continue

        # Accept either the named component itself or a Cube child
        # nested below one of those component Xforms.
        path_parts = [
            part
            for part in path.split("/")
            if part
        ]

        if not any(
            name in path_parts
            for name in component_names
        ):
            continue

        matches.append(
            path
        )

    matches = sorted(
        set(matches)
    )

    if not matches:
        print()
        print(
            "[BASKET OBSTACLE ERROR] "
            "No Cube prims found below a Basket path."
        )

        print(
            "Basket-related prims in the stage:"
        )

        for prim in stage.Traverse():
            path = str(
                prim.GetPath()
            )

            if "Basket" in path:
                print(
                    f"  {path} "
                    f"(type={prim.GetTypeName()})"
                )

        raise RuntimeError(
            "Could not locate Basket cuboid components."
        )

    return matches




def find_basket_physics_geometry_paths(
    stage,
):
    """
    Find physical geometry for all five Basket components:

        Bottom, Front, Rear, Left, Right

    Unlike find_basket_cube_paths(), this function is for PhysX only and
    accepts any UsdGeom.Gprim geometry, not just Cube prims.

    A component may itself be geometry, or it may be an Xform containing
    one or more geometry descendants.

    The function requires every named component to contribute at least
    one physical geometry prim. Component-name matching is case-insensitive,
    so a USD typo such as RIght is accepted as canonical Right.  If one is missing, fail before physics
    starts instead of discovering it only when the truss falls through.
    """
    by_component = {
        name: []
        for name in BASKET_COMPONENT_NAMES
    }

    basket_related = []

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        path = str(
            prim.GetPath()
        )

        if "/Basket/" not in path:
            continue

        basket_related.append(
            (
                path,
                prim.GetTypeName(),
            )
        )

        path_parts = [
            part
            for part in path.split("/")
            if part
        ]

        # CollisionAPI belongs on actual geometry, not an empty Xform.
        try:
            is_gprim = bool(
                UsdGeom.Gprim(
                    prim
                )
            )
        except Exception:
            is_gprim = False

        if not is_gprim:
            continue

        # Match Basket component names case-insensitively.
        #
        # The current USD contains:
        #     Basket/RIght
        # instead of:
        #     Basket/Right
        #
        # Do not fail the whole physical-drop pipeline because of that
        # capitalization typo. Keep the canonical keys
        # Bottom/Front/Rear/Left/Right for validation and logging.
        path_parts_lower = [
            part.lower()
            for part in path_parts
        ]

        for component_name in BASKET_COMPONENT_NAMES:
            if (
                component_name.lower()
                in path_parts_lower
            ):
                by_component[
                    component_name
                ].append(
                    path
                )

    # De-duplicate and sort for deterministic behavior.
    for component_name in BASKET_COMPONENT_NAMES:
        by_component[
            component_name
        ] = sorted(
            set(
                by_component[
                    component_name
                ]
            )
        )

    print()
    print("========================================")
    print("[BASKET] PHYSICAL GEOMETRY DISCOVERY")
    print("========================================")

    missing = []

    all_paths = []

    for component_name in BASKET_COMPONENT_NAMES:
        paths = by_component[
            component_name
        ]

        if not paths:
            missing.append(
                component_name
            )

            print(
                f"  {component_name:8s}: MISSING"
            )
            continue

        print(
            f"  {component_name:8s}: "
            f"{len(paths)} geometry prim(s)"
        )

        for path in paths:
            prim = stage.GetPrimAtPath(
                path
            )

            actual_leaf_name = (
                path.rstrip("/")
                .split("/")[-1]
            )

            name_note = ""

            if (
                actual_leaf_name
                != component_name
            ):
                name_note = (
                    f" | accepted as {component_name} "
                    "(case-insensitive match)"
                )

            print(
                f"      {path} "
                f"(type={prim.GetTypeName()})"
                f"{name_note}"
            )

            all_paths.append(
                path
            )

    if missing:
        print()
        print(
            "[BASKET COLLIDER ERROR] Missing physical geometry for: "
            + ", ".join(
                missing
            )
        )

        print(
            "[BASKET COLLIDER ERROR] All Basket-related prims:"
        )

        for path, type_name in basket_related:
            print(
                f"  {path} "
                f"(type={type_name})"
            )

        raise RuntimeError(
            "Basket does not expose physical geometry for all five "
            "required components."
        )

    all_paths = sorted(
        set(
            all_paths
        )
    )

    print(
        f"[BASKET] Total physical geometry prims: "
        f"{len(all_paths)}"
    )

    return all_paths




def find_basket_root_prim_from_prim(
    prim,
):
    """
    Walk upward from a Basket component until the Basket root is found.
    """
    current = prim

    while current.IsValid():
        if (
            str(
                current.GetName()
            ).lower()
            == "basket"
        ):
            return current

        parent = current.GetParent()

        if (
            not parent.IsValid()
            or parent == current
        ):
            break

        current = parent

    return None


def find_basket_component_geometry_prim(
    stage,
    component_name,
):
    """
    Return one geometry prim belonging to the requested Basket component.
    Matching is case-insensitive, so Basket/RIght is accepted as Right.
    """
    candidates = []

    component_lower = (
        component_name.lower()
    )

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        path = str(
            prim.GetPath()
        )

        if "/Basket/" not in path:
            continue

        try:
            is_gprim = bool(
                UsdGeom.Gprim(
                    prim
                )
            )
        except Exception:
            is_gprim = False

        if not is_gprim:
            continue

        parts = [
            part.lower()
            for part in path.split("/")
            if part
        ]

        if component_lower not in parts:
            continue

        candidates.append(
            prim
        )

    if not candidates:
        raise RuntimeError(
            f"Could not find Basket/{component_name} geometry."
        )

    candidates.sort(
        key=lambda prim: (
            len(
                str(
                    prim.GetPath()
                )
            ),
            str(
                prim.GetPath()
            ),
        )
    )

    return candidates[0]


def get_basket_catch_floor_path(
    stage,
):
    """
    Return the expected PhysicsCatchFloor path under the live Basket.
    """
    bottom_prim = (
        find_basket_component_geometry_prim(
            stage,
            "Bottom",
        )
    )

    basket_root = (
        find_basket_root_prim_from_prim(
            bottom_prim
        )
    )

    if basket_root is None:
        raise RuntimeError(
            "Could not locate Basket root above Bottom."
        )

    return (
        f"{basket_root.GetPath()}/"
        f"{BASKET_CATCH_FLOOR_NAME}"
    )


def get_basket_catcher_paths(
    stage,
):
    """
    Invisible catcher prims: thick floor plus the four low retaining
    walls. These are permanently collidable, so a released truss stays
    in the basket even while the visible basket is soft.
    """
    floor_path = (
        get_basket_catch_floor_path(
            stage
        )
    )

    basket_root_path = (
        floor_path.rsplit("/", 1)[0]
    )

    paths = [
        floor_path
    ]

    for wall_name in BASKET_CATCH_WALL_NAMES:
        paths.append(
            f"{basket_root_path}/{wall_name}"
        )

    return paths


def define_invisible_collider_cube(
    stage,
    path,
    center,
    size,
):
    """
    Define (or rebuild) an invisible unit Cube scaled to `size` and
    placed at `center`, both in Basket-local coordinates.

    The prim gets CollisionAPI + PhysxCollisionAPI with the basket
    contact/rest offsets. Collision is left enabled: these helpers are
    never toggled, unlike the visible basket components.
    """
    cube = UsdGeom.Cube.Define(
        stage,
        path,
    )

    cube.CreateSizeAttr(
        1.0
    ).Set(
        1.0
    )

    prim = cube.GetPrim()

    # Repeated Script Editor runs must not stack duplicate xform ops.
    xform = UsdGeom.Xformable(
        prim
    )

    xform.ClearXformOpOrder()

    xform.AddTranslateOp().Set(
        Gf.Vec3d(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )
    )

    xform.AddScaleOp().Set(
        Gf.Vec3f(
            float(size[0]),
            float(size[1]),
            float(size[2]),
        )
    )

    UsdGeom.Imageable(
        prim
    ).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )

    if prim.HasAPI(
        UsdPhysics.CollisionAPI
    ):
        collision_api = (
            UsdPhysics.CollisionAPI(
                prim
            )
        )
    else:
        collision_api = (
            UsdPhysics.CollisionAPI.Apply(
                prim
            )
        )

    collision_api.CreateCollisionEnabledAttr(
        True
    ).Set(
        True
    )

    if prim.HasAPI(
        PhysxSchema.PhysxCollisionAPI
    ):
        physx_collision = (
            PhysxSchema.PhysxCollisionAPI(
                prim
            )
        )
    else:
        physx_collision = (
            PhysxSchema.PhysxCollisionAPI.Apply(
                prim
            )
        )

    physx_collision.CreateContactOffsetAttr(
        BASKET_CONTACT_OFFSET
    ).Set(
        BASKET_CONTACT_OFFSET
    )

    physx_collision.CreateRestOffsetAttr(
        BASKET_REST_OFFSET
    ).Set(
        BASKET_REST_OFFSET
    )

    return prim


def compute_catch_wall_specs(
    pmin,
    pmax,
    floor_top_z,
    wall_height,
):
    """
    Lay out the four retaining walls around the Bottom footprint,
    all in Basket-local coordinates.

    Each wall's inner face sits BASKET_CATCH_WALL_INNER_INSET inside
    the footprint edge and the slab grows outward from there, so the
    usable basket volume is unchanged. The two X walls span the full
    Y extent (and vice versa) so the corners cannot leak.

    Returns a tuple of (name, center_xyz, size_xyz).
    """
    inset = float(
        BASKET_CATCH_WALL_INNER_INSET
    )

    thickness = float(
        BASKET_CATCH_WALL_THICKNESS
    )

    wall_height = float(
        wall_height
    )

    center_x = float(
        0.5 * (pmin[0] + pmax[0])
    )

    center_y = float(
        0.5 * (pmin[1] + pmax[1])
    )

    center_z = float(
        floor_top_z
        + 0.5 * wall_height
    )

    span_x = float(
        (pmax[0] - pmin[0])
        - 2.0 * inset
        + 2.0 * thickness
    )

    span_y = float(
        (pmax[1] - pmin[1])
        - 2.0 * inset
        + 2.0 * thickness
    )

    return (
        (
            BASKET_CATCH_WALL_NAMES[0],
            (
                float(
                    pmin[0]
                    + inset
                    - 0.5 * thickness
                ),
                center_y,
                center_z,
            ),
            (
                thickness,
                span_y,
                wall_height,
            ),
        ),
        (
            BASKET_CATCH_WALL_NAMES[1],
            (
                float(
                    pmax[0]
                    - inset
                    + 0.5 * thickness
                ),
                center_y,
                center_z,
            ),
            (
                thickness,
                span_y,
                wall_height,
            ),
        ),
        (
            BASKET_CATCH_WALL_NAMES[2],
            (
                center_x,
                float(
                    pmin[1]
                    + inset
                    - 0.5 * thickness
                ),
                center_z,
            ),
            (
                span_x,
                thickness,
                wall_height,
            ),
        ),
        (
            BASKET_CATCH_WALL_NAMES[3],
            (
                center_x,
                float(
                    pmax[1]
                    - inset
                    + 0.5 * thickness
                ),
                center_z,
            ),
            (
                span_x,
                thickness,
                wall_height,
            ),
        ),
    )


def prepare_invisible_basket_catcher_before_physics(
    stage,
):
    """
    Pre-create the invisible physics catcher inside the Basket:
    a thick floor below Basket/Bottom plus four low retaining walls.

    Geometry is derived from the Bottom bound expressed in Basket-local
    coordinates, so the catcher follows the real Basket dimensions
    instead of hard-coding x/y size.

    The catcher:
      - is a child of Basket, therefore moves with Jackal/base_link
      - is invisible and is never an RMPflow obstacle
      - is collidable at ALL times, unlike the visible components

    That last point is what keeps a released truss in the basket. The
    visible walls are softened while the arm approaches the drop pose,
    and a truss resting on them would otherwise drop straight out.
    The retaining walls stop below the rim, so they cannot fight the
    gripper, which always releases from above the basket.
    """
    bottom_prim = (
        find_basket_component_geometry_prim(
            stage,
            "Bottom",
        )
    )

    basket_root = (
        find_basket_root_prim_from_prim(
            bottom_prim
        )
    )

    if basket_root is None:
        raise RuntimeError(
            "Could not locate Basket root for PhysicsCatchFloor."
        )

    # Bound Bottom in Basket-local coordinates.
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=False,
        ignoreVisibility=True,
    )

    relative_bbox = (
        bbox_cache.ComputeRelativeBound(
            bottom_prim,
            basket_root,
        )
    )

    aligned_range = (
        relative_bbox.ComputeAlignedRange()
    )

    pmin = np.asarray(
        aligned_range.GetMin(),
        dtype=np.float64,
    )

    pmax = np.asarray(
        aligned_range.GetMax(),
        dtype=np.float64,
    )

    bottom_size = (
        pmax
        - pmin
    )

    if (
        bottom_size[0] <= 0.0
        or bottom_size[1] <= 0.0
    ):
        raise RuntimeError(
            "Basket Bottom has invalid XY bound; cannot build catch floor."
        )

    catch_size_x = float(
        bottom_size[0]
        + 2.0 * BASKET_CATCH_FLOOR_XY_MARGIN
    )

    catch_size_y = float(
        bottom_size[1]
        + 2.0 * BASKET_CATCH_FLOOR_XY_MARGIN
    )

    catch_size_z = float(
        BASKET_CATCH_FLOOR_THICKNESS
    )

    catch_center_x = float(
        0.5
        * (
            pmin[0]
            + pmax[0]
        )
    )

    catch_center_y = float(
        0.5
        * (
            pmin[1]
            + pmax[1]
        )
    )

    # Place the top face at (Bottom top + small overlap).
    catch_top_z = float(
        pmax[2]
        + BASKET_CATCH_FLOOR_TOP_OVERLAP
    )

    catch_center_z = float(
        catch_top_z
        - 0.5 * catch_size_z
    )

    catch_path = (
        f"{basket_root.GetPath()}/"
        f"{BASKET_CATCH_FLOOR_NAME}"
    )

    define_invisible_collider_cube(
        stage,
        catch_path,
        (
            catch_center_x,
            catch_center_y,
            catch_center_z,
        ),
        (
            catch_size_x,
            catch_size_y,
            catch_size_z,
        ),
    )

    print()
    print("========================================")
    print("[BASKET] INVISIBLE PHYSICS CATCHER")
    print("========================================")

    print(
        f"Bottom:     {bottom_prim.GetPath()}"
    )

    print(
        f"CatchFloor: {catch_path}"
    )

    print(
        f"Bottom Basket-local min: "
        f"{np.round(pmin, 4)}"
    )

    print(
        f"Bottom Basket-local max: "
        f"{np.round(pmax, 4)}"
    )

    print(
        f"CatchFloor center: "
        f"{np.round([catch_center_x, catch_center_y, catch_center_z], 4)}"
    )

    print(
        f"CatchFloor size:   "
        f"{np.round([catch_size_x, catch_size_y, catch_size_z], 4)}"
    )

    catcher_paths = [
        catch_path
    ]

    # ----------------------------------------------------------
    # Low retaining walls
    # ----------------------------------------------------------
    #
    # Height is clamped to the visible rim so the walls stay well
    # below the gripper's release pose.
    rim_top_z = catch_top_z

    for geometry_path in find_basket_physics_geometry_paths(
        stage
    ):
        geometry_prim = stage.GetPrimAtPath(
            geometry_path
        )

        if not geometry_prim.IsValid():
            continue

        geometry_range = (
            bbox_cache.ComputeRelativeBound(
                geometry_prim,
                basket_root,
            ).ComputeAlignedRange()
        )

        rim_top_z = max(
            rim_top_z,
            float(
                geometry_range.GetMax()[2]
            ),
        )

    wall_height = min(
        float(
            BASKET_CATCH_WALL_HEIGHT
        ),
        float(
            rim_top_z
            - catch_top_z
            - BASKET_CATCH_WALL_RIM_CLEARANCE
        ),
    )

    if wall_height <= 0.0:
        print(
            "[BASKET WARNING] Basket rim is too shallow for invisible "
            "retaining walls; only the catch floor is active."
        )

        return catcher_paths

    wall_specs = compute_catch_wall_specs(
        pmin,
        pmax,
        catch_top_z,
        wall_height,
    )

    for wall_name, center, size in wall_specs:
        wall_path = (
            f"{basket_root.GetPath()}/{wall_name}"
        )

        define_invisible_collider_cube(
            stage,
            wall_path,
            center,
            size,
        )

        catcher_paths.append(
            wall_path
        )

        print(
            f"  [CATCH WALL] {wall_name:22s} "
            f"center={np.round(center, 4)} | "
            f"size={np.round(size, 4)}"
        )

    print(
        f"Basket-local rim top z: {rim_top_z:.4f} | "
        f"catcher floor top z: {catch_top_z:.4f} | "
        f"wall height: {wall_height:.4f} m"
    )

    print(
        "[BASKET] Catcher is invisible and stays collidable at all "
        "times, so released trusses cannot leave the basket while "
        "the visible components are softened."
    )

    return catcher_paths


def get_all_basket_release_collider_paths(
    stage,
):
    """
    The five visible Basket component geometries.

    The invisible catcher is deliberately excluded: it must stay
    collidable at all times so a truss already lying in the basket
    keeps its support while these components are softened.
    """
    return sorted(
        set(
            find_basket_physics_geometry_paths(
                stage
            )
        )
    )



def find_rigid_body_ancestor_path(
    stage,
    prim_path,
):
    """
    Return the nearest ancestor (including self) that already has
    UsdPhysics.RigidBodyAPI, or None.

    A basket Cube with a rigid-body ancestor moves with that body.
    Otherwise PhysX treats the collider as static.
    """
    prim = stage.GetPrimAtPath(
        prim_path
    )

    while prim.IsValid():
        if prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        ):
            return str(
                prim.GetPath()
            )

        parent = prim.GetParent()

        if (
            not parent.IsValid()
            or parent == prim
        ):
            break

        prim = parent

    return None



def prepare_basket_physx_colliders_before_physics(
    stage,
):
    """
    Make Basket/Bottom/Front/Rear/Left/Right real PhysX colliders.

    IMPORTANT:
    This runs BEFORE timeline.play() / articulation initialization.

    The Basket geometry already exists. We only apply physics schemas
    to those existing geometry prims; no new collision geometry is
    created.

    NOTE:
    RMPflow obstacle discovery remains Cube-only. PhysX collision
    discovery is broader so non-Cube wall geometry is not missed.
    """
    geometry_paths = (
        find_basket_physics_geometry_paths(
            stage
        )
    )

    # Create the robust invisible catcher before PhysX/articulation
    # initialization. Unlike the visible components it is collidable
    # from the start and is never softened.
    catcher_paths = (
        prepare_invisible_basket_catcher_before_physics(
            stage
        )
    )

    print()
    print("========================================")
    print("[BASKET] PREPARE REAL PHYSX COLLIDERS")
    print("========================================")

    prepared = []

    for geometry_path in geometry_paths:
        prim = stage.GetPrimAtPath(
            geometry_path
        )

        if not prim.IsValid():
            raise RuntimeError(
                f"Invalid basket geometry: {geometry_path}"
            )

        # ----------------------------------------------------
        # USD Physics collider
        # ----------------------------------------------------
        if prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            collision_api = (
                UsdPhysics.CollisionAPI(
                    prim
                )
            )
        else:
            collision_api = (
                UsdPhysics.CollisionAPI.Apply(
                    prim
                )
            )

        # Pre-create the real PhysX collider before physics starts,
        # but keep it DISABLED during all navigation/manipulation/arm
        # motion. It will be enabled only immediately before the
        # harvested truss is released into the basket.
        collision_api.CreateCollisionEnabledAttr(
            False
        ).Set(
            False
        )

        # ----------------------------------------------------
        # PhysX contact/rest offsets
        # ----------------------------------------------------
        if prim.HasAPI(
            PhysxSchema.PhysxCollisionAPI
        ):
            physx_collision = (
                PhysxSchema.PhysxCollisionAPI(
                    prim
                )
            )
        else:
            physx_collision = (
                PhysxSchema.PhysxCollisionAPI.Apply(
                    prim
                )
            )

        physx_collision.CreateContactOffsetAttr(
            BASKET_CONTACT_OFFSET
        ).Set(
            BASKET_CONTACT_OFFSET
        )

        physx_collision.CreateRestOffsetAttr(
            BASKET_REST_OFFSET
        ).Set(
            BASKET_REST_OFFSET
        )

        # Thin mesh basket walls often tunnel under default triangle
        # meshes. Prefer convex hulls when the prim is a Mesh.
        if prim.IsA(UsdGeom.Mesh):
            if prim.HasAPI(
                UsdPhysics.MeshCollisionAPI
            ):
                mesh_collision = (
                    UsdPhysics.MeshCollisionAPI(
                        prim
                    )
                )
            else:
                mesh_collision = (
                    UsdPhysics.MeshCollisionAPI.Apply(
                        prim
                    )
                )

            mesh_collision.CreateApproximationAttr(
                "convexHull"
            ).Set(
                "convexHull"
            )

        rigid_ancestor = (
            find_rigid_body_ancestor_path(
                stage,
                geometry_path,
            )
        )

        prepared.append(
            geometry_path
        )

        print(
            f"  collider PREPARED / currently OFF | {geometry_path}"
        )

        if rigid_ancestor is not None:
            print(
                f"    rigid-body ancestor: "
                f"{rigid_ancestor}"
            )
        else:
            print(
                "    WARNING: no RigidBodyAPI ancestor; "
                "PhysX will treat this basket collider as static."
            )

    # The catcher prims already have their CollisionAPI prepared, and
    # they are intentionally left enabled.
    for catcher_path in catcher_paths:
        if catcher_path not in prepared:
            prepared.append(
                catcher_path
            )

    print(
        f"[BASKET] Prepared "
        f"{len(prepared)} real PhysX collider(s), including "
        f"{len(catcher_paths)} invisible catcher prim(s)."
    )

    print(
        "[BASKET] Visible component collisions are currently DISABLED "
        "and are enabled only around payload release. The invisible "
        "catcher stays enabled so released trusses stay in the basket."
    )

    return prepared



def set_basket_physx_collision_enabled(
    stage,
    enabled,
):
    """
    Toggle existing Basket/Bottom/Front/Rear/Left/Right CollisionAPI
    attributes.

    All schemas are created before physics starts; at runtime we only
    change the existing collisionEnabled values.

    Use:
        False during navigation / manipulation / arm-to-basket motion
        True  immediately before harvested-truss release

    The invisible catcher is never toggled here. It holds already
    released trusses while these visible components are soft.
    """
    geometry_paths = (
        get_all_basket_release_collider_paths(
            stage
        )
    )

    changed = 0

    for geometry_path in geometry_paths:
        prim = stage.GetPrimAtPath(
            geometry_path
        )

        if not prim.IsValid():
            continue

        if not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            raise RuntimeError(
                f"Basket collider missing CollisionAPI: {geometry_path}"
            )

        collision_api = (
            UsdPhysics.CollisionAPI(
                prim
            )
        )

        attr = (
            collision_api
            .GetCollisionEnabledAttr()
        )

        if not attr:
            attr = (
                collision_api
                .CreateCollisionEnabledAttr(
                    bool(enabled)
                )
            )

        attr.Set(
            bool(enabled)
        )

        changed += 1

        print(
            f"  [BASKET COLLIDER] "
            f"enabled={bool(enabled)} | "
            f"{geometry_path}"
        )

    print(
        f"[BASKET COLLISION] enabled={bool(enabled)} "
        f"on {changed} basket collider(s)."
    )

    return changed


def ensure_basket_catcher_collision_enabled(
    stage,
):
    """
    Re-assert that every invisible catcher prim is collidable.

    Cheap insurance before a release: if the catcher were off, the
    truss would drop straight through the softened basket.
    """
    enabled = 0

    for catcher_path in get_basket_catcher_paths(
        stage
    ):
        prim = stage.GetPrimAtPath(
            catcher_path
        )

        if not prim.IsValid():
            continue

        if not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            continue

        UsdPhysics.CollisionAPI(
            prim
        ).CreateCollisionEnabledAttr(
            True
        ).Set(
            True
        )

        enabled += 1

    print(
        f"[BASKET CATCHER] {enabled} invisible collider(s) confirmed "
        "enabled."
    )

    return enabled


def enable_physx_scene_ccd_before_physics(
    stage,
):
    """
    Enable CCD at the PhysX-scene level when a PhysicsScene prim exists.
    """
    scene_count = 0

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        if not prim.IsA(
            UsdPhysics.Scene
        ):
            continue

        if prim.HasAPI(
            PhysxSchema.PhysxSceneAPI
        ):
            physx_scene = (
                PhysxSchema.PhysxSceneAPI(
                    prim
                )
            )
        else:
            physx_scene = (
                PhysxSchema.PhysxSceneAPI.Apply(
                    prim
                )
            )

        physx_scene.CreateEnableCCDAttr(
            True
        ).Set(
            True
        )

        scene_count += 1

        print(
            f"[PHYSX CCD] Scene CCD enabled: "
            f"{prim.GetPath()}"
        )

    if scene_count == 0:
        print(
            "[PHYSX CCD WARNING] No UsdPhysics.Scene prim found. "
            "Rigid-body CCD will still be authored on the truss."
        )

    return scene_count


def create_basket_obstacle_wrappers_before_physics(
    stage,
):
    """
    IMPORTANT ISAAC SIM LIFECYCLE RULE:

    Create/wrap the Basket Cube prims BEFORE SingleArticulation
    initialization. Constructing new Core API prim wrappers after
    PhysX tensor articulation views are active can invalidate those
    tensor views.

    These wrappers point to the already-existing Basket Cube prims.
    No basket geometry is created here.
    """
    cube_paths = (
        find_basket_cube_paths(
            stage
        )
    )

    print()
    print("========================================")
    print("[BASKET] PREPARE RMPFLOW WRAPPERS")
    print("========================================")

    obstacles = []

    for index, cube_path in enumerate(
        cube_paths
    ):
        obstacle = VisualCuboid(
            prim_path=cube_path,
            name=(
                f"rmpflow_basket_"
                f"{index}"
            ),
        )

        obstacles.append(
            obstacle
        )

        print(
            f"  prepared | {cube_path}"
        )

    print(
        f"[BASKET] Prepared "
        f"{len(obstacles)} cuboid wrappers "
        "before physics initialization."
    )

    return obstacles


def add_prepared_basket_obstacles_to_rmpflow(
    rmpflow,
    basket_obstacles,
):
    """
    Register already-prepared wrappers with RMPflow.

    RMPflow is only used after Jackal has parked, so the basket is
    treated as STATIC for this manipulation phase. Its world pose is
    read when it is added; no per-frame obstacle update is required.
    """
    print()
    print("========================================")
    print("[RMPFLOW] REGISTER PARKED BASKET")
    print("========================================")

    active = []

    for obstacle in basket_obstacles:
        success = (
            rmpflow.add_obstacle(
                obstacle,
                static=True,
            )
        )

        print(
            f"  {'OK' if success else 'FAILED'} | "
            f"{obstacle.prim_path}"
        )

        if success:
            active.append(
                obstacle
            )

    if len(active) != len(
        basket_obstacles
    ):
        raise RuntimeError(
            "RMPflow did not accept every Basket cuboid. "
            f"accepted={len(active)}, "
            f"prepared={len(basket_obstacles)}"
        )

    print(
        f"[RMPFLOW] Static basket obstacles active: "
        f"{len(active)}"
    )

    return active
