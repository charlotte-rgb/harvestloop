"""Truss carry ops: logical grasp, payload transport and physical release."""

import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from .config import *  # noqa: F401,F403
from .simctl import next_frame
from .usd_pose import pose_position
from .arm_control import hold_arm_pose
from .base_control import apply_base_velocity_raw
from .basket import (
    ensure_basket_catcher_collision_enabled,
    set_basket_physx_collision_enabled,
)


def prepare_truss_carry_ops_before_physics(
    stage,
):
    """
    Prepare every Truss_* root BEFORE articulation tensor views exist.

    Covers both aisle-1 beds (left Row_02_Left and right Row_01_Left).

    For each truss:
      1. create/reuse the zero harvestCarry translation op
      2. ensure UsdPhysics.RigidBodyAPI exists
      3. author rigidBodyEnabled=True
      4. author kinematicEnabled=True
      5. pre-create velocity/angularVelocity attrs

    During harvesting:
      - before CUT: kinematic = attached to plant
      - after CUT/carry: still kinematic; carry op follows GraspTip
      - basket release: kinematic -> dynamic, gravity takes over

    No RigidBodyAPI is created while physics is live.
    """
    print()
    print("========================================")
    print("[TRUSS] PREPARE CARRY + RELEASE PHYSICS")
    print("========================================")

    row_roots = (
        AISLE1_LEFT_ROW_ROOT,
        AISLE1_RIGHT_ROW_ROOT,
    )

    print(
        "Row roots: "
        + ", ".join(row_roots)
    )

    carry_ops = {}

    rigid_body_count = 0

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        path = str(
            prim.GetPath()
        )

        if not any(
            path.startswith(root + "/")
            for root in row_roots
        ):
            continue

        name = prim.GetName()

        if not name.startswith(
            "Truss_"
        ):
            continue

        if prim.GetTypeName() != "Xform":
            continue

        # ----------------------------------------------------
        # Existing/prepared carry translation op
        # ----------------------------------------------------
        xformable = UsdGeom.Xformable(
            prim
        )

        wanted_name = (
            "xformOp:translate:"
            + TRUSS_CARRY_OP_SUFFIX
        )

        carry_op = None

        for op in xformable.GetOrderedXformOps():
            if str(
                op.GetOpName()
            ) == wanted_name:
                carry_op = op
                break

        if carry_op is None:
            carry_op = (
                xformable.AddTranslateOp(
                    UsdGeom.XformOp.PrecisionDouble,
                    TRUSS_CARRY_OP_SUFFIX,
                )
            )

        carry_op.Set(
            Gf.Vec3d(
                0.0,
                0.0,
                0.0,
            )
        )

        carry_ops[path] = carry_op

        # ----------------------------------------------------
        # PRE-CREATE rigid-body physics
        # ----------------------------------------------------
        if prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        ):
            rigid_body = (
                UsdPhysics.RigidBodyAPI(
                    prim
                )
            )
        else:
            rigid_body = (
                UsdPhysics.RigidBodyAPI.Apply(
                    prim
                )
            )

        rigid_body.CreateRigidBodyEnabledAttr(
            True
        ).Set(
            True
        )

        # Attached-to-plant / carried state is kinematic.
        rigid_body.CreateKinematicEnabledAttr(
            True
        ).Set(
            True
        )

        # Pre-create these attrs now so release only changes values.
        rigid_body.CreateVelocityAttr(
            Gf.Vec3f(
                0.0,
                0.0,
                0.0,
            )
        )

        rigid_body.CreateAngularVelocityAttr(
            Gf.Vec3f(
                0.0,
                0.0,
                0.0,
            )
        )

        # ----------------------------------------------------
        # PRE-CREATE / ENABLE PHYSX CCD
        # ----------------------------------------------------
        if prim.HasAPI(
            PhysxSchema.PhysxRigidBodyAPI
        ):
            physx_rigid_body = (
                PhysxSchema.PhysxRigidBodyAPI(
                    prim
                )
            )
        else:
            physx_rigid_body = (
                PhysxSchema.PhysxRigidBodyAPI.Apply(
                    prim
                )
            )

        physx_rigid_body.CreateEnableCCDAttr(
            ENABLE_TRUSS_CCD
        ).Set(
            ENABLE_TRUSS_CCD
        )

        physx_rigid_body.CreateEnableSpeculativeCCDAttr(
            ENABLE_TRUSS_SPECULATIVE_CCD
        ).Set(
            ENABLE_TRUSS_SPECULATIVE_CCD
        )

        rigid_body_count += 1

    if not carry_ops:
        raise RuntimeError(
            "No Truss_* Xform roots found under "
            + " or ".join(row_roots)
        )

    print(
        f"[TRUSS] Prepared carry ops: "
        f"{len(carry_ops)}"
    )

    print(
        f"[TRUSS] Prepared kinematic rigid bodies: "
        f"{rigid_body_count}"
    )

    print(
        "[TRUSS] Gravity will be enabled automatically when "
        "the selected truss becomes dynamic at basket release."
    )

    print(
        f"[TRUSS] CCD enabled={ENABLE_TRUSS_CCD}, "
        f"speculative_CCD={ENABLE_TRUSS_SPECULATIVE_CCD}"
    )

    return carry_ops



def find_cutter_tip_path(stage):
    if stage.GetPrimAtPath(
        CUTTER_TIP_PATH
    ).IsValid():
        return CUTTER_TIP_PATH

    for prim in stage.Traverse():
        path = str(
            prim.GetPath()
        )

        if (
            path.startswith(
                ARM_PATH + "/"
            )
            and prim.GetName()
            == "CutterTip"
        ):
            print(
                f"[CUTTER] Found CutterTip: "
                f"{path}"
            )
            return path

    raise RuntimeError(
        "Could not find CutterTip under UR5e."
    )


def make_logical_grasp(
    stage,
    grasp_path,
    cut_path,
):
    """
    No finger simulation. Grasp is a logical state once the physical
    GraspTip is within the accepted GraspPoint tolerance.

    The truss is NOT moved before CUT.
    """
    truss_path = str(
        stage.GetPrimAtPath(
            grasp_path
        ).GetPath().GetParentPath()
    )

    print()
    print("========================================")
    print("[LOGICAL GRASP]")
    print("========================================")

    print(
        f"Truss:      {truss_path}"
    )

    print(
        f"GraspPoint: {grasp_path}"
    )

    print(
        f"CutPoint:   {cut_path}"
    )

    print(
        "[GRASP] Logical attachment armed. "
        "Truss remains plant-fixed until CUT."
    )

    return truss_path



def disable_truss_collisions(
    stage,
    truss_path,
):
    """
    Disable collision on every collision-enabled descendant of the
    selected harvested truss.

    Why:
      After CUT, this project treats the harvested truss as a logical
      payload rigidly attached to GraspTip. Leaving the original plant
      colliders active can make PhysX fight the replayed robot motion.

    This changes only collisionEnabled on already-existing collision
    APIs. It does NOT create/remove/reparent prims while articulation
    tensor views are active.
    """
    truss_prim = stage.GetPrimAtPath(
        truss_path
    )

    if not truss_prim.IsValid():
        raise RuntimeError(
            f"Invalid truss path: {truss_path}"
        )

    disabled = []

    print()
    print("========================================")
    print("[PAYLOAD] DISABLE TRUSS COLLISIONS")
    print("========================================")

    for prim in stage.Traverse():
        path = str(
            prim.GetPath()
        )

        if not (
            path == truss_path
            or path.startswith(
                truss_path + "/"
            )
        ):
            continue

        # Only touch prims that already have CollisionAPI.
        if not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            continue

        api = UsdPhysics.CollisionAPI(
            prim
        )

        attr = (
            api.GetCollisionEnabledAttr()
        )

        # If the attribute is not authored yet, CreateCollisionEnabledAttr
        # authors it on the already-existing CollisionAPI.
        if not attr:
            attr = (
                api.CreateCollisionEnabledAttr()
            )

        old_value = attr.Get()

        attr.Set(
            False
        )

        disabled.append(
            {
                "path": path,
                "old_value": old_value,
            }
        )

        print(
            f"  collision OFF | {path}"
        )

    print(
        f"[PAYLOAD] Disabled collision on "
        f"{len(disabled)} prim(s)."
    )

    return disabled



def force_payload_collisions_enabled(
    stage,
    payload,
):
    """
    Force EVERY existing CollisionAPI under the selected harvested
    Truss_* to collisionEnabled=True.

    Do not restore the original value here. Some original collider
    attributes may have been False. A dynamic released rigid body with
    all of its usable collider shapes restored to False will simply fall
    through the basket.

    No CollisionAPI schema is created at runtime; we only toggle schemas
    that already existed before physics initialization.
    """
    if payload is None:
        return 0

    truss_path = payload[
        "truss_path"
    ]

    enabled = 0

    print()
    print("========================================")
    print("[PAYLOAD] FORCE ALL TRUSS COLLIDERS ON")
    print("========================================")

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue

        path = str(
            prim.GetPath()
        )

        if not (
            path == truss_path
            or path.startswith(
                truss_path + "/"
            )
        ):
            continue

        if not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            continue

        collision_api = (
            UsdPhysics.CollisionAPI(
                prim
            )
        )

        attr = (
            collision_api
            .GetCollisionEnabledAttr()
        )

        # CollisionAPI already exists. Author the enabled attribute if
        # that specific attribute has not yet been authored.
        if not attr:
            attr = (
                collision_api
                .CreateCollisionEnabledAttr(
                    True
                )
            )

        attr.Set(
            True
        )

        enabled += 1

        print(
            f"  collision ON | {path}"
        )

    print(
        f"[PAYLOAD] Forced collision ON for "
        f"{enabled} truss collider(s)."
    )

    if enabled == 0:
        print(
            "[PAYLOAD COLLISION ERROR] The selected Truss_* has no "
            "existing CollisionAPI geometry to collide with the basket."
        )

    return enabled




def activate_cut_payload(
    stage,
    truss_path,
    carry_ops,
):
    """
    After CUT, the selected truss starts following the GraspTip.

    We use translation-only following because build_scene(3).py places
    each Truss_* root at identity under a translated Stem root. The
    whole truss subtree therefore moves rigidly with this one prepared
    translation op.
    """
    if truss_path not in carry_ops:
        raise RuntimeError(
            f"No prepared carry op for {truss_path}"
        )

    # Once CUT happens, the truss is no longer treated as part of the
    # plant collision world. It becomes a logical payload attached to
    # GraspTip, so disable its old crop colliders before retreat.
    disabled_collision_state = (
        disable_truss_collisions(
            stage,
            truss_path,
        )
    )

    grasp_tip_at_cut = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    payload = {
        "active": True,
        "truss_path": truss_path,
        "carry_op": carry_ops[
            truss_path
        ],
        "grasp_tip_at_cut": (
            grasp_tip_at_cut.copy()
        ),
        "released": False,
        "disabled_collision_state": (
            disabled_collision_state
        ),
    }

    print()
    print("========================================")
    print("[CUT]")
    print("========================================")

    print(
        f"[CUT] Detached logical truss: "
        f"{truss_path}"
    )

    print(
        f"[CUT] Attachment reference GraspTip: "
        f"{np.round(grasp_tip_at_cut, 4)}"
    )

    print(
        "[CUT] Truss will now follow GraspTip "
        "during retreat."
    )

    return payload


def update_cut_payload(
    stage,
    payload,
):
    """
    Translate selected truss by the GraspTip displacement since CUT.
    """
    if payload is None:
        return

    if not payload.get(
        "active",
        False,
    ):
        return

    if payload.get(
        "released",
        False,
    ):
        return

    current_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    delta = (
        current_tip
        - payload[
            "grasp_tip_at_cut"
        ]
    )

    payload[
        "carry_op"
    ].Set(
        Gf.Vec3d(
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
        )
    )


async def physical_release_payload(
    stage,
    payload,
    jackal,
    arm,
    arm_indices,
    q_hold,
):
    """
    Release the harvested truss as a real dynamic rigid body.

    Sequence:
      1. update logical payload to its final carried pose
      2. stop following GraspTip
      3. enable all five Basket colliders + invisible thick catch floor
      4. FORCE every existing selected-Truss collider ON
      5. zero linear/angular velocity
      6. set kinematicEnabled=False
      7. gravity drops it onto the thick Basket catcher
      8. keep Jackal parked and arm fixed while it settles

    The RigidBodyAPI and its attributes were prepared before physics
    initialization, so no rigid-body schema is created here.
    """
    if payload is None:
        return False

    # Final kinematic carry update before handing motion to PhysX.
    update_cut_payload(
        stage,
        payload,
    )

    truss_path = payload[
        "truss_path"
    ]

    truss_prim = stage.GetPrimAtPath(
        truss_path
    )

    if not truss_prim.IsValid():
        print(
            f"[RELEASE FAILED] Invalid truss: "
            f"{truss_path}"
        )
        return False

    if not truss_prim.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        print(
            "[RELEASE FAILED] Truss has no pre-created "
            "RigidBodyAPI."
        )
        return False

    # Freeze logical attachment at current release transform.
    payload[
        "active"
    ] = False

    payload[
        "released"
    ] = True

    release_position = pose_position(
        stage,
        truss_path,
    ).copy()

    print()
    print("========================================")
    print("[PHYSICAL BASKET RELEASE]")
    print("========================================")

    # --------------------------------------------------------
    # ACTIVATE BASKET PHYSICS ONLY NOW
    # --------------------------------------------------------
    #
    # Up to this point the basket has existed visually and as an RMPflow
    # obstacle, but its PhysX collisions were disabled so it could not
    # physically fight the UR5e/gripper during the basket approach.
    #
    # The arm is now at the basket transport pose, so enable the five
    # existing basket colliders immediately before making the harvested
    # truss dynamic.
    enabled_basket_colliders = (
        set_basket_physx_collision_enabled(
            stage,
            True,
        )
    )

    if enabled_basket_colliders < 5:
        print(
            f"[RELEASE FAILED] Only "
            f"{enabled_basket_colliders} Basket collider(s) "
            "were enabled; expected the five visible Basket "
            "components."
        )
        return False

    # The invisible catcher should already be on; confirm it, because
    # it is what retains the truss once these components soften again.
    enabled_catcher_colliders = (
        ensure_basket_catcher_collision_enabled(
            stage
        )
    )

    if enabled_catcher_colliders < 1:
        print(
            "[RELEASE FAILED] The invisible basket catcher has no "
            "active collider."
        )
        return False

    # Let PhysX fully register basket colliders before the truss becomes
    # dynamic — too short a warmup is a common tunnel-through cause.
    for _ in range(
        PHYSICAL_DROP_COLLISION_WARMUP_FRAMES
    ):
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        hold_arm_pose(
            arm,
            arm_indices,
            q_hold,
        )

        await next_frame()

        update_cut_payload(
            stage,
            payload,
        )

    print(
        "[RELEASE] Basket PhysX collisions are now ACTIVE."
    )

    print(
        f"Released truss: "
        f"{truss_path}"
    )

    print(
        f"Release world position: "
        f"{np.round(release_position, 4)}"
    )

    # --------------------------------------------------------
    # FORCE every existing selected-Truss collider ON.
    # --------------------------------------------------------
    #
    # Do not restore the old values. A collider whose original value
    # was False would otherwise remain non-colliding after release.
    enabled_truss_colliders = (
        force_payload_collisions_enabled(
            stage,
            payload,
        )
    )

    if enabled_truss_colliders == 0:
        print(
            "[RELEASE FAILED] Selected harvested truss has no active "
            "collision geometry."
        )
        return False

    rigid_body = (
        UsdPhysics.RigidBodyAPI(
            truss_prim
        )
    )

    rigid_body.GetRigidBodyEnabledAttr().Set(
        True
    )

    # Start release from rest. The arm is already settled at the basket
    # pose, so zero release velocity is appropriate and prevents a
    # kinematic-following velocity spike.
    velocity_attr = (
        rigid_body.GetVelocityAttr()
    )

    if velocity_attr:
        velocity_attr.Set(
            Gf.Vec3f(
                0.0,
                0.0,
                0.0,
            )
        )

    angular_velocity_attr = (
        rigid_body.GetAngularVelocityAttr()
    )

    if angular_velocity_attr:
        angular_velocity_attr.Set(
            Gf.Vec3f(
                0.0,
                0.0,
                0.0,
            )
        )

    # This is the actual release:
    # kinematic -> dynamic. Scene gravity now acts on the truss.
    # Re-assert CCD + damping so the payload cannot tunnel the walls.
    if truss_prim.HasAPI(
        PhysxSchema.PhysxRigidBodyAPI
    ):
        physx_rigid_body = (
            PhysxSchema.PhysxRigidBodyAPI(
                truss_prim
            )
        )
    else:
        physx_rigid_body = (
            PhysxSchema.PhysxRigidBodyAPI.Apply(
                truss_prim
            )
        )

    physx_rigid_body.CreateEnableCCDAttr(
        True
    ).Set(
        True
    )

    physx_rigid_body.CreateEnableSpeculativeCCDAttr(
        True
    ).Set(
        True
    )

    physx_rigid_body.CreateLinearDampingAttr(
        3.0
    ).Set(
        3.0
    )

    physx_rigid_body.CreateAngularDampingAttr(
        3.0
    ).Set(
        3.0
    )

    physx_rigid_body.CreateMaxLinearVelocityAttr(
        1.5
    ).Set(
        1.5
    )

    print(
        "[RELEASE] CCD=True, speculative_CCD=True, "
        "linearDamping=3.0, maxLinearVelocity=1.5"
    )

    rigid_body.GetKinematicEnabledAttr().Set(
        False
    )

    print(
        "[RELEASE] kinematicEnabled=False"
    )

    print(
        "[RELEASE] Five Basket components plus the invisible catcher "
        "(thick floor + low retaining walls) are collidable."
    )

    print(
        f"[RELEASE] Forced {enabled_truss_colliders} selected-truss "
        "collider(s) ON; gravity is now active."
    )


    lowest_z = float(
        release_position[2]
    )

    final_position = (
        release_position.copy()
    )

    # --------------------------------------------------------
    # Let the payload fall while robot stays still.
    # --------------------------------------------------------
    for frame in range(
        PHYSICAL_DROP_SETTLE_FRAMES
    ):
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        hold_arm_pose(
            arm,
            arm_indices,
            q_hold,
        )

        await next_frame()

        current_position = pose_position(
            stage,
            truss_path,
        )

        final_position = (
            current_position.copy()
        )

        lowest_z = min(
            lowest_z,
            float(
                current_position[2]
            ),
        )

        if (
            frame == 0
            or (
                frame + 1
            )
            % PHYSICAL_DROP_PRINT_EVERY
            == 0
            or frame
            == PHYSICAL_DROP_SETTLE_FRAMES - 1
        ):
            print(
                f"[PHYSICAL DROP] "
                f"frame={frame + 1:3d}/"
                f"{PHYSICAL_DROP_SETTLE_FRAMES:3d} | "
                f"truss_pos="
                f"{np.round(current_position, 4)}"
            )

    drop_distance = float(
        release_position[2]
        - final_position[2]
    )

    print()
    print("========================================")
    print("[PHYSICAL DROP COMPLETE]")
    print("========================================")

    print(
        f"Release position: "
        f"{np.round(release_position, 4)}"
    )

    print(
        f"Final position:   "
        f"{np.round(final_position, 4)}"
    )

    print(
        f"Vertical displacement: "
        f"{drop_distance:+.4f} m"
    )

    print(
        f"Lowest observed z: "
        f"{lowest_z:.4f} m"
    )

    return True
