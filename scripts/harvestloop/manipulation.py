"""RMPflow arm stages: viewpoint, pregrasp, grasp, cut, retreat, basket."""

import math

import numpy as np

from .config import *  # noqa: F401,F403
from .mathutils import (
    quaternion_angle,
    quaternion_conjugate,
    quaternion_from_two_vectors,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)
from .simctl import (
    ensure_articulations_initialized,
    next_frame,
    rebuild_articulation_motion_policy,
    recover_rmpflow_action,
)
from .usd_pose import (
    grasp_tip_position_in_arm_base,
    pose_position,
    pose_position_orientation,
)
from .arm_control import (
    get_active_arm_positions,
    hold_arm_pose,
    update_rmpflow_base_pose,
)
from .base_control import apply_base_velocity_raw
from .basket import basket_drop_task_errors
from .payload import update_cut_payload
from .perception import wrist_camera_framing


def compute_pregrasp(
    grasp_position,
    side_name="left",
):
    """
    Aisle-side PREGRASP from GraspPoint.

    left  (+Y bed): GraspPoint + [0, -PREGRASP_OFFSET, 0]
    right (-Y bed): GraspPoint + [0, +PREGRASP_OFFSET, 0]
    """
    side = str(
        side_name
    ).strip().lower()

    if side == "right":
        y_offset = float(
            PREGRASP_OFFSET
        )
    else:
        y_offset = -float(
            PREGRASP_OFFSET
        )

    return (
        np.asarray(
            grasp_position,
            dtype=np.float64,
        )
        + np.array(
            [
                0.0,
                y_offset,
                0.0,
            ],
            dtype=np.float64,
        )
    )




async def move_arm_to_basket_transport_with_rmpflow(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    q_target,
    payload,
    reference_drop_offset,
):
    """
    Drive the arm to the basket transport / home joint pose and release
    from that configuration.

    Success = physical joints within RMPFLOW_BASKET_CSPACE_TOLERANCE of
    q_target (UR5E_HOME_Q). DropPoint geometry is logged only.
    """
    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    ).reshape(-1).copy()

    reference_drop_offset = np.asarray(
        reference_drop_offset,
        dtype=np.float64,
    )

    print()
    print("========================================")
    print("[RMPFLOW BASKET HOME TRANSITION]")
    print("========================================")

    print(
        f"Home / release q: "
        f"{np.round(q_target, 4)}"
    )

    print(
        f"Known-good GraspTip-DropPoint offset "
        f"(diagnostic): "
        f"{np.round(reference_drop_offset, 4)}"
    )

    q_start = get_active_arm_positions(
        arm,
        arm_indices,
    )

    initial_joint_error = float(
        np.max(
            np.abs(
                q_start
                - q_target
            )
        )
    )

    initial_task = basket_drop_task_errors(
        stage,
        reference_drop_offset,
    )

    print(
        f"Initial home joint error: "
        f"{initial_joint_error:.4f} rad "
        f"(tol={RMPFLOW_BASKET_CSPACE_TOLERANCE:.4f})"
    )

    print(
        f"Initial drop task error (diagnostic): "
        f"xy={initial_task['xy_error']:.4f} m | "
        f"z={initial_task['z_error']:.4f} m | "
        f"3d={initial_task['error_3d']:.4f} m"
    )

    rmpflow.set_end_effector_target(
        None,
        None,
    )

    rmpflow.set_cspace_target(
        q_target
    )

    best_joint_error = (
        initial_joint_error
    )

    best_q = q_start.copy()
    divergence_count = 0

    async def _hold_home_for_release():
        for _ in range(
            RMPFLOW_BASKET_CSPACE_SETTLE_FRAMES
        ):
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )
            hold_arm_pose(
                arm,
                arm_indices,
                q_target,
            )
            await next_frame()
            update_cut_payload(
                stage,
                payload,
            )

        return q_target.copy()

    for step in range(
        RMPFLOW_BASKET_CSPACE_MAX_STEPS
    ):
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        update_rmpflow_base_pose(
            stage,
            rmpflow,
        )

        action = (
            articulation_policy
            .get_next_articulation_action(
                RMPFLOW_PHYSICS_DT
            )
        )

        arm.apply_action(
            action
        )

        await next_frame()

        update_cut_payload(
            stage,
            payload,
        )

        q_actual = get_active_arm_positions(
            arm,
            arm_indices,
        )

        joint_error = float(
            np.max(
                np.abs(
                    q_actual
                    - q_target
                )
            )
        )

        task = basket_drop_task_errors(
            stage,
            reference_drop_offset,
        )

        if (
            joint_error + 1e-6
            < best_joint_error
        ):
            best_joint_error = joint_error
            best_q = q_actual.copy()

        if (
            step == 0
            or (step + 1)
            % RMPFLOW_BASKET_CSPACE_PRINT_EVERY
            == 0
        ):
            print(
                f"[RMPFLOW BASKET] "
                f"step={step:4d} | "
                f"home_err={joint_error:.4f} rad | "
                f"drop_xy={task['xy_error']:.4f} m | "
                f"drop_z={task['z_error']:.4f} m | "
                f"drop_3d={task['error_3d']:.4f} m"
            )

        if (
            joint_error
            <= RMPFLOW_BASKET_CSPACE_TOLERANCE
        ):
            print()
            print(
                "[BASKET HOME REACHED] "
                f"max_joint_error={joint_error:.4f} rad"
            )
            print(
                f"DropPoint offset (diagnostic): "
                f"{np.round(task['current_offset'], 4)}"
            )

            release_q = (
                await _hold_home_for_release()
            )

            print(
                f"Release-hold q (= home): "
                f"{np.round(release_q, 4)}"
            )

            return (
                True,
                release_q,
            )

        if (
            joint_error
            > best_joint_error
            + RMPFLOW_BASKET_CSPACE_DIVERGENCE_MARGIN
        ):
            divergence_count += 1
        else:
            divergence_count = 0

        if (
            divergence_count
            >= RMPFLOW_BASKET_CSPACE_DIVERGENCE_STEPS
        ):
            print()
            print(
                "[BASKET] Home guidance diverging; "
                "holding home for release."
            )
            break

    # Timeout / divergence: still command home for the release hold.
    release_q = await _hold_home_for_release()

    final_q = get_active_arm_positions(
        arm,
        arm_indices,
    )

    final_joint_error = float(
        np.max(
            np.abs(
                final_q
                - q_target
            )
        )
    )

    final_task = basket_drop_task_errors(
        stage,
        reference_drop_offset,
    )

    if (
        final_joint_error
        <= ARM_HOME_ACCEPT_TOLERANCE
    ):
        print()
        print(
            "[BASKET HOME ACCEPT] "
            f"max_joint_error={final_joint_error:.4f} rad "
            f"(accept_tol={ARM_HOME_ACCEPT_TOLERANCE:.4f})"
        )
        print(
            f"DropPoint (diagnostic): "
            f"xy={final_task['xy_error']:.4f} m | "
            f"z={final_task['z_error']:.4f} m | "
            f"3d={final_task['error_3d']:.4f} m"
        )
        print(
            f"Release-hold q (= home): "
            f"{np.round(release_q, 4)}"
        )

        return (
            True,
            release_q,
        )

    print()
    print(
        "[FAILED] Basket home transition timed out; "
        "still releasing at commanded home hold."
    )
    print(
        f"Final home joint error="
        f"{final_joint_error:.4f} rad "
        f"(best={best_joint_error:.4f})"
    )
    print(
        f"DropPoint (diagnostic): "
        f"xy={final_task['xy_error']:.4f} m | "
        f"z={final_task['z_error']:.4f} m | "
        f"3d={final_task['error_3d']:.4f} m"
    )
    print(
        f"Release-hold q (= home): "
        f"{np.round(release_q, 4)}"
    )

    return (
        False,
        release_q,
    )


async def move_arm_to_pregrasp(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    target_position,
):
    print()
    print("========================================")
    print("[RMPFLOW] MOVE TO PREGRASP")
    print("========================================")

    print(
        f"Target: "
        f"{np.round(target_position, 4)}"
    )

    # IMPORTANT:
    # Do NOT add a c-space target during PREGRASP.
    #
    # The earlier working HarvestLoop pregrasp test used only the
    # Cartesian GraspTip target here. Adding an upright c-space
    # attractor creates a competing objective and can make RMPflow
    # settle far away from PREGRASP.
    #
    # We already placed the arm in UR5E_UPRIGHT_Q before creating
    # this fresh RMPflow instance, so upright is only the INITIAL
    # configuration, not a simultaneous optimization target.
    rmpflow.set_end_effector_target(
        np.asarray(
            target_position,
            dtype=np.float64,
        ),
        None,
    )

    initial_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    # Record the ACTUAL physical six-joint trajectory. The first point
    # is the upright configuration from which RMPflow starts.
    joint_trajectory = [
        get_active_arm_positions(
            arm,
            arm_indices,
        ).copy()
    ]

    initial_delta = (
        np.asarray(
            target_position,
            dtype=np.float64,
        )
        - initial_tip
    )

    print(
        f"Initial GraspTip: "
        f"{np.round(initial_tip, 4)}"
    )

    print(
        f"Initial target delta [dx,dy,dz]: "
        f"{np.round(initial_delta, 4)}"
    )

    print(
        f"Initial target error: "
        f"{np.linalg.norm(initial_delta):.4f} m"
    )

    for step in range(
        PREGRASP_MAX_STEPS
    ):
        # Jackal must remain completely parked during manipulation.
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        update_rmpflow_base_pose(
            stage,
            rmpflow,
        )

        action = (
            articulation_policy
            .get_next_articulation_action(
                RMPFLOW_PHYSICS_DT
            )
        )

        arm.apply_action(
            action
        )

        await next_frame()

        # Store the measured configuration after this RMPflow step.
        joint_trajectory.append(
            get_active_arm_positions(
                arm,
                arm_indices,
            ).copy()
        )

        tip_position = pose_position(
            stage,
            GRASP_TIP_PATH,
        )

        error = float(
            np.linalg.norm(
                tip_position
                - target_position
            )
        )

        if step % 30 == 0:
            delta = (
                np.asarray(
                    target_position,
                    dtype=np.float64,
                )
                - tip_position
            )

            print(
                f"[PREGRASP] step={step:4d} | "
                f"tip={np.round(tip_position, 4)} | "
                f"delta={np.round(delta, 4)} | "
                f"error={error:.4f} m"
            )

        if error <= PREGRASP_TOLERANCE:
            print()
            print(
                "[PREGRASP REACHED]"
            )

            print(
                f"Physical GraspTip: "
                f"{np.round(tip_position, 4)}"
            )

            print(
                f"Target error: "
                f"{error:.4f} m"
            )

            print(
                "[GRASPPOINT] Accepted immediately at "
                "current physical pose; no best-pose restore."
            )

            print(
                f"Recorded return trajectory points: "
                f"{len(joint_trajectory)}"
            )

            return True, joint_trajectory

    final_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    final_error = float(
        np.linalg.norm(
            final_tip
            - target_position
        )
    )

    print()
    print(
        "[FAILED] PREGRASP timed out."
    )

    print(
        f"Final GraspTip: "
        f"{np.round(final_tip, 4)}"
    )

    print(
        f"Final target error: "
        f"{final_error:.4f} m"
    )

    return False, joint_trajectory



async def reset_grasp_retry_seed(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    reset_position,
):
    """
    Short bounded reset to a nearby PREGRASP seed before another
    GraspPoint attempt.
    """
    reset_position = np.asarray(
        reset_position,
        dtype=np.float64,
    )

    rmpflow.set_end_effector_target(
        reset_position,
        None,
    )

    best_error = float("inf")
    no_progress = 0
    previous_best = float("inf")

    for step in range(
        GRASP_RETRY_RESET_MAX_STEPS
    ):
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        update_rmpflow_base_pose(
            stage,
            rmpflow,
        )

        action = (
            articulation_policy
            .get_next_articulation_action(
                RMPFLOW_PHYSICS_DT
            )
        )

        arm.apply_action(
            action
        )

        await next_frame()

        tip = pose_position(
            stage,
            GRASP_TIP_PATH,
        )

        error = float(
            np.linalg.norm(
                tip
                - reset_position
            )
        )

        if error < best_error:
            best_error = error

        if (
            step == 0
            or (step + 1)
            % GRASP_RETRY_RESET_PRINT_EVERY
            == 0
        ):
            print(
                f"[GRASP RETRY RESET] "
                f"step={step:3d} | "
                f"error={error:.4f} m | "
                f"best={best_error:.4f} m"
            )

        if (
            error
            <= GRASP_RETRY_RESET_TOLERANCE
        ):
            print(
                f"[GRASP RETRY RESET REACHED] "
                f"error={error:.4f} m"
            )
            return True

        # Stop a reset that has genuinely stopped improving.
        if (
            previous_best
            - best_error
            >= GRASP_PLATEAU_MIN_IMPROVEMENT
        ):
            no_progress = 0
            previous_best = best_error
        else:
            no_progress += 1

        if (
            no_progress
            >= GRASP_PLATEAU_CONSECUTIVE_STEPS
        ):
            print(
                "[GRASP RETRY RESET] Plateau detected."
            )
            break

    print(
        f"[GRASP RETRY RESET FAILED] "
        f"best_error={best_error:.4f} m"
    )

    return False


async def move_arm_to_grasp_point(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    target_position,
    pregrasp_position,
):
    """
    PREGRASP -> actual GraspPoint with bounded local retries.

    RMPflow is a local reactive controller, so one very long attempt is
    not treated as more reliable than several short, supervised attempts.

    Workflow:
        attempt 1 from nominal PREGRASP
        if plateau/divergence:
            reset to nearby PREGRASP seed
        attempt 2
        if plateau/divergence:
            reset to another nearby PREGRASP seed
        attempt 3

    Success remains physical Cartesian distance <= 3 cm.
    No "best pose" joint restoration is used.
    """
    target_position = np.asarray(
        target_position,
        dtype=np.float64,
    )

    pregrasp_position = np.asarray(
        pregrasp_position,
        dtype=np.float64,
    )

    print()
    print("========================================")
    print("[RMPFLOW] BOUNDED GRASPPOINT RETRIES")
    print("========================================")

    print(
        f"GraspPoint target: "
        f"{np.round(target_position, 4)}"
    )

    print(
        f"Success tolerance: "
        f"{GRASP_TOLERANCE:.3f} m"
    )

    overall_best_error = float("inf")
    overall_best_tip = None

    # Only the successful attempt's trajectory is relevant.
    successful_trajectory = []

    for attempt_index in range(
        GRASP_MAX_ATTEMPTS
    ):
        attempt_number = (
            attempt_index + 1
        )

        z_offset = (
            GRASP_RETRY_PREGRASP_Z_OFFSETS[
                attempt_index
            ]
        )

        retry_seed = (
            pregrasp_position
            + np.array(
                [
                    0.0,
                    0.0,
                    z_offset,
                ],
                dtype=np.float64,
            )
        )

        print()
        print("----------------------------------------")
        print(
            f"[GRASP ATTEMPT "
            f"{attempt_number}/{GRASP_MAX_ATTEMPTS}]"
        )
        print("----------------------------------------")

        # Attempt 1 starts from the already-reached nominal PREGRASP.
        # Later attempts explicitly reset to a slightly different seed.
        if attempt_index > 0:
            print(
                f"[GRASP RETRY] Reset seed: "
                f"{np.round(retry_seed, 4)}"
            )

            reset_ok = await reset_grasp_retry_seed(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=(
                    articulation_policy
                ),
                reset_position=(
                    retry_seed
                ),
            )

            if not reset_ok:
                print(
                    "[GRASP RETRY] Could not recover "
                    "a PREGRASP seed; stopping retries."
                )
                break

        initial_tip = pose_position(
            stage,
            GRASP_TIP_PATH,
        )

        initial_error = float(
            np.linalg.norm(
                target_position
                - initial_tip
            )
        )

        print(
            f"Initial tip: "
            f"{np.round(initial_tip, 4)}"
        )

        print(
            f"Initial error: "
            f"{initial_error:.4f} m"
        )

        if (
            initial_error
            <= GRASP_TOLERANCE
            + GRASP_TOLERANCE_EPS
        ):
            print(
                "[GRASPPOINT REACHED BEFORE NEW COMMAND]"
            )

            return (
                True,
                [
                    get_active_arm_positions(
                        arm,
                        arm_indices,
                    ).copy()
                ],
            )

        rmpflow.set_end_effector_target(
            target_position,
            None,
        )

        q0 = get_active_arm_positions(
            arm,
            arm_indices,
        ).copy()

        attempt_trajectory = [
            q0
        ]

        best_error = (
            initial_error
        )

        best_tip = (
            initial_tip.copy()
        )

        significant_best = (
            initial_error
        )

        plateau_count = 0
        divergence_count = 0

        for step in range(
            GRASP_ATTEMPT_MAX_STEPS
        ):
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )

            update_rmpflow_base_pose(
                stage,
                rmpflow,
            )

            action = (
                articulation_policy
                .get_next_articulation_action(
                    RMPFLOW_PHYSICS_DT
                )
            )

            arm.apply_action(
                action
            )

            await next_frame()

            q_actual = get_active_arm_positions(
                arm,
                arm_indices,
            ).copy()

            attempt_trajectory.append(
                q_actual
            )

            tip_position = pose_position(
                stage,
                GRASP_TIP_PATH,
            )

            error = float(
                np.linalg.norm(
                    target_position
                    - tip_position
                )
            )

            if error < best_error:
                best_error = error
                best_tip = (
                    tip_position.copy()
                )

            if error < overall_best_error:
                overall_best_error = error
                overall_best_tip = (
                    tip_position.copy()
                )

            # ----------------------------------------------
            # Plateau supervision
            # ----------------------------------------------
            if (
                significant_best
                - best_error
                >= GRASP_PLATEAU_MIN_IMPROVEMENT
            ):
                significant_best = (
                    best_error
                )
                plateau_count = 0
            else:
                plateau_count += 1

            # ----------------------------------------------
            # Divergence supervision
            # ----------------------------------------------
            if (
                error
                > best_error
                + GRASP_DIVERGENCE_MARGIN
            ):
                divergence_count += 1
            else:
                divergence_count = 0

            if (
                step == 0
                or (step + 1)
                % GRASP_PROGRESS_PRINT_EVERY
                == 0
            ):
                print(
                    f"[GRASPPOINT] "
                    f"attempt={attempt_number} | "
                    f"step={step:3d} | "
                    f"error={error:.4f} m | "
                    f"best={best_error:.4f} m | "
                    f"plateau={plateau_count}/"
                    f"{GRASP_PLATEAU_CONSECUTIVE_STEPS}"
                )

            # ----------------------------------------------
            # Actual task success
            # ----------------------------------------------
            if (
                error
                <= GRASP_TOLERANCE
                + GRASP_TOLERANCE_EPS
            ):
                print()
                print(
                    "[GRASPPOINT REACHED]"
                )

                print(
                    f"Attempt: "
                    f"{attempt_number}/{GRASP_MAX_ATTEMPTS}"
                )

                print(
                    f"Physical GraspTip: "
                    f"{np.round(tip_position, 4)}"
                )

                print(
                    f"Target error: "
                    f"{error:.4f} m"
                )

                successful_trajectory = (
                    attempt_trajectory
                )

                return (
                    True,
                    successful_trajectory,
                )

            if (
                plateau_count
                >= GRASP_PLATEAU_CONSECUTIVE_STEPS
            ):
                print()
                print(
                    f"[GRASP ATTEMPT {attempt_number}] "
                    "Plateau detected; stop this local solve."
                )

                print(
                    f"Best attempt error: "
                    f"{best_error:.4f} m"
                )

                break

            if (
                divergence_count
                >= GRASP_DIVERGENCE_CONSECUTIVE_STEPS
            ):
                print()
                print(
                    f"[GRASP ATTEMPT {attempt_number}] "
                    "Divergence detected; stop this local solve."
                )

                print(
                    f"Best attempt error: "
                    f"{best_error:.4f} m"
                )

                break

        print(
            f"[GRASP ATTEMPT {attempt_number} ENDED] "
            f"best={best_error:.4f} m | "
            f"best_tip={np.round(best_tip, 4)}"
        )

    # --------------------------------------------------------
    # All bounded attempts failed.
    # --------------------------------------------------------
    final_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    final_error = float(
        np.linalg.norm(
            final_tip
            - target_position
        )
    )

    print()
    print("========================================")
    print("[FAILED] ALL GRASPPOINT ATTEMPTS")
    print("========================================")

    print(
        f"Final error: "
        f"{final_error:.4f} m"
    )

    print(
        f"Best error across attempts: "
        f"{overall_best_error:.4f} m"
    )

    if overall_best_tip is not None:
        print(
            f"Best GraspTip across attempts: "
            f"{np.round(overall_best_tip, 4)}"
        )

    return (
        False,
        successful_trajectory,
    )





async def align_cutter_and_cut(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    grasp_position,
    cut_position,
    cutter_tip_path,
    pregrasp_position=None,
):
    """
    Keep GraspTip at GraspPoint while orienting the rigid tool so
    CutterTip reaches CutPoint.

    Hard success criterion (unchanged):
        GraspTip  -> GraspPoint <= 3.0 cm
        CutterTip -> CutPoint   <= 4.0 cm

    Robustness (like GRASP):
        - up to CUT_MAX_ATTEMPTS orientation attempts
        - track best_q / best grasp+cutter errors
        - on diverge/timeout: restore best_q, hold, remeasure
        - accept only if restored pose meets the hard criterion
        - otherwise back off to a nearby PREGRASP seed and retry

    best_score / CUT_NEAR_SCORE are diagnostics + divergence triggers only.
    They never grant automatic CUT success.

    Returns:
        success, joint_trajectory
    """
    grasp_position = np.asarray(
        grasp_position,
        dtype=np.float64,
    )

    cut_position = np.asarray(
        cut_position,
        dtype=np.float64,
    )

    if pregrasp_position is None:
        pregrasp_position = (
            grasp_position.copy()
        )
    else:
        pregrasp_position = np.asarray(
            pregrasp_position,
            dtype=np.float64,
        )

    def _cut_errors_from_pose():
        tip, _ = pose_position_orientation(
            stage,
            GRASP_TIP_PATH,
        )
        cutter = pose_position(
            stage,
            cutter_tip_path,
        )
        g_err = float(
            np.linalg.norm(
                tip - grasp_position
            )
        )
        c_err = float(
            np.linalg.norm(
                cutter - cut_position
            )
        )
        return tip, cutter, g_err, c_err

    def _in_cut_region(g_err, c_err):
        return (
            g_err
            <= CUT_GRASP_HOLD_TOLERANCE
            + CUT_TOLERANCE_EPS
            and c_err
            <= CUTTER_TOLERANCE
            + CUT_TOLERANCE_EPS
        )

    (
        current_grasp_tip,
        current_grasp_orientation,
    ) = pose_position_orientation(
        stage,
        GRASP_TIP_PATH,
    )

    current_cutter_tip = pose_position(
        stage,
        cutter_tip_path,
    )

    tool_vector = (
        current_cutter_tip
        - current_grasp_tip
    )

    crop_vector = (
        cut_position
        - grasp_position
    )

    tool_length = float(
        np.linalg.norm(
            tool_vector
        )
    )

    crop_length = float(
        np.linalg.norm(
            crop_vector
        )
    )

    print()
    print("========================================")
    print(
        "[CUT ALIGNMENT] BOUNDED RETRIES "
        f"({CUT_MAX_ATTEMPTS} max)"
    )
    print("========================================")

    print(
        f"Physical GraspTip: "
        f"{np.round(current_grasp_tip, 4)}"
    )

    print(
        f"Physical CutterTip: "
        f"{np.round(current_cutter_tip, 4)}"
    )

    print(
        f"GraspPoint: "
        f"{np.round(grasp_position, 4)}"
    )

    print(
        f"CutPoint: "
        f"{np.round(cut_position, 4)}"
    )

    print(
        f"Hard success: grasp <= "
        f"{CUT_GRASP_HOLD_TOLERANCE:.3f} m | "
        f"cutter <= {CUTTER_TOLERANCE:.3f} m"
    )

    print(
        f"Tool marker spacing: "
        f"{tool_length:.4f} m"
    )

    print(
        f"Target marker spacing: "
        f"{crop_length:.4f} m"
    )

    print(
        f"Unavoidable spacing mismatch: "
        f"{abs(tool_length - crop_length):.4f} m"
    )

    current_grasp_error = float(
        np.linalg.norm(
            current_grasp_tip
            - grasp_position
        )
    )

    current_cutter_error = float(
        np.linalg.norm(
            current_cutter_tip
            - cut_position
        )
    )

    print(
        f"Current GraspTip error: "
        f"{current_grasp_error:.4f} m"
    )

    print(
        f"Current CutterTip error: "
        f"{current_cutter_error:.4f} m"
    )

    if _in_cut_region(
        current_grasp_error,
        current_cutter_error,
    ):
        current_q = get_active_arm_positions(
            arm,
            arm_indices,
        ).copy()

        print()
        print(
            "[CUTTERTIP ALREADY AT CUTPOINT]"
        )

        print(
            f"GraspTip error: "
            f"{current_grasp_error:.4f} m "
            f"(tol={CUT_GRASP_HOLD_TOLERANCE:.4f})"
        )

        print(
            f"CutterTip error: "
            f"{current_cutter_error:.4f} m "
            f"(tol={CUTTER_TOLERANCE:.4f})"
        )

        print(
            "[CUT ALIGNMENT] Current physical pose is inside the "
            "logical CUT region."
        )

        print(
            "[CUT ALIGNMENT] CUT accepted immediately; "
            "no extra RMPflow motion."
        )

        return (
            True,
            [current_q],
        )

    overall_trajectory = []
    overall_best_grasp = float("inf")
    overall_best_cutter = float("inf")

    async def _restore_best_and_remeasure(
        best_q,
        attempt_trajectory,
    ):
        print(
            "[CUT ALIGN] Restoring best_q for "
            "remeasurement (not soft-accept)."
        )

        for _ in range(
            CUT_BEST_RESTORE_HOLD_FRAMES
        ):
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )

            hold_arm_pose(
                arm,
                arm_indices,
                best_q,
            )

            await next_frame()

            attempt_trajectory.append(
                get_active_arm_positions(
                    arm,
                    arm_indices,
                ).copy()
            )

        tip, cutter, g_err, c_err = (
            _cut_errors_from_pose()
        )

        print(
            f"[CUT ALIGN] Remeasure after restore: "
            f"grasp={g_err:.4f} m | "
            f"cutter={c_err:.4f} m | "
            f"tip={np.round(tip, 4)} | "
            f"cutter_tip={np.round(cutter, 4)}"
        )

        return g_err, c_err

    async def _backoff_cut_retry_seed(
        attempt_index,
    ):
        z_offset = float(
            CUT_RETRY_SEED_Z_OFFSETS[
                min(
                    attempt_index,
                    len(CUT_RETRY_SEED_Z_OFFSETS)
                    - 1,
                )
            ]
        )

        seed = (
            pregrasp_position
            + np.array(
                [
                    0.0,
                    0.0,
                    z_offset,
                ],
                dtype=np.float64,
            )
        )

        print(
            f"[CUT RETRY] Back off to seed: "
            f"{np.round(seed, 4)} "
            f"(PREGRASP + z={z_offset:+.3f})"
        )

        rmpflow.set_end_effector_target(
            seed,
            None,
        )

        best_seed_err = float("inf")

        for step in range(
            CUT_RETRY_RESET_MAX_STEPS
        ):
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )

            update_rmpflow_base_pose(
                stage,
                rmpflow,
            )

            action = (
                articulation_policy
                .get_next_articulation_action(
                    RMPFLOW_PHYSICS_DT
                )
            )

            arm.apply_action(
                action
            )

            await next_frame()

            tip = pose_position(
                stage,
                GRASP_TIP_PATH,
            )

            err = float(
                np.linalg.norm(
                    tip - seed
                )
            )

            best_seed_err = min(
                best_seed_err,
                err,
            )

            if (
                err
                <= CUT_RETRY_RESET_TOLERANCE
            ):
                print(
                    f"[CUT RETRY] Seed reached "
                    f"(error={err:.4f} m)"
                )
                break
        else:
            print(
                f"[CUT RETRY] Seed approach incomplete "
                f"(best={best_seed_err:.4f} m); "
                "continuing to re-grasp."
            )

        # Re-approach GraspPoint (position only) before next
        # orientation solve.
        print(
            "[CUT RETRY] Re-approach GraspPoint "
            "before next orientation attempt."
        )

        rmpflow.set_end_effector_target(
            grasp_position,
            None,
        )

        best_grasp_err = float("inf")

        for step in range(
            CUT_RETRY_REGRASP_MAX_STEPS
        ):
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )

            update_rmpflow_base_pose(
                stage,
                rmpflow,
            )

            action = (
                articulation_policy
                .get_next_articulation_action(
                    RMPFLOW_PHYSICS_DT
                )
            )

            arm.apply_action(
                action
            )

            await next_frame()

            tip = pose_position(
                stage,
                GRASP_TIP_PATH,
            )

            err = float(
                np.linalg.norm(
                    tip - grasp_position
                )
            )

            best_grasp_err = min(
                best_grasp_err,
                err,
            )

            if (
                err
                <= CUT_RETRY_REGRASP_TOLERANCE
            ):
                print(
                    f"[CUT RETRY] Re-grasp reached "
                    f"(error={err:.4f} m)"
                )
                return True

        print(
            f"[CUT RETRY] Re-grasp incomplete "
            f"(best={best_grasp_err:.4f} m)"
        )

        return best_grasp_err <= (
            CUT_RETRY_REGRASP_TOLERANCE
            + 0.020
        )

    for attempt_index in range(
        CUT_MAX_ATTEMPTS
    ):
        attempt_number = (
            attempt_index + 1
        )

        print()
        print("----------------------------------------")
        print(
            f"[CUT ATTEMPT "
            f"{attempt_number}/{CUT_MAX_ATTEMPTS}]"
        )
        print("----------------------------------------")

        if attempt_index > 0:
            seed_ok = await _backoff_cut_retry_seed(
                attempt_index
            )

            if not seed_ok:
                print(
                    "[CUT RETRY] Could not recover a "
                    "nearby seed; stopping retries."
                )
                break

        (
            tip_now,
            orient_now,
        ) = pose_position_orientation(
            stage,
            GRASP_TIP_PATH,
        )

        cutter_now = pose_position(
            stage,
            cutter_tip_path,
        )

        g0 = float(
            np.linalg.norm(
                tip_now - grasp_position
            )
        )

        c0 = float(
            np.linalg.norm(
                cutter_now - cut_position
            )
        )

        print(
            f"Attempt start errors: "
            f"grasp={g0:.4f} m | "
            f"cutter={c0:.4f} m"
        )

        if _in_cut_region(g0, c0):
            q_now = get_active_arm_positions(
                arm,
                arm_indices,
            ).copy()

            print(
                "[CUT REGION REACHED] Already inside "
                "hard criterion at attempt start."
            )

            return (
                True,
                overall_trajectory + [q_now],
            )

        tool_vector = (
            cutter_now - tip_now
        )

        crop_vector = (
            cut_position
            - grasp_position
        )

        try:
            delta_q = (
                quaternion_from_two_vectors(
                    tool_vector,
                    crop_vector,
                )
            )
        except Exception as exc:
            print(
                f"[CUT ATTEMPT] Orientation solve failed: "
                f"{exc}"
            )
            continue

        desired_grasp_orientation = (
            quaternion_multiply(
                delta_q,
                orient_now,
            )
        )

        print(
            f"Desired GraspTip orientation [wxyz]: "
            f"{np.round(desired_grasp_orientation, 4)}"
        )

        rmpflow.set_end_effector_target(
            grasp_position,
            desired_grasp_orientation,
        )

        q0 = get_active_arm_positions(
            arm,
            arm_indices,
        ).copy()

        trajectory = [
            q0
        ]

        best_score = float("inf")
        best_q = q0.copy()
        best_grasp_error = float("inf")
        best_cutter_error = float("inf")
        divergence_count = 0
        ended_by = "timeout"

        for step in range(
            CUT_ATTEMPT_MAX_STEPS
        ):
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )

            update_rmpflow_base_pose(
                stage,
                rmpflow,
            )

            action = (
                articulation_policy
                .get_next_articulation_action(
                    RMPFLOW_PHYSICS_DT
                )
            )

            arm.apply_action(
                action
            )

            await next_frame()

            q_actual = get_active_arm_positions(
                arm,
                arm_indices,
            ).copy()

            trajectory.append(
                q_actual
            )

            grasp_tip = pose_position(
                stage,
                GRASP_TIP_PATH,
            )

            cutter_tip = pose_position(
                stage,
                cutter_tip_path,
            )

            grasp_error = float(
                np.linalg.norm(
                    grasp_tip
                    - grasp_position
                )
            )

            cutter_error = float(
                np.linalg.norm(
                    cutter_tip
                    - cut_position
                )
            )

            # Diagnostic / divergence only — never a success gate.
            score = max(
                grasp_error
                / CUT_GRASP_HOLD_TOLERANCE,
                cutter_error
                / CUTTER_TOLERANCE,
            )

            if score < best_score:
                best_score = score
                best_q = q_actual.copy()
                best_grasp_error = (
                    grasp_error
                )
                best_cutter_error = (
                    cutter_error
                )
                divergence_count = 0

            elif (
                best_score <= CUT_NEAR_SCORE
                and score
                > best_score
                + CUT_DIVERGENCE_MARGIN
            ):
                divergence_count += 1

            else:
                divergence_count = 0

            if (
                step % CUT_PROGRESS_PRINT_EVERY
                == 0
            ):
                print(
                    f"[CUT ALIGN] step={step:4d} | "
                    f"grasp_err={grasp_error:.4f} m | "
                    f"cutter_err={cutter_error:.4f} m | "
                    f"score={score:.3f} | "
                    f"best={best_score:.3f}"
                )

            if _in_cut_region(
                grasp_error,
                cutter_error,
            ):
                print()
                print(
                    "[CUT REGION REACHED]"
                )

                print(
                    f"GraspTip error: "
                    f"{grasp_error:.4f} m "
                    f"(tol={CUT_GRASP_HOLD_TOLERANCE:.4f})"
                )

                print(
                    f"CutterTip error: "
                    f"{cutter_error:.4f} m "
                    f"(tol={CUTTER_TOLERANCE:.4f})"
                )

                print(
                    "[CUT ALIGNMENT] CUT accepted at the "
                    "first valid physical pose "
                    f"(attempt {attempt_number})."
                )

                return (
                    True,
                    overall_trajectory
                    + trajectory,
                )

            if (
                best_score
                <= CUT_NEAR_SCORE
                and divergence_count
                >= CUT_DIVERGENCE_CONSECUTIVE_STEPS
            ):
                ended_by = "divergence"
                print()
                print(
                    "[CUT ALIGN] RMPflow diverging before "
                    "entering the logical CUT region."
                )

                print(
                    f"[CUT ALIGN] Best observed errors: "
                    f"grasp={best_grasp_error:.4f} m | "
                    f"cutter={best_cutter_error:.4f} m"
                )
                break

        else:
            print()
            print(
                f"[CUT ALIGN] Attempt {attempt_number} "
                "timed out."
            )

            print(
                f"[CUT ALIGN] Best observed errors: "
                f"grasp={best_grasp_error:.4f} m | "
                f"cutter={best_cutter_error:.4f} m"
            )

        overall_best_grasp = min(
            overall_best_grasp,
            best_grasp_error,
        )

        overall_best_cutter = min(
            overall_best_cutter,
            best_cutter_error,
        )

        g_rem, c_rem = (
            await _restore_best_and_remeasure(
                best_q,
                trajectory,
            )
        )

        overall_trajectory.extend(
            trajectory
        )

        if _in_cut_region(g_rem, c_rem):
            print()
            print(
                "[CUT REGION REACHED] After best_q restore "
                "+ remeasure (hard 3 cm / 4 cm)."
            )

            print(
                f"GraspTip error: "
                f"{g_rem:.4f} m "
                f"(tol={CUT_GRASP_HOLD_TOLERANCE:.4f})"
            )

            print(
                f"CutterTip error: "
                f"{c_rem:.4f} m "
                f"(tol={CUTTER_TOLERANCE:.4f})"
            )

            return (
                True,
                overall_trajectory,
            )

        print(
            f"[CUT ALIGN] Restore did not meet hard "
            f"criterion after {ended_by}; "
            f"remeasure grasp={g_rem:.4f} m | "
            f"cutter={c_rem:.4f} m"
        )

        if attempt_number < CUT_MAX_ATTEMPTS:
            print(
                "[CUT ALIGN] Will back off and retry "
                "orientation from a nearby seed."
            )

    print()
    print(
        "[FAILED] Could not align CutterTip with CutPoint "
        "while holding GraspTip within hard tolerances."
    )

    print(
        f"Best grasp across attempts: "
        f"{overall_best_grasp:.4f} m "
        f"(need <= {CUT_GRASP_HOLD_TOLERANCE:.3f})"
    )

    print(
        f"Best cutter across attempts: "
        f"{overall_best_cutter:.4f} m "
        f"(need <= {CUTTER_TOLERANCE:.3f})"
    )

    return (
        False,
        overall_trajectory,
    )


async def _post_cut_retreat_attempt(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    pregrasp_position,
    payload,
    attempt_index,
    state,
):
    """
    One receding-horizon post-CUT Cartesian retreat attempt.

    Goal:
        move physical GraspTip back into the PREGRASP safe region.

    Strategy:
        current physical GraspTip
        -> 1 cm toward PREGRASP
        -> require the physical tip to come within 3 mm of that target
           (or use up the short 20-frame command burst)
        -> re-measure actual GraspTip
        -> generate a NEW 1 cm target
        -> repeat

    This avoids the fixed-waypoint failure mode where early waypoints
    were accepted without actual motion and later targets became too far
    from the real physical state.

    No historical joint replay.
    No orientation constraint.

    `state` carries the best observed retreat pose across attempts so
    the caller can restore it; it is never used to accept a retreat.

    Returns:
        True only if the unchanged POSTCUT_RETREAT_TOLERANCE criterion
        is physically satisfied.
    """
    pregrasp_position = np.asarray(
        pregrasp_position,
        dtype=np.float64,
    )

    print()
    print("========================================")
    print(
        f"[POST-CUT RECEDING CARTESIAN RETREAT] "
        f"attempt {attempt_index}/"
        f"{POSTCUT_RETREAT_MAX_ATTEMPTS}"
    )
    print("CURRENT TIP -> 1 CM TARGET -> RE-MEASURE")
    print("========================================")

    initial_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    initial_error = float(
        np.linalg.norm(
            initial_tip
            - pregrasp_position
        )
    )

    print(
        f"Current GraspTip: "
        f"{np.round(initial_tip, 4)}"
    )

    print(
        f"PREGRASP target:  "
        f"{np.round(pregrasp_position, 4)}"
    )

    print(
        f"Initial PREGRASP error: "
        f"{initial_error:.4f} m"
    )

    if initial_error < state["best_error"]:
        state["best_error"] = initial_error
        state["best_q"] = get_active_arm_positions(
            arm,
            arm_indices,
        ).copy()

    if (
        initial_error
        <= POSTCUT_RETREAT_TOLERANCE
    ):
        print()
        print(
            "[PREGRASP RECOVERED AFTER CUT]"
        )

        print(
            "[RETREAT] Current CUT pose is already inside "
            "the PREGRASP safe region."
        )

        return True

    best_pregrasp_error = (
        initial_error
    )

    previous_pregrasp_error = (
        initial_error
    )

    no_progress_count = 0
    divergence_count = 0

    # --------------------------------------------------------
    # Receding-horizon loop
    # --------------------------------------------------------
    for target_index in range(
        1,
        POSTCUT_RETREAT_MAX_TARGETS + 1,
    ):
        current_tip = pose_position(
            stage,
            GRASP_TIP_PATH,
        )

        to_pregrasp = (
            pregrasp_position
            - current_tip
        )

        pregrasp_error_before = float(
            np.linalg.norm(
                to_pregrasp
            )
        )

        # Final task criterion always wins.
        if (
            pregrasp_error_before
            <= POSTCUT_RETREAT_TOLERANCE
        ):
            print()
            print(
                "[PREGRASP RECOVERED AFTER CUT]"
            )

            print(
                f"Reached before target "
                f"{target_index}/{POSTCUT_RETREAT_MAX_TARGETS}"
            )

            print(
                f"Physical GraspTip: "
                f"{np.round(current_tip, 4)}"
            )

            print(
                f"Cartesian error: "
                f"{pregrasp_error_before:.4f} m"
            )

            return True

        direction_norm = float(
            np.linalg.norm(
                to_pregrasp
            )
        )

        if direction_norm <= 1e-9:
            print()
            print(
                "[PREGRASP RECOVERED AFTER CUT]"
            )
            return True

        direction = (
            to_pregrasp
            / direction_norm
        )

        step_distance = min(
            POSTCUT_RETREAT_STEP_DISTANCE,
            pregrasp_error_before,
        )

        local_target = (
            current_tip
            + direction * step_distance
        )

        if (
            target_index
            % POSTCUT_RETREAT_PRINT_EVERY_TARGET
            == 0
        ):
            print()
            print(
                f"[RETREAT TARGET "
                f"{target_index:02d}/"
                f"{POSTCUT_RETREAT_MAX_TARGETS:02d}]"
            )

            print(
                f"  current_tip="
                f"{np.round(current_tip, 4)}"
            )

            print(
                f"  local_target="
                f"{np.round(local_target, 4)}"
            )

            print(
                f"  PREGRASP error before="
                f"{pregrasp_error_before:.4f} m"
            )

            print(
                f"  local target tolerance="
                f"{POSTCUT_RETREAT_LOCAL_TARGET_TOLERANCE:.4f} m"
            )

        # Position-only command. No orientation constraint.
        rmpflow.set_end_effector_target(
            local_target,
            None,
        )

        best_local_error = float(
            np.linalg.norm(
                current_tip
                - local_target
            )
        )

        local_reached = False

        # ----------------------------------------------------
        # Move only briefly, then re-measure and regenerate.
        # ----------------------------------------------------
        for frame in range(
            POSTCUT_RETREAT_FRAMES_PER_TARGET
        ):
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )

            update_rmpflow_base_pose(
                stage,
                rmpflow,
            )

            action = (
                articulation_policy
                .get_next_articulation_action(
                    RMPFLOW_PHYSICS_DT
                )
            )

            arm.apply_action(
                action
            )

            await next_frame()

            update_cut_payload(
                stage,
                payload,
            )

            tip_now = pose_position(
                stage,
                GRASP_TIP_PATH,
            )

            local_error = float(
                np.linalg.norm(
                    tip_now
                    - local_target
                )
            )

            pregrasp_error_now = float(
                np.linalg.norm(
                    tip_now
                    - pregrasp_position
                )
            )

            if (
                pregrasp_error_now
                < best_pregrasp_error
            ):
                best_pregrasp_error = (
                    pregrasp_error_now
                )

            if (
                pregrasp_error_now
                < state["best_error"]
            ):
                state["best_error"] = (
                    pregrasp_error_now
                )

                state["best_q"] = (
                    get_active_arm_positions(
                        arm,
                        arm_indices,
                    ).copy()
                )

            best_local_error = min(
                best_local_error,
                local_error,
            )

            # Final PREGRASP region reached: stop immediately.
            if (
                pregrasp_error_now
                <= POSTCUT_RETREAT_TOLERANCE
            ):
                print()
                print(
                    "[PREGRASP RECOVERED AFTER CUT]"
                )

                print(
                    f"Reached during target "
                    f"{target_index}, frame={frame}"
                )

                print(
                    f"Physical GraspTip: "
                    f"{np.round(tip_now, 4)}"
                )

                print(
                    f"Cartesian error: "
                    f"{pregrasp_error_now:.4f} m"
                )

                return True

            # Short target is good enough ONLY after real motion has
            # brought the physical tip inside the tight 3 mm region.
            # Do not use a tolerance larger than the 1 cm target step.
            if (
                local_error
                <= POSTCUT_RETREAT_LOCAL_TARGET_TOLERANCE
            ):
                local_reached = True

                print(
                    f"[RETREAT LOCAL TARGET REACHED] "
                    f"target={target_index:02d} | "
                    f"frame={frame:02d} | "
                    f"local_error={local_error:.4f} m"
                )

                break

        # ----------------------------------------------------
        # Re-measure actual state after this short command burst
        # ----------------------------------------------------
        current_tip_after = pose_position(
            stage,
            GRASP_TIP_PATH,
        )

        pregrasp_error_after = float(
            np.linalg.norm(
                current_tip_after
                - pregrasp_position
            )
        )

        actual_progress = (
            pregrasp_error_before
            - pregrasp_error_after
        )

        print(
            f"[POST-CUT RETREAT] "
            f"target={target_index:02d} | "
            f"before={pregrasp_error_before:.4f} m | "
            f"after={pregrasp_error_after:.4f} m | "
            f"progress={actual_progress:+.4f} m | "
            f"best={best_pregrasp_error:.4f} m | "
            f"local_best={best_local_error:.4f} m | "
            f"local_reached={local_reached}"
        )

        # Final task criterion after burst.
        if (
            pregrasp_error_after
            <= POSTCUT_RETREAT_TOLERANCE
        ):
            print()
            print(
                "[PREGRASP RECOVERED AFTER CUT]"
            )

            print(
                f"Physical GraspTip: "
                f"{np.round(current_tip_after, 4)}"
            )

            print(
                f"Cartesian error: "
                f"{pregrasp_error_after:.4f} m"
            )

            return True

        # ----------------------------------------------------
        # Progress / divergence supervision
        # ----------------------------------------------------
        if (
            actual_progress
            >= POSTCUT_RETREAT_MIN_PROGRESS
        ):
            no_progress_count = 0
        else:
            no_progress_count += 1

        if (
            pregrasp_error_after
            > best_pregrasp_error
            + POSTCUT_RETREAT_DIVERGENCE_MARGIN
        ):
            divergence_count += 1
        else:
            divergence_count = 0

        if (
            divergence_count
            >= POSTCUT_RETREAT_DIVERGENCE_LIMIT
        ):
            print()
            print(
                f"[RETREAT ATTEMPT {attempt_index}] "
                "Stopped: receding post-CUT retreat is "
                "consistently diverging."
            )

            print(
                f"Best PREGRASP error: "
                f"{best_pregrasp_error:.4f} m"
            )

            print(
                f"Current PREGRASP error: "
                f"{pregrasp_error_after:.4f} m"
            )

            return False

        if (
            no_progress_count
            >= POSTCUT_RETREAT_NO_PROGRESS_LIMIT
        ):
            print()
            print(
                f"[RETREAT ATTEMPT {attempt_index}] "
                "Stopped: receding post-CUT retreat made "
                "insufficient physical progress."
            )

            print(
                f"Best PREGRASP error: "
                f"{best_pregrasp_error:.4f} m"
            )

            print(
                f"Current PREGRASP error: "
                f"{pregrasp_error_after:.4f} m"
            )

            return False

        previous_pregrasp_error = (
            pregrasp_error_after
        )

    # --------------------------------------------------------
    # Global timeout
    # --------------------------------------------------------
    final_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    final_error = float(
        np.linalg.norm(
            final_tip
            - pregrasp_position
        )
    )

    print()
    print(
        f"[RETREAT ATTEMPT {attempt_index}] "
        "Stopped: receding post-CUT retreat reached "
        "the target-count limit."
    )

    print(
        f"Final PREGRASP error: "
        f"{final_error:.4f} m"
    )

    print(
        f"Best PREGRASP error: "
        f"{best_pregrasp_error:.4f} m"
    )

    return False


async def _restore_best_retreat_pose(
    stage,
    jackal,
    arm,
    arm_indices,
    payload,
    best_q,
    pregrasp_position,
):
    """
    Hold the best observed retreat pose, then remeasure the PHYSICAL
    PREGRASP error.

    This is recovery and remeasurement only. The returned error is
    still judged against the unchanged POSTCUT_RETREAT_TOLERANCE.
    """
    print()
    print(
        "[POST-CUT RETREAT] Restoring best observed retreat "
        "state for remeasurement (not a soft accept)."
    )

    for _ in range(
        POSTCUT_RETREAT_BEST_RESTORE_HOLD_FRAMES
    ):
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        hold_arm_pose(
            arm,
            arm_indices,
            best_q,
        )

        await next_frame()

        update_cut_payload(
            stage,
            payload,
        )

    restored_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    restored_error = float(
        np.linalg.norm(
            restored_tip
            - pregrasp_position
        )
    )

    print(
        f"[POST-CUT RETREAT] Remeasured after restore: "
        f"tip={np.round(restored_tip, 4)} | "
        f"error={restored_error:.4f} m "
        f"(need <= {POSTCUT_RETREAT_TOLERANCE:.3f})"
    )

    return restored_error


async def rmpflow_retreat_to_pregrasp_after_cut(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    pregrasp_position,
    payload=None,
):
    """
    Bounded post-CUT retreat: up to POSTCUT_RETREAT_MAX_ATTEMPTS runs
    of the receding-horizon controller.

    On divergence or timeout the local solve is stopped, the best
    observed retreat state is restored and remeasured, and the retreat
    runs again from there.

    The success criterion never changes: the physical GraspTip must be
    within POSTCUT_RETREAT_TOLERANCE of PREGRASP. Restored poses are
    accepted only when they satisfy that same criterion.
    """
    pregrasp_position = np.asarray(
        pregrasp_position,
        dtype=np.float64,
    )

    state = {
        "best_error": float("inf"),
        "best_q": None,
    }

    attempts_run = 0

    for attempt_index in range(
        1,
        POSTCUT_RETREAT_MAX_ATTEMPTS + 1,
    ):
        attempts_run = attempt_index

        reached = await _post_cut_retreat_attempt(
            stage=stage,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            rmpflow=rmpflow,
            articulation_policy=articulation_policy,
            pregrasp_position=pregrasp_position,
            payload=payload,
            attempt_index=attempt_index,
            state=state,
        )

        if reached:
            return True

        if state["best_q"] is None:
            break

        restored_error = (
            await _restore_best_retreat_pose(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                payload=payload,
                best_q=state["best_q"],
                pregrasp_position=pregrasp_position,
            )
        )

        if (
            restored_error
            <= POSTCUT_RETREAT_TOLERANCE
        ):
            print()
            print(
                "[PREGRASP RECOVERED AFTER CUT]"
            )

            print(
                "Restored best retreat state satisfies the "
                "original criterion: "
                f"{restored_error:.4f} m"
            )

            return True

        if (
            attempt_index
            < POSTCUT_RETREAT_MAX_ATTEMPTS
        ):
            print(
                f"[POST-CUT RETREAT] Attempt "
                f"{attempt_index} did not satisfy "
                f"{POSTCUT_RETREAT_TOLERANCE:.3f} m; "
                "retrying the receding retreat from the "
                "restored state."
            )

    print()
    print(
        "[FAILED] Post-CUT retreat did not satisfy the "
        f"{POSTCUT_RETREAT_TOLERANCE:.3f} m PREGRASP criterion "
        f"after {attempts_run} attempt(s)."
    )

    print(
        f"Best PREGRASP error across attempts: "
        f"{state['best_error']:.4f} m"
    )

    return False


async def replay_pregrasp_path_to_upright(
    stage,
    jackal,
    arm,
    arm_indices,
    joint_trajectory,
    recorded_upright_tip,
    recorded_upright_tip_in_arm_base,
    payload=None,
):
    """
    Replay the measured successful UPRIGHT -> PREGRASP joint path
    in reverse.

    Key difference from the old implementation:
      - each recorded waypoint is held until the physical joints catch
        up (or a small per-waypoint timeout is reached)
      - the final target is the ACTUAL recorded upright configuration
        joint_trajectory[0], not the ideal UR5E_UPRIGHT_Q array

    That recorded start pose was physically achieved and validated
    immediately before the successful RMPflow PREGRASP motion.
    """
    print()
    print("========================================")
    print("[RETURN] REPLAY PREGRASP PATH BACKWARD")
    print("========================================")

    if (
        joint_trajectory is None
        or len(joint_trajectory) < 2
    ):
        print(
            "[FAILED] No valid PREGRASP trajectory "
            "was recorded."
        )
        return False

    recorded_upright_q = np.asarray(
        joint_trajectory[0],
        dtype=np.float64,
    ).copy()

    reverse_path = list(
        reversed(
            joint_trajectory[
                ::RETURN_REPLAY_STRIDE
            ]
        )
    )

    # Ensure the exact measured start configuration is the final target.
    if not np.allclose(
        reverse_path[-1],
        recorded_upright_q,
    ):
        reverse_path.append(
            recorded_upright_q.copy()
        )

    total_points = len(
        reverse_path
    )

    print(
        f"[RETURN] Replaying "
        f"{total_points} joint configurations."
    )

    print(
        "[RETURN] Final target is recorded physical "
        "SAFE UPRIGHT configuration:"
    )

    print(
        f"         {np.round(recorded_upright_q, 4)}"
    )

    worst_tracking_error = 0.0

    for point_index, q_target in enumerate(
        reverse_path
    ):
        q_target = np.asarray(
            q_target,
            dtype=np.float64,
        )

        tracking_error = float("inf")
        frames_used = 0

        # Hold each waypoint until the physical arm catches up.
        for frame in range(
            RETURN_WAYPOINT_MAX_FRAMES
        ):
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )

            hold_arm_pose(
                arm,
                arm_indices,
                q_target,
            )

            await next_frame()

            update_cut_payload(
                stage,
                payload,
            )

            q_actual = (
                get_active_arm_positions(
                    arm,
                    arm_indices,
                )
            )

            tracking_error = float(
                np.max(
                    np.abs(
                        q_actual
                        - q_target
                    )
                )
            )

            frames_used = frame + 1

            if (
                tracking_error
                <= RETURN_WAYPOINT_TOLERANCE
            ):
                break

        worst_tracking_error = max(
            worst_tracking_error,
            tracking_error,
        )

        if (
            point_index % 10 == 0
            or point_index
            == total_points - 1
        ):
            print(
                f"[RETURN REPLAY] "
                f"{point_index + 1:4d}/"
                f"{total_points:4d} | "
                f"tracking_error="
                f"{tracking_error:.4f} rad | "
                f"frames={frames_used}"
            )

    # --------------------------------------------------------
    # Final settle at the ACTUAL recorded upright configuration
    # --------------------------------------------------------
    final_error = float("inf")

    for frame in range(
        RETURN_FINAL_HOLD_MAX_FRAMES
    ):
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        hold_arm_pose(
            arm,
            arm_indices,
            recorded_upright_q,
        )

        await next_frame()

        update_cut_payload(
            stage,
            payload,
        )

        q_actual = get_active_arm_positions(
            arm,
            arm_indices,
        )

        final_error = float(
            np.max(
                np.abs(
                    q_actual
                    - recorded_upright_q
                )
            )
        )

        if (
            final_error
            <= RETURN_FINAL_TOLERANCE
        ):
            print(
                "[UPRIGHT REPLAY] Joint-space tolerance reached; "
                "validating physical GraspTip pose."
            )
            break

    q_actual = get_active_arm_positions(
        arm,
        arm_indices,
    )

    final_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    recorded_upright_tip = np.asarray(
        recorded_upright_tip,
        dtype=np.float64,
    )

    # World-space error is retained as a diagnostic only.
    upright_world_cart_error = float(
        np.linalg.norm(
            final_tip
            - recorded_upright_tip
        )
    )

    current_tip_in_arm_base = (
        grasp_tip_position_in_arm_base(
            stage
        )
    )

    recorded_upright_tip_in_arm_base = np.asarray(
        recorded_upright_tip_in_arm_base,
        dtype=np.float64,
    )

    upright_base_relative_error = float(
        np.linalg.norm(
            current_tip_in_arm_base
            - recorded_upright_tip_in_arm_base
        )
    )

    # Recompute from the final physical q so the value printed here is
    # independent of where the settle loop happened to break.
    final_error = float(
        np.max(
            np.abs(
                q_actual
                - recorded_upright_q
            )
        )
    )

    print()
    print("========================================")
    print("[UPRIGHT RETURN VALIDATION]")
    print("========================================")

    print(
        f"Recorded upright q: "
        f"{np.round(recorded_upright_q, 4)}"
    )

    print(
        f"Actual q:           "
        f"{np.round(q_actual, 4)}"
    )

    print(
        f"Joint error: "
        f"{final_error:.4f} rad | "
        f"tol={RETURN_FINAL_TOLERANCE:.4f}"
    )

    print(
        f"Recorded upright GraspTip WORLD: "
        f"{np.round(recorded_upright_tip, 4)}"
    )

    print(
        f"Current GraspTip WORLD:          "
        f"{np.round(final_tip, 4)}"
    )

    print(
        f"World Cartesian error "
        f"(diagnostic only): "
        f"{upright_world_cart_error:.4f} m"
    )

    print(
        f"Recorded GraspTip in UR5e base: "
        f"{np.round(recorded_upright_tip_in_arm_base, 4)}"
    )

    print(
        f"Current GraspTip in UR5e base:  "
        f"{np.round(current_tip_in_arm_base, 4)}"
    )

    print(
        f"Base-relative Cartesian error: "
        f"{upright_base_relative_error:.4f} m | "
        f"tol={RETURN_UPRIGHT_CART_TOLERANCE:.4f}"
    )

    # --------------------------------------------------------
    # SAFE UPRIGHT PASS/FAIL
    # --------------------------------------------------------
    #
    # Joint-space recovery is sufficient by itself. This was the
    # original physically meaningful safe-upright condition.
    #
    # A base-relative Cartesian match is an alternative acceptance path
    # for redundant wrist configurations.
    #
    # WORLD-space tip displacement does NOT reject a valid return.
    joint_ok = (
        final_error
        <= RETURN_FINAL_TOLERANCE
    )

    base_relative_ok = (
        upright_base_relative_error
        <= RETURN_UPRIGHT_CART_TOLERANCE
    )

    if joint_ok:
        print()
        print(
            "[UPRIGHT RECOVERED BY JOINT STATE]"
        )

        if (
            upright_world_cart_error
            > RETURN_UPRIGHT_CART_TOLERANCE
        ):
            print(
                "[UPRIGHT NOTE] World GraspTip moved because the "
                "mobile base/world pose changed; this does not invalidate "
                "the recovered UR5e joint state."
            )

        return True

    if base_relative_ok:
        print()
        print(
            "[UPRIGHT RECOVERED BASE-RELATIVE CARTESIAN]"
        )

        print(
            "[UPRIGHT NOTE] Joint configuration differs from the "
            "recorded upright q, but the tool returned to the same "
            "safe region relative to UR5e base_link."
        )

        return True

    print()
    print(
        "[FAILED] SAFE UPRIGHT was not recovered in either "
        "joint space or UR5e-base-relative task space."
    )

    return False


def end_effector_target_for_camera_pose(
    stage,
    camera_position,
    camera_orientation,
):
    """
    Tool pose that puts the wrist camera at the requested pose.

    The camera is bolted to the gripper, so the tool->camera
    transform is rigid. It is measured from USD instead of taken
    from the mount constants, which keeps this correct whether the
    camera came from harvest_bot.usd or from the fallback mount.
    """
    (
        camera_now_position,
        camera_now_orientation,
    ) = pose_position_orientation(
        stage,
        WRIST_CAMERA_PATH,
    )

    (
        tip_now_position,
        tip_now_orientation,
    ) = pose_position_orientation(
        stage,
        GRASP_TIP_PATH,
    )

    # Tool pose expressed in the camera frame.
    camera_to_tip_rotation = quaternion_multiply(
        quaternion_conjugate(
            camera_now_orientation
        ),
        tip_now_orientation,
    )

    camera_to_tip_offset = (
        quaternion_to_rotation_matrix(
            camera_now_orientation
        ).T
        @ (
            tip_now_position
            - camera_now_position
        )
    )

    target_rotation = quaternion_to_rotation_matrix(
        camera_orientation
    )

    return (
        np.asarray(
            camera_position,
            dtype=np.float64,
        )
        + target_rotation
        @ camera_to_tip_offset,
        quaternion_multiply(
            camera_orientation,
            camera_to_tip_rotation,
        ),
    )


async def _run_rmpflow_pose_stage(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    target_position,
    target_orientation,
    label,
):
    """
    Drive the tool to a full pose target and report how close it got.

    Returns:
        (reached, best_error, articulation_policy)
    """
    if ensure_articulations_initialized(
        jackal,
        arm,
    ):
        articulation_policy = (
            rebuild_articulation_motion_policy(
                arm,
                rmpflow,
            )
        )

    rmpflow.set_end_effector_target(
        np.asarray(
            target_position,
            dtype=np.float64,
        ),
        (
            None
            if target_orientation is None
            else np.asarray(
                target_orientation,
                dtype=np.float64,
            )
        ),
    )

    best_error = float("inf")
    steps_since_improve = 0
    reinit_count = 0

    for step in range(
        VIEW_MAX_STEPS
    ):
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        update_rmpflow_base_pose(
            stage,
            rmpflow,
        )

        try:
            action = (
                articulation_policy
                .get_next_articulation_action(
                    RMPFLOW_PHYSICS_DT
                )
            )
        except Exception as exc:
            print(
                f"[RMPFLOW] get_next_articulation_action failed: "
                f"{exc}"
            )

            (
                action,
                articulation_policy,
                reinit_count,
                give_up,
            ) = recover_rmpflow_action(
                jackal=jackal,
                arm=arm,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                reinit_count=reinit_count,
                max_reinits=1,
            )

            if give_up or action is None:
                break

            update_rmpflow_base_pose(
                stage,
                rmpflow,
            )

            rmpflow.set_end_effector_target(
                np.asarray(
                    target_position,
                    dtype=np.float64,
                ),
                (
                    None
                    if target_orientation is None
                    else np.asarray(
                        target_orientation,
                        dtype=np.float64,
                    )
                ),
            )

            try:
                action = (
                    articulation_policy
                    .get_next_articulation_action(
                        RMPFLOW_PHYSICS_DT
                    )
                )
            except Exception:
                break

        arm.apply_action(
            action
        )

        await next_frame()

        error = float(
            np.linalg.norm(
                pose_position(
                    stage,
                    GRASP_TIP_PATH,
                )
                - target_position
            )
        )

        if error + 1e-4 < best_error:
            best_error = error
            steps_since_improve = 0
        else:
            steps_since_improve += 1

        if step % VIEW_PRINT_EVERY == 0:
            print(
                f"[VIEW AIM] {label} | step={step:4d} | "
                f"tip_error={error:.4f} m | "
                f"best={best_error:.4f} m"
            )

        if error <= VIEW_TOLERANCE:
            for _ in range(
                VIEW_SETTLE_FRAMES
            ):
                apply_base_velocity_raw(
                    jackal,
                    0.0,
                    0.0,
                )

                update_rmpflow_base_pose(
                    stage,
                    rmpflow,
                )

                arm.apply_action(
                    articulation_policy
                    .get_next_articulation_action(
                        RMPFLOW_PHYSICS_DT
                    )
                )

                await next_frame()

            return (
                True,
                best_error,
                articulation_policy,
            )

        if (
            steps_since_improve
            >= VIEW_NO_IMPROVE_STEPS
        ):
            # Stuck (often against beds/foliage): stop early instead
            # of burning the full step budget with re-inits.
            reached = (
                best_error
                <= VIEW_ACCEPT_TOLERANCE
            )
            print(
                f"[VIEW AIM] plateau after "
                f"{steps_since_improve} stagnant steps | "
                f"best={best_error:.4f} m | "
                f"accept={reached}"
            )
            return (
                reached,
                best_error,
                articulation_policy,
            )

    return (
        False,
        best_error,
        articulation_policy,
    )


async def move_arm_to_aimed_viewpoint(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    camera_position,
    camera_orientation,
    look_target,
    label,
):
    """
    Put the wrist camera at a pose that frames the truss.

    The tool is commanded, but what matters is measured: after each
    attempt the camera's real pose is read back from USD and the
    commanded camera pose is corrected by the world-frame error.
    That closes the loop over the whole chain — RMPflow tracking
    error, the tool frame convention, the camera mount — without
    assuming any of them are exact.

    Returns:
        (ok, articulation_policy, framing)
    """
    print()
    print("========================================")
    print(f"[VIEWPOINT] {label}")
    print("========================================")

    print(
        f"Camera target: {np.round(camera_position, 4)}"
    )

    print(
        f"Look target:   {np.round(look_target, 4)}"
    )

    commanded_position = np.asarray(
        camera_position,
        dtype=np.float64,
    ).copy()

    commanded_orientation = np.asarray(
        camera_orientation,
        dtype=np.float64,
    ).copy()

    framing = None

    for attempt in range(
        1,
        VIEW_AIM_MAX_ATTEMPTS + 1,
    ):
        (
            tool_position,
            tool_orientation,
        ) = end_effector_target_for_camera_pose(
            stage,
            commanded_position,
            commanded_orientation,
        )

        (
            reached,
            best_error,
            articulation_policy,
        ) = await _run_rmpflow_pose_stage(
            stage=stage,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            rmpflow=rmpflow,
            articulation_policy=articulation_policy,
            target_position=tool_position,
            target_orientation=tool_orientation,
            label=f"{label} | attempt {attempt}",
        )

        (
            achieved_position,
            achieved_orientation,
        ) = pose_position_orientation(
            stage,
            WRIST_CAMERA_PATH,
        )

        position_error = achieved_position - camera_position

        orientation_error = quaternion_multiply(
            camera_orientation,
            quaternion_conjugate(
                achieved_orientation
            ),
        )

        framing = wrist_camera_framing(
            stage,
            look_target,
        )

        print(
            f"[VIEW AIM] attempt {attempt}/"
            f"{VIEW_AIM_MAX_ATTEMPTS} | "
            f"tool_reached={reached} "
            f"(best={best_error:.4f} m) | "
            f"camera_offset="
            f"{np.linalg.norm(position_error):.4f} m | "
            f"camera_angle="
            f"{math.degrees(quaternion_angle(orientation_error)):+.2f} deg"
        )

        print(
            f"[VIEW AIM] truss framing: "
            f"pixel="
            f"{np.round(framing['pixel'], 1) if framing['pixel'] is not None else 'behind camera'} | "
            f"center_fraction={framing['center_fraction']:.2f} | "
            f"depth={framing['depth']:+.3f} m"
        )

        if (
            framing["center_fraction"]
            <= VIEW_AIM_CENTER_FRACTION
        ):
            print(
                f"[VIEWPOINT FRAMED] truss within the central "
                f"{VIEW_AIM_CENTER_FRACTION:.0%} of the image."
            )
            return (
                True,
                articulation_policy,
                framing,
            )

        if attempt == VIEW_AIM_MAX_ATTEMPTS:
            break

        # Correct the commanded camera pose by the world-frame error
        # that was actually measured, then command it again.
        commanded_position = (
            commanded_position
            - position_error
        )

        commanded_orientation = quaternion_multiply(
            orientation_error,
            commanded_orientation,
        )

        print(
            "[VIEW AIM] correcting commanded camera pose by "
            f"{np.round(-position_error, 4)} m"
        )

    if (
        framing is not None
        and framing["in_view"]
    ):
        print(
            "[VIEWPOINT ACCEPT] truss is in frame but off center "
            f"(center_fraction={framing['center_fraction']:.2f})."
        )
        return (
            True,
            articulation_policy,
            framing,
        )

    if not VIEW_AIM_FALLBACK_TO_POSITION_ONLY:
        print(
            "[FAILED] Could not aim the wrist camera at the truss."
        )
        return (
            False,
            articulation_policy,
            framing,
        )

    # Aiming failed. Fall back to the old position-only viewpoint so
    # the truss still gets an image and the route behaves as before:
    # tool tip at the original VIEW_OFFSET standoff, orientation free.
    standoff_direction = (
        np.asarray(
            camera_position,
            dtype=np.float64,
        )
        - np.asarray(
            look_target,
            dtype=np.float64,
        )
    )

    standoff_direction = standoff_direction / max(
        float(
            np.linalg.norm(
                standoff_direction
            )
        ),
        1e-9,
    )

    fallback_target = (
        np.asarray(
            look_target,
            dtype=np.float64,
        )
        + standoff_direction
        * VIEW_OFFSET
    )

    print(
        "[VIEW AIM] Aiming failed; falling back to the "
        "position-only viewpoint at "
        f"{VIEW_OFFSET:.3f} m standoff."
    )

    (
        fallback_ok,
        articulation_policy,
    ) = await move_arm_to_viewpoint(
        stage=stage,
        jackal=jackal,
        arm=arm,
        arm_indices=arm_indices,
        rmpflow=rmpflow,
        articulation_policy=articulation_policy,
        target_position=fallback_target,
        label=f"{label} | position-only fallback",
    )

    framing = wrist_camera_framing(
        stage,
        look_target,
    )

    return (
        fallback_ok,
        articulation_policy,
        framing,
    )


async def move_arm_to_viewpoint(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    target_position,
    label,
):
    """
    RMPflow Cartesian move of GraspTip to a privileged viewing pose.
    No grasp / cut / finger actuation.

    Returns:
        (ok, articulation_policy)
    """
    print()
    print("========================================")
    print(f"[VIEWPOINT] {label}")
    print("========================================")

    print(
        f"GraspTip target: "
        f"{np.round(target_position, 4)}"
    )

    if ensure_articulations_initialized(
        jackal,
        arm,
    ):
        articulation_policy = (
            rebuild_articulation_motion_policy(
                arm,
                rmpflow,
            )
        )

        print(
            "[RMPFLOW] Rebuilt ArticulationMotionPolicy "
            "after articulation re-init."
        )

    rmpflow.set_end_effector_target(
        np.asarray(
            target_position,
            dtype=np.float64,
        ),
        None,
    )

    initial_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    print(
        f"Initial GraspTip: "
        f"{np.round(initial_tip, 4)}"
    )

    print(
        "Initial error: "
        f"{np.linalg.norm(target_position - initial_tip):.4f} m"
    )

    best_error = float("inf")
    best_tip = np.asarray(
        initial_tip,
        dtype=np.float64,
    ).copy()
    steps_since_improve = 0
    reinit_count = 0

    for step in range(
        VIEW_MAX_STEPS
    ):
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )

        update_rmpflow_base_pose(
            stage,
            rmpflow,
        )

        try:
            action = (
                articulation_policy
                .get_next_articulation_action(
                    RMPFLOW_PHYSICS_DT
                )
            )
        except Exception as exc:
            print(
                f"[RMPFLOW] get_next_articulation_action failed: "
                f"{exc}"
            )

            (
                action,
                articulation_policy,
                reinit_count,
                give_up,
            ) = recover_rmpflow_action(
                jackal=jackal,
                arm=arm,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                reinit_count=reinit_count,
                max_reinits=1,
            )

            if give_up or action is None:
                break

            update_rmpflow_base_pose(
                stage,
                rmpflow,
            )

            rmpflow.set_end_effector_target(
                np.asarray(
                    target_position,
                    dtype=np.float64,
                ),
                None,
            )

            try:
                action = (
                    articulation_policy
                    .get_next_articulation_action(
                        RMPFLOW_PHYSICS_DT
                    )
                )
            except Exception:
                break

        arm.apply_action(
            action
        )

        await next_frame()

        tip_position = pose_position(
            stage,
            GRASP_TIP_PATH,
        )

        error = float(
            np.linalg.norm(
                tip_position
                - target_position
            )
        )

        if error + 1e-4 < best_error:
            best_error = error
            best_tip = np.asarray(
                tip_position,
                dtype=np.float64,
            ).copy()
            steps_since_improve = 0
        else:
            steps_since_improve += 1

        if step % VIEW_PRINT_EVERY == 0:
            print(
                f"[VIEW] step={step:4d} | "
                f"tip={np.round(tip_position, 4)} | "
                f"error={error:.4f} m | "
                f"best={best_error:.4f} m"
            )

        if error <= VIEW_TOLERANCE:
            print(
                f"[VIEWPOINT REACHED] error={error:.4f} m"
            )

            for _ in range(
                VIEW_SETTLE_FRAMES
            ):
                apply_base_velocity_raw(
                    jackal,
                    0.0,
                    0.0,
                )

                update_rmpflow_base_pose(
                    stage,
                    rmpflow,
                )

                action = (
                    articulation_policy
                    .get_next_articulation_action(
                        RMPFLOW_PHYSICS_DT
                    )
                )

                arm.apply_action(
                    action
                )

                await next_frame()

            return True, articulation_policy

        # Plateau: accept if close enough, otherwise abort early when
        # stuck against beds instead of re-init thrashing for minutes.
        if (
            steps_since_improve
            >= VIEW_NO_IMPROVE_STEPS
        ):
            if best_error <= VIEW_ACCEPT_TOLERANCE:
                print(
                    f"[VIEWPOINT ACCEPT] plateau best_error="
                    f"{best_error:.4f} m | "
                    f"best_tip={np.round(best_tip, 4)} | "
                    f"target={np.round(target_position, 4)}"
                )
                return True, articulation_policy

            print(
                f"[VIEWPOINT STUCK] no progress for "
                f"{steps_since_improve} steps | "
                f"best_error={best_error:.4f} m | aborting stage"
            )
            break

    final_tip = pose_position(
        stage,
        GRASP_TIP_PATH,
    )

    final_error = float(
        np.linalg.norm(
            final_tip
            - target_position
        )
    )

    if best_error <= VIEW_ACCEPT_TOLERANCE:
        print(
            f"[VIEWPOINT ACCEPT] timeout best_error="
            f"{best_error:.4f} m | "
            f"final_error={final_error:.4f} m | "
            f"best_tip={np.round(best_tip, 4)}"
        )
        return True, articulation_policy

    print(
        f"[FAILED] Viewpoint timed out. "
        f"final_error={final_error:.4f} m | "
        f"best_error={best_error:.4f} m"
    )

    return False, articulation_policy
