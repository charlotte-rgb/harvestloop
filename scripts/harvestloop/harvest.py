"""Per-truss harvest cycle and per-stop orchestration."""

import time

import numpy as np

from .config import *  # noqa: F401,F403
from .simctl import (
    ensure_articulations_initialized,
    next_frame,
    rebuild_articulation_motion_policy,
)
from .usd_pose import (
    get_base_state,
    grasp_tip_position_in_arm_base,
    pose_position,
    world_point_to_frame,
)
from .arm_control import (
    get_active_arm_positions,
    hold_arm_pose,
    move_arm_to_fixed_home,
    update_rmpflow_base_pose,
)
from .base_control import (
    apply_base_velocity_raw,
    follow_waypoint,
    smooth_stop,
    turn_to_yaw,
)
from .basket import set_basket_physx_collision_enabled
from .payload import (
    activate_cut_payload,
    make_logical_grasp,
    physical_release_payload,
)
from .metrics import (
    STAGE_BASKET,
    STAGE_CUT,
    STAGE_GRASP,
    STAGE_NAV,
    STAGE_PERCEPTION,
    STAGE_RETREAT,
    STAGE_SETUP,
    STAGE_UPRIGHT,
)
from .perception import capture_wrist_rgbd
from .perf import measure
from .estimator import localization_errors
from .keypoint_estimate import (
    render_keypoint_overlay,
    show_keypoint_overlay,
)
from .routing import (
    compute_named_camera_viewpoint_pose,
    compute_stem_camera_column_xy,
)
from .manipulation import (
    align_cutter_and_cut,
    compute_pregrasp,
    move_arm_to_aimed_viewpoint,
    move_arm_to_basket_transport_with_rmpflow,
    move_arm_to_grasp_point,
    move_arm_to_pregrasp,
    replay_pregrasp_path_to_upright,
    rmpflow_retreat_to_pregrasp_after_cut,
)


def _tip_error(
    stage,
    tip_path,
    target_position,
):
    """Distance from a tool tip to the position it was sent to."""
    return float(
        np.linalg.norm(
            pose_position(
                stage,
                tip_path,
            )
            - np.asarray(
                target_position,
                dtype=np.float64,
            )
        )
    )


def _label_record_fields(
    labels,
):
    """Flatten wrist-camera labels into per-truss record fields."""
    fields = {}

    for name, label in labels.items():
        fields[f"{name}_world"] = label["world"]
        fields[f"{name}_cam"] = label["camera"]
        fields[f"{name}_px"] = label["pixel"]
        fields[f"{name}_depth_m"] = label["depth"]
        fields[f"{name}_in_view"] = label["in_view"]

    return fields


def _keypoint_record_fields(
    estimate,
    gt_k1_world,
    gt_k2_world,
):
    """
    Per-truss keypoint columns: detection, confidence, 3D error.

    Estimators that do not produce keypoints still emit the columns,
    empty, so runs from different perception modes stay comparable.
    """
    keypoints = (
        estimate.get(
            "keypoints"
        )
        or {}
    )

    fields = {
        "gt_k1_world": gt_k1_world,
        "gt_k2_world": gt_k2_world,
        "keypoint_needs_another_view": bool(
            estimate.get(
                "needs_another_view",
                False,
            )
        ),
        "keypoint_reason": estimate.get(
            "reason",
            "",
        ),
        "peduncle_length_m": (
            estimate.get(
                "diagnostics",
                {},
            )
            or {}
        ).get(
            "peduncle_length_m"
        ),
    }

    for name, gt_point in (
        (
            "k1",
            gt_k1_world,
        ),
        (
            "k2",
            gt_k2_world,
        ),
    ):
        keypoint = keypoints.get(
            name
        )

        fields[f"est_{name}_world"] = (
            keypoint["world"]
            if keypoint
            else None
        )

        fields[f"{name}_px"] = (
            keypoint["pixel"]
            if keypoint
            else None
        )

        fields[f"{name}_confidence"] = (
            float(
                keypoint["confidence"]
            )
            if keypoint
            else None
        )

        fields[f"{name}_depth_valid_fraction"] = (
            float(
                keypoint["depth_valid_fraction"]
            )
            if keypoint
            else None
        )

        fields[f"{name}_error_m"] = (
            float(
                np.linalg.norm(
                    np.asarray(
                        keypoint["world"],
                        dtype=np.float64,
                    )
                    - np.asarray(
                        gt_point,
                        dtype=np.float64,
                    )
                )
            )
            if (
                keypoint
                and gt_point is not None
            )
            else None
        )

    return fields


def _estimate_is_valid(estimate):
    return (
        estimate is not None
        and estimate.get(
            "grasp_point_world"
        ) is not None
        and estimate.get(
            "cut_point_world"
        ) is not None
    )


def _estimate_selection_key(estimate):
    """
    Rank estimates for active perception: valid first, then ones that
    do not ask for another view, then higher confidence.
    """
    if estimate is None:
        return (
            0,
            0,
            -1.0,
        )

    return (
        1 if _estimate_is_valid(
            estimate
        ) else 0,
        0 if estimate.get(
            "needs_another_view",
            True,
        ) else 1,
        float(
            estimate.get(
                "confidence",
                0.0,
            )
        ),
    )


def _shared_view_protocol():
    """
    Fixed and active share one trial: canonical capture once, then
    optional lateral views in the same acquisition.
    """
    return RUN_CONDITION_LABEL in (
        "shared_view",
        "active_perception",
    )


def _viewpoint_sequence_for_condition():
    if RUN_CONDITION_LABEL == "fixed_view":
        # Legacy one-capture policy (optional full e2e later).
        return [
            ACTIVE_VIEWPOINT_SEQUENCE[0]
        ]

    return list(
        ACTIVE_VIEWPOINT_SEQUENCE[
            :ACTIVE_PERCEPTION_MAX_VIEWS
        ]
    )


def _canonical_view_is_bad(record):
    """Failed / bad canonical view for the rescue-rate denominator."""
    if record is None:
        return True

    if not record.get(
        "ok",
        False,
    ):
        return True

    estimate = record.get(
        "estimate"
    )

    if estimate is None:
        return True

    if not _estimate_is_valid(
        estimate
    ):
        return True

    if bool(
        estimate.get(
            "needs_another_view",
            False,
        )
    ):
        return True

    return False


def _rescue_outcome(
    canonical_record,
    selected_record,
):
    """
    Classify the shared-view trial.

    Returns one of: no_rescue_needed, rescued, not_rescued.
    """
    canonical_bad = _canonical_view_is_bad(
        canonical_record
    )

    if not canonical_bad:
        return "no_rescue_needed"

    if (
        selected_record is not None
        and selected_record.get(
            "ok",
            False,
        )
        and _estimate_is_valid(
            selected_record.get(
                "estimate"
            )
        )
        and not bool(
            (
                selected_record.get(
                    "estimate"
                )
                or {}
            ).get(
                "needs_another_view",
                True,
            )
        )
    ):
        return "rescued"

    return "not_rescued"


async def recover_upright_after_harvest_skip(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    truss_name,
    reason,
):
    """
    After a per-truss harvest failure, leave the plant and continue.

    SAFE UPRIGHT is best-effort. Incomplete recovery must NOT abort the
    whole aisle route.

    Returns:
        (ok, articulation_policy)  # ok is True unless articulations die
    """
    print(
        f"[HARVEST SKIP] {truss_name}: {reason} "
        "Recovering SAFE UPRIGHT and continuing route."
    )

    set_basket_physx_collision_enabled(
        stage,
        False,
    )

    upright_ok = await move_arm_to_fixed_home(
        arm=arm,
        arm_indices=arm_indices,
        q_home=UR5E_UPRIGHT_Q,
        label=(
            f"SAFE UPRIGHT after skip {truss_name}"
        ),
    )

    if not upright_ok:
        print(
            "[HARVEST SKIP] SAFE UPRIGHT incomplete after "
            f"{truss_name}; forcing upright hold and moving on."
        )

    # Always command upright briefly so the next truss starts from a
    # known joint hold, even if the residual exceeded tolerance.
    for _ in range(6):
        hold_arm_pose(
            arm,
            arm_indices,
            UR5E_UPRIGHT_Q,
        )
        apply_base_velocity_raw(
            jackal,
            0.0,
            0.0,
        )
        await next_frame()

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

    return True, articulation_policy


async def harvest_one_truss(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    side_name,
    truss,
    grasp_position,
    cut_position,
    cutter_tip_path,
    truss_carry_ops,
    basket_drop_reference_offset,
    q_hold,
    recorded_upright_tip,
    recorded_upright_tip_in_arm_base,
    harvest_log,
    timing=None,
):
    """
    Baseline grasp -> cut -> carry -> basket release for one truss.

    `grasp_position` and `cut_position` come from the perception
    seam (`estimate_target`). This function must not read USD
    GraspPoint / CutPoint for control.

    Assumes arm starts at SAFE UPRIGHT with upright tips already recorded.
    Ends at the accepted basket transport posture (q_hold updated),
    or back at SAFE UPRIGHT if this truss is skipped.

    Every exit records a stage-tagged outcome in `harvest_log`, so a
    post-CUT controller failure is not counted as a perception failure.

    Returns:
        (ok, articulation_policy, q_hold)

        ok=False only for unrecoverable failures that should abort
        the whole route. Exhausted PREGRASP / GraspPoint / CUT retries
        skip the truss and return ok=True so the aisle route continues.
    """
    if cut_position is None:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_PERCEPTION,
            "Estimator returned no CutPoint; nothing to aim at.",
        )

        print(
            f"[HARVEST SKIP] {truss['truss_name']}: "
            "missing estimated CutPoint"
        )
        return True, articulation_policy, q_hold

    if grasp_position is None:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_PERCEPTION,
            "Estimator returned no GraspPoint.",
        )

        print(
            f"[HARVEST SKIP] {truss['truss_name']}: "
            "missing estimated GraspPoint"
        )
        return True, articulation_policy, q_hold

    grasp_position = np.asarray(
        grasp_position,
        dtype=np.float64,
    ).reshape(3)

    cut_position = np.asarray(
        cut_position,
        dtype=np.float64,
    ).reshape(3)

    pregrasp_position = compute_pregrasp(
        grasp_position,
        side_name=side_name,
    )

    print()
    print("========================================")
    print(
        f"[HARVEST] {side_name} | {truss['truss_name']}"
    )
    print("========================================")

    print(
        f"Estimated GraspPoint: {np.round(grasp_position, 4)}"
    )

    print(
        f"Estimated CutPoint:   {np.round(cut_position, 4)}"
    )

    print(
        f"PREGRASP:   {np.round(pregrasp_position, 4)} "
        f"(side={side_name}, offset={PREGRASP_OFFSET:.3f} m)"
    )

    harvest_log.note(
        grasp_target=grasp_position,
        cut_target=cut_position,
        pregrasp_target=pregrasp_position,
    )

    # Where the base and the shoulder actually stand for this truss.
    # The route is nominally symmetric, so the left row failing far
    # more often than the right has to show up in one of these.
    base_state = get_base_state(
        stage
    )

    grasp_in_arm_base = world_point_to_frame(
        stage,
        ARM_BASE_PATH,
        grasp_position,
    )

    pregrasp_in_arm_base = world_point_to_frame(
        stage,
        ARM_BASE_PATH,
        pregrasp_position,
    )

    harvest_log.note(
        base_x=float(base_state["x"]),
        base_y=float(base_state["y"]),
        base_yaw=float(base_state["yaw"]),
        arm_base_world=pose_position(
            stage,
            ARM_BASE_PATH,
        ),
        grasp_in_arm_base=grasp_in_arm_base,
        pregrasp_in_arm_base=pregrasp_in_arm_base,
        # Straight-line reach the shoulder is being asked for. The
        # UR5e is scaled to 0.6, so its envelope is ~0.51 m.
        grasp_reach_m=float(
            np.linalg.norm(
                grasp_in_arm_base
            )
        ),
        pregrasp_reach_m=float(
            np.linalg.norm(
                pregrasp_in_arm_base
            )
        ),
    )

    print(
        f"Reach from arm base: "
        f"grasp={np.linalg.norm(grasp_in_arm_base):.4f} m | "
        f"pregrasp={np.linalg.norm(pregrasp_in_arm_base):.4f} m | "
        f"grasp_in_arm_base={np.round(grasp_in_arm_base, 4)}"
    )

    # Keep basket soft while the arm approaches the drop region.
    set_basket_physx_collision_enabled(
        stage,
        False,
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

    update_rmpflow_base_pose(
        stage,
        rmpflow,
    )

    with measure(timing, "PREGRASP"):
        (
            pregrasp_ok,
            pregrasp_joint_trajectory,
        ) = await move_arm_to_pregrasp(
            stage=stage,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            rmpflow=rmpflow,
            articulation_policy=articulation_policy,
            target_position=pregrasp_position,
        )

    harvest_log.note(
        pregrasp_error_m=_tip_error(
            stage,
            GRASP_TIP_PATH,
            pregrasp_position,
        ),
        q_at_pregrasp=get_active_arm_positions(
            arm,
            arm_indices,
        ),
        tip_in_arm_base_at_pregrasp=(
            grasp_tip_position_in_arm_base(
                stage
            )
        ),
    )

    if not pregrasp_ok:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_GRASP,
            "PREGRASP pose not reached.",
        )

        recover_ok, articulation_policy = (
            await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason="PREGRASP failed.",
            )
        )
        return recover_ok, articulation_policy, q_hold

    with measure(timing, "GRASP"):
        (
            grasp_ok,
            grasp_joint_trajectory,
        ) = await move_arm_to_grasp_point(
            stage=stage,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            rmpflow=rmpflow,
            articulation_policy=articulation_policy,
            target_position=grasp_position,
            pregrasp_position=pregrasp_position,
        )

    harvest_log.note(
        grasp_error_m=_tip_error(
            stage,
            GRASP_TIP_PATH,
            grasp_position,
        ),
        q_at_grasp=get_active_arm_positions(
            arm,
            arm_indices,
        ),
        # Where the tip actually stalled, in shoulder coordinates.
        # Compared against grasp_in_arm_base this says whether the
        # arm ran out of reach or stopped short in some direction.
        tip_in_arm_base_at_grasp=(
            grasp_tip_position_in_arm_base(
                stage
            )
        ),
    )

    if not grasp_ok:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_GRASP,
            "GraspTip did not reach GraspPoint within "
            f"{GRASP_TOLERANCE:.3f} m after retries.",
        )

        recover_ok, articulation_policy = (
            await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason=(
                    "GraspPoint failed after retries "
                    "(hard 3 cm not met)."
                ),
            )
        )
        return recover_ok, articulation_policy, q_hold

    # Silence unused trajectory binding (retries use it internally).
    _ = grasp_joint_trajectory

    truss_path = make_logical_grasp(
        stage=stage,
        grasp_path=truss["grasp_path"],
        cut_path=truss["cut_path"],
    )

    with measure(timing, "CUT"):
        (
            cut_align_ok,
            cut_joint_trajectory,
        ) = await align_cutter_and_cut(
            stage=stage,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            rmpflow=rmpflow,
            articulation_policy=articulation_policy,
            grasp_position=grasp_position,
            cut_position=cut_position,
            cutter_tip_path=cutter_tip_path,
            pregrasp_position=pregrasp_position,
        )

    harvest_log.note(
        cut_grasp_error_m=_tip_error(
            stage,
            GRASP_TIP_PATH,
            grasp_position,
        ),
        cut_cutter_error_m=_tip_error(
            stage,
            cutter_tip_path,
            cut_position,
        ),
    )

    if not cut_align_ok:
        # Logical grasp was armed but CUT never declared; plant stays
        # fixed. Recover upright so the next truss / stop can proceed.
        _ = cut_joint_trajectory

        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_CUT,
            "CUT criterion not met after retries "
            f"(GraspTip <= {GRASP_TOLERANCE:.3f} m, "
            f"CutterTip <= {CUTTER_TOLERANCE:.3f} m).",
        )

        recover_ok, articulation_policy = (
            await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason=(
                    "CUT failed after retries "
                    "(hard 3 cm / 4 cm not met)."
                ),
            )
        )
        return recover_ok, articulation_policy, q_hold

    try:
        payload = activate_cut_payload(
            stage=stage,
            truss_path=truss_path,
            carry_ops=truss_carry_ops,
        )
    except Exception as exc:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_RETREAT,
            f"Payload attach after CUT failed: {exc}",
        )

        print(
            "[STOPPING TEST] Payload attach failed: "
            f"{exc}"
        )
        return False, articulation_policy, q_hold

    # Skip the old CUT->GRASP joint replay holdup. After CUT, retreat
    # directly toward PREGRASP with the payload attached.
    _ = cut_joint_trajectory

    with measure(timing, "RETREAT"):
        retreat_ok = (
            await rmpflow_retreat_to_pregrasp_after_cut(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                pregrasp_position=pregrasp_position,
                payload=payload,
            )
        )

    harvest_log.note(
        retreat_error_m=_tip_error(
            stage,
            GRASP_TIP_PATH,
            pregrasp_position,
        ),
    )

    if not retreat_ok:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_RETREAT,
            "Post-CUT retreat did not reach the PREGRASP "
            f"region within {POSTCUT_RETREAT_TOLERANCE:.3f} m "
            f"in {POSTCUT_RETREAT_MAX_ATTEMPTS} attempts.",
        )

        if not POSTCUT_RETREAT_FAILURE_CONTINUES_ROUTE:
            print(
                "[STOPPING TEST] Post-cut PREGRASP retreat failed."
            )
            return False, articulation_policy, q_hold

        # The truss is already severed and the retreat controller
        # failed, so it is dropped where it is. Carrying on to the
        # basket would deliver it through a path that bypasses the
        # retreat stage and corrupt the end-to-end metric.
        await physical_release_payload(
            stage=stage,
            payload=payload,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            q_hold=get_active_arm_positions(
                arm,
                arm_indices,
            ),
        )

        recover_ok, articulation_policy = (
            await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason=(
                    "Post-CUT retreat failed; severed truss "
                    "released in place."
                ),
            )
        )
        return recover_ok, articulation_policy, q_hold

    with measure(timing, "UPRIGHT"):
        return_ok = (
            await replay_pregrasp_path_to_upright(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                joint_trajectory=pregrasp_joint_trajectory,
                recorded_upright_tip=recorded_upright_tip,
                recorded_upright_tip_in_arm_base=(
                    recorded_upright_tip_in_arm_base
                ),
                payload=payload,
            )
        )

    if not return_ok:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_UPRIGHT,
            "Return to SAFE UPRIGHT with payload failed.",
        )

        print(
            "[STOPPING TEST] Return to SAFE UPRIGHT with payload failed."
        )
        return False, articulation_policy, q_hold

    with measure(timing, "BASKET TRANSPORT"):
        (
            basket_ok,
            basket_release_q,
        ) = await move_arm_to_basket_transport_with_rmpflow(
            stage=stage,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            rmpflow=rmpflow,
            articulation_policy=articulation_policy,
            q_target=q_hold,
            payload=payload,
            reference_drop_offset=basket_drop_reference_offset,
        )

    if basket_release_q is None:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_BASKET,
            "Basket transition returned no release pose.",
        )

        print(
            "[STOPPING TEST] Basket transition returned no release pose."
        )
        return False, articulation_policy, q_hold

    q_hold = basket_release_q.copy()

    if not basket_ok:
        print(
            f"[HARVEST WARN] {truss['truss_name']}: "
            "Home joint tolerance not fully met; "
            "force-releasing at commanded UR5E_HOME_Q hold "
            "and continuing."
        )

    with measure(timing, "BASKET RELEASE"):
        release_ok = await physical_release_payload(
            stage=stage,
            payload=payload,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            q_hold=q_hold,
        )

    if not release_ok:
        harvest_log.record_failure(
            truss["truss_name"],
            STAGE_BASKET,
            "Physical basket release failed after drop approach.",
        )

        print(
            f"[HARVEST SKIP] {truss['truss_name']}: "
            "Physical basket release failed after drop approach. "
            "Recovering upright and continuing route."
        )
        recover_ok, articulation_policy = (
            await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason="Basket physical release failed.",
            )
        )
        return recover_ok, articulation_policy, q_hold

    # Soften the visible basket again so the next approach is not
    # fought by PhysX. The released truss keeps its support from the
    # invisible catcher, which is never toggled.
    set_basket_physx_collision_enabled(
        stage,
        False,
    )

    harvest_log.record_success(
        truss["truss_name"]
    )

    print(
        f"[HARVEST OK] Released {truss['truss_name']} into basket."
    )

    return True, articulation_policy, q_hold


async def recenter_jackal_on_aisle_stop(
    stage,
    commander,
    arm,
    arm_indices,
    q_hold,
    stop_x,
    label,
):
    """
    After arm manipulation, Jackal often yaws/drifts off the aisle.

    Re-face -X and return base_link to (stop_x, AISLE_Y).
    """
    print()
    print("========================================")
    print(f"[NAV] Recenter Jackal | {label}")
    print("========================================")

    before = get_base_state(
        stage
    )

    print(
        f"Before: xy=({before['x']:+.3f}, {before['y']:+.3f}) | "
        f"yaw={before['yaw']:+.3f} | "
        f"target=({stop_x:+.3f}, {AISLE_Y:+.3f})"
    )

    if not await turn_to_yaw(
        stage,
        commander,
        arm,
        arm_indices,
        q_hold,
        LEFT_YAW,
        f"{label} | face -X",
    ):
        return False

    current = get_base_state(
        stage
    )

    dx = float(
        stop_x - current["x"]
    )
    dy = float(
        AISLE_Y - current["y"]
    )

    need_reposition = (
        abs(dx) > 0.045
        or abs(dy) > 0.030
    )

    if need_reposition:
        # Facing -X: forward decreases X. If we need larger X, reverse.
        reverse = dx > 0.0

        print(
            f"[NAV] Reposition dx={dx:+.3f} dy={dy:+.3f} | "
            f"reverse={reverse}"
        )

        if not await follow_waypoint(
            stage=stage,
            commander=commander,
            arm=arm,
            arm_indices=arm_indices,
            q_hold=q_hold,
            target_x=float(stop_x),
            target_y=float(AISLE_Y),
            reverse=reverse,
            speed_limit=min(
                MAX_AISLE_SPEED,
                0.95,
            ),
            tolerance=FINAL_WAYPOINT_TOLERANCE,
            label=f"{label} | return stop plane",
            stop_at_target=True,
            path_axis="horizontal",
        ):
            return False
    else:
        print(
            "[NAV] Already near stop plane; yaw-only recenter."
        )

    if not await turn_to_yaw(
        stage,
        commander,
        arm,
        arm_indices,
        q_hold,
        LEFT_YAW,
        f"{label} | hold -X",
    ):
        return False

    await smooth_stop(
        commander,
        settle_frames=4,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
    )

    after = get_base_state(
        stage
    )

    print(
        f"[NAV] Recentered: "
        f"xy=({after['x']:+.3f}, {after['y']:+.3f}) | "
        f"yaw={after['yaw']:+.3f}"
    )

    return True


async def inspect_and_harvest_stem_trusses_at_stop(
    stage,
    jackal,
    arm,
    arm_indices,
    rmpflow,
    articulation_policy,
    capture_session,
    stop_index,
    side_name,
    stem,
    approach_offset,
    cutter_tip_path,
    truss_carry_ops,
    basket_drop_reference_offset,
    q_hold,
    commander,
    stop_x,
    harvest_log,
    estimator,
    run=None,
    timing=None,
):
    """
    Per truss on one parked stem (fixed-view baseline):
      1. vertical wrist viewpoint
      2. one RGB-D capture
      3. estimate_target(...) for grasp/cut
      4. SAFE UPRIGHT
      5. baseline grasp / cut / carry / basket release

    Viewpoint placement still uses the planned stem look target so
    every condition starts from the same canonical camera pose.
    Manipulation uses only the estimator output.

    Returns:
        (ok, articulation_policy, q_hold)
    """
    if stem is None:
        print(
            f"[STOP {stop_index:02d}] side={side_name} | "
            "no stem assigned; skipping."
        )
        return True, articulation_policy, q_hold

    print()
    print("========================================")
    print(
        f"[STOP {stop_index:02d}] VIEW+HARVEST "
        f"{side_name.upper()} {stem['stem_name']}"
    )
    print("========================================")

    print(
        f"Stem path: {stem['stem_path']}"
    )

    print(
        f"Stem root: "
        f"{np.round(stem['root_position'], 4)}"
    )

    print(
        f"Truss count: {len(stem['trusses'])}"
    )

    if not stem["trusses"]:
        print(
            "[WARN] Stem has no trusses."
        )
        return True, articulation_policy, q_hold

    stem_camera_xy = compute_stem_camera_column_xy(
        stem,
        approach_offset,
    )

    trusses_by_height = sorted(
        stem["trusses"],
        key=lambda item: float(
            item["look_target"][2]
        ),
    )

    print(
        f"Fixed stem camera XY: "
        f"({stem_camera_xy[0]:+.4f}, {stem_camera_xy[1]:+.4f}) | "
        f"standoff={VIEW_CAMERA_STANDOFF:.3f} m | "
        f"approach_offset={np.round(approach_offset, 4)} | "
        f"sweep={len(trusses_by_height)} trusses by increasing Z"
    )

    for truss_i, truss in enumerate(
        trusses_by_height
    ):
        image_stem = (
            f"stop_{stop_index:02d}_"
            f"{side_name}_"
            f"stem_{stem['stem_index']:02d}_"
            f"truss_{truss['truss_index']:02d}"
        )

        # Open this truss's record now, so a failure before the
        # viewpoint is still attributed to the right truss.
        harvest_log.begin_truss(
            truss["truss_name"],
            stop_index=stop_index,
            side=side_name,
            stem_name=stem["stem_name"],
            stem_index=stem["stem_index"],
            truss_index=truss["truss_index"],
            image_stem=image_stem,
            truss_path=truss["truss_path"],
            scene_seed=SCENE_SEED,
            occlusion_level=OCCLUSION_LEVEL,
        )

        # After a prior basket drop the arm is not upright; restore
        # SAFE UPRIGHT before the next vertical viewpoint.
        if truss_i > 0:
            upright_ok = await move_arm_to_fixed_home(
                arm=arm,
                arm_indices=arm_indices,
                q_home=UR5E_UPRIGHT_Q,
                label=(
                    f"SAFE UPRIGHT before "
                    f"{truss['truss_name']}"
                ),
            )

            if not upright_ok:
                harvest_log.record_failure(
                    truss["truss_name"],
                    STAGE_SETUP,
                    "SAFE UPRIGHT between trusses incomplete; "
                    "forcing hold and continuing.",
                )

                print(
                    "[HARVEST SKIP] SAFE UPRIGHT incomplete between "
                    f"trusses before {truss['truss_name']}; continuing."
                )

            for _ in range(8):
                hold_arm_pose(
                    arm,
                    arm_indices,
                    UR5E_UPRIGHT_Q,
                )
                apply_base_velocity_raw(
                    jackal,
                    0.0,
                    0.0,
                )
                await next_frame()

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

        if truss.get("cut_position") is None:
            harvest_log.record_failure(
                truss["truss_name"],
                STAGE_PERCEPTION,
                "Truss has no CutPoint marker; cannot score or bind.",
            )

            print(
                f"[HARVEST SKIP] {truss['truss_name']}: "
                "missing CutPoint marker"
            )
            continue

        viewpoint_names = _viewpoint_sequence_for_condition()

        print()
        print(
            f"[TARGET] stop={stop_index:02d} | "
            f"side={side_name} | "
            f"stem={stem['stem_name']} | "
            f"truss={truss['truss_name']} | "
            f"condition={RUN_CONDITION_LABEL}"
        )

        print(
            f"Look target (USD): "
            f"{np.round(truss['look_target'], 4)}"
        )

        # Hidden USD GT for scoring only. Not passed into the
        # manipulation controller.
        gt_grasp_world = pose_position(
            stage,
            truss["grasp_path"],
        )

        gt_cut_world = None

        if truss.get("cut_path") is not None:
            cut_prim = stage.GetPrimAtPath(
                truss["cut_path"]
            )

            if cut_prim.IsValid():
                gt_cut_world = pose_position(
                    stage,
                    truss["cut_path"],
                )

        # K1 = CutPoint, K2 = GraspPoint.
        gt_k1_world = gt_cut_world
        gt_k2_world = gt_grasp_world

        label_points = {
            "grasp": gt_grasp_world,
        }

        if gt_cut_world is not None:
            label_points["cut"] = gt_cut_world

        if hasattr(
            estimator,
            "bind_truss",
        ):
            estimator.bind_truss(
                truss["grasp_path"],
                truss["cut_path"],
            )

        view_records = []
        acquisition_t0 = time.perf_counter()
        canonical_done_t = None
        shared = _shared_view_protocol()

        for view_index, viewpoint_name in enumerate(
            viewpoint_names
        ):
            (
                camera_target,
                camera_target_orientation,
            ) = compute_named_camera_viewpoint_pose(
                stem_camera_xy,
                truss["look_target"],
                viewpoint_name,
            )

            view_stem = (
                f"{image_stem}_v{view_index + 1:02d}"
                f"_{viewpoint_name}"
            )

            print(
                f"[VIEW {view_index + 1}/{len(viewpoint_names)}] "
                f"{viewpoint_name} | "
                f"camera_target={np.round(camera_target, 4)}"
            )

            harvest_log.note(
                view_target=camera_target,
                look_target=truss["look_target"],
            )

            stage_name = (
                "VIEWPOINT"
                if view_index == 0
                else "ACTIVE_VIEWPOINT"
            )

            with measure(timing, stage_name):
                (
                    ok,
                    articulation_policy,
                    framing,
                ) = await move_arm_to_aimed_viewpoint(
                    stage=stage,
                    jackal=jackal,
                    arm=arm,
                    arm_indices=arm_indices,
                    rmpflow=rmpflow,
                    articulation_policy=articulation_policy,
                    camera_position=camera_target,
                    camera_orientation=camera_target_orientation,
                    look_target=truss["look_target"],
                    label=view_stem,
                )

            if view_index == 0:
                harvest_log.note(
                    framing_center_fraction=(
                        framing["center_fraction"]
                        if framing is not None
                        and np.isfinite(
                            framing["center_fraction"]
                        )
                        else None
                    ),
                    framing_in_view=(
                        False
                        if framing is None
                        else framing["in_view"]
                    ),
                )

            if not ok:
                # Canonical pose is required for the shared-view trial.
                if view_index == 0 and not shared:
                    harvest_log.record_failure(
                        truss["truss_name"],
                        STAGE_PERCEPTION,
                        f"Viewpoint not reached for {image_stem}.",
                    )

                    print(
                        f"[STOPPING TEST] Failed viewpoint for "
                        f"{image_stem}"
                    )
                    return False, articulation_policy, q_hold

                print(
                    f"[VIEW SKIP] {viewpoint_name}: "
                    "aimed viewpoint not reached; trying next."
                )
                view_records.append(
                    {
                        "viewpoint": viewpoint_name,
                        "view_index": view_index + 1,
                        "ok": False,
                        "estimate": None,
                        "capture_info": None,
                        "confidence": 0.0,
                        "needs_another_view": True,
                        "reason": "viewpoint_not_reached",
                    }
                )

                if view_index == 0:
                    canonical_done_t = time.perf_counter()

                if view_index == 0 and not shared:
                    break

                continue

            try:
                with measure(timing, "CAPTURE"):
                    capture_info = await capture_wrist_rgbd(
                        stage,
                        view_stem,
                        capture_session,
                        output_dir=(
                            run.images_dir
                            if run is not None
                            else None
                        ),
                        label_points=label_points,
                    )
            except Exception as exc:
                if view_index == 0 and not shared:
                    harvest_log.record_failure(
                        truss["truss_name"],
                        STAGE_PERCEPTION,
                        f"Wrist RGB-D capture failed: {exc}",
                    )

                    print(
                        f"[STOPPING TEST] RGB-D capture failed for "
                        f"{image_stem}: {exc}"
                    )
                    return False, articulation_policy, q_hold

                print(
                    f"[VIEW SKIP] {viewpoint_name}: "
                    f"capture failed ({exc})"
                )
                view_records.append(
                    {
                        "viewpoint": viewpoint_name,
                        "view_index": view_index + 1,
                        "ok": False,
                        "estimate": None,
                        "capture_info": None,
                        "confidence": 0.0,
                        "needs_another_view": True,
                        "reason": f"capture_failed: {exc}",
                    }
                )

                if view_index == 0:
                    canonical_done_t = time.perf_counter()

                continue

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

            try:
                with measure(timing, "ESTIMATE"):
                    estimate = estimator.estimate_target(
                        rgb=capture_info["rgb"],
                        depth=capture_info["depth"],
                        camera_pose=capture_info[
                            "camera_pose"
                        ],
                    )
            except Exception as exc:
                if view_index == 0 and not shared:
                    harvest_log.record_failure(
                        truss["truss_name"],
                        STAGE_PERCEPTION,
                        f"estimate_target failed: {exc}",
                    )

                    print(
                        f"[STOPPING TEST] estimate_target failed for "
                        f"{image_stem}: {exc}"
                    )
                    return False, articulation_policy, q_hold

                print(
                    f"[VIEW SKIP] {viewpoint_name}: "
                    f"estimate_target failed ({exc})"
                )
                view_records.append(
                    {
                        "viewpoint": viewpoint_name,
                        "view_index": view_index + 1,
                        "ok": False,
                        "estimate": None,
                        "capture_info": capture_info,
                        "confidence": 0.0,
                        "needs_another_view": True,
                        "reason": f"estimate_failed: {exc}",
                    }
                )

                if view_index == 0:
                    canonical_done_t = time.perf_counter()

                continue

            confidence = float(
                estimate.get(
                    "confidence",
                    0.0,
                )
            )

            needs_another = bool(
                estimate.get(
                    "needs_another_view",
                    False,
                )
            )

            print(
                f"[ESTIMATE] view={viewpoint_name} | "
                f"source={estimate.get('source', '?')} | "
                f"confidence={confidence:.3f} | "
                f"another_view={needs_another}"
                + (
                    f" | {estimate.get('reason')}"
                    if estimate.get(
                        "reason"
                    )
                    else ""
                )
            )

            if SHOW_KEYPOINT_OVERLAY:
                overlay_image = render_keypoint_overlay(
                    capture_info["rgb"],
                    estimate,
                    capture_info["camera_pose"],
                    label=(
                        f"{stem['stem_name']} "
                        f"{truss['truss_name']} "
                        f"{viewpoint_name}"
                    ),
                    gt_k1_world=gt_k1_world,
                    gt_k2_world=gt_k2_world,
                    gt_grasp_world=gt_grasp_world,
                    gt_cut_world=gt_cut_world,
                )

                if run is not None:
                    run.overlay_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    overlay_path = (
                        run.overlay_dir
                        / f"{view_stem}.png"
                    )

                    overlay_image.save(
                        overlay_path
                    )

                    print(
                        f"[OVERLAY] {overlay_path}"
                    )

                show_keypoint_overlay(
                    overlay_image,
                    caption=(
                        f"{view_stem}  "
                        f"conf={confidence:.2f}"
                    ),
                )

            view_records.append(
                {
                    "viewpoint": viewpoint_name,
                    "view_index": view_index + 1,
                    "ok": True,
                    "estimate": estimate,
                    "capture_info": capture_info,
                    "camera_target": camera_target,
                    "confidence": confidence,
                    "needs_another_view": needs_another,
                    "reason": estimate.get(
                        "reason",
                        "",
                    ),
                }
            )

            if view_index == 0:
                canonical_done_t = time.perf_counter()

            # Fixed-view legacy: stop after the shared canonical capture.
            if not shared:
                break

            # Shared-view: fixed result is view 1. Active stops here
            # too unless the canonical estimate asks for another view
            # or the canonical view failed / is invalid.
            canonical_bad = _canonical_view_is_bad(
                view_records[0]
            )

            if not canonical_bad:
                print(
                    "[SHARED VIEW] canonical OK — "
                    "no rescue needed; active stops."
                )
                break

            if view_index + 1 >= len(
                viewpoint_names
            ):
                break

            print(
                "[SHARED VIEW] canonical bad / needs_another_view — "
                "continuing active views in the same trial."
            )

        perception_camera_motion_s = float(
            time.perf_counter() - acquisition_t0
        )

        if canonical_done_t is None:
            canonical_camera_motion_s = perception_camera_motion_s
        else:
            canonical_camera_motion_s = float(
                canonical_done_t - acquisition_t0
            )

        extra_camera_motion_s = max(
            0.0,
            perception_camera_motion_s
            - canonical_camera_motion_s,
        )

        if not view_records:
            harvest_log.record_failure(
                truss["truss_name"],
                STAGE_PERCEPTION,
                "No viewpoint estimates were obtained.",
            )

            (
                ok,
                articulation_policy,
            ) = await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason="no viewpoint estimates",
            )

            if not ok:
                return False, articulation_policy, q_hold

            continue

        canonical = view_records[0]
        canonical_estimate = canonical.get(
            "estimate"
        )
        canonical_errors = localization_errors(
            canonical_estimate or {},
            gt_grasp_world,
            gt_cut_world,
        )
        canonical_bad = _canonical_view_is_bad(
            canonical
        )

        selected = max(
            view_records,
            key=lambda record: _estimate_selection_key(
                record.get(
                    "estimate"
                )
            ),
        )

        rescue_status = _rescue_outcome(
            canonical,
            selected,
        )

        estimate = selected.get(
            "estimate"
        )

        capture_info = selected.get(
            "capture_info"
        )

        camera_target = selected.get(
            "camera_target"
        )

        if estimate is None:
            harvest_log.record_failure(
                truss["truss_name"],
                STAGE_PERCEPTION,
                selected.get(
                    "reason"
                )
                or "No usable estimate across views.",
            )

            harvest_log.note(
                views_used=len(
                    view_records
                ),
                viewpoints_used=[
                    record["viewpoint"]
                    for record in view_records
                ],
                view_confidences=[
                    record["confidence"]
                    for record in view_records
                ],
                view_needs_another_view=[
                    record["needs_another_view"]
                    for record in view_records
                ],
                selected_viewpoint=selected[
                    "viewpoint"
                ],
                perception_camera_motion_s=(
                    perception_camera_motion_s
                ),
                canonical_camera_motion_s=(
                    canonical_camera_motion_s
                ),
                extra_camera_motion_s=(
                    extra_camera_motion_s
                ),
                canonical_needs_another_view=bool(
                    canonical.get(
                        "needs_another_view",
                        True,
                    )
                ),
                canonical_view_bad=canonical_bad,
                canonical_grasp_localization_error_m=(
                    canonical_errors[
                        "grasp_error_m"
                    ]
                ),
                canonical_cut_localization_error_m=(
                    canonical_errors[
                        "cut_error_m"
                    ]
                ),
                rescue_status=rescue_status,
                rescued=(
                    rescue_status == "rescued"
                ),
                no_rescue_needed=(
                    rescue_status
                    == "no_rescue_needed"
                ),
                shared_view_protocol=shared,
            )

            (
                ok,
                articulation_policy,
            ) = await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason=(
                    selected.get("reason")
                    or "no usable estimate"
                ),
            )

            if not ok:
                return False, articulation_policy, q_hold

            continue

        # Mark how many views the acquisition used (not the
        # estimator's internal fixed-view counter).
        estimate = dict(
            estimate
        )

        estimate["views_used"] = len(
            view_records
        )

        errors = localization_errors(
            estimate,
            gt_grasp_world,
            gt_cut_world,
        )

        print(
            f"[ESTIMATE SELECTED] "
            f"view={selected['viewpoint']} | "
            f"views_used={len(view_records)} | "
            f"rescue={rescue_status} | "
            f"confidence={float(estimate.get('confidence', 0.0)):.3f} | "
            f"grasp_err="
            f"{errors['grasp_error_m'] if errors['grasp_error_m'] is not None else float('nan'):.4f} m | "
            f"cut_err="
            f"{errors['cut_error_m'] if errors['cut_error_m'] is not None else float('nan'):.4f} m | "
            f"canonical_cut_err="
            f"{canonical_errors['cut_error_m'] if canonical_errors['cut_error_m'] is not None else float('nan'):.4f} m | "
            f"extra_motion={extra_camera_motion_s:.2f}s"
        )

        print(
            f"[LOG] harvest_stop={stop_index:02d} | "
            f"side={side_name} | "
            f"stem={stem['stem_name']} | "
            f"truss={truss['truss_name']} | "
            f"selected_view={selected['viewpoint']} | "
            f"camera_target={np.round(camera_target, 4)} | "
            f"wrist_cam_world="
            f"{np.round(capture_info['camera_position'], 4)} | "
            f"image={capture_info['image_path']}"
        )

        harvest_log.note(
            image=(
                run.relative_image(
                    capture_info["image_path"]
                )
                if run is not None
                else capture_info["image_path"]
            ),
            depth_image=(
                run.relative_image(
                    capture_info["depth_path"]
                )
                if (
                    run is not None
                    and capture_info.get(
                        "depth_path"
                    )
                )
                else capture_info.get(
                    "depth_path"
                )
            ),
            camera_position=capture_info[
                "camera_position"
            ],
            camera_quat=capture_info[
                "camera_orientation"
            ],
            gt_grasp_world=gt_grasp_world,
            gt_cut_world=gt_cut_world,
            est_grasp_world=estimate.get(
                "grasp_point_world"
            ),
            est_cut_world=estimate.get(
                "cut_point_world"
            ),
            estimate_confidence=float(
                estimate.get(
                    "confidence",
                    0.0,
                )
            ),
            estimate_source=estimate.get(
                "source"
            ),
            views_used=len(
                view_records
            ),
            viewpoints_used=[
                record["viewpoint"]
                for record in view_records
            ],
            view_confidences=[
                record["confidence"]
                for record in view_records
            ],
            view_needs_another_view=[
                record["needs_another_view"]
                for record in view_records
            ],
            selected_viewpoint=selected[
                "viewpoint"
            ],
            selected_view_index=int(
                selected["view_index"]
            ),
            perception_camera_motion_s=(
                perception_camera_motion_s
            ),
            canonical_camera_motion_s=(
                canonical_camera_motion_s
            ),
            extra_camera_motion_s=(
                extra_camera_motion_s
            ),
            canonical_viewpoint=canonical[
                "viewpoint"
            ],
            canonical_needs_another_view=bool(
                canonical.get(
                    "needs_another_view",
                    True,
                )
            ),
            canonical_view_bad=canonical_bad,
            canonical_confidence=float(
                canonical.get(
                    "confidence",
                    0.0,
                )
            ),
            canonical_est_grasp_world=(
                None
                if canonical_estimate is None
                else canonical_estimate.get(
                    "grasp_point_world"
                )
            ),
            canonical_est_cut_world=(
                None
                if canonical_estimate is None
                else canonical_estimate.get(
                    "cut_point_world"
                )
            ),
            canonical_grasp_localization_error_m=(
                canonical_errors[
                    "grasp_error_m"
                ]
            ),
            canonical_cut_localization_error_m=(
                canonical_errors[
                    "cut_error_m"
                ]
            ),
            grasp_localization_error_m=errors[
                "grasp_error_m"
            ],
            cut_localization_error_m=errors[
                "cut_error_m"
            ],
            rescue_status=rescue_status,
            rescued=(
                rescue_status == "rescued"
            ),
            no_rescue_needed=(
                rescue_status
                == "no_rescue_needed"
            ),
            shared_view_protocol=shared,
            estimate_diagnostics=estimate.get(
                "diagnostics"
            ),
            **_label_record_fields(
                capture_info["labels"]
            ),
            **_keypoint_record_fields(
                estimate,
                gt_k1_world,
                gt_k2_world,
            ),
        )

        if (
            estimate.get(
                "grasp_point_world"
            ) is None
            or estimate.get(
                "cut_point_world"
            ) is None
        ):
            harvest_log.record_failure(
                truss["truss_name"],
                STAGE_PERCEPTION,
                estimate.get(
                    "reason"
                )
                or "Estimator did not return grasp and cut points.",
            )

            (
                ok,
                articulation_policy,
            ) = await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason=(
                    estimate.get(
                        "reason"
                    )
                    or "perception failed"
                ),
            )

            if not ok:
                return False, articulation_policy, q_hold

            continue

        # Baseline harvest starts from SAFE UPRIGHT, not the view pose.
        upright_ok = await move_arm_to_fixed_home(
            arm=arm,
            arm_indices=arm_indices,
            q_home=UR5E_UPRIGHT_Q,
            label=(
                f"SAFE UPRIGHT before harvest "
                f"{truss['truss_name']}"
            ),
        )

        if not upright_ok:
            harvest_log.record_failure(
                truss["truss_name"],
                STAGE_SETUP,
                "SAFE UPRIGHT before harvest incomplete; "
                "forcing hold and skipping this truss.",
            )

            print(
                "[HARVEST SKIP] SAFE UPRIGHT incomplete before "
                f"harvest {truss['truss_name']}; continuing stop."
            )

            (
                ok,
                articulation_policy,
            ) = await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason="SAFE UPRIGHT before harvest incomplete",
            )

            if not ok:
                return False, articulation_policy, q_hold

            continue

        for _ in range(8):
            hold_arm_pose(
                arm,
                arm_indices,
                UR5E_UPRIGHT_Q,
            )
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )
            await next_frame()

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

        recorded_upright_tip = pose_position(
            stage,
            GRASP_TIP_PATH,
        ).copy()

        recorded_upright_tip_in_arm_base = (
            grasp_tip_position_in_arm_base(
                stage
            ).copy()
        )

        (
            harvest_ok,
            articulation_policy,
            q_hold,
        ) = await harvest_one_truss(
            stage=stage,
            jackal=jackal,
            arm=arm,
            arm_indices=arm_indices,
            rmpflow=rmpflow,
            articulation_policy=articulation_policy,
            side_name=side_name,
            truss=truss,
            grasp_position=estimate[
                "grasp_point_world"
            ],
            cut_position=estimate[
                "cut_point_world"
            ],
            cutter_tip_path=cutter_tip_path,
            truss_carry_ops=truss_carry_ops,
            basket_drop_reference_offset=(
                basket_drop_reference_offset
            ),
            q_hold=q_hold,
            recorded_upright_tip=recorded_upright_tip,
            recorded_upright_tip_in_arm_base=(
                recorded_upright_tip_in_arm_base
            ),
            harvest_log=harvest_log,
            timing=timing,
        )

        if not harvest_ok:
            (
                ok,
                articulation_policy,
            ) = await recover_upright_after_harvest_skip(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                truss_name=truss["truss_name"],
                reason="harvest incomplete; continuing remaining trusses",
            )

            if not ok:
                return False, articulation_policy, q_hold

            continue

        # Arm reaction during grasp/cut/release yaws the Jackal off
        # the aisle; snap back to the planned stop before the next
        # truss / drive segment.
        with measure(timing, "RECENTER"):
            recenter_ok = await recenter_jackal_on_aisle_stop(
                stage=stage,
                commander=commander,
                arm=arm,
                arm_indices=arm_indices,
                q_hold=q_hold,
                stop_x=stop_x,
                label=(
                    f"stop_{stop_index:02d} after "
                    f"{truss['truss_name']}"
                ),
            )

        if not recenter_ok:
            # Keep harvesting remaining trusses at this stop from the
            # measured base pose rather than dropping the rest of the
            # stem. Next stop still re-drives from wherever we are.
            remaining = (
                len(trusses_by_height) - truss_i - 1
            )

            harvest_log.record_event(
                STAGE_NAV,
                f"stop_{stop_index:02d}",
                (
                    "Jackal recenter incomplete after "
                    f"{truss['truss_name']}; continuing with "
                    f"{remaining} remaining truss(es) "
                    "from measured pose."
                ),
            )

            print(
                f"[NAV] Recenter incomplete after "
                f"{truss['truss_name']}; continuing stop."
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

        update_rmpflow_base_pose(
            stage,
            rmpflow,
        )

    return True, articulation_policy, q_hold
