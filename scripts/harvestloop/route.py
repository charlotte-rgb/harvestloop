"""Full aisle-1 route entry point."""

import json
import math

import numpy as np

import omni.usd
import omni.timeline
import omni.kit.app
from isaacsim.core.prims import SingleArticulation

from harvest_config import JACKAL_PATH

from .config import *  # noqa: F401,F403
from .config import (
    resolve_run_condition,
    resolve_run_experiment,
)
from .simctl import next_frame, wait_frames
from .usd_pose import get_base_state
from .arm_control import (
    create_rmpflow,
    get_fixed_arm_home_pose,
    hold_arm_pose,
    move_arm_to_fixed_home,
    update_rmpflow_base_pose,
)
from .base_control import (
    SmoothVelocityCommander,
    apply_base_velocity_raw,
    follow_waypoint_list,
    get_wheel_indices,
    reset_jackal_to_exact_home,
    smooth_stop,
    turn_to_yaw,
)
from .basket import (
    add_prepared_basket_obstacles_to_rmpflow,
    create_basket_obstacle_wrappers_before_physics,
    enable_physx_scene_ccd_before_physics,
    find_basket_drop_point_path,
    grasp_tip_drop_offset_in_basket_frame,
    prepare_basket_physx_colliders_before_physics,
    set_basket_physx_collision_enabled,
)
from .payload import (
    find_cutter_tip_path,
    prepare_truss_carry_ops_before_physics,
)
from .perception import (
    ensure_wrist_camera,
    setup_persistent_wrist_rgbd_capture,
)
from .estimator import (
    make_estimator,
    measure_tool_tip_spacing,
)
from .routing import (
    build_aisle1_harvest_stops,
    dense_aisle_waypoints,
    discover_stems_in_row,
    filter_stops_to_targets,
    flatten_harvest_targets,
    print_harvest_stop_plan,
)
from .metrics import HarvestLog, STAGE_NAV, STAGE_SETUP
from .perf import (
    RunLoopRateLimit,
    StageTimer,
    ViewportUpdates,
    measure,
)
from .runlog import RunArtifacts
from .harvest import inspect_and_harvest_stem_trusses_at_stop


def _apply_target_manifest(
    harvest_stops,
    run,
):
    """
    Lock the target list for a paired experiment.

    First condition discovers trusses, optionally keeps the first
    MAX_PAIRED_TARGETS, and writes targets.json. The second condition
    reads that list so both harvest the same plants after a scene
    rebuild with the same seed.
    """
    planned = flatten_harvest_targets(
        harvest_stops
    )

    manifest_path = (
        run.experiment_dir
        / "targets.json"
    )

    if manifest_path.exists():
        payload = json.loads(
            manifest_path.read_text()
        )
        allowed = [
            item["truss_path"]
            for item in payload.get(
                "targets",
                [],
            )
            if item.get(
                "truss_path"
            )
        ]

        print(
            f"[PAIR] Reusing {len(allowed)} targets from "
            f"{manifest_path}"
        )
    else:
        allowed_items = planned

        if MAX_PAIRED_TARGETS is not None:
            allowed_items = planned[
                : int(
                    MAX_PAIRED_TARGETS
                )
            ]

        allowed = [
            item["truss_path"]
            for item in allowed_items
        ]

        payload = {
            "experiment_id": run.experiment_id,
            "scene_seed": SCENE_SEED,
            "occlusion_level": OCCLUSION_LEVEL,
            "max_targets": MAX_PAIRED_TARGETS,
            "discovered": len(
                planned
            ),
            "protocol": "shared_view_rescue",
            "gt_scoring_only": True,
            "fixed_views": 1,
            "active_max_views": ACTIVE_PERCEPTION_MAX_VIEWS,
            "targets": allowed_items,
        }

        manifest_path.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n"
        )

        print(
            f"[PAIR] Wrote {len(allowed)}/"
            f"{len(planned)} targets to {manifest_path}"
        )

    filtered = filter_stops_to_targets(
        harvest_stops,
        allowed,
    )

    missing = [
        path
        for path in allowed
        if not any(
            truss.get(
                "truss_path"
            ) == path
            for stop in filtered
            for truss in (
                stop.get(
                    "stem",
                    {},
                ).get(
                    "trusses",
                    [],
                )
            )
        )
    ]

    if missing:
        detail = "\n".join(
            f"  missing: {path}"
            for path in missing
        )
        raise RuntimeError(
            f"{len(missing)} manifest targets not on this stage "
            f"(seed={SCENE_SEED}, occlusion={OCCLUSION_LEVEL}). "
            "Matched LOW/MEDIUM/HIGH sweeps require identical "
            "truss paths after rebuild.\n"
            f"{detail}"
        )

    print(
        f"[PAIR] Manifest locked: {len(allowed)} targets | "
        f"seed={SCENE_SEED} occlusion={OCCLUSION_LEVEL} | "
        f"stops={len(filtered)} "
        f"targets={sum(len(stop['stem']['trusses']) for stop in filtered)}"
    )

    print_harvest_stop_plan(
        filtered
    )

    experiment_path = (
        run.experiment_dir
        / "experiment.json"
    )

    experiment = {}

    if experiment_path.exists():
        try:
            experiment = json.loads(
                experiment_path.read_text()
            )
        except json.JSONDecodeError:
            experiment = {}

    experiment.update(
        {
            "experiment_id": run.experiment_id,
            "protocol": "shared_view_rescue",
            "scene_seed": SCENE_SEED,
            "occlusion_level": OCCLUSION_LEVEL,
            "max_targets": MAX_PAIRED_TARGETS,
            "gt_scoring_only": True,
            "fixed_views": 1,
            "active_max_views": ACTIVE_PERCEPTION_MAX_VIEWS,
            "pair_state_restore": False,
            "matched_manifest": True,
        }
    )

    experiment_path.write_text(
        json.dumps(
            experiment,
            indent=2,
        )
        + "\n"
    )

    return filtered


# ============================================================
# MAIN
# ============================================================

async def main(condition=None, experiment=None):
    """
    Run the aisle-1 route and always leave artifacts behind.

    The route has many early returns. Writing artifacts here rather
    than at each of them means an aborted run still keeps its images,
    per-truss records and outcome summary.

    `condition` may be "fixed_view" or "active_perception". If omitted,
    --condition / HARVESTLOOP_CONDITION / config default is used.

    Artifacts land in:
      results/runs/<experiment>/<condition>/
    """
    resolved = resolve_run_condition(
        explicit=condition
    )

    experiment_id = resolve_run_experiment(
        explicit=experiment,
        condition=resolved,
    )

    print(
        f"[RUN] experiment={experiment_id!r} "
        f"condition={resolved!r}"
        + (
            f" (launcher condition={condition!r})"
            if condition is not None
            else " (CLI / env / config default)"
        )
    )

    run = RunArtifacts(
        condition=resolved,
        experiment_id=experiment_id,
    )

    harvest_log = HarvestLog()

    timing = StageTimer()

    rate_limit = RunLoopRateLimit()

    viewport = ViewportUpdates()

    progress = {
        "stops_planned": None,
        "stops_visited": 0,
        "note": "",
    }

    try:
        rate_limit.disable()

        viewport.pause()

        await run_route(
            run=run,
            harvest_log=harvest_log,
            progress=progress,
            timing=timing,
        )
    except Exception as exc:
        progress["note"] = (
            f"Route raised: {exc}"
        )
        raise
    finally:
        viewport.resume()

        rate_limit.restore()

        harvest_log.print_summary()

        timing.print_summary()

        # Never let an artifact-writing problem mask a route failure.
        try:
            run.write(
                harvest_log,
                stops_planned=progress[
                    "stops_planned"
                ],
                stops_visited=progress[
                    "stops_visited"
                ],
                note=progress[
                    "note"
                ],
                timing=timing,
            )
        except Exception as exc:
            print(
                f"[RUN ARTIFACTS] Could not write run files: {exc}"
            )


async def run_route(
    run,
    harvest_log,
    progress,
    timing=None,
):
    print()
    print("========================================")
    print(
        "HarvestLoop — AISLE-1 ROUTE + VIEW + "
        "GRASP/CUT/CARRY/RELEASE"
    )
    print("========================================")

    print(
        f"Central centerline: x={CORRIDOR_X:.2f}"
    )

    print(
        f"Aisle 1 centerline: y={AISLE_Y:.3f}"
    )

    print(
        f"Cross-aisle gap: {CROSS_AISLE_GAP:.2f} m "
        f"(narrow scene)"
    )

    print(
        "Aisle nav: fixed y=AISLE_Y, heading -X; "
        "stop x from stem only"
    )

    print(
        f"Left bed X span: "
        f"{LEFT_BED_OUTER_X:.3f} -> {LEFT_BED_INNER_X:.3f} m"
    )

    print(
        "Per truss: wrist viewpoint -> "
        "grasp -> cut -> carry -> basket release."
    )

    print(
        f"PREGRASP offset: {PREGRASP_OFFSET:.3f} m "
        f"(left -Y / right +Y)"
    )

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        print("[ERROR] No USD stage.")
        return

    required_paths = [
        JACKAL_BASE_PATH,
        ARM_PATH,
        ARM_BASE_PATH,
        GRASP_TIP_PATH,
        AISLE1_LEFT_ROW_ROOT,
        AISLE1_RIGHT_ROW_ROOT,
    ]

    for required_path in required_paths:
        if not stage.GetPrimAtPath(
            required_path
        ).IsValid():
            print(
                f"[ERROR] Missing: "
                f"{required_path}"
            )
            return

    timeline = (
        omni.timeline
        .get_timeline_interface()
    )

    if timeline.is_playing():
        timeline.stop()

        for _ in range(8):
            await omni.kit.app.get_app().next_update_async()

    # Real PhysX basket colliders + CCD + truss carry ops BEFORE
    # articulation views (same order as the stable baseline).
    try:
        prepare_basket_physx_colliders_before_physics(
            stage
        )

        set_basket_physx_collision_enabled(
            stage,
            False,
        )

        enable_physx_scene_ccd_before_physics(
            stage
        )

    except Exception as exc:
        print()
        print(
            "[STOPPING TEST] Could not prepare Basket "
            f"PhysX colliders / CCD: {exc}"
        )
        return

    try:
        basket_obstacles = (
            create_basket_obstacle_wrappers_before_physics(
                stage
            )
        )
    except Exception as exc:
        print()
        print(
            "[STOPPING TEST] Could not prepare Basket "
            f"obstacle wrappers: {exc}"
        )
        return

    try:
        truss_carry_ops = (
            prepare_truss_carry_ops_before_physics(
                stage
            )
        )
    except Exception as exc:
        print()
        print(
            "[STOPPING TEST] Could not prepare "
            f"truss carry transforms: {exc}"
        )
        return

    try:
        cutter_tip_path = find_cutter_tip_path(
            stage
        )
    except Exception as exc:
        print()
        print(
            f"[STOPPING TEST] {exc}"
        )
        return

    print(
        f"[CUTTER] Using tip: {cutter_tip_path}"
    )

    try:
        ensure_wrist_camera(
            stage
        )
    except Exception as exc:
        print()
        print(
            f"[STOPPING TEST] WristCamera setup failed: {exc}"
        )
        return

    # Discover privileged stems/trusses BEFORE physics starts moving.
    left_stems = discover_stems_in_row(
        stage,
        AISLE1_LEFT_ROW_ROOT,
    )

    right_stems = discover_stems_in_row(
        stage,
        AISLE1_RIGHT_ROW_ROOT,
    )

    if (
        not left_stems
        and not right_stems
    ):
        print(
            "[STOPPING TEST] No stems found under aisle-1 beds."
        )
        return

    harvest_stops = build_aisle1_harvest_stops(
        left_stems,
        right_stems,
    )

    harvest_stops = _apply_target_manifest(
        harvest_stops,
        run,
    )

    if not harvest_stops:
        print(
            "[STOPPING TEST] No harvest stops planned."
        )
        return

    progress["stops_planned"] = len(
        harvest_stops
    )

    timeline.play()

    await wait_frames(
        15
    )

    # Create Replicator render product BEFORE articulation views exist.
    try:
        capture_session = (
            await setup_persistent_wrist_rgbd_capture(
                stage
            )
        )
    except Exception as exc:
        print()
        print(
            "[STOPPING TEST] Persistent wrist capture "
            f"setup failed: {exc}"
        )
        return

    jackal = SingleArticulation(
        prim_path=JACKAL_PATH,
        name="harvestloop_aisle1_viewpoint_jackal",
    )

    arm = SingleArticulation(
        prim_path=ARM_PATH,
        name="harvestloop_aisle1_viewpoint_arm",
    )

    jackal.initialize()
    arm.initialize()

    await wait_frames(
        5
    )

    get_wheel_indices(
        jackal
    )

    arm_indices, q_hold = get_fixed_arm_home_pose(
        arm
    )

    home_ok = await move_arm_to_fixed_home(
        arm=arm,
        arm_indices=arm_indices,
        q_home=q_hold,
        label="BASKET TRANSPORT POSE",
    )

    if not home_ok:
        print(
            "[STOPPING TEST] UR5e did not reach transport pose."
        )
        return

    # Record known-good GraspTip↔DropPoint offset while the arm is
    # still at the basket transport posture (baseline sequence).
    try:
        basket_drop_reference_offset = (
            grasp_tip_drop_offset_in_basket_frame(
                stage
            ).copy()
        )

        basket_drop_point_path = (
            find_basket_drop_point_path(
                stage
            )
        )

        print()
        print("========================================")
        print("[BASKET TASK REFERENCE]")
        print("========================================")

        print(
            f"DropPoint: {basket_drop_point_path}"
        )

        print(
            "Known-good GraspTip-DropPoint offset "
            "in Basket frame: "
            f"{np.round(basket_drop_reference_offset, 4)}"
        )

    except Exception as exc:
        print(
            "[STOPPING TEST] Could not record Basket "
            f"DropPoint reference: {exc}"
        )
        return

    tool_spacing_m = measure_tool_tip_spacing(
        stage,
        GRASP_TIP_PATH,
        CUTTER_TIP_PATH,
    )

    estimator = make_estimator(
        stage,
        mode=ESTIMATOR_MODE,
        tool_spacing_m=tool_spacing_m,
    )

    print()
    print("========================================")
    print("[PERCEPTION SEAM]")
    print("========================================")

    print(
        f"Condition: {RUN_CONDITION_LABEL}"
    )

    if RUN_CONDITION_LABEL == "active_perception":
        print(
            f"Active views: up to {ACTIVE_PERCEPTION_MAX_VIEWS} "
            f"({', '.join(ACTIVE_VIEWPOINT_SEQUENCE)}) | "
            f"lateral offset ±{ACTIVE_VIEW_LATERAL_OFFSET_M:.3f} m"
        )
    else:
        print(
            "Fixed view: one canonical wrist capture per target"
        )

    print(
        f"Estimator: {getattr(estimator, 'name', ESTIMATOR_MODE)}"
    )

    print(
        f"Tool tip spacing: {tool_spacing_m:.4f} m"
    )

    print(
        "Controller consumes only estimate_target(...); "
        "USD GraspPoint/CutPoint are hidden GT for scoring."
    )

    start_ok = await reset_jackal_to_exact_home(
        stage=stage,
        jackal=jackal,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
    )

    if not start_ok:
        print(
            "[STOPPING TEST] Jackal could not be reset to HOME."
        )
        return

    commander = SmoothVelocityCommander(
        jackal
    )

    await smooth_stop(
        commander,
        settle_frames=8,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
    )

    initial = get_base_state(
        stage
    )

    print(
        f"[INITIAL] "
        f"xy=({initial['x']:+.3f}, {initial['y']:+.3f}) | "
        f"yaw={initial['yaw']:+.3f}"
    )

    # --------------------------------------------------------
    # ENTER AISLE 1
    # --------------------------------------------------------

    if not await turn_to_yaw(
        stage,
        commander,
        arm,
        arm_indices,
        q_hold,
        CORRIDOR_YAW,
        "HOME -> face +Y",
    ):
        return

    if not await follow_waypoint_list(
        stage=stage,
        commander=commander,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
        waypoints=CENTRAL_FORWARD_WAYPOINTS,
        reverse=False,
        speed_limit=MAX_MAIN_SPEED,
        label="central corridor -> aisle 1",
        path_axis="vertical",
    ):
        return

    if not await turn_to_yaw(
        stage,
        commander,
        arm,
        arm_indices,
        q_hold,
        LEFT_YAW,
        "aisle intersection -> face -X",
    ):
        return

    rmpflow = None
    articulation_policy = None
    active_basket_obstacles = None

    last_stop_x = None

    for stop in harvest_stops:
        stop_index = stop[
            "stop_index"
        ]
        stop_x = stop[
            "stop_x"
        ]

        print()
        print("========================================")
        print(
            f"[NAV] Drive to harvest stop {stop_index:02d} "
            f"@ x={stop_x:+.3f}"
        )
        print("========================================")

        from_x = (
            CORRIDOR_X
            if last_stop_x is None
            else last_stop_x
        )

        current = get_base_state(
            stage
        )

        # Consecutive stems (e.g. left then right at similar X) often
        # produce nearly identical stop_x. If we are already at/past
        # this stop toward -X, do not drive — that caused ~160 deg
        # yaw spins trying to reverse a few centimeters.
        already_at_or_past = (
            current["x"]
            <= stop_x
            + AISLE_STOP_SKIP_TOLERANCE
        )

        print(
            f"[NAV] from_x={from_x:+.3f} -> stop_x={stop_x:+.3f} | "
            f"current_x={current['x']:+.3f} | "
            f"stem_ws={stop['stem_workspace_x']:+.3f} | "
            f"before=+{STOP_BEFORE_STEM_OFFSET:.3f} m"
        )

        if already_at_or_past:
            print(
                f"[NAV] stop_{stop_index:02d} already at/past "
                f"target (tol={AISLE_STOP_SKIP_TOLERANCE:.3f} m); "
                "skip aisle drive"
            )

            await smooth_stop(
                commander,
                settle_frames=4,
                arm=arm,
                arm_indices=arm_indices,
                q_hold=q_hold,
            )
        else:
            aisle_waypoints = dense_aisle_waypoints(
                current["x"],
                stop_x,
                step=AISLE_WAYPOINT_STEP,
            )

            # Outbound only: always drive forward toward -X.
            aisle_reverse = False

            if stop_x > CORRIDOR_X:
                print(
                    "[NAV WARNING] stop_x is on the RIGHT side of the "
                    f"corridor ({stop_x:+.3f}); clamping motion to left aisle."
                )
                stop_x = CORRIDOR_X - STOP_X_MARGIN
                aisle_waypoints = dense_aisle_waypoints(
                    current["x"],
                    stop_x,
                    step=AISLE_WAYPOINT_STEP,
                )

            print(
                f"[NAV] driving reverse={aisle_reverse} | "
                f"waypoints={len(aisle_waypoints)}"
            )

            with measure(timing, "DRIVE"):
                drive_ok = await follow_waypoint_list(
                    stage=stage,
                    commander=commander,
                    arm=arm,
                    arm_indices=arm_indices,
                    q_hold=q_hold,
                    waypoints=aisle_waypoints,
                    reverse=aisle_reverse,
                    speed_limit=MAX_AISLE_SPEED,
                    label=(
                        f"aisle 1 -> stop_{stop_index:02d}"
                    ),
                    path_axis="horizontal",
                )

            if not drive_ok:
                # Skip this stop instead of ending the route: the
                # next stop plans from the measured pose, so one bad
                # drive costs one stem rather than the whole run.
                harvest_log.record_event(
                    STAGE_NAV,
                    f"stop_{stop_index:02d}",
                    "Aisle drive to the stop failed; stop skipped.",
                )

                last_stop_x = stop_x
                continue

        parked = get_base_state(
            stage
        )

        ahead_of_stem = (
            stop["stem_workspace_x"]
            - parked["x"]
        )

        print(
            f"[PARKED] stop_{stop_index:02d} at "
            f"({parked['x']:+.3f}, {parked['y']:+.3f}) | "
            f"planned={stop_x:+.3f} | "
            f"stem_ws={stop['stem_workspace_x']:+.3f} | "
            f"stem_ahead_by={ahead_of_stem:+.3f} m "
            f"(>0 means stem still ahead while facing -X)"
        )

        await smooth_stop(
            commander,
            settle_frames=8,
            arm=arm,
            arm_indices=arm_indices,
            q_hold=q_hold,
        )

        # Keep aisle heading horizontal (-X); no lateral refine.
        with measure(timing, "DRIVE"):
            heading_ok = await turn_to_yaw(
                stage,
                commander,
                arm,
                arm_indices,
                q_hold,
                LEFT_YAW,
                f"stop_{stop_index:02d} hold aisle heading -X",
            )

        if not heading_ok:
            harvest_log.record_event(
                STAGE_NAV,
                f"stop_{stop_index:02d}",
                "Could not hold the aisle heading; stop skipped.",
            )

            last_stop_x = stop_x
            continue

        with measure(timing, "STOP SETUP"):
            upright_ok = await move_arm_to_fixed_home(
                arm=arm,
                arm_indices=arm_indices,
                q_home=UR5E_UPRIGHT_Q,
                label=(
                    f"SAFE UPRIGHT @ stop_{stop_index:02d}"
                ),
            )

        if not upright_ok:
            # Do not abandon the stop — force an upright hold and
            # continue so remaining trusses are still attempted.
            harvest_log.record_event(
                STAGE_SETUP,
                f"stop_{stop_index:02d}",
                "SAFE UPRIGHT at the stop incomplete; "
                "forcing hold and continuing.",
            )

            print(
                f"[STOP SETUP] upright incomplete @ "
                f"stop_{stop_index:02d}; forcing hold."
            )

        for _ in range(8):
            hold_arm_pose(
                arm,
                arm_indices,
                UR5E_UPRIGHT_Q,
            )
            commander.emergency_stop()
            await next_frame()

        if rmpflow is None:
            (
                rmpflow,
                articulation_policy,
            ) = create_rmpflow(
                arm
            )

            update_rmpflow_base_pose(
                stage,
                rmpflow,
            )

            try:
                active_basket_obstacles = (
                    add_prepared_basket_obstacles_to_rmpflow(
                        rmpflow=rmpflow,
                        basket_obstacles=basket_obstacles,
                    )
                )
            except Exception as exc:
                print(
                    "[STOPPING TEST] Basket RMPflow obstacles "
                    f"failed: {exc}"
                )
                commander.emergency_stop()
                return

            print(
                f"[RMPFLOW] Registered "
                f"{len(active_basket_obstacles)} "
                "static basket cuboids."
            )

        # Keep wrappers alive for the whole multi-stop session.
        _ = active_basket_obstacles

        side_ok, articulation_policy, q_hold = (
            await inspect_and_harvest_stem_trusses_at_stop(
                stage=stage,
                jackal=jackal,
                arm=arm,
                arm_indices=arm_indices,
                rmpflow=rmpflow,
                articulation_policy=articulation_policy,
                capture_session=capture_session,
                stop_index=stop_index,
                side_name=stop[
                    "side_name"
                ],
                stem=stop[
                    "stem"
                ],
                approach_offset=stop[
                    "approach_offset"
                ],
                cutter_tip_path=cutter_tip_path,
                truss_carry_ops=truss_carry_ops,
                basket_drop_reference_offset=(
                    basket_drop_reference_offset
                ),
                q_hold=q_hold,
                commander=commander,
                stop_x=stop_x,
                harvest_log=harvest_log,
                estimator=estimator,
                run=run,
                timing=timing,
            )
        )

        if not side_ok:
            print(
                f"[NAV WARN] Stop {stop_index} harvest incomplete; "
                "continuing to remaining stops."
            )
            progress["note"] = (
                f"Incomplete stop_{stop_index:02d}; continuing."
            )

            for _ in range(10):
                hold_arm_pose(
                    arm,
                    arm_indices,
                    q_hold,
                )
                apply_base_velocity_raw(
                    jackal,
                    0.0,
                    0.0,
                )
                await next_frame()

            continue

        # Return to transport pose before the next Jackal move.
        transport_ok = await move_arm_to_fixed_home(
            arm=arm,
            arm_indices=arm_indices,
            q_home=q_hold,
            label=(
                f"TRANSPORT POSE after stop_{stop_index:02d}"
            ),
        )

        if not transport_ok:
            print(
                "[NAV WARN] Transport pose incomplete after "
                f"stop_{stop_index:02d}; forcing hold and moving on."
            )

        # Always hold q_hold while driving, even if residuals remain.
        for _ in range(8):
            hold_arm_pose(
                arm,
                arm_indices,
                q_hold,
            )
            apply_base_velocity_raw(
                jackal,
                0.0,
                0.0,
            )
            await next_frame()

        last_stop_x = stop_x

        progress["stops_visited"] += 1

    # --------------------------------------------------------
    # RETURN HOME ALONG THE SAME CENTERLINES
    # --------------------------------------------------------

    print()
    print("========================================")
    print("[NAV] All stops done — return HOME")
    print("========================================")

    if last_stop_x is None:
        print(
            "[STOPPING TEST] No stop was visited."
        )
        return

    return_aisle_waypoints = dense_aisle_waypoints(
        last_stop_x,
        CORRIDOR_X,
        step=AISLE_WAYPOINT_STEP,
    )

    if not await follow_waypoint_list(
        stage=stage,
        commander=commander,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
        waypoints=return_aisle_waypoints,
        reverse=True,
        speed_limit=MAX_AISLE_SPEED,
        label="last stop -> aisle intersection",
        path_axis="horizontal",
    ):
        return

    if not await turn_to_yaw(
        stage,
        commander,
        arm,
        arm_indices,
        q_hold,
        CORRIDOR_YAW,
        "intersection -> face +Y",
    ):
        return

    if not await follow_waypoint_list(
        stage=stage,
        commander=commander,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
        waypoints=CENTRAL_REVERSE_WAYPOINTS,
        reverse=True,
        speed_limit=MAX_MAIN_SPEED,
        label="aisle 1 -> HOME",
        path_axis="vertical",
    ):
        return

    if not await turn_to_yaw(
        stage,
        commander,
        arm,
        arm_indices,
        q_hold,
        HOME_YAW,
        "restore HOME orientation",
    ):
        return

    await smooth_stop(
        commander,
        settle_frames=8,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
    )

    final = get_base_state(
        stage
    )

    print()
    print("========================================")
    print(
        "[TEST PASSED] AISLE-1 ROUTE + VIEW + "
        "GRASP/CUT/CARRY/RELEASE"
    )
    print("========================================")

    print(
        f"Stops visited: {progress['stops_visited']}"
    )

    print(
        f"Run directory: {run.run_dir}"
    )

    print(
        f"Final xy=({final['x']:+.4f}, "
        f"{final['y']:+.4f})"
    )

    print(
        f"Final yaw={final['yaw']:+.4f} rad"
    )

    print(
        f"Final roll={math.degrees(final['roll']):+.2f} deg | "
        f"pitch={math.degrees(final['pitch']):+.2f} deg"
    )
