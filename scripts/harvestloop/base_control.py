"""Jackal wheel control, smoothed velocity and waypoint following."""

import math

import numpy as np

from isaacsim.core.utils.types import ArticulationAction

from .config import *  # noqa: F401,F403
from .mathutils import quaternion_multiply, wrap_angle, yaw_quaternion
from .simctl import next_frame
from .usd_pose import get_base_state
from .arm_control import check_tip, hold_arm_pose


# ============================================================
# WHEEL CONTROL
# ============================================================

def get_wheel_indices(jackal):
    result = {}

    for name in LEFT_WHEELS + RIGHT_WHEELS:
        if name not in jackal.dof_names:
            raise RuntimeError(
                f"Missing wheel DOF '{name}'. "
                f"Available DOFs: {list(jackal.dof_names)}"
            )

        result[name] = jackal.dof_names.index(
            name
        )

    return result


def set_wheel_velocity(
    jackal,
    left_velocity,
    right_velocity,
):
    indices = get_wheel_indices(
        jackal
    )

    action = ArticulationAction(
        joint_velocities=np.array(
            [
                left_velocity,
                right_velocity,
                left_velocity,
                right_velocity,
            ],
            dtype=np.float32,
        ),
        joint_indices=np.array(
            [
                indices["front_left_wheel_joint"],
                indices["front_right_wheel_joint"],
                indices["rear_left_wheel_joint"],
                indices["rear_right_wheel_joint"],
            ],
            dtype=np.int32,
        ),
    )

    jackal.apply_action(
        action
    )


def apply_base_velocity_raw(
    jackal,
    linear_velocity,
    angular_velocity,
):
    left_linear = (
        linear_velocity
        - angular_velocity
        * TRACK_WIDTH
        / 2.0
    )

    right_linear = (
        linear_velocity
        + angular_velocity
        * TRACK_WIDTH
        / 2.0
    )

    set_wheel_velocity(
        jackal,
        left_linear / WHEEL_RADIUS,
        right_linear / WHEEL_RADIUS,
    )


# ============================================================
# SMOOTH COMMANDER
# ============================================================

class SmoothVelocityCommander:
    def __init__(self, jackal):
        self.jackal = jackal
        self.current_v = 0.0
        self.current_w = 0.0

    @staticmethod
    def _ramp(
        current,
        target,
        accel,
        decel,
    ):
        current = float(current)
        target = float(target)

        same_sign = (
            abs(current) < 1e-9
            or abs(target) < 1e-9
            or math.copysign(1.0, current)
            == math.copysign(1.0, target)
        )

        accelerating = (
            same_sign
            and abs(target) > abs(current)
        )

        rate = accel if accelerating else decel

        max_change = rate * CONTROL_DT

        return current + float(
            np.clip(
                target - current,
                -max_change,
                +max_change,
            )
        )

    def command(
        self,
        target_v,
        target_w,
        angular_accel=DRIVE_ANGULAR_ACCEL,
        angular_decel=DRIVE_ANGULAR_DECEL,
    ):
        self.current_v = self._ramp(
            self.current_v,
            target_v,
            MAX_LINEAR_ACCEL,
            MAX_LINEAR_DECEL,
        )

        self.current_w = self._ramp(
            self.current_w,
            target_w,
            angular_accel,
            angular_decel,
        )

        apply_base_velocity_raw(
            self.jackal,
            self.current_v,
            self.current_w,
        )

    def emergency_stop(self):
        self.current_v = 0.0
        self.current_w = 0.0

        apply_base_velocity_raw(
            self.jackal,
            0.0,
            0.0,
        )


async def smooth_stop(
    commander,
    settle_frames=12,
    arm=None,
    arm_indices=None,
    q_hold=None,
):
    for _ in range(90):
        commander.command(
            0.0,
            0.0,
            angular_accel=TURN_ANGULAR_ACCEL,
            angular_decel=TURN_ANGULAR_DECEL,
        )

        if (
            arm is not None
            and arm_indices is not None
            and q_hold is not None
        ):
            hold_arm_pose(
                arm,
                arm_indices,
                q_hold,
            )

        await next_frame()

        if (
            abs(commander.current_v) < 0.002
            and abs(commander.current_w) < 0.005
        ):
            break

    commander.emergency_stop()

    for _ in range(settle_frames):
        if (
            arm is not None
            and arm_indices is not None
            and q_hold is not None
        ):
            hold_arm_pose(
                arm,
                arm_indices,
                q_hold,
            )

        await next_frame()


# ============================================================
# FAST IN-PLACE TURN
# ============================================================

async def turn_to_yaw(
    stage,
    commander,
    arm,
    arm_indices,
    q_hold,
    target_yaw,
    label,
):
    print()
    print("========================================")
    print(f"[TURN] {label}")
    print("========================================")

    await smooth_stop(
        commander,
        settle_frames=15,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
    )

    target_yaw = wrap_angle(
        target_yaw
    )

    for step in range(
        MAX_TURN_STEPS
    ):
        safe, state = check_tip(
            stage,
            commander,
        )

        if not safe:
            return False

        error = wrap_angle(
            target_yaw
            - state["yaw"]
        )

        if abs(error) <= TURN_TOLERANCE:
            await smooth_stop(
                commander,
                settle_frames=12,
                arm=arm,
                arm_indices=arm_indices,
                q_hold=q_hold,
            )

            state = get_base_state(
                stage
            )

            print(
                f"[TURN COMPLETE] "
                f"xy=({state['x']:+.3f}, {state['y']:+.3f}) | "
                f"yaw={state['yaw']:+.3f}"
            )

            return True

        target_w = float(
            np.clip(
                K_TURN_YAW * error,
                -MAX_TURN_ANGULAR_SPEED,
                +MAX_TURN_ANGULAR_SPEED,
            )
        )

        target_w = math.copysign(
            max(
                abs(target_w),
                MIN_TURN_ANGULAR_SPEED,
            ),
            error,
        )

        if state["tilt"] >= TIP_WARNING:
            target_w = float(
                np.clip(
                    target_w,
                    -0.22,
                    +0.22,
                )
            )

        commander.command(
            0.0,
            target_w,
            angular_accel=TURN_ANGULAR_ACCEL,
            angular_decel=TURN_ANGULAR_DECEL,
        )

        hold_arm_pose(
            arm,
            arm_indices,
            q_hold,
        )

        if step % PRINT_INTERVAL == 0:
            print(
                f"[TURN] step={step:4d} | "
                f"yaw={state['yaw']:+.3f} | "
                f"err={math.degrees(error):+.1f} deg | "
                f"w={commander.current_w:+.3f} | "
                f"tilt={math.degrees(state['tilt']):.2f} deg"
            )

        await next_frame()

    await smooth_stop(
        commander,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
    )

    print("[FAILED] turn timed out.")
    return False


# ============================================================
# DIRECT POINT FOLLOWER
# ============================================================

async def follow_waypoint(
    stage,
    commander,
    arm,
    arm_indices,
    q_hold,
    target_x,
    target_y,
    reverse,
    speed_limit,
    tolerance,
    label,
    stop_at_target,
    path_axis,
):
    """
    Follow one waypoint while tracking the EXACT global centerline.

    path_axis == "vertical":
        exact centerline x = CORRIDOR_X

    path_axis == "horizontal":
        exact centerline y = AISLE_Y

    Dense waypoints only define longitudinal progress. They no longer
    define a new diagonal path from the Jackal's current position.

    Most importantly, a final stop is accepted as soon as the robot
    reaches/crosses the target's longitudinal plane. Lateral error
    NEVER blocks an intersection stop.
    """

    print()
    print(
        f"[WAYPOINT] {label}: "
        f"({target_x:+.3f}, {target_y:+.3f}) "
        f"{'(reverse)' if reverse else ''} "
        f"{'[STOP]' if stop_at_target else '[PASS]'} "
        f"axis={path_axis}"
    )

    initial_state = get_base_state(
        stage
    )

    if path_axis == "vertical":
        initial_longitudinal = initial_state["y"]
        target_longitudinal = float(target_y)
        centerline_value = CORRIDOR_X

    elif path_axis == "horizontal":
        initial_longitudinal = initial_state["x"]
        target_longitudinal = float(target_x)
        centerline_value = AISLE_Y

    else:
        raise ValueError(
            f"Unknown path_axis: {path_axis}"
        )

    delta_longitudinal = (
        target_longitudinal
        - initial_longitudinal
    )

    if abs(delta_longitudinal) < 1e-8:
        direction_sign = 1.0
    else:
        direction_sign = math.copysign(
            1.0,
            delta_longitudinal,
        )

    # Outbound aisle harvest faces -X (reverse=False, horizontal):
    # if we are already at/past the stop plane toward -X, accept
    # immediately instead of turning ~180 deg to drive back.
    if (
        path_axis == "horizontal"
        and stop_at_target
        and (not reverse)
    ):
        if (
            initial_state["x"]
            <= target_x
            + AISLE_STOP_SKIP_TOLERANCE
        ):
            print(
                f"[STOP ALREADY REACHED/PASSED] "
                f"x={initial_state['x']:+.3f} "
                f"(target={target_x:+.3f}, "
                f"tol={AISLE_STOP_SKIP_TOLERANCE:.3f} m) — "
                "skip drive, hold heading"
            )
            await smooth_stop(
                commander,
                settle_frames=20,
                arm=arm,
                arm_indices=arm_indices,
                q_hold=q_hold,
            )
            return True

    for step in range(
        MAX_SEGMENT_STEPS
    ):
        safe, state = check_tip(
            stage,
            commander,
        )

        if not safe:
            return False

        # ----------------------------------------------------
        # EXACT CENTERLINE + LONGITUDINAL PROGRESS
        # ----------------------------------------------------

        if path_axis == "vertical":
            current_longitudinal = state["y"]

            remaining_along = (
                direction_sign
                * (
                    target_longitudinal
                    - current_longitudinal
                )
            )

            signed_cross_track = (
                CORRIDOR_X
                - state["x"]
            )

            cross_track = abs(
                signed_cross_track
            )

            # Virtual point lies ahead on x=0 exactly.
            virtual_x = CORRIDOR_X
            virtual_y = (
                state["y"]
                + direction_sign
                * CENTERLINE_LOOKAHEAD
            )

        else:
            current_longitudinal = state["x"]

            remaining_along = (
                direction_sign
                * (
                    target_longitudinal
                    - current_longitudinal
                )
            )

            # Fixed aisle centerline: Jackal base_link on y = AISLE_Y.
            # Do not track UR5e offset — that steered into the beds.
            signed_cross_track = (
                AISLE_Y
                - state["y"]
            )

            cross_track = abs(
                signed_cross_track
            )

            # Virtual point lies ahead on the fixed aisle midline.
            virtual_x = (
                state["x"]
                + direction_sign
                * CENTERLINE_LOOKAHEAD
            )

            virtual_y = AISLE_Y

        # ----------------------------------------------------
        # PASS / STOP
        # ----------------------------------------------------

        if stop_at_target:
            # IMPORTANT:
            # Stop based ONLY on longitudinal progress.
            #
            # If remaining_along <= tolerance, the Jackal has reached
            # or crossed the desired intersection/end plane.
            #
            # Lateral error is deliberately NOT part of this condition.
            if (
                remaining_along
                <= FINAL_LONGITUDINAL_TOLERANCE
            ):
                await smooth_stop(
                    commander,
                    settle_frames=12,
                    arm=arm,
                    arm_indices=arm_indices,
                    q_hold=q_hold,
                )

                final_state = get_base_state(
                    stage
                )

                if path_axis == "vertical":
                    final_cross = abs(
                        CORRIDOR_X
                        - final_state["x"]
                    )
                else:
                    final_cross = abs(
                        AISLE_Y
                        - final_state["y"]
                    )

                print(
                    f"[STOP WAYPOINT REACHED] "
                    f"actual=({final_state['x']:+.3f}, "
                    f"{final_state['y']:+.3f}) | "
                    f"along={remaining_along:+.3f} m | "
                    f"cross={final_cross:.3f} m | "
                    f"aisle_y={AISLE_Y:+.3f}"
                )

                return True

        else:
            # Intermediate dense points are pass-through markers.
            if (
                remaining_along
                <= PASS_LONGITUDINAL_TOLERANCE
            ):
                print(
                    f"[PASS WAYPOINT] "
                    f"actual=({state['x']:+.3f}, "
                    f"{state['y']:+.3f}) | "
                    f"along={remaining_along:+.3f} m | "
                    f"cross={cross_track:.3f} m"
                )

                return True

        # ----------------------------------------------------
        # PURE-PURSUIT STYLE CENTERLINE HEADING
        # ----------------------------------------------------

        dx = (
            virtual_x
            - state["x"]
        )

        dy = (
            virtual_y
            - state["y"]
        )

        motion_yaw = math.atan2(
            dy,
            dx,
        )

        desired_body_yaw = (
            wrap_angle(
                motion_yaw
                + math.pi
            )
            if reverse
            else wrap_angle(
                motion_yaw
            )
        )

        yaw_error = wrap_angle(
            desired_body_yaw
            - state["yaw"]
        )

        target_w = float(
            np.clip(
                K_POINT_YAW
                * yaw_error,
                -MAX_DRIVE_ANGULAR_SPEED,
                +MAX_DRIVE_ANGULAR_SPEED,
            )
        )

        # ----------------------------------------------------
        # LINEAR SPEED
        # ----------------------------------------------------

        if stop_at_target:
            longitudinal_distance = max(
                remaining_along,
                0.0,
            )

            speed = min(
                speed_limit,
                K_LINEAR
                * longitudinal_distance,
            )

            # Shorter approach taper so high aisle/main speeds stay fast
            # until close to the stop plane.
            if (
                longitudinal_distance
                < 0.50
            ):
                speed = min(
                    speed,
                    0.70,
                )

            if (
                longitudinal_distance
                < 0.22
            ):
                speed = min(
                    speed,
                    0.35,
                )

            if (
                longitudinal_distance
                < 0.10
            ):
                speed = min(
                    speed,
                    0.15,
                )

        else:
            speed = speed_limit

        # Heading-error slowdown.
        abs_yaw_error = abs(
            yaw_error
        )

        if (
            abs_yaw_error
            >= HEADING_VERY_SLOW_ANGLE
        ):
            speed = min(
                speed,
                HEADING_VERY_SLOW_SPEED,
            )

        elif (
            abs_yaw_error
            >= HEADING_SLOW_ANGLE
        ):
            speed = min(
                speed,
                HEADING_SLOW_SPEED,
            )

        # Additional slowdown only if the base has genuinely
        # drifted away from the exact global centerline.
        if (
            cross_track
            >= CENTERLINE_DRIFT_VERY_SLOW
        ):
            speed = min(
                speed,
                CENTERLINE_DRIFT_VERY_SLOW_SPEED,
            )

        elif (
            cross_track
            >= CENTERLINE_DRIFT_SLOW
        ):
            speed = min(
                speed,
                CENTERLINE_DRIFT_SPEED,
            )

        if state["tilt"] >= TIP_WARNING:
            speed = min(
                speed,
                0.040,
            )

            target_w = float(
                np.clip(
                    target_w,
                    -0.10,
                    +0.10,
                )
            )

        target_v = (
            -speed
            if reverse
            else speed
        )

        commander.command(
            target_v,
            target_w,
            angular_accel=DRIVE_ANGULAR_ACCEL,
            angular_decel=DRIVE_ANGULAR_DECEL,
        )

        hold_arm_pose(
            arm,
            arm_indices,
            q_hold,
        )

        if (
            step
            % PRINT_INTERVAL
            == 0
        ):
            print(
                f"           step={step:4d} | "
                f"xy=({state['x']:+.3f}, "
                f"{state['y']:+.3f}) | "
                f"along={remaining_along:+.3f} | "
                f"cross={cross_track:.3f} | "
                f"yaw_err="
                f"{math.degrees(yaw_error):+.1f} deg | "
                f"v={commander.current_v:+.3f} | "
                f"w={commander.current_w:+.3f}"
            )

        await next_frame()

    await smooth_stop(
        commander,
        arm=arm,
        arm_indices=arm_indices,
        q_hold=q_hold,
    )

    print(
        f"[FAILED] waypoint timed out: "
        f"({target_x:+.3f}, "
        f"{target_y:+.3f})"
    )

    return False


async def follow_waypoint_list(
    stage,
    commander,
    arm,
    arm_indices,
    q_hold,
    waypoints,
    reverse,
    speed_limit,
    label,
    path_axis,
):
    print()
    print("========================================")
    print(f"[PATH] {label}")
    print(
        f"[CENTERLINE] "
        f"{'x=' + format(CORRIDOR_X, '.2f') if path_axis == 'vertical' else 'y=' + format(AISLE_Y, '.2f')}"
    )
    print("========================================")

    for index, (
        target_x,
        target_y,
    ) in enumerate(
        waypoints
    ):
        final_point = (
            index
            == len(waypoints) - 1
        )

        tolerance = (
            FINAL_WAYPOINT_TOLERANCE
            if final_point
            else PASS_THROUGH_TOLERANCE
        )

        ok = await follow_waypoint(
            stage=stage,
            commander=commander,
            arm=arm,
            arm_indices=arm_indices,
            q_hold=q_hold,
            target_x=target_x,
            target_y=target_y,
            reverse=reverse,
            speed_limit=speed_limit,
            tolerance=tolerance,
            label=(
                f"{label} "
                f"{index + 1}/{len(waypoints)}"
            ),
            stop_at_target=final_point,
            path_axis=path_axis,
        )

        if not ok:
            return False

    final = get_base_state(
        stage
    )

    print(
        f"[PATH COMPLETE] {label} | "
        f"xy=({final['x']:+.3f}, "
        f"{final['y']:+.3f})"
    )

    return True


# ============================================================
# EXACT JACKAL START POSE
# ============================================================

async def reset_jackal_to_exact_home(
    stage,
    jackal,
    arm,
    arm_indices,
    q_hold,
):
    """
    Teleport/correct the Jackal articulation root until the actual
    USD base_link is at:

        x   = 0.0
        y   = 0.0
        yaw = 0.0

    We correct from the measured base_link world pose rather than
    assuming that the articulation root and base_link have identical
    local origins.
    """

    print()
    print("========================================")
    print("[JACKAL] RESET TO EXACT HOME")
    print("========================================")

    # Wheels must not be commanding motion while relocating.
    apply_base_velocity_raw(
        jackal,
        0.0,
        0.0,
    )

    for attempt in range(
        START_RESET_ATTEMPTS
    ):
        state = get_base_state(
            stage
        )

        x_error = (
            HOME_X
            - state["x"]
        )

        y_error = (
            HOME_Y
            - state["y"]
        )

        yaw_error = wrap_angle(
            HOME_YAW
            - state["yaw"]
        )

        xy_error = math.hypot(
            x_error,
            y_error,
        )

        print(
            f"[START RESET] attempt={attempt + 1} | "
            f"base=({state['x']:+.5f}, "
            f"{state['y']:+.5f}) | "
            f"yaw={math.degrees(state['yaw']):+.3f} deg | "
            f"xy_err={xy_error:.5f} m | "
            f"yaw_err={math.degrees(yaw_error):+.3f} deg"
        )

        if (
            xy_error
            <= START_XY_TOLERANCE
            and abs(yaw_error)
            <= START_YAW_TOLERANCE
        ):
            print(
                "[START RESET COMPLETE] "
                "Jackal base_link is at HOME."
            )
            return True

        root_position, root_orientation = (
            jackal.get_world_pose()
        )

        root_position = np.asarray(
            root_position,
            dtype=np.float64,
        ).copy()

        root_orientation = np.asarray(
            root_orientation,
            dtype=np.float64,
        ).copy()

        # Correct world yaw of the articulation root.
        yaw_correction_q = yaw_quaternion(
            yaw_error
        )

        corrected_orientation = (
            quaternion_multiply(
                yaw_correction_q,
                root_orientation,
            )
        )

        # Correct world XY. Rotation can slightly change base_link
        # XY if root/base origins differ, so this is repeated.
        corrected_position = (
            root_position.copy()
        )

        corrected_position[0] += x_error
        corrected_position[1] += y_error

        jackal.set_world_pose(
            position=corrected_position,
            orientation=corrected_orientation,
        )

        # Continue holding the upright UR5e during relocation.
        for _ in range(
            START_RESET_SETTLE_FRAMES
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

    # Final verification.
    state = get_base_state(
        stage
    )

    xy_error = math.hypot(
        HOME_X - state["x"],
        HOME_Y - state["y"],
    )

    yaw_error = wrap_angle(
        HOME_YAW
        - state["yaw"]
    )

    if (
        xy_error
        <= START_XY_TOLERANCE
        and abs(yaw_error)
        <= START_YAW_TOLERANCE
    ):
        print(
            "[START RESET COMPLETE] "
            "Jackal base_link is at HOME."
        )
        return True

    print()
    print(
        "[FAILED] Could not place Jackal base_link "
        "at exact HOME."
    )

    print(
        f"Final base=({state['x']:+.5f}, "
        f"{state['y']:+.5f}) | "
        f"yaw={math.degrees(state['yaw']):+.3f} deg"
    )

    return False
