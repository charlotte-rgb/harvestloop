"""UR5e joint-space control and RMPflow policy setup."""

import math

import numpy as np

from isaacsim.core.utils.types import ArticulationAction
import isaacsim.robot_motion.motion_generation as mg

from usd_utils import get_world_pose

from .config import *  # noqa: F401,F403
from .simctl import next_frame
from .usd_pose import find_project_file, get_base_state


def check_tip(
    stage,
    commander,
):
    state = get_base_state(
        stage
    )

    if state["tilt"] >= TIP_HARD_STOP:
        commander.emergency_stop()

        print()
        print("========================================")
        print("[EMERGENCY STOP — TIP LIMIT]")
        print("========================================")

        print(
            f"roll={math.degrees(state['roll']):+.2f} deg | "
            f"pitch={math.degrees(state['pitch']):+.2f} deg"
        )

        return False, state

    return True, state


# ============================================================
# UR5e TRANSPORT / UPRIGHT HOLD
# ============================================================

def get_arm_joint_indices(arm):
    """
    Return articulation indices for the six UR5e joints in the
    ACTIVE_ARM_JOINTS order.
    """
    indices = []

    for name in ACTIVE_ARM_JOINTS:
        if name not in arm.dof_names:
            raise RuntimeError(
                f"Missing UR5e DOF '{name}'. "
                f"Available DOFs: {list(arm.dof_names)}"
            )

        indices.append(
            arm.dof_names.index(name)
        )

    return np.asarray(
        indices,
        dtype=np.int32,
    )


def get_active_arm_positions(
    arm,
    arm_indices,
):
    all_positions = np.asarray(
        arm.get_joint_positions(),
        dtype=np.float64,
    )

    return all_positions[
        arm_indices
    ].copy()


def get_fixed_arm_home_pose(arm):
    """
    Use the manually captured HarvestLoop basket transport pose.

    Joint order:
        shoulder_pan_joint
        shoulder_lift_joint
        elbow_joint
        wrist_1_joint
        wrist_2_joint
        wrist_3_joint

    MANUAL DROP/BASKET POSE:
        [0.0, -pi/2, 0.0, 0.0, 0.0, 0.0]
    """
    indices = get_arm_joint_indices(
        arm
    )

    q_home = UR5E_HOME_Q.copy()

    print()
    print("========================================")
    print("[UR5e MANUAL BASKET TRANSPORT POSE]")
    print("========================================")

    for name, q in zip(
        ACTIVE_ARM_JOINTS,
        q_home,
    ):
        print(
            f"  {name:22s} = {q:+.4f} rad"
        )

    return (
        indices,
        q_home,
    )


def hold_arm_pose(
    arm,
    arm_indices,
    q_hold,
):
    """
    Re-command the fixed six-joint UR5e HOME/transport pose.

    This is called every simulation frame during navigation and
    turning so the arm does not sag or swing with base motion.
    """
    action = ArticulationAction(
        joint_positions=np.asarray(
            q_hold,
            dtype=np.float32,
        ),
        joint_indices=np.asarray(
            arm_indices,
            dtype=np.int32,
        ),
    )

    arm.apply_action(
        action
    )


async def move_arm_to_fixed_home(
    arm,
    arm_indices,
    q_home,
    label="FIXED JOINT POSE",
):
    """
    Move the six UR5e joints directly to a specified configuration.
    """
    print()
    print("========================================")
    print(f"[UR5e] MOVING TO {label}")
    print("========================================")

    best_error = float("inf")
    steps_since_improve = 0

    for step in range(
        ARM_HOME_MAX_STEPS
    ):
        hold_arm_pose(
            arm,
            arm_indices,
            q_home,
        )

        await next_frame()

        q = get_active_arm_positions(
            arm,
            arm_indices,
        )

        error = np.abs(
            q - q_home
        )

        max_error = float(
            np.max(error)
        )

        if max_error + 1e-4 < best_error:
            best_error = max_error
            steps_since_improve = 0
        else:
            steps_since_improve += 1

        if step % 30 == 0:
            print(
                f"[ARM HOME] step={step:4d} | "
                f"max_joint_error={max_error:.4f} rad | "
                f"best={best_error:.4f} rad"
            )

        if max_error <= ARM_HOME_TOLERANCE:
            print(
                f"[ARM HOME REACHED] "
                f"max_joint_error={max_error:.4f} rad"
            )

            for name, value in zip(
                ACTIVE_ARM_JOINTS,
                q,
            ):
                print(
                    f"  {name:22s} = {value:+.4f} rad"
                )

            return True

        # Plateau near target: do not wait out the full budget.
        if (
            best_error
            <= ARM_HOME_ACCEPT_TOLERANCE
            and steps_since_improve
            >= ARM_HOME_PLATEAU_STEPS
        ):
            print(
                f"[ARM HOME ACCEPT] plateau "
                f"best={best_error:.4f} rad "
                f"(tol={ARM_HOME_TOLERANCE:.4f}, "
                f"accept<={ARM_HOME_ACCEPT_TOLERANCE:.4f})"
            )
            return True

    q = get_active_arm_positions(
        arm,
        arm_indices,
    )

    max_error = float(
        np.max(
            np.abs(
                q - q_home
            )
        )
    )

    if max_error <= ARM_HOME_ACCEPT_TOLERANCE:
        print()
        print(
            f"[ARM HOME ACCEPT] timeout "
            f"max_joint_error={max_error:.4f} rad"
        )
        return True

    print()
    print(
        f"[FAILED] UR5e could not reach fixed HOME. "
        f"max_joint_error={max_error:.4f} rad | "
        f"best={best_error:.4f} rad"
    )

    return False


def create_rmpflow(arm):
    urdf_path = find_project_file(
        URDF_FILENAME
    )

    robot_description_path = (
        find_project_file(
            ROBOT_DESCRIPTION_FILENAME
        )
    )

    rmpflow_config_path = (
        find_project_file(
            RMPFLOW_CONFIG_FILENAME
        )
    )

    print()
    print("========================================")
    print("[RMPFLOW] SCALED UR5e CONFIG")
    print("========================================")

    print(
        f"URDF:       {urdf_path}"
    )

    print(
        f"Robot desc: {robot_description_path}"
    )

    print(
        f"RMP config: {rmpflow_config_path}"
    )

    rmpflow = (
        mg.lula.motion_policies.RmpFlow(
            robot_description_path=str(
                robot_description_path
            ),
            urdf_path=str(
                urdf_path
            ),
            rmpflow_config_path=str(
                rmpflow_config_path
            ),
            end_effector_frame_name=(
                "grasp_tip"
            ),
            maximum_substep_size=(
                RMPFLOW_MAX_SUBSTEP
            ),
        )
    )

    articulation_policy = (
        mg.ArticulationMotionPolicy(
            arm,
            rmpflow,
            RMPFLOW_PHYSICS_DT,
        )
    )

    return (
        rmpflow,
        articulation_policy,
    )


def update_rmpflow_base_pose(
    stage,
    rmpflow,
):
    """
    Important: use the actual USD ur5e/base_link world transform.
    """
    (
        base_position,
        base_orientation,
    ) = get_world_pose(
        stage,
        ARM_BASE_PATH,
    )

    base_position = np.asarray(
        base_position,
        dtype=np.float64,
    )

    base_orientation = np.asarray(
        base_orientation,
        dtype=np.float64,
    )

    rmpflow.set_robot_base_pose(
        base_position,
        base_orientation,
    )

    return (
        base_position,
        base_orientation,
    )
