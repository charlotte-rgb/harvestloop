"""Simulation stepping and articulation (re)initialisation helpers."""

import time

import omni.kit.app
import isaacsim.robot_motion.motion_generation as mg

from .config import *  # noqa: F401,F403

# Avoid thrashing expensive SingleArticulation.initialize() when
# RMPflow briefly fails against foliage/beds.
_LAST_ARTICULATION_REINIT_T = 0.0
_ARTICULATION_REINIT_COOLDOWN_S = 2.0


async def next_frame():
    await omni.kit.app.get_app().next_update_async()


async def wait_frames(count):
    for _ in range(count):
        await next_frame()


def _articulation_ready(articulation):
    ready = getattr(
        articulation,
        "is_initialized",
        None,
    )

    if callable(ready):
        try:
            return bool(ready())
        except Exception:
            return False

    if ready is None:
        # Older views may not expose the flag; treat as ready so we
        # do not re-init on every query.
        return True

    return bool(ready)


def ensure_articulations_initialized(
    jackal,
    arm,
    force=False,
):
    """
    Re-initialize articulations if a prior capture/USD change
    invalidated their PhysX tensor views.

    Rate-limited: bed/foliage RMPflow failures used to call this every
    control step and stall the run for minutes.
    """
    global _LAST_ARTICULATION_REINIT_T

    jackal_ready = _articulation_ready(jackal)
    arm_ready = _articulation_ready(arm)

    if (
        jackal_ready
        and arm_ready
        and not force
    ):
        return False

    now = time.perf_counter()

    if (
        not force
        and (now - _LAST_ARTICULATION_REINIT_T)
        < _ARTICULATION_REINIT_COOLDOWN_S
    ):
        return False

    reinited = False

    if force or not jackal_ready:
        print(
            "[ARTICULATION] Re-initializing Jackal."
        )
        jackal.initialize()
        reinited = True

    if force or not arm_ready:
        print(
            "[ARTICULATION] Re-initializing UR5e."
        )
        arm.initialize()
        reinited = True

    if reinited:
        _LAST_ARTICULATION_REINIT_T = now

    return reinited


def rebuild_articulation_motion_policy(
    arm,
    rmpflow,
):
    """
    ArticulationMotionPolicy caches the articulation handle.
    Rebuild it after any articulation re-initialize.
    """
    return mg.ArticulationMotionPolicy(
        arm,
        rmpflow,
        RMPFLOW_PHYSICS_DT,
    )


def recover_rmpflow_action(
    jackal,
    arm,
    rmpflow,
    articulation_policy,
    reinit_count,
    max_reinits=1,
):
    """
    One-shot recovery after get_next_articulation_action fails.

    Returns:
        (action_or_None, articulation_policy, reinit_count, give_up)
    """
    if reinit_count >= max_reinits:
        print(
            "[RMPFLOW] Articulation still failing after "
            f"{reinit_count} re-init(s); aborting stage early."
        )
        return (
            None,
            articulation_policy,
            reinit_count,
            True,
        )

    print(
        "[RMPFLOW] Forcing articulation re-init + policy rebuild "
        f"(attempt {reinit_count + 1}/{max_reinits})."
    )

    ensure_articulations_initialized(
        jackal,
        arm,
        force=True,
    )

    articulation_policy = (
        rebuild_articulation_motion_policy(
            arm,
            rmpflow,
        )
    )

    reinit_count += 1

    try:
        action = (
            articulation_policy
            .get_next_articulation_action(
                RMPFLOW_PHYSICS_DT
            )
        )
    except Exception as exc:
        print(
            "[RMPFLOW] Still failing after re-init: "
            f"{exc}"
        )
        return (
            None,
            articulation_policy,
            reinit_count,
            True,
        )

    return (
        action,
        articulation_policy,
        reinit_count,
        False,
    )
