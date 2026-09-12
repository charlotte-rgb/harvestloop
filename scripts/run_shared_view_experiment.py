# Shared-view rescue experiment (Script Editor).
#
# One trial per truss:
#   1) move to canonical wrist pose
#   2) capture view 1 ONCE  → FIXED-VIEW result
#   3) estimate
#   4) if needs_another_view / view-1 fails → views 2–3 in the SAME trial
#   5) select best estimate → ACTIVE / rescue result
#
# Fixed and active share the exact same first RGB-D capture.
# No rebuild between fixed and active. Estimator + tolerances unchanged.
#
# Artifacts:
#   results/runs/<experiment_id>/shared_view/
#
# Paste:
#   MAX_TARGETS = 16
#   OCCLUSION = "medium"
#   SEED = 42
#   exec(open("/home/charlotte/harvestloop_sim/scripts/run_shared_view_experiment.py", encoding="utf-8").read(), globals())
#
# Score:
#   python3 scripts/analyze_rescue_benchmark.py <experiment_id>

if "EXPERIMENT" not in globals():
    EXPERIMENT = None

if "MAX_TARGETS" not in globals():
    MAX_TARGETS = 16

if "OCCLUSION" not in globals():
    OCCLUSION = "medium"

if "SEED" not in globals():
    SEED = 42

if "REBUILD_FIRST" not in globals():
    REBUILD_FIRST = True

import asyncio
import importlib
import sys
import traceback

SCRIPTS_DIR = "/home/charlotte/harvestloop_sim/scripts"

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import harvest_config
import usd_utils

importlib.reload(harvest_config)
importlib.reload(usd_utils)

for _name in [
    _m for _m in sys.modules
    if (
        _m == "harvestloop"
        or _m.startswith("harvestloop.")
        or _m == "scenegen"
        or _m.startswith("scenegen.")
    )
]:
    del sys.modules[_name]

from harvestloop.config import (
    OCCLUSION_CHOICES,
    apply_scene_identity,
    resolve_run_experiment,
)
from harvestloop.route import main
from harvestloop.simctl import wait_frames
from scenegen.builder import rebuild_greenhouse


def _optional_str(name):
    value = globals().get(name)

    if isinstance(value, str):
        value = value.strip() or None

    return value


_experiment = _optional_str("EXPERIMENT")
_occlusion = _optional_str("OCCLUSION") or "medium"
_seed = globals().get("SEED", 42)
_max_targets = globals().get("MAX_TARGETS", 16)
_rebuild_first = bool(globals().get("REBUILD_FIRST", True))

if _occlusion not in OCCLUSION_CHOICES:
    raise ValueError(
        f"OCCLUSION must be one of {OCCLUSION_CHOICES}"
    )

identity = apply_scene_identity(
    seed=_seed,
    occlusion=_occlusion,
    max_targets=_max_targets,
)


async def _stop_sim_for_rebuild():
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()

    if timeline.is_playing():
        print("[SHARED] Stopping timeline before rebuild...")
        timeline.stop()

    await wait_frames(30)


async def _run():
    experiment_id = resolve_run_experiment(
        explicit=_experiment,
        condition="shared_view",
    )

    print(
        "[SHARED VIEW LAUNCH] "
        f"EXPERIMENT={experiment_id!r} "
        f"SEED={identity['scene_seed']!r} "
        f"OCCLUSION={identity['occlusion_level']!r} "
        f"MAX_TARGETS={identity['max_paired_targets']!r} | "
        "protocol=shared_view_rescue (one trial, shared capture)"
    )

    try:
        if _rebuild_first:
            await _stop_sim_for_rebuild()
            rebuild_greenhouse(
                seed=identity["scene_seed"],
                occlusion=identity["occlusion_level"],
            )
            await wait_frames(45)
            print("[SHARED] Greenhouse ready.")

        await main(
            condition="shared_view",
            experiment=experiment_id,
        )

        print(
            f"[SHARED DONE] {experiment_id} | "
            "score with: python3 scripts/analyze_rescue_benchmark.py "
            f"{experiment_id}"
        )
    except Exception:
        print("[SHARED] Aborted:")
        traceback.print_exc()
        raise


def _on_done(task):
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        print("[SHARED] Task cancelled.")
        return

    if exc is not None:
        print("[SHARED] FATAL:")
        traceback.print_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )


_task = asyncio.ensure_future(_run())
_task.add_done_callback(_on_done)
