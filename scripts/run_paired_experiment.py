# DEPRECATED for the perception experiment.
# Prefer scripts/run_shared_view_experiment.py:
#   one trial per truss, shared canonical RGB-D, in-place rescue.
#
# This launcher still runs the older rebuild-between-conditions
# fixed then active full routes (optional later e2e policy compare).
#
# Artifacts:
#   results/runs/<experiment_id>/
#       experiment.json
#       targets.json
#       fixed_view/
#       active_perception/
#
# Preferred paste (runs BOTH conditions, deterministic rebuild):
#   MAX_TARGETS = 16
#   OCCLUSION = "medium"
#   SEED = 42
#   exec(open("/home/charlotte/harvestloop_sim/scripts/run_paired_experiment.py", encoding="utf-8").read(), globals())
#
# Active view-1 fingerprint restore is OFF by default
# (PAIR_STATE_RESTORE=False). Paired equality comes from:
# same seed + occlusion rebuild, locked targets.json, and a
# full aisle re-drive for both conditions.
#
# For LOW/MEDIUM/HIGH in one session use run_occlusion_suite.py.
# Resume active only (after fixed already finished):
#   CONDITION = "active_perception"
#   EXPERIMENT = "<id from fixed run>"
#   MAX_TARGETS = 16
#   OCCLUSION = "medium"
#   SEED = 42
#   exec(open("/home/charlotte/harvestloop_sim/scripts/run_paired_experiment.py", encoding="utf-8").read(), globals())

if "CONDITION" not in globals():
    CONDITION = None

if "EXPERIMENT" not in globals():
    EXPERIMENT = None

if "MAX_TARGETS" not in globals():
    MAX_TARGETS = 16

if "OCCLUSION" not in globals():
    OCCLUSION = "medium"

if "SEED" not in globals():
    SEED = 42

# When CONDITION is unset, run fixed then active automatically.
if "RUN_BOTH" not in globals():
    RUN_BOTH = CONDITION is None

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
    RUN_ARTIFACTS_ROOT,
    RUN_CONDITION_CHOICES,
    apply_scene_identity,
    resolve_run_condition,
    resolve_run_experiment,
)
from harvestloop.route import main
from harvestloop.simctl import wait_frames
from scenegen.builder import rebuild_greenhouse


def _optional_str(name):
    value = globals().get(
        name
    )

    if isinstance(
        value,
        str,
    ):
        value = value.strip() or None

    return value


def _first_missing_condition(
    experiment_id,
):
    if experiment_id is None:
        return "fixed_view"

    root = (
        RUN_ARTIFACTS_ROOT
        / experiment_id
    )

    for condition in RUN_CONDITION_CHOICES:
        if not (
            root
            / condition
            / "trusses.csv"
        ).exists():
            return condition

    return None


async def _stop_sim_for_rebuild():
    """
    Physics must be stopped before RemovePrim on the greenhouse.
    Leaving the timeline playing after fixed_view is what killed the
    handoff to active_perception.
    """
    import omni.timeline

    timeline = (
        omni.timeline
        .get_timeline_interface()
    )

    if timeline.is_playing():
        print(
            "[PAIR] Stopping timeline before greenhouse rebuild..."
        )
        timeline.stop()

    await wait_frames(
        30
    )


async def _rebuild_for_pair(
    seed,
    occlusion,
    label,
):
    print(
        f"[PAIR] {label}"
    )
    await _stop_sim_for_rebuild()

    try:
        rebuild_greenhouse(
            seed=seed,
            occlusion=occlusion,
        )
    except Exception:
        print(
            "[PAIR] Greenhouse rebuild FAILED:"
        )
        traceback.print_exc()
        raise

    await wait_frames(
        45
    )
    print(
        "[PAIR] Greenhouse rebuild settled."
    )


def _on_pair_task_done(
    task,
):
    """Surface async failures that Script Editor otherwise swallows."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        print(
            "[PAIR] Task cancelled."
        )
        return

    if exc is not None:
        print(
            "[PAIR] FATAL — chain stopped:"
        )
        traceback.print_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )


_condition = _optional_str(
    "CONDITION"
)
_experiment = _optional_str(
    "EXPERIMENT"
)
_occlusion = _optional_str(
    "OCCLUSION"
) or "medium"
_seed = globals().get(
    "SEED",
    42,
)
_max_targets = globals().get(
    "MAX_TARGETS",
    16,
)
_run_both = bool(
    globals().get(
        "RUN_BOTH",
        _condition is None,
    )
)

if _occlusion not in OCCLUSION_CHOICES:
    raise ValueError(
        f"OCCLUSION must be one of {OCCLUSION_CHOICES}"
    )

identity = apply_scene_identity(
    seed=_seed,
    occlusion=_occlusion,
    max_targets=_max_targets,
)


async def _run_paired_both():
    experiment_id = resolve_run_experiment(
        explicit=_experiment,
        condition="fixed_view",
    )

    print(
        "[PAIR LAUNCH] BOTH conditions | "
        f"EXPERIMENT={experiment_id!r} "
        f"SEED={identity['scene_seed']!r} "
        f"OCCLUSION={identity['occlusion_level']!r} "
        f"MAX_TARGETS={identity['max_paired_targets']!r}"
    )

    try:
        await _rebuild_for_pair(
            identity["scene_seed"],
            identity["occlusion_level"],
            "Rebuild before fixed_view",
        )

        print(
            "[PAIR] === fixed_view ==="
        )
        await main(
            condition="fixed_view",
            experiment=experiment_id,
        )
        print(
            "[PAIR] fixed_view finished OK"
        )

        await _rebuild_for_pair(
            identity["scene_seed"],
            identity["occlusion_level"],
            "Rebuild before active_perception "
            "(same seed/occlusion)",
        )

        print(
            "[PAIR] === active_perception ==="
        )
        await main(
            condition="active_perception",
            experiment=experiment_id,
        )
        print(
            "[PAIR] active_perception finished OK"
        )

        print(
            f"[PAIR DONE] {experiment_id} | "
            "score with: python3 scripts/analyze_paired_benchmark.py "
            f"{experiment_id}"
        )
    except Exception:
        print(
            "[PAIR] Chain aborted after an error "
            f"(experiment={experiment_id!r})."
        )
        traceback.print_exc()

        fixed_csv = (
            RUN_ARTIFACTS_ROOT
            / experiment_id
            / "fixed_view"
            / "trusses.csv"
        )

        if fixed_csv.exists():
            print(
                "[PAIR] fixed_view artifacts exist. Resume with:\n"
                "  CONDITION = \"active_perception\"\n"
                f"  EXPERIMENT = \"{experiment_id}\"\n"
                f"  MAX_TARGETS = {identity['max_paired_targets']!r}\n"
                f"  OCCLUSION = \"{identity['occlusion_level']}\"\n"
                f"  SEED = {identity['scene_seed']}\n"
                "  exec(open(\"/home/charlotte/harvestloop_sim/scripts/"
                "run_paired_experiment.py\", encoding=\"utf-8\").read(), "
                "globals())"
            )

        raise


async def _run_one():
    condition = _condition

    if condition is None:
        condition = _first_missing_condition(
            _experiment
        )

    if condition is None:
        raise RuntimeError(
            "Both conditions already have trusses.csv under "
            f"{_experiment}. Set EXPERIMENT to a new id "
            "or RUN_BOTH = True."
        )

    resolve_run_condition(
        explicit=condition
    )

    experiment_id = _experiment

    if experiment_id is None:
        experiment_id = resolve_run_experiment(
            condition=condition
        )

    print(
        "[PAIR LAUNCH] "
        f"EXPERIMENT={experiment_id!r} "
        f"CONDITION={condition!r} "
        f"SEED={identity['scene_seed']!r} "
        f"OCCLUSION={identity['occlusion_level']!r} "
        f"MAX_TARGETS={identity['max_paired_targets']!r}"
    )

    try:
        # Always restore plants before a condition that needs a
        # fresh greenhouse (active after fixed, or a lone fixed).
        await _rebuild_for_pair(
            identity["scene_seed"],
            identity["occlusion_level"],
            f"Rebuild before {condition}",
        )

        await main(
            condition=condition,
            experiment=experiment_id,
        )
        print(
            f"[PAIR] {condition} finished OK"
        )
    except Exception:
        print(
            f"[PAIR] {condition} aborted "
            f"(experiment={experiment_id!r})."
        )
        traceback.print_exc()
        raise


_task = asyncio.ensure_future(
    _run_paired_both()
    if _run_both
    else _run_one()
)
_task.add_done_callback(
    _on_pair_task_done
)
