# ============================================================
# harvest_aisle1_route_viewpoint_test.py
#
# Entry point for the full aisle-1 harvesting route:
# drive -> viewpoint capture -> pregrasp -> grasp -> cut ->
# carry -> basket release, for every truss of every stem pair.
#
# The implementation lives in the harvestloop package; this file
# only refreshes the modules (so edits take effect without an
# Isaac restart) and schedules the route on the Kit event loop.
#
# Run from the Isaac Script Editor via run_aisle1_viewpoint.py.
#
# Condition / experiment (optional):
#   CONDITION = "fixed_view"|"active_perception"
#   EXPERIMENT = "<YYYYMMDD_HHMMSS>"   # shared pair folder
#   --condition / HARVESTLOOP_CONDITION
#   --experiment / HARVESTLOOP_EXPERIMENT
# ============================================================

import asyncio
import importlib
import sys

SCRIPTS_DIR = "/home/charlotte/harvestloop_sim/scripts"

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import harvest_config
import usd_utils

importlib.reload(harvest_config)
importlib.reload(usd_utils)

# Drop cached submodules so a re-run always picks up edited sources.
for _name in [
    _m for _m in sys.modules
    if _m == "harvestloop" or _m.startswith("harvestloop.")
]:
    del sys.modules[_name]

from harvestloop.route import main


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


_condition = _optional_str(
    "CONDITION"
)
_experiment = _optional_str(
    "EXPERIMENT"
)

print(
    f"[LAUNCH] EXPERIMENT={_experiment!r} "
    f"CONDITION={_condition!r}"
)

asyncio.ensure_future(
    main(
        condition=_condition,
        experiment=_experiment,
    )
)
