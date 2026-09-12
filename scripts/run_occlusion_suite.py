# ============================================================
# run_occlusion_suite.py
#
# Strictly matched LOW / MEDIUM / HIGH shared-view rescue sweep.
#
# For one seed:
#   1) rebuild once and lock targets.json (same truss paths, stops,
#      order, count)
#   2) for each occlusion level: copy that exact manifest, rebuild
#      foliage only, run shared_view (canonical + rescue in one trial)
#
# Estimator, active-view policy (max 3), grasp/cut controller, and
# thresholds are unchanged.
#
# Script Editor — one seed:
#   SEED = 42
#   MAX_TARGETS = 16
#   LEVELS = ("low", "medium", "high")
#   exec(open("/home/charlotte/harvestloop_sim/scripts/run_occlusion_suite.py", encoding="utf-8").read(), globals())
#
# Resume / continue an existing suite (reuse locked targets.json):
#   SUITE = "20260912_095224__s7"
#   SEED = 7
#   LEVELS = ("low", "medium", "high")   # or only remaining, e.g. ("medium", "high")
#   exec(open("/home/charlotte/harvestloop_sim/scripts/run_occlusion_suite.py", encoding="utf-8").read(), globals())
#
# Script Editor — then 2–3 more seeds (no algorithm tuning):
#   SEEDS = (42, 7, 99)
#   MAX_TARGETS = 16
#   exec(open("/home/charlotte/harvestloop_sim/scripts/run_occlusion_suite.py", encoding="utf-8").read(), globals())
#
# Score (outside Isaac):
#   python3 scripts/analyze_occlusion_suite.py <suite_id>
# ============================================================

if "SEED" not in globals():
    SEED = 42

if "SEEDS" not in globals():
    SEEDS = None

if "MAX_TARGETS" not in globals():
    MAX_TARGETS = 16

if "LEVELS" not in globals():
    LEVELS = (
        "low",
        "medium",
        "high",
    )

if "SUITE" not in globals():
    SUITE = None

# Occlusion used only to discover the locked stem/truss topology.
# Foliage differs per level; structure RNG is isolation-safe.
if "MANIFEST_OCCLUSION" not in globals():
    MANIFEST_OCCLUSION = "medium"

# If True (default), reuse suite/targets.json when present and skip
# finished levels that already have a clean shared_view run.
if "RESUME" not in globals():
    RESUME = True

import asyncio
import importlib
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import omni.usd

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
    apply_scene_identity,
)
from harvestloop.route import main
from harvestloop.routing import build_target_manifest_from_stage
from harvestloop.simctl import wait_frames
from scenegen.builder import rebuild_greenhouse

SUITES_ROOT = (
    Path(SCRIPTS_DIR).resolve().parents[0]
    / "results"
    / "suites"
)


def _optional_str(name):
    value = globals().get(name)

    if isinstance(value, str):
        value = value.strip() or None

    return value


_max_targets = globals().get("MAX_TARGETS", 16)
_levels = tuple(
    str(level).strip().lower()
    for level in globals().get(
        "LEVELS",
        ("low", "medium", "high"),
    )
)
_suite_prefix = _optional_str("SUITE")
_manifest_occlusion = (
    _optional_str("MANIFEST_OCCLUSION")
    or "medium"
)
_resume = bool(globals().get("RESUME", True))


_seeds_raw = globals().get("SEEDS", None)

if _seeds_raw is None:
    _seeds = (int(globals().get("SEED", 42)),)
else:
    _seeds = tuple(int(s) for s in _seeds_raw)

for level in _levels:
    if level not in OCCLUSION_CHOICES:
        raise ValueError(
            f"Unknown occlusion {level!r}; expected {OCCLUSION_CHOICES}"
        )

if _manifest_occlusion not in OCCLUSION_CHOICES:
    raise ValueError(
        f"MANIFEST_OCCLUSION must be one of {OCCLUSION_CHOICES}"
    )


async def _stop_sim_for_rebuild():
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()

    if timeline.is_playing():
        print("[SUITE] Stopping timeline before rebuild...")
        timeline.stop()

    await wait_frames(30)


async def _rebuild(seed, occlusion, label):
    print(f"[SUITE] {label}")
    await _stop_sim_for_rebuild()
    rebuild_greenhouse(seed=seed, occlusion=occlusion)
    await wait_frames(45)
    print(
        f"[SUITE] Rebuild settled | seed={seed} "
        f"occlusion={occlusion}"
    )


def _write_suite_json(suite_id, seed, experiments, locked):
    SUITES_ROOT.mkdir(parents=True, exist_ok=True)
    path = SUITES_ROOT / suite_id / "suite.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "suite_id": suite_id,
        "scene_seed": int(seed),
        "max_targets": _max_targets,
        "levels": list(_levels),
        "protocol": (
            "shared_view_rescue_matched_manifest"
        ),
        "gt_scoring_only": True,
        "matched_targets": True,
        "locked_target_count": (
            None
            if locked is None
            else len(locked.get("targets", []))
        ),
        "locked_truss_paths": (
            None
            if locked is None
            else list(locked.get("truss_paths", []))
        ),
        "experiments": experiments,
    }
    path.write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return path


def _install_manifest(suite_id, experiment_id):
    src = SUITES_ROOT / suite_id / "targets.json"
    dst_dir = RUN_ARTIFACTS_ROOT / experiment_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "targets.json"

    if not src.exists():
        raise FileNotFoundError(
            f"Missing suite-locked targets at {src}"
        )

    shutil.copy2(src, dst)
    print(f"[SUITE] Installed locked manifest -> {dst}")
    return dst


def _load_locked(suite_id):
    path = SUITES_ROOT / suite_id / "targets.json"

    if not path.exists():
        return None

    return json.loads(path.read_text())


def _load_suite_experiments(suite_id):
    path = SUITES_ROOT / suite_id / "suite.json"

    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}

    experiments = payload.get("experiments") or {}
    return dict(experiments)


def _level_already_done(suite_id, level):
    experiment_id = f"{suite_id}__{level}"
    run_json = (
        RUN_ARTIFACTS_ROOT
        / experiment_id
        / "shared_view"
        / "run.json"
    )
    trusses = (
        RUN_ARTIFACTS_ROOT
        / experiment_id
        / "shared_view"
        / "trusses.csv"
    )

    if not run_json.exists() or not trusses.exists():
        return False

    payload = json.loads(run_json.read_text())
    note = str(payload.get("note") or "").strip()

    if note:
        return False

    # Count only rows with a real truss_path (drop ghost log lines).
    import csv

    with trusses.open(newline="") as handle:
        valid_n = sum(
            1
            for row in csv.DictReader(handle)
            if str(row.get("truss_path") or "").strip()
        )

    locked = _load_locked(suite_id)
    expected = (
        len(locked.get("targets", []))
        if locked is not None
        else None
    )

    if expected is not None and valid_n != expected:
        print(
            f"[SUITE] RESUME: {level} has {valid_n} valid "
            f"targets, expected {expected}; will re-run."
        )
        return False

    return valid_n > 0


async def _lock_manifest(suite_id, seed):
    """
    Rebuild once, discover aisle-1 targets, write suite/targets.json.
    Same paths/stops/order/count are reused for every level.
    """
    existing = _load_locked(suite_id)

    if _resume and existing is not None:
        n = len(existing.get("targets", []))
        print(
            f"[SUITE] RESUME: reusing locked {n} targets at "
            f"{SUITES_ROOT / suite_id / 'targets.json'}"
        )
        return existing

    await _rebuild(
        seed,
        _manifest_occlusion,
        (
            f"Manifest discovery rebuild "
            f"(seed={seed}, occlusion={_manifest_occlusion})"
        ),
    )

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError(
            "No USD stage after manifest rebuild."
        )

    locked = build_target_manifest_from_stage(
        stage,
        scene_seed=seed,
        occlusion_level=_manifest_occlusion,
        max_targets=_max_targets,
        suite_id=suite_id,
    )

    n = len(locked.get("targets", []))

    if n <= 0:
        raise RuntimeError(
            "Manifest discovery found zero targets."
        )

    suite_dir = SUITES_ROOT / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    path = suite_dir / "targets.json"
    path.write_text(
        json.dumps(locked, indent=2) + "\n"
    )

    print(
        f"[SUITE] Locked {n} matched targets at {path}"
    )

    for item in locked["targets"]:
        print(
            f"  stop_{int(item['stop_index']):02d} "
            f"{item['side']} {item['truss_path']}"
        )

    return locked


async def _run_level(suite_id, seed, level):
    experiment_id = f"{suite_id}__{level}"
    apply_scene_identity(
        seed=seed,
        occlusion=level,
        max_targets=_max_targets,
    )
    _install_manifest(suite_id, experiment_id)

    await _rebuild(
        seed,
        level,
        f"Rebuild before shared_view ({level})",
    )

    # Re-install after rebuild in case anything wiped the run dir.
    _install_manifest(suite_id, experiment_id)

    print(
        f"[SUITE] === {experiment_id} / shared_view ==="
    )
    await main(
        condition="shared_view",
        experiment=experiment_id,
    )
    print(
        f"[SUITE] Level {level} finished OK -> {experiment_id}"
    )
    return experiment_id


async def _run_one_suite(seed, suite_id):
    experiments = _load_suite_experiments(suite_id)
    locked = None

    print(
        "[SUITE LAUNCH] "
        f"id={suite_id!r} seed={seed} "
        f"max_targets={_max_targets} "
        f"levels={list(_levels)} resume={_resume} | "
        "protocol=shared_view_rescue_matched_manifest"
    )
    _write_suite_json(
        suite_id,
        seed,
        experiments,
        locked,
    )

    try:
        locked = await _lock_manifest(
            suite_id,
            seed,
        )
        _write_suite_json(
            suite_id,
            seed,
            experiments,
            locked,
        )

        for level in _levels:
            exp_id = f"{suite_id}__{level}"

            if _resume and _level_already_done(
                suite_id,
                level,
            ):
                print(
                    f"[SUITE] RESUME: skip finished {level} "
                    f"({exp_id})"
                )
                experiments[level] = exp_id
                _write_suite_json(
                    suite_id,
                    seed,
                    experiments,
                    locked,
                )
                continue

            exp_id = await _run_level(
                suite_id,
                seed,
                level,
            )
            experiments[level] = exp_id

            cur = json.loads(
                (
                    RUN_ARTIFACTS_ROOT
                    / exp_id
                    / "targets.json"
                ).read_text()
            )
            n = len(cur.get("targets", []))
            locked_n = len(
                locked.get("targets", [])
            )

            if n != locked_n:
                raise RuntimeError(
                    f"Target count drift at {level}: "
                    f"{n} != locked {locked_n}"
                )

            locked_paths = locked.get(
                "truss_paths"
            ) or [
                item["truss_path"]
                for item in locked.get(
                    "targets",
                    [],
                )
            ]
            cur_paths = [
                item["truss_path"]
                for item in cur.get(
                    "targets",
                    [],
                )
            ]

            if cur_paths != locked_paths:
                raise RuntimeError(
                    f"Target path drift at {level}."
                )

            _write_suite_json(
                suite_id,
                seed,
                experiments,
                locked,
            )

        print(
            f"[SUITE DONE] {suite_id}\n"
            "Score with:\n"
            f"  python3 scripts/analyze_occlusion_suite.py {suite_id}"
        )
        return suite_id
    except Exception:
        print(f"[SUITE] Aborted (suite={suite_id!r}).")
        traceback.print_exc()
        _write_suite_json(
            suite_id,
            seed,
            experiments,
            locked,
        )
        raise


async def _run_all():
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    finished = []

    def _suite_id_for_seed(seed):
        """
        Always keep a per-seed suite id: <batch>__s{seed}.

        Avoid double tags like ...__s7__s7 when SUITE already ends
        with __s{seed}. Never leave a bare batch root for a single
        SEED (that produced 20260912_095224 for seed 99).
        """
        import re

        if not _suite_prefix:
            if len(_seeds) == 1:
                return f"{stamp}__s{seed}"
            return f"{stamp}__s{seed}"

        prefix = _suite_prefix

        if prefix.endswith(f"__s{seed}"):
            return prefix

        batch = re.sub(r"__s\d+$", "", prefix)
        return f"{batch}__s{seed}"

    for index, seed in enumerate(_seeds):
        suite_id = _suite_id_for_seed(seed)

        print()
        print("=" * 60)
        print(
            f"[SUITE BATCH] seed {index + 1}/"
            f"{len(_seeds)} -> {suite_id}"
        )
        print("=" * 60)

        finished.append(
            await _run_one_suite(
                seed,
                suite_id,
            )
        )

    print()
    print("[SUITE BATCH DONE]")

    for suite_id in finished:
        print(
            f"  python3 scripts/analyze_occlusion_suite.py {suite_id}"
        )


def _on_done(task):
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        print("[SUITE] Cancelled.")
        return

    if exc is not None:
        print("[SUITE] FATAL:")
        traceback.print_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )


_task = asyncio.ensure_future(_run_all())
_task.add_done_callback(_on_done)
