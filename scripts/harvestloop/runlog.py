"""
Per-run artifact directory.

Matched fixed/active pairs share one experiment folder:

    results/runs/<experiment_id>/
        experiment.json
        fixed_view/
            images/
            trusses.csv
            trusses.json
            summary.txt
            run.json
        active_perception/
            ...

Images and metrics from different conditions never overwrite each
other, and the shared parent makes the controlled pair obvious.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import *  # noqa: F401,F403


# Suffixes used when a vector-valued record field is flattened into
# CSV columns. Anything of another length is written as a string.
_VECTOR_SUFFIXES = {
    2: ("u", "v"),
    3: ("x", "y", "z"),
    4: ("w", "qx", "qy", "qz"),
    # Arm configurations, in ACTIVE_ARM_JOINTS order.
    6: (
        "pan",
        "lift",
        "elbow",
        "wrist1",
        "wrist2",
        "wrist3",
    ),
}


def _is_vector(value):
    return isinstance(
        value,
        (list, tuple, np.ndarray),
    )


# Naming rules that fix a field's column layout even when every
# record happens to be missing it, so the CSV schema depends on the
# field names rather than on how one run went.
_ARITY_BY_SUFFIX = (
    ("_px", 2),
    ("_quat", 4),
    ("_world", 3),
    ("_cam", 3),
    ("_target", 3),
    ("_position", 3),
    ("_in_arm_base", 3),
    ("q_at_pregrasp", 6),
    ("q_at_grasp", 6),
)


def _vector_arities(records):
    """
    Component count per vector-valued field, across all records.

    A field that is None in one record (a label behind the camera, a
    stage never reached) must still produce the same columns as the
    records where it has a value, or the CSV ends up with two
    different columns for one quantity.
    """
    arities = {}

    for record in records:
        for key, value in record.items():
            if key in arities:
                continue

            if _is_vector(value):
                arities[key] = np.asarray(
                    value
                ).size
                continue

            if value is not None:
                continue

            for suffix, arity in _ARITY_BY_SUFFIX:
                if key.endswith(
                    suffix
                ):
                    arities[key] = arity
                    break

    return arities


def flatten_record(
    record,
    arities=None,
):
    """
    Turn one truss record into flat scalar CSV columns.

    Vector fields become <name>_x / <name>_y / ... so the CSV can be
    loaded by anything without parsing nested values.
    """
    arities = (
        arities
        if arities is not None
        else _vector_arities(
            [record]
        )
    )

    flat = {}

    for key, value in record.items():
        arity = arities.get(
            key
        )

        suffixes = (
            _VECTOR_SUFFIXES.get(
                arity
            )
            if arity is not None
            else None
        )

        if value is None:
            if suffixes is None:
                flat[key] = ""
            else:
                for suffix in suffixes:
                    flat[f"{key}_{suffix}"] = ""

            continue

        if not _is_vector(value):
            flat[key] = value
            continue

        # Non-numeric sequences (viewpoint names, flags) stay as one
        # semicolon-separated column instead of being cast to floats.
        try:
            components = np.asarray(
                value,
                dtype=np.float64,
            ).ravel()
        except (TypeError, ValueError):
            flat[key] = ";".join(
                str(
                    item
                )
                for item in value
            )
            continue

        if suffixes is None:
            flat[key] = " ".join(
                f"{component:.6f}"
                for component in components
            )
            continue

        for suffix, component in zip(
            suffixes,
            components,
        ):
            flat[f"{key}_{suffix}"] = float(
                component
            )

    return flat


def _jsonable(value):
    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            float(component)
            for component in value.ravel()
        ]

    if isinstance(
        value,
        (np.floating, np.integer),
    ):
        return value.item()

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _jsonable(item)
            for item in value
        ]

    return value


def _config_snapshot():
    """
    The tunables that change what a run measures.

    Two runs are only comparable if these match, so they travel with
    the results instead of living only in the source tree.
    """
    return {
        # Wall-clock only: neither changes what is computed, but a
        # run's timings are meaningless without knowing them.
        "unthrottle_run_loop": UNTHROTTLE_RUN_LOOP,
        "gate_wrist_render_product": GATE_WRIST_RENDER_PRODUCT,
        "freeze_viewport_during_route": (
            FREEZE_VIEWPORT_DURING_ROUTE
        ),
        "estimator_mode": ESTIMATOR_MODE,
        "run_condition_label": RUN_CONDITION_LABEL,
        "run_experiment_id": RUN_EXPERIMENT_ID,
        "scene_seed": SCENE_SEED,
        "occlusion_level": OCCLUSION_LEVEL,
        "max_paired_targets": MAX_PAIRED_TARGETS,
        "pair_view1_sanity": PAIR_VIEW1_SANITY,
        "pair_state_restore": PAIR_STATE_RESTORE,
        "active_perception_max_views": ACTIVE_PERCEPTION_MAX_VIEWS,
        "active_view_lateral_offset_m": (
            ACTIVE_VIEW_LATERAL_OFFSET_M
        ),
        "active_viewpoint_sequence": list(
            ACTIVE_VIEWPOINT_SEQUENCE
        ),
        "keypoint_cut_offset_m": KEYPOINT_CUT_OFFSET_M,
        "keypoint_spacing_tol_m": KEYPOINT_SPACING_TOL_M,
        "keypoint_min_confidence": KEYPOINT_MIN_CONFIDENCE,
        "keypoint_min_single_confidence": (
            KEYPOINT_MIN_SINGLE_CONFIDENCE
        ),
        "view_camera_standoff_m": VIEW_CAMERA_STANDOFF,
        "view_aim_center_fraction": VIEW_AIM_CENTER_FRACTION,
        "view_aim_max_attempts": VIEW_AIM_MAX_ATTEMPTS,
        "view_offset_m": VIEW_OFFSET,
        "view_tolerance_m": VIEW_TOLERANCE,
        "view_accept_tolerance_m": VIEW_ACCEPT_TOLERANCE,
        "pregrasp_offset_m": PREGRASP_OFFSET,
        "grasp_tolerance_m": GRASP_TOLERANCE,
        "cutter_tolerance_m": CUTTER_TOLERANCE,
        "postcut_retreat_tolerance_m": (
            POSTCUT_RETREAT_TOLERANCE
        ),
        "postcut_retreat_max_attempts": (
            POSTCUT_RETREAT_MAX_ATTEMPTS
        ),
        "wrist_capture_resolution": list(
            WRIST_CAPTURE_RESOLUTION
        ),
        "ur5e_home_q": _jsonable(
            UR5E_HOME_Q
        ),
        "ur5e_upright_q": _jsonable(
            UR5E_UPRIGHT_Q
        ),
    }


class RunArtifacts:
    """One condition subfolder under a shared experiment directory."""

    def __init__(
        self,
        condition=None,
        experiment_id=None,
        root=None,
    ):
        self.condition = (
            condition
            or RUN_CONDITION_LABEL
        )

        self.experiment_id = (
            experiment_id
            or RUN_EXPERIMENT_ID
        )

        if not self.experiment_id:
            self.experiment_id = (
                datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

        self.started_at = datetime.now()

        # Stable id used in CSV / run.json; path-like so nested
        # layout stays recoverable from a single string.
        self.run_id = (
            f"{self.experiment_id}"
            f"/{self.condition}"
        )

        artifacts_root = Path(
            root
            or RUN_ARTIFACTS_ROOT
        )

        self.experiment_dir = (
            artifacts_root
            / self.experiment_id
        )

        self.run_dir = (
            self.experiment_dir
            / self.condition
        )

        if self.run_dir.exists() and any(
            self.run_dir.iterdir()
        ):
            print(
                "[WARN] Reusing non-empty condition dir "
                f"{self.run_dir} — previous files may be mixed."
            )

        self.images_dir = (
            self.run_dir
            / "images"
        )

        self.overlay_dir = (
            self.run_dir
            / "keypoints_overlay"
        )

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.images_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if SHOW_KEYPOINT_OVERLAY:
            self.overlay_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._write_experiment_stub()

        print()
        print("========================================")
        print("[RUN ARTIFACTS]")
        print("========================================")

        print(
            f"Experiment: {self.experiment_id}"
        )

        print(
            f"Condition:  {self.condition}"
        )

        print(
            f"Run id:     {self.run_id}"
        )

        print(
            f"Directory:  {self.run_dir}"
        )

        if SHOW_KEYPOINT_OVERLAY:
            print(
                f"Keypoint overlays: {self.overlay_dir}"
            )

    def _write_experiment_stub(
        self,
    ):
        """Parent-folder marker so the next condition can find us."""
        path = (
            self.experiment_dir
            / "experiment.json"
        )

        payload = {
            "experiment_id": self.experiment_id,
            "scene_seed": SCENE_SEED,
            "occlusion_level": OCCLUSION_LEVEL,
            "max_targets": MAX_PAIRED_TARGETS,
            "gt_scoring_only": True,
            "fixed_views": 1,
            "active_max_views": ACTIVE_PERCEPTION_MAX_VIEWS,
            "protocol": "shared_view_rescue",
            "pair_state_restore": False,
            "conditions": {},
        }

        if path.exists():
            try:
                payload = json.loads(
                    path.read_text()
                )
            except json.JSONDecodeError:
                pass

        payload[
            "experiment_id"
        ] = self.experiment_id
        payload.setdefault(
            "scene_seed",
            SCENE_SEED,
        )
        payload.setdefault(
            "occlusion_level",
            OCCLUSION_LEVEL,
        )
        payload.setdefault(
            "gt_scoring_only",
            True,
        )
        payload.setdefault(
            "protocol",
            "rebuild_between_conditions",
        )

        conditions = payload.setdefault(
            "conditions",
            {},
        )

        conditions[
            self.condition
        ] = {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(
                timespec="seconds"
            ),
            "path": self.condition,
        }

        path.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n"
        )

    def relative_image(
        self,
        image_path,
    ):
        """Image path as stored in the records (run-relative)."""
        try:
            return str(
                Path(
                    image_path
                ).relative_to(
                    self.run_dir
                )
            )
        except ValueError:
            return str(
                image_path
            )

    def write(
        self,
        harvest_log,
        stops_planned=None,
        stops_visited=None,
        note="",
        timing=None,
    ):
        """
        Write every artifact for this run.

        Safe to call from a `finally`: a run that aborted mid-route
        still leaves its images, records and summary behind.
        """
        harvest_log.finalize_pending(
            "Run ended before this truss reached an outcome."
        )

        records = [
            {
                "run_id": self.run_id,
                "experiment_id": self.experiment_id,
                "condition": self.condition,
                **record,
            }
            for record in harvest_log.records
        ]

        summary_lines = (
            harvest_log.summary_lines()
        )

        # Where the wall clock went, kept next to the outcomes: a
        # speed change is only acceptable if these shrink while the
        # outcomes above stay put.
        if timing is not None:
            summary_lines = (
                summary_lines
                + [""]
                + timing.summary_lines()
            )

        summary_path = (
            self.run_dir
            / "summary.txt"
        )

        summary_path.write_text(
            "\n".join(
                summary_lines
            )
            + "\n"
        )

        json_path = (
            self.run_dir
            / "trusses.json"
        )

        json_path.write_text(
            json.dumps(
                [
                    {
                        key: _jsonable(value)
                        for key, value in record.items()
                    }
                    for record in records
                ],
                indent=2,
            )
            + "\n"
        )

        csv_path = (
            self.run_dir
            / "trusses.csv"
        )

        arities = _vector_arities(
            records
        )

        flat_records = [
            flatten_record(
                record,
                arities,
            )
            for record in records
        ]

        # Records differ in which stages they reached, so the header
        # is the union of all columns in first-seen order.
        columns = []

        for flat in flat_records:
            for key in flat:
                if key not in columns:
                    columns.append(
                        key
                    )

        with csv_path.open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
            )

            writer.writeheader()

            for flat in flat_records:
                writer.writerow(
                    flat
                )

        metadata_path = (
            self.run_dir
            / "run.json"
        )

        metadata_path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "experiment_id": self.experiment_id,
                    "condition": self.condition,
                    "started_at": (
                        self.started_at.isoformat(
                            timespec="seconds"
                        )
                    ),
                    "finished_at": (
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    ),
                    "stops_planned": stops_planned,
                    "stops_visited": stops_visited,
                    "trusses_attempted": len(
                        records
                    ),
                    "perception_successes": sum(
                        1 for record in records
                        if harvest_log.perception_ok(
                            record
                        )
                    ),
                    "end_to_end_successes": sum(
                        1 for record in records
                        if harvest_log.end_to_end_ok(
                            record
                        )
                    ),
                    "note": note,
                    "route_events": harvest_log.events,
                    "wall_clock_s": (
                        round(
                            timing.elapsed(),
                            1,
                        )
                        if timing is not None
                        else None
                    ),
                    "stage_seconds": (
                        {
                            stage: round(
                                seconds,
                                1,
                            )
                            for stage, seconds
                            in timing.totals.items()
                        }
                        if timing is not None
                        else None
                    ),
                    "config": _config_snapshot(),
                },
                indent=2,
            )
            + "\n"
        )

        print()
        print("========================================")
        print("[RUN ARTIFACTS WRITTEN]")
        print("========================================")

        print(
            f"Records: {len(records)}"
        )

        print(
            f"CSV:     {csv_path}"
        )

        print(
            f"JSON:    {json_path}"
        )

        print(
            f"Summary: {summary_path}"
        )

        print(
            f"Images:  {self.images_dir}"
        )

        return self.run_dir
