"""
Stage-tagged harvest outcomes and the two success metrics.

Two separate metrics are reported, because a controller failure after
CUT must not be counted against perception:

    1. Perception / manipulation success
       viewpoint reached, GraspTip <= 3 cm, CutterTip <= 4 cm, CUT
       declared. This is the metric that compares fixed-view against
       active perception.

    2. End-to-end harvest success
       the above, plus post-CUT retreat within its unchanged
       tolerance, upright recovery with payload, and basket release.

Every failure is tagged with the stage it happened in so the two can
be told apart afterwards.

Each attempted truss also accumulates one record: where it was, what
the wrist camera saw, the privileged grasp/cut ground truth in world
and camera frames, and the error the arm actually achieved at every
stage it reached. `runlog.RunArtifacts` writes those records out.
"""

from .config import RUN_CONDITION_LABEL

# Arm staging before the truss was ever attempted: the SAFE UPRIGHT
# holds taken before the viewpoint and before the manipulation. Kept
# apart from STAGE_UPRIGHT, which is the post-CUT return carrying the
# payload, because a failure here means nothing was attempted yet.
STAGE_SETUP = "SETUP"

STAGE_PERCEPTION = "PERCEPTION"
STAGE_GRASP = "GRASP"
STAGE_CUT = "CUT"
STAGE_RETREAT = "RETREAT"
STAGE_UPRIGHT = "UPRIGHT"
STAGE_BASKET = "BASKET"

# The run ended (aborted or stopped) before this truss reached any
# outcome of its own. Fails both metrics.
STAGE_ABORTED = "ABORTED"

# Route-level navigation trouble: driving to a stop, holding the aisle
# heading, or recentering the base after a harvest. Recorded as an
# event rather than a truss outcome, because it costs coverage (the
# trusses at that stop are never attempted) instead of failing a
# manipulation attempt.
STAGE_NAV = "NAV"

STAGES = (
    STAGE_SETUP,
    STAGE_PERCEPTION,
    STAGE_GRASP,
    STAGE_CUT,
    STAGE_RETREAT,
    STAGE_UPRIGHT,
    STAGE_BASKET,
    STAGE_ABORTED,
)

# Failing in one of these stages leaves metric 1 intact: CUT already
# met its criterion, so what broke afterwards was the controller and
# not perception. Every other stage fails both metrics.
POST_PERCEPTION_STAGES = (
    STAGE_RETREAT,
    STAGE_UPRIGHT,
    STAGE_BASKET,
)


class HarvestLog:
    """Per-truss outcome records for one route run."""

    def __init__(self):
        self.records = []
        self.pending = None

        # Route-level events, kept apart from per-truss outcomes so
        # they cannot move either success rate.
        self.events = []

    def record_event(
        self,
        kind,
        label,
        reason,
    ):
        """
        Note something that cost coverage rather than failing a truss.

        A skipped stop means its trusses were never attempted, so the
        summary reports it next to the per-truss metrics instead of
        silently shrinking the denominator.
        """
        self.events.append(
            {
                "kind": kind,
                "label": label,
                "reason": reason,
            }
        )

        print(
            f"[ROUTE EVENT] {kind} | {label} | {reason}"
        )

    # --------------------------------------------------------
    # RECORD LIFECYCLE
    # --------------------------------------------------------

    def begin_truss(
        self,
        truss_label,
        **context,
    ):
        """
        Open the record for one truss attempt.

        Everything measured between here and the outcome call is
        attached with `note()`.
        """
        self.finalize_pending(
            "Previous truss never reached an outcome."
        )

        self.pending = {
            "truss": truss_label,
            "stage": "",
            "reason": "",
            "perception_success": False,
            "end_to_end_success": False,
        }

        self.pending.update(
            context
        )

    def note(
        self,
        **fields,
    ):
        """Attach measurements to the truss currently being attempted."""
        if self.pending is None:
            return

        self.pending.update(
            fields
        )

    def _close(
        self,
        truss_label,
        stage,
        reason,
    ):
        if self.pending is None:
            # An outcome without a matching begin_truss(): keep the
            # result rather than dropping it.
            self.pending = {
                "truss": truss_label,
                "stage": "",
                "reason": "",
                "perception_success": False,
                "end_to_end_success": False,
            }

        record = self.pending
        self.pending = None

        record["truss"] = truss_label
        record["stage"] = stage or ""
        record["reason"] = reason

        record["perception_success"] = (
            self.perception_ok(
                record
            )
        )

        record["end_to_end_success"] = (
            self.end_to_end_ok(
                record
            )
        )

        # Drop ghost outcomes that never got a USD truss_path (can
        # appear if record_success/failure is called without begin).
        if not str(
            record.get("truss_path") or ""
        ).strip():
            print(
                "[HARVEST LOG] Skipping pathless truss record "
                f"label={record.get('truss')!r}"
            )
            return record

        self.records.append(
            record
        )

        return record

    def record_success(
        self,
        truss_label,
    ):
        self._close(
            truss_label,
            None,
            "",
        )

    def record_failure(
        self,
        truss_label,
        stage,
        reason,
    ):
        if stage not in STAGES:
            raise ValueError(
                f"Unknown harvest stage: {stage}"
            )

        self._close(
            truss_label,
            stage,
            reason,
        )

        print(
            f"[STAGE FAIL] {stage} | {truss_label} | "
            f"{reason}"
        )

    def finalize_pending(
        self,
        reason,
    ):
        """
        Close an open truss record that never reached an outcome.

        Called when the next truss starts and again before artifacts
        are written, so an aborted route still reports every truss it
        touched.
        """
        if self.pending is None:
            return

        truss_label = self.pending.get(
            "truss",
            "<unknown truss>",
        )

        self._close(
            truss_label,
            STAGE_ABORTED,
            reason,
        )

        print(
            f"[STAGE FAIL] {STAGE_ABORTED} | {truss_label} | "
            f"{reason}"
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    def perception_ok(
        self,
        record,
    ):
        stage = record["stage"]

        return (
            not stage
            or stage in POST_PERCEPTION_STAGES
        )

    def end_to_end_ok(
        self,
        record,
    ):
        return not record["stage"]

    def _values(
        self,
        field,
        cast=float,
    ):
        return [
            cast(
                record[field]
            )
            for record in self.records
            if record.get(
                field
            ) is not None
        ]

    def _keypoint_lines(self):
        """
        K1 / K2 detection quality, when a keypoint estimator ran.

        Keypoint error is reported next to target error because the
        two separate detection quality from the derivation that turns
        keypoints into GraspPoint and CutPoint.
        """
        lines = []

        summaries = [
            (
                "K1 (cut)   ",
                "k1_confidence",
                "k1_error_m",
            ),
            (
                "K2 (grasp) ",
                "k2_confidence",
                "k2_error_m",
            ),
        ]

        if not any(
            self._values(
                confidence_field
            )
            for _, confidence_field, _ in summaries
        ):
            return lines

        lines.append(
            ""
        )

        lines.append(
            "Keypoints (K1 = CutPoint, K2 = GraspPoint):"
        )

        for label, confidence_field, error_field in summaries:
            confidences = self._values(
                confidence_field
            )

            errors = self._values(
                error_field
            )

            detail = (
                f"  {label} detected {len(confidences)}"
                f"/{len(self.records)}"
            )

            if confidences:
                detail += (
                    "  mean conf "
                    f"{sum(confidences) / len(confidences):.2f}"
                )

            if errors:
                detail += (
                    "  mean err "
                    f"{sum(errors) / len(errors):.4f} m"
                )

            lines.append(
                detail
            )

        flags = self._values(
            "keypoint_needs_another_view",
            cast=bool,
        )

        if flags:
            lines.append(
                "  Flagged for another view: "
                f"{sum(flags)}/{len(flags)} "
                "(fixed view still uses 1 capture)"
            )

        return lines

    def summary_lines(self):
        """The harvest summary as lines, for printing and for a file."""
        total = len(
            self.records
        )

        lines = [
            "========================================",
            "[HARVEST OUTCOME SUMMARY]",
            "========================================",
        ]

        if total == 0:
            lines.append(
                "No trusses were attempted."
            )
            return lines

        perception_hits = sum(
            1 for record in self.records
            if self.perception_ok(
                record
            )
        )

        end_to_end_hits = sum(
            1 for record in self.records
            if self.end_to_end_ok(
                record
            )
        )

        lines.append(
            f"Trusses attempted: {total}"
        )

        lines.append(
            "Perception/manipulation success: "
            f"{perception_hits}/{total} "
            f"({100.0 * perception_hits / total:.1f}%)"
        )

        lines.append(
            "End-to-end harvest success:      "
            f"{end_to_end_hits}/{total} "
            f"({100.0 * end_to_end_hits / total:.1f}%)"
        )

        grasp_errs = [
            float(record["grasp_localization_error_m"])
            for record in self.records
            if record.get(
                "grasp_localization_error_m"
            ) is not None
        ]

        cut_errs = [
            float(record["cut_localization_error_m"])
            for record in self.records
            if record.get(
                "cut_localization_error_m"
            ) is not None
        ]

        views = [
            int(record["views_used"])
            for record in self.records
            if record.get(
                "views_used"
            ) is not None
        ]

        if grasp_errs or cut_errs or views:
            lines.append(
                ""
            )

            lines.append(
                f"{RUN_CONDITION_LABEL} localization (vs hidden USD GT):"
            )

            if grasp_errs:
                lines.append(
                    "  Mean grasp-point error: "
                    f"{sum(grasp_errs) / len(grasp_errs):.4f} m "
                    f"(n={len(grasp_errs)})"
                )

            if cut_errs:
                lines.append(
                    "  Mean cut-point error:   "
                    f"{sum(cut_errs) / len(cut_errs):.4f} m "
                    f"(n={len(cut_errs)})"
                )

            if views:
                lines.append(
                    "  Mean views / target:    "
                    f"{sum(views) / len(views):.2f}"
                )

        motion_times = self._values(
            "perception_camera_motion_s"
        )

        if motion_times:
            lines.append(
                "  Mean perception+camera motion: "
                f"{sum(motion_times) / len(motion_times):.2f} s "
                f"(n={len(motion_times)})"
            )

        selected = [
            record.get(
                "selected_viewpoint"
            )
            for record in self.records
            if record.get(
                "selected_viewpoint"
            )
        ]

        if selected:
            counts = {}

            for name in selected:
                counts[name] = counts.get(
                    name,
                    0,
                ) + 1

            lines.append(
                "  Selected viewpoints: "
                + ", ".join(
                    f"{name}={counts[name]}"
                    for name in sorted(
                        counts
                    )
                )
            )

        lines.extend(
            self._keypoint_lines()
        )

        lines.append(
            ""
        )

        lines.append(
            "Failures by stage:"
        )

        for stage in STAGES:
            hits = [
                record for record in self.records
                if record["stage"] == stage
            ]

            if not hits:
                continue

            lines.append(
                f"  {stage:11s} {len(hits):3d}"
            )

            for record in hits:
                lines.append(
                    f"      {record['truss']}: "
                    f"{record['reason']}"
                )

        if self.events:
            lines.append(
                ""
            )

            lines.append(
                f"Route events (cost coverage, not counted above): "
                f"{len(self.events)}"
            )

            for event in self.events:
                lines.append(
                    f"  {event['kind']:5s} {event['label']}: "
                    f"{event['reason']}"
                )

        return lines

    def print_summary(self):
        print()

        for line in self.summary_lines():
            print(
                line
            )
