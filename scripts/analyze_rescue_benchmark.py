#!/usr/bin/env python3
"""
Score a shared-view rescue experiment.

Primary metric:

    rescue_rate =
        # failed/bad canonical views recovered by extra views
        / # failed/bad canonical views

Reads:
    results/runs/<experiment_id>/shared_view/trusses.csv

Writes:
    results/comparisons/<experiment_id>/
        rescue.md
        rescue.csv
        figures/rescue_summary.png

Usage:
    python3 scripts/analyze_rescue_benchmark.py
    python3 scripts/analyze_rescue_benchmark.py <experiment_id>
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"
COMPARISONS_ROOT = PROJECT_ROOT / "results" / "comparisons"
DEMO_ROOT = PROJECT_ROOT / "docs" / "demo"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_conditions import (  # noqa: E402
    load_meta,
    load_rows,
    number,
    resolve_user_path,
    truthy,
)


def mean(values):
    values = [
        value
        for value in values
        if value is not None and not (
            isinstance(value, float) and math.isnan(value)
        )
    ]

    if not values:
        return None

    return sum(values) / len(values)


def fmt_mean(value, unit="m", digits=4):
    if value is None:
        return "—"

    if unit == "m":
        return f"{value:.{digits}f} m"

    if unit == "s":
        return f"{value:.2f} s"

    return f"{value:.{digits}f}"


def fmt_rate(hits, total):
    if not total:
        return "— (n=0)"

    return f"{hits}/{total} ({100.0 * hits / total:.1f}%)"


def newest_shared_run():
    candidates = []

    if not RUNS_ROOT.exists():
        raise SystemExit(f"No runs under {RUNS_ROOT}")

    for path in RUNS_ROOT.iterdir():
        if not path.is_dir():
            continue

        for name in (
            "shared_view",
            "active_perception",
        ):
            csv_path = path / name / "trusses.csv"

            if csv_path.exists() and load_rows(csv_path.parent):
                candidates.append(csv_path.parent)

    if not candidates:
        raise SystemExit(
            f"No shared_view runs with trusses.csv under {RUNS_ROOT}"
        )

    return max(
        candidates,
        key=lambda path: (
            path.parent.name,
            path.name,
        ),
    )


def summarize(run_dir: Path):
    rows = [
        row
        for row in load_rows(run_dir)
        if str(row.get("truss_path") or "").strip()
    ]
    meta = load_meta(run_dir)
    experiment = {}
    exp_path = run_dir.parent / "experiment.json"

    if exp_path.exists():
        experiment = json.loads(exp_path.read_text())

    bad = [
        row
        for row in rows
        if truthy(row, "canonical_view_bad")
        or (
            row.get("canonical_view_bad") in ("", None)
            and (
                truthy(row, "canonical_needs_another_view")
                or truthy(row, "keypoint_needs_another_view")
            )
        )
    ]
    rescued = [
        row
        for row in rows
        if (
            row.get("rescue_status") == "rescued"
            or truthy(row, "rescued")
        )
    ]
    no_need = [
        row
        for row in rows
        if (
            row.get("rescue_status") == "no_rescue_needed"
            or truthy(row, "no_rescue_needed")
        )
    ]
    not_rescued = [
        row
        for row in rows
        if row.get("rescue_status") == "not_rescued"
        or (
            row in bad
            and row not in rescued
            and row not in no_need
        )
    ]

    # Prefer explicit canonical_view_bad when present.
    bad_n = sum(
        1
        for row in rows
        if truthy(row, "canonical_view_bad")
        or (
            row.get("canonical_view_bad") in ("", None)
            and row.get("rescue_status")
            in ("rescued", "not_rescued")
        )
    )

    if bad_n == 0:
        bad_n = len(bad)

    rescued_n = sum(
        1
        for row in rows
        if row.get("rescue_status") == "rescued"
        or truthy(row, "rescued")
    )

    n = len(rows)
    canonical_failure_rate = (
        None if n <= 0 else bad_n / n
    )
    canonical_success_rate = (
        None
        if canonical_failure_rate is None
        else 1.0 - canonical_failure_rate
    )
    rescue_rate = (
        None if bad_n <= 0 else rescued_n / bad_n
    )
    # Final success after active perception:
    #   canonical_success + canonical_fail × rescue_rate
    # Equivalent to (n - bad + rescued) / n when rates defined.
    if n <= 0:
        final_success_rate = None
    elif bad_n <= 0:
        final_success_rate = 1.0
    elif rescue_rate is None:
        final_success_rate = canonical_success_rate
    else:
        final_success_rate = (
            canonical_success_rate
            + canonical_failure_rate * rescue_rate
        )

    return {
        "run_dir": run_dir,
        "experiment_id": run_dir.parent.name,
        "condition": run_dir.name,
        "rows": rows,
        "meta": meta,
        "experiment": experiment,
        "n": n,
        "bad_n": bad_n,
        "rescued_n": rescued_n,
        "not_rescued_n": len(
            [
                row
                for row in rows
                if row.get("rescue_status") == "not_rescued"
            ]
        ),
        "no_need_n": sum(
            1
            for row in rows
            if row.get("rescue_status") == "no_rescue_needed"
            or truthy(row, "no_rescue_needed")
        ),
        "rescue_rate": rescue_rate,
        "canonical_failure_rate": canonical_failure_rate,
        "canonical_success_rate": canonical_success_rate,
        "final_success_rate": final_success_rate,
        "mean_views": mean(
            [
                number(row, "views_used")
                for row in rows
            ]
        ),
        "mean_extra_motion_s": mean(
            [
                number(row, "extra_camera_motion_s")
                for row in rows
            ]
        ),
        "mean_canonical_cut_err": mean(
            [
                number(
                    row,
                    "canonical_cut_localization_error_m",
                )
                for row in rows
            ]
        ),
        "mean_selected_cut_err": mean(
            [
                number(row, "cut_localization_error_m")
                for row in rows
            ]
        ),
        "mean_canonical_grasp_err": mean(
            [
                number(
                    row,
                    "canonical_grasp_localization_error_m",
                )
                for row in rows
            ]
        ),
        "mean_selected_grasp_err": mean(
            [
                number(row, "grasp_localization_error_m")
                for row in rows
            ]
        ),
        "rescued_rows": [
            row
            for row in rows
            if row.get("rescue_status") == "rescued"
            or truthy(row, "rescued")
        ],
        "not_rescued_rows": [
            row
            for row in rows
            if row.get("rescue_status") == "not_rescued"
        ],
    }


def plot_summary(summary, out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        "No rescue needed",
        "Rescued",
        "Not rescued",
    ]
    values = [
        summary["no_need_n"],
        summary["rescued_n"],
        summary["not_rescued_n"],
    ]
    colors = ["#6B8F71", "#4C6A92", "#A65D57"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("Targets")
    axes[0].set_title("Shared-view rescue outcomes")

    before = summary["mean_canonical_cut_err"] or 0.0
    after = summary["mean_selected_cut_err"] or 0.0
    axes[1].bar(
        ["Canonical (fixed)", "After active"],
        [before * 1000.0, after * 1000.0],
        color=["#4C6A92", "#C4A35A"],
    )
    axes[1].set_ylabel("Mean cut error (mm)")
    axes[1].set_title("Localization before vs after")

    rate = summary["rescue_rate"]
    fig.suptitle(
        f"{summary['experiment_id']}  |  rescue_rate="
        + (
            "n/a"
            if rate is None
            else f"{100.0 * rate:.1f}%"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_markdown(summary):
    rate = summary["rescue_rate"]
    lines = [
        f"# Shared-view rescue: `{summary['experiment_id']}`",
        "",
        f"- Condition folder: `{summary['condition']}`",
        f"- Targets: `{summary['n']}`",
        f"- Protocol: shared canonical capture → optional active views",
        f"- Hidden USD GT: scoring-only",
        "",
        "## Primary metric",
        "",
        f"- Bad canonical views: **{summary['bad_n']}**",
        f"- Rescued: **{summary['rescued_n']}**",
        f"- Rescue rate: **{fmt_rate(summary['rescued_n'], summary['bad_n'])}**",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Targets | {summary['n']} |",
        f"| No rescue needed | {summary['no_need_n']} |",
        f"| Rescued | {summary['rescued_n']} |",
        f"| Not rescued | {summary['not_rescued_n']} |",
        f"| Mean views | {fmt_mean(summary['mean_views'], unit='views', digits=2)} |",
        f"| Mean extra camera-motion | {fmt_mean(summary['mean_extra_motion_s'], unit='s')} |",
        f"| Canonical cut error | {fmt_mean(summary['mean_canonical_cut_err'])} |",
        f"| Selected cut error | {fmt_mean(summary['mean_selected_cut_err'])} |",
        f"| Canonical grasp error | {fmt_mean(summary['mean_canonical_grasp_err'])} |",
        f"| Selected grasp error | {fmt_mean(summary['mean_selected_grasp_err'])} |",
        "",
        "## Rescued targets",
        "",
    ]

    if not summary["rescued_rows"]:
        lines.append("_None_")
    else:
        for row in summary["rescued_rows"]:
            lines.append(
                f"- stop {row.get('stop_index')} {row.get('side')} "
                f"{row.get('truss') or row.get('truss_name')} → "
                f"{row.get('selected_viewpoint')} "
                f"({row.get('views_used')} views)"
            )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![rescue_summary](figures/rescue_summary.png)",
            "",
        ]
    )

    return "\n".join(lines)


def main(argv):
    if len(argv) >= 2:
        path = resolve_user_path(argv[1])

        if path.name == "trusses.csv":
            run_dir = path.parent
        elif (path / "shared_view" / "trusses.csv").exists():
            run_dir = path / "shared_view"
        elif (path / "trusses.csv").exists():
            run_dir = path
        else:
            raise SystemExit(f"No trusses.csv under {path}")
    else:
        run_dir = newest_shared_run()

    summary = summarize(run_dir)
    out_dir = COMPARISONS_ROOT / summary["experiment_id"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_summary(summary, fig_dir / "rescue_summary.png")

    demo = DEMO_ROOT / "rescue_summary.png"
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    demo.write_bytes((fig_dir / "rescue_summary.png").read_bytes())

    markdown = build_markdown(summary)
    (out_dir / "rescue.md").write_text(markdown + "\n")

    with (out_dir / "rescue.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "value",
            ],
        )
        writer.writeheader()

        for metric, value in (
            ("targets", summary["n"]),
            ("bad_canonical", summary["bad_n"]),
            ("rescued", summary["rescued_n"]),
            ("not_rescued", summary["not_rescued_n"]),
            ("no_rescue_needed", summary["no_need_n"]),
            ("rescue_rate", summary["rescue_rate"]),
            ("mean_views", summary["mean_views"]),
            ("mean_extra_camera_motion_s", summary["mean_extra_motion_s"]),
            (
                "mean_canonical_cut_error_m",
                summary["mean_canonical_cut_err"],
            ),
            (
                "mean_selected_cut_error_m",
                summary["mean_selected_cut_err"],
            ),
        ):
            writer.writerow(
                {
                    "metric": metric,
                    "value": value,
                }
            )

    print(markdown)
    print()
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main(sys.argv)
