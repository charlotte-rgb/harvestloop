#!/usr/bin/env python3
"""
Score a paired fixed_view vs active_perception experiment.

Reads finished artifacts only. Writes:

    results/comparisons/<experiment_id>/
        benchmark.md
        comparison.csv
        paired.csv
        paired_targets.csv
        figures/localization_error.png
        figures/harvest_success.png
        figures/success_vs_occlusion.png
        figures/views_time_cost.png

Also copies the figures to results/figures/ and docs/demo/.

Usage:
    python3 scripts/analyze_paired_benchmark.py
    python3 scripts/analyze_paired_benchmark.py 20260909_192714
    python3 scripts/analyze_paired_benchmark.py results/runs/<id>
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"
COMPARISONS_ROOT = PROJECT_ROOT / "results" / "comparisons"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures"
DEMO_ROOT = PROJECT_ROOT / "docs" / "demo"

FIXED = "fixed_view"
ACTIVE = "active_perception"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_conditions import (  # noqa: E402
    load_meta,
    load_rows,
    newest_experiment_pair,
    number,
    paired_rows as _flat_paired_rows,
    resolve_user_path,
    summarize,
    truthy,
)


def paired_rows(fixed, active):
    """
    Join on truss_path and keep the raw rows for rescue/harm scoring.
    """
    flat = _flat_paired_rows(
        fixed,
        active,
    )

    fixed_by = {
        row.get(
            "truss_path"
        )
        or row.get(
            "image_stem"
        ): row
        for row in fixed["rows"]
    }
    active_by = {
        row.get(
            "truss_path"
        )
        or row.get(
            "image_stem"
        ): row
        for row in active["rows"]
    }

    enriched = []

    for pair in flat:
        key = pair[
            "key"
        ]
        enriched.append(
            {
                **pair,
                "fixed": fixed_by[
                    key
                ],
                "active": active_by[
                    key
                ],
            }
        )

    return enriched


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


def rate(hits, total):
    if not total:
        return None

    return hits / total


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
        return "—"

    return f"{hits}/{total} ({100.0 * hits / total:.1f}%)"


def stage_of(row):
    return (
        row.get(
            "stage"
        ) or ""
    ).strip()


def grasp_ok(row):
    if row.get(
        "grasp_success"
    ) not in (
        "",
        None,
    ):
        flagged = truthy(
            row,
            "grasp_success",
        )

        if flagged is not None:
            return flagged

    stage = stage_of(
        row
    )

    if not stage:
        return True

    return stage in (
        "CUT",
        "RETREAT",
        "UPRIGHT",
        "BASKET",
    )


def cut_ok(row):
    if row.get(
        "cut_success"
    ) not in (
        "",
        None,
    ):
        flagged = truthy(
            row,
            "cut_success",
        )

        if flagged is not None:
            return flagged

    flagged = truthy(
        row,
        "perception_success",
    )

    if flagged is not None:
        return flagged

    stage = stage_of(
        row
    )

    return (
        not stage
        or stage in (
            "RETREAT",
            "UPRIGHT",
            "BASKET",
        )
    )


def e2e_ok(row):
    flagged = truthy(
        row,
        "end_to_end_success",
    )

    if flagged is not None:
        return flagged

    return not stage_of(
        row
    )


def occlusion_of(row, meta, experiment):
    return (
        row.get(
            "occlusion_level"
        )
        or (meta.get("config") or {}).get(
            "occlusion_level"
        )
        or experiment.get(
            "occlusion_level"
        )
        or "medium"
    )


def seed_of(row, meta, experiment):
    value = (
        row.get(
            "scene_seed"
        )
        or (meta.get("config") or {}).get(
            "scene_seed"
        )
        or experiment.get(
            "scene_seed"
        )
        or 42
    )

    try:
        return int(
            float(
                value
            )
        )
    except (TypeError, ValueError):
        return value


def condition_metrics(summary, experiment):
    rows = summary["rows"]
    meta = summary["meta"]
    total = len(
        rows
    )
    grasp_hits = sum(
        1 for row in rows if grasp_ok(row)
    )
    cut_hits = sum(
        1 for row in rows if cut_ok(row)
    )
    e2e_hits = sum(
        1 for row in rows if e2e_ok(row)
    )

    wall = meta.get(
        "wall_clock_s"
    )

    return {
        **summary,
        "grasp_hits": grasp_hits,
        "cut_hits": cut_hits,
        "e2e_hits": e2e_hits,
        "grasp_rate": rate(
            grasp_hits,
            total,
        ),
        "cut_rate": rate(
            cut_hits,
            total,
        ),
        "e2e_rate": rate(
            e2e_hits,
            total,
        ),
        "mean_grasp_err": mean(
            summary["grasp_errs"]
        ),
        "mean_cut_err": mean(
            summary["cut_errs"]
        ),
        "mean_views": mean(
            summary["views"]
        ),
        "mean_motion": mean(
            summary["motion"]
        ),
        "mean_total": mean(
            [
                number(
                    row,
                    "total_time_s",
                )
                for row in rows
            ]
        ),
        "wall_clock_s": wall,
        "occlusion": occlusion_of(
            rows[0] if rows else {},
            meta,
            experiment,
        ),
        "scene_seed": seed_of(
            rows[0] if rows else {},
            meta,
            experiment,
        ),
        "estimator": (
            (meta.get("config") or {}).get(
                "estimator_mode"
            )
            or "keypoints"
        ),
        "oracle_used": any(
            (
                row.get(
                    "estimate_source"
                ) or ""
            ).startswith(
                "oracle"
            )
            for row in rows
        ),
    }


def pair_outcomes(pairs):
    rescued = []
    harmed = []
    improved = 0
    worsened = 0

    for pair in pairs:
        fixed_cut = cut_ok(
            pair["fixed"]
        )
        active_cut = cut_ok(
            pair["active"]
        )

        if active_cut and not fixed_cut:
            rescued.append(
                pair
            )

        if fixed_cut and not active_cut:
            harmed.append(
                pair
            )

        fixed_err = number(
            pair["fixed"],
            "cut_localization_error_m",
        )
        active_err = number(
            pair["active"],
            "cut_localization_error_m",
        )

        if (
            fixed_err is not None
            and active_err is not None
        ):
            delta = active_err - fixed_err

            if delta < -0.005:
                improved += 1

            if delta > 0.005:
                worsened += 1

    return {
        "rescued": rescued,
        "harmed": harmed,
        "improved_5mm": improved,
        "worsened_5mm": worsened,
    }


def target_label(row):
    stop = row.get(
        "stop_index"
    ) or "?"
    side = row.get(
        "side"
    ) or "?"
    truss = row.get(
        "truss"
    ) or row.get(
        "truss_name"
    ) or "?"

    return f"stop {stop} {side} {truss}"


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def _bar_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def plot_localization(fixed, active, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    _bar_style()

    labels = [
        "Grasp-point error",
        "Cut-point error",
    ]
    fixed_vals = [
        fixed["mean_grasp_err"] or 0.0,
        fixed["mean_cut_err"] or 0.0,
    ]
    active_vals = [
        active["mean_grasp_err"] or 0.0,
        active["mean_cut_err"] or 0.0,
    ]

    x = np.arange(
        len(labels)
    )
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(7.2, 4.2)
    )
    ax.bar(
        x - width / 2,
        [
            value * 1000.0
            for value in fixed_vals
        ],
        width,
        label="Fixed view (1 capture)",
        color="#4C6A92",
    )
    ax.bar(
        x + width / 2,
        [
            value * 1000.0
            for value in active_vals
        ],
        width,
        label="Active perception (≤3 views)",
        color="#C47B2B",
    )
    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        labels
    )
    ax.set_ylabel(
        "Mean 3D error vs hidden USD GT (mm)"
    )
    ax.set_title(
        "Fixed vs active localization error"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=160,
    )
    plt.close(
        fig
    )


def plot_success(fixed, active, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    _bar_style()

    labels = [
        "Grasp",
        "Cut",
        "End-to-end",
    ]
    fixed_vals = [
        100.0 * (fixed["grasp_rate"] or 0.0),
        100.0 * (fixed["cut_rate"] or 0.0),
        100.0 * (fixed["e2e_rate"] or 0.0),
    ]
    active_vals = [
        100.0 * (active["grasp_rate"] or 0.0),
        100.0 * (active["cut_rate"] or 0.0),
        100.0 * (active["e2e_rate"] or 0.0),
    ]

    x = np.arange(
        len(labels)
    )
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(7.2, 4.2)
    )
    ax.bar(
        x - width / 2,
        fixed_vals,
        width,
        label="Fixed view",
        color="#4C6A92",
    )
    ax.bar(
        x + width / 2,
        active_vals,
        width,
        label="Active perception",
        color="#C47B2B",
    )
    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        labels
    )
    ax.set_ylabel(
        "Success rate (%)"
    )
    ax.set_ylim(
        0,
        100,
    )
    ax.set_title(
        "Fixed vs active harvest success"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=160,
    )
    plt.close(
        fig
    )


def plot_occlusion(pairs, experiment, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    _bar_style()

    grouped = defaultdict(
        lambda: {
            "fixed_cut": [],
            "active_cut": [],
        }
    )

    default_level = experiment.get(
        "occlusion_level"
    ) or "medium"

    for pair in pairs:
        level = (
            pair["fixed"].get(
                "occlusion_level"
            )
            or pair["active"].get(
                "occlusion_level"
            )
            or default_level
        )
        grouped[level]["fixed_cut"].append(
            cut_ok(
                pair["fixed"]
            )
        )
        grouped[level]["active_cut"].append(
            cut_ok(
                pair["active"]
            )
        )

    order = [
        level
        for level in (
            "low",
            "medium",
            "high",
        )
        if level in grouped
    ]

    if not order:
        order = sorted(
            grouped
        )

    x = np.arange(
        len(order)
    )
    width = 0.36
    fixed_vals = [
        100.0 * (
            sum(grouped[level]["fixed_cut"])
            / max(len(grouped[level]["fixed_cut"]), 1)
        )
        for level in order
    ]
    active_vals = [
        100.0 * (
            sum(grouped[level]["active_cut"])
            / max(len(grouped[level]["active_cut"]), 1)
        )
        for level in order
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.2)
    )
    ax.bar(
        x - width / 2,
        fixed_vals,
        width,
        label="Fixed view",
        color="#4C6A92",
    )
    ax.bar(
        x + width / 2,
        active_vals,
        width,
        label="Active perception",
        color="#C47B2B",
    )
    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        [
            f"{level}\n(n={len(grouped[level]['fixed_cut'])})"
            for level in order
        ]
    )
    ax.set_ylabel(
        "Cut / perception success (%)"
    )
    ax.set_ylim(
        0,
        100,
    )
    ax.set_title(
        "Success vs occlusion"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=160,
    )
    plt.close(
        fig
    )


def plot_cost(fixed, active, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    _bar_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.6, 4.2),
    )

    axes[0].bar(
        [
            "Fixed",
            "Active",
        ],
        [
            fixed["mean_views"] or 1.0,
            active["mean_views"] or 1.0,
        ],
        color=[
            "#4C6A92",
            "#C47B2B",
        ],
    )
    axes[0].set_ylabel(
        "Mean views / target"
    )
    axes[0].set_title(
        "Views used"
    )
    axes[0].set_ylim(
        0,
        3.2,
    )

    axes[1].bar(
        [
            "Fixed",
            "Active",
        ],
        [
            fixed["mean_motion"] or 0.0,
            active["mean_motion"] or 0.0,
        ],
        color=[
            "#4C6A92",
            "#C47B2B",
        ],
    )
    axes[1].set_ylabel(
        "Mean perception + camera-motion time (s)"
    )
    axes[1].set_title(
        "Time cost"
    )

    fig.suptitle(
        "Views and time cost"
    )
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=160,
    )
    plt.close(
        fig
    )


def build_markdown(
    experiment_id,
    experiment,
    fixed,
    active,
    pairs,
    outcomes,
):
    rescued_lines = [
        f"- {target_label(pair['fixed'])} → "
        f"{pair['active'].get('selected_viewpoint') or 'canonical'} "
        f"({pair['active'].get('views_used') or '?'} views)"
        for pair in outcomes["rescued"]
    ]
    harmed_lines = [
        f"- {target_label(pair['fixed'])}"
        for pair in outcomes["harmed"]
    ]

    warnings = []

    if fixed["oracle_used"] or active["oracle_used"]:
        warnings.append(
            "Oracle estimator rows were found. Hidden USD GT must "
            "be scoring-only for the paper table."
        )

    if len(pairs) != min(
        fixed["total"],
        active["total"],
    ):
        warnings.append(
            f"Only {len(pairs)}/{min(fixed['total'], active['total'])} "
            "targets share the same `truss_path`. Rebuild the scene "
            "between conditions and use `run_paired_experiment.py`."
        )

    lines = [
        f"# Paired benchmark: `{experiment_id}`",
        "",
        f"- Scene seed: `{fixed['scene_seed']}`",
        f"- Occlusion: `{fixed['occlusion']}`",
        f"- Estimator: `{fixed['estimator']}`",
        f"- Protocol: `{experiment.get('protocol') or 'legacy_manual_pair'}`",
        f"- Max targets: `{experiment.get('max_targets')}`",
        f"- Paired targets (same `truss_path`): {len(pairs)}",
        f"- Hidden USD GT: scoring-only "
        f"({'violated' if (fixed['oracle_used'] or active['oracle_used']) else 'yes'})",
        "",
        "## Summary",
        "",
        "| Metric | Fixed view | Active perception |",
        "|---|---:|---:|",
        f"| Targets | {fixed['total']} | {active['total']} |",
        f"| Grasp error | {fmt_mean(fixed['mean_grasp_err'])} | {fmt_mean(active['mean_grasp_err'])} |",
        f"| Cut error | {fmt_mean(fixed['mean_cut_err'])} | {fmt_mean(active['mean_cut_err'])} |",
        f"| Grasp success | {fmt_rate(fixed['grasp_hits'], fixed['total'])} | {fmt_rate(active['grasp_hits'], active['total'])} |",
        f"| Cut success | {fmt_rate(fixed['cut_hits'], fixed['total'])} | {fmt_rate(active['cut_hits'], active['total'])} |",
        f"| End-to-end success | {fmt_rate(fixed['e2e_hits'], fixed['total'])} | {fmt_rate(active['e2e_hits'], active['total'])} |",
        f"| Mean views | {fmt_mean(fixed['mean_views'], unit='', digits=2)} | {fmt_mean(active['mean_views'], unit='', digits=2)} |",
        f"| Camera-motion time | {fmt_mean(fixed['mean_motion'], 's')} | {fmt_mean(active['mean_motion'], 's')} |",
        f"| Condition wall clock | {fmt_mean(fixed['wall_clock_s'], 's')} | {fmt_mean(active['wall_clock_s'], 's')} |",
        "",
        "## Rescued / harmed (paired cut success)",
        "",
        f"- Rescued by active: **{len(outcomes['rescued'])}**",
        f"- Harmed by active: **{len(outcomes['harmed'])}**",
        f"- Cut error improved >5 mm: {outcomes['improved_5mm']}",
        f"- Cut error worsened >5 mm: {outcomes['worsened_5mm']}",
        "",
    ]

    if rescued_lines:
        lines.append(
            "Rescued:"
        )
        lines.extend(
            rescued_lines
        )
        lines.append(
            ""
        )

    if harmed_lines:
        lines.append(
            "Harmed:"
        )
        lines.extend(
            harmed_lines
        )
        lines.append(
            ""
        )

    lines.extend(
        [
            "## Figures",
            "",
            f"![localization_error](figures/localization_error.png)",
            "",
            f"![harvest_success](figures/harvest_success.png)",
            "",
            f"![success_vs_occlusion](figures/success_vs_occlusion.png)",
            "",
            f"![views_time_cost](figures/views_time_cost.png)",
            "",
        ]
    )

    if warnings:
        lines.append(
            "## Warnings"
        )
        lines.append(
            ""
        )

        for warning in warnings:
            lines.append(
                f"- {warning}"
            )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    )


def copy_figures(src_dir):
    for dest in (
        FIGURES_ROOT,
        DEMO_ROOT,
    ):
        dest.mkdir(
            parents=True,
            exist_ok=True,
        )

        for path in src_dir.glob(
            "*.png"
        ):
            shutil.copy2(
                path,
                dest / path.name,
            )


def resolve_experiment_dir(argv):
    if len(
        argv
    ) >= 2:
        path = resolve_user_path(
            argv[1]
        )

        if path.name in (
            FIXED,
            ACTIVE,
        ):
            return path.parent

        return path

    pair = newest_experiment_pair()

    if pair is None:
        raise SystemExit(
            f"No paired experiment under {RUNS_ROOT}"
        )

    return pair


def main(argv):
    experiment_dir = resolve_experiment_dir(
        argv
    )
    experiment_id = experiment_dir.name

    experiment_path = (
        experiment_dir / "experiment.json"
    )
    experiment = {}

    if experiment_path.exists():
        experiment = json.loads(
            experiment_path.read_text()
        )

    fixed_dir = experiment_dir / FIXED
    active_dir = experiment_dir / ACTIVE

    if not (
        fixed_dir / "trusses.csv"
    ).exists():
        raise SystemExit(
            f"Missing {fixed_dir / 'trusses.csv'}"
        )

    if not (
        active_dir / "trusses.csv"
    ).exists():
        raise SystemExit(
            f"Missing {active_dir / 'trusses.csv'}"
        )

    fixed = condition_metrics(
        summarize(
            fixed_dir
        ),
        experiment,
    )
    active = condition_metrics(
        summarize(
            active_dir
        ),
        experiment,
    )
    pairs = paired_rows(
        fixed,
        active,
    )
    outcomes = pair_outcomes(
        pairs
    )

    out_dir = (
        COMPARISONS_ROOT
        / experiment_id
    )
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_localization(
        fixed,
        active,
        fig_dir / "localization_error.png",
    )
    plot_success(
        fixed,
        active,
        fig_dir / "harvest_success.png",
    )
    plot_occlusion(
        pairs,
        experiment,
        fig_dir / "success_vs_occlusion.png",
    )
    plot_cost(
        fixed,
        active,
        fig_dir / "views_time_cost.png",
    )
    copy_figures(
        fig_dir
    )

    markdown = build_markdown(
        experiment_id,
        experiment,
        fixed,
        active,
        pairs,
        outcomes,
    )
    (
        out_dir / "benchmark.md"
    ).write_text(
        markdown
    )

    write_csv(
        out_dir / "comparison.csv",
        [
            {
                "metric": "targets",
                "fixed_view": fixed["total"],
                "active_perception": active["total"],
            },
            {
                "metric": "mean_grasp_error_m",
                "fixed_view": fixed["mean_grasp_err"],
                "active_perception": active["mean_grasp_err"],
            },
            {
                "metric": "mean_cut_error_m",
                "fixed_view": fixed["mean_cut_err"],
                "active_perception": active["mean_cut_err"],
            },
            {
                "metric": "grasp_success",
                "fixed_view": fixed["grasp_rate"],
                "active_perception": active["grasp_rate"],
            },
            {
                "metric": "cut_success",
                "fixed_view": fixed["cut_rate"],
                "active_perception": active["cut_rate"],
            },
            {
                "metric": "end_to_end_success",
                "fixed_view": fixed["e2e_rate"],
                "active_perception": active["e2e_rate"],
            },
            {
                "metric": "mean_views",
                "fixed_view": fixed["mean_views"],
                "active_perception": active["mean_views"],
            },
            {
                "metric": "mean_camera_motion_s",
                "fixed_view": fixed["mean_motion"],
                "active_perception": active["mean_motion"],
            },
            {
                "metric": "rescued",
                "fixed_view": 0,
                "active_perception": len(
                    outcomes["rescued"]
                ),
            },
            {
                "metric": "harmed",
                "fixed_view": len(
                    outcomes["harmed"]
                ),
                "active_perception": 0,
            },
        ],
        [
            "metric",
            "fixed_view",
            "active_perception",
        ],
    )

    paired_out = []

    for pair in pairs:
        paired_out.append(
            {
                "truss_path": pair["key"],
                "label": target_label(
                    pair["fixed"]
                ),
                "occlusion": occlusion_of(
                    pair["fixed"],
                    fixed["meta"],
                    experiment,
                ),
                "scene_seed": seed_of(
                    pair["fixed"],
                    fixed["meta"],
                    experiment,
                ),
                "fixed_grasp_error_m": number(
                    pair["fixed"],
                    "grasp_localization_error_m",
                ),
                "active_grasp_error_m": number(
                    pair["active"],
                    "grasp_localization_error_m",
                ),
                "fixed_cut_error_m": number(
                    pair["fixed"],
                    "cut_localization_error_m",
                ),
                "active_cut_error_m": number(
                    pair["active"],
                    "cut_localization_error_m",
                ),
                "fixed_grasp_success": grasp_ok(
                    pair["fixed"]
                ),
                "active_grasp_success": grasp_ok(
                    pair["active"]
                ),
                "fixed_cut_success": cut_ok(
                    pair["fixed"]
                ),
                "active_cut_success": cut_ok(
                    pair["active"]
                ),
                "fixed_e2e_success": e2e_ok(
                    pair["fixed"]
                ),
                "active_e2e_success": e2e_ok(
                    pair["active"]
                ),
                "fixed_views": number(
                    pair["fixed"],
                    "views_used",
                ),
                "active_views": number(
                    pair["active"],
                    "views_used",
                ),
                "fixed_motion_s": number(
                    pair["fixed"],
                    "perception_camera_motion_s",
                ),
                "active_motion_s": number(
                    pair["active"],
                    "perception_camera_motion_s",
                ),
                "selected_viewpoint": pair["active"].get(
                    "selected_viewpoint"
                ),
                "rescued": (
                    cut_ok(
                        pair["active"]
                    )
                    and not cut_ok(
                        pair["fixed"]
                    )
                ),
                "harmed": (
                    cut_ok(
                        pair["fixed"]
                    )
                    and not cut_ok(
                        pair["active"]
                    )
                ),
            }
        )

    if paired_out:
        write_csv(
            out_dir / "paired_targets.csv",
            paired_out,
            list(
                paired_out[0].keys()
            ),
        )
        write_csv(
            out_dir / "paired.csv",
            paired_out,
            list(
                paired_out[0].keys()
            ),
        )

    print(
        markdown
    )
    print()
    print(
        f"Wrote {out_dir}"
    )


if __name__ == "__main__":
    main(
        sys.argv
    )
