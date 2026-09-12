#!/usr/bin/env python3
"""
Compare fixed_view vs active_perception harvest runs.

Builds the controlled-comparison table from run artifacts. Does not
re-simulate anything: it only reads trusses.csv / run.json.

Usage:
    python3 scripts/compare_conditions.py
    python3 scripts/compare_conditions.py results/runs/<experiment_id>
    python3 scripts/compare_conditions.py \\
        results/runs/<exp>/fixed_view results/runs/<exp>/active_perception

With no arguments, picks the newest experiment folder that has both
condition subfolders (or falls back to newest legacy flat runs).
Writes:

    results/comparisons/<experiment_id>/
        comparison.md
        comparison.csv
        paired.csv
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

FIXED = "fixed_view"
ACTIVE = "active_perception"


def resolve_user_path(raw: str) -> Path:
    """
    Resolve a CLI path from any cwd.

    Tries, in order:
      1. absolute path as given
      2. path relative to the current working directory
      3. path relative to the project root
      4. experiment id under results/runs/
    """
    path = Path(
        raw
    ).expanduser()

    if path.is_absolute():
        return path.resolve()

    candidates = [
        (
            Path.cwd() / path
        ).resolve(),
        (
            PROJECT_ROOT / path
        ).resolve(),
        (
            RUNS_ROOT / path
        ).resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Prefer the project-root interpretation in error messages when
    # the user passed results/runs/... from scripts/.
    if str(
        path
    ).startswith(
        "results/"
    ):
        return (
            PROJECT_ROOT / path
        ).resolve()

    return candidates[
        0
    ]


def number(row, key):
    value = row.get(key, "")

    if value in (
        "",
        None,
    ):
        return None

    try:
        return float(
            value
        )
    except ValueError:
        return None


def truthy(row, key):
    value = str(
        row.get(
            key,
            "",
        )
    ).strip().lower()

    if value in (
        "1",
        "true",
        "yes",
    ):
        return True

    if value in (
        "0",
        "false",
        "no",
        "",
    ):
        return False

    return None


def load_rows(run_dir: Path):
    path = run_dir / "trusses.csv"

    if not path.exists():
        raise SystemExit(
            f"Missing {path}"
        )

    with path.open(
        newline=""
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def load_meta(run_dir: Path):
    path = run_dir / "run.json"

    if not path.exists():
        return {}

    return json.loads(
        path.read_text()
    )


def condition_of(run_dir: Path, meta, rows):
    if rows and rows[0].get(
        "condition"
    ):
        return rows[0]["condition"]

    if meta.get(
        "condition"
    ):
        return meta["condition"]

    config = meta.get(
        "config"
    ) or {}

    if config.get(
        "run_condition_label"
    ):
        return config[
            "run_condition_label"
        ]

    name = run_dir.name

    if name in (
        FIXED,
        ACTIVE,
    ):
        return name

    if name.endswith(
        f"_{FIXED}"
    ):
        return FIXED

    if name.endswith(
        f"_{ACTIVE}"
    ):
        return ACTIVE

    return "unknown"


def experiment_of(run_dir: Path, meta=None):
    meta = meta or {}

    if meta.get(
        "experiment_id"
    ):
        return meta[
            "experiment_id"
        ]

    config = meta.get(
        "config"
    ) or {}

    if config.get(
        "run_experiment_id"
    ):
        return config[
            "run_experiment_id"
        ]

    if run_dir.name in (
        FIXED,
        ACTIVE,
    ):
        return run_dir.parent.name

    return run_dir.name


def iter_condition_dirs(condition: str):
    """Yield finished condition run dirs (nested + legacy flat)."""
    if not RUNS_ROOT.exists():
        return

    for path in RUNS_ROOT.iterdir():
        if not path.is_dir():
            continue

        nested = path / condition

        if (
            nested.is_dir()
            and (
                nested / "trusses.csv"
            ).exists()
        ):
            yield nested
            continue

        if (
            path.name.endswith(
                f"_{condition}"
            )
            and (
                path / "trusses.csv"
            ).exists()
        ):
            yield path


def newest_run(condition: str) -> Path:
    candidates = list(
        iter_condition_dirs(
            condition
        )
    )

    # Prefer dirs with at least one truss row.
    nonempty = []

    for path in candidates:
        rows = load_rows(
            path
        )

        if not rows:
            continue

        meta = load_meta(
            path
        )

        if condition_of(
            path,
            meta,
            rows,
        ) == condition:
            nonempty.append(
                path
            )

    if not nonempty:
        raise SystemExit(
            f"No {condition} runs with trusses.csv under {RUNS_ROOT}"
        )

    return max(
        nonempty,
        key=lambda path: (
            experiment_of(
                path,
                load_meta(
                    path
                ),
            ),
            path.name,
        ),
    )


def newest_experiment_pair():
    """Newest parent folder that has both condition subfolders."""
    if not RUNS_ROOT.exists():
        return None

    candidates = []

    for path in RUNS_ROOT.iterdir():
        if not path.is_dir():
            continue

        fixed_dir = path / FIXED
        active_dir = path / ACTIVE

        if not (
            fixed_dir / "trusses.csv"
        ).exists():
            continue

        if not (
            active_dir / "trusses.csv"
        ).exists():
            continue

        if not load_rows(
            fixed_dir
        ):
            continue

        if not load_rows(
            active_dir
        ):
            continue

        candidates.append(
            path
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.name,
    )


def mean(values):
    values = [
        value
        for value in values
        if value is not None
        and math.isfinite(
            value
        )
    ]

    if not values:
        return None

    return sum(
        values
    ) / len(
        values
    )


def fmt_mean(values, unit="", digits=4):
    value = mean(
        values
    )

    if value is None:
        return "—"

    if unit == "m":
        return f"{value:.{digits}f} m (n={len([v for v in values if v is not None])})"

    if unit == "s":
        return f"{value:.2f} s (n={len([v for v in values if v is not None])})"

    if unit == "views":
        return f"{value:.2f}"

    return f"{value:.{digits}f}"


def fmt_rate(hits, total):
    if total <= 0:
        return "—"

    return (
        f"{hits}/{total} "
        f"({100.0 * hits / total:.1f}%)"
    )


def summarize(run_dir: Path):
    rows = load_rows(
        run_dir
    )
    meta = load_meta(
        run_dir
    )
    condition = condition_of(
        run_dir,
        meta,
        rows,
    )

    total = len(
        rows
    )

    perception = sum(
        1
        for row in rows
        if truthy(
            row,
            "perception_success",
        )
    )

    end_to_end = sum(
        1
        for row in rows
        if truthy(
            row,
            "end_to_end_success",
        )
    )

    grasp_errs = [
        number(
            row,
            "grasp_localization_error_m",
        )
        for row in rows
    ]

    cut_errs = [
        number(
            row,
            "cut_localization_error_m",
        )
        for row in rows
    ]

    views = [
        number(
            row,
            "views_used",
        )
        for row in rows
    ]

    motion = [
        number(
            row,
            "perception_camera_motion_s",
        )
        for row in rows
    ]

    flagged = sum(
        1
        for row in rows
        if truthy(
            row,
            "keypoint_needs_another_view",
        )
    )

    selected = {}

    for row in rows:
        name = row.get(
            "selected_viewpoint",
            "",
        ) or ""

        if not name:
            continue

        selected[name] = selected.get(
            name,
            0,
        ) + 1

    return {
        "run_dir": run_dir,
        "run_id": (
            meta.get(
                "run_id"
            )
            or f"{experiment_of(run_dir, meta)}/{condition}"
        ),
        "experiment_id": experiment_of(
            run_dir,
            meta,
        ),
        "condition": condition,
        "total": total,
        "perception": perception,
        "end_to_end": end_to_end,
        "grasp_errs": grasp_errs,
        "cut_errs": cut_errs,
        "views": views,
        "motion": motion,
        "flagged": flagged,
        "selected": selected,
        "rows": rows,
        "meta": meta,
    }


def paired_rows(fixed, active):
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

    keys = sorted(
        set(
            fixed_by
        )
        & set(
            active_by
        )
    )

    pairs = []

    for key in keys:
        left = fixed_by[
            key
        ]
        right = active_by[
            key
        ]

        pairs.append(
            {
                "key": key,
                "truss": left.get(
                    "truss"
                )
                or right.get(
                    "truss"
                ),
                "stop_index": left.get(
                    "stop_index"
                )
                or right.get(
                    "stop_index"
                ),
                "side": left.get(
                    "side"
                )
                or right.get(
                    "side"
                ),
                "fixed_views": number(
                    left,
                    "views_used",
                ),
                "active_views": number(
                    right,
                    "views_used",
                ),
                "fixed_flagged": truthy(
                    left,
                    "keypoint_needs_another_view",
                ),
                "active_selected": right.get(
                    "selected_viewpoint",
                    "",
                ),
                "fixed_cut_err": number(
                    left,
                    "cut_localization_error_m",
                ),
                "active_cut_err": number(
                    right,
                    "cut_localization_error_m",
                ),
                "fixed_grasp_err": number(
                    left,
                    "grasp_localization_error_m",
                ),
                "active_grasp_err": number(
                    right,
                    "grasp_localization_error_m",
                ),
                "fixed_perception": truthy(
                    left,
                    "perception_success",
                ),
                "active_perception": truthy(
                    right,
                    "perception_success",
                ),
                "fixed_e2e": truthy(
                    left,
                    "end_to_end_success",
                ),
                "active_e2e": truthy(
                    right,
                    "end_to_end_success",
                ),
            }
        )

    return pairs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
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
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key in fieldnames
                }
            )


def build_markdown(fixed, active, pairs):
    lines = [
        "# Fixed view vs active perception",
        "",
        f"- Fixed: `{fixed['run_id']}`",
        f"- Active: `{active['run_id']}`",
        f"- Paired trusses (same `truss_path`): {len(pairs)}",
        "",
        "## Summary table",
        "",
        "| Metric | Fixed view | Active perception |",
        "|---|---:|---:|",
        f"| Trusses attempted | {fixed['total']} | {active['total']} |",
        f"| Grasp-point error | {fmt_mean(fixed['grasp_errs'], 'm')} | {fmt_mean(active['grasp_errs'], 'm')} |",
        f"| Cut-point error | {fmt_mean(fixed['cut_errs'], 'm')} | {fmt_mean(active['cut_errs'], 'm')} |",
        f"| Perception / manipulation success | {fmt_rate(fixed['perception'], fixed['total'])} | {fmt_rate(active['perception'], active['total'])} |",
        f"| End-to-end harvest success | {fmt_rate(fixed['end_to_end'], fixed['total'])} | {fmt_rate(active['end_to_end'], active['total'])} |",
        f"| Mean views / target | {fmt_mean(fixed['views'], 'views')} | {fmt_mean(active['views'], 'views')} |",
        f"| Mean perception + camera-motion time | {fmt_mean(fixed['motion'], 's')} | {fmt_mean(active['motion'], 's')} |",
        f"| Flagged needs_another_view | {fixed['flagged']}/{fixed['total']} | {active['flagged']}/{active['total']} |",
        "",
    ]

    if active[
        "selected"
    ]:
        selected = ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                active[
                    "selected"
                ].items()
            )
        )

        lines.extend(
            [
                f"Active selected viewpoints: {selected}",
                "",
            ]
        )

    if pairs:
        helped = []
        hurt = []
        rescued = []

        for pair in pairs:
            fixed_cut = pair[
                "fixed_cut_err"
            ]
            active_cut = pair[
                "active_cut_err"
            ]

            if (
                fixed_cut is not None
                and active_cut is not None
            ):
                delta = active_cut - fixed_cut

                if delta < -0.005:
                    helped.append(
                        pair
                    )
                elif delta > 0.005:
                    hurt.append(
                        pair
                    )

            if (
                pair[
                    "fixed_perception"
                ]
                is False
                and pair[
                    "active_perception"
                ]
                is True
            ):
                rescued.append(
                    pair
                )

        lines.extend(
            [
                "## Paired targets",
                "",
                f"- Cut error improved by >5 mm: {len(helped)}",
                f"- Cut error worsened by >5 mm: {len(hurt)}",
                f"- Perception failures rescued by active: {len(rescued)}",
                "",
            ]
        )

        if rescued:
            lines.append(
                "Rescued:"
            )

            for pair in rescued:
                lines.append(
                    f"- stop {pair['stop_index']} {pair['side']} "
                    f"{pair['truss']} → selected "
                    f"{pair['active_selected']} "
                    f"({pair['active_views']} views)"
                )

            lines.append(
                ""
            )

        flagged_fixed = [
            pair
            for pair in pairs
            if pair[
                "fixed_flagged"
            ]
        ]

        if flagged_fixed:
            lines.extend(
                [
                    "## Targets fixed-view flagged for another view",
                    "",
                    "| Target | Fixed cut err | Active cut err | Active views | Selected | Fixed OK | Active OK |",
                    "|---|---:|---:|---:|---|---|---|",
                ]
            )

            for pair in flagged_fixed:
                lines.append(
                    "| "
                    f"stop {pair['stop_index']} {pair['side']} {pair['truss']}"
                    " | "
                    f"{pair['fixed_cut_err'] if pair['fixed_cut_err'] is not None else float('nan'):.4f}"
                    " | "
                    f"{pair['active_cut_err'] if pair['active_cut_err'] is not None else float('nan'):.4f}"
                    " | "
                    f"{pair['active_views'] if pair['active_views'] is not None else '—'}"
                    " | "
                    f"{pair['active_selected'] or '—'}"
                    " | "
                    f"{pair['fixed_perception']}"
                    " | "
                    f"{pair['active_perception']}"
                    " |"
                )

            lines.append(
                ""
            )

    if fixed["total"] != active["total"] or len(pairs) < min(fixed["total"], active["total"]):
        lines.extend(
            [
                "## Warning",
                "",
                "These runs are not a perfect matched pair "
                f"(fixed={fixed['total']}, active={active['total']}, "
                f"paired={len(pairs)}). Prefer two runs on the same "
                "built scene before treating this as the paper result.",
                "",
            ]
        )

    return "\n".join(
        lines
    )


def main(argv):
    if len(
        argv
    ) >= 3:
        fixed_dir = resolve_user_path(
            argv[1]
        )

        active_dir = resolve_user_path(
            argv[2]
        )
    elif len(
        argv
    ) == 2:
        experiment_dir = resolve_user_path(
            argv[1]
        )

        # Bare condition dir accidentally passed: climb to parent.
        if experiment_dir.name in (
            FIXED,
            ACTIVE,
        ) and (
            experiment_dir.parent / FIXED
        ).exists():
            experiment_dir = experiment_dir.parent

        fixed_dir = (
            experiment_dir / FIXED
        )
        active_dir = (
            experiment_dir / ACTIVE
        )

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
    else:
        pair = newest_experiment_pair()

        if pair is not None:
            fixed_dir = pair / FIXED
            active_dir = pair / ACTIVE
        else:
            fixed_dir = newest_run(
                FIXED
            )
            active_dir = newest_run(
                ACTIVE
            )

    fixed = summarize(
        fixed_dir
    )
    active = summarize(
        active_dir
    )

    if fixed[
        "condition"
    ] != FIXED:
        print(
            f"[WARN] {fixed_dir} condition={fixed['condition']!r}, expected {FIXED!r}"
        )

    if active[
        "condition"
    ] != ACTIVE:
        print(
            f"[WARN] {active_dir} condition={active['condition']!r}, expected {ACTIVE!r}"
        )

    pairs = paired_rows(
        fixed,
        active,
    )

    experiment_id = (
        experiment_of(
            fixed_dir,
            load_meta(
                fixed_dir
            ),
        )
        if fixed_dir.parent == active_dir.parent
        else f"{fixed['run_id']}_vs_{active['run_id']}"
    )

    # Nested pair: parent folder name. Flat legacy: keep _vs_ name.
    if (
        fixed_dir.name == FIXED
        and active_dir.name == ACTIVE
        and fixed_dir.parent == active_dir.parent
    ):
        experiment_id = fixed_dir.parent.name

    out_dir = (
        COMPARISONS_ROOT
        / experiment_id
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown = build_markdown(
        fixed,
        active,
        pairs,
    )

    (
        out_dir / "comparison.md"
    ).write_text(
        markdown
    )

    write_csv(
        out_dir / "comparison.csv",
        [
            {
                "metric": "trusses_attempted",
                "fixed_view": fixed[
                    "total"
                ],
                "active_perception": active[
                    "total"
                ],
            },
            {
                "metric": "mean_grasp_error_m",
                "fixed_view": mean(
                    fixed[
                        "grasp_errs"
                    ]
                ),
                "active_perception": mean(
                    active[
                        "grasp_errs"
                    ]
                ),
            },
            {
                "metric": "mean_cut_error_m",
                "fixed_view": mean(
                    fixed[
                        "cut_errs"
                    ]
                ),
                "active_perception": mean(
                    active[
                        "cut_errs"
                    ]
                ),
            },
            {
                "metric": "perception_success_rate",
                "fixed_view": (
                    fixed[
                        "perception"
                    ]
                    / fixed[
                        "total"
                    ]
                    if fixed[
                        "total"
                    ]
                    else None
                ),
                "active_perception": (
                    active[
                        "perception"
                    ]
                    / active[
                        "total"
                    ]
                    if active[
                        "total"
                    ]
                    else None
                ),
            },
            {
                "metric": "end_to_end_success_rate",
                "fixed_view": (
                    fixed[
                        "end_to_end"
                    ]
                    / fixed[
                        "total"
                    ]
                    if fixed[
                        "total"
                    ]
                    else None
                ),
                "active_perception": (
                    active[
                        "end_to_end"
                    ]
                    / active[
                        "total"
                    ]
                    if active[
                        "total"
                    ]
                    else None
                ),
            },
            {
                "metric": "mean_views_per_target",
                "fixed_view": mean(
                    fixed[
                        "views"
                    ]
                ),
                "active_perception": mean(
                    active[
                        "views"
                    ]
                ),
            },
            {
                "metric": "mean_perception_camera_motion_s",
                "fixed_view": mean(
                    fixed[
                        "motion"
                    ]
                ),
                "active_perception": mean(
                    active[
                        "motion"
                    ]
                ),
            },
        ],
        [
            "metric",
            "fixed_view",
            "active_perception",
        ],
    )

    if pairs:
        write_csv(
            out_dir / "paired.csv",
            pairs,
            list(
                pairs[0].keys()
            ),
        )

    print(
        markdown
    )
    print(
        ""
    )
    print(
        f"Wrote {out_dir}"
    )


if __name__ == "__main__":
    main(
        sys.argv
    )
