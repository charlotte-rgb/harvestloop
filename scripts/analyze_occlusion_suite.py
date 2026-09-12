#!/usr/bin/env python3
"""
Aggregate matched LOW / MEDIUM / HIGH shared-view rescue suites.

Headline comparison:
  canonical_success = 1 - canonical_fail_rate
  final_success     = canonical_success + canonical_fail_rate × rescue_rate

Claims to keep precise:
  - AP improves failure recovery / robustness
  - Do NOT claim monotonic improvement with occlusion
  - Do NOT claim AP improves mean localization / cut accuracy

Writes per suite:
    results/suites/<suite_id>/
        suite_benchmark.md
        suite_summary.csv
        figures/
            canonical_vs_final_success.png
            canonical_failure_vs_occlusion.png
            rescue_rate_vs_occlusion.png
            localization_before_after.png
            views_vs_occlusion.png

Batch pool (seeds 7/42/99):
    python3 scripts/analyze_occlusion_suite.py --batch 20260912_095224
    -> results/suites/<batch>/batch_benchmark.md

Usage:
    python3 scripts/analyze_occlusion_suite.py <suite_id>
    python3 scripts/analyze_occlusion_suite.py --batch 20260912_095224
    python3 scripts/analyze_occlusion_suite.py --expect-n 16 <suite_id>
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"
SUITES_ROOT = PROJECT_ROOT / "results" / "suites"
SCRIPTS = PROJECT_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))

from analyze_rescue_benchmark import (  # noqa: E402
    fmt_mean,
    fmt_rate,
    summarize,
)

LEVEL_ORDER = ("low", "medium", "high")
SHARED = "shared_view"


def resolve_suite(raw: str) -> Path:
    path = Path(raw).expanduser()

    if path.is_dir() and (path / "suite.json").exists():
        return path.resolve()

    candidate = SUITES_ROOT / raw

    if (candidate / "suite.json").exists():
        return candidate.resolve()

    raise SystemExit(f"No suite.json for {raw!r} under {SUITES_ROOT}")


def enrich_summary(summary, occlusion):
    summary = dict(summary)
    summary["occlusion"] = occlusion

    if summary.get("canonical_success_rate") is None:
        fail = summary.get("canonical_failure_rate")
        summary["canonical_success_rate"] = (
            None if fail is None else 1.0 - fail
        )

    if summary.get("final_success_rate") is None:
        n = summary["n"]
        bad = summary["bad_n"]
        rescued = summary["rescued_n"]

        if n <= 0:
            summary["final_success_rate"] = None
        else:
            summary["final_success_rate"] = (
                n - bad + rescued
            ) / n

    return summary


def load_level(experiment_id: str, occlusion: str, expect_n=None):
    run_dir = RUNS_ROOT / experiment_id / SHARED

    if not (run_dir / "trusses.csv").exists():
        raise SystemExit(f"Missing {run_dir / 'trusses.csv'}")

    summary = enrich_summary(
        summarize(run_dir),
        occlusion,
    )
    summary["experiment_id"] = experiment_id

    if expect_n is not None and summary["n"] != expect_n:
        raise SystemExit(
            f"{experiment_id}: expected N={expect_n}, got "
            f"N={summary['n']} (valid truss_path rows only)"
        )

    return summary


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for suite plots"
        ) from exc

    return plt


def _pct(rate):
    if rate is None:
        return float("nan")

    return 100.0 * rate


def plot_canonical_vs_final(levels, out_path):
    plt = _require_matplotlib()
    labels = [row["occlusion"] for row in levels]
    canonical = [
        _pct(row["canonical_success_rate"]) for row in levels
    ]
    final = [
        _pct(row["final_success_rate"]) for row in levels
    ]
    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(
        [i - width / 2 for i in x],
        canonical,
        width,
        label="Canonical success",
        color="#A65D57",
    )
    ax.bar(
        [i + width / 2 for i in x],
        final,
        width,
        label="Final success (after AP)",
        color="#4C6A92",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Occlusion level")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Canonical-view success vs final success after active perception"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_canonical_failure(levels, out_path):
    plt = _require_matplotlib()
    labels = [row["occlusion"] for row in levels]
    values = [
        _pct(row["canonical_failure_rate"]) for row in levels
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, values, color="#A65D57")
    ax.set_ylabel("Canonical failure rate (%)")
    ax.set_xlabel("Occlusion level")
    ax.set_ylim(0, 100)
    ax.set_title("Bad first views / total targets")

    for label, row, value in zip(labels, levels, values):
        ax.annotate(
            f"{row['bad_n']}/{row['n']}",
            (label, value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rescue_rate(levels, out_path):
    plt = _require_matplotlib()
    labels = [row["occlusion"] for row in levels]
    values = [_pct(row["rescue_rate"]) for row in levels]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, values, color="#4C6A92")
    ax.set_ylabel("Rescue rate (%)")
    ax.set_xlabel("Occlusion level")
    ax.set_ylim(0, 100)
    ax.set_title("Rescued / bad canonical views")

    for label, row, value in zip(labels, levels, values):
        ax.annotate(
            f"{row['rescued_n']}/{row['bad_n']}",
            (label, value if not math.isnan(value) else 0.0),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_localization(levels, out_path):
    plt = _require_matplotlib()
    labels = [row["occlusion"] for row in levels]
    x = range(len(labels))
    width = 0.2

    series = [
        (
            "canonical grasp",
            [row["mean_canonical_grasp_err"] for row in levels],
        ),
        (
            "after grasp",
            [row["mean_selected_grasp_err"] for row in levels],
        ),
        (
            "canonical cut",
            [row["mean_canonical_cut_err"] for row in levels],
        ),
        (
            "after cut",
            [row["mean_selected_cut_err"] for row in levels],
        ),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    for index, (name, values) in enumerate(series):
        vals = [
            (1000.0 * v if v is not None else float("nan"))
            for v in values
        ]
        ax.bar(
            [i + (index - 1.5) * width for i in x],
            vals,
            width,
            label=name,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean localization error (mm)")
    ax.set_xlabel("Occlusion level")
    ax.set_title(
        "Cut/grasp error before vs after (diagnostic; not an AP claim)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_views(levels, out_path):
    plt = _require_matplotlib()
    labels = [row["occlusion"] for row in levels]
    views = [row["mean_views"] or 0.0 for row in levels]
    motion = [
        row["mean_extra_motion_s"] or 0.0 for row in levels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].bar(labels, views, color="#6B8F71")
    axes[0].set_ylabel("Mean views / target")
    axes[0].set_title("Views used")
    axes[0].set_ylim(0, 3.2)

    axes[1].bar(labels, motion, color="#C4A35A")
    axes[1].set_ylabel("Mean extra camera-motion (s)")
    axes[1].set_title("Extra motion after canonical")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(levels, out_path):
    fields = [
        "occlusion",
        "experiment_id",
        "n",
        "bad_n",
        "canonical_failure_rate",
        "canonical_success_rate",
        "rescued_n",
        "not_rescued_n",
        "no_need_n",
        "rescue_rate",
        "final_success_rate",
        "mean_views",
        "mean_extra_motion_s",
        "mean_canonical_grasp_err",
        "mean_selected_grasp_err",
        "mean_canonical_cut_err",
        "mean_selected_cut_err",
    ]

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in levels:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fields
                }
            )


def write_markdown(suite, levels, out_path):
    lines = [
        f"# Matched occlusion suite: `{suite['suite_id']}`",
        "",
        f"- Scene seed: `{suite.get('scene_seed')}`",
        f"- Max targets: `{suite.get('max_targets')}`",
        f"- Locked targets: `{suite.get('locked_target_count')}`",
        f"- Protocol: `{suite.get('protocol')}`",
        f"- Levels: {', '.join(suite.get('levels') or [])}",
        "",
        "## Headline: canonical success vs final success after AP",
        "",
        "```text",
        "canonical_success = 1 - canonical_fail_rate",
        "final_success     = canonical_success",
        "                  + canonical_fail_rate × rescue_rate",
        "```",
        "",
        "| Occlusion | N | Canonical success | Final success (AP) | "
        "Δ (pp) | Rescue |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in levels:
        canon = row["canonical_success_rate"]
        final = row["final_success_rate"]
        delta = (
            None
            if canon is None or final is None
            else 100.0 * (final - canon)
        )
        lines.append(
            "| {occ} | {n} | {canon} | {final} | {delta} | {rescue} |".format(
                occ=row["occlusion"],
                n=row["n"],
                canon=fmt_rate(
                    row["n"] - row["bad_n"],
                    row["n"],
                ),
                final=(
                    f"{100.0 * final:.1f}%"
                    if final is not None
                    else "—"
                ),
                delta=(
                    f"{delta:+.1f}"
                    if delta is not None
                    else "—"
                ),
                rescue=fmt_rate(
                    row["rescued_n"],
                    row["bad_n"],
                ),
            )
        )

    lines.extend(
        [
            "",
            "### Claim scope",
            "",
            "- Active perception improves **failure recovery / "
            "robustness** (rescue of bad canonical views).",
            "- Do **not** claim monotonic improvement with occlusion.",
            "- Do **not** claim AP improves mean localization / cut "
            "accuracy.",
            "",
            "## Supporting metrics",
            "",
            "| Occlusion | Canonical fail | Rescue | Views | "
            "Extra motion | Cut before | Cut after |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in levels:
        lines.append(
            "| {occ} | {fail} | {rescue} | {views} | "
            "{motion} | {before} | {after} |".format(
                occ=row["occlusion"],
                fail=fmt_rate(row["bad_n"], row["n"]),
                rescue=fmt_rate(
                    row["rescued_n"],
                    row["bad_n"],
                ),
                views=(
                    f"{row['mean_views']:.2f}"
                    if row["mean_views"] is not None
                    else "—"
                ),
                motion=fmt_mean(
                    row["mean_extra_motion_s"],
                    unit="s",
                ),
                before=fmt_mean(
                    row["mean_canonical_cut_err"]
                ),
                after=fmt_mean(
                    row["mean_selected_cut_err"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![canonical_vs_final_success]"
            "(figures/canonical_vs_final_success.png)",
            "",
            "![canonical_failure_vs_occlusion]"
            "(figures/canonical_failure_vs_occlusion.png)",
            "",
            "![rescue_rate_vs_occlusion]"
            "(figures/rescue_rate_vs_occlusion.png)",
            "",
            "![localization_before_after]"
            "(figures/localization_before_after.png)",
            "",
            "![views_vs_occlusion](figures/views_vs_occlusion.png)",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def analyze_suite(suite_dir: Path, expect_n=None):
    suite = json.loads((suite_dir / "suite.json").read_text())
    experiments = suite.get("experiments") or {}

    if not experiments:
        raise SystemExit(
            f"No experiments listed in {suite_dir / 'suite.json'}"
        )

    levels = []

    for occlusion in LEVEL_ORDER:
        if occlusion not in experiments:
            continue

        levels.append(
            load_level(
                experiments[occlusion],
                occlusion,
                expect_n=expect_n,
            )
        )

    if not levels:
        for occlusion, exp_id in experiments.items():
            levels.append(
                load_level(
                    exp_id,
                    occlusion,
                    expect_n=expect_n,
                )
            )

    fig_dir = suite_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_canonical_vs_final(
        levels,
        fig_dir / "canonical_vs_final_success.png",
    )
    plot_canonical_failure(
        levels,
        fig_dir / "canonical_failure_vs_occlusion.png",
    )
    plot_rescue_rate(
        levels,
        fig_dir / "rescue_rate_vs_occlusion.png",
    )
    plot_localization(
        levels,
        fig_dir / "localization_before_after.png",
    )
    plot_views(
        levels,
        fig_dir / "views_vs_occlusion.png",
    )

    write_csv(levels, suite_dir / "suite_summary.csv")
    write_markdown(
        suite,
        levels,
        suite_dir / "suite_benchmark.md",
    )

    print(f"Wrote {suite_dir / 'suite_benchmark.md'}")
    print(f"Wrote {suite_dir / 'suite_summary.csv'}")
    print(f"Figures under {fig_dir}")

    for row in levels:
        print(
            f"  {row['occlusion']}: N={row['n']} | "
            f"canon_success="
            f"{fmt_rate(row['n'] - row['bad_n'], row['n'])} | "
            f"final_success="
            f"{(100.0 * row['final_success_rate']):.1f}% | "
            f"rescue={fmt_rate(row['rescued_n'], row['bad_n'])}"
        )

    return suite, levels


def _pool_levels(seed_levels):
    """
    seed_levels: list of (seed, levels_list)
    """
    by_occ = defaultdict(list)

    for seed, levels in seed_levels:
        for row in levels:
            item = dict(row)
            item["seed"] = seed
            by_occ[row["occlusion"]].append(item)

    pooled = []

    for occ in LEVEL_ORDER:
        rows = by_occ.get(occ) or []

        if not rows:
            continue

        n = sum(r["n"] for r in rows)
        bad = sum(r["bad_n"] for r in rows)
        rescued = sum(r["rescued_n"] for r in rows)
        fail = None if n <= 0 else bad / n
        canon = None if fail is None else 1.0 - fail
        rescue = None if bad <= 0 else rescued / bad
        final = None if n <= 0 else (n - bad + rescued) / n

        def _mean(key):
            vals = [
                r[key]
                for r in rows
                if r.get(key) is not None
            ]
            return (
                None
                if not vals
                else sum(vals) / len(vals)
            )

        pooled.append(
            {
                "occlusion": occ,
                "experiment_id": "pooled",
                "n": n,
                "bad_n": bad,
                "rescued_n": rescued,
                "not_rescued_n": sum(
                    r["not_rescued_n"] for r in rows
                ),
                "no_need_n": sum(
                    r["no_need_n"] for r in rows
                ),
                "canonical_failure_rate": fail,
                "canonical_success_rate": canon,
                "rescue_rate": rescue,
                "final_success_rate": final,
                "mean_views": _mean("mean_views"),
                "mean_extra_motion_s": _mean(
                    "mean_extra_motion_s"
                ),
                "mean_canonical_grasp_err": _mean(
                    "mean_canonical_grasp_err"
                ),
                "mean_selected_grasp_err": _mean(
                    "mean_selected_grasp_err"
                ),
                "mean_canonical_cut_err": _mean(
                    "mean_canonical_cut_err"
                ),
                "mean_selected_cut_err": _mean(
                    "mean_selected_cut_err"
                ),
                "seed_rows": rows,
            }
        )

    return pooled


def write_batch_markdown(batch_id, seed_levels, pooled, out_path):
    seeds = [seed for seed, _ in seed_levels]
    lines = [
        f"# Matched occlusion batch: `{batch_id}`",
        "",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        "- Protocol: shared_view_rescue_matched_manifest",
        "- Each seed × occlusion cell uses locked N=16 targets",
        "",
        "## Headline (pooled): canonical vs final success after AP",
        "",
        "```text",
        "canonical_success = 1 - canonical_fail_rate",
        "final_success     = canonical_success",
        "                  + canonical_fail_rate × rescue_rate",
        "```",
        "",
        "| Occlusion | N | Canonical success | Final success (AP) | "
        "Δ (pp) | Rescue |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in pooled:
        canon = row["canonical_success_rate"]
        final = row["final_success_rate"]
        delta = (
            None
            if canon is None or final is None
            else 100.0 * (final - canon)
        )
        lines.append(
            "| {occ} | {n} | {canon} | {final} | {delta} | {rescue} |".format(
                occ=row["occlusion"],
                n=row["n"],
                canon=fmt_rate(
                    row["n"] - row["bad_n"],
                    row["n"],
                ),
                final=(
                    f"{100.0 * final:.1f}%"
                    if final is not None
                    else "—"
                ),
                delta=(
                    f"{delta:+.1f}"
                    if delta is not None
                    else "—"
                ),
                rescue=fmt_rate(
                    row["rescued_n"],
                    row["bad_n"],
                ),
            )
        )

    lines.extend(
        [
            "",
            "### Claim scope",
            "",
            "- Active perception improves **failure recovery / "
            "robustness**.",
            "- Do **not** claim monotonic improvement with occlusion "
            "(pooled canonical-fail is not monotone here).",
            "- Do **not** claim AP improves mean localization / cut "
            "accuracy.",
            "",
            "## Per-seed cells (verify N=16)",
            "",
            "| Seed | Occlusion | N | Canonical success | "
            "Final success | Rescue |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )

    for seed, levels in seed_levels:
        for row in levels:
            lines.append(
                "| {seed} | {occ} | {n} | {canon} | {final} | "
                "{rescue} |".format(
                    seed=seed,
                    occ=row["occlusion"],
                    n=row["n"],
                    canon=fmt_rate(
                        row["n"] - row["bad_n"],
                        row["n"],
                    ),
                    final=(
                        f"{100.0 * row['final_success_rate']:.1f}%"
                        if row["final_success_rate"] is not None
                        else "—"
                    ),
                    rescue=fmt_rate(
                        row["rescued_n"],
                        row["bad_n"],
                    ),
                )
            )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![canonical_vs_final_success]"
            "(figures/canonical_vs_final_success.png)",
            "",
            "![canonical_failure_vs_occlusion]"
            "(figures/canonical_failure_vs_occlusion.png)",
            "",
            "![rescue_rate_vs_occlusion]"
            "(figures/rescue_rate_vs_occlusion.png)",
            "",
            "![views_vs_occlusion](figures/views_vs_occlusion.png)",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def analyze_batch(batch_id: str, expect_n=16):
    pattern = re.compile(
        rf"^{re.escape(batch_id)}__s(\d+)$"
    )
    seed_dirs = []

    for path in sorted(SUITES_ROOT.iterdir()):
        if not path.is_dir():
            continue

        match = pattern.match(path.name)

        if not match:
            continue

        seed_dirs.append(
            (int(match.group(1)), path)
        )

    if not seed_dirs:
        raise SystemExit(
            f"No per-seed suites matching {batch_id}__s*"
        )

    seed_levels = []

    for seed, suite_dir in seed_dirs:
        print(f"\n=== seed {seed}: {suite_dir.name} ===")
        _, levels = analyze_suite(
            suite_dir,
            expect_n=expect_n,
        )
        seed_levels.append((seed, levels))

    pooled = _pool_levels(seed_levels)
    out_dir = SUITES_ROOT / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_canonical_vs_final(
        pooled,
        fig_dir / "canonical_vs_final_success.png",
    )
    plot_canonical_failure(
        pooled,
        fig_dir / "canonical_failure_vs_occlusion.png",
    )
    plot_rescue_rate(
        pooled,
        fig_dir / "rescue_rate_vs_occlusion.png",
    )
    plot_views(
        pooled,
        fig_dir / "views_vs_occlusion.png",
    )
    write_csv(pooled, out_dir / "batch_summary.csv")
    write_batch_markdown(
        batch_id,
        seed_levels,
        pooled,
        out_dir / "batch_benchmark.md",
    )

    payload = {
        "batch_id": batch_id,
        "seeds": [seed for seed, _ in seed_levels],
        "expect_n": expect_n,
        "protocol": "shared_view_rescue_matched_manifest",
        "claim_scope": {
            "ap_improves_failure_recovery": True,
            "monotonic_with_occlusion": False,
            "mean_localization_improvement": False,
        },
    }
    (out_dir / "batch.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    print(f"\nWrote {out_dir / 'batch_benchmark.md'}")

    for row in pooled:
        print(
            f"  pooled {row['occlusion']}: N={row['n']} | "
            f"canon="
            f"{fmt_rate(row['n'] - row['bad_n'], row['n'])} | "
            f"final={100.0 * row['final_success_rate']:.1f}% | "
            f"rescue={fmt_rate(row['rescued_n'], row['bad_n'])}"
        )


def main(argv):
    args = argv[1:]
    expect_n = 16
    batch_id = None
    suite_ids = []

    i = 0

    while i < len(args):
        if args[i] == "--expect-n":
            expect_n = int(args[i + 1])
            i += 2
            continue

        if args[i] == "--batch":
            batch_id = args[i + 1]
            i += 2
            continue

        if args[i] == "--no-expect-n":
            expect_n = None
            i += 1
            continue

        suite_ids.append(args[i])
        i += 1

    if batch_id:
        analyze_batch(batch_id, expect_n=expect_n)
        return

    if not suite_ids:
        raise SystemExit(
            "Usage:\n"
            "  python3 scripts/analyze_occlusion_suite.py <suite_id>\n"
            "  python3 scripts/analyze_occlusion_suite.py "
            "--batch 20260912_095224"
        )

    for suite_id in suite_ids:
        analyze_suite(
            resolve_suite(suite_id),
            expect_n=expect_n,
        )


if __name__ == "__main__":
    main(sys.argv)
