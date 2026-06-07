#!/usr/bin/env python
"""Aggregate StyleGAN2-ADA training runs into learning curves and summary tables.

Reads each run's ``stats.jsonl`` (training-time scalars) and ``metric-*.jsonl``
(offline / training-time metric snapshots), then emits:

- ``<outdir>/learning_curves.csv``: one row per (run, kimg) with every metric and
  the key training scalars (ADA p, G/D loss, D scores, sec/kimg, peak GPU mem).
- ``<outdir>/summary.csv``: one row per run with the final/best metrics and the
  total training cost (hours, mean sec/kimg).
- Optional PNG curves (FID vs kimg, ADA p vs kimg, G/D loss vs kimg) when
  matplotlib is available.
- With ``--fair-kimg``: ``<outdir>/fair_comparison.csv`` (+ bar plot), one row per
  run at the snapshot closest to that kimg, for a matched-budget ranking.
- With ``--fair-kimg`` and ``--seed-group``: ``<outdir>/seed_aggregate.csv``
  (+ ``seed_fid_comparison.png``), one row per configuration giving the
  mean/min/max/range of each metric across its seeds.

Usage::

    python scripts/analyze_results.py \
        --run E1=results/runs/p1-lsun-church256-100k-baseline \
        --run E2=results/runs/p2-lsun-church256-100k-noada-1500 \
        --outdir results/analysis

    # Cross-seed aggregation at a matched budget (E1+E1b are the same config,
    # different seeds):
    python scripts/analyze_results.py \
        --run E1=<run> --run E1b=<run> --run E2=<run> --run E2b=<run> \
        --fair-kimg 1500 --seed-group E1=E1,E1b --seed-group E2=E2,E2b \
        --outdir results/analysis

The run path may point either at a training subfolder (the one that contains
``stats.jsonl``) or at the parent ``outdir`` that holds numbered subfolders; in
the latter case the most recently modified subfolder is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import resolve_path  # noqa: E402


SNAPSHOT_RE = re.compile(r"network-snapshot-(\d+)\.pkl")

# Training scalars worth carrying into the learning-curve table, mapped to short
# column names. Each value in stats.jsonl is a {"mean", "std", "num"} dict.
STAT_COLUMNS = {
    "Progress/augment": "ada_p",
    "Loss/G/loss": "g_loss",
    "Loss/D/loss": "d_loss",
    "Loss/scores/real": "d_score_real",
    "Loss/scores/fake": "d_score_fake",
    "Loss/r1_penalty": "r1_penalty",
    "Timing/sec_per_kimg": "sec_per_kimg",
    "Resources/peak_gpu_mem_gb": "peak_gpu_mem_gb",
    "Timing/total_hours": "total_hours",
}


def _mean(entry: Any) -> Optional[float]:
    if isinstance(entry, dict):
        return entry.get("mean")
    if isinstance(entry, (int, float)):
        return float(entry)
    return None


def find_run_dir(path: Path) -> Path:
    """Return the folder that actually contains stats.jsonl for ``path``."""
    if (path / "stats.jsonl").is_file():
        return path
    candidates = sorted(
        path.glob("**/stats.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No stats.jsonl found under {}".format(path))
    return candidates[-1].parent


def parse_stats(run_dir: Path) -> Dict[int, Dict[str, float]]:
    """Map rounded kimg -> selected training scalars."""
    rows: Dict[int, Dict[str, float]] = {}
    stats_file = run_dir / "stats.jsonl"
    with stats_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kimg = _mean(record.get("Progress/kimg"))
            if kimg is None:
                continue
            key = int(round(kimg))
            row = rows.setdefault(key, {})
            for stat_key, column in STAT_COLUMNS.items():
                value = _mean(record.get(stat_key))
                if value is not None:
                    row[column] = value
    return rows


def parse_metrics(run_dir: Path) -> Dict[int, Dict[str, float]]:
    """Map snapshot kimg -> flattened metric results from every metric-*.jsonl."""
    rows: Dict[int, Dict[str, float]] = {}
    for metric_file in sorted(run_dir.glob("metric-*.jsonl")):
        with metric_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                snapshot = record.get("snapshot_pkl", "")
                match = SNAPSHOT_RE.search(snapshot)
                if not match:
                    continue
                kimg = int(match.group(1))
                row = rows.setdefault(kimg, {})
                for name, value in record.get("results", {}).items():
                    if isinstance(value, (int, float)):
                        row[name] = float(value)
    return rows


def collect_run(name: str, path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    run_dir = find_run_dir(path)
    stats = parse_stats(run_dir)
    metrics = parse_metrics(run_dir)

    all_kimg = sorted(set(stats) | set(metrics))
    curve: List[Dict[str, Any]] = []
    for kimg in all_kimg:
        row: Dict[str, Any] = {"run": name, "kimg": kimg}
        row.update(stats.get(kimg, {}))
        row.update(metrics.get(kimg, {}))
        curve.append(row)

    # Summary: final training scalars + best/last metric values.
    summary: Dict[str, Any] = {"run": name, "run_dir": str(run_dir)}
    if stats:
        last_kimg = max(stats)
        summary["final_kimg"] = last_kimg
        for column in ("total_hours", "peak_gpu_mem_gb", "ada_p"):
            if column in stats[last_kimg]:
                summary[column] = stats[last_kimg][column]
        sec_values = [r["sec_per_kimg"] for r in stats.values() if "sec_per_kimg" in r]
        if sec_values:
            summary["mean_sec_per_kimg"] = sum(sec_values) / len(sec_values)
    # Best (minimum) FID and the kimg where it occurred, plus latest of every metric.
    metric_names = sorted({k for row in metrics.values() for k in row})
    for metric_name in metric_names:
        points = [(kimg, row[metric_name]) for kimg, row in sorted(metrics.items()) if metric_name in row]
        if not points:
            continue
        summary["{}_final".format(metric_name)] = points[-1][1]
        if "fid" in metric_name:
            best_kimg, best_val = min(points, key=lambda kv: kv[1])
            summary["{}_best".format(metric_name)] = best_val
            summary["{}_best_kimg".format(metric_name)] = best_kimg
    return curve, summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


FAIR_METRICS = [
    "fid50k_full",
    "kid50k_full",
    "pr50k3_full_precision",
    "pr50k3_full_recall",
]


def fair_comparison(
    curves: Dict[str, List[Dict[str, Any]]], target_kimg: int
) -> List[Dict[str, Any]]:
    """One row per run: the metric snapshot whose kimg is closest to ``target_kimg``.

    Runs use slightly different snapshot grids (e.g. E1 lands on 1512 while the
    1500-budget runs land on 1500), so an exact kimg match is not guaranteed; the
    nearest snapshot that actually carries an FID value is used instead. This
    aligns the matrix at a common training budget rather than mixing each run's
    own final kimg.
    """
    rows: List[Dict[str, Any]] = []
    for name, curve in curves.items():
        points = [r for r in curve if r.get("fid50k_full") is not None]
        if not points:
            continue
        chosen = min(points, key=lambda r: abs(r["kimg"] - target_kimg))
        row: Dict[str, Any] = {
            "run": name,
            "target_kimg": target_kimg,
            "snapshot_kimg": chosen["kimg"],
        }
        for metric in FAIR_METRICS:
            row[metric] = chosen.get(metric)
        rows.append(row)
    return rows


def make_fair_plot(rows: List[Dict[str, Any]], outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print("[fair plot skipped] matplotlib unavailable: {}".format(exc))
        return

    bars = [(r["run"], r["fid50k_full"]) for r in rows if r.get("fid50k_full") is not None]
    if not bars:
        return
    names, vals = zip(*bars)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = range(len(names))
    ax.bar(positions, vals, color="#4C72B0")
    for x, v in zip(positions, vals):
        ax.text(x, v, "{:.2f}".format(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names)
    ax.set_ylabel("FID50k_full")
    ax.set_title("FID at matched budget (~{} kimg)".format(rows[0]["target_kimg"]))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fair_fid_comparison.png", dpi=150)
    plt.close(fig)
    print("[plot] {}".format(outdir / "fair_fid_comparison.png"))


def seed_aggregate(
    fair_rows: List[Dict[str, Any]], groups: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """One row per seed-group: mean / min / max / range of each metric across seeds.

    ``groups`` maps a group label (e.g. ``E1``) to the run names that are the same
    configuration under different seeds (e.g. ``["E1", "E1b"]``). The aggregate is
    computed over the fair-budget snapshot of each member, so every seed is compared
    at the same kimg. ``range`` is ``max - min`` and doubles as the error-bar half-
    span only loosely (it is the full span for n=2). ``num_gpus``-style metadata is
    not aggregated; only the FAIR_METRICS are.
    """
    by_name = {row["run"]: row for row in fair_rows}
    rows: List[Dict[str, Any]] = []
    for group, members in groups.items():
        present = [by_name[m] for m in members if m in by_name]
        if not present:
            print("[seed-group skipped] {}: no members found in fair rows".format(group))
            continue
        row: Dict[str, Any] = {
            "group": group,
            "n_seeds": len(present),
            "members": "+".join(m for m in members if m in by_name),
            "snapshot_kimg": "/".join(str(r.get("snapshot_kimg")) for r in present),
        }
        for metric in FAIR_METRICS:
            vals = [r[metric] for r in present if r.get(metric) is not None]
            if not vals:
                row[metric + "_mean"] = None
                row[metric + "_min"] = None
                row[metric + "_max"] = None
                row[metric + "_range"] = None
                continue
            lo, hi = min(vals), max(vals)
            row[metric + "_mean"] = sum(vals) / len(vals)
            row[metric + "_min"] = lo
            row[metric + "_max"] = hi
            row[metric + "_range"] = hi - lo
        rows.append(row)
    return rows


def make_seed_plot(agg_rows: List[Dict[str, Any]], outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print("[seed plot skipped] matplotlib unavailable: {}".format(exc))
        return

    bars = [r for r in agg_rows if r.get("fid50k_full_mean") is not None]
    if not bars:
        return
    names = [r["group"] for r in bars]
    means = [r["fid50k_full_mean"] for r in bars]
    # Asymmetric error bars from the observed min/max around the mean.
    lower = [r["fid50k_full_mean"] - r["fid50k_full_min"] for r in bars]
    upper = [r["fid50k_full_max"] - r["fid50k_full_mean"] for r in bars]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = range(len(names))
    ax.bar(positions, means, color="#4C72B0", yerr=[lower, upper], capsize=6,
           error_kw={"ecolor": "#333333", "elinewidth": 1.2})
    for x, r in zip(positions, bars):
        ax.text(x, r["fid50k_full_max"], "{:.2f}".format(r["fid50k_full_mean"]),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(["{}\n(n={})".format(r["group"], r["n_seeds"]) for r in bars])
    ax.set_ylabel("FID50k_full (mean, bars = seed min/max)")
    ax.set_title("FID across seeds at matched budget")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "seed_fid_comparison.png", dpi=150)
    plt.close(fig)
    print("[plot] {}".format(outdir / "seed_fid_comparison.png"))


def make_plots(curves: Dict[str, List[Dict[str, Any]]], outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print("[plots skipped] matplotlib unavailable: {}".format(exc))
        return

    panels = [
        ("fid50k_full", "FID50k_full", "fid_learning_curve.png", True),
        ("ada_p", "ADA probability", "ada_p_curve.png", False),
        ("g_loss", "G loss", "g_loss_curve.png", False),
        ("d_loss", "D loss", "d_loss_curve.png", False),
    ]
    for column, ylabel, filename, log_y in panels:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for name, rows in curves.items():
            xy = [(r["kimg"], r[column]) for r in rows if column in r and r[column] is not None]
            if not xy:
                continue
            xy.sort()
            xs, ys = zip(*xy)
            ax.plot(xs, ys, marker="o", markersize=3, label=name)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("kimg")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=150)
        plt.close(fig)
        print("[plot] {}".format(outdir / filename))


def parse_group_arg(value: str) -> Tuple[str, List[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--seed-group expects GROUP=NAME1,NAME2, got: {}".format(value)
        )
    group, raw_members = value.split("=", 1)
    group = group.strip()
    members = [m.strip() for m in raw_members.split(",") if m.strip()]
    if not group or not members:
        raise argparse.ArgumentTypeError(
            "--seed-group needs a label and >=1 run name in: {}".format(value)
        )
    return group, members


def parse_run_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--run expects NAME=PATH, got: {}".format(value)
        )
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("--run name is empty in: {}".format(value))
    return name, resolve_path(raw_path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_arg,
        required=True,
        metavar="NAME=PATH",
        help="Label and path of a run (repeatable). Path may be the run subfolder or its parent outdir.",
    )
    parser.add_argument("--outdir", default="results/analysis")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--fair-kimg",
        type=int,
        default=None,
        help="Emit fair_comparison.csv aligning every run at the snapshot closest to this kimg.",
    )
    parser.add_argument(
        "--seed-group",
        action="append",
        type=parse_group_arg,
        default=None,
        metavar="GROUP=NAME1,NAME2",
        help="Aggregate the named runs as one configuration across seeds (repeatable). "
        "Requires --fair-kimg; emits seed_aggregate.csv + seed_fid_comparison.png.",
    )
    args = parser.parse_args()

    outdir = resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_curve_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    curves: Dict[str, List[Dict[str, Any]]] = {}
    for name, path in args.run:
        curve, summary = collect_run(name, path)
        curves[name] = curve
        all_curve_rows.extend(curve)
        summaries.append(summary)
        fid_best = summary.get("fid50k_full_best")
        print(
            "[{}] {} -> {} curve points, final_kimg={}, best FID={}".format(
                name,
                summary.get("run_dir", path),
                len(curve),
                summary.get("final_kimg"),
                round(fid_best, 3) if isinstance(fid_best, float) else fid_best,
            )
        )

    write_csv(outdir / "learning_curves.csv", all_curve_rows)
    write_csv(outdir / "summary.csv", summaries)
    print("[csv] {}".format(outdir / "learning_curves.csv"))
    print("[csv] {}".format(outdir / "summary.csv"))

    if args.fair_kimg is not None:
        fair_rows = fair_comparison(curves, args.fair_kimg)
        write_csv(outdir / "fair_comparison.csv", fair_rows)
        print("[csv] {}".format(outdir / "fair_comparison.csv"))
        for row in fair_rows:
            print(
                "  [fair] {} @ {} kimg: FID={}".format(
                    row["run"],
                    row["snapshot_kimg"],
                    round(row["fid50k_full"], 3) if row.get("fid50k_full") else None,
                )
            )
        if not args.no_plots:
            make_fair_plot(fair_rows, outdir)

        if args.seed_group:
            groups = dict(args.seed_group)
            agg_rows = seed_aggregate(fair_rows, groups)
            write_csv(outdir / "seed_aggregate.csv", agg_rows)
            print("[csv] {}".format(outdir / "seed_aggregate.csv"))
            for row in agg_rows:
                mean = row.get("fid50k_full_mean")
                print(
                    "  [seed] {} (n={}): FID mean={} range=[{}, {}]".format(
                        row["group"],
                        row["n_seeds"],
                        round(mean, 3) if isinstance(mean, float) else mean,
                        round(row["fid50k_full_min"], 3) if row.get("fid50k_full_min") else None,
                        round(row["fid50k_full_max"], 3) if row.get("fid50k_full_max") else None,
                    )
                )
            if not args.no_plots:
                make_seed_plot(agg_rows, outdir)
    elif args.seed_group:
        print("[seed-group ignored] --seed-group requires --fair-kimg")

    if not args.no_plots:
        make_plots(curves, outdir)


if __name__ == "__main__":
    main()
