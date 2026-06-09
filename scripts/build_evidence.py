#!/usr/bin/env python
"""Build and validate the project evidence package from retained artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_results import collect_run, parse_metrics  # noqa: E402
from stylegan_course.project import resolve_path  # noqa: E402


RUNS: Dict[str, Dict[str, Any]] = {
    "E1": {
        "config": "configs/baseline/p1_lsun_church256_baseline.json",
        "run_dir": (
            "results/runs/p1-lsun-church256-100k-baseline/"
            "00003-lsun-church-256-100k-mirror-paper256-kimg2000-ada-target0.6-bgc"
        ),
        "fair_kimg": 1512,
        "schedule": "single two-GPU run",
        "scope": "central baseline; 2000 kimg retained for final display",
    },
    "E2": {
        "config": "configs/baseline/p2_lsun_church256_noada_1500.json",
        "run_dir": (
            "results/runs/p2-lsun-church256-100k-noada-1500/"
            "00003-lsun-church-256-100k-mirror-paper256-kimg1500-noaug"
        ),
        "fair_kimg": 1500,
        "schedule": "two-GPU run; one of two runs scheduled on a four-GPU machine",
        "scope": "no-augmentation control; dataset x-flip remains enabled",
    },
    "E3": {
        "config": "configs/baseline/p2_lsun_church256_fixedp02_1500.json",
        "run_dir": (
            "results/runs/p2-lsun-church256-100k-fixedp02-1500/"
            "00003-lsun-church-256-100k-mirror-paper256-kimg1500-fixed-p0.2-bgc"
        ),
        "fair_kimg": 1500,
        "schedule": "two-GPU run; one of two runs scheduled on a four-GPU machine",
        "scope": "single-seed exploratory fixed-augmentation control",
    },
    "E4": {
        "config": "configs/baseline/p2_lsun_church256_subset50k_ada_1500.json",
        "run_dir": (
            "results/runs/p2-lsun-church256-50k-ada-1500/"
            "00000-lsun-church-256-100k-subset50000-mirror-paper256-kimg1500-"
            "ada-target0.6-bgc"
        ),
        "fair_kimg": 1500,
        "schedule": "two-GPU run; one of two runs scheduled on a four-GPU machine",
        "scope": "50k condition at fixed image-exposure budget",
    },
    "E5": {
        "config": "configs/baseline/p2_lsun_church256_target04_1500.json",
        "run_dir": (
            "results/runs/p2-lsun-church256-100k-ada-target04-1500/"
            "00000-lsun-church-256-100k-mirror-paper256-kimg1500-ada-target0.4-bgc"
        ),
        "fair_kimg": 1500,
        "schedule": "two-GPU run; one of two runs scheduled on a four-GPU machine",
        "scope": "single-seed exploratory ADA-target sensitivity result",
    },
    "E1b": {
        "config": "configs/baseline/p2_lsun_church256_baseline_seed1_1500.json",
        "run_dir": (
            "results/runs/p2-lsun-church256-100k-ada-seed1-1500/"
            "00000-lsun-church-256-100k-mirror-paper256-kimg1500-ada-target0.6-bgc"
        ),
        "fair_kimg": 1500,
        "schedule": "two-GPU run; one of three runs scheduled on a six-GPU machine",
        "scope": "second random condition for E1",
    },
    "E2b": {
        "config": "configs/baseline/p2_lsun_church256_noada_seed1_1500.json",
        "run_dir": (
            "results/runs/p2-lsun-church256-100k-noada-seed1-1500/"
            "00000-lsun-church-256-100k-mirror-paper256-kimg1500-noaug"
        ),
        "fair_kimg": 1500,
        "schedule": "two-GPU run; one of three runs scheduled on a six-GPU machine",
        "scope": "second random condition for E2",
    },
    "E4b": {
        "config": "configs/baseline/p2_lsun_church256_subset50k_ada_seed1_1500.json",
        "run_dir": (
            "results/runs/p2-lsun-church256-50k-ada-seed1-1500/"
            "00000-lsun-church-256-100k-subset50000-mirror-paper256-kimg1500-"
            "ada-target0.6-bgc"
        ),
        "fair_kimg": 1500,
        "schedule": "two-GPU run; one of three runs scheduled on a six-GPU machine",
        "scope": "second random condition for E4; also selects a different 50k subset",
    },
}

METRICS = (
    "fid50k_full",
    "kid50k_full",
    "pr50k3_full_precision",
    "pr50k3_full_recall",
)

INDEPENDENT_SAMPLE_GROUPS = {
    "E1-final-trunc07": {
        "directory": "results/samples/show_trunc07",
        "count": 24,
        "present": {4, 15, 16},
        "possible": set(),
    },
    "E2-noaug": {
        "directory": "results/samples/p2-lsun-church256-100k-noada-1500",
        "count": 16,
        "present": {2},
        "possible": set(),
    },
    "E3-fixedp02": {
        "directory": "results/samples/p2-lsun-church256-100k-fixedp02-1500",
        "count": 16,
        "present": {2},
        "possible": set(),
    },
    "E4-50k-ada": {
        "directory": "results/samples/p2-lsun-church256-50k-ada-1500",
        "count": 16,
        "present": {2},
        "possible": set(),
    },
    "E5-target04": {
        "directory": "results/samples/p2-lsun-church256-100k-ada-target04-1500",
        "count": 16,
        "present": {10, 13},
        "possible": {7},
    },
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_sources(
    run_dir: Path, snapshot_kimg: int
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, int]]:
    metrics, metadata = parse_metrics(run_dir)
    if snapshot_kimg not in metrics:
        raise ValueError("{} has no metrics at {} kimg".format(run_dir, snapshot_kimg))
    values = metrics[snapshot_kimg]
    missing = [name for name in METRICS if name not in values]
    if missing:
        raise ValueError(
            "{} missing metrics at {} kimg: {}".format(
                run_dir, snapshot_kimg, ", ".join(missing)
            )
        )
    sources = {
        name: relative(run_dir / metadata[(snapshot_kimg, name)]["source_file"])
        for name in METRICS
    }
    counts = {
        name: int(metadata[(snapshot_kimg, name)]["count"]) for name in METRICS
    }
    return values, sources, counts


def build_manifest() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, spec in RUNS.items():
        config_path = resolve_path(spec["config"])
        run_dir = resolve_path(spec["run_dir"])
        if not config_path.is_file() or not run_dir.is_dir():
            raise FileNotFoundError("{} inputs are incomplete".format(name))
        config = load_json(config_path)
        train = config["train"]
        curve, summary = collect_run(name, run_dir)
        del curve
        fair_kimg = int(spec["fair_kimg"])
        values, sources, counts = metric_sources(run_dir, fair_kimg)
        options_path = run_dir / "training_options.json"
        options = load_json(options_path) if options_path.is_file() else {}
        dataset_size = int(train.get("subset", 100000))
        aug = str(train["aug"])
        if aug == "ada":
            aug_parameter = "target={}".format(train["target"])
        elif aug == "fixed":
            aug_parameter = "p={}".format(train["p"])
        else:
            aug_parameter = "none"
        snapshot = run_dir / "network-snapshot-{:06d}.pkl".format(fair_kimg)
        rows.append(
            {
                "run": name,
                "config_path": relative(config_path),
                "run_dir": relative(run_dir),
                "dataset_path": config["data"],
                "dataset_size": dataset_size,
                "resolution": options.get("training_set_kwargs", {}).get(
                    "resolution", 256
                ),
                "train_seed": train["seed"],
                "augmentation": aug,
                "augmentation_parameter": aug_parameter,
                "mirror_train": train["mirror"],
                "budget_kimg": train["kimg"],
                "fair_snapshot_kimg": fair_kimg,
                "fair_snapshot_pkl": snapshot.name,
                "fair_snapshot_path": (
                    relative(snapshot) if snapshot.is_file() else "not retained"
                ),
                "fair_snapshot_retained": snapshot.is_file(),
                "train_gpus_per_run": train["gpus"],
                "machine_schedule_context": spec["schedule"],
                "evaluation_gpus": config["evaluate"]["gpus"],
                "evaluation_mirror": config["evaluate"].get("mirror", False),
                "fid50k_full": values["fid50k_full"],
                "kid50k_full": values["kid50k_full"],
                "precision": values["pr50k3_full_precision"],
                "recall": values["pr50k3_full_recall"],
                "final_training_kimg": summary.get("final_kimg"),
                "final_ada_p": summary.get("ada_p"),
                "total_hours": summary.get("total_hours"),
                "peak_gpu_mem_gb": summary.get("peak_gpu_mem_gb"),
                "mean_sec_per_kimg": summary.get("mean_sec_per_kimg"),
                "fid_source": sources["fid50k_full"],
                "fid_eval_count": counts["fid50k_full"],
                "kid_source": sources["kid50k_full"],
                "kid_eval_count": counts["kid50k_full"],
                "pr_source": sources["pr50k3_full_precision"],
                "pr_eval_count": counts["pr50k3_full_precision"],
                "stats_source": relative(run_dir / "stats.jsonl"),
                "training_options_source": (
                    relative(options_path) if options_path.is_file() else "not retained"
                ),
                "protocol_source": relative(config_path),
                "interpretation_scope": spec["scope"],
            }
        )
    return rows


def build_claims() -> List[Dict[str, str]]:
    return [
        {
            "claim_id": "C1",
            "strength": "strong within retained two-seed experiment",
            "allowed_wording": (
                "At the matched budget, ADA improved FID relative to noaug in both "
                "retained random conditions."
            ),
            "evidence": (
                "evidence/experiment_manifest.csv (E1/E1b versus E2/E2b); "
                "results/analysis/seed_aggregate.csv"
            ),
            "limitation": "n=2 conditions; no statistical significance claim.",
        },
        {
            "claim_id": "C2",
            "strength": "bounded repeated observation",
            "allowed_wording": (
                "Under the fixed 1500 kimg image-exposure budget, the retained 50k "
                "conditions did not show worse FID than their 100k counterparts."
            ),
            "evidence": (
                "evidence/experiment_manifest.csv (E1/E1b and E4/E4b); "
                "results/analysis/seed_aggregate.csv"
            ),
            "limitation": (
                "50k sees more equivalent epochs; E4/E4b also use different "
                "deterministically sampled subsets. Do not claim less data is better."
            ),
        },
        {
            "claim_id": "C3",
            "strength": "exploratory single-seed result",
            "allowed_wording": (
                "Fixed p=0.2 produced the highest precision and lowest recall among "
                "the matched-budget seed=42 groups."
            ),
            "evidence": "evidence/experiment_manifest.csv (E3)",
            "limitation": "No second seed; do not generalize beyond this experiment.",
        },
        {
            "claim_id": "C4",
            "strength": "exploratory single-seed result",
            "allowed_wording": (
                "ADA target=0.4 had the best single-seed FID/KID in the retained "
                "matched-budget matrix."
            ),
            "evidence": "evidence/experiment_manifest.csv (E5)",
            "limitation": "No second seed; do not call target=0.4 universally optimal.",
        },
        {
            "claim_id": "C5",
            "strength": "scoped visual observation",
            "allowed_wording": (
                "Watermark-like contamination is reproduced in retained generated "
                "samples and persists through part of the interpolation trajectory."
            ),
            "evidence": (
                "evidence/visual/watermark_annotations.csv; "
                "evidence/visual/interpolation_watermark_audit.csv; "
                "evidence/figures/watermark_trajectory.png"
            ),
            "limitation": (
                "This is evidence of learned contamination, not proof that a specific "
                "training image was memorized."
            ),
        },
        {
            "claim_id": "C6",
            "strength": "scoped nearest-neighbor audit",
            "allowed_wording": (
                "In the current eight-sample Inception-feature nearest-neighbor audit, "
                "no near-pixel copy of a training sample was observed."
            ),
            "evidence": (
                "evidence/nearest_neighbor_audit.md; "
                "results/nn/e1-final/neighbors.csv; results/nn/e1-final/nn_seed*.png"
            ),
            "limitation": (
                "Eight generated samples and one feature space cannot establish the "
                "absence of memorization for all outputs."
            ),
        },
        {
            "claim_id": "C7",
            "strength": "direct protocol evidence",
            "allowed_wording": (
                "Each formal training run used two GPUs; four- and six-GPU machines "
                "were used to schedule multiple independent two-GPU runs in parallel."
            ),
            "evidence": (
                "evidence/experiment_manifest.csv; configs/baseline/*.json; "
                "scripts/run_p2_parallel.sh; scripts/run_p2_seed1.sh"
            ),
            "limitation": "Do not describe a single model as four- or six-GPU training.",
        },
    ]


def ensure_visual_tables(evidence_dir: Path) -> Tuple[List[Dict[str, str]], ...]:
    visual = evidence_dir / "visual"
    failures = read_csv(visual / "failure_cases.csv")
    independent = read_csv(visual / "watermark_annotations.csv")
    interpolation = read_csv(visual / "interpolation_watermark_audit.csv")
    if len(failures) != 64:
        raise ValueError("failure_cases.csv must contain 64 rows")
    if len(independent) != 88:
        raise ValueError("watermark_annotations.csv must contain 88 rows")
    if len(interpolation) != 151:
        raise ValueError("interpolation_watermark_audit.csv must contain 151 rows")
    for rows, label in (
        (failures, "failure"),
        (independent, "watermark"),
        (interpolation, "interpolation"),
    ):
        for row in rows:
            source = resolve_path(row["source_path"])
            if not source.is_file():
                raise FileNotFoundError("{} annotation missing {}".format(label, source))
    return failures, independent, interpolation


def write_watermark_audits(evidence_dir: Path) -> None:
    visual = evidence_dir / "visual"
    independent: List[Dict[str, Any]] = []
    for group, spec in INDEPENDENT_SAMPLE_GROUPS.items():
        for seed in range(spec["count"]):
            if seed in spec["present"]:
                status = "present"
                confidence = "high"
                notes = "readable or strongly letter-like stock-photo watermark artifact"
            elif seed in spec["possible"]:
                status = "possible"
                confidence = "medium"
                notes = "faint letter-like artifact; retained separately from confirmed cases"
            else:
                status = "absent"
                confidence = "high"
                notes = "no watermark-like text observed in manual review"
            independent.append(
                {
                    "group": group,
                    "seed": seed,
                    "source_path": "{}/seed{:04d}.png".format(
                        spec["directory"], seed
                    ),
                    "watermark": status,
                    "confidence": confidence,
                    "notes": notes,
                }
            )
    write_csv(
        visual / "watermark_annotations.csv",
        independent,
        ["group", "seed", "source_path", "watermark", "confidence", "notes"],
    )

    interpolation: List[Dict[str, Any]] = []
    for frame in range(151):
        if 103 <= frame <= 148:
            status = "present"
            confidence = "high"
            notes = "letter-like watermark persists continuously through this segment"
        elif frame in {102, 149}:
            status = "possible"
            confidence = "medium"
            notes = "transition frame with faint partial letter-like artifact"
        else:
            status = "absent"
            confidence = "high"
            notes = "no watermark-like text observed"
        interpolation.append(
            {
                "frame": frame,
                "source_path": (
                    "results/samples/interp_e1/frame_{:04d}.png".format(frame)
                ),
                "watermark": status,
                "confidence": confidence,
                "notes": notes,
            }
        )
    write_csv(
        visual / "interpolation_watermark_audit.csv",
        interpolation,
        ["frame", "source_path", "watermark", "confidence", "notes"],
    )


def image_grid(
    entries: Sequence[Tuple[Path, str]],
    output: Path,
    columns: int,
    tile_size: int = 256,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    label_height = 30
    rows = math.ceil(len(entries) / columns)
    canvas = Image.new(
        "RGB", (columns * tile_size, rows * (tile_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (path, label) in enumerate(entries):
        row, column = divmod(index, columns)
        with Image.open(path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((tile_size, tile_size))
            x = column * tile_size + (tile_size - tile.width) // 2
            y = row * (tile_size + label_height) + label_height
            canvas.paste(tile, (x, y))
        draw.text(
            (column * tile_size + 5, row * (tile_size + label_height) + 8),
            label,
            fill="black",
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_figures(
    evidence_dir: Path,
    failures: List[Dict[str, str]],
    interpolation: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    figure_dir = evidence_dir / "figures"
    comparison_groups = [
        ("E2 noaug", "results/samples/p2-lsun-church256-100k-noada-1500"),
        ("E3 fixed p=0.2", "results/samples/p2-lsun-church256-100k-fixedp02-1500"),
        ("E4 50k ADA", "results/samples/p2-lsun-church256-50k-ada-1500"),
        (
            "E5 ADA target=0.4",
            "results/samples/p2-lsun-church256-100k-ada-target04-1500",
        ),
    ]
    comparison_entries: List[Tuple[Path, str]] = []
    for group, directory in comparison_groups:
        for seed in (0, 3, 7, 12):
            comparison_entries.append(
                (
                    resolve_path("{}/seed{:04d}.png".format(directory, seed)),
                    "{} seed{:04d}".format(group, seed),
                )
            )
    comparison_path = figure_dir / "p2_fixed_seed_comparison.png"
    image_grid(comparison_entries, comparison_path, columns=4)

    severity_rank = {"high": 0, "medium": 1, "low": 2, "none": 3}
    selected = sorted(
        failures,
        key=lambda row: (
            severity_rank.get(row["severity"].lower(), 9),
            row["group"],
            int(row["seed"]),
        ),
    )[:12]
    failure_path = figure_dir / "failure_cases_grid.png"
    image_grid(
        [
            (
                resolve_path(row["source_path"]),
                "{} s{} {}: {}".format(
                    row["group"], row["seed"], row["severity"], row["notes"][:30]
                ),
            )
            for row in selected
        ],
        failure_path,
        columns=4,
    )

    trajectory_frames = [0, 30, 60, 90, 105, 115, 125, 135, 145, 150]
    by_frame = {int(row["frame"]): row for row in interpolation}
    trajectory_path = figure_dir / "watermark_trajectory.png"
    image_grid(
        [
            (
                resolve_path(by_frame[frame]["source_path"]),
                "frame {:03d}: {}".format(frame, by_frame[frame]["watermark"]),
            )
            for frame in trajectory_frames
        ],
        trajectory_path,
        columns=5,
    )

    return [
        {
            "figure_id": "F1",
            "path": relative(comparison_path),
            "purpose": "Same-seed qualitative comparison across four retained P2 groups.",
            "source_scope": "seeds 0, 3, 7, 12; no E1 fair-point samples retained",
            "restriction": "Do not present as a five-group E1-E5 comparison.",
        },
        {
            "figure_id": "F2",
            "path": relative(failure_path),
            "purpose": "Representative high/medium severity generated-image failures.",
            "source_scope": "selected from the complete 64-image annotation table",
            "restriction": "Selection illustrates failure modes; it is not a frequency plot.",
        },
        {
            "figure_id": "F3",
            "path": relative(trajectory_path),
            "purpose": "Watermark-like contamination along the E1 W-space interpolation.",
            "source_scope": "10 selected frames from the independently audited 151-frame trajectory",
            "restriction": "Trajectory frames are correlated and excluded from independent-sample rates.",
        },
        {
            "figure_id": "F4",
            "path": "results/samples/stylemix/grid.png",
            "purpose": "Style-mixing and contamination-transfer example.",
            "source_scope": "retained E1 final-model style-mixing grid",
            "restriction": "Not used for watermark frequency estimates.",
        },
        {
            "figure_id": "F5",
            "path": "results/nn/e1-final/nn_seed0000.png",
            "purpose": "Example generated image with three nearest real images.",
            "source_scope": "one of eight audited generated seeds",
            "restriction": "Read together with the full nearest-neighbor audit.",
        },
    ]


def audit_nearest_neighbors(evidence_dir: Path) -> None:
    nn_dir = resolve_path("results/nn/e1-final")
    rows = read_csv(nn_dir / "neighbors.csv")
    errors: List[str] = []
    if len(rows) != 24:
        errors.append("expected 24 CSV rows, found {}".format(len(rows)))
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    if sorted(grouped) != list(range(8)):
        errors.append("expected seeds 0-7, found {}".format(sorted(grouped)))
    for seed, entries in grouped.items():
        entries.sort(key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in entries]
        distances = [float(row["distance"]) for row in entries]
        if ranks != [1, 2, 3]:
            errors.append("seed {} ranks are {}".format(seed, ranks))
        if distances != sorted(distances):
            errors.append("seed {} distances are not monotonic".format(seed))
        montage = nn_dir / "nn_seed{:04d}.png".format(seed)
        if not montage.is_file():
            errors.append("missing {}".format(montage))
    if errors:
        raise ValueError("; ".join(errors))

    distance_min = min(float(row["distance"]) for row in rows)
    distance_max = max(float(row["distance"]) for row in rows)
    text = """# E1 最近邻审计

## 核验结果

- 生成样本：8 个，seed 0-7。
- 每个生成样本：3 个训练集近邻。
- `neighbors.csv`：24 行，rank 均为 1-3，距离按 rank 单调不减。
- 拼图：8 张，均与 CSV 中的 seed 一一对应。
- 距离范围：{distance_min:.4f}-{distance_max:.4f}。
- 特征空间：StyleGAN2-ADA 官方 FID Inception 特征，L2 距离。
- 真实集扫描：脚本默认扫描完整训练集；本地已不保留正式 ZIP，无法在当前机器重跑。

## 允许表述

> 在当前 8 个生成样本的 Inception 特征近邻审计中，没有观察到近乎像素复制的训练样本。

## 限制

该结果只覆盖 8 个生成 seed、一个模型快照和一种特征空间。它不能证明模型不存在记忆，
也不能推广为全部生成图均不是训练样本复制。水印现象应解释为数据污染特征的学习与复现，
不能仅凭水印认定模型记忆了某一张训练图。
""".format(
        distance_min=distance_min, distance_max=distance_max
    )
    (evidence_dir / "nearest_neighbor_audit.md").write_text(text, encoding="utf-8")


def validate_references(evidence_dir: Path, figures: Sequence[Dict[str, str]]) -> None:
    for figure in figures:
        path = resolve_path(figure["path"])
        if not path.is_file():
            raise FileNotFoundError("figure source missing: {}".format(path))
    required = [
        evidence_dir / "experiment_manifest.csv",
        evidence_dir / "claim_evidence_matrix.csv",
        evidence_dir / "visual/failure_cases.csv",
        evidence_dir / "visual/watermark_annotations.csv",
        evidence_dir / "visual/interpolation_watermark_audit.csv",
        evidence_dir / "visual/stylemix_audit.md",
        evidence_dir / "nearest_neighbor_audit.md",
        evidence_dir / "figure_manifest.csv",
        evidence_dir / "provenance/README.md",
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("evidence outputs missing: {}".format(", ".join(missing)))


def write_index(evidence_dir: Path) -> None:
    text = """# 项目证据索引

更新时间：2026-06-09

本目录只负责封闭实验、视觉和数据溯源证据，不包含正式课程报告或最终提交包。

## 证据优先级

1. 原始指标 JSONL、训练 `stats.jsonl` 和 `training_options.json`；
2. 实验配置与自动生成的汇总表；
3. 生成图像、人工视觉标注和近邻拼图；
4. Claude 配置或会话记录只用于寻找线索，不作为正式证据。

## 文件

- `experiment_manifest.csv`：E1-E5、E1b、E2b、E4b 的协议、资源、指标值和原始来源。
- `claim_evidence_matrix.csv`：允许使用的结论、证据强度和禁止越过的边界。
- `visual/failure_cases.csv`：四组 P2 固定 seed 样例，共 64 张的完整失败模式标注。
- `visual/watermark_annotations.csv`：88 张独立生成样本的水印审计。
- `visual/interpolation_watermark_audit.csv`：151 帧插值轨迹的独立水印审计。
- `nearest_neighbor_audit.md`：8 个生成样本、24 条近邻记录的完整性和结论边界。
- `figure_manifest.csv`：证据图、来源范围和使用限制。
- `provenance/`：目标机数据元数据。运行 `scripts/collect_dataset_provenance.py` 后生成
  `lsun_target_machine.json`；在文件回传前，实际来源、下载日期、文件大小和哈希仍不得推测。

## 固定结论

- ADA 相比 noaug 的 FID 改善在两个保留随机条件下排序一致。
- 50k 结果只描述固定 1500 kimg 图像曝光预算，不解释为“数据越少越好”。
- E3 和 E5 均为单 seed 探索性结果。
- 单个正式训练 run 使用双卡；四卡和六卡机器用于并行调度多个双卡 run。
- 水印是训练数据污染特征被学习和复现的证据，不等价于特定样本记忆。
- 最近邻结论只覆盖当前 8 个生成样本，不证明模型总体不存在记忆。

## 重建与验证

```bash
python scripts/build_evidence.py
python -m unittest discover -s tests -v
```
"""
    (evidence_dir / "evidence_index.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", default="evidence")
    args = parser.parse_args()
    evidence_dir = resolve_path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    write_csv(
        evidence_dir / "experiment_manifest.csv",
        manifest,
        list(manifest[0].keys()),
    )
    claims = build_claims()
    write_csv(
        evidence_dir / "claim_evidence_matrix.csv",
        claims,
        ["claim_id", "strength", "allowed_wording", "evidence", "limitation"],
    )
    write_watermark_audits(evidence_dir)
    failures, _independent, interpolation = ensure_visual_tables(evidence_dir)
    figures = make_figures(evidence_dir, failures, interpolation)
    write_csv(
        evidence_dir / "figure_manifest.csv",
        figures,
        ["figure_id", "path", "purpose", "source_scope", "restriction"],
    )
    audit_nearest_neighbors(evidence_dir)
    write_index(evidence_dir)
    validate_references(evidence_dir, figures)
    print("Evidence package validated at {}".format(evidence_dir))


if __name__ == "__main__":
    main()
