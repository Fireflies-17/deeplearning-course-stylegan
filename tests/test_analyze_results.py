from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_results import collect_run, find_run_dir, parse_metrics


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class AnalyzeResultsTests(unittest.TestCase):
    def test_repeated_metric_uses_latest_timestamp_and_counts_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_jsonl(
                run_dir / "metric-fid50k_full.jsonl",
                [
                    {
                        "snapshot_pkl": "network-snapshot-001500.pkl",
                        "timestamp": 20,
                        "results": {"fid50k_full": 16.0},
                    },
                    {
                        "snapshot_pkl": "network-snapshot-001500.pkl",
                        "timestamp": 10,
                        "results": {"fid50k_full": 17.0},
                    },
                ],
            )

            metrics, metadata = parse_metrics(run_dir)
            self.assertEqual(metrics[1500]["fid50k_full"], 16.0)
            self.assertEqual(metadata[(1500, "fid50k_full")]["count"], 2)

    def test_summary_records_each_metrics_snapshot_kimg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_jsonl(
                run_dir / "stats.jsonl",
                [
                    {
                        "Progress/kimg": {"mean": 2000},
                        "Timing/total_hours": {"mean": 1.0},
                    }
                ],
            )
            write_jsonl(
                run_dir / "metric-fid50k_full.jsonl",
                [
                    {
                        "snapshot_pkl": "network-snapshot-001500.pkl",
                        "timestamp": 1,
                        "results": {"fid50k_full": 16.0},
                    },
                    {
                        "snapshot_pkl": "network-snapshot-002000.pkl",
                        "timestamp": 2,
                        "results": {"fid50k_full": 13.0},
                    },
                ],
            )
            write_jsonl(
                run_dir / "metric-kid50k_full.jsonl",
                [
                    {
                        "snapshot_pkl": "network-snapshot-001500.pkl",
                        "timestamp": 3,
                        "results": {"kid50k_full": 0.01},
                    }
                ],
            )

            _, summary = collect_run("E1", run_dir)
            self.assertEqual(summary["fid50k_full_final_kimg"], 2000)
            self.assertEqual(summary["kid50k_full_final_kimg"], 1500)

    def test_parent_run_selection_uses_numeric_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older_number = root / "00009-old"
            newer_number = root / "00010-new"
            older_number.mkdir()
            newer_number.mkdir()
            (older_number / "stats.jsonl").touch()
            (newer_number / "stats.jsonl").touch()

            self.assertEqual(find_run_dir(root), newer_number)


if __name__ == "__main__":
    unittest.main()
