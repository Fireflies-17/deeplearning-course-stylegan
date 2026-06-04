#!/usr/bin/env python
"""Calculate the non-report-grade P0 metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import resolve_path  # noqa: E402
from stylegan_course.smoke_metric import calculate, write_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--fake", required=True)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--output", default="results/tables/p0-smoke-metric.json")
    args = parser.parse_args()

    result = calculate(
        resolve_path(args.real),
        resolve_path(args.fake),
        args.max_images,
    )
    output = resolve_path(args.output)
    write_result(result, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
