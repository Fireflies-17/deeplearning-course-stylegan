#!/usr/bin/env python
"""Run the complete first-stage P0 pipeline on a GPU machine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.commands import build_generate_command, build_train_command  # noqa: E402
from stylegan_course.project import (  # noqa: E402
    find_latest_snapshot,
    load_config,
    require_backend,
    resolve_path,
    run_command,
)


def script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/p0_smoke.json")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.skip_bootstrap:
        run_command([sys.executable, script("bootstrap_stylegan2_ada.py")])
    require_backend(config)

    run_command([sys.executable, script("preflight.py"), "--config", args.config, "--strict"])
    run_command([sys.executable, script("prepare_data.py"), "synthetic"])
    run_command([sys.executable, script("capture_environment.py")])

    run_command(build_train_command(config))
    first_snapshot = find_latest_snapshot(config)
    print("Initial snapshot: {}".format(first_snapshot))

    final_snapshot = first_snapshot
    if not args.skip_resume:
        run_command(build_train_command(config, resume=first_snapshot, kimg=1))
        final_snapshot = find_latest_snapshot(config)
        print("Resume snapshot: {}".format(final_snapshot))

    run_command(build_generate_command(config, final_snapshot))
    run_command(
        [
            sys.executable,
            script("smoke_metric.py"),
            "--real",
            str(resolve_path(config["data"])),
            "--fake",
            str(resolve_path(config["samples_outdir"])),
            "--output",
            str(resolve_path(config["smoke_metric_output"])),
        ]
    )


if __name__ == "__main__":
    main()
