#!/usr/bin/env python
"""Unified command entry point for training, resume, generation, and evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.commands import (  # noqa: E402
    build_evaluate_command,
    build_generate_command,
    build_train_command,
)
from stylegan_course.project import (  # noqa: E402
    load_config,
    require_backend,
    resolve_network,
    run_command,
)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the backend command without executing it.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    add_common(train)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute the official backend's configuration dry run.",
    )

    resume = subparsers.add_parser("resume")
    add_common(resume)
    resume.add_argument("--network", default="latest")
    resume.add_argument("--kimg", type=int)
    resume.add_argument("--dry-run", action="store_true")

    generate = subparsers.add_parser("generate")
    add_common(generate)
    generate.add_argument("--network", default="latest")
    generate.add_argument("--seeds")
    generate.add_argument("--trunc", type=float)
    generate.add_argument("--outdir")
    generate.add_argument("--class", dest="class_idx", type=int)

    evaluate = subparsers.add_parser("evaluate")
    add_common(evaluate)
    evaluate.add_argument("--network", default="latest")
    evaluate.add_argument("--metrics")

    args = parser.parse_args()
    config = load_config(args.config)
    if not args.print_only:
        require_backend(config)

    if args.command == "train":
        command = build_train_command(config, backend_dry_run=args.dry_run)
    elif args.command == "resume":
        network = resolve_network(config, args.network)
        command = build_train_command(
            config,
            resume=network,
            kimg=args.kimg,
            backend_dry_run=args.dry_run,
        )
    elif args.command == "generate":
        network = resolve_network(config, args.network)
        command = build_generate_command(
            config,
            network,
            seeds=args.seeds,
            trunc=args.trunc,
            outdir=args.outdir,
            class_idx=args.class_idx,
        )
    else:
        network = resolve_network(config, args.network)
        command = build_evaluate_command(config, network, metrics=args.metrics)

    run_command(command, print_only=args.print_only)


if __name__ == "__main__":
    main()
