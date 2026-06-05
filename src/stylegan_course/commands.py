"""Build commands for the pinned StyleGAN2-ADA backend."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .project import backend_path, resolve_path


TRAIN_OPTIONS = {
    "gpus",
    "snap",
    "metrics",
    "seed",
    "cond",
    "subset",
    "mirror",
    "cfg",
    "gamma",
    "kimg",
    "batch",
    "aug",
    "p",
    "target",
    "augpipe",
    "resume",
    "freezed",
    "fp32",
    "nhwc",
    "nobench",
    "allow_tf32",
    "workers",
}


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _options(values: Dict[str, Any]) -> List[str]:
    options = []
    for key, value in values.items():
        if value is None:
            continue
        options.append("--{}={}".format(key.replace("_", "-"), _value(value)))
    return options


def build_train_command(
    config: Dict[str, Any],
    resume: Optional[Path] = None,
    kimg: Optional[int] = None,
    backend_dry_run: bool = False,
) -> List[str]:
    backend = backend_path(config)
    train = dict(config["train"])
    unknown = sorted(set(train) - TRAIN_OPTIONS)
    if unknown:
        raise ValueError("Unsupported StyleGAN2-ADA train options: {}".format(", ".join(unknown)))
    if kimg is not None:
        train["kimg"] = kimg
    if resume is not None:
        train["resume"] = str(resume)

    command = [
        sys.executable,
        str(backend / "train.py"),
        "--outdir={}".format(resolve_path(config["outdir"])),
        "--data={}".format(resolve_path(config["data"])),
    ]
    command.extend(_options(train))
    if backend_dry_run:
        command.append("--dry-run")
    return command


def build_generate_command(
    config: Dict[str, Any],
    network: Path,
    seeds: Optional[str] = None,
    trunc: Optional[float] = None,
    outdir: Optional[str] = None,
    class_idx: Optional[int] = None,
) -> List[str]:
    generate = dict(config["generate"])
    if seeds is not None:
        generate["seeds"] = seeds
    if trunc is not None:
        generate["trunc"] = trunc
    if class_idx is not None:
        generate["class"] = class_idx

    command = [
        sys.executable,
        str(backend_path(config) / "generate.py"),
        "--network={}".format(network),
        "--outdir={}".format(resolve_path(outdir or config["samples_outdir"])),
    ]
    command.extend(_options(generate))
    return command


def build_evaluate_command(
    config: Dict[str, Any],
    network: Path,
    metrics: Optional[str] = None,
) -> List[str]:
    evaluate = dict(config["evaluate"])
    if metrics is not None:
        evaluate["metrics"] = metrics

    command = [
        sys.executable,
        str(backend_path(config) / "calc_metrics.py"),
        "--network={}".format(network),
        "--data={}".format(resolve_path(config["data"])),
    ]
    command.extend(_options(evaluate))
    return command
