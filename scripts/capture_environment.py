#!/usr/bin/env python
"""Record the software and GPU environment used by an experiment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import resolve_path  # noqa: E402


def command_output(command: list) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return "unavailable: {}".format(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/logs/environment.json")
    args = parser.parse_args()

    backend = resolve_path("third_party/stylegan2-ada-pytorch")
    record: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "backend_commit": command_output(["git", "-C", str(backend), "rev-parse", "HEAD"]),
        "backend_diff_stat": command_output(["git", "-C", str(backend), "diff", "--stat"]),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvidia_smi_topology": command_output(["nvidia-smi", "topo", "-m"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "compiler": command_output(["g++", "--version"]),
        "ninja": command_output(["ninja", "--version"]),
    }
    try:
        import torch

        record["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError as error:
        record["torch"] = {"unavailable": str(error)}

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
