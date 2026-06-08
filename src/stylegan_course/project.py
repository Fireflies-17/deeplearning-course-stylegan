"""Project paths, configuration loading, and subprocess helpers."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
STYLEGAN2_ADA_URL = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
STYLEGAN2_ADA_COMMIT = "d72cc7d041b42ec8e806021a205ed9349f87c6a4"
SNAPSHOT_RE = re.compile(r"network-snapshot-(\d+)\.pkl$")


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: str) -> Dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = [
        "schema_version",
        "name",
        "backend",
        "data",
        "outdir",
        "samples_outdir",
        "train",
        "generate",
        "evaluate",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Missing config fields: {}".format(", ".join(missing)))
    if config["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version: {}".format(config["schema_version"]))
    return config


def backend_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config["backend"])


def require_backend(config: Dict[str, Any]) -> Path:
    backend = backend_path(config)
    required = ["train.py", "generate.py", "calc_metrics.py", "dataset_tool.py"]
    missing = [name for name in required if not (backend / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "StyleGAN2-ADA backend is missing {}. Run: "
            "python scripts/bootstrap_stylegan2_ada.py".format(", ".join(missing))
        )
    return backend


def format_command(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    print_only: bool = False,
) -> None:
    print("$ {}".format(format_command(command)), flush=True)
    if not print_only:
        subprocess.run(command, cwd=str(cwd or ROOT), check=True)


def find_latest_snapshot(config: Dict[str, Any]) -> Path:
    outdir = resolve_path(config["outdir"])
    snapshots = list(outdir.glob("**/network-snapshot-*.pkl"))
    if not snapshots:
        raise FileNotFoundError("No network snapshot found below {}".format(outdir))

    def sort_key(path: Path) -> tuple:
        match = SNAPSHOT_RE.search(path.name)
        kimg = int(match.group(1)) if match else -1
        return kimg, path.stat().st_mtime

    return max(snapshots, key=sort_key)


def resolve_network(config: Dict[str, Any], value: str) -> Path:
    if value == "latest":
        return find_latest_snapshot(config)
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError("Network snapshot does not exist: {}".format(path))
    return path
