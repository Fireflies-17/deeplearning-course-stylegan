#!/usr/bin/env python
"""Collect dataset provenance facts on the training machine.

The output distinguishes directly observed file facts from download-history
evidence. File modification times are never promoted to download dates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit, urlunsplit


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SOURCE_MARKERS = (
    "opendatalab",
    "openxlab",
    "download_lsun_church.py",
)
DOWNLOAD_CLIENTS = (
    "wget ",
    "curl ",
    "aria2c ",
)
DATASET_TERMS = ("lsun", "church_outdoor", "church-outdoor", "church")
SECRET_OPTION_RE = re.compile(
    r"(?i)(--?(?:token|access[-_]?token|api[-_]?key|password|passwd|secret)"
    r"(?:\s+|=))([^\s]+)"
)
AUTH_HEADER_RE = re.compile(r"(?i)(authorization:\s*(?:bearer|basic)\s+)([^\s\"']+)")
URL_RE = re.compile(r"https?://[^\s\"']+")


def iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def command_output(command: List[str]) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return "unavailable: {}".format(error)


def hash_file(path: Path) -> Dict[str, str]:
    hashes = {"sha256": hashlib.sha256(), "md5": hashlib.md5()}
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            for digest in hashes.values():
                digest.update(chunk)
    return {name: digest.hexdigest() for name, digest in hashes.items()}


def inspect_zip(path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
    }
    if not path.is_file():
        record["status"] = "missing"
        return record

    stat = path.stat()
    record.update(
        {
            "status": "observed",
            "size_bytes": stat.st_size,
            "modified_time_utc": iso_time(stat.st_mtime),
            "modified_time_interpretation": "file timestamp; not a download date",
            "hashes": hash_file(path),
        }
    )

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        image_names = [
            name for name in names if Path(name).suffix.lower() in IMAGE_SUFFIXES
        ]
        record["zip_entry_count"] = len(names)
        record["image_count"] = len(image_names)
        record["dataset_json_present"] = "dataset.json" in names
        if "dataset.json" in names:
            try:
                record["dataset_json"] = json.loads(
                    archive.read("dataset.json").decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                record["dataset_json_error"] = str(error)

        if image_names:
            try:
                from PIL import Image

                dimensions = set()
                for name in image_names[: min(32, len(image_names))]:
                    with archive.open(name) as image_file:
                        with Image.open(image_file) as image:
                            dimensions.add((image.width, image.height, image.mode))
                record["sampled_image_count"] = min(32, len(image_names))
                record["sampled_dimensions"] = [
                    {"width": width, "height": height, "mode": mode}
                    for width, height, mode in sorted(dimensions)
                ]
            except ImportError as error:
                record["image_inspection_error"] = "Pillow unavailable: {}".format(error)
            except Exception as error:
                record["image_inspection_error"] = str(error)
    return record


def inspect_raw(path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
    }
    if not path.exists():
        record["status"] = "missing"
        return record

    files = [item for item in path.rglob("*") if item.is_file()]
    stats = [item.stat() for item in files]
    record.update(
        {
            "status": "observed",
            "file_count": len(files),
            "size_bytes": sum(stat.st_size for stat in stats),
        }
    )
    if stats:
        record["earliest_modified_time_utc"] = iso_time(
            min(stat.st_mtime for stat in stats)
        )
        record["latest_modified_time_utc"] = iso_time(
            max(stat.st_mtime for stat in stats)
        )
        record["modified_time_interpretation"] = (
            "file timestamps; not download dates"
        )
    return record


def redact_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    suffix = ""
    while raw and raw[-1] in ".,);]":
        suffix = raw[-1] + suffix
        raw = raw[:-1]
    parts = urlsplit(raw)
    query = "<redacted-query>" if parts.query else ""
    netloc = parts.netloc
    if "@" in netloc:
        host = netloc.rsplit("@", 1)[1]
        netloc = "<redacted-userinfo>@{}".format(host)
    return urlunsplit((parts.scheme, netloc, parts.path, query, "")) + suffix


def redact_command(command: str) -> str:
    command = SECRET_OPTION_RE.sub(r"\1<redacted>", command)
    command = AUTH_HEADER_RE.sub(r"\1<redacted>", command)
    return URL_RE.sub(redact_url, command)


def is_download_candidate(command: str) -> bool:
    lowered = command.lower()
    if any(marker in lowered for marker in SOURCE_MARKERS):
        return True
    has_client = any(client in lowered for client in DOWNLOAD_CLIENTS)
    has_url = "http://" in lowered or "https://" in lowered
    has_dataset = any(term in lowered for term in DATASET_TERMS)
    return has_client and has_url and has_dataset


def parse_history(lines: Iterable[str]) -> List[Dict[str, Optional[str]]]:
    records: List[Dict[str, Optional[str]]] = []
    pending_time: Optional[str] = None
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            continue

        bash_timestamp = re.fullmatch(r"#(\d{9,})", line)
        if bash_timestamp:
            pending_time = iso_time(float(bash_timestamp.group(1)))
            continue

        zsh_match = re.match(r"^:\s*(\d{9,}):\d+;(.*)$", line)
        if zsh_match:
            timestamp = iso_time(float(zsh_match.group(1)))
            command = zsh_match.group(2)
        else:
            timestamp = pending_time
            command = line
        pending_time = None

        if not is_download_candidate(command):
            continue
        records.append(
            {
                "timestamp_utc": timestamp,
                "command": redact_command(command),
            }
        )
    return records


def inspect_history(path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": str(path.expanduser().resolve()),
        "exists": path.expanduser().is_file(),
    }
    if not path.expanduser().is_file():
        record.update(
            {
                "status": "missing",
                "candidate_commands": [],
                "assessment": "download history unavailable",
                "source_status": "unrecoverable",
            }
        )
        return record

    with path.expanduser().open("r", encoding="utf-8", errors="replace") as handle:
        commands = parse_history(handle)
    timestamped = [item for item in commands if item["timestamp_utc"]]
    if timestamped:
        assessment = "actual source command and timestamp recovered"
        source_status = "recovered_with_date"
    elif commands:
        assessment = "actual source command recovered; download date unknown"
        source_status = "recovered_without_date"
    else:
        assessment = "actual source and download date not retained or recoverable"
        source_status = "unrecoverable"
    record.update(
        {
            "status": "observed",
            "candidate_commands": commands,
            "assessment": assessment,
            "source_status": source_status,
        }
    )
    return record


def collect_environment() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvcc": command_output(["nvcc", "--version"]),
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
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError as error:
        record["torch"] = {"unavailable": str(error)}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", required=True)
    parser.add_argument("--raw", dest="raw_path", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser()
    record = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_policy": {
            "file_timestamps_are_download_dates": False,
            "planned_sources_are_actual_sources": False,
            "missing_values_are_fabricated": False,
        },
        "dataset_zip": inspect_zip(Path(args.zip_path).expanduser()),
        "raw_dataset": inspect_raw(Path(args.raw_path).expanduser()),
        "download_history": inspect_history(Path(args.history)),
        "environment": collect_environment(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("Wrote {}".format(output.resolve()))


if __name__ == "__main__":
    main()
