#!/usr/bin/env python
"""Download and optionally extract the LSUN Church Outdoor train LMDB zip."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_URL = "http://dl.yf.io/lsun/scenes/church_outdoor_train_lmdb.zip"
EXPECTED_SHA256 = "91128ae026840ac0c5982b4445ab5fc4e092d6847cca76793b2b1a0815c2e74a"
EXPECTED_SIZE = 2448463466


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/downloads/church_outdoor_train_lmdb.zip")
    parser.add_argument("--extract-dir", default="data/raw/lsun")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA256 validation for a trusted custom mirror.",
    )
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, skip_hash: bool) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != EXPECTED_SIZE:
        return False
    return skip_hash or sha256(path) == EXPECTED_SHA256


def download_once(url: str, target: Path, args: argparse.Namespace) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    existing = part.stat().st_size if part.is_file() else 0
    headers = {"Range": "bytes={}-".format(existing)} if existing else {}
    mode = "ab" if existing else "wb"
    response = requests.get(url, stream=True, timeout=args.timeout, headers=headers)
    if existing and response.status_code != 206:
        part.unlink()
        existing = 0
        mode = "wb"
        response.close()
        response = requests.get(url, stream=True, timeout=args.timeout)
    response.raise_for_status()

    total_header = response.headers.get("Content-Length")
    total = int(total_header) + existing if total_header and total_header.isdigit() else None
    with part.open(mode) as handle:
        progress = tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=target.name,
        )
        with progress:
            for chunk in response.iter_content(chunk_size=args.chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
    part.replace(target)


def download(args: argparse.Namespace) -> Path:
    target = Path(args.out)
    if validate(target, args.skip_hash):
        print("local LSUN Church zip already verified: {}".format(target))
        return target

    for attempt in range(1, args.retries + 1):
        try:
            print("trying {} (attempt {}/{})".format(args.url, attempt, args.retries))
            download_once(args.url, target, args)
            if validate(target, args.skip_hash):
                print("download complete: {}".format(target))
                return target
            actual_size = target.stat().st_size if target.is_file() else 0
            actual_hash = sha256(target) if target.is_file() else "missing"
            raise RuntimeError(
                "validation failed: expected size {} and sha256 {}, got size {} and sha256 {}".format(
                    EXPECTED_SIZE,
                    EXPECTED_SHA256,
                    actual_size,
                    actual_hash,
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(
                "failed attempt {}/{}: {}".format(attempt, args.retries, exc),
                file=sys.stderr,
            )
            if attempt < args.retries:
                time.sleep(args.sleep)
    raise RuntimeError("Unable to download LSUN Church from {}".format(args.url))


def extract(zip_path: Path, extract_dir: Path) -> None:
    expected_dir = extract_dir / "church_outdoor_train_lmdb"
    if expected_dir.is_dir():
        print("LSUN Church LMDB already exists: {}".format(expected_dir))
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    if not expected_dir.is_dir():
        raise RuntimeError(
            "Expected {} after extraction. Check archive layout.".format(expected_dir)
        )
    print("extracted LSUN Church LMDB to {}".format(expected_dir))


def main() -> None:
    args = parse_args()
    zip_path = download(args)
    if args.extract:
        extract(zip_path, Path(args.extract_dir))


if __name__ == "__main__":
    main()
