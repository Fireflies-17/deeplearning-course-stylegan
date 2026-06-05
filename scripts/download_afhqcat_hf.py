#!/usr/bin/env python
"""Download AFHQv2 train/cat images from Hugging Face with resumable saves."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ryushinn/AFHQv2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prefix", default="train/cat/")
    parser.add_argument("--outdir", default="data/raw/afhq/train/cat")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=10.0)
    parser.add_argument(
        "--hf-endpoint",
        default="",
        help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    return parser.parse_args()


def save_matching_rows(args: argparse.Namespace) -> tuple[int, int]:
    from datasets import load_dataset

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    saved = 0
    skipped = 0
    for row in tqdm(ds, desc="AFHQv2 rows"):
        relpath = str(row.get("image_relpath", "")).replace("\\", "/")
        if not relpath.startswith(args.prefix):
            continue

        target = outdir / Path(relpath).name
        if target.is_file() and target.stat().st_size > 0:
            skipped += 1
            continue

        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL image in row['image']")
        image.convert("RGB").save(target)
        saved += 1
    return saved, skipped


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    for attempt in range(1, args.retries + 1):
        try:
            saved, skipped = save_matching_rows(args)
            print(
                "download complete: saved {} new images, skipped {} existing images".format(
                    saved, skipped
                )
            )
            return
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(
                "attempt {}/{} failed: {}".format(attempt, args.retries, exc),
                file=sys.stderr,
            )
            if attempt == args.retries:
                raise
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
