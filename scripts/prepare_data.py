#!/usr/bin/env python
"""Prepare synthetic P0 data or convert an image directory with dataset_tool.py."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import require_backend, resolve_path  # noqa: E402


def generate_synthetic(output: Path, count: int, resolution: int, seed: int) -> None:
    existing = sorted(output.glob("*.png")) if output.exists() else []
    if len(existing) == count:
        print("Synthetic source already exists: {}".format(output))
        return
    if existing:
        raise RuntimeError(
            "{} contains {} PNG files; expected {}. Remove it or choose another output.".format(
                output, len(existing), count
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(count):
        background = tuple(rng.randint(8, 96) for _ in range(3))
        foreground = tuple(rng.randint(128, 255) for _ in range(3))
        image = Image.new("RGB", (resolution, resolution), background)
        draw = ImageDraw.Draw(image)
        margin = rng.randint(max(2, resolution // 12), max(3, resolution // 5))
        x0, y0 = margin, margin
        x1, y1 = resolution - margin - 1, resolution - margin - 1
        shape = index % 3
        if shape == 0:
            draw.ellipse((x0, y0, x1, y1), fill=foreground)
        elif shape == 1:
            draw.rectangle((x0, y0, x1, y1), fill=foreground)
        else:
            draw.polygon(
                [
                    (resolution // 2, y0),
                    (x1, y1),
                    (x0, y1),
                ],
                fill=foreground,
            )
        angle = (index * 37) % 360
        radius = resolution * 0.45
        center = resolution / 2
        end_x = center + radius * math.cos(math.radians(angle))
        end_y = center + radius * math.sin(math.radians(angle))
        draw.line((center, center, end_x, end_y), fill=(255, 255, 255), width=1)
        image.save(output / "{:05d}.png".format(index))
    print("Created {} synthetic images in {}".format(count, output))


def convert_dataset(
    source: Path,
    dest: Path,
    backend: Path,
    resolution: str,
    transform: str,
    max_images: int,
) -> None:
    if dest.exists():
        print("Converted dataset already exists: {}".format(dest))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(backend / "dataset_tool.py"),
        "--source={}".format(source),
        "--dest={}".format(dest),
    ]
    if resolution:
        try:
            width, height = [int(value) for value in resolution.lower().split("x", 1)]
        except ValueError:
            raise ValueError("Resolution must use the form WIDTHxHEIGHT, e.g. 256x256")
        command.extend(["--width={}".format(width), "--height={}".format(height)])
    if transform:
        command.append("--transform={}".format(transform))
    if max_images:
        command.append("--max-images={}".format(max_images))
    print("$ {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--source", default="data/raw/p0-shapes-32")
    synthetic.add_argument("--dest", default="data/processed/p0-shapes-32.zip")
    synthetic.add_argument("--count", type=int, default=64)
    synthetic.add_argument("--resolution", type=int, default=32)
    synthetic.add_argument("--seed", type=int, default=20260604)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--source", required=True)
    convert.add_argument("--dest", required=True)
    convert.add_argument("--resolution", default="")
    convert.add_argument(
        "--transform",
        choices=["", "center-crop", "center-crop-wide"],
        default="",
    )
    convert.add_argument("--max-images", type=int, default=0)

    args = parser.parse_args()
    config = {"backend": "third_party/stylegan2-ada-pytorch"}
    backend = require_backend(config)

    if args.command == "synthetic":
        source = resolve_path(args.source)
        dest = resolve_path(args.dest)
        generate_synthetic(source, args.count, args.resolution, args.seed)
        resolution = "{}x{}".format(args.resolution, args.resolution)
        convert_dataset(source, dest, backend, resolution, "", 0)
    else:
        convert_dataset(
            resolve_path(args.source),
            resolve_path(args.dest),
            backend,
            args.resolution,
            args.transform,
            args.max_images,
        )


if __name__ == "__main__":
    main()
