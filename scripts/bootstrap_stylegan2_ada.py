#!/usr/bin/env python
"""Clone and verify the pinned official StyleGAN2-ADA PyTorch backend."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import (  # noqa: E402
    STYLEGAN2_ADA_COMMIT,
    STYLEGAN2_ADA_URL,
    resolve_path,
)


PATCHES = [
    ROOT / "patches" / "stylegan2-ada-pytorch-modern-pytorch.patch",
    ROOT / "patches" / "stylegan2-ada-pytorch-modern-warnings.patch",
    ROOT / "patches" / "stylegan2-ada-pytorch-python312-distutils.patch",
    ROOT / "patches" / "stylegan2-ada-pytorch-ddp-noise-buffer.patch",
    ROOT / "patches" / "stylegan2-ada-pytorch-numpy-scalar.patch",
]


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def git_succeeds(*args: str) -> bool:
    result = subprocess.run(
        ["git", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def apply_compatibility_patch(target: Path, patch: Path, verify_only: bool) -> None:
    if not patch.is_file():
        raise SystemExit("Compatibility patch is missing: {}".format(patch))
    if git_succeeds("-C", str(target), "apply", "--reverse", "--check", str(patch)):
        print("Compatibility patch already applied: {}".format(patch.name))
        return
    if verify_only:
        raise SystemExit("Compatibility patch is not applied: {}".format(patch))
    if not git_succeeds("-C", str(target), "apply", "--check", str(patch)):
        raise SystemExit("Compatibility patch does not apply cleanly: {}".format(patch))
    subprocess.run(["git", "-C", str(target), "apply", str(patch)], check=True)
    print("Applied compatibility patch: {}".format(patch.name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="third_party/stylegan2-ada-pytorch",
        help="Clone destination relative to the project root.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that the expected commit is already present.",
    )
    args = parser.parse_args()

    target = resolve_path(args.target)
    if not target.exists():
        if args.verify_only:
            raise SystemExit("Backend does not exist: {}".format(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", STYLEGAN2_ADA_URL, str(target)], check=True)

    if not (target / ".git").exists():
        raise SystemExit("Target exists but is not a Git checkout: {}".format(target))

    current = git_output("-C", str(target), "rev-parse", "HEAD")
    if current != STYLEGAN2_ADA_COMMIT:
        if args.verify_only:
            raise SystemExit(
                "Backend commit mismatch: expected {}, found {}".format(
                    STYLEGAN2_ADA_COMMIT, current
                )
            )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", STYLEGAN2_ADA_COMMIT],
            check=True,
        )
        current = git_output("-C", str(target), "rev-parse", "HEAD")

    for patch in PATCHES:
        apply_compatibility_patch(target, patch, verify_only=args.verify_only)
    print("StyleGAN2-ADA backend ready: {}".format(current))


if __name__ == "__main__":
    main()
