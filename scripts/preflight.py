#!/usr/bin/env python
"""Validate the GPU machine before launching the P0 training flow."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import load_config, require_backend  # noqa: E402


BUILD_PACKAGE_FIX = (
    'python -m pip install -U "setuptools>=70" "wheel>=0.43" '
    "-i https://pypi.tuna.tsinghua.edu.cn/simple"
)
REQUIRED_PACKAGES = [
    "click",
    "numpy",
    "PIL",
    "requests",
    "scipy",
    "setuptools",
    "torch",
    "tqdm",
    "wheel",
]
RECOMMENDED_PACKAGES = [
    "ninja",
    "pyspng",
    "imageio",
    "imageio_ffmpeg",
    "matplotlib",
    "lmdb",
    "cv2",
]


def package_status(names: List[str]) -> List[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def check_python_build_packages() -> None:
    try:
        import setuptools  # noqa: F401
        import torch.utils.cpp_extension  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Python build packages are incompatible with this Python runtime. "
            "PyTorch CUDA extension compilation imports setuptools and "
            "torch.utils.cpp_extension before training. "
            "Run: {}".format(BUILD_PACKAGE_FIX)
        ) from exc


def check_grid_sample_gradfix(backend: Path, torch: object) -> None:
    sys.path.insert(0, str(backend))
    from torch_utils.ops import grid_sample_gradfix

    grid_sample_gradfix.enabled = True
    x = torch.randn(1, 1, 4, 4, device="cuda", requires_grad=True)
    theta = torch.eye(2, 3, device="cuda").unsqueeze(0)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    output = grid_sample_gradfix.grid_sample(x, grid)
    first_grad = torch.autograd.grad(output.square().sum(), x, create_graph=True)[0]
    torch.autograd.grad(first_grad.square().sum(), x)
    print("[ok] modern PyTorch grid_sample second-order gradient")


def check_fused_ops(backend: Path, torch: object) -> None:
    sys.path.insert(0, str(backend))
    from torch_utils.ops import bias_act, upfirdn2d

    x = torch.randn(1, 4, 8, 8, device="cuda")
    b = torch.randn(4, device="cuda")
    bias_act.bias_act(x, b)
    f = upfirdn2d.setup_filter([1, 3, 3, 1], device=x.device)
    upfirdn2d.upfirdn2d(x, f, padding=1)
    torch.cuda.synchronize()
    if bias_act._plugin is None or upfirdn2d._plugin is None:
        raise SystemExit(
            "Fused CUDA ops failed to compile. Use a CUDA 12.8 devel image with nvcc, "
            "a compatible C++ compiler, and ninja before training."
        )
    print("[ok] fused CUDA ops compiled and executed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/p0_smoke.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require the CUDA compilation toolchain and execute fused ops.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    backend = require_backend(config)

    missing = package_status(REQUIRED_PACKAGES)
    if missing:
        raise SystemExit(
            "Missing required packages: {}. Run: python -m pip install -r requirements-gpu.txt".format(
                ", ".join(missing)
            )
        )
    recommended_missing = package_status(RECOMMENDED_PACKAGES)
    if recommended_missing:
        print("[warn] optional performance packages missing: {}".format(", ".join(recommended_missing)))

    check_python_build_packages()

    import torch

    print("Python: {}".format(sys.version.split()[0]))
    print("PyTorch: {}".format(torch.__version__))
    print("PyTorch CUDA: {}".format(torch.version.cuda))
    if sys.version_info[:2] != (3, 12):
        print("[warn] Python 3.12 is the selected baseline; current Python is {}.{}.".format(
            sys.version_info.major, sys.version_info.minor
        ))
    if not torch.__version__.startswith("2.8."):
        print("[warn] PyTorch 2.8 is the selected baseline.")
    if not str(torch.version.cuda).startswith("12.8"):
        print("[warn] CUDA 12.8 is the selected baseline.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch.")
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
    print("Compute capability: {}".format(torch.cuda.get_device_capability(0)))

    tools = {
        "nvcc": shutil.which("nvcc"),
        "compiler": shutil.which("g++") or shutil.which("c++") or shutil.which("cl"),
        "ninja": shutil.which("ninja"),
    }
    missing_tools = [name for name, path in tools.items() if path is None]
    if missing_tools and args.strict:
        raise SystemExit(
            "Missing required CUDA build tools: {}. Select a CUDA 12.8 devel image and "
            "install ninja before training.".format(", ".join(missing_tools))
        )
    if missing_tools:
        print("[warn] missing CUDA build tools: {}; fused ops may use slow fallbacks.".format(
            ", ".join(missing_tools)
        ))
    if tools["nvcc"] is not None:
        version = subprocess.check_output(["nvcc", "--version"], text=True).strip().splitlines()[-1]
        print("NVCC: {}".format(version))

    check_grid_sample_gradfix(backend, torch)
    if args.strict:
        check_fused_ops(backend, torch)
    print("[ok] preflight passed")


if __name__ == "__main__":
    main()
