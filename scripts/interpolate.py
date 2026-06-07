#!/usr/bin/env python
"""Latent-space interpolation for a trained StyleGAN2-ADA generator.

Walks a smooth path through a sequence of anchor seeds and renders the frames.
Interpolation is done in W space by default (linear interpolation of the mapped
``w`` vectors, the standard "smooth" path); Z space (spherical interpolation of
the input noise) is also available via ``--space z``. Truncation ``--trunc`` is
applied at mapping time, so you can produce clean, report-ready transitions.

It writes:

- ``<outdir>/frame_<idx>.png``: one PNG per interpolation frame.
- ``<outdir>/contact_sheet.png``: every frame tiled into a single grid image,
  convenient for embedding a strip of the walk directly in the report.
- ``<outdir>/interpolation.mp4``: optional, only if ``imageio`` (with an ffmpeg
  plugin) is importable; failure to write the video is non-fatal.

This is a non-training visualization: a smooth, non-repeating walk is evidence
that the model learned a continuous latent space rather than memorizing a few
training images, and it pairs well with the truncation and style-mixing figures.

Requires the pinned StyleGAN2-ADA backend (run scripts/bootstrap_stylegan2_ada.py)
and a CUDA GPU.

Usage::

    python scripts/interpolate.py \
        --network results/runs/.../network-snapshot-002000.pkl \
        --seeds 0,1,2,3,0 --steps 30 --trunc 0.7 \
        --outdir results/samples/interp_e1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import resolve_path  # noqa: E402


def parse_seeds(value: str) -> List[int]:
    seeds: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError(
            "Need at least 2 anchor seeds to interpolate, got: {}".format(value)
        )
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, help="Generator .pkl snapshot.")
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,0",
        type=parse_seeds,
        help="Anchor seeds in walk order; repeat the first to make a loop.",
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Frames per anchor-to-anchor segment."
    )
    parser.add_argument("--trunc", type=float, default=0.7, help="Truncation psi.")
    parser.add_argument(
        "--space", choices=["w", "z"], default="w", help="Interpolation space."
    )
    parser.add_argument(
        "--cols", type=int, default=0, help="Contact-sheet columns (0 = auto)."
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Video frame rate (if imageio available)."
    )
    parser.add_argument("--outdir", default="results/samples/interp")
    parser.add_argument("--backend", default="third_party/stylegan2-ada-pytorch")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    backend = resolve_path(args.backend)
    if not (backend / "legacy.py").is_file():
        raise FileNotFoundError(
            "StyleGAN2-ADA backend not found at {}. Run: "
            "python scripts/bootstrap_stylegan2_ada.py".format(backend)
        )
    sys.path.insert(0, str(backend))

    # Heavy imports are deferred so --help works without torch/backend installed.
    import numpy as np
    import torch
    import PIL.Image

    import dnnlib  # type: ignore  # noqa: E402  (provided by backend)
    import legacy  # type: ignore  # noqa: E402

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    network_path = resolve_path(args.network)
    outdir = resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading generator: {}".format(network_path))
    with dnnlib.util.open_url(str(network_path)) as handle:
        G = legacy.load_network_pkl(handle)["G_ema"].to(device).eval()

    label = torch.zeros([1, G.c_dim], device=device)

    def seed_z(seed: int) -> "torch.Tensor":
        return torch.from_numpy(
            np.random.RandomState(seed).randn(1, G.z_dim)
        ).to(device).float()

    def slerp(a: "torch.Tensor", b: "torch.Tensor", t: float) -> "torch.Tensor":
        # Spherical interpolation between two z vectors, falls back to lerp when
        # the endpoints are nearly colinear.
        a_n = a / a.norm(dim=1, keepdim=True)
        b_n = b / b.norm(dim=1, keepdim=True)
        dot = (a_n * b_n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        if float(sin_omega.abs().min()) < 1e-6:
            return (1.0 - t) * a + t * b
        return (
            torch.sin((1.0 - t) * omega) / sin_omega * a
            + torch.sin(t * omega) / sin_omega * b
        )

    # Build the anchor latents in the chosen space.
    if args.space == "w":
        anchors = []
        for seed in args.seeds:
            with torch.no_grad():
                w = G.mapping(seed_z(seed), label, truncation_psi=args.trunc)
            anchors.append(w)  # (1, num_ws, w_dim)
    else:
        anchors = [seed_z(seed) for seed in args.seeds]  # (1, z_dim)

    def render(latent: "torch.Tensor") -> "np.ndarray":
        with torch.no_grad():
            if args.space == "w":
                img = G.synthesis(latent, noise_mode="const")
            else:
                img = G(latent, label, truncation_psi=args.trunc, noise_mode="const")
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].permute(1, 2, 0).cpu().numpy()

    # Render every frame along the piecewise path between consecutive anchors.
    frames: List["np.ndarray"] = []
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        # Drop the duplicated endpoint between segments to avoid a stutter.
        last = (i == len(anchors) - 2)
        count = args.steps + (1 if last else 0)
        for s in range(count):
            t = s / args.steps
            if args.space == "w":
                latent = (1.0 - t) * a + t * b
            else:
                latent = slerp(a, b, t)
            frames.append(render(latent))
        print("  segment {}/{} rendered".format(i + 1, len(anchors) - 1))

    for idx, frame in enumerate(frames):
        PIL.Image.fromarray(frame, "RGB").save(outdir / "frame_{:04d}.png".format(idx))
    print("[frames] {} PNGs in {}".format(len(frames), outdir))

    # Contact sheet: tile every frame into one grid image.
    cols = args.cols if args.cols > 0 else int(np.ceil(np.sqrt(len(frames))))
    rows = int(np.ceil(len(frames) / cols))
    h, w = frames[0].shape[:2]
    pad = 2
    sheet = np.full(
        (rows * h + pad * (rows - 1), cols * w + pad * (cols - 1), 3), 255, dtype=np.uint8
    )
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        y, x = r * (h + pad), c * (w + pad)
        sheet[y : y + h, x : x + w] = frame[..., :3]
    sheet_path = outdir / "contact_sheet.png"
    PIL.Image.fromarray(sheet, "RGB").save(sheet_path)
    print("[contact_sheet] {}".format(sheet_path))

    # Optional MP4 (soft dependency on imageio + an ffmpeg plugin).
    try:
        import imageio  # type: ignore

        video_path = outdir / "interpolation.mp4"
        imageio.mimsave(str(video_path), frames, fps=args.fps)
        print("[video] {}".format(video_path))
    except Exception as exc:  # noqa: BLE001  (video is a best-effort extra)
        print("[video] skipped ({}). Frames and contact sheet were written.".format(exc))


if __name__ == "__main__":
    main()
