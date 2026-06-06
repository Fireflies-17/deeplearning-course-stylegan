#!/usr/bin/env python
"""Training-set nearest-neighbor retrieval for generated samples.

For a set of generator seeds, this generates images, embeds them with the same
official Inception feature space used by FID, then streams the real training set
once to find each generated image's Top-K nearest real neighbors (L2 distance in
Inception feature space). It writes:

- ``<outdir>/neighbors.csv``: rows of (seed, rank, real_index, distance).
- ``<outdir>/nn_seed<seed>.png``: a montage [generated | neighbor_1 ... neighbor_K].

This is used for the data-contamination / memorization-risk analysis. Without a
clear nearest-neighbor match we describe results as "data-contamination
reproduction" (e.g. learned stock-photo watermarks), not "memorization".

Requires the pinned StyleGAN2-ADA backend (run scripts/bootstrap_stylegan2_ada.py)
and a CUDA GPU. Mirrors the feature extraction in the backend's
``metrics/frechet_inception_distance.py``.

Usage::

    python scripts/nearest_neighbors.py \
        --network results/runs/.../network-snapshot-002000.pkl \
        --data data/processed/lsun-church-256-100k.zip \
        --seeds 0-15 --topk 3 --outdir results/nn/e1-final
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import resolve_path  # noqa: E402

# Same detector the backend uses for FID, so neighbor distances live in the FID space.
DETECTOR_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/"
    "metrics/inception-2015-12-05.pkl"
)


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
    if not seeds:
        raise argparse.ArgumentTypeError("No seeds parsed from: {}".format(value))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, help="Generator .pkl snapshot.")
    parser.add_argument("--data", required=True, help="Real dataset zip/dir (the training set).")
    parser.add_argument("--seeds", default="0-15", type=parse_seeds)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--trunc", type=float, default=1.0, help="Truncation psi for generation.")
    parser.add_argument("--outdir", default="results/nn")
    parser.add_argument("--backend", default="third_party/stylegan2-ada-pytorch")
    parser.add_argument("--max-real", type=int, default=0, help="Cap real images scanned (0 = all).")
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
    from metrics import metric_utils  # type: ignore  # noqa: E402
    from training import dataset as dataset_lib  # type: ignore  # noqa: E402

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    network_path = resolve_path(args.network)
    data_path = resolve_path(args.data)
    outdir = resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading generator: {}".format(network_path))
    with dnnlib.util.open_url(str(network_path)) as handle:
        G = legacy.load_network_pkl(handle)["G_ema"].to(device).eval()

    print("Loading Inception feature detector")
    detector = metric_utils.get_feature_detector(DETECTOR_URL, device=device)
    detector_kwargs = dict(return_features=True)

    def features_from_uint8(images: "torch.Tensor") -> "np.ndarray":
        # images: (N, C, H, W) uint8 on device -> (N, 2048) float32 numpy.
        with torch.no_grad():
            feats = detector(images, **detector_kwargs)
        return feats.cpu().numpy().astype(np.float64)

    # --- Generate images and embed them. ---
    label = torch.zeros([1, G.c_dim], device=device)
    gen_uint8: List["np.ndarray"] = []
    gen_feats_list: List["np.ndarray"] = []
    for seed in args.seeds:
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, G.z_dim)
        ).to(device).float()
        with torch.no_grad():
            img = G(z, label, truncation_psi=args.trunc, noise_mode="const")
        img_u8 = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        gen_feats_list.append(features_from_uint8(img_u8))
        gen_uint8.append(img_u8[0].permute(1, 2, 0).cpu().numpy())
    gen_feats = np.concatenate(gen_feats_list, axis=0)  # (S, 2048)
    num_gen = gen_feats.shape[0]

    # Per-generated-image running Top-K as list of (distance, real_index).
    best: List[List[Tuple[float, int]]] = [[] for _ in range(num_gen)]

    def consider(gen_idx: int, dist: float, real_idx: int) -> None:
        bucket = best[gen_idx]
        if len(bucket) < args.topk:
            bucket.append((dist, real_idx))
            bucket.sort(key=lambda kv: kv[0])
        elif dist < bucket[-1][0]:
            bucket[-1] = (dist, real_idx)
            bucket.sort(key=lambda kv: kv[0])

    # --- Stream the real training set once. ---
    print("Scanning real set: {}".format(data_path))
    real_ds = dataset_lib.ImageFolderDataset(
        path=str(data_path), use_labels=False, max_size=None, xflip=False
    )
    limit = args.max_real if args.max_real > 0 else len(real_ds)
    for real_idx in range(min(limit, len(real_ds))):
        image, _ = real_ds[real_idx]  # (C, H, W) uint8 numpy
        tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        feat = features_from_uint8(tensor)[0]  # (2048,)
        dists = np.linalg.norm(gen_feats - feat[None, :], axis=1)  # (S,)
        for gen_idx in range(num_gen):
            consider(gen_idx, float(dists[gen_idx]), real_idx)
        if (real_idx + 1) % 5000 == 0:
            print("  scanned {}/{}".format(real_idx + 1, min(limit, len(real_ds))))

    # --- Write table and montages. ---
    csv_path = outdir / "neighbors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "rank", "real_index", "distance"])
        for gen_idx, seed in enumerate(args.seeds):
            for rank, (dist, real_idx) in enumerate(best[gen_idx], start=1):
                writer.writerow([seed, rank, real_idx, round(dist, 4)])
    print("[csv] {}".format(csv_path))

    for gen_idx, seed in enumerate(args.seeds):
        tiles = [gen_uint8[gen_idx]]
        for _, real_idx in best[gen_idx]:
            real_img, _ = real_ds[real_idx]
            tiles.append(np.transpose(real_img, (1, 2, 0)))
        h = max(t.shape[0] for t in tiles)
        w = max(t.shape[1] for t in tiles)
        pad = 4
        canvas = np.full((h, w * len(tiles) + pad * (len(tiles) - 1), 3), 255, dtype=np.uint8)
        for col, tile in enumerate(tiles):
            x = col * (w + pad)
            canvas[: tile.shape[0], x : x + tile.shape[1]] = tile[..., :3]
        montage_path = outdir / "nn_seed{:04d}.png".format(seed)
        PIL.Image.fromarray(canvas, "RGB").save(montage_path)
    print("[montages] {} files in {}".format(len(args.seeds), outdir))


if __name__ == "__main__":
    main()
