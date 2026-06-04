"""Small deterministic Fréchet-style metric for pipeline validation only."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _feature(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((8, 8), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float64) / 255.0
    pixels = array.reshape(-1)
    means = array.mean(axis=(0, 1))
    stds = array.std(axis=(0, 1))
    return np.concatenate([pixels, means, stds])


def _directory_features(path: Path, max_images: int) -> np.ndarray:
    files = sorted(
        item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    return np.stack([_feature(Image.open(item)) for item in files[:max_images]])


def _zip_features(path: Path, max_images: int) -> np.ndarray:
    features: List[np.ndarray] = []
    with zipfile.ZipFile(str(path)) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_SUFFIXES
        )
        for name in names[:max_images]:
            with archive.open(name) as handle:
                features.append(_feature(Image.open(io.BytesIO(handle.read()))))
    return np.stack(features)


def load_features(path: Path, max_images: int) -> np.ndarray:
    if path.is_dir():
        features = _directory_features(path, max_images)
    elif path.is_file() and path.suffix.lower() == ".zip":
        features = _zip_features(path, max_images)
    else:
        raise ValueError("Expected an image directory or zip archive: {}".format(path))
    if features.shape[0] < 2:
        raise ValueError("At least two images are required: {}".format(path))
    return features


def _statistics(features: np.ndarray) -> tuple:
    return features.mean(axis=0), np.cov(features, rowvar=False)


def _sqrt_trace_product(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    cov_a = (cov_a + cov_a.T) * 0.5
    cov_b = (cov_b + cov_b.T) * 0.5
    values, vectors = np.linalg.eigh(cov_a)
    sqrt_a = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
    middle = sqrt_a @ cov_b @ sqrt_a
    middle = (middle + middle.T) * 0.5
    return float(np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None)).sum())


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    mean_a, cov_a = _statistics(features_a)
    mean_b, cov_b = _statistics(features_b)
    mean_term = float(np.square(mean_a - mean_b).sum())
    covariance_term = float(
        np.trace(cov_a) + np.trace(cov_b) - 2.0 * _sqrt_trace_product(cov_a, cov_b)
    )
    return max(0.0, mean_term + covariance_term)


def calculate(real_path: Path, fake_path: Path, max_images: int) -> Dict[str, object]:
    real = load_features(real_path, max_images)
    fake = load_features(fake_path, max_images)
    return {
        "metric": "p0_smoke_frechet_not_report_grade",
        "value": frechet_distance(real, fake),
        "real_images": int(real.shape[0]),
        "fake_images": int(fake.shape[0]),
        "feature_dimensions": int(real.shape[1]),
        "warning": "Pipeline validation only. Do not report as FID or KID.",
    }


def write_result(result: Dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
