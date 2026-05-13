"""Helpers for DINOv3 future-observation feature caches."""

import hashlib
from pathlib import Path
from typing import Union

import numpy as np


def image_sha1(image: np.ndarray) -> str:
    """Stable content hash for a decoded RGB image array."""
    arr = np.ascontiguousarray(image)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def feature_path(cache_dir: Union[str, Path], image_hash: str) -> Path:
    cache_dir = Path(cache_dir)
    return cache_dir / image_hash[:2] / f"{image_hash}.npy"


def load_feature(cache_dir: Union[str, Path], image: np.ndarray) -> np.ndarray:
    path = feature_path(cache_dir, image_sha1(image))
    if not path.exists():
        raise FileNotFoundError(
            f"Missing DINOv3 feature cache for image hash {path.stem}: {path}. "
            "Run vla-scripts/precompute_dinov3_features.py first."
        )
    return np.load(path)


def save_feature(cache_dir: Union[str, Path], image_hash: str, feature: np.ndarray) -> Path:
    path = feature_path(cache_dir, image_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feature)
    return path
