"""Helpers for DINOv3 future-observation feature caches."""

import hashlib
from pathlib import Path
from typing import Optional, Union

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


def _normalize_dataset_name(dataset_name: object) -> Optional[str]:
    if dataset_name is None:
        return None
    if isinstance(dataset_name, np.ndarray):
        if dataset_name.size == 0:
            return None
        dataset_name = dataset_name.reshape(-1)[0].item()
    if isinstance(dataset_name, bytes):
        dataset_name = dataset_name.decode("utf-8")
    return str(dataset_name)


def resolve_cache_dir(cache_dir: Union[str, Path], dataset_name: object = None) -> Path:
    cache_dir = Path(cache_dir)
    dataset_name = _normalize_dataset_name(dataset_name)
    if dataset_name is None:
        return cache_dir

    dataset_cache_dir = cache_dir / dataset_name
    return dataset_cache_dir if dataset_cache_dir.exists() else cache_dir


def load_feature(cache_dir: Union[str, Path], image: np.ndarray, dataset_name: object = None) -> np.ndarray:
    resolved_cache_dir = resolve_cache_dir(cache_dir, dataset_name)
    path = feature_path(resolved_cache_dir, image_sha1(image))
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
