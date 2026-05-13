"""
Precompute frozen DINOv3 features per decoded primary image.

This script is intentionally **independent of Prismatic / VLA** so it can run in a
separate environment (e.g. newer `transformers` for DINOv3) while RLDS metadata is
supplied via a pickle produced in the VLA environment:

  # In VLA env:
  python vla-scripts/export_dinov3_rlds_spec.py \\
      --data_root_dir /path/to/rlds \\
      --dataset_name my_dataset \\
      --output dinov3_rlds_spec.pkl

  # In DINOv3-friendly env (only needs this file + deps below):
  python vla-scripts/precompute_dinov3_features.py \\
      --spec_pickle dinov3_rlds_spec.pkl \\
      --output_dir /path/to/cache \\
      --resize_resolution 224,224

Optional: encode all images under a directory (no TensorFlow):

  python vla-scripts/precompute_dinov3_features.py \\
      --images_dir /path/to/images \\
      --output_dir /path/to/cache

Cache layout matches `prismatic/vla/datasets/dinov3_features.py` (sha1 of uint8
decoded+resized HxWx3 array -> .npy under output_dir/xx/hash.npy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm
from PIL import Image

# --- cache helpers (must stay compatible with prismatic/vla/datasets/dinov3_features.py) ---


def image_sha1(image: np.ndarray) -> str:
    arr = np.ascontiguousarray(image)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def feature_path(cache_dir: Union[str, Path], image_hash: str) -> Path:
    cache_dir = Path(cache_dir)
    return cache_dir / image_hash[:2] / f"{image_hash}.npy"


def save_feature(cache_dir: Union[str, Path], image_hash: str, feature: np.ndarray) -> Path:
    path = feature_path(cache_dir, image_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feature)
    return path


# --- minimal RLDS pipeline (vendored from prismatic/vla/datasets/rlds/dataset.py) ---


def tree_map(fn: Callable, tree: Dict) -> Dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def parse_relative_action_mask(mask):
    if mask is None:
        return None
    if isinstance(mask, str):
        if not mask.strip():
            return None
        truthy = {"1", "true", "t", "yes", "y"}
        falsy = {"0", "false", "f", "no", "n"}
        parsed = []
        for item in mask.split(","):
            value = item.strip().lower()
            if value in truthy:
                parsed.append(True)
            elif value in falsy:
                parsed.append(False)
            else:
                raise ValueError(f"Invalid relative_action_mask value: {item!r}")
        return tuple(parsed)
    return tuple(bool(x) for x in mask)


def _add_action_normalization_mask(dataset_statistics: dict, action_normalization_mask: Optional[List[bool]]) -> dict:
    if action_normalization_mask is None:
        return dataset_statistics
    if len(action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
        raise ValueError(
            f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
            f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
        )
    dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)
    return dataset_statistics


def _norm_type_str(normalization_type: Any) -> str:
    if hasattr(normalization_type, "value"):
        return str(normalization_type.value)
    return str(normalization_type)


def normalize_action_and_proprio(traj: Dict, metadata: Dict, normalization_type: Any):
    """Same behavior as prismatic RLDS util; depends only on dlimp + tf."""
    import dlimp as dl
    import tensorflow as tf

    nt = _norm_type_str(normalization_type)
    keys_to_normalize = {"action": "action", "proprio": "observation/proprio"}

    if nt == "normal":
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x),
            )
        return traj

    if nt in ("bounds", "bounds_q99"):
        for key, traj_key in keys_to_normalize.items():
            if nt == "bounds":
                low, high = metadata[key]["min"], metadata[key]["max"]
            else:
                low, high = metadata[key]["q01"], metadata[key]["q99"]
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1),
                    x,
                ),
            )
            zeros_mask = metadata[key]["min"] == metadata[key]["max"]
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == traj_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )
        return traj

    raise ValueError(f"Unknown normalization type {normalization_type!r} ({nt!r})")


def make_rlds_dataset_from_spec_entry(
    cfg: Dict[str, Any],
    *,
    train: bool,
    shuffle: bool,
    num_parallel_reads: int,
    num_parallel_calls: int,
):
    """
    Build a trajectory dataset equivalent to `make_dataset_from_rlds` when
    `dataset_statistics` is already provided (from export script).
    """
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    name = cfg["name"]
    data_dir = cfg["data_dir"]
    standardize_fn = cfg.get("standardize_fn")
    image_obs_keys = cfg["image_obs_keys"]
    depth_obs_keys = cfg.get("depth_obs_keys", {})
    state_obs_keys = cfg.get("state_obs_keys", ())
    language_key = cfg.get("language_key")
    action_proprio_normalization_type = cfg["action_proprio_normalization_type"]
    absolute_action_mask = cfg.get("absolute_action_mask")
    action_normalization_mask = cfg.get("action_normalization_mask")
    use_relative_action = cfg.get("use_relative_action", False)
    relative_action_mask = parse_relative_action_mask(cfg.get("relative_action_mask"))
    if use_relative_action and relative_action_mask is None:
        raise ValueError("use_relative_action=True requires relative_action_mask in spec entry.")

    dataset_statistics = cfg.get("dataset_statistics")
    if dataset_statistics is None:
        raise ValueError(
            "Spec entry missing `dataset_statistics`. Run `vla-scripts/export_dinov3_rlds_spec.py` in the VLA env."
        )

    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        if standardize_fn is not None:
            traj = standardize_fn(traj)
        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Did you write a `standardize_fn`?"
            )
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    (
                        tf.zeros((traj_len, 1), dtype=tf.float32)
                        if key is None
                        else tf.cast(old_obs[key], tf.float32)
                    )
                    for key in state_obs_keys
                ],
                axis=1,
            )

        new_obs["timestep"] = tf.range(traj_len)
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, " "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)

        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )
        return traj

    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            ds_stats = json.load(f)
    else:
        ds_stats = dataset_statistics

    ds_stats = tree_map(np.array, ds_stats)
    ds_stats = _add_action_normalization_mask(ds_stats, action_normalization_mask)

    split = "train" if train else "val"
    builder = tfds.builder(name, data_dir=data_dir)
    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads)
    dataset = dataset.traj_map(restructure, num_parallel_calls)
    if not use_relative_action:
        dataset = dataset.traj_map(
            partial(
                normalize_action_and_proprio,
                metadata=ds_stats,
                normalization_type=action_proprio_normalization_type,
            ),
            num_parallel_calls,
        )
    return dataset


def decode_and_resize_primary(frame: dict, resize_size: Tuple[int, int]) -> dict:
    import dlimp as dl
    import tensorflow as tf

    image = frame["observation"]["image_primary"]
    if image.dtype == tf.string:
        image = tf.cond(
            tf.equal(tf.strings.length(image), 0),
            lambda: tf.zeros((*resize_size, 3), dtype=tf.uint8),
            lambda: dl.transforms.resize_image(
                tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8),
                size=resize_size,
            ),
        )
    elif image.dtype == tf.uint8:
        image = dl.transforms.resize_image(image, size=resize_size)
    else:
        raise ValueError(f"Unsupported image_primary dtype: {image.dtype}")
    frame["observation"]["image_primary"] = image
    return frame


def parse_resolution(value: str) -> Tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected H,W or HxW, e.g. 224,224")
    return int(parts[0]), int(parts[1])


def _iter_images_dir(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


@torch.inference_mode()
def encode_and_save(model, processor, images, hashes, output_dir: Path, device: str):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        if hidden.shape[1] > 1:
            hidden = hidden[:, 1:]
        features = hidden.mean(dim=1).float().cpu().numpy().astype(np.float16)

    for image_hash, feature in zip(hashes, features):
        save_feature(output_dir, image_hash, feature)


def main() -> None:
    import pickle

    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--spec_pickle",
        type=Path,
        help="Pickle from `export_dinov3_rlds_spec.py` (RLDS / OXE, no Prismatic needed at runtime).",
    )
    src.add_argument(
        "--images_dir",
        type=Path,
        help="Recursively encode all images under this directory (no RLDS / TensorFlow).",
    )

    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--resize_resolution", type=parse_resolution, default=(224, 224))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle_rlds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_parallel_calls", type=int, default=16)
    parser.add_argument("--num_parallel_reads", type=int, default=-1, help="TF autotune if -1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True).to(args.device).eval()

    pending_images: List[Image.Image] = []
    pending_hashes: List[str] = []
    seen: set[str] = set()
    written = 0
    samples = 0

    def flush():
        nonlocal written, pending_images, pending_hashes
        if pending_images:
            encode_and_save(model, processor, pending_images, pending_hashes, args.output_dir, args.device)
            written += len(pending_images)
            pending_images, pending_hashes = [], []

    def process_numpy_image(image: np.ndarray):
        nonlocal samples
        image_hash = image_sha1(image)
        if image_hash in seen:
            samples += 1
            return
        seen.add(image_hash)
        target_path = feature_path(args.output_dir, image_hash)
        if target_path.exists() and not args.overwrite:
            samples += 1
            return
        pending_hashes.append(image_hash)
        pending_images.append(Image.fromarray(np.asarray(image)))
        if len(pending_images) >= args.batch_size:
            flush()
        samples += 1

    if args.images_dir is not None:
        if not args.images_dir.is_dir():
            raise FileNotFoundError(args.images_dir)
        for path in tqdm.tqdm(list(_iter_images_dir(args.images_dir)), desc="images_dir"):
            if args.max_samples is not None and samples >= args.max_samples:
                break
            img = Image.open(path).convert("RGB")
            img = img.resize((args.resize_resolution[1], args.resize_resolution[0]), Image.BICUBIC)
            process_numpy_image(np.asarray(img))
        flush()
        print(f"Wrote {written} DINOv3 feature files to {args.output_dir}")
        return

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    num_reads = args.num_parallel_reads if args.num_parallel_reads >= 0 else tf.data.AUTOTUNE

    with open(args.spec_pickle, "rb") as f:
        entries: List[Dict[str, Any]] = pickle.load(f)

    for cfg in entries:
        if args.max_samples is not None and samples >= args.max_samples:
            break
        dataset = make_rlds_dataset_from_spec_entry(
            cfg,
            train=args.train,
            shuffle=args.shuffle_rlds,
            num_parallel_reads=num_reads,
            num_parallel_calls=args.num_parallel_calls,
        )
        dataset = dataset.flatten(num_parallel_calls=args.num_parallel_calls)
        dataset = dataset.frame_map(
            partial(decode_and_resize_primary, resize_size=args.resize_resolution),
            args.num_parallel_calls,
        )
        desc = f"precompute {cfg.get('name', 'dataset')}"
        for frame in tqdm.tqdm(dataset.as_numpy_iterator(), desc=desc):
            if args.max_samples is not None and samples >= args.max_samples:
                break
            process_numpy_image(frame["observation"]["image_primary"])

    flush()
    print(f"Wrote {written} DINOv3 feature files to {args.output_dir}")


if __name__ == "__main__":
    main()
