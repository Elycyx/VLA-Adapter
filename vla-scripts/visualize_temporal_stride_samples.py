"""
Visualize how temporal frame intervals sample images from an RLDS dataset.

Example:
  python vla-scripts/visualize_temporal_stride_samples.py \
      --data_root_dir /mnt/lx/cyx/lerobot/dataset \
      --dataset_name pick_place_conveyor \
      --output_dir outputs/temporal_stride_vis \
      --num_temporal_frames 2 \
      --intervals 1,2,3 \
      --num_examples 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must be formatted as HEIGHT,WIDTH or HEIGHTxWIDTH.")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive.")
    return height, width


def parse_intervals(value: str) -> list[int]:
    intervals = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not intervals:
        raise argparse.ArgumentTypeError("At least one interval is required.")
    if any(interval < 1 for interval in intervals):
        raise argparse.ArgumentTypeError("All intervals must be >= 1.")
    return intervals


def get_camera_keys(observation: dict[str, np.ndarray]) -> list[str]:
    keys = ["image_primary"] if "image_primary" in observation else []
    keys.extend(sorted(k for k in observation if "wrist" in k and k.startswith("image_")))
    return keys


def temporal_indices(window_size: int, num_temporal_frames: int, interval: int) -> list[int]:
    start = window_size - 1 - (num_temporal_frames - 1) * interval
    if start < 0:
        raise ValueError(
            f"window_size={window_size} is too small for num_temporal_frames={num_temporal_frames} "
            f"and interval={interval}."
        )
    return [start + i * interval for i in range(num_temporal_frames)]


def to_rgb_image(array: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.BILINEAR)
    return image


def make_grid(
    sample: dict,
    *,
    intervals: Iterable[int],
    num_temporal_frames: int,
    tile_size: tuple[int, int] | None,
) -> Image.Image:
    observation = sample["observation"]
    camera_keys = get_camera_keys(observation)
    if not camera_keys:
        raise ValueError("No image_primary or image_*wrist keys found in the RLDS sample.")

    window_size = int(observation[camera_keys[0]].shape[0])
    intervals = list(intervals)
    if tile_size is None:
        frame_h = int(observation[camera_keys[0]].shape[1])
        frame_w = int(observation[camera_keys[0]].shape[2])
    else:
        frame_h, frame_w = tile_size

    label_h = 28
    row_label_w = 190
    gap = 8
    cols = num_temporal_frames
    rows = len(intervals) * len(camera_keys)
    width = row_label_w + cols * frame_w + (cols + 1) * gap
    height = label_h + rows * (frame_h + label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    title = "Temporal stride sampling visualization"
    language = sample.get("task", {}).get("language_instruction", b"")
    if isinstance(language, np.ndarray):
        language = language.item() if language.shape == () else language.reshape(-1)[0]
    if isinstance(language, bytes):
        language = language.decode(errors="ignore")
    if language:
        title += f" | {language[:90]}"
    draw.text((gap, 6), title, fill=(20, 20, 20))

    for row_idx, (interval, camera_key) in enumerate((i, k) for i in intervals for k in camera_keys):
        y0 = label_h + gap + row_idx * (frame_h + label_h + gap)
        row_label = f"{camera_key}\ninterval={interval}"
        draw.text((gap, y0 + label_h + 4), row_label, fill=(30, 30, 30))

        indices = temporal_indices(window_size, num_temporal_frames, interval)
        for col_idx, frame_idx in enumerate(indices):
            x0 = row_label_w + gap + col_idx * (frame_w + gap)
            label = f"idx {frame_idx}"
            if col_idx == len(indices) - 1:
                label += " (t)"
            else:
                offset = (indices[-1] - frame_idx)
                label += f" (t-{offset})"
            draw.text((x0, y0), label, fill=(30, 30, 30))
            image = to_rgb_image(observation[camera_key][frame_idx], size=(frame_w, frame_h))
            canvas.paste(image, (x0, y0 + label_h))
            draw.rectangle((x0, y0 + label_h, x0 + frame_w - 1, y0 + label_h + frame_h - 1), outline=(80, 80, 80))

    return canvas


def build_dataset(args: argparse.Namespace):
    mixture_spec = OXE_NAMED_MIXTURES.get(args.dataset_name, [(args.dataset_name, 1.0)])
    load_camera_views = ("primary", "left_wrist", "right_wrist") if "aloha" in args.dataset_name else ("primary", "wrist")
    per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        args.data_root_dir,
        mixture_spec,
        load_camera_views=load_camera_views,
        load_depth=False,
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
    )

    max_interval = max(args.intervals)
    temporal_window_size = (args.num_temporal_frames - 1) * max_interval + 1
    dataset, _, _ = make_interleaved_dataset(
        dataset_kwargs_list=per_dataset_kwargs,
        sample_weights=weights,
        train=args.train,
        shuffle_buffer_size=args.shuffle_buffer_size,
        traj_transform_kwargs=dict(
            window_size=temporal_window_size,
            future_action_window_size=0,
            future_obs_window_size=0,
            skip_unlabeled=True,
            goal_relabeling_strategy="uniform",
        ),
        frame_transform_kwargs=dict(
            resize_size=args.resize_resolution,
            num_parallel_calls=args.num_parallel_calls,
        ),
        balance_weights=True,
        traj_transform_threads=len(mixture_spec),
        traj_read_threads=len(mixture_spec),
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Save image grids visualizing temporal stride sampling from RLDS.")
    parser.add_argument("--data_root_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_temporal_frames", type=int, default=2)
    parser.add_argument("--intervals", type=parse_intervals, default=parse_intervals("1,2"))
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--skip_examples", type=int, default=0)
    parser.add_argument("--resize_resolution", type=parse_resolution, default=parse_resolution("224,224"))
    parser.add_argument("--tile_size", type=parse_resolution, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=256)
    parser.add_argument("--num_parallel_calls", type=int, default=8)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.num_temporal_frames < 1:
        raise ValueError("--num_temporal_frames must be >= 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args)
    iterator = dataset.as_numpy_iterator()

    for _ in range(args.skip_examples):
        next(iterator)

    for example_idx in range(args.num_examples):
        sample = next(iterator)
        grid = make_grid(
            sample,
            intervals=args.intervals,
            num_temporal_frames=args.num_temporal_frames,
            tile_size=args.tile_size,
        )
        output_path = args.output_dir / f"temporal_stride_sample_{example_idx:04d}.png"
        grid.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
