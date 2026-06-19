# DINOv3 Precompute Environment

This environment is only for running `vla-scripts/precompute_dinov3_features.py`.
It is intentionally separate from the main VLA / Prismatic environment because
DINOv3 may require a newer `transformers` stack.

## Base Packages

Install these packages for both `--images_dir` and `--spec_pickle` modes:

```bash
pip install -r vla-scripts/dinov3_requirements.txt
```

The requirements file is pinned from a known-working `dinov3` environment on
Python 3.11. It uses PyTorch CUDA 12.6 wheels via the PyTorch extra index.

## Extra Packages for RLDS Spec Mode

When running with `--spec_pickle`, the script also reconstructs the RLDS input
pipeline. These dependencies are already included in
`vla-scripts/dinov3_requirements.txt`:

```bash
pip install tensorflow==2.15.0 tensorflow-graphics draccus
pip install "tensorflow-datasets==4.9.3" "tensorflow-metadata==1.16.1"
pip install git+https://github.com/kvablack/dlimp.git
```

Pin `protobuf` to match TensorFlow 2.15 and avoid TFDS / metadata version skew:

```bash
# Python 3.10 only, if recreating an environment closer to the main VLA env
pip install "protobuf==3.20.3"

# Python 3.11 only (tensorflow-metadata 1.16.1 requires protobuf>=4.25.2)
pip install "protobuf>=4.25.2,<5"
```

Do **not** install the latest unpinned `tensorflow-datasets` on Python 3.11.
Newer releases pull in `tensorflow-metadata` built against protobuf 6.x, which
conflicts with TensorFlow 2.15.

The repository root must also be available on disk because old spec pickle files
may contain references to `prismatic.vla.datasets.rlds.oxe.transforms` functions.
Running the script from this repository is enough.

## Suggested Setup

```bash
conda create -n dinov3 python=3.11
conda activate dinov3

pip install -r vla-scripts/dinov3_requirements.txt
```

If this is used on a machine that needs a different PyTorch CUDA wheel, update
the `--extra-index-url` and `torch==...` line in
`vla-scripts/dinov3_requirements.txt`.

## Fix Protobuf / TFDS Version Errors

If you see errors like `cannot import name 'runtime_version'` or
`gencode 6.31.1 runtime 5.x`, downgrade the TF metadata stack:

```bash
pip install "protobuf>=4.25.2,<5" "tensorflow-datasets==4.9.3" "tensorflow-metadata==1.16.1"
```

On Python 3.10 you can use `protobuf==3.20.3` instead.

## Workflow Reminder

Create the RLDS spec pickle in the main VLA environment:

```bash
python vla-scripts/export_dinov3_rlds_spec.py \
    --data_root_dir /path/to/rlds \
    --dataset_name /dataset/name \
    --output /path/to/dinov3_rlds_spec.pkl
```

Then precompute DINOv3 features in this DINOv3 environment:

```bash
CUDA_VISIBLE_DEVICES=0 python vla-scripts/precompute_dinov3_features.py \
    --spec_pickle /path/to/dinov3_rlds_spec.pkl \
    --output_dir /path/to/dinov3_feature_cache \
    --resize_resolution 224,224 \
    --model_id ./dinov3-vitl16-pretrain-lvd1689m \
    --batch_size 64
```
