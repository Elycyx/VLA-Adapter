"""
Export a pickle spec for `precompute_dinov3_features.py` (DINOv3-only environment).

Run this script in the **same Python environment as VLA / Prismatic** (where RLDS + OXE
helpers work). It calls `make_dataset_from_rlds` once per dataset to materialize
`dataset_statistics`, then saves kwargs + stats so the precompute script does not need
to import Prismatic or recompute statistics.
"""

from __future__ import annotations

import argparse
import inspect
import pickle
from pathlib import Path
from typing import Any, Dict, List

from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def _to_serializable_norm_type(obj: Any) -> Any:
    """Store normalization type as plain string for the standalone consumer."""
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


def _prepare_entry(raw: Dict[str, Any], train: bool) -> Dict[str, Any]:
    raw = dict(raw)
    raw.pop("dataset_frame_transform_kwargs", None)
    sig = inspect.signature(make_dataset_from_rlds)
    skip = {"train", "dataset_statistics"}
    call_kw = {k: raw[k] for k in sig.parameters if k not in skip and k in raw}

    _, stats = make_dataset_from_rlds(**call_kw, train=train, dataset_statistics=None)
    out = {**call_kw, "dataset_statistics": stats}
    out["action_proprio_normalization_type"] = _to_serializable_norm_type(out["action_proprio_normalization_type"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path to write spec .pkl")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.dataset_name in OXE_NAMED_MIXTURES:
        mixture_spec = OXE_NAMED_MIXTURES[args.dataset_name]
    else:
        mixture_spec = [(args.dataset_name, 1.0)]

    per_dataset_kwargs, _ = get_oxe_dataset_kwargs_and_weights(
        args.data_root_dir,
        mixture_spec,
        load_camera_views=("primary", "left_wrist", "right_wrist")
        if "aloha" in args.dataset_name
        else ("primary", "wrist"),
        load_depth=False,
        load_proprio=True,
        load_language=True,
    )

    entries: List[Dict[str, Any]] = []
    for raw in per_dataset_kwargs:
        entries.append(_prepare_entry(raw, train=args.train))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(entries, f, protocol=4)

    names = [e["name"] for e in entries]
    print(f"Wrote RLDS spec for {len(entries)} dataset(s): {names} -> {args.output}")


if __name__ == "__main__":
    main()
