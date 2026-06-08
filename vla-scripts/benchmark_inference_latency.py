#!/usr/bin/env python3
"""Benchmark VLA-Adapter local inference latency with policy_server-compatible inputs."""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULTS: dict[str, Any] = {
    "pretrained_checkpoint": "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
    "device": None,
    "robot_platform": "pick_place_conveyor",
    "unnorm_key": "",
    "control_mode": "joint_pos",
    "num_images_in_input": 2,
    "num_temporal_frames": 1,
    "temporal_fusion_type": "attention",
    "use_current_query_temporal_attention": False,
    "use_mid_layer_temporal_fusion": False,
    "action_horizon": 8,
    "action_dim": 8,
    "proprio_dim": 8,
    "use_proprio": True,
    "use_l1_regression": True,
    "use_diffusion": False,
    "use_film": False,
    "use_minivlm": True,
    "use_pro_version": True,
    "use_future_pred": False,
    "pred_tokens_before_action": False,
    "use_future_conf": False,
    "future_confidence_gamma": 1.0,
    "use_latency_conditioning": False,
    "latency_steps": 1,
    "latency_steps_max": 5,
    "use_relative_action": False,
    "relative_action_mask": None,
    "center_crop": False,
    "load_in_8bit": False,
    "load_in_4bit": False,
    "compile": False,
    "use_cuda_graph": False,
    "cuda_graph_warmup": 3,
    "return_confidence": False,
    "confidence_threshold": 0.65,
    "min_action_horizon": 2,
    "confidence_cumulative_min": True,
}


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value!r}")


def _encode_jpeg(image: Any, quality: int) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _camera_payload(
    *,
    rng: Any,
    batch_size: int,
    num_temporal_frames: int,
    image_size: int,
    jpeg_quality: int,
) -> list[str] | list[list[str]]:
    import numpy as np

    payload: list[str] | list[list[str]] = []
    for _ in range(batch_size):
        frames = []
        for _ in range(num_temporal_frames):
            image = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            frames.append(_encode_jpeg(image, jpeg_quality))
        payload.append(frames if num_temporal_frames > 1 else frames[-1])
    return payload


def build_synthetic_request(
    cfg: Any,
    *,
    batch_size: int,
    image_size: int,
    jpeg_quality: int,
    seed: int,
    task: str,
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    num_temporal_frames = int(cfg.num_temporal_frames)
    if num_temporal_frames < 1:
        raise ValueError("--num_temporal_frames must be >= 1")

    images: dict[str, Any] = {
        "fixed_cam": _camera_payload(
            rng=rng,
            batch_size=batch_size,
            num_temporal_frames=num_temporal_frames,
            image_size=image_size,
            jpeg_quality=jpeg_quality,
        )
    }

    for cam_idx in range(max(0, cfg.num_images_in_input - 1)):
        key = "wrist_cam" if cam_idx == 0 else f"wrist_cam_{cam_idx}"
        images[key] = _camera_payload(
            rng=rng,
            batch_size=batch_size,
            num_temporal_frames=num_temporal_frames,
            image_size=image_size,
            jpeg_quality=jpeg_quality,
        )

    request: dict[str, Any] = {
        "type": "vla",
        "num_envs": batch_size,
        "control_mode": cfg.control_mode,
        "images": images,
        "task_description": task,
        "step_ids": list(range(batch_size)),
    }
    if cfg.use_proprio or cfg.use_relative_action:
        request["state"] = np.zeros((batch_size, cfg.proprio_dim), dtype=np.float32).tolist()
    if cfg.use_latency_conditioning:
        request["latency_steps"] = [int(cfg.latency_steps)] * batch_size
    return request


def _cuda_synchronize(device: Any) -> None:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _summarize(latencies_ms: list[float]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _event_device_time_us(event: Any) -> float:
    for attr in ("device_time_total", "cuda_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _event_self_device_time_us(event: Any) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _event_device_memory_bytes(event: Any) -> int:
    for attr in ("device_memory_usage", "cuda_memory_usage"):
        value = getattr(event, attr, None)
        if value is not None:
            return int(value)
    return 0


def _profile_event_to_dict(event: Any) -> dict[str, Any]:
    input_shapes = getattr(event, "input_shapes", None)
    device_time_us = _event_device_time_us(event)
    return {
        "key": event.key,
        "count": int(event.count),
        "cpu_time_total_us": float(event.cpu_time_total),
        "cpu_time_avg_us": float(event.cpu_time_total / max(event.count, 1)),
        "device_time_total_us": device_time_us,
        "device_time_avg_us": float(device_time_us / max(event.count, 1)),
        "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "self_device_time_total_us": _event_self_device_time_us(event),
        "cpu_memory_usage_bytes": int(getattr(event, "cpu_memory_usage", 0)),
        "device_memory_usage_bytes": _event_device_memory_bytes(event),
        "flops": int(getattr(event, "flops", 0) or 0),
        "input_shapes": _json_safe(input_shapes) if input_shapes else None,
    }


def _install_profiler_labels(policy: Any) -> None:
    """Attach profiler labels to hot methods without changing default inference semantics."""
    import torch

    def wrap_attr(obj: Any, attr: str, label: str) -> None:
        if obj is None or not hasattr(obj, attr):
            return
        original = getattr(obj, attr)
        if getattr(original, "_vla_profile_wrapped", False):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)

        wrapped._vla_profile_wrapped = True  # type: ignore[attr-defined]
        setattr(obj, attr, wrapped)

    def wrap_class_call(cls: Any, label: str) -> None:
        if cls is None or not hasattr(cls, "__call__"):
            return
        original = cls.__call__
        if getattr(original, "_vla_profile_wrapped", False):
            return

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            with torch.profiler.record_function(label):
                return original(self, *args, **kwargs)

        wrapped._vla_profile_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, "__call__", wrapped)

    wrap_attr(policy, "predict_batch", "server.predict_batch/full")
    wrap_attr(policy, "_prepare_images_for_vla", "cpu.prepare_images_for_vla")
    wrap_attr(policy, "_normalize_proprio", "cpu.normalize_proprio")
    wrap_class_call(type(getattr(policy, "processor", None)), "cpu.processor/tokenizer_image")
    wrap_class_call(type(getattr(getattr(policy, "processor", None), "image_processor", None)), "cpu.image_processor")

    try:
        import policy_server as policy_server_module

        wrap_attr(policy_server_module, "decode_image", "cpu.decode_image")
        wrap_attr(policy_server_module, "_decode_image_entry", "cpu.decode_image_entry")
        wrap_attr(policy_server_module, "_select_image_frames", "cpu.select_primary_frames")
        wrap_attr(policy_server_module, "_select_wrist_image_frames", "cpu.select_wrist_frames")
        wrap_attr(policy_server_module, "_build_proprio_state", "cpu.build_proprio_state")
    except Exception:
        pass

    model = policy.model
    wrap_attr(model, "predict_action", "model.predict_action/full")
    wrap_attr(model, "_process_vision_features", "model.process_vision_features")
    wrap_attr(model, "_build_multimodal_attention", "model.build_multimodal_attention")
    wrap_attr(model, "_regression_or_discrete_prediction", "model.regression_or_discrete_prediction")
    wrap_attr(model, "_unnormalize_actions", "model.unnormalize_actions")
    wrap_attr(getattr(model, "vision_backbone", None), "forward", "model.vision_backbone")
    wrap_attr(getattr(getattr(model, "vision_backbone", None), "temporal_patch_attention", None), "forward", "model.temporal_patch_attention")
    wrap_attr(getattr(model, "projector", None), "forward", "model.projector")
    wrap_attr(getattr(model, "language_model", None), "forward", "model.language_model")

    action_head = getattr(policy, "action_head", None)
    wrap_attr(action_head, "predict_action", "action_head.predict_action")
    wrap_attr(getattr(action_head, "model", None), "forward", "action_head.mlp_resnet")
    blocks = getattr(getattr(action_head, "model", None), "mlp_resnet_blocks", [])
    for block in blocks:
        wrap_attr(block, "forward", "action_head.block")


def run_operator_profile(
    *,
    policy: Any,
    request: dict[str, Any],
    batch_size: int,
    device: Any,
    profile_dir: Path,
    profile_iters: int,
    top_k: int,
    with_stack: bool,
) -> dict[str, Any]:
    import torch

    profile_dir.mkdir(parents=True, exist_ok=True)
    _install_profiler_labels(policy)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    _cuda_synchronize(device)
    trace_path = profile_dir / "operator_trace.json"
    summary_path = profile_dir / "operator_profile.json"
    table_path = profile_dir / "operator_profile_top.txt"

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
        with_flops=True,
    ) as prof:
        for idx in range(profile_iters):
            with torch.profiler.record_function(f"profile.iteration_{idx:04d}"):
                policy.predict_batch(request, batch_size)
            prof.step()
    _cuda_synchronize(device)

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    events = prof.key_averages(group_by_input_shape=False)
    event_time = _event_device_time_us if device.type == "cuda" else lambda evt: float(evt.cpu_time_total)
    top_events = sorted(events, key=event_time, reverse=True)[:top_k]
    top_memory_events = sorted(events, key=lambda evt: abs(_event_device_memory_bytes(evt)), reverse=True)[:top_k]
    top_flop_events = sorted(events, key=lambda evt: getattr(evt, "flops", 0) or 0, reverse=True)[:top_k]

    table = events.table(
        sort_by=sort_by,
        row_limit=top_k,
    )
    prof.export_chrome_trace(str(trace_path))
    table_path.write_text(table + "\n", encoding="utf-8")

    summary = {
        "profile_iters": profile_iters,
        "sort_by": sort_by,
        "trace_path": str(trace_path),
        "table_path": str(table_path),
        "top_time_events": [_profile_event_to_dict(evt) for evt in top_events],
        "top_cuda_memory_events": [_profile_event_to_dict(evt) for evt in top_memory_events],
        "top_flop_events": [_profile_event_to_dict(evt) for evt in top_flop_events],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nOperator profile top events:")
    print(table)
    print(f"Saved profiler trace to: {trace_path}")
    print(f"Saved profiler summary to: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VLA-Adapter local inference latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pretrained_checkpoint", "--pretrained_path", dest="pretrained_checkpoint", default=DEFAULTS["pretrained_checkpoint"])
    parser.add_argument("--device", default=DEFAULTS["device"], help="cuda:0 / cuda:1 / cpu")
    parser.add_argument("--robot_platform", default=DEFAULTS["robot_platform"])
    parser.add_argument("--unnorm_key", default=DEFAULTS["unnorm_key"])
    parser.add_argument("--control_mode", default=DEFAULTS["control_mode"])
    parser.add_argument("--num_images_in_input", type=int, default=DEFAULTS["num_images_in_input"])
    parser.add_argument("--num_temporal_frames", type=int, default=DEFAULTS["num_temporal_frames"])
    parser.add_argument("--temporal_fusion_type", default=DEFAULTS["temporal_fusion_type"], choices=("attention", "delta_mlp"))
    parser.add_argument("--use_current_query_temporal_attention", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_current_query_temporal_attention"])
    parser.add_argument("--use_mid_layer_temporal_fusion", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_mid_layer_temporal_fusion"])
    parser.add_argument("--action_horizon", type=int, default=DEFAULTS["action_horizon"])
    parser.add_argument("--action_dim", type=int, default=DEFAULTS["action_dim"])
    parser.add_argument("--proprio_dim", type=int, default=DEFAULTS["proprio_dim"])
    parser.add_argument("--use_proprio", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_proprio"])
    parser.add_argument("--use_l1_regression", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_l1_regression"])
    parser.add_argument("--use_diffusion", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_diffusion"])
    parser.add_argument("--use_film", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_film"])
    parser.add_argument("--use_minivlm", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_minivlm"])
    parser.add_argument("--use_pro_version", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_pro_version"])
    parser.add_argument("--use_future_pred", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_future_pred"])
    parser.add_argument("--use_latency_conditioning", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_latency_conditioning"])
    parser.add_argument("--latency_steps", type=int, default=DEFAULTS["latency_steps"])
    parser.add_argument("--latency_steps_max", type=int, default=DEFAULTS["latency_steps_max"])
    parser.add_argument("--use_future_conf", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_future_conf"])
    parser.add_argument("--future_confidence_gamma", type=float, default=DEFAULTS["future_confidence_gamma"])
    parser.add_argument("--pred_tokens_before_action", action=argparse.BooleanOptionalAction, default=DEFAULTS["pred_tokens_before_action"])
    parser.add_argument("--use_relative_action", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_relative_action"])
    parser.add_argument("--relative_action_mask", default=DEFAULTS["relative_action_mask"])
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=DEFAULTS["center_crop"])
    parser.add_argument("--load_in_8bit", action=argparse.BooleanOptionalAction, default=DEFAULTS["load_in_8bit"])
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=DEFAULTS["load_in_4bit"])
    parser.add_argument("--compile", action="store_true", default=DEFAULTS["compile"])
    parser.add_argument("--use_cuda_graph", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_cuda_graph"])
    parser.add_argument("--cuda_graph_warmup", type=int, default=DEFAULTS["cuda_graph_warmup"])
    parser.add_argument("--return_confidence", action=argparse.BooleanOptionalAction, default=DEFAULTS["return_confidence"])
    parser.add_argument("--confidence_threshold", type=float, default=DEFAULTS["confidence_threshold"])
    parser.add_argument("--min_action_horizon", type=int, default=DEFAULTS["min_action_horizon"])
    parser.add_argument("--confidence_cumulative_min", action=argparse.BooleanOptionalAction, default=DEFAULTS["confidence_cumulative_min"])

    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations excluded from stats")
    parser.add_argument("--iters", type=int, default=50, help="Measured inference iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of envs per predict_batch call")
    parser.add_argument("--image_size", type=int, default=224, help="Synthetic RGB image height/width")
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=Path, default=None, help="Optional path to save summary and raw latencies")
    parser.add_argument("--print_each", type=_str2bool, default=False, help="Print each measured iteration latency")
    parser.add_argument("--profile", action="store_true", help="Run torch profiler after the latency benchmark")
    parser.add_argument("--profile_dir", type=Path, default=Path("logs/operator_profile"), help="Directory for profiler trace and summary")
    parser.add_argument("--profile_iters", type=int, default=3, help="Number of measured iterations under torch profiler")
    parser.add_argument("--profile_top_k", type=int, default=60, help="Rows to include in profiler top tables")
    parser.add_argument("--profile_with_stack", action="store_true", help="Collect Python stacks in profiler trace")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    from policy_server import ServerConfig, VLAAdapterPolicy

    if args.iters < 1:
        raise ValueError("--iters must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.profile_iters < 1:
        raise ValueError("--profile_iters must be >= 1")

    cfg_keys = set(ServerConfig.__dataclass_fields__.keys())
    cfg_kwargs = {key: getattr(args, key) for key in cfg_keys if hasattr(args, key)}
    cfg = ServerConfig(**cfg_kwargs)

    # prismatic.vla.constants may inspect argv to select robot-specific dimensions.
    if cfg.robot_platform.lower() not in " ".join(sys.argv).lower():
        sys.argv.append(cfg.robot_platform)

    request = build_synthetic_request(
        cfg,
        batch_size=args.batch_size,
        image_size=args.image_size,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
        task=args.task,
    )

    print(f"Loading policy from: {cfg.pretrained_checkpoint}", flush=True)
    policy = VLAAdapterPolicy(cfg)
    print(
        "Benchmark config: "
        f"device={policy.device}, batch_size={args.batch_size}, warmup={args.warmup}, iters={args.iters}, "
        f"num_images={cfg.num_images_in_input}, temporal_frames={cfg.num_temporal_frames}",
        flush=True,
    )

    for idx in range(args.warmup):
        _cuda_synchronize(policy.device)
        policy.predict_batch(request, args.batch_size)
        _cuda_synchronize(policy.device)
        print(f"warmup {idx + 1}/{args.warmup} done", flush=True)

    latencies_ms: list[float] = []
    for idx in range(args.iters):
        _cuda_synchronize(policy.device)
        start = time.perf_counter()
        actions, _ = policy.predict_batch(request, args.batch_size)
        _cuda_synchronize(policy.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        if args.print_each:
            print(f"iter {idx + 1:04d}: {elapsed_ms:.3f} ms", flush=True)

    summary = _summarize(latencies_ms)
    actions_shape = list(np.asarray(actions, dtype=object).shape)
    result = {
        "checkpoint": cfg.pretrained_checkpoint,
        "device": str(policy.device),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "num_images_in_input": cfg.num_images_in_input,
        "num_temporal_frames": cfg.num_temporal_frames,
        "use_cuda_graph": cfg.use_cuda_graph,
        "cuda_graph_warmup": cfg.cuda_graph_warmup,
        "summary": summary,
        "latencies_ms": latencies_ms,
        "actions_shape": actions_shape,
    }

    if args.profile:
        result["operator_profile"] = run_operator_profile(
            policy=policy,
            request=request,
            batch_size=args.batch_size,
            device=policy.device,
            profile_dir=args.profile_dir,
            profile_iters=args.profile_iters,
            top_k=args.profile_top_k,
            with_stack=args.profile_with_stack,
        )

    print("\nLatency summary (ms):")
    print(
        "  "
        f"mean={summary['mean_ms']:.3f}, median={summary['median_ms']:.3f}, "
        f"p90={summary['p90_ms']:.3f}, p95={summary['p95_ms']:.3f}, p99={summary['p99_ms']:.3f}, "
        f"min={summary['min_ms']:.3f}, max={summary['max_ms']:.3f}, std={summary['std_ms']:.3f}"
    )
    print(f"Throughput: {args.batch_size * 1000.0 / summary['mean_ms']:.3f} env/s")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON result to: {args.output_json}")


if __name__ == "__main__":
    main()
