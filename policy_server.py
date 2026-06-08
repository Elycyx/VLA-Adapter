#!/usr/bin/env python3
"""HTTP policy server for VLA-Adapter inference.

The API matches POLICY_SERVER.md and is consumed by policy_client.py:
    GET  /info
    POST /predict
    POST /reset

Debug:
    --debug 会在每次 POST /predict 打印完整请求与完整响应 JSON（同时 stdout + logger）。
    为避免日志爆炸，默认不会打印 images 里的 base64 正文，只会打印长度与短 hash。

Confidence:
    --return_confidence 与 --confidence-score-log PATH 同时开启时，每次推理将各步得分以 JSON 一行
    追加写入 PATH，并在 logger INFO 打同样一条紧凑 JSON。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import gc
import io
import json
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _status(message: str) -> None:
    print(f"[policy_server] {message}", flush=True)
    logger.info(message)


@dataclass
class ServerConfig:
    pretrained_checkpoint: str = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    device: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    robot_platform: str = "pick_place_conveyor"
    model_family: str = "openvla"
    unnorm_key: str = ""
    control_mode: str = "joint_pos"
    num_images_in_input: int = 2
    num_temporal_frames: int = 1
    temporal_fusion_type: str = "attention"
    use_current_query_temporal_attention: bool = False
    use_mid_layer_temporal_fusion: bool = False
    action_horizon: int = 8
    action_dim: int = 8
    proprio_dim: int = 8
    use_proprio: bool = True
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_minivlm: bool = True
    use_pro_version: bool = True
    use_future_pred: bool = False
    pred_tokens_before_action: bool = False
    use_future_conf: bool = False
    future_confidence_gamma: float = 1.0
    use_latency_conditioning: bool = False
    latency_steps: int = 1
    latency_steps_max: int = 5
    use_relative_action: bool = False
    relative_action_mask: str | tuple[bool, ...] | list[bool] | None = None
    center_crop: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    compile: bool = False
    use_cuda_graph: bool = False
    cuda_graph_warmup: int = 3
    debug: bool = False
    save_model_images: str | None = None
    return_confidence: bool = False
    confidence_threshold: float = 0.65
    min_action_horizon: int = 2
    confidence_cumulative_min: bool = True
    # When set alongside return_confidence: append each inference's per-step scores (JSONL) and log a summary.
    confidence_score_log: str | None = None

    @property
    def num_open_loop_steps(self) -> int:
        return self.action_horizon


_SAVE_MODEL_IMAGES_SEQ = 0
_SAVE_MODEL_IMAGES_LOCK = threading.Lock()
_CONFIDENCE_SCORE_LOG_LOCK = threading.Lock()


def _append_confidence_score_log(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object per line (thread-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _CONFIDENCE_SCORE_LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def _confidence_log_record(
    *,
    request: dict[str, Any],
    num_envs: int,
    latency_s: float,
    confidence_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a serializable record for file + logger (per-step scores per env)."""
    per_env: list[dict[str, Any]] = []
    action_conf = confidence_payload.get("action_confidence")
    eff = confidence_payload.get("effective_horizon")
    details = {
        k: confidence_payload[k]
        for k in (
            "raw_pred_confidence",
            "pred_to_action_attention",
            "action_attention_mass",
        )
        if k in confidence_payload
    }
    for i in range(num_envs):
        entry: dict[str, Any] = {"env_idx": i}
        if isinstance(action_conf, list) and i < len(action_conf):
            entry["action_confidence"] = action_conf[i]
        if isinstance(eff, list) and i < len(eff):
            entry["effective_horizon"] = eff[i]
        for k, v in details.items():
            if isinstance(v, list) and i < len(v):
                entry[k] = v[i]
        per_env.append(entry)
    return {
        "ts": time.time(),
        "latency_s": latency_s,
        "num_envs": num_envs,
        "step_ids": request.get("step_ids"),
        "task_description": request.get("task_description"),
        "per_env": per_env,
    }


class CUDAGraphActionRunner:
    """Replay the fixed-shape GPU action path with CUDA Graph.

    CPU preprocessing stays outside the graph. Captured graph covers VLM/action-head
    forward plus action unnormalization into a static output buffer.
    """

    _INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values")

    def __init__(
        self,
        *,
        policy: "VLAAdapterPolicy",
        predict_kwargs: dict[str, Any],
        warmup: int,
    ) -> None:
        if policy.device.type != "cuda":
            raise ValueError("CUDA Graph requires a CUDA device.")
        self.policy = policy
        self.device = policy.device
        self.warmup = max(1, int(warmup))
        self.graph = torch.cuda.CUDAGraph()
        self.static_kwargs: dict[str, Any] = {}
        self.static_inputs: dict[str, torch.Tensor] = {}
        self.signature = self._signature(predict_kwargs)
        self._validate_graph_inputs(predict_kwargs)
        self._init_static_kwargs(predict_kwargs)
        self._init_action_stats()
        self.static_output: torch.Tensor | None = None
        self._capture()

    @staticmethod
    def _tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
        return (tuple(value.shape), str(value.dtype), str(value.device))

    def _signature(self, predict_kwargs: dict[str, Any]) -> tuple[Any, ...]:
        parts: list[Any] = []
        for key in self._INPUT_KEYS:
            value = predict_kwargs[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"CUDA Graph input {key!r} must be a tensor.")
            parts.append((key, self._tensor_signature(value)))
        proprio = predict_kwargs.get("proprio")
        if proprio is None:
            parts.append(("proprio", None))
        else:
            proprio_tensor = torch.as_tensor(proprio)
            parts.append(("proprio", tuple(proprio_tensor.shape)))
        parts.append(("use_film", bool(predict_kwargs.get("use_film", False))))
        parts.append(("latency_steps", self._latency_signature(predict_kwargs.get("latency_steps"))))
        return tuple(parts)

    @staticmethod
    def _latency_signature(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
        if isinstance(value, (list, tuple)):
            return (len(value),)
        return ()

    def matches(self, predict_kwargs: dict[str, Any]) -> bool:
        return self.signature == self._signature(predict_kwargs)

    def _validate_graph_inputs(self, predict_kwargs: dict[str, Any]) -> None:
        attention_mask = predict_kwargs["attention_mask"]
        if not bool(torch.all(attention_mask == 1).item()):
            raise ValueError(
                "--use_cuda_graph currently requires an all-ones attention_mask "
                "(no padded prompts within the captured batch)."
            )

    def _clone_static_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().clone().to(self.device)

    def _init_static_kwargs(self, predict_kwargs: dict[str, Any]) -> None:
        for key, value in predict_kwargs.items():
            if key in self._INPUT_KEYS:
                static = self._clone_static_tensor(value)
                self.static_inputs[key] = static
                self.static_kwargs[key] = static
            elif key == "proprio" and value is not None:
                static = torch.as_tensor(value, device=self.device, dtype=torch.bfloat16).detach().clone()
                self.static_inputs[key] = static
                self.static_kwargs[key] = static
            elif key == "latency_steps" and value is not None:
                static = torch.as_tensor(value, device=self.device, dtype=torch.bfloat16).detach().clone()
                self.static_inputs[key] = static
                self.static_kwargs[key] = static
            else:
                self.static_kwargs[key] = value
        self.static_kwargs["return_normalized_tensor"] = True

    def _init_action_stats(self) -> None:
        stats = self.policy.model.get_action_stats(self.policy.unnorm_key)
        if "q01" in stats and "q99" in stats:
            low = np.asarray(stats["q01"], dtype=np.float32)
            high = np.asarray(stats["q99"], dtype=np.float32)
        else:
            low = np.asarray(stats["min"], dtype=np.float32)
            high = np.asarray(stats["max"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
        self.action_low = torch.as_tensor(low, device=self.device, dtype=torch.float32).view(1, 1, -1)
        self.action_high = torch.as_tensor(high, device=self.device, dtype=torch.float32).view(1, 1, -1)
        self.action_mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).view(1, 1, -1)

    def _unnormalize_tensor(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        normalized_actions = normalized_actions.float()
        unnormalized = 0.5 * (normalized_actions + 1.0) * (self.action_high - self.action_low + 1e-8) + self.action_low
        return torch.where(self.action_mask, unnormalized, normalized_actions)

    def _run_static_model(self) -> torch.Tensor:
        normalized_actions, _ = self.policy.model.predict_action(**self.static_kwargs)
        return self._unnormalize_tensor(normalized_actions)

    def _capture(self) -> None:
        torch.cuda.synchronize(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            with torch.inference_mode():
                for _ in range(self.warmup):
                    actions = self._run_static_model()
                self.static_output = torch.empty_like(actions)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        if self.static_output is None:
            raise RuntimeError("Failed to initialize CUDA Graph output buffer.")

        with torch.cuda.graph(self.graph):
            with torch.inference_mode():
                actions = self._run_static_model()
                self.static_output.copy_(actions)

    def _copy_inputs(self, predict_kwargs: dict[str, Any]) -> None:
        for key in self._INPUT_KEYS:
            self.static_inputs[key].copy_(predict_kwargs[key])
        if "proprio" in self.static_inputs:
            self.static_inputs["proprio"].copy_(
                torch.as_tensor(predict_kwargs["proprio"], device=self.device, dtype=self.static_inputs["proprio"].dtype)
            )
        if "latency_steps" in self.static_inputs:
            self.static_inputs["latency_steps"].copy_(
                torch.as_tensor(
                    predict_kwargs["latency_steps"],
                    device=self.device,
                    dtype=self.static_inputs["latency_steps"].dtype,
                )
            )

    def replay(self, predict_kwargs: dict[str, Any]) -> np.ndarray:
        self._copy_inputs(predict_kwargs)
        self.graph.replay()
        if self.static_output is None:
            raise RuntimeError("CUDA Graph output buffer is not initialized.")
        return self.static_output.detach().cpu().numpy()


class VLAAdapterPolicy:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        if cfg.use_future_conf and not cfg.use_future_pred:
            raise ValueError("--use_future_conf requires --use_future_pred.")
        if cfg.use_relative_action and cfg.relative_action_mask is None:
            raise ValueError("--use_relative_action requires --relative_action_mask.")
        if cfg.num_temporal_frames < 1:
            raise ValueError("--num_temporal_frames must be >= 1.")
        if cfg.use_latency_conditioning and (cfg.latency_steps < 0 or cfg.latency_steps_max < 1):
            raise ValueError("--use_latency_conditioning requires latency_steps >= 0 and latency_steps_max >= 1.")
        if cfg.use_cuda_graph and not torch.cuda.is_available():
            raise ValueError("--use_cuda_graph requires CUDA.")
        self.device = torch.device(cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.cuda_graph_runner: CUDAGraphActionRunner | None = None

        # Import after parsing robot_platform so prismatic.vla.constants chooses
        # the same dimensions used by the checkpoint.
        _status(f"debug={cfg.debug}, device={self.device}, checkpoint={cfg.pretrained_checkpoint}")
        _status("importing VLA-Adapter utilities")
        import experiments.robot.openvla_utils as openvla_utils

        openvla_utils.DEVICE = self.device
        from experiments.robot.openvla_utils import (
            absolute_actions_from_relative,
            get_action_head,
            get_latency_projector,
            get_noisy_action_projector,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
            normalize_proprio,
            prepare_images_for_vla,
        )

        _status("loading main VLA model")
        self._absolute_actions_from_relative = absolute_actions_from_relative
        self._get_vla_action = get_vla_action
        self._normalize_proprio = normalize_proprio
        self._prepare_images_for_vla = prepare_images_for_vla
        self.model = get_vla(cfg)
        _status("loading processor")
        self.processor = get_processor(cfg)
        _status("resolving unnorm_key")
        self.unnorm_key = self._resolve_unnorm_key(cfg.unnorm_key)

        self.proprio_projector = None
        if cfg.use_proprio:
            _status("loading proprio projector")
            self.proprio_projector = get_proprio_projector(cfg, self.model.llm_dim, proprio_dim=cfg.proprio_dim)

        self.latency_projector = None
        if cfg.use_latency_conditioning:
            _status("loading latency projector")
            self.latency_projector = get_latency_projector(cfg, self.model.llm_dim)

        if cfg.use_future_pred:
            _status("loading future-prediction components and enabling future-pred branch")
            from experiments.robot.openvla_utils import find_checkpoint_file

            try:
                ckpt_path = find_checkpoint_file(cfg.pretrained_checkpoint, "pred_components")
            except AssertionError as exc:
                raise FileNotFoundError(
                    "--use_future_pred was set, but this checkpoint does not contain "
                    "`pred_components--*_checkpoint.pt`. Use a checkpoint trained with "
                    "`--use_future_pred True`, or remove `--use_future_pred` for this checkpoint."
                ) from exc
            _status(f"loading pred_components from {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model.pred_queries.load_state_dict(state["pred_queries"])
            pred_head_state = state["pred_head"]
            if "weight" not in pred_head_state and "base_layer.weight" in pred_head_state:
                # Backward compatibility for early checkpoints where pred_head was accidentally
                # LoRA-wrapped by target_modules="all-linear". Training used lora_alpha=2*r, so
                # the PEFT scaling is alpha / r = 2. Merge it into the base Linear weight.
                base_weight = pred_head_state["base_layer.weight"]
                lora_a = pred_head_state["lora_A.default.weight"]
                lora_b = pred_head_state["lora_B.default.weight"]
                pred_head_state = {"weight": base_weight + 2.0 * (lora_b @ lora_a)}
            self.model.pred_head.load_state_dict(pred_head_state)
            use_future_conf = bool(state.get("use_future_conf", cfg.use_future_conf))
            if use_future_conf:
                if "pred_confidence_head" not in state:
                    raise FileNotFoundError(
                        "--use_future_conf was set, but pred_components does not contain "
                        "`pred_confidence_head`."
                    )
                self.model.pred_confidence_head.load_state_dict(state["pred_confidence_head"])
            self.model.pred_queries.to(self.device, dtype=torch.bfloat16)
            self.model.pred_head.to(self.device, dtype=torch.bfloat16)
            self.model.pred_confidence_head.to(self.device, dtype=torch.bfloat16)
            self.model.set_use_future_pred(True)
            pred_before_action = bool(state.get("pred_tokens_before_action", cfg.pred_tokens_before_action))
            self.cfg.pred_tokens_before_action = pred_before_action
            self.model.set_pred_tokens_before_action(pred_before_action)
            self.cfg.use_future_conf = use_future_conf
            self.cfg.future_confidence_gamma = float(
                state.get("future_confidence_gamma", cfg.future_confidence_gamma)
            )
            self.model.set_use_future_conf(
                self.cfg.use_future_conf,
                self.cfg.future_confidence_gamma,
            )
            _status("future-pred branch enabled")

        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            _status("loading action head")
            self.action_head = get_action_head(cfg, self.model.llm_dim)
            if hasattr(self.action_head, "use_x0_prediction"):
                self.action_head.use_x0_prediction = False

        self.noisy_action_projector = None
        if cfg.use_diffusion:
            _status("loading noisy action projector")
            self.noisy_action_projector = get_noisy_action_projector(cfg, self.model.llm_dim)

        if cfg.compile:
            if cfg.use_cuda_graph:
                raise ValueError("--compile and --use_cuda_graph should not be enabled together.")
            _status("compiling VLA model with torch.compile")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        _status(
            "loaded VLA-Adapter policy on "
            f"{self.device}; unnorm_key={self.unnorm_key!r}, "
            f"action_horizon={cfg.action_horizon}, action_dim={cfg.action_dim}"
        )
        if cfg.confidence_score_log and not cfg.return_confidence:
            logger.warning(
                "--confidence-score-log is set but --return_confidence is off; "
                "no per-step scores will be produced. Enable --return_confidence."
            )
        elif cfg.return_confidence and cfg.confidence_score_log:
            _status(f"per-step confidence scores -> JSONL append: {cfg.confidence_score_log}")

    def _resolve_unnorm_key(self, requested: str) -> str:
        norm_stats = getattr(self.model, "norm_stats", {})
        if requested:
            if requested not in norm_stats:
                raise ValueError(f"Unknown unnorm_key {requested!r}; available keys: {list(norm_stats.keys())}")
            return requested
        if len(norm_stats) == 1:
            key = next(iter(norm_stats.keys()))
            self.cfg.unnorm_key = key
            return key
        if not norm_stats:
            raise ValueError("Checkpoint does not provide norm_stats; cannot unnormalize VLA actions.")
        raise ValueError(f"Please pass --unnorm_key. Available keys: {list(norm_stats.keys())}")

    def info(self) -> dict[str, Any]:
        return {
            "action_dim": self.cfg.action_dim,
            "action_horizon": self.cfg.action_horizon,
            "model_name": Path(self.cfg.pretrained_checkpoint).name,
            "control_mode": self.cfg.control_mode,
            "proprio_dim": self.cfg.proprio_dim,
            "unnorm_key": self.unnorm_key,
            "num_images_in_input": self.cfg.num_images_in_input,
            "num_temporal_frames": self.cfg.num_temporal_frames,
            "temporal_fusion_type": self.cfg.temporal_fusion_type,
            "use_current_query_temporal_attention": self.cfg.use_current_query_temporal_attention,
            "use_mid_layer_temporal_fusion": self.cfg.use_mid_layer_temporal_fusion,
            "use_cuda_graph": self.cfg.use_cuda_graph,
            "use_relative_action": self.cfg.use_relative_action,
            "relative_action_mask": self.cfg.relative_action_mask,
            "use_future_pred": self.cfg.use_future_pred,
            "pred_tokens_before_action": self.cfg.pred_tokens_before_action,
            "use_future_conf": self.cfg.use_future_conf,
            "future_confidence_gamma": self.cfg.future_confidence_gamma,
            "return_confidence": self.cfg.return_confidence,
            "confidence_threshold": self.cfg.confidence_threshold,
            "min_action_horizon": self.cfg.min_action_horizon,
            "confidence_score_log_enabled": bool(self.cfg.confidence_score_log),
        }

    def predict_one(self, request: dict[str, Any], env_idx: int) -> np.ndarray:
        if self.cfg.debug:
            logger.info("env %s: decoding images and building observation", env_idx)

        num_temporal_frames = int(self.cfg.num_temporal_frames)
        primary_frames = _select_image_frames(
            request,
            env_idx,
            ("fixed_cam", "static", "rgb_static", "image", "full_image"),
            num_temporal_frames,
        )
        wrist_frame_lists = _select_wrist_image_frames(request, env_idx, num_temporal_frames)

        obs = {
            "full_image": primary_frames if num_temporal_frames > 1 else primary_frames[-1],
        }
        for i, frames in enumerate(wrist_frame_lists):
            obs[f"wrist_{i}"] = frames if num_temporal_frames > 1 else frames[-1]

        if len(obs) < self.cfg.num_images_in_input:
            raise ValueError(
                f"Request supplied {len(obs)} image(s), but the model expects {self.cfg.num_images_in_input}."
            )

        if self.cfg.use_proprio or self.cfg.use_relative_action:
            obs["state"] = _build_proprio_state(request, env_idx)

        if self.cfg.save_model_images:
            _save_policy_input_images(obs, Path(self.cfg.save_model_images), env_idx)

        task_description = request.get("task_description", "")
        if isinstance(task_description, list):
            task_description = task_description[env_idx]

        if self.cfg.debug:
            logger.info("env %s: running VLA inference", env_idx)
        latency_steps = _resolve_latency_steps(request, self.cfg, env_idx)
        actions = self._get_vla_action(
            cfg=self.cfg,
            vla=self.model,
            processor=self.processor,
            obs=obs,
            task_label=str(task_description),
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            latency_projector=self.latency_projector,
            latency_steps=latency_steps,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm,
        )
        action_chunk = _ensure_action_chunk(np.asarray(actions, dtype=np.float32), self.cfg.action_horizon, self.cfg.action_dim)
        if self.cfg.debug:
            logger.info("env %s: inference done, action_chunk shape=%s", env_idx, action_chunk.shape)
        return action_chunk

    def predict_batch(self, request: dict[str, Any], num_envs: int) -> tuple[np.ndarray | list[list[list[float]]], dict[str, Any] | None]:
        if self.cfg.debug:
            logger.info("batch: decoding %s envs and building batched observation", num_envs)

        num_temporal_frames = int(self.cfg.num_temporal_frames)
        temporal_primary_by_env = [
            _select_image_frames(
                request,
                i,
                ("fixed_cam", "static", "rgb_static", "image", "full_image"),
                num_temporal_frames,
            )
            for i in range(num_envs)
        ]
        temporal_wrist_by_env = [_select_wrist_image_frames(request, i, num_temporal_frames) for i in range(num_envs)]

        for i, wrist_frame_lists in enumerate(temporal_wrist_by_env):
            supplied = 1 + len(wrist_frame_lists)
            if supplied < self.cfg.num_images_in_input:
                raise ValueError(
                    f"Env {i} supplied {supplied} image(s), but the model expects {self.cfg.num_images_in_input}."
                )

        if self.cfg.save_model_images:
            out_dir = Path(self.cfg.save_model_images)
            for env_idx, primary_frames in enumerate(temporal_primary_by_env):
                obs = {"full_image": primary_frames[-1]}
                for wrist_idx, wrist_frames in enumerate(temporal_wrist_by_env[env_idx]):
                    obs[f"wrist_{wrist_idx}"] = wrist_frames[-1]
                _save_policy_input_images(obs, out_dir, env_idx)

        task_description = request.get("task_description", "")
        if isinstance(task_description, list):
            prompts = [self._build_prompt(str(task_description[i])) for i in range(num_envs)]
        else:
            prompts = [self._build_prompt(str(task_description)) for _ in range(num_envs)]

        with torch.inference_mode():
            processed_primary = self._prepare_images_for_vla(
                [frames[-1] for frames in temporal_primary_by_env],
                self.cfg,
            )
            inputs = self.processor(prompts, processed_primary, padding=True).to(self.device, dtype=torch.bfloat16)

            wrist_slots = min(
                max((len(wrist_frames) for wrist_frames in temporal_wrist_by_env), default=0),
                self.cfg.num_images_in_input - 1,
            )
            pixel_value_chunks = []
            for frame_idx in range(num_temporal_frames):
                frame_images = [frames[frame_idx] for frames in temporal_primary_by_env]
                processed_frame = self._prepare_images_for_vla(frame_images, self.cfg)
                frame_inputs = self.processor(prompts, processed_frame, padding=True).to(self.device, dtype=torch.bfloat16)
                pixel_value_chunks.append(frame_inputs["pixel_values"])

            for slot in range(wrist_slots):
                for frame_idx in range(num_temporal_frames):
                    slot_images = []
                    for env_idx, env_wrist_frames in enumerate(temporal_wrist_by_env):
                        if slot >= len(env_wrist_frames):
                            raise ValueError(f"Env {env_idx} is missing wrist image slot {slot}.")
                        slot_images.append(env_wrist_frames[slot][frame_idx])

                    processed_slot = self._prepare_images_for_vla(slot_images, self.cfg)
                    slot_inputs = self.processor(prompts, processed_slot, padding=True).to(self.device, dtype=torch.bfloat16)
                    pixel_value_chunks.append(slot_inputs["pixel_values"])

            if pixel_value_chunks:
                inputs["pixel_values"] = torch.cat(pixel_value_chunks, dim=1)

            proprio = None
            proprio_raw = None
            if self.cfg.use_proprio or self.cfg.use_relative_action:
                proprio_raw = np.stack([_build_proprio_state(request, i) for i in range(num_envs)], axis=0)
            if self.cfg.use_proprio:
                proprio_norm_stats = self.model.norm_stats[self.cfg.unnorm_key]["proprio"]
                proprio = self._normalize_proprio(proprio_raw, proprio_norm_stats)

            if self.cfg.debug:
                logger.info(
                    "batch: running VLA inference, input_ids=%s, pixel_values=%s, proprio=%s",
                    tuple(inputs["input_ids"].shape),
                    tuple(inputs["pixel_values"].shape),
                    None if proprio is None else proprio.shape,
                )

            latency_steps = _resolve_latency_steps(request, self.cfg)
            predict_kwargs = dict(
                **inputs,
                unnorm_key=self.cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                latency_steps=latency_steps,
                latency_projector=self.latency_projector,
                latency_steps_scale=self.cfg.latency_steps_max,
                noisy_action_projector=self.noisy_action_projector,
                action_head=self.action_head,
                use_film=self.cfg.use_film,
            )
            confidence_info = None
            if self.cfg.use_cuda_graph:
                if self.cfg.return_confidence:
                    raise ValueError("--use_cuda_graph does not support --return_confidence.")
                if self.action_head is None:
                    raise ValueError("--use_cuda_graph currently requires an action head.")
                if self.cuda_graph_runner is None or not self.cuda_graph_runner.matches(predict_kwargs):
                    _status("capturing CUDA Graph for fixed-shape action inference")
                    self.cuda_graph_runner = CUDAGraphActionRunner(
                        policy=self,
                        predict_kwargs=predict_kwargs,
                        warmup=self.cfg.cuda_graph_warmup,
                    )
                actions = self.cuda_graph_runner.replay(predict_kwargs)
            elif self.cfg.return_confidence:
                actions, _, confidence_info = self.model.predict_action(
                    **predict_kwargs,
                    return_pred_confidence=True,
                    pred_confidence_cumulative_min=self.cfg.confidence_cumulative_min,
                )
            else:
                actions, _ = self.model.predict_action(**predict_kwargs)

        if self.cfg.use_relative_action:
            if proprio_raw is None:
                raise ValueError("use_relative_action=True requires proprio state in request.")
            actions = self._absolute_actions_from_relative(actions, proprio_raw, self.cfg.relative_action_mask)

        action_batch = _ensure_action_batch(
            np.asarray(actions, dtype=np.float32),
            num_envs,
            self.cfg.action_horizon,
            self.cfg.action_dim,
        )
        response_info = None
        if confidence_info is not None and "pred_confidence" in confidence_info:
            confidence = _ensure_confidence_batch(
                np.asarray(confidence_info["pred_confidence"], dtype=np.float32),
                num_envs,
                self.cfg.action_horizon,
            )
            effective_horizon = _effective_horizons(
                confidence,
                threshold=self.cfg.confidence_threshold,
                min_horizon=self.cfg.min_action_horizon,
                max_horizon=self.cfg.action_horizon,
            )
            response_info = {
                "action_confidence": confidence.tolist(),
                "effective_horizon": effective_horizon.tolist(),
            }
            action_payload = _truncate_action_batch(action_batch, effective_horizon)
            for name in (
                "raw_pred_confidence",
                "pred_to_action_attention",
                "action_attention_mass",
            ):
                if name in confidence_info:
                    response_info[name] = _ensure_confidence_batch(
                        np.asarray(confidence_info[name], dtype=np.float32),
                        num_envs,
                        self.cfg.action_horizon,
                    ).tolist()
        else:
            action_payload = action_batch
        if self.cfg.debug:
            if isinstance(action_payload, np.ndarray):
                logger.info("batch: inference done, action_batch shape=%s", action_payload.shape)
            else:
                logger.info(
                    "batch: inference done, action lengths=%s",
                    [len(chunk) for chunk in action_payload],
                )
        return action_payload, response_info

    def _build_prompt(self, task_label: str) -> str:
        if not self.cfg.use_minivlm:
            return f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        return (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def reset(self, env_ids: list[int]) -> None:
        del env_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def decode_image(b64_str: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"), dtype=np.uint8)


def _describe_b64_blob(blob: Any) -> dict[str, Any]:
    s = str(blob)
    digest = hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return {"base64_chars": len(s), "sha256_12": digest}


def _describe_image_entry(entry: Any) -> Any:
    if isinstance(entry, list):
        return [_describe_b64_blob(blob) for blob in entry]
    return _describe_b64_blob(entry)


def _decode_image_entry(entry: Any, num_temporal_frames: int, camera_name: str, env_idx: int) -> list[np.ndarray]:
    """Decode one env's camera entry: either a single base64 string or a temporal frame list."""
    if isinstance(entry, str):
        if num_temporal_frames > 1:
            raise ValueError(
                f"Env {env_idx} camera {camera_name!r} sent a single image, but the server expects "
                f"{num_temporal_frames} temporal frames per camera. "
                f"Send a list like [prev_frame_b64, curr_frame_b64]."
            )
        return [decode_image(entry)]

    if isinstance(entry, list):
        if not entry:
            raise ValueError(f"Env {env_idx} camera {camera_name!r} has an empty temporal frame list.")
        if len(entry) != num_temporal_frames:
            raise ValueError(
                f"Env {env_idx} camera {camera_name!r} sent {len(entry)} frame(s), "
                f"but num_temporal_frames={num_temporal_frames}."
            )
        return [decode_image(blob) for blob in entry]

    raise ValueError(
        f"Env {env_idx} camera {camera_name!r} must be a base64 string or a list of base64 strings, "
        f"got {type(entry).__name__}."
    )


def _redact_predict_request_for_debug(request: dict[str, Any]) -> dict[str, Any]:
    """用于 debug：保留除 images 外的原始字段；images 只保留元信息，不打印 base64。"""
    out = dict(request)
    images = out.get("images")
    if isinstance(images, dict):
        redacted: dict[str, Any] = {}
        for cam, blobs in images.items():
            if isinstance(blobs, list):
                redacted[cam] = [_describe_image_entry(b) for b in blobs]
            else:
                redacted[cam] = blobs
        out["images"] = redacted
    return out


def _debug_json_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _log_predict_debug(phase: str, payload: dict[str, Any]) -> None:
    text = _debug_json_dump(payload)
    print(f"[policy_server][DEBUG] {phase}\n{text}", flush=True)
    logger.info("[DEBUG] %s\n%s", phase, text)


def _save_policy_input_images(obs: dict[str, Any], out_dir: Path, env_idx: int) -> None:
    global _SAVE_MODEL_IMAGES_SEQ
    with _SAVE_MODEL_IMAGES_LOCK:
        _SAVE_MODEL_IMAGES_SEQ += 1
        seq = _SAVE_MODEL_IMAGES_SEQ

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in obs.items():
        if isinstance(value, list):
            value = value[-1] if value else None
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        Image.fromarray(value).save(out_dir / f"{seq:08d}_{name}_env{env_idx}.png")


def _select_image_frames(
    request: dict[str, Any],
    env_idx: int,
    preferred_keys: tuple[str, ...],
    num_temporal_frames: int,
) -> list[np.ndarray]:
    images = request.get("images", {})
    for key in preferred_keys:
        values = images.get(key)
        if values is not None:
            return _decode_image_entry(values[env_idx], num_temporal_frames, key, env_idx)
    raise ValueError(f"Missing primary image. Expected one of: {preferred_keys}")


def _select_wrist_image_frames(
    request: dict[str, Any],
    env_idx: int,
    num_temporal_frames: int,
) -> list[list[np.ndarray]]:
    images = request.get("images", {})
    preferred = ("wrist_cam", "gripper", "rgb_gripper", "wrist_image")
    selected: list[list[np.ndarray]] = []
    used: set[str] = set()

    for key in preferred:
        values = images.get(key)
        if values is not None:
            selected.append(_decode_image_entry(values[env_idx], num_temporal_frames, key, env_idx))
            used.add(key)

    for key in sorted(images.keys()):
        if key in used or key in {"fixed_cam", "static", "rgb_static", "image", "full_image"}:
            continue
        if "wrist" in key or "gripper" in key:
            selected.append(_decode_image_entry(images[key][env_idx], num_temporal_frames, key, env_idx))

    return selected


def _build_proprio_state(request: dict[str, Any], env_idx: int) -> np.ndarray:
    proprio = request.get("proprioception", {})
    if request.get("state") is not None:
        return np.asarray(request["state"][env_idx], dtype=np.float32).reshape(-1)

    if "state" in proprio:
        return np.asarray(proprio["state"][env_idx], dtype=np.float32)

    gripper = np.asarray(proprio.get("gripper_state", [[0.0]])[env_idx], dtype=np.float32).reshape(-1)
    if "joint_positions" in proprio:
        joints = np.asarray(proprio["joint_positions"][env_idx], dtype=np.float32).reshape(-1)
        return np.concatenate([joints, gripper], axis=0)

    if "eef_pos" in proprio and "eef_orient" in proprio:
        eef_pos = np.asarray(proprio["eef_pos"][env_idx], dtype=np.float32).reshape(-1)
        eef_orient = np.asarray(proprio["eef_orient"][env_idx], dtype=np.float32).reshape(-1)
        return np.concatenate([eef_pos, eef_orient, gripper], axis=0)

    raise ValueError("Missing proprioception. Expected state, joint_positions, or eef_pos/eef_orient.")


def _resolve_latency_steps(request: dict[str, Any], cfg: ServerConfig, env_idx: int | None = None) -> int | list[int] | None:
    if not cfg.use_latency_conditioning:
        return None

    raw_steps = request.get("latency_steps", cfg.latency_steps)
    if env_idx is not None and isinstance(raw_steps, (list, tuple)):
        raw_steps = raw_steps[env_idx]

    if env_idx is None and isinstance(raw_steps, (list, tuple)):
        values = [int(round(float(value))) for value in raw_steps]
        if any(value < 0 for value in values):
            raise ValueError("latency_steps must be non-negative.")
        return values

    value = int(round(float(raw_steps)))
    if value < 0:
        raise ValueError("latency_steps must be non-negative.")
    return value


def _ensure_action_chunk(actions: np.ndarray, horizon: int, action_dim: int) -> np.ndarray:
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(f"Expected a single action chunk, got batched actions with shape {actions.shape}.")
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk with shape (H, D), got {actions.shape}.")
    if actions.shape[-1] != action_dim:
        raise ValueError(f"Expected action_dim={action_dim}, got {actions.shape[-1]}.")
    if actions.shape[0] >= horizon:
        return actions[:horizon]

    pad = np.repeat(actions[-1:], horizon - actions.shape[0], axis=0)
    return np.concatenate([actions, pad], axis=0)


def _ensure_action_batch(actions: np.ndarray, num_envs: int, horizon: int, action_dim: int) -> np.ndarray:
    if actions.ndim == 2:
        if num_envs != 1:
            raise ValueError(f"Expected batched actions for {num_envs} envs, got shape {actions.shape}.")
        actions = actions[None, :, :]
    if actions.ndim != 3:
        raise ValueError(f"Expected action batch with shape (N, H, D), got {actions.shape}.")
    if actions.shape[0] != num_envs:
        raise ValueError(f"Expected num_envs={num_envs}, got actions batch size {actions.shape[0]}.")
    if actions.shape[-1] != action_dim:
        raise ValueError(f"Expected action_dim={action_dim}, got {actions.shape[-1]}.")
    if actions.shape[1] >= horizon:
        return actions[:, :horizon, :]

    pad = np.repeat(actions[:, -1:, :], horizon - actions.shape[1], axis=1)
    return np.concatenate([actions, pad], axis=1)


def _ensure_confidence_batch(confidence: np.ndarray, num_envs: int, horizon: int) -> np.ndarray:
    if confidence.ndim == 1:
        if num_envs != 1:
            raise ValueError(f"Expected batched confidence for {num_envs} envs, got shape {confidence.shape}.")
        confidence = confidence[None, :]
    if confidence.ndim != 2:
        raise ValueError(f"Expected confidence batch with shape (N, H), got {confidence.shape}.")
    if confidence.shape[0] != num_envs:
        raise ValueError(f"Expected num_envs={num_envs}, got confidence batch size {confidence.shape[0]}.")
    confidence = confidence.astype(np.float32)
    if confidence.shape[1] >= horizon:
        return confidence[:, :horizon]

    pad = np.repeat(confidence[:, -1:], horizon - confidence.shape[1], axis=1)
    return np.concatenate([confidence, pad], axis=1)


def _effective_horizons(confidence: np.ndarray, threshold: float, min_horizon: int, max_horizon: int) -> np.ndarray:
    min_horizon = max(1, min(int(min_horizon), int(max_horizon)))
    effective = np.full((confidence.shape[0],), int(max_horizon), dtype=np.int64)
    low_confidence = confidence < float(threshold)
    for env_idx in range(confidence.shape[0]):
        low_steps = np.flatnonzero(low_confidence[env_idx])
        if low_steps.size:
            effective[env_idx] = max(min_horizon, int(low_steps[0]))
    return effective


def _truncate_action_batch(actions: np.ndarray, effective_horizon: np.ndarray) -> list[list[list[float]]]:
    truncated: list[list[list[float]]] = []
    for env_idx, horizon in enumerate(effective_horizon):
        horizon = int(horizon)
        truncated.append(actions[env_idx, :horizon].tolist())
    return truncated


CONFIG = ServerConfig()
POLICY: VLAAdapterPolicy | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global POLICY
    POLICY = VLAAdapterPolicy(CONFIG)
    _status("FastAPI lifespan ready; waiting for HTTP requests")
    try:
        yield
    finally:
        _status("shutting down policy server")
        POLICY = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


app = FastAPI(title="VLA-Adapter Policy Server", lifespan=lifespan)


@app.get("/info")
def info() -> dict[str, Any]:
    if POLICY is None:
        raise HTTPException(status_code=503, detail="Policy is not loaded.")
    resp = POLICY.info()
    if CONFIG.debug:
        _status(f"GET /info -> {resp}")
    return resp


@app.post("/predict")
def predict(request: dict[str, Any]) -> dict[str, Any]:
    if POLICY is None:
        raise HTTPException(status_code=503, detail="Policy is not loaded.")
    if request.get("type") != "vla":
        err = {"error": "This server only supports requests with type='vla'.", "latency_s": 0.0}
        if CONFIG.debug:
            _log_predict_debug("POST /predict 输出 (full)", err)
        return err

    t0 = time.monotonic()
    if CONFIG.debug:
        req_dbg = _redact_predict_request_for_debug(request)
        _log_predict_debug("POST /predict 输入 (full, images redacted)", req_dbg)

    try:
        num_envs = int(request["num_envs"])
        stacked, confidence_info = POLICY.predict_batch(request, num_envs)
    except Exception as exc:
        err = {"error": str(exc), "latency_s": round(time.monotonic() - t0, 4)}
        logger.exception("POST /predict failed")
        if CONFIG.debug:
            _log_predict_debug("POST /predict 输出 (full)", err)
        return err

    actions_payload = stacked.tolist() if isinstance(stacked, np.ndarray) else stacked
    out = {"actions": actions_payload, "latency_s": round(time.monotonic() - t0, 4)}
    if CONFIG.use_latency_conditioning:
        out["latency_steps"] = _resolve_latency_steps(request, CONFIG)
    if confidence_info is not None:
        out.update(confidence_info)

    if (
        CONFIG.return_confidence
        and CONFIG.confidence_score_log
        and confidence_info is not None
    ):
        record = _confidence_log_record(
            request=request,
            num_envs=num_envs,
            latency_s=out["latency_s"],
            confidence_payload=confidence_info,
        )
        log_path = Path(CONFIG.confidence_score_log)
        _append_confidence_score_log(log_path, record)
        logger.info(
            "confidence_scores inference -> %s | %s",
            log_path,
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str),
        )

    if CONFIG.debug:
        _log_predict_debug("POST /predict 输出 (full)", out)
    return out


@app.post("/reset")
async def reset(request: dict[str, Any]) -> dict[str, str]:
    if POLICY is not None:
        POLICY.reset([int(env_id) for env_id in request.get("env_ids", [])])
    return {"status": "ok"}


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(
        description="VLA-Adapter Policy Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python policy_server.py --pretrained_path outputs/..._chkpt --device cuda:0 --debug
  python policy_server.py --pretrained_checkpoint outputs/..._chkpt --port 9000
""",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        "--pretrained_path",
        dest="pretrained_checkpoint",
        default=CONFIG.pretrained_checkpoint,
        help="HuggingFace repo id 或本地 VLA-Adapter checkpoint 路径",
    )
    parser.add_argument("--device", default=CONFIG.device, help="推理设备，如 cuda:0 / cuda:1 / cpu")
    parser.add_argument("--host", default=CONFIG.host)
    parser.add_argument("--port", type=int, default=CONFIG.port)
    parser.add_argument("--robot_platform", default=CONFIG.robot_platform)
    parser.add_argument("--unnorm_key", default=CONFIG.unnorm_key)
    parser.add_argument("--control_mode", default=CONFIG.control_mode)
    parser.add_argument("--num_images_in_input", type=int, default=CONFIG.num_images_in_input)
    parser.add_argument("--num_temporal_frames", type=int, default=CONFIG.num_temporal_frames)
    parser.add_argument("--temporal_fusion_type", default=CONFIG.temporal_fusion_type, choices=("attention", "delta_mlp"))
    parser.add_argument(
        "--use_current_query_temporal_attention",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.use_current_query_temporal_attention,
    )
    parser.add_argument(
        "--use_mid_layer_temporal_fusion",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.use_mid_layer_temporal_fusion,
    )
    parser.add_argument("--action_horizon", type=int, default=CONFIG.action_horizon)
    parser.add_argument("--action_dim", type=int, default=CONFIG.action_dim)
    parser.add_argument("--proprio_dim", type=int, default=CONFIG.proprio_dim)
    parser.add_argument("--use_proprio", action=argparse.BooleanOptionalAction, default=CONFIG.use_proprio)
    parser.add_argument("--use_l1_regression", action=argparse.BooleanOptionalAction, default=CONFIG.use_l1_regression)
    parser.add_argument("--use_diffusion", action=argparse.BooleanOptionalAction, default=CONFIG.use_diffusion)
    parser.add_argument("--use_film", action=argparse.BooleanOptionalAction, default=CONFIG.use_film)
    parser.add_argument("--use_minivlm", action=argparse.BooleanOptionalAction, default=CONFIG.use_minivlm)
    parser.add_argument("--use_pro_version", action=argparse.BooleanOptionalAction, default=CONFIG.use_pro_version)
    parser.add_argument("--use_future_pred", action=argparse.BooleanOptionalAction, default=CONFIG.use_future_pred)
    parser.add_argument("--use_latency_conditioning", action=argparse.BooleanOptionalAction, default=CONFIG.use_latency_conditioning)
    parser.add_argument("--latency_steps", type=int, default=CONFIG.latency_steps)
    parser.add_argument("--latency_steps_max", type=int, default=CONFIG.latency_steps_max)
    parser.add_argument(
        "--use_future_conf",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.use_future_conf,
    )
    parser.add_argument("--future_confidence_gamma", type=float, default=CONFIG.future_confidence_gamma)
    parser.add_argument(
        "--pred_tokens_before_action",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.pred_tokens_before_action,
    )
    parser.add_argument("--use_relative_action", action=argparse.BooleanOptionalAction, default=CONFIG.use_relative_action)
    parser.add_argument(
        "--relative_action_mask",
        default=CONFIG.relative_action_mask,
        help="Comma-separated bool mask, e.g. true,true,true,true,true,true,true,false",
    )
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=CONFIG.center_crop)
    parser.add_argument("--load_in_8bit", action=argparse.BooleanOptionalAction, default=CONFIG.load_in_8bit)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=CONFIG.load_in_4bit)
    parser.add_argument("--compile", action="store_true", default=CONFIG.compile)
    parser.add_argument("--use_cuda_graph", action=argparse.BooleanOptionalAction, default=CONFIG.use_cuda_graph)
    parser.add_argument("--cuda_graph_warmup", type=int, default=CONFIG.cuda_graph_warmup)
    parser.add_argument("--debug", action="store_true", default=CONFIG.debug)
    parser.add_argument("--save-model-images", dest="save_model_images", default=CONFIG.save_model_images, metavar="DIR")
    parser.add_argument("--return_confidence", action=argparse.BooleanOptionalAction, default=CONFIG.return_confidence)
    parser.add_argument("--confidence_threshold", type=float, default=CONFIG.confidence_threshold)
    parser.add_argument("--min_action_horizon", type=int, default=CONFIG.min_action_horizon)
    parser.add_argument(
        "--confidence_cumulative_min",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.confidence_cumulative_min,
    )
    parser.add_argument(
        "--confidence-score-log",
        dest="confidence_score_log",
        default=CONFIG.confidence_score_log,
        metavar="PATH",
        help="With --return_confidence: append each inference's per-step scores as one JSON line; also log the same line at INFO.",
    )
    args = parser.parse_args()
    return ServerConfig(**vars(args))


if __name__ == "__main__":
    CONFIG = parse_args()
    if CONFIG.robot_platform.lower() not in " ".join(sys.argv).lower():
        sys.argv.append(CONFIG.robot_platform)
    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port)
