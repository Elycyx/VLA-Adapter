#!/usr/bin/env python3
"""HTTP policy server for VLA-Adapter inference.

The API matches POLICY_SERVER.md and is consumed by policy_client.py:
    GET  /info
    POST /predict
    POST /reset

Debug:
    --debug 会在每次 POST /predict 打印完整请求与完整响应 JSON（同时 stdout + logger）。
    为避免日志爆炸，默认不会打印 images 里的 base64 正文，只会打印长度与短 hash。
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
    action_horizon: int = 16
    action_dim: int = 8
    proprio_dim: int = 8
    use_proprio: bool = True
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_minivlm: bool = True
    use_pro_version: bool = True
    center_crop: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    compile: bool = False
    debug: bool = False
    save_model_images: str | None = None

    @property
    def num_open_loop_steps(self) -> int:
        return self.action_horizon


_SAVE_MODEL_IMAGES_SEQ = 0
_SAVE_MODEL_IMAGES_LOCK = threading.Lock()


class VLAAdapterPolicy:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # Import after parsing robot_platform so prismatic.vla.constants chooses
        # the same dimensions used by the checkpoint.
        _status(f"debug={cfg.debug}, device={self.device}, checkpoint={cfg.pretrained_checkpoint}")
        _status("importing VLA-Adapter utilities")
        import experiments.robot.openvla_utils as openvla_utils

        openvla_utils.DEVICE = self.device
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_noisy_action_projector,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
            normalize_proprio,
            prepare_images_for_vla,
        )

        _status("loading main VLA model")
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
            _status("compiling VLA model with torch.compile")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        _status(
            "loaded VLA-Adapter policy on "
            f"{self.device}; unnorm_key={self.unnorm_key!r}, "
            f"action_horizon={cfg.action_horizon}, action_dim={cfg.action_dim}"
        )

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
        }

    def predict_one(self, request: dict[str, Any], env_idx: int) -> np.ndarray:
        if self.cfg.debug:
            logger.info("env %s: decoding images and building observation", env_idx)

        obs = {
            "full_image": _select_image(request, env_idx, ("fixed_cam", "static", "rgb_static", "image", "full_image")),
        }

        wrist_images = _select_wrist_images(request, env_idx)
        for i, image in enumerate(wrist_images):
            obs[f"wrist_{i}"] = image

        if len(obs) < self.cfg.num_images_in_input:
            raise ValueError(
                f"Request supplied {len(obs)} image(s), but the model expects {self.cfg.num_images_in_input}."
            )

        if self.cfg.use_proprio:
            obs["state"] = _build_proprio_state(request, env_idx)

        if self.cfg.save_model_images:
            _save_policy_input_images(obs, Path(self.cfg.save_model_images), env_idx)

        task_description = request.get("task_description", "")
        if isinstance(task_description, list):
            task_description = task_description[env_idx]

        if self.cfg.debug:
            logger.info("env %s: running VLA inference", env_idx)
        actions = self._get_vla_action(
            cfg=self.cfg,
            vla=self.model,
            processor=self.processor,
            obs=obs,
            task_label=str(task_description),
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm,
        )
        action_chunk = _ensure_action_chunk(np.asarray(actions, dtype=np.float32), self.cfg.action_horizon, self.cfg.action_dim)
        if self.cfg.debug:
            logger.info("env %s: inference done, action_chunk shape=%s", env_idx, action_chunk.shape)
        return action_chunk

    def predict_batch(self, request: dict[str, Any], num_envs: int) -> np.ndarray:
        if self.cfg.debug:
            logger.info("batch: decoding %s envs and building batched observation", num_envs)

        primary_images = [
            _select_image(request, i, ("fixed_cam", "static", "rgb_static", "image", "full_image"))
            for i in range(num_envs)
        ]
        wrist_images_by_env = [_select_wrist_images(request, i) for i in range(num_envs)]

        for i, wrist_images in enumerate(wrist_images_by_env):
            supplied = 1 + len(wrist_images)
            if supplied < self.cfg.num_images_in_input:
                raise ValueError(
                    f"Env {i} supplied {supplied} image(s), but the model expects {self.cfg.num_images_in_input}."
                )

        if self.cfg.save_model_images:
            out_dir = Path(self.cfg.save_model_images)
            for env_idx, primary in enumerate(primary_images):
                obs = {"full_image": primary}
                for wrist_idx, wrist in enumerate(wrist_images_by_env[env_idx]):
                    obs[f"wrist_{wrist_idx}"] = wrist
                _save_policy_input_images(obs, out_dir, env_idx)

        task_description = request.get("task_description", "")
        if isinstance(task_description, list):
            prompts = [self._build_prompt(str(task_description[i])) for i in range(num_envs)]
        else:
            prompts = [self._build_prompt(str(task_description)) for _ in range(num_envs)]

        with torch.inference_mode():
            processed_primary = self._prepare_images_for_vla(primary_images, self.cfg)
            inputs = self.processor(prompts, processed_primary, padding=True).to(self.device, dtype=torch.bfloat16)

            wrist_slots = min(max((len(images) for images in wrist_images_by_env), default=0), self.cfg.num_images_in_input - 1)
            wrist_pixel_values = []
            for slot in range(wrist_slots):
                slot_images = []
                for env_idx, env_wrist_images in enumerate(wrist_images_by_env):
                    if slot >= len(env_wrist_images):
                        raise ValueError(f"Env {env_idx} is missing wrist image slot {slot}.")
                    slot_images.append(env_wrist_images[slot])

                processed_slot = self._prepare_images_for_vla(slot_images, self.cfg)
                slot_inputs = self.processor(prompts, processed_slot, padding=True).to(self.device, dtype=torch.bfloat16)
                wrist_pixel_values.append(slot_inputs["pixel_values"])

            if wrist_pixel_values:
                inputs["pixel_values"] = torch.cat([inputs["pixel_values"]] + wrist_pixel_values, dim=1)

            proprio = None
            if self.cfg.use_proprio:
                proprio = np.stack([_build_proprio_state(request, i) for i in range(num_envs)], axis=0)
                proprio_norm_stats = self.model.norm_stats[self.cfg.unnorm_key]["proprio"]
                proprio = self._normalize_proprio(proprio, proprio_norm_stats)

            if self.cfg.debug:
                logger.info(
                    "batch: running VLA inference, input_ids=%s, pixel_values=%s, proprio=%s",
                    tuple(inputs["input_ids"].shape),
                    tuple(inputs["pixel_values"].shape),
                    None if proprio is None else proprio.shape,
                )

            actions, _ = self.model.predict_action(
                **inputs,
                unnorm_key=self.cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                action_head=self.action_head,
                use_film=self.cfg.use_film,
            )

        action_batch = _ensure_action_batch(
            np.asarray(actions, dtype=np.float32),
            num_envs,
            self.cfg.action_horizon,
            self.cfg.action_dim,
        )
        if self.cfg.debug:
            logger.info("batch: inference done, action_batch shape=%s", action_batch.shape)
        return action_batch

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


def _redact_predict_request_for_debug(request: dict[str, Any]) -> dict[str, Any]:
    """用于 debug：保留除 images 外的原始字段；images 只保留元信息，不打印 base64。"""
    out = dict(request)
    images = out.get("images")
    if isinstance(images, dict):
        redacted: dict[str, Any] = {}
        for cam, blobs in images.items():
            if isinstance(blobs, list):
                redacted[cam] = [_describe_b64_blob(b) for b in blobs]
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
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        Image.fromarray(value).save(out_dir / f"{seq:08d}_{name}_env{env_idx}.png")


def _select_image(request: dict[str, Any], env_idx: int, preferred_keys: tuple[str, ...]) -> np.ndarray:
    images = request.get("images", {})
    for key in preferred_keys:
        values = images.get(key)
        if values is not None:
            return decode_image(values[env_idx])
    raise ValueError(f"Missing primary image. Expected one of: {preferred_keys}")


def _select_wrist_images(request: dict[str, Any], env_idx: int) -> list[np.ndarray]:
    images = request.get("images", {})
    preferred = ("wrist_cam", "gripper", "rgb_gripper", "wrist_image")
    selected: list[np.ndarray] = []
    used: set[str] = set()

    for key in preferred:
        values = images.get(key)
        if values is not None:
            selected.append(decode_image(values[env_idx]))
            used.add(key)

    for key in sorted(images.keys()):
        if key in used or key in {"fixed_cam", "static", "rgb_static", "image", "full_image"}:
            continue
        if "wrist" in key or "gripper" in key:
            selected.append(decode_image(images[key][env_idx]))

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
        stacked = POLICY.predict_batch(request, num_envs)
    except Exception as exc:
        err = {"error": str(exc), "latency_s": round(time.monotonic() - t0, 4)}
        logger.exception("POST /predict failed")
        if CONFIG.debug:
            _log_predict_debug("POST /predict 输出 (full)", err)
        return err

    out = {"actions": stacked.tolist(), "latency_s": round(time.monotonic() - t0, 4)}
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
    parser.add_argument("--action_horizon", type=int, default=CONFIG.action_horizon)
    parser.add_argument("--action_dim", type=int, default=CONFIG.action_dim)
    parser.add_argument("--proprio_dim", type=int, default=CONFIG.proprio_dim)
    parser.add_argument("--use_proprio", action=argparse.BooleanOptionalAction, default=CONFIG.use_proprio)
    parser.add_argument("--use_l1_regression", action=argparse.BooleanOptionalAction, default=CONFIG.use_l1_regression)
    parser.add_argument("--use_diffusion", action=argparse.BooleanOptionalAction, default=CONFIG.use_diffusion)
    parser.add_argument("--use_film", action=argparse.BooleanOptionalAction, default=CONFIG.use_film)
    parser.add_argument("--use_minivlm", action=argparse.BooleanOptionalAction, default=CONFIG.use_minivlm)
    parser.add_argument("--use_pro_version", action=argparse.BooleanOptionalAction, default=CONFIG.use_pro_version)
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=CONFIG.center_crop)
    parser.add_argument("--load_in_8bit", action=argparse.BooleanOptionalAction, default=CONFIG.load_in_8bit)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=CONFIG.load_in_4bit)
    parser.add_argument("--compile", action="store_true", default=CONFIG.compile)
    parser.add_argument("--debug", action="store_true", default=CONFIG.debug)
    parser.add_argument("--save-model-images", dest="save_model_images", default=CONFIG.save_model_images, metavar="DIR")
    args = parser.parse_args()
    return ServerConfig(**vars(args))


if __name__ == "__main__":
    CONFIG = parse_args()
    if CONFIG.robot_platform.lower() not in " ".join(sys.argv).lower():
        sys.argv.append(CONFIG.robot_platform)
    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port)
