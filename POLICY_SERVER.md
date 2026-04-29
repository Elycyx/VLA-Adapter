# Policy Server 实现规范

本文档定义 `eval.py` 所需的 **Policy Server** HTTP 接口，兼容 RL 策略、VLA、Diffusion Policy、ACT 等 robot learning 方法。

---

## 1. 推理模式说明

`eval.py` 通过两种延迟模拟模式评估策略在存在推理延迟时的真实表现。
两种模式**都在推理时暂停仿真**（阻塞调用 `/predict`），共享同一套 Server API，
区别仅在客户端侧的**动作调度逻辑**。

两种模式均支持两种推理频率来源：
- `--infer_freq <Hz>`（固定频率，内部换算为 `infer_period_ms = 1000 / infer_freq`，确定性，适合对比实验；默认 20 Hz）
- `--use_real_latency`（每次推理测量实际 wall-clock 耗时 ms，适合真实性能评估）

推理周期在内部换算为仿真子步数：`sim_steps = round(infer_period_ms / (sim_dt × 1000))`，
与机械臂控制频率 `--ctrl_freq` 相互独立。例如 `--infer_freq 10` 在 100 Hz 仿真下始终等于 10 个物理子步 (100 ms)。

### Simulated Synchronous Inference (`--inference_mode sync`)

模拟**单线程串行**部署：推理完成后先用零动作快进 `infer_period_ms` 对应的仿真时间（惩罚），
再执行刚算出的动作 chunk（每个控制步执行 chunk 中的一步 action，共 H 步）。

```
cycle at sim time t:
  ① 暂停仿真，采集 O_t，阻塞调用 /predict → 得到 A_t
  ② 快进 infer_period_ms 零动作（惩罚：物理世界在无控制下演化）
     → 内部换算为 N 个物理子步，可能跨越多个控制步或不足一个控制步
  ③ 执行 A_t 的 H 步动作（每控制步一步 action）
  ④ 回到 ①
```

**核心效果**：走走停停。策略必须承担"思考太久导致物理世界失控"的后果。
使用 `--use_real_latency` 时，延迟随每次推理的实际耗时动态变化。

### Simulated Asynchronous Inference (`--inference_mode async`)

模拟**多线程流水线**部署：推理完成后执行**上一轮**的动作，本轮结果留给下一轮用。

```
cycle at sim time t:
  ① 暂停仿真，采集 O_t，阻塞调用 /predict → 得到 A_t
  ② 执行 prev_actions（上一轮推理结果）infer_period_ms 时长
     → 每个控制步执行 chunk 中的一步 action，可能跨越多个控制步或不足一个
  ③ 存储 A_t 作为下一轮的 prev_actions
  ④ 回到 ①

首轮无 prev_actions → 执行零动作。
```

**核心效果**：动作永远比观测滞后一个推理周期，还原真实世界异步流水线中的**观测滞后性**：
"我此刻看到的画面 O_t，我针对它做出的决策，要在 infer_period_ms 之后才真正作用于环境"。
如果 action_horizon(H) 不够填满延迟期间的控制步，不足部分退化为零动作。
使用 `--use_real_latency` 时，延迟随实际推理耗时波动。

---

## 2. API 端点

### `GET /info`

返回模型元信息，客户端启动时调用一次。

```json
{
    "action_dim": 8,
    "action_horizon": 16,
    "model_name": "diffusion_policy_franka",
    "control_mode": "joint_pos"
}
```

- `action_dim` **(必需)**：单步动作维度
- `action_horizon` **(必需)**：每次推理返回的动作步数 `H`

### `POST /predict`

核心推理端点。接收批量观测，返回批量动作序列。

**请求** — State 模式（RL 策略）：

```json
{
    "type": "state",
    "num_envs": 5,
    "state": [[0.1, 0.2, ...], ...],
    "step_ids": [42, 10, 5, 3, 27]
}
```

**请求** — VLA 模式（视觉-语言-动作模型）：

```json
{
    "type": "vla",
    "num_envs": 5,
    "control_mode": "joint_pos",
    "state_format": "both",
    "orientation_rep": "quat",
    "proprioception": {
        "joint_positions": [[...], ...],
        "gripper_state": [[1.0], ...],
        "eef_pos": [[0.3, 0.1, 0.4], ...],
        "eef_orient": [[1.0, 0.0, 0.0, 0.0], ...]
    },
    "images": {
        "fixed_cam": ["<base64 JPEG>", ...],
        "wrist_cam": ["<base64 JPEG>", ...]
    },
    "task_description": "pick up the red cube from the conveyor",
    "step_ids": [42, 10, 5, 3, 27]
}
```

VLA 请求附加字段说明：

| 字段 | 含义 | 取值 |
|------|------|------|
| `control_mode` | 当前动作空间 | `joint_pos` / `ik_abs` / `ik_rel` |
| `state_format` | proprioception 中包含哪些项 | `joint`（仅关节）/ `eef_pose`（仅末端）/ `both` |
| `orientation_rep` | `eef_orient` 的表示方式 | `quat`（w,x,y,z 4D）/ `euler`（roll,pitch,yaw 3D）|

- `state_format=joint` 时：proprioception 只含 `joint_positions` + `gripper_state`
- `state_format=eef_pose` 时：proprioception 只含 `eef_pos` + `eef_orient` + `gripper_state`
- `state_format=both` 时：全部包含
```

图像编码：RGB uint8，JPEG quality=90，base64 编码。服务器侧解码：

```python
import base64, io, numpy as np
from PIL import Image

def decode_image(b64_str: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"), dtype=np.uint8)
```

**响应**：

```json
{
    "actions": [[[0.1, 0.2, ...], ...], ...],
    "latency_s": 0.15
}
```

- `actions` **(必需)**：形状 `(N, H, action_dim)`。即使 `H=1` 也必须保留中间维度。
- `latency_s`（可选）：服务器端推理耗时，用于日志。

### `POST /reset`（可选）

Episode 重置通知。服务器可据此清理 history buffer 等内部状态。
未实现时客户端静默忽略。

```json
{"env_ids": [0, 3]}
```

---

## 3. 动作空间（`--control`）

| Control Mode | action_dim | 说明 |
|-------------|-----------|------|
| `joint_pos` | 8 | 7 **绝对**关节位置 + 1 夹爪（与 `eval.py --backend local` 的 ACT 输出一致；eval 在 `env.step` 前按当前关节转为相对增量） |
| `ik_abs` | 7 | 6 绝对末端位姿(xyz+rpy/quat) + 1 夹爪 |
| `ik_rel` | 7 | 6 相对末端位姿增量 (dx,dy,dz + roll,pitch,yaw 欧拉角) + 1 夹爪 |

`joint_pos`：夹爪通道由 eval 按 `>0.5` 映射为张开/闭合命令（与 ACT 数据收集一致时多为 0/1）。`ik_*`：夹爪为 **-1.0 闭合，+1.0 张开**，动作范围 **[-1, 1]**。

`eval.py` 的 `--control` 参数会覆盖环境的 arm_action 配置，切换到对应的控制模式。
Policy server 返回的 `actions` 维度和语义必须与所选 `control_mode` 一致。

---

## 4. 参考实现（FastAPI）

```python
#!/usr/bin/env python3
"""Minimal policy server template."""

import base64, io, time
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_DIM = 8
ACTION_HORIZON = 16
MODEL = None


def load_model():
    global MODEL
    # MODEL = YourPolicy.load("checkpoint.pt").to(DEVICE).eval()
    print(f"[INFO] Model loaded on {DEVICE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


def decode_image(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8)


@app.get("/info")
def info():
    return {"action_dim": ACTION_DIM, "action_horizon": ACTION_HORIZON, "control_mode": "joint_pos"}


@app.post("/predict")
async def predict(request: dict):
    t0 = time.monotonic()
    N = request["num_envs"]
    all_actions = []

    for i in range(N):
        if request.get("type") == "state":
            state = torch.tensor(request["state"][i], device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a = MODEL(state)                              # → (1, D) or (1, H, D)
            if a.dim() == 2:
                a = a.unsqueeze(1).expand(-1, ACTION_HORIZON, -1)
            all_actions.append(a.squeeze(0).cpu().numpy())
        else:
            imgs = {c: decode_image(vs[i]) for c, vs in request.get("images", {}).items()}
            proprio = {k: np.array(v[i]) for k, v in request.get("proprioception", {}).items()}
            lang = request.get("task_description", "")
            # action_chunk = MODEL.predict(imgs, proprio, lang)
            action_chunk = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
            all_actions.append(action_chunk)

    return {"actions": np.stack(all_actions).tolist(), "latency_s": round(time.monotonic() - t0, 4)}


@app.post("/reset")
async def reset(request: dict):
    # 清理 request["env_ids"] 对应的 history buffer
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

启动：`pip install fastapi uvicorn && python policy_server.py`
