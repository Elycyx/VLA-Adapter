"""
Train an image reconstruction probe from frozen VLA future-prediction features.

The probe is for visualization only: the VLA, pred queries, and pred head are
kept frozen, and only ImageReconstructionProbe parameters are optimized.
"""

import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from accelerate import PartialState
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader

import experiments.robot.openvla_utils as openvla_utils
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import DINO_V3_FEATURE_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.probes import ImageReconstructionProbe


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ReconProbeConfig:
    # Model/checkpoint
    pretrained_checkpoint: str = "openvla/openvla-7b"
    probe_resume_path: Optional[Path] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_film: bool = False
    use_minivlm: bool = False
    num_images_in_input: int = 1
    pred_tokens_before_action: bool = False
    use_future_conf: bool = False
    future_confidence_gamma: float = 1.0

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "aloha_scoop_x_into_bowl"
    shuffle_buffer_size: int = 100_000
    image_aug: bool = True
    use_relative_action: bool = False
    relative_action_mask: Optional[str] = None

    # Probe
    recon_target_size: Tuple[int, int] = (224, 224)
    probe_base_channels: int = 256
    probe_latent_size: Tuple[int, int] = (7, 7)
    recon_mse_weight: float = 0.25

    # Optimization
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_steps: int = 50_000
    grad_accumulation_steps: int = 1
    save_freq: int = 5_000
    save_latest_checkpoint_only: bool = False

    # Logging
    run_root_dir: Path = Path("runs")
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_entity: str = "cyx0307-shanghai-jiao-tong-university"
    wandb_project: str = "vla-adapter"
    use_wandb: bool = False
    wandb_log_freq: int = 10
    recon_log_freq: int = 500
    seed: int = 7


def get_run_id(cfg: ReconProbeConfig) -> str:
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    run_id = f"RECON-PROBE-{cfg.dataset_name}-{time.strftime('%Y_%m_%d-%H_%M_%S')}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def serialize_config(cfg: ReconProbeConfig) -> Dict[str, Any]:
    config = asdict(cfg)
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def wrap_ddp(module: nn.Module, device_id: int) -> DDP:
    return DDP(module, device_ids=[device_id], gradient_as_bucket_view=True)


def load_future_pred_components(vla: torch.nn.Module, cfg: ReconProbeConfig, device: torch.device) -> None:
    ckpt_path = openvla_utils.find_checkpoint_file(cfg.pretrained_checkpoint, "pred_components")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    vla.pred_queries.load_state_dict(state["pred_queries"])
    pred_head_state = state["pred_head"]
    if "weight" not in pred_head_state and "base_layer.weight" in pred_head_state:
        base_weight = pred_head_state["base_layer.weight"]
        lora_a = pred_head_state["lora_A.default.weight"]
        lora_b = pred_head_state["lora_B.default.weight"]
        pred_head_state = {"weight": base_weight + 2.0 * (lora_b @ lora_a)}
    vla.pred_head.load_state_dict(pred_head_state)

    vla.pred_queries.to(device, dtype=torch.bfloat16)
    vla.pred_head.to(device, dtype=torch.bfloat16)
    vla.set_use_future_pred(True)
    pred_before_action = bool(state.get("pred_tokens_before_action", cfg.pred_tokens_before_action))
    cfg.pred_tokens_before_action = pred_before_action
    vla.set_pred_tokens_before_action(pred_before_action)

    use_future_conf = bool(state.get("use_future_conf", cfg.use_future_conf))
    cfg.use_future_conf = use_future_conf
    cfg.future_confidence_gamma = float(state.get("future_confidence_gamma", cfg.future_confidence_gamma))
    if hasattr(vla, "set_use_future_conf"):
        vla.set_use_future_conf(cfg.use_future_conf, cfg.future_confidence_gamma)


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def maybe_enable_minivlm_tokenization(vla: torch.nn.Module, cfg: ReconProbeConfig) -> None:
    llm_backbone_id = str(getattr(vla.config, "llm_backbone_id", "")).lower()
    if not cfg.use_minivlm and ("qwen" in llm_backbone_id or "minivlm" in llm_backbone_id):
        cfg.use_minivlm = True
        print(
            f"Detected `{getattr(vla.config, 'llm_backbone_id', '')}`; "
            "enabling use_minivlm so action labels use raw action token ids.",
            flush=True,
        )


def extract_predicted_future_features(
    vla: torch.nn.Module,
    output,
    pred_mask: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    last_hidden = output.hidden_states[-1]
    batch_size, seq_len, _ = last_hidden.shape
    language_len = pred_mask.shape[1]
    patch_offset = seq_len - language_len

    batch_idx, lang_idx = torch.where(pred_mask)
    expected = batch_size * chunk
    if batch_idx.numel() != expected:
        raise ValueError(f"Expected {expected} pred tokens, got {batch_idx.numel()}.")

    mm_idx = torch.where(lang_idx == 0, torch.zeros_like(lang_idx), patch_offset + lang_idx)
    mm_pred_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=last_hidden.device)
    mm_pred_mask[batch_idx, mm_idx] = True
    pred_h = last_hidden[mm_pred_mask].reshape(batch_size, chunk, -1).to(torch.bfloat16)

    pred = vla.pred_head(pred_h).float()
    if pred.shape[-1] != DINO_V3_FEATURE_DIM:
        raise ValueError(f"Expected pred feature dim {DINO_V3_FEATURE_DIM}, got {pred.shape[-1]}.")
    return F.normalize(pred, dim=-1).detach()


def compute_recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    mse_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    valid = valid_mask.float()
    denom = valid.sum().clamp_min(1.0)

    l1_by_frame = (recon - target).abs().mean(dim=(2, 3, 4))
    mse_by_frame = ((recon - target) ** 2).mean(dim=(2, 3, 4))
    l1 = (l1_by_frame * valid).sum() / denom
    mse = (mse_by_frame * valid).sum() / denom
    loss = l1 + mse_weight * mse
    return loss, {
        "recon_loss": loss.item(),
        "recon_l1": l1.item(),
        "recon_mse": mse.item(),
        "valid_future_frames": valid.sum().item(),
    }


def save_probe_checkpoint(
    cfg: ReconProbeConfig,
    run_dir: Path,
    probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{step}_chkpt")
        suffix = f"{step}_checkpoint.pt"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "probe": (probe.module if hasattr(probe, "module") else probe).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": serialize_config(cfg),
        },
        checkpoint_dir / f"image_recon_probe--{suffix}",
    )


def load_probe_checkpoint_if_needed(
    probe: ImageReconstructionProbe,
    probe_resume_path: Optional[Path],
    device: torch.device,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    if probe_resume_path is None:
        return 0, None

    state = torch.load(probe_resume_path, map_location=device, weights_only=True)
    if "probe" in state:
        probe.load_state_dict(state["probe"])
        return int(state.get("step", 0)), state.get("optimizer")

    probe.load_state_dict(state)
    return 0, None


def save_reconstruction_grid(
    target: torch.Tensor,
    recon: torch.Tensor,
    valid_mask: torch.Tensor,
    output_path: Path,
) -> None:
    target = target[0].detach().cpu().clamp(0, 1)
    recon = recon[0].detach().cpu().clamp(0, 1)
    valid = valid_mask[0].detach().cpu().bool()
    frames = []
    for row in (target, recon):
        row_frames = []
        for idx in range(row.shape[0]):
            image = row[idx]
            if not valid[idx]:
                image = image * 0.25
            array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            row_frames.append(array)
        frames.append(np.concatenate(row_frames, axis=1))

    grid = np.concatenate(frames, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)


@draccus.wrap()
def train_recon_probe(cfg: ReconProbeConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    openvla_utils.DEVICE = device
    run_dir = cfg.run_root_dir / get_run_id(cfg)
    if distributed_state.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    if not openvla_utils.model_is_on_hf_hub(cfg.pretrained_checkpoint):
        if distributed_state.is_main_process:
            openvla_utils.update_auto_map(cfg.pretrained_checkpoint)
            openvla_utils.check_model_logic_mismatch(cfg.pretrained_checkpoint)
        barrier()
        openvla_utils.update_auto_map = lambda _: None
        openvla_utils.check_model_logic_mismatch = lambda _: None

    vla = openvla_utils.get_vla(cfg)
    processor = openvla_utils.get_processor(cfg)
    load_future_pred_components(vla, cfg, device)
    freeze_module(vla)
    maybe_enable_minivlm_tokenization(vla, cfg)

    target_size = tuple(cfg.recon_target_size)
    if hasattr(vla.config, "image_sizes") and vla.config.image_sizes is not None:
        target_size = tuple(vla.config.image_sizes)
        cfg.recon_target_size = target_size

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_minivlm=cfg.use_minivlm,
        use_future_pred=True,
        pred_tokens_before_action=cfg.pred_tokens_before_action,
        load_future_pred_features=False,
        load_future_recon_pixels=True,
        future_recon_size=target_size,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=target_size,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        use_relative_action=cfg.use_relative_action,
        relative_action_mask=cfg.relative_action_mask,
        use_future_pred=True,
    )
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    barrier()

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    probe = ImageReconstructionProbe(
        input_dim=DINO_V3_FEATURE_DIM,
        output_size=target_size,
        base_channels=cfg.probe_base_channels,
        latent_size=cfg.probe_latent_size,
    ).to(device)
    start_step, optimizer_state = load_probe_checkpoint_if_needed(probe, cfg.probe_resume_path, device)
    if torch.cuda.is_available() and getattr(distributed_state, "num_processes", 1) > 1:
        probe = wrap_ddp(probe, device_id)
    optimizer = AdamW(probe.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if distributed_state.is_main_process and cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_dir.name,
            config=serialize_config(cfg),
        )

    progress = tqdm.tqdm(
        total=cfg.max_steps,
        initial=start_step,
        leave=False,
        disable=not distributed_state.is_main_process,
    )
    optimizer.zero_grad(set_to_none=True)
    last_step = start_step
    for batch_idx, batch in enumerate(dataloader):
        step = start_step + batch_idx
        if step >= cfg.max_steps:
            break
        last_step = step

        pred_mask = batch["pred_mask"].to(device)
        target = batch["future_recon_pixels"].to(device, dtype=torch.float32)
        valid_mask = batch["future_pad_mask"].to(device)

        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with torch.no_grad(), autocast_context:
            output = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device, dtype=torch.bfloat16),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                use_film=cfg.use_film,
                pred_mask=pred_mask,
            )
            pred_features = extract_predicted_future_features(
                vla=vla,
                output=output,
                pred_mask=pred_mask,
                chunk=target.shape[1],
            )

        recon = probe(pred_features)
        loss, metrics = compute_recon_loss(
            recon=recon,
            target=target,
            valid_mask=valid_mask,
            mse_weight=cfg.recon_mse_weight,
        )
        (loss / cfg.grad_accumulation_steps).backward()

        if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress.update()

        if distributed_state.is_main_process and step % cfg.wandb_log_freq == 0:
            print(f"step={step} recon_loss={metrics['recon_loss']:.6f} recon_l1={metrics['recon_l1']:.6f}")
            if cfg.use_wandb:
                wandb.log({f"Recon Probe/{k}": v for k, v in metrics.items()}, step=step)

        if distributed_state.is_main_process and step > 0 and step % cfg.recon_log_freq == 0:
            save_reconstruction_grid(
                target=target,
                recon=recon,
                valid_mask=valid_mask,
                output_path=run_dir / "recon_grids" / f"step_{step:07d}.png",
            )

        if step > 0 and step % cfg.save_freq == 0:
            barrier()
            if distributed_state.is_main_process:
                save_probe_checkpoint(cfg, run_dir, probe, optimizer, step)
            barrier()

    barrier()
    if distributed_state.is_main_process:
        final_step = min(cfg.max_steps, last_step)
        save_probe_checkpoint(cfg, run_dir, probe, optimizer, final_step)
    if distributed_state.is_main_process and cfg.use_wandb:
        wandb.finish()
    barrier()


if __name__ == "__main__":
    train_recon_probe()
