"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, DINO_V3_FEATURE_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK, NUM_PRED_TOKENS, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
from prismatic.vla.datasets.dinov3_features import load_feature
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights



@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False
    use_future_pred: bool = False
    pred_tokens_before_action: bool = False
    future_pred_feature_dir: Optional[Path] = None
    load_future_pred_features: bool = True
    load_future_recon_pixels: bool = False
    future_recon_size: Optional[Tuple[int, int]] = None
    num_temporal_frames: int = 1
    temporal_frame_interval: int = 1
    use_latency_conditioning: bool = False
    latency_steps_min: int = 1
    latency_steps_max: int = 5


    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        current_idx = max(0, (self.num_temporal_frames - 1) * self.temporal_frame_interval)
        dataset_name = rlds_batch["dataset_name"]
        latency_steps = self._sample_latency_steps()
        action_start_idx = current_idx + latency_steps
        action_end_idx = action_start_idx + NUM_ACTIONS_CHUNK
        actions = rlds_batch["action"][action_start_idx:action_end_idx]
        action_pad_mask = np.asarray(
            rlds_batch["pad_mask_future_actions"][action_start_idx:action_end_idx],
            dtype=np.bool_,
        )
        current_action = actions[0]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = actions[1:]

        if self.use_minivlm:
            self.prompt_builder_fn = QwenPromptBuilder
            prompt_builder = self.prompt_builder_fn("openvla")
            # Get action chunk string
            future_actions_string = self.action_tokenizer(future_actions,self.use_minivlm)
            current_action_string = self.action_tokenizer(current_action,self.use_minivlm)

            action_chunk_string = [current_action_string] + future_actions_string
            flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]
            action_chunk_len = len(flattened_action_chunk_string) 

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ''},
            ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

            if len(input_ids) >= 3:
                del input_ids[-3] 
                del input_ids[-2] 
                del input_ids[-1] 

            if NUM_TOKENS<len(flattened_action_chunk_string):
                action_token_ids = flattened_action_chunk_string[:NUM_TOKENS]
            else:
                remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
                extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
                action_token_ids = flattened_action_chunk_string + extended_array

            # Optionally add NUM_PRED_TOKENS predictive-token placeholders. We use STOP_INDEX
            # (not pad) because the collator builds `attention_mask = input_ids.ne(pad_token_id)`
            # and pred positions must be attended.
            prompt_len = len(input_ids)
            if self.use_future_pred:
                pred_placeholders = [STOP_INDEX] * NUM_PRED_TOKENS
                if self.pred_tokens_before_action:
                    input_ids = input_ids + pred_placeholders + action_token_ids
                    pred_start = prompt_len
                    action_start = prompt_len + NUM_PRED_TOKENS
                else:
                    input_ids = input_ids + action_token_ids + pred_placeholders
                    action_start = prompt_len
                    pred_start = prompt_len + NUM_TOKENS
            else:
                input_ids = input_ids + action_token_ids
                action_start = prompt_len

            labels = list(input_ids)
            action_chunk_len = NUM_TOKENS

        else:
            future_actions_string = ''.join(self.action_tokenizer(future_actions, use_minivlm=False))

            # Get action chunk string
            current_action_string = self.action_tokenizer(current_action, use_minivlm=False)
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string[0]},
            ]
            # remove action token
            # conversation = [
            #     {"from": "human", "value": f"What action should the robot take to {lang}?"},
            #     {"from": "gpt", "value": ""},
            # ]
            action_chunk_len = 1


            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            # Tokenize (w/ `base_tokenizer`)
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            prompt_len = max(0, len(input_ids) - action_chunk_len)
            action_start = prompt_len
            if self.use_future_pred:
                pred_placeholders = [STOP_INDEX] * NUM_PRED_TOKENS
                if self.pred_tokens_before_action:
                    input_ids = input_ids[:prompt_len] + pred_placeholders + input_ids[prompt_len:]
                    pred_start = prompt_len
                    action_start = prompt_len + NUM_PRED_TOKENS
                else:
                    input_ids = input_ids + pred_placeholders
                    pred_start = len(input_ids) - NUM_PRED_TOKENS
            labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self._temporal_images_to_pixels(rlds_batch["observation"]["image_primary"])

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)
        if self.use_latency_conditioning:
            return_dict["latency_steps"] = np.asarray(latency_steps, dtype=np.float32)
            return_dict["action_pad_mask"] = torch.from_numpy(action_pad_mask.copy())

        # Future-vision prediction extras
        if self.use_future_pred:
            if self.load_future_pred_features and self.future_pred_feature_dir is None:
                raise ValueError(
                    "use_future_pred=True requires future_pred_feature_dir. "
                    "Run vla-scripts/precompute_dinov3_features.py and pass its output directory."
                )
            # Explicitly mark only the action span as supervised and keep all pred positions ignored.
            labels[:] = IGNORE_INDEX
            labels[action_start : action_start + action_chunk_len] = input_ids[
                action_start : action_start + action_chunk_len
            ]
            seq_len = input_ids.shape[0]
            pred_mask = torch.zeros(seq_len, dtype=torch.bool)
            pred_mask[pred_start : pred_start + NUM_PRED_TOKENS] = True
            # Force pred-position labels to IGNORE_INDEX (in case the trailing slice above missed any).
            labels[pred_start : pred_start + NUM_PRED_TOKENS] = IGNORE_INDEX

            # Future prediction remains anchored at the observation time t; latency only shifts action targets.
            future_imgs = rlds_batch["image_primary_future"]  # (chunk, H, W, 3) uint8
            future_pad_mask = torch.from_numpy(
                np.asarray(rlds_batch["pad_mask_future_obs"], dtype=np.bool_).copy()
            )
            return_dict["pred_mask"] = pred_mask
            return_dict["future_pad_mask"] = future_pad_mask
            if self.load_future_recon_pixels:
                return_dict["future_recon_pixels"] = self._future_images_to_rgb_tensor(future_imgs)
            if self.load_future_pred_features:
                future_features = np.stack(
                    [
                        load_feature(self.future_pred_feature_dir, future_imgs[i], dataset_name=dataset_name)
                        for i in range(future_imgs.shape[0])
                    ],
                    axis=0,
                )
                if future_features.shape[-1] != DINO_V3_FEATURE_DIM:
                    raise ValueError(
                        f"Expected cached DINOv3 features with dim {DINO_V3_FEATURE_DIM}, "
                        f"got {future_features.shape[-1]} from {self.future_pred_feature_dir}."
                    )
                return_dict["future_pred_features"] = torch.from_numpy(future_features).to(torch.float32)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    pixel_values_wrist = self._temporal_images_to_pixels(rlds_batch["observation"][k])
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"][current_idx]
            return_dict["proprio"] = proprio

        return return_dict

    def _sample_latency_steps(self) -> int:
        """Sample synthetic inference delay in control steps."""
        if not self.use_latency_conditioning:
            return 0
        latency_min = int(self.latency_steps_min)
        latency_max = int(self.latency_steps_max)
        if latency_min < 0 or latency_max < latency_min:
            raise ValueError(
                f"Invalid latency range [{latency_min}, {latency_max}]. "
                "Expected 0 <= latency_steps_min <= latency_steps_max."
            )
        return random.randint(latency_min, latency_max)

    def _temporal_images_to_pixels(self, images: np.ndarray) -> torch.Tensor:
        """Transform a temporal window of RGB images into view-local channel stacks."""
        frames = []
        start_idx = max(0, images.shape[0] - 1 - (self.num_temporal_frames - 1) * self.temporal_frame_interval)
        frame_indices = [start_idx + i * self.temporal_frame_interval for i in range(self.num_temporal_frames)]
        for frame_idx in frame_indices:
            img = Image.fromarray(images[frame_idx])
            frames.append(self.image_transform(img))
        return torch.cat(frames, dim=0)

    def _future_images_to_rgb_tensor(self, future_imgs: np.ndarray) -> torch.Tensor:
        """Convert future uint8 RGB frames to (chunk, 3, H, W) float tensors in [0, 1]."""
        frames = []
        for i in range(future_imgs.shape[0]):
            frame = Image.fromarray(future_imgs[i])
            if self.future_recon_size is not None:
                height, width = self.future_recon_size
                frame = frame.resize((width, height), resample=Image.BILINEAR)
            array = np.asarray(frame, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(frames, dim=0)
    
    

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        use_relative_action: bool = False,
        relative_action_mask: Optional[Tuple[bool, ...]] = None,
        use_future_pred: bool = False,
        num_temporal_frames: int = 1,
        temporal_frame_interval: int = 1,
        use_latency_conditioning: bool = False,
        latency_steps_min: int = 1,
        latency_steps_max: int = 5,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        if num_temporal_frames < 1:
            raise ValueError(f"num_temporal_frames must be >= 1, got {num_temporal_frames}.")
        if temporal_frame_interval < 1:
            raise ValueError(f"temporal_frame_interval must be >= 1, got {temporal_frame_interval}.")
        temporal_window_size = (num_temporal_frames - 1) * temporal_frame_interval + 1
        latency_steps_max = int(latency_steps_max) if use_latency_conditioning else 0
        latency_steps_min = int(latency_steps_min) if use_latency_conditioning else 0
        if latency_steps_min < 0 or latency_steps_max < latency_steps_min:
            raise ValueError(
                f"Invalid latency range [{latency_steps_min}, {latency_steps_max}]. "
                "Expected 0 <= latency_steps_min <= latency_steps_max."
            )
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.batch_transform.num_temporal_frames = num_temporal_frames
        self.batch_transform.temporal_frame_interval = temporal_frame_interval
        self.batch_transform.use_latency_conditioning = use_latency_conditioning
        self.batch_transform.latency_steps_min = latency_steps_min
        self.batch_transform.latency_steps_max = latency_steps_max

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        for dataset_kwargs in per_dataset_kwargs:
            dataset_kwargs["use_relative_action"] = use_relative_action
            dataset_kwargs["relative_action_mask"] = relative_action_mask

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=temporal_window_size,                   # Full history window before interval sampling
                future_action_window_size=NUM_ACTIONS_CHUNK - 1 + latency_steps_max,  # For delayed action chunking
                future_obs_window_size=NUM_ACTIONS_CHUNK if use_future_pred else 0,
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
                use_relative_action=use_relative_action,
                relative_action_mask=relative_action_mask,
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
