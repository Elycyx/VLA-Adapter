"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions.
Inherits from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained,
but exactly replicate the logic in `prismatic.models.vlms.prismatic.py`.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    DINO_V3_FEATURE_DIM,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    NUM_PRED_TOKENS,
    STOP_INDEX,
    NormalizationType,
    NUM_TOKENS
)
from .configuration_prismatic import OpenVLAConfig, PrismaticConfig



# Set up logger
logger = logging.getLogger(__name__)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper



# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor



def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma



# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class CausalTemporalPatchAttention(nn.Module):
    """Temporal fusion over same-location patches, then keep the current frame."""

    def __init__(self, vision_dim: int, temporal_dim: int = 256) -> None:
        super().__init__()
        temporal_dim = min(temporal_dim, vision_dim)
        attn_dim = temporal_dim
        if attn_dim >= 4:
            attn_dim -= attn_dim % 4
        attn_dim = max(1, attn_dim)
        num_heads = 4 if attn_dim >= 4 and attn_dim % 4 == 0 else 1
        temporal_dim = max(1, temporal_dim)
        self.fusion_type = "attention"
        self.use_current_query_attention = False

        self.attn_norm = nn.LayerNorm(vision_dim)
        self.down_proj = nn.Linear(vision_dim, attn_dim)
        self.temporal_attn = nn.MultiheadAttention(attn_dim, num_heads=num_heads, batch_first=True)
        self.up_proj = nn.Linear(attn_dim, vision_dim)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        self.delta_norm = nn.LayerNorm(vision_dim * 2)
        self.delta_fc1 = nn.Linear(vision_dim * 2, temporal_dim)
        self.delta_act = nn.GELU()
        self.delta_fc2 = nn.Linear(temporal_dim, vision_dim)
        self.delta_gate = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.delta_fc2.weight)
        nn.init.zeros_(self.delta_fc2.bias)

    def set_fusion_type(self, fusion_type: str) -> None:
        normalized = fusion_type.replace("-", "_").lower()
        if normalized in {"delta", "delta_mlp"}:
            normalized = "delta_mlp"
        elif normalized != "attention":
            raise ValueError(f"Unsupported temporal_fusion_type={fusion_type!r}; use 'attention' or 'delta_mlp'.")
        self.fusion_type = normalized

    def set_use_current_query_attention(self, use_current_query_attention: bool) -> None:
        self.use_current_query_attention = bool(use_current_query_attention)

    def forward(self, patches: torch.Tensor, return_delta: bool = False) -> torch.Tensor:
        """
        Args:
            patches: (B, V, T, P, D), view-major temporal patch features.

        Returns:
            (B, V * P, D), current-frame tokens enriched by past frames.
        """
        bsz, num_views, num_frames, num_patches, dim = patches.shape
        if num_frames == 1:
            current = patches[:, :, -1]
            if return_delta:
                return torch.zeros_like(current).reshape(bsz, num_views * num_patches, dim)
            return current.reshape(bsz, num_views * num_patches, dim)

        current = patches[:, :, -1]
        if self.fusion_type == "delta_mlp":
            previous = patches[:, :, -2]
            delta = current - previous
            x = torch.cat([current, delta], dim=-1)
            current_delta = self.delta_fc2(self.delta_act(self.delta_fc1(self.delta_norm(x))))
            current_delta = self.delta_gate.to(current_delta.dtype) * current_delta
            if return_delta:
                return current_delta.reshape(bsz, num_views * num_patches, dim)
            fused = current + current_delta
            return fused.reshape(bsz, num_views * num_patches, dim)

        x = patches.permute(0, 1, 3, 2, 4).reshape(bsz * num_views * num_patches, num_frames, dim)
        x = self.down_proj(self.attn_norm(x))
        x = x + self._temporal_position_encoding(num_frames, x.shape[-1], x.device, x.dtype)
        if self.use_current_query_attention:
            attended, _ = self.temporal_attn(x[:, -1:], x, x, need_weights=False)
            current_delta = self.up_proj(attended[:, 0])
            current_delta = current_delta.reshape(bsz, num_views, num_patches, dim)
            if return_delta:
                return current_delta.reshape(bsz, num_views * num_patches, dim)
            return (current + current_delta).reshape(bsz, num_views * num_patches, dim)

        causal_mask = torch.triu(
            torch.ones(num_frames, num_frames, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attended, _ = self.temporal_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        current_delta = self.up_proj(attended[:, -1])
        current_delta = current_delta.reshape(bsz, num_views, num_patches, dim)
        if return_delta:
            return current_delta.reshape(bsz, num_views * num_patches, dim)
        return (current + current_delta).reshape(bsz, num_views * num_patches, dim)

    @staticmethod
    def _temporal_position_encoding(
        num_frames: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(-(num_frames - 1), 1, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-np.log(10000.0) / max(dim, 1))
        )
        pe = torch.zeros(num_frames, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        pe[-1] = 0.0
        return pe.unsqueeze(0).to(dtype=dtype)


class PrismaticVisionBackbone(nn.Module):
    """
    Vision backbone for Prismatic models that handles image feature extraction.

    Supports both single backbone (e.g., SigLIP) and fused backbone (e.g., SigLIP + DINOv2) configurations.
    For fused backbones, features from both models are concatenated along the feature dimension.
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        Initialize the vision backbone.

        Args:
            use_fused_vision_backbone: Whether to use two backbones and fuse their features
            image_sizes: List of image sizes for each backbone
            timm_model_ids: List of TIMM model IDs to use for each backbone
            timm_override_act_layers: List of activation layer overrides for each backbone
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # Default value, can be overridden later
        self.num_temporal_frames = 1  # Default value preserves legacy single-frame behavior
        self.use_mid_layer_temporal_fusion = False

        # Validate number of (fused) vision backbones
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        # Create primary featurizer
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # Create secondary featurizer if using fused backbone
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim
        self.temporal_patch_attention = CausalTemporalPatchAttention(self.embed_dim)

        # Patch LayerScale modules for HF compatibility
        self._patch_layer_scales()


    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        Create a TIMM-based featurizer model with appropriate configurations.

        Args:
            model_id: The TIMM model ID to load
            img_size: Input image size for the model
            act_layer: Override for the activation layer type

        Returns:
            A configured featurizer model
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch the forward function to extract the second-to-last layer features
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer


    def _patch_layer_scales(self) -> None:
        """
        Patch all LayerScale modules to be compatible with HF's parameter naming.

        HF Transformers overwrites parameters with names containing 'gamma',
        so we need to rename and modify the forward method.
        """
        # Patch primary featurizer
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # Patch secondary featurizer if it exists
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)


    def get_num_patches(self) -> int:
        """
        Returns the number of vision patches output by the vision backbone.

        Returns:
            Number of patches per image
        """
        return self.featurizer.patch_embed.num_patches


    def get_num_images_in_input(self) -> int:
        """
        Returns the number of input images for the vision backbone.

        Returns:
            Number of images expected in the input
        """
        return self.num_images_in_input


    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        Sets the number of input images for the vision backbone.

        Args:
            num_images_in_input: Number of images to expect in the input
        """
        self.num_images_in_input = num_images_in_input


    def get_num_temporal_frames(self) -> int:
        return self.num_temporal_frames


    def set_num_temporal_frames(self, num_temporal_frames: int) -> None:
        if num_temporal_frames < 1:
            raise ValueError(f"num_temporal_frames must be >= 1, got {num_temporal_frames}.")
        self.num_temporal_frames = num_temporal_frames


    def get_temporal_fusion_type(self) -> str:
        return self.temporal_patch_attention.fusion_type


    def set_temporal_fusion_type(self, temporal_fusion_type: str) -> None:
        self.temporal_patch_attention.set_fusion_type(temporal_fusion_type)


    def get_use_current_query_attention(self) -> bool:
        return self.temporal_patch_attention.use_current_query_attention


    def set_use_current_query_attention(self, use_current_query_attention: bool) -> None:
        self.temporal_patch_attention.set_use_current_query_attention(use_current_query_attention)


    def get_use_mid_layer_temporal_fusion(self) -> bool:
        return self.use_mid_layer_temporal_fusion


    def set_use_mid_layer_temporal_fusion(self, use_mid_layer_temporal_fusion: bool) -> None:
        self.use_mid_layer_temporal_fusion = bool(use_mid_layer_temporal_fusion)


    @staticmethod
    def _temporal_mid_layer_index(featurizer: nn.Module) -> int:
        final_idx = max(0, len(featurizer.blocks) - 2)
        return max(0, final_idx // 2)


    def _forward_fused_image_middle_and_final(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_regular, img_fused = torch.split(img, [3, 3], dim=1)
        mid_idx = self._temporal_mid_layer_index(self.featurizer)
        final_idx = max(0, len(self.featurizer.blocks) - 2)
        fused_mid_idx = self._temporal_mid_layer_index(self.fused_featurizer)
        fused_final_idx = max(0, len(self.fused_featurizer.blocks) - 2)
        patches_mid, patches_final = self.featurizer.get_intermediate_layers(img_regular, n={mid_idx, final_idx})
        patches_fused_mid, patches_fused_final = self.fused_featurizer.get_intermediate_layers(
            img_fused, n={fused_mid_idx, fused_final_idx}
        )
        middle = torch.cat([patches_mid, patches_fused_mid], dim=2)
        final = torch.cat([patches_final, patches_fused_final], dim=2)
        return middle, final


    def _forward_fused_image(self, img: torch.Tensor) -> torch.Tensor:
        img_regular, img_fused = torch.split(img, [3, 3], dim=1)
        patches = self.featurizer(img_regular)
        patches_fused = self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone.

        If `self.use_fused_vision_backbone == True`, uses both SigLIP and DINOv2 transformers to extract visual features
        (otherwise uses SigLIP only). Allows multi-image inputs (but only for fused vision backbone).

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
        """
        if self.num_images_in_input == 1 and self.num_temporal_frames == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            return self._forward_fused_image(pixel_values)

        else:
            assert self.use_fused_vision_backbone, "Multi-image or multi-frame inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            expected_images = self.num_images_in_input * self.num_temporal_frames
            expected_channels = 6 * expected_images
            if pixel_values.shape[1] != expected_channels:
                raise ValueError(
                    f"Expected pixel_values with {expected_channels} channels "
                    f"({self.num_images_in_input} view(s) * {self.num_temporal_frames} frame(s) * 6), "
                    f"got {pixel_values.shape[1]}."
                )
            images = torch.split(pixel_values, [6] * expected_images, dim=1)

            # Process each image and collect patches
            all_patches = []
            all_middle_patches = [] if self.use_mid_layer_temporal_fusion and self.num_temporal_frames > 1 else None
            for img in images:
                if all_middle_patches is None:
                    all_patches.append(self._forward_fused_image(img))
                else:
                    middle_patches, final_patches = self._forward_fused_image_middle_and_final(img)
                    all_middle_patches.append(middle_patches)
                    all_patches.append(final_patches)

            if self.num_temporal_frames == 1:
                return torch.cat(all_patches, dim=1)

            patch_stack = torch.stack(all_patches, dim=1)
            bsz, _, num_patches, dim = patch_stack.shape
            patch_stack = patch_stack.reshape(
                bsz, self.num_images_in_input, self.num_temporal_frames, num_patches, dim
            )
            if all_middle_patches is not None:
                middle_patch_stack = torch.stack(all_middle_patches, dim=1).reshape(
                    bsz, self.num_images_in_input, self.num_temporal_frames, num_patches, dim
                )
                temporal_delta = self.temporal_patch_attention(middle_patch_stack, return_delta=True)
                current_final = patch_stack[:, :, -1].reshape(bsz, self.num_images_in_input * num_patches, dim)
                return current_final + temporal_delta
            return self.temporal_patch_attention(patch_stack)



# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features



# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None



class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa



class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )
        
        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size
        
        #Action query token
        self.action_queries = nn.Embedding(NUM_TOKENS, self.llm_dim)
        self.action_queries.weight.data.zero_()

        # Optional future-vision predictive tokens (gated by `use_future_pred`).
        #   * `pred_queries`: learnable token embeddings, one per future step.
        #   * `pred_head`: maps pred-token last-layer hidden state into frozen DINOv3 feature space.
        # Initialized as zeros / identity-like; the `use_future_pred` flag is False by default so
        # these modules contribute nothing unless explicitly enabled at training time.
        self.use_future_pred: bool = False
        self.pred_tokens_before_action: bool = False
        self.use_future_conf: bool = False
        self.future_confidence_gamma: float = 1.0
        self.pred_queries = nn.Embedding(NUM_PRED_TOKENS, self.llm_dim)
        self.pred_queries.weight.data.zero_()
        self.pred_head = nn.Linear(self.llm_dim, DINO_V3_FEATURE_DIM, bias=False)
        confidence_hidden_dim = max(128, self.llm_dim // 4)
        self.pred_confidence_head = nn.Sequential(
            nn.Linear(self.llm_dim, confidence_hidden_dim),
            nn.GELU(),
            nn.Linear(confidence_hidden_dim, 1),
            nn.Sigmoid(),
        )

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    def set_use_future_pred(self, flag: bool) -> None:
        """Toggle the future-vision prediction branch (affects forward + predict_action)."""
        self.use_future_pred = bool(flag)

    def set_pred_tokens_before_action(self, flag: bool) -> None:
        """Choose whether pred-token slots are inserted before or after action-query slots."""
        self.pred_tokens_before_action = bool(flag)

    def set_use_future_conf(self, flag: bool, gamma: float = 1.0) -> None:
        """Toggle the learned future-confidence head used for dynamic chunking."""
        self.use_future_conf = bool(flag)
        self.future_confidence_gamma = float(gamma)

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()
    def set_version(self, version: str):
        self.version = version
        return self.version


    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        Replace embeddings in input_embeddings at positions where all_actions_mask is True
        with embeddings from noisy_action_features, using vectorized operations.

        Args:
            input_embeddings: Tensor of shape (B, S, D)
            all_actions_mask: Boolean tensor of shape (B, S)
            noisy_action_features: Tensor of shape (B, K, D) where K is the number of True values in mask per sample

        Returns:
            Modified input_embeddings tensor
        """
        # Rank True positions per row, then gather replacement tokens without dynamic nonzero indices.
        ranks = all_actions_mask.to(torch.long).cumsum(dim=1) - 1
        ranks = ranks.clamp(min=0, max=noisy_action_features.shape[1] - 1)
        gather_index = ranks.unsqueeze(-1).expand(-1, -1, input_embeddings.shape[-1])
        repositioned_features = noisy_action_features.gather(1, gather_index)
        return torch.where(all_actions_mask.unsqueeze(-1), repositioned_features, input_embeddings)

    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """Process vision features with optional FiLM conditioning"""
        if use_film:
            # FiLM: Infuse language inputs into visual features
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        # Project patch embeddings into language embedding space
        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """Build multimodal embeddings and attention mask"""
        # Update attention mask
        
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """Build multimodal labels with IGNORE_INDEX for patch embeddings"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
        use_film: bool = False,
        pred_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

            # Get input embeddings (from language model embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

            
            # Extract action masks
            all_actions_mask = self._process_action_masks(labels)
            non_language_mask = all_actions_mask
            if self.use_future_pred and pred_mask is not None:
                non_language_mask = non_language_mask | pred_mask

            # Extract the language portion of the input embeddings (i.e. remove the action tokens portion)
            
            # print(input_embeddings[~all_actions_mask].size())
            language_embeddings = input_embeddings[~non_language_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )  # (B, lang_seq_len, llm_dim)

            # Get visual features
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

            # Process action embeddings
            if noisy_actions is not None:
                

                action_queries = self.action_queries.weight  # (1, h)
                action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
                all_actions_mask = self._process_action_masks(labels)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, action_queries)
                

            else:
                action_queries = self.action_queries.weight  # (1, h)
                action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
                all_actions_mask = self._process_action_masks(labels)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, action_queries)

            # Optionally inject learnable predictive tokens at positions marked by `pred_mask`.
            if self.use_future_pred and pred_mask is not None:
                pred_q = self.pred_queries.weight.view(1, NUM_PRED_TOKENS, -1).repeat(
                    input_embeddings.shape[0], 1, 1
                )  # (B, NUM_PRED_TOKENS, llm_dim)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, pred_mask, pred_q
                )

            # Build multimodal embeddings & attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
            
            # Build labels for multimodal sequence if needed
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

            # Dispatch to language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
            )


    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)



class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask, pad_token_id: Optional[int] = None):
        """Prepares input for action prediction by adding necessary tokens.

        When `self.use_future_pred=True`, also reserves NUM_PRED_TOKENS predictive-token slots.
        Their position is controlled by `self.pred_tokens_before_action`. Returns an additional
        `pred_mask` tensor (or None).
        """
        del pad_token_id  # unused; pred placeholders use STOP_INDEX (must be attended, not pad).
        bsz = input_ids.shape[0]
        prompt_len = input_ids.shape[-1]
        device, dtype = input_ids.device, input_ids.dtype

        placeholder_action_token_ids = torch.ones((bsz, NUM_TOKENS), device=device, dtype=dtype)
        pred_start = None

        if self.use_future_pred:
            # Predictive-token placeholders use STOP_INDEX rather than pad_token_id so they are NOT
            # zeroed-out in the attention mask. Their embeddings get replaced by `pred_queries` in
            # forward(); the actual id contents are irrelevant for downstream behavior.
            pred_placeholders = torch.full((bsz, NUM_PRED_TOKENS), STOP_INDEX, device=device, dtype=dtype)
            if self.pred_tokens_before_action:
                pred_start = prompt_len
                input_ids = torch.cat([input_ids, pred_placeholders, placeholder_action_token_ids], dim=-1)
            else:
                pred_start = prompt_len + NUM_TOKENS
                input_ids = torch.cat([input_ids, placeholder_action_token_ids, pred_placeholders], dim=-1)
        else:
            input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((bsz, 1), device=device, dtype=dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = torch.ones(
            (attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        # Build pred_mask spanning the predictive slots within `input_ids`
        # (no vision patches are inserted yet; consumers must offset accordingly when used post-vision).
        pred_mask = None
        if self.use_future_pred:
            pred_mask = torch.zeros((bsz, input_ids.shape[-1]), dtype=torch.bool, device=device)
            pred_mask[:, pred_start : pred_start + NUM_PRED_TOKENS] = True

        return input_ids, attention_mask, pred_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = torch.full(
            (labels.shape[0], input_ids.shape[-1] - labels.shape[-1]),
            ARBITRARY_ACTION_TOKEN_IDX,
            device=labels.device,
            dtype=labels.dtype,
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def _compute_pred_attention_confidence(
        self,
        language_model_output,
        normalized_actions,
        NUM_PATCHES: int,
        NUM_PROMPT_TOKENS: int,
        cumulative_min: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Training-free confidence from pred-token attention to action-token slots."""
        batch_size = normalized_actions.shape[0]
        if not (self.use_future_pred and language_model_output.attentions is not None):
            confidence = np.ones((batch_size, NUM_ACTIONS_CHUNK), dtype=np.float32)
            return {
                "pred_confidence": confidence,
                "raw_pred_confidence": confidence.copy(),
                "pred_to_action_attention": confidence.copy(),
                "action_attention_mass": confidence.copy(),
            }

        action_offset = NUM_PRED_TOKENS if self.pred_tokens_before_action else 0
        action_start = NUM_PATCHES + NUM_PROMPT_TOKENS + action_offset
        if self.pred_tokens_before_action:
            pred_start = NUM_PATCHES + NUM_PROMPT_TOKENS
        else:
            pred_start = action_start + NUM_TOKENS

        # Last-layer attention: (B, heads, S, S). Average heads, then measure how much
        # each pred query attends to action tokens for the corresponding future step.
        attn = language_model_output.attentions[-1].float().mean(dim=1)
        pred_to_actions = attn[
            :,
            pred_start : pred_start + NUM_PRED_TOKENS,
            action_start : action_start + NUM_TOKENS,
        ]
        by_action_step = pred_to_actions.reshape(
            batch_size, NUM_PRED_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM
        ).sum(dim=-1)
        same_step = by_action_step.diagonal(dim1=1, dim2=2)
        action_mass = by_action_step.sum(dim=-1).clamp_min(1e-8)
        raw_confidence = (same_step / action_mass).clamp(0.0, 1.0)
        confidence = raw_confidence
        if cumulative_min:
            confidence = torch.cummin(confidence, dim=1).values

        return {
            "pred_confidence": confidence.detach().cpu().numpy().astype(np.float32),
            "raw_pred_confidence": raw_confidence.detach().cpu().numpy().astype(np.float32),
            "pred_to_action_attention": raw_confidence.detach().cpu().numpy().astype(np.float32),
            "action_attention_mass": action_mass.detach().cpu().numpy().astype(np.float32),
        }

    def _compute_pred_learned_confidence(
        self,
        language_model_output,
        normalized_actions,
        NUM_PATCHES: int,
        NUM_PROMPT_TOKENS: int,
        cumulative_min: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Learned rollout reliability confidence from pred-token hidden states."""
        batch_size = normalized_actions.shape[0]
        if not (self.use_future_pred and self.use_future_conf):
            confidence = np.ones((batch_size, NUM_ACTIONS_CHUNK), dtype=np.float32)
            return {
                "pred_confidence": confidence,
                "raw_pred_confidence": confidence.copy(),
            }

        action_offset = NUM_PRED_TOKENS if self.pred_tokens_before_action else 0
        action_start = NUM_PATCHES + NUM_PROMPT_TOKENS + action_offset
        pred_start = NUM_PATCHES + NUM_PROMPT_TOKENS if self.pred_tokens_before_action else action_start + NUM_TOKENS

        last_hidden = language_model_output.hidden_states[-1]
        pred_h = last_hidden[:, pred_start : pred_start + NUM_PRED_TOKENS, :].to(torch.bfloat16)
        raw_confidence = self.pred_confidence_head(pred_h).squeeze(-1).float()
        confidence = raw_confidence
        if cumulative_min:
            confidence = torch.cummin(confidence, dim=1).values

        return {
            "pred_confidence": confidence.detach().cpu().numpy().astype(np.float32),
            "raw_pred_confidence": raw_confidence.detach().cpu().numpy().astype(np.float32),
        }


    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        proprio=None,
        proprio_projector=None,
        latency_steps=None,
        latency_projector=None,
        latency_steps_scale=1.0,
        pred_mask=None,
        return_pred_confidence: bool = False,
        pred_confidence_cumulative_min: bool = True,
        return_tensor: bool = False,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""

        action_queries = self.action_queries.weight  # (1, h)
        action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
        # Replace action token embeddings with noisy action embeddings
        input_embeddings = self._replace_input_embeddings(input_embeddings.clone(), all_actions_mask, action_queries)

        # Optionally inject predictive-token embeddings (no-op when disabled / mask is None)
        if self.use_future_pred and pred_mask is not None:
            pred_q = self.pred_queries.weight.view(1, NUM_PRED_TOKENS, -1).repeat(
                input_embeddings.shape[0], 1, 1
            )
            input_embeddings = self._replace_input_embeddings(input_embeddings, pred_mask, pred_q)

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        # Transformers' SDPA mask helper checks torch.all(attention_mask == 1),
        # which is not CUDA Graph capturable. The graph path is fixed-shape and
        # assumes no padded language tokens, so None is equivalent to an all-ones mask.
        lm_attention_mask = None if return_tensor else multimodal_attention_mask
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=lm_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=return_pred_confidence and not self.use_future_conf,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens. If pred tokens are before action tokens, action slots
        # are shifted by NUM_PRED_TOKENS within the language segment.
        action_offset = NUM_PRED_TOKENS if (self.use_future_pred and self.pred_tokens_before_action) else 0
        multi_layer_hidden_states = []

        for item in language_model_output.hidden_states[0:]:
            text_hidden_states = item
            batch_size = item.shape[0]
            start = NUM_PATCHES + NUM_PROMPT_TOKENS + action_offset
            actions_hidden_states = text_hidden_states[
                :, start : start + NUM_TOKENS, :,
            ].reshape(batch_size, 1, NUM_TOKENS, -1).to(torch.bfloat16)

            task_latten_states = item[:, :NUM_PATCHES].reshape(batch_size, 1, NUM_PATCHES , -1)
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2)
            multi_layer_hidden_states.append(all_hidden_states)

        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)
        

        # Handle different prediction methods
        if action_head is not None:
            # L1 regression prediction
            normalized_actions = action_head.predict_action(multi_layer_hidden_states,
                                                proprio=proprio,
                                                proprio_projector=proprio_projector,
                                                latency_steps=latency_steps,
                                                latency_projector=latency_projector,
                                                latency_steps_scale=latency_steps_scale)
            normalized_actions = normalized_actions.reshape(input_embeddings.shape[0], NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float()
            if not return_tensor:
                normalized_actions = normalized_actions.cpu().detach().numpy()
        else:
            if return_tensor:
                raise ValueError("return_tensor=True is only supported with a regression action_head.")
            # Discrete token-based prediction
            disc_start = NUM_PATCHES + NUM_PROMPT_TOKENS + action_offset
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    disc_start : disc_start + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(input_embeddings.shape[0], NUM_ACTIONS_CHUNK, ACTION_DIM)

        pred_info = {}
        if return_pred_confidence:
            if self.use_future_conf:
                pred_info = self._compute_pred_learned_confidence(
                    language_model_output=language_model_output,
                    normalized_actions=normalized_actions,
                    NUM_PATCHES=NUM_PATCHES,
                    NUM_PROMPT_TOKENS=NUM_PROMPT_TOKENS,
                    cumulative_min=pred_confidence_cumulative_min,
                )
            else:
                pred_info = self._compute_pred_attention_confidence(
                    language_model_output=language_model_output,
                    normalized_actions=normalized_actions,
                    NUM_PATCHES=NUM_PATCHES,
                    NUM_PROMPT_TOKENS=NUM_PROMPT_TOKENS,
                    cumulative_min=pred_confidence_cumulative_min,
                )

        return normalized_actions, actions_hidden_states, pred_info


    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        latency_steps=None,
        latency_projector=None,
        latency_steps_scale=1.0,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """

        pixel_values = kwargs["pixel_values"] # [1, 12, 224, 224]
        attention_mask = kwargs["attention_mask"] # 
        return_normalized_tensor = bool(kwargs.pop("return_normalized_tensor", False))
        if return_normalized_tensor and kwargs.get("return_pred_confidence", False):
            raise ValueError("CUDA graph tensor path does not support return_pred_confidence.")

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token
        return_pred_confidence = bool(kwargs.pop("return_pred_confidence", False))

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask, pred_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)
        # Force pred-token positions to IGNORE so the action-mask heuristic ignores them.
        if pred_mask is not None:
            labels = labels.clone()
            labels[pred_mask] = IGNORE_INDEX

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)
        non_language_mask = all_actions_mask
        if self.use_future_pred and pred_mask is not None:
            non_language_mask = non_language_mask | pred_mask

        # Language tokens are the original prompt plus the final stop token; action/pred
        # placeholders are inserted between them. Static slicing keeps CUDA Graph capture valid.
        language_embeddings = torch.cat(
            [input_embeddings[:, : NUM_PROMPT_TOKENS + 1, :], input_embeddings[:, -1:, :]],
            dim=1,
        )

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            if isinstance(proprio, torch.Tensor):
                proprio = proprio.to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            else:
                proprio = torch.as_tensor(
                    proprio,
                    device=projected_patch_embeddings.device,
                    dtype=projected_patch_embeddings.dtype,
                )

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()

        # Run regression or discrete token-based prediction
        normalized_actions, actions_hidden_states, pred_info = self._regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            NUM_PATCHES,
            NUM_PROMPT_TOKENS,
            action_head=action_head,
            proprio=proprio, # [8]
            proprio_projector=proprio_projector,
            latency_steps=latency_steps,
            latency_projector=latency_projector,
            latency_steps_scale=latency_steps_scale,
            pred_mask=pred_mask,
            return_pred_confidence=return_pred_confidence,
            pred_confidence_cumulative_min=kwargs.pop("pred_confidence_cumulative_min", True),
            return_tensor=return_normalized_tensor,
            )
           
        if return_normalized_tensor:
            return normalized_actions, actions_hidden_states

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        if return_pred_confidence:
            return actions, actions_hidden_states, pred_info
        return actions, actions_hidden_states



    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
