"""Implementation of additional modules for the VLA's vision transformer."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class FiLMedVisionTransformerBlock(nn.Module):
    """
    Wrapper for ViT blocks that adds components to implement FiLM language conditioning.

    Modulates visual feature embeddings via
        x = (1 + gamma) * x + beta,
    where x is visual feature and gamma and beta are learned projections of the average language embedding.
    gamma and beta have D dimensions each, where D is the number of hidden dimensions in the ViT's features.

    NOTE #1 (Moo Jin):
    In convolutional neural architectures, the "feature" in FiLM is an entire feature map, i.e., each channel in a
    convolutional layer (so gamma and beta have C dimensions, where C is the number of channels). Therefore, FiLM's
    scaling and shifting is applied across all spatial locations for conv nets -- i.e., it is spatially agnostic.

    For vision transformer architectures, you may consider individual patch embeddings as individual "features" at first
    instinct, but this would make FiLM scaling and shifting spatially local. In order to make the modulation spatially
    global like in convolutional architectures, we should apply the scaling and shifting to each dimension of each patch
    embedding. I.e., gamma and beta should have D dimensions, where D is the number of dimensions in a visual embedding.

    NOTE #2 (Moo Jin):
    x = (1 + gamma) * x + beta is used in the original FiLM paper as opposed to x = gamma * x + beta (see section 7.2 in
    https://arxiv.org/pdf/1709.07871.pdf). Since gamma and beta are close to zero upon initialization, this leads to an
    identity transformation at the beginning of training, which minimizes perturbation to the pretrained representation.
    """

    def __init__(
        self,
        block,
        vision_dim: int,
        llm_dim: int,
    ):
        """
        Initializes FiLM ViT block wrapper.

        Args:
            block (timm.models.vision_transformer.Block): Vision transformer block.
            vision_dim (int): Number of hidden dimensions in visual embeddings.
            llm_dim (int): Number of hidden dimensions in language embeddings.
        """
        super().__init__()
        self.block = block
        # Initialize gamma and beta projectors
        self.scale = nn.Linear(llm_dim, vision_dim)
        self.shift = nn.Linear(llm_dim, vision_dim)

    def forward(self, x, average_language_embedding):
        """
        Overrides the vision transformer block forward pass to use FiLM.

        Args:
            x (torch.Tensor): Visual input embeddings, (batch_size, vision_seq_len, vision_dim).
            average_language_embedding (torch.Tensor): Average language embedding for task, (batch_size, llm_dim).
        """
        # Project average language embedding to visual embedding space to get gamma and beta
        gamma = self.scale(average_language_embedding)  # (batch_size, vision_dim)
        beta = self.shift(average_language_embedding)  # (batch_size, vision_dim)

        # Pass visual inputs through attention portion of original block
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))

        # Modulate intermediate visual representations via FiLM
        x = x * (1 + gamma.view(gamma.shape[0], 1, gamma.shape[1])) + beta.view(beta.shape[0], 1, beta.shape[1])

        # Pass visual inputs through feedforward portion of original block
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        return x


class NullVisionTransformerBlockWrapper(nn.Module):
    """
    Null wrapper for ViT blocks that doesn't do anything; just calls the original block's forward function.
    Useful if you want to use a block wrapper every X blocks instead of every block (e.g., to reduce the number of new
    parameters introduced by a new wrapper).
    """

    def __init__(
        self,
        block,
    ):
        super().__init__()
        self.block = block

    def forward(self, x, average_language_embedding):
        return self.block(x)


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """Utility function for monkey-patching functions."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


class FiLMedVisionTransformer(VisionTransformer):
    """
    Wrapper for timm.models.vision_transformer.VisionTransformer that overrides functions to enable infusing language
    embeddings into visual embeddings via FiLM.
    """

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        language_embeddings: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ):
        """
        Copy of timm.models.vision_transformer.VisionTransformer._intermediate_layers() with modifications
        to take in language embeddings as additional input.
        """
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, language_embeddings)  # Modified to receive language_embeddings
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        language_embeddings: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Copy of timm.models.vision_transformer.VisionTransformer.get_intermediate_layers() with modifications
        to allow language embeddings as additional input.
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, language_embeddings, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


class FiLMedPrismaticVisionBackbone(nn.Module):
    """
    Wrapper for OpenVLA's vision backbone that implements feature-wise linear modulation (FiLM).

    Wraps the Vision Transformers in the vision backbone to enable language conditioning through FiLM.
    Supports processing 1-3 images using dual vision backbones (SigLIP + DINOv2).
    """

    def __init__(
        self,
        vision_backbone,
        llm_dim: int = 4096,  # 4096 for Llama-2 7B
    ) -> None:
        """
        Initializes FiLM wrapper.

        Args:
            vision_backbone (PrismaticVisionBackbone): Base vision backbone.
            llm_dim (int): Dimension of language model embeddings.
        """
        super().__init__()
        self.vision_backbone = vision_backbone
        self.llm_dim = llm_dim

        # Wrap vision transformers
        self._wrap_vit(self.vision_backbone.featurizer)  # SigLIP
        if self.vision_backbone.use_fused_vision_backbone:
            self._wrap_vit(self.vision_backbone.fused_featurizer)  # DINOv2

    def _wrap_vit(self, vit) -> None:
        """
        Creates wrapper around an individual vision transformer to allow for infusion of language inputs.

        Args:
            vit (VisionTransformer): Original vision transformer.
        """
        # Wrap vision transformer blocks
        block_wrappers = []
        for block in vit.blocks:
            block_wrappers.append(
                FiLMedVisionTransformerBlock(block=block, vision_dim=vit.num_features, llm_dim=self.llm_dim)
            )
        vit.blocks = nn.Sequential(*block_wrappers)

        # Wrap vision transformer with new class that overrides functions used for forward pass
        vit.__class__ = FiLMedVisionTransformer
        vit.forward = unpack_tuple(partial(vit.get_intermediate_layers, n={len(vit.blocks) - 2}))

    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self) -> int:
        """Returns the number of input images for the vision backbone."""
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Sets the number of input images for the vision backbone."""
        self.vision_backbone.set_num_images_in_input(num_images_in_input)

    def get_num_temporal_frames(self) -> int:
        """Returns the number of temporal frames per camera view."""
        if hasattr(self.vision_backbone, "get_num_temporal_frames"):
            return self.vision_backbone.get_num_temporal_frames()
        return 1

    def set_num_temporal_frames(self, num_temporal_frames: int) -> None:
        """Sets the number of temporal frames per camera view."""
        if hasattr(self.vision_backbone, "set_num_temporal_frames"):
            self.vision_backbone.set_num_temporal_frames(num_temporal_frames)

    def get_temporal_fusion_type(self) -> str:
        """Returns the temporal fusion implementation."""
        if hasattr(self.vision_backbone, "get_temporal_fusion_type"):
            return self.vision_backbone.get_temporal_fusion_type()
        return "attention"

    def set_temporal_fusion_type(self, temporal_fusion_type: str) -> None:
        """Sets the temporal fusion implementation."""
        if hasattr(self.vision_backbone, "set_temporal_fusion_type"):
            self.vision_backbone.set_temporal_fusion_type(temporal_fusion_type)

    def get_use_current_query_attention(self) -> bool:
        """Returns whether attention fusion uses current-frame queries only."""
        if hasattr(self.vision_backbone, "get_use_current_query_attention"):
            return self.vision_backbone.get_use_current_query_attention()
        return False

    def set_use_current_query_attention(self, use_current_query_attention: bool) -> None:
        """Sets whether attention fusion uses current-frame queries only."""
        if hasattr(self.vision_backbone, "set_use_current_query_attention"):
            self.vision_backbone.set_use_current_query_attention(use_current_query_attention)

    def get_use_mid_layer_temporal_fusion(self) -> bool:
        """Returns whether temporal fusion uses middle-layer features."""
        if hasattr(self.vision_backbone, "get_use_mid_layer_temporal_fusion"):
            return self.vision_backbone.get_use_mid_layer_temporal_fusion()
        return False

    def set_use_mid_layer_temporal_fusion(self, use_mid_layer_temporal_fusion: bool) -> None:
        """Sets whether temporal fusion uses middle-layer features."""
        if hasattr(self.vision_backbone, "set_use_mid_layer_temporal_fusion"):
            self.vision_backbone.set_use_mid_layer_temporal_fusion(use_mid_layer_temporal_fusion)

    @staticmethod
    def _temporal_mid_layer_index(featurizer) -> int:
        final_idx = max(0, len(featurizer.blocks) - 2)
        return max(0, final_idx // 2)

    def _forward_fused_image_middle_and_final(
        self,
        img: torch.Tensor,
        average_language_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_regular, img_fused = torch.split(img, [3, 3], dim=1)
        mid_idx = self._temporal_mid_layer_index(self.vision_backbone.featurizer)
        final_idx = max(0, len(self.vision_backbone.featurizer.blocks) - 2)
        fused_mid_idx = self._temporal_mid_layer_index(self.vision_backbone.fused_featurizer)
        fused_final_idx = max(0, len(self.vision_backbone.fused_featurizer.blocks) - 2)
        patches_mid, patches_final = self.vision_backbone.featurizer.get_intermediate_layers(
            img_regular, average_language_embedding, n={mid_idx, final_idx}
        )
        patches_fused_mid, patches_fused_final = self.vision_backbone.fused_featurizer.get_intermediate_layers(
            img_fused, average_language_embedding, n={fused_mid_idx, fused_final_idx}
        )
        middle = torch.cat([patches_mid, patches_fused_mid], dim=2)
        final = torch.cat([patches_final, patches_fused_final], dim=2)
        return middle, final

    def _forward_fused_image(self, img: torch.Tensor, average_language_embedding: torch.Tensor) -> torch.Tensor:
        img_regular, img_fused = torch.split(img, [3, 3], dim=1)
        patches = self.vision_backbone.featurizer(img_regular, average_language_embedding)
        patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)
        return torch.cat([patches, patches_fused], dim=2)

    def forward(self, pixel_values: torch.Tensor, language_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone with FiLM to infuse language inputs into visual features.

        Identical to PrismaticVisionBackbone.forward() except that language embeddings are also used as input.

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
            language_embeddings (torch.Tensor): Language embeddings for the task description, (B, seq_len, llm_dim).
        """
        # For FiLM: Average the language embeddings of the task description
        average_language_embedding = language_embeddings.mean(dim=1)

        num_images = self.get_num_images_in_input()
        num_frames = self.get_num_temporal_frames()

        if num_images == 1 and num_frames == 1:
            if not self.vision_backbone.use_fused_vision_backbone:
                return self.vision_backbone.featurizer(pixel_values, average_language_embedding)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            return self._forward_fused_image(pixel_values, average_language_embedding)

        else:
            assert self.vision_backbone.use_fused_vision_backbone, "Multi-image or multi-frame inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            expected_images = num_images * num_frames
            expected_channels = 6 * expected_images
            if pixel_values.shape[1] != expected_channels:
                raise ValueError(
                    f"Expected pixel_values with {expected_channels} channels "
                    f"({num_images} view(s) * {num_frames} frame(s) * 6), got {pixel_values.shape[1]}."
                )
            images = torch.split(pixel_values, [6] * expected_images, dim=1)

            # Process each image and collect patches
            all_patches = []
            use_mid_layer_fusion = self.get_use_mid_layer_temporal_fusion() and num_frames > 1
            all_middle_patches = [] if use_mid_layer_fusion else None
            for img in images:
                if all_middle_patches is None:
                    all_patches.append(self._forward_fused_image(img, average_language_embedding))
                else:
                    middle_patches, final_patches = self._forward_fused_image_middle_and_final(
                        img, average_language_embedding
                    )
                    all_middle_patches.append(middle_patches)
                    all_patches.append(final_patches)

            if num_frames == 1:
                return torch.cat(all_patches, dim=1)

            patch_stack = torch.stack(all_patches, dim=1)
            bsz, _, num_patches, dim = patch_stack.shape
            patch_stack = patch_stack.reshape(bsz, num_images, num_frames, num_patches, dim)
            if all_middle_patches is not None:
                middle_patch_stack = torch.stack(all_middle_patches, dim=1).reshape(
                    bsz, num_images, num_frames, num_patches, dim
                )
                temporal_delta = self.vision_backbone.temporal_patch_attention(middle_patch_stack, return_delta=True)
                current_final = patch_stack[:, :, -1].reshape(bsz, num_images * num_patches, dim)
                return current_final + temporal_delta
            return self.vision_backbone.temporal_patch_attention(patch_stack)
