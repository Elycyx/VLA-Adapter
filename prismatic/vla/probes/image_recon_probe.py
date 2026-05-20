"""
Lightweight image decoder probes for visualizing predicted future features.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(32, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageReconstructionProbe(nn.Module):
    """Decode predicted future features into RGB frames for visualization only."""

    def __init__(
        self,
        input_dim: int = 1024,
        output_size: Tuple[int, int] = (224, 224),
        base_channels: int = 256,
        latent_size: Tuple[int, int] = (7, 7),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_size = tuple(output_size)
        self.base_channels = base_channels
        self.latent_size = tuple(latent_size)

        latent_dim = base_channels * self.latent_size[0] * self.latent_size[1]
        self.feature_projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, latent_dim),
            nn.GELU(),
        )

        channels = [
            base_channels,
            base_channels,
            base_channels // 2,
            base_channels // 4,
            base_channels // 8,
            base_channels // 16,
        ]
        self.decoder = nn.ModuleList(
            ConvBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        )
        self.output_head = nn.Conv2d(channels[-1], 3, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return RGB reconstructions in [0, 1].

        Args:
            features: Tensor with shape (batch, horizon, input_dim).
        """
        if features.ndim != 3:
            raise ValueError(f"Expected features with shape (B, T, D), got {tuple(features.shape)}.")

        batch_size, horizon, feature_dim = features.shape
        if feature_dim != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {feature_dim}.")

        x = features.reshape(batch_size * horizon, feature_dim).float()
        x = self.feature_projector(x)
        x = x.reshape(
            batch_size * horizon,
            self.base_channels,
            self.latent_size[0],
            self.latent_size[1],
        )

        for block in self.decoder:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)

        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        x = torch.sigmoid(self.output_head(x))
        return x.reshape(batch_size, horizon, 3, self.output_size[0], self.output_size[1])
