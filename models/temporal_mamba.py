"""
Temporal Mamba Block (TMB) module for Vivim.
Flattens feature maps along spatial dimensions and applies Mamba selective scan
across sequence time steps to capture video dynamics in linear time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import MambaLayer


class TemporalMambaBlock(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = MambaLayer(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] - Sequence of feature maps across T frames
        Returns:
            out: [B, T, C, H, W] - Temporal-enhanced sequence features
        """
        B, T, C, H, W = x.shape

        # Reshape to scan across time: [B * H * W, T, C]
        x_perm = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        x_norm = self.norm(x_perm)

        mamba_out = self.mamba(x_norm)  # [B * H * W, T, C]
        out_flat = x_perm + self.proj(mamba_out)

        # Reshape back to [B, T, C, H, W]
        out = out_flat.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        return out
