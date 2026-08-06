"""
Selective State Space Model (SSM) Layer wrapping the official Mamba-3 module.
Uses mamba_ssm.modules.mamba3.Mamba3 directly from the state-spaces/mamba package (v2.3.2+).

This replaces the previous manual selective_scan_fn / C++ kernel binding approach
with the official Mamba-3 (arXiv:2603.15569) implementation, which includes:
  - Exponential-Trapezoidal Discretization
  - Complex-Valued State Update (RoPE angles)
  - SISO / MIMO fused Triton kernels
"""

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba3 import Mamba3


class MambaLayer(nn.Module):
    """
    Drop-in replacement for the previous hand-rolled MambaLayer.
    Interface contract:  forward(x: [B, L, D]) -> [B, L, D]
    All internal SSM logic is delegated to the official Mamba3 module.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.mamba3 = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        return self.mamba3(x)
