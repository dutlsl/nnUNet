"""
Selective State Space Model (SSM) Layer wrapping the official Mamba-3 module.
Uses mamba_ssm.modules.mamba3.Mamba3 directly from the official state-spaces/mamba package.

Supports PyTorch float8 compatibility for Cutlass/Quack API.
"""

import torch
import torch.nn as nn

# Compatibility polyfill for torch float8 types required by Cutlass/Quack
for dt in ['float8_e8m0fnu', 'float4_e2m1fn_x2', 'float8_e4m3fnuz', 'float8_e5m2fnuz']:
    if not hasattr(torch, dt):
        setattr(torch, dt, getattr(torch, 'float8_e5m2', torch.float32))

try:
    from mamba_ssm.modules.mamba3 import Mamba3
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False


class MambaLayer(nn.Module):
    """
    MambaLayer delegating sequence scanning to the official Mamba3 module.
    Interface contract: forward(x: [B, L, D]) -> [B, L, D]
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
        if HAS_MAMBA3:
            self.mamba3 = Mamba3(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                chunk_size=chunk_size,
            )
        else:
            raise ImportError("mamba_ssm.modules.mamba3.Mamba3 could not be imported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        return self.mamba3(x)
