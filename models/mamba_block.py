"""
Selective State Space Model (SSM) Layer for Mamba.
Includes automatic Pure PyTorch Fallback if CUDA mamba_ssm is not available.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import selective_scan_cuda
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except (ImportError, RuntimeError, ModuleNotFoundError):
    HAS_MAMBA_CUDA = False


def selective_scan_pytorch(u, delta, A, B, C, D=None):
    """
    Pure PyTorch autograd implementation of Selective Scan SSM.
    Input:
        u: [B, D, L]
        delta: [B, D, L]
        A: [D, N]
        B: [B, N, L]
        C: [B, N, L]
        D: [D] (optional)
    Output:
        y: [B, D, L]
    """
    b, d, l = u.shape
    n = A.shape[1]

    # Discretize A: deltaA = exp(delta * A) -> [B, D, N, L]
    # delta: [B, D, 1, L], A: [1, D, N, 1]
    deltaA = torch.exp(delta.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1))

    # deltaB_u: [B, D, N, L]
    # delta: [B, D, 1, L], B: [B, 1, N, L], u: [B, D, 1, L]
    deltaB_u = delta.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)

    x = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    ys = []

    for i in range(l):
        x = deltaA[:, :, :, i] * x + deltaB_u[:, :, :, i]
        y_i = (x * C[:, :, i].unsqueeze(1)).sum(dim=-1)  # [B, D]
        ys.append(y_i)

    y = torch.stack(ys, dim=-1)  # [B, D, L]

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)

    return y


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize A parameter
        A = torch.repeat_interleave(torch.arange(1, self.d_state + 1, dtype=torch.float32), self.d_inner).view(self.d_inner, self.d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)  # [B, L, 2 * d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # [B, L, d_inner] each

        # Conv1d along sequence length
        x_conv = x_proj.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_act = F.silu(x_conv)  # [B, d_inner, L]

        # SSM Projection
        x_ssm_in = x_act.transpose(1, 2)  # [B, L, d_inner]
        ssm_params = self.x_proj(x_ssm_in)  # [B, L, 2*d_state + 1]

        dt, B_param, C_param = torch.split(ssm_params, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).transpose(1, 2).to(dtype=x_act.dtype)  # [B, d_inner, L]

        B_param = B_param.transpose(1, 2).to(dtype=x_act.dtype)  # [B, d_state, L]
        C_param = C_param.transpose(1, 2).to(dtype=x_act.dtype)  # [B, d_state, L]

        A = -torch.exp(self.A_log.to(dtype=x_act.dtype))  # [d_inner, d_state]

        if HAS_MAMBA_CUDA:
            y = selective_scan_fn(x_act, dt, A, B_param, C_param, self.D)
        else:
            y = selective_scan_pytorch(x_act, dt, A, B_param, C_param, self.D)

        y = y.transpose(1, 2)  # [B, L, d_inner]
        y = y * F.silu(z)      # Gated activation

        out = self.out_proj(y)  # [B, L, D]
        return out
