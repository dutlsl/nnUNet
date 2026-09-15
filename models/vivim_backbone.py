"""
Vivim: Video Vision Mamba for Medical Video Segmentation.
Combines 2D UNet feature extractor with Temporal Mamba Blocks (TMB) at encoder bottleneck.
Includes high-throughput Bottleneck Feature FIFO Caching for real-time 120Hz+ streaming inference.
"""

import os
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = None
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = None
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_mamba import TemporalMambaBlock


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VivimBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba: bool = True,
        mamba_version: str = "mamba3",
        mamba_headdim: int = 64,
        temporal_window: int = int(os.environ.get("TEMPORAL_WINDOW", "8")),
        use_fifo_cache: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.use_mamba = use_mamba
        self.temporal_window = temporal_window
        self.use_fifo_cache = use_fifo_cache
        self._fifo_cache = []

        # Encoder stages
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Temporal Mamba Bottleneck
        if self.use_mamba:
            self.temporal_mamba = TemporalMambaBlock(
                channels=base_channels * 8,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.final_cls = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    # -------------------------------------------------------------------------
    # FIFO Inference Cache Management
    # -------------------------------------------------------------------------
    def reset_cache(self):
        """Clear the FIFO bottleneck feature cache."""
        self._fifo_cache = []

    def enable_fifo_cache(self, enabled: bool = True, temporal_window: int = None):
        """Enable or disable FIFO bottleneck feature caching during inference."""
        self.use_fifo_cache = enabled
        if temporal_window is not None:
            self.temporal_window = temporal_window
        self.reset_cache()

    def disable_fifo_cache(self):
        """Disable FIFO caching (always recomputes all frames)."""
        self.enable_fifo_cache(enabled=False)

    def set_temporal_window(self, window_size: int):
        """Update temporal window size for FIFO cache."""
        self.temporal_window = window_size
        self.reset_cache()

    def _is_cache_valid(self, b_curr_seq: torch.Tensor) -> bool:
        """
        Check if cached bottleneck features match the incoming frame:
        device, dtype, batch size, and spatial resolution.
        """
        if not self._fifo_cache:
            return False
        if len(self._fifo_cache) != self.temporal_window - 1:
            return False
        cached = self._fifo_cache[0]
        return (
            cached.device == b_curr_seq.device
            and cached.dtype == b_curr_seq.dtype
            and cached.shape[0] == b_curr_seq.shape[0]
            and cached.shape[2:] == b_curr_seq.shape[2:]
        )

    # -------------------------------------------------------------------------
    # Forward Pass Subroutines
    # -------------------------------------------------------------------------
    def _encode_single_frame(self, x_frame: torch.Tensor):
        """
        Encode a single frame [B, C, H, W] through 2D CNN encoder stages.
        Returns:
            e1: [B, base, H, W]
            e2: [B, base*2, H/2, W/2]
            e3: [B, base*4, H/4, W/4]
            b:  [B, base*8, H/8, W/8]
        """
        e1 = self.enc1(x_frame)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        return e1, e2, e3, b

    def _decode_single_frame(
        self,
        b_feat: torch.Tensor,
        e3_feat: torch.Tensor,
        e2_feat: torch.Tensor,
        e1_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Decode single frame from bottleneck and skip features."""
        d3 = self.up3(b_feat)
        d3 = self.dec3(torch.cat([d3, e3_feat], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2_feat], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1_feat], dim=1))

        out = self.final_cls(d1)
        return out

    def _forward_full(self, x: torch.Tensor, return_b_seq: bool = False):
        """Standard full forward pass across all T frames (used in training and non-cached inference)."""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        e1 = self.enc1(x_flat)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        _, C_b, H_b, W_b = b.shape

        b_seq = b.view(B, T, C_b, H_b, W_b)
        if self.use_mamba:
            b_mamba = self.temporal_mamba(b_seq)
            b_last = b_mamba[:, -1]
        else:
            b_last = b_seq[:, -1]

        e3_last = e3.view(B, T, -1, H // 4, W // 4)[:, -1]
        e2_last = e2.view(B, T, -1, H // 2, W // 2)[:, -1]
        e1_last = e1.view(B, T, -1, H, W)[:, -1]

        out = self._decode_single_frame(b_last, e3_last, e2_last, e1_last)
        if return_b_seq:
            return out, b_seq
        return out

    # -------------------------------------------------------------------------
    # Main Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] - Video frame sequence OR
               [B, C, H, W]    - Single frame streaming (transparently handled)
        Returns:
            out: [B, Num_Classes, H, W] - Output prediction mask/logits for the last frame
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]

        B, T, C, H, W = x.shape

        # 1. Training mode: ALWAYS bypass FIFO cache to guarantee clean autograd graph and zero stale features
        if self.training or not self.use_fifo_cache:
            return self._forward_full(x)

        # 2. Streaming Inference (T == 1): High-throughput FIFO cache accelerated path
        if T == 1:
            x_frame = x[:, 0]  # [B, C, H, W]
            e1, e2, e3, b_curr = self._encode_single_frame(x_frame)
            b_curr_seq = b_curr.unsqueeze(1)  # [B, 1, C_b, H_b, W_b]

            # Cache validation & cold-start handling (boundary padding replication)
            if not self._is_cache_valid(b_curr_seq):
                self._fifo_cache = [
                    b_curr_seq.detach().clone() for _ in range(self.temporal_window - 1)
                ]

            # Assemble full T-frame temporal bottleneck sequence: [B, T, C_b, H_b, W_b]
            b_seq = torch.cat(self._fifo_cache + [b_curr_seq], dim=1)

            # Update FIFO queue (evict oldest past frame, enqueue current frame)
            self._fifo_cache.pop(0)
            self._fifo_cache.append(b_curr_seq.detach())

            # Temporal Mamba sequence scan
            if self.use_mamba:
                b_mamba = self.temporal_mamba(b_seq)
                b_last = b_mamba[:, -1]
            else:
                b_last = b_curr

            # Decode current frame using its own skip connections
            out = self._decode_single_frame(b_last, e3, e2, e1)
            return out

        # 3. Batch Multi-frame Inference (T > 1): full pass + warm up FIFO cache
        out, b_seq = self._forward_full(x, return_b_seq=True)
        needed = self.temporal_window - 1
        if needed > 0:
            if T >= needed:
                self._fifo_cache = [
                    b_seq[:, i : i + 1].detach().clone() for i in range(T - needed, T)
                ]
            else:
                self._fifo_cache = [
                    b_seq[:, 0:1].detach().clone() for _ in range(needed - T)
                ] + [b_seq[:, i : i + 1].detach().clone() for i in range(T)]
        return out
