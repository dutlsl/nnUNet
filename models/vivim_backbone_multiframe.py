"""
VivimBackboneMultiFrame: Phase 2 extension of VivimBackbone.
Decodes ALL T frames during training (multi-frame output),
but at inference time decodes only the last frame (identical to base).
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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class VivimBackboneMultiFrame(nn.Module):
    """
    Same architecture as VivimBackbone, but forward() can optionally decode
    all T frames instead of just the last one.

    - output_all_frames=False (inference): returns [B, C, H, W] — identical to base
    - output_all_frames=True (training):   returns [B, T, C, H, W]
    """
    def __init__(self, in_channels=1, num_classes=4, base_channels=32,
                 d_state=16, d_conv=4, expand=2, use_mamba=True,
                 mamba_version="mamba3", mamba_headdim=64,
                 temporal_window=int(os.environ.get("TEMPORAL_WINDOW", "8")),
                 use_fifo_cache=True, **kwargs):
        super().__init__()
        self.use_mamba = use_mamba
        self.temporal_window = temporal_window
        self.use_fifo_cache = use_fifo_cache
        self._fifo_cache = []

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        if self.use_mamba:
            self.temporal_mamba = TemporalMambaBlock(
                channels=base_channels * 8, d_state=d_state,
                d_conv=d_conv, expand=expand)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.final_cls = nn.Conv2d(base_channels, num_classes, 1)

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

    def _encode_single_frame(self, x_frame: torch.Tensor):
        """Encode single frame [B, C, H, W] through 2D CNN encoder."""
        e1 = self.enc1(x_frame)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        return e1, e2, e3, b

    def _decode_single_frame(self, b_feat, e3_feat, e2_feat, e1_feat):
        """Decode a single frame given bottleneck + skip features."""
        d3 = self.up3(b_feat)
        d3 = self.dec3(torch.cat([d3, e3_feat], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2_feat], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1_feat], dim=1))
        return self.final_cls(d1)

    def forward(self, x, output_all_frames=False):
        """
        Args:
            x: [B, T, C, H, W] or [B, C, H, W] (single frame)
            output_all_frames: if True, decode all T frames → [B, T, Cls, H, W]
        Returns:
            output_all_frames=False: [B, Cls, H, W]  (last frame only — inference mode)
            output_all_frames=True:  [B, T, Cls, H, W] (all frames — training mode)
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)

        B, T, C, H, W = x.shape

        # 1. Multi-frame training / decoding path
        if output_all_frames or self.training or not self.use_fifo_cache:
            x_flat = x.view(B * T, C, H, W)
            e1 = self.enc1(x_flat)
            p1 = self.pool1(e1)
            e2 = self.enc2(p1)
            p2 = self.pool2(e2)
            e3 = self.enc3(p2)
            p3 = self.pool3(e3)
            b = self.bottleneck(p3)
            _, C_b, H_b, W_b = b.shape

            if self.use_mamba:
                b_seq = b.view(B, T, C_b, H_b, W_b)
                b_mamba = self.temporal_mamba(b_seq)
            else:
                b_mamba = b.view(B, T, C_b, H_b, W_b)

            e3_all = e3.view(B, T, -1, H // 4, W // 4)
            e2_all = e2.view(B, T, -1, H // 2, W // 2)
            e1_all = e1.view(B, T, -1, H, W)

            if not output_all_frames:
                # Inference on multi-frame input: warm up cache
                needed = self.temporal_window - 1
                if needed > 0:
                    if T >= needed:
                        self._fifo_cache = [
                            b_mamba[:, i : i + 1].detach().clone() for i in range(T - needed, T)
                        ]
                    else:
                        self._fifo_cache = [
                            b_mamba[:, 0:1].detach().clone() for _ in range(needed - T)
                        ] + [b_mamba[:, i : i + 1].detach().clone() for i in range(T)]

                return self._decode_single_frame(
                    b_mamba[:, -1], e3_all[:, -1], e2_all[:, -1], e1_all[:, -1])

            # Training: decode all T frames with gradient checkpointing
            outputs = []
            for t in range(T):
                out_t = torch.utils.checkpoint.checkpoint(
                    self._decode_single_frame,
                    b_mamba[:, t], e3_all[:, t], e2_all[:, t], e1_all[:, t],
                    use_reentrant=False,
                )
                outputs.append(out_t)

            return torch.stack(outputs, dim=1)

        # 2. Streaming Inference (T == 1, output_all_frames=False): FIFO Cache
        if T == 1:
            x_frame = x[:, 0]
            e1, e2, e3, b_curr = self._encode_single_frame(x_frame)
            b_curr_seq = b_curr.unsqueeze(1)

            if not self._is_cache_valid(b_curr_seq):
                self._fifo_cache = [
                    b_curr_seq.detach().clone() for _ in range(self.temporal_window - 1)
                ]

            b_seq = torch.cat(self._fifo_cache + [b_curr_seq], dim=1)
            self._fifo_cache.pop(0)
            self._fifo_cache.append(b_curr_seq.detach())

            if self.use_mamba:
                b_mamba = self.temporal_mamba(b_seq)
                b_last = b_mamba[:, -1]
            else:
                b_last = b_curr

            return self._decode_single_frame(b_last, e3, e2, e1)
