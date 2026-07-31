"""
Vivim: Video Vision Mamba for Medical Video Segmentation.
Combines 2D UNet feature extractor with Temporal Mamba Blocks (TMB) at encoder bottleneck.
Optionally includes a SphereHead for parametric eyeball circle estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_mamba import TemporalMambaBlock
from models.sphere_head import SphereHead


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
        use_sphere_head: bool = False,
        sphere_head_cfg: dict = None,
    ):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_sphere_head = use_sphere_head

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

        # Optional Sphere Head for parametric eyeball estimation
        if self.use_sphere_head and sphere_head_cfg is not None:
            self.sphere_head = SphereHead(
                feature_dim=base_channels * 8,
                hidden_dim=sphere_head_cfg.get('hidden_dim', 128) if isinstance(sphere_head_cfg, dict) else getattr(sphere_head_cfg, 'hidden_dim', 128),
                max_radius=sphere_head_cfg.get('max_radius', 200.0) if isinstance(sphere_head_cfg, dict) else getattr(sphere_head_cfg, 'max_radius', 200.0),
                min_radius=sphere_head_cfg.get('min_radius', 30.0) if isinstance(sphere_head_cfg, dict) else getattr(sphere_head_cfg, 'min_radius', 30.0),
                sharpness=sphere_head_cfg.get('sharpness', 20.0) if isinstance(sphere_head_cfg, dict) else getattr(sphere_head_cfg, 'sharpness', 20.0),
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, T, C, H, W] - Video frame sequence
        Returns:
            dict with:
                'seg_logits': [B, Num_Classes, H, W] - Segmentation logits
                'sphere_params': [B, 3] - (cx, cy, r) if sphere_head enabled
                'sphere_mask': [B, 1, H, W] - Rendered circle mask if sphere_head enabled
        """
        B, T, C, H, W = x.shape

        # Reshape to process frame-by-frame through 2D encoder: [B*T, C, H, W]
        x_flat = x.view(B * T, C, H, W)

        e1 = self.enc1(x_flat)              # [B*T, 32, H, W]
        p1 = self.pool1(e1)                 # [B*T, 32, H/2, W/2]

        e2 = self.enc2(p1)                  # [B*T, 64, H/2, W/2]
        p2 = self.pool2(e2)                 # [B*T, 64, H/4, W/4]

        e3 = self.enc3(p2)                  # [B*T, 128, H/4, W/4]
        p3 = self.pool3(e3)                 # [B*T, 128, H/8, W/8]

        b = self.bottleneck(p3)             # [B*T, 256, H/8, W/8]
        _, C_b, H_b, W_b = b.shape

        # Temporal Mamba selective scan at bottleneck
        if self.use_mamba:
            b_seq = b.view(B, T, C_b, H_b, W_b)
            b_mamba = self.temporal_mamba(b_seq)
            b_last = b_mamba[:, -1]          # Take the temporal-enhanced last frame bottleneck [B, 256, H/8, W/8]
        else:
            b_last = b.view(B, T, C_b, H_b, W_b)[:, -1]

        # Extract last frame skip connections
        e3_last = e3.view(B, T, -1, H // 4, W // 4)[:, -1]
        e2_last = e2.view(B, T, -1, H // 2, W // 2)[:, -1]
        e1_last = e1.view(B, T, -1, H, W)[:, -1]

        # Decoder pass
        d3 = self.up3(b_last)
        d3 = self.dec3(torch.cat([d3, e3_last], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2_last], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1_last], dim=1))

        seg_logits = self.final_cls(d1)  # [B, Num_Classes, H, W]

        result = {'seg_logits': seg_logits}

        # Sphere Head branch (from bottleneck features)
        if self.use_sphere_head and hasattr(self, 'sphere_head'):
            sphere_out = self.sphere_head(b_last, H=H, W=W)
            result['sphere_params'] = sphere_out['params']  # [B, 3]
            result['sphere_mask'] = sphere_out['mask']       # [B, 1, H, W]

        return result
