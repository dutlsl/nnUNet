"""
Vivim: Video Vision Mamba for Medical Video Segmentation.
Combines 2D UNet feature extractor with Temporal Mamba Blocks (TMB) at encoder bottleneck.

SAGD branch: SphereHead/WeakMed removed. Added SAGD hook points for
Structure-Aware Serialization (SAS), Hierarchical Domain Modeling (HDM),
and Spectral Graph Alignment (SGA).
"""

import torch
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
        use_sadg: bool = False,
        sadg_cfg=None,
        # Legacy parameters kept for config compatibility but ignored on SAGD branch
        use_sphere_head: bool = False,
        sphere_head_cfg: dict = None,
        use_eyeball_head: bool = False,
    ):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_sadg = use_sadg

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

        # SAGD modules (initialized externally by trainer for config access)
        # These are set by nnUNetTrainer_Vivim_SADG after construction
        self.sas = None  # StructureAwareSerializer
        self.hdm = None  # HDM2D
        self.sga = None  # SpectralGraphAlignment

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.final_cls = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _encode(self, x_flat: torch.Tensor, H: int, W: int):
        """
        Shared encoder path. Returns bottleneck and skip connections.

        Args:
            x_flat: [B*T, C, H, W]
            H, W: original spatial dims

        Returns:
            b: [B*T, 256, H/8, W/8]
            e1, e2, e3: encoder skip connection features
        """
        e1 = self.enc1(x_flat)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)
        return b, e1, e2, e3

    def _decode(self, b_last, e3_last, e2_last, e1_last):
        """
        Shared decoder path. Returns segmentation logits.

        Args:
            b_last: [B, 256, H/8, W/8]
            e3_last, e2_last, e1_last: skip connections for last frame
        """
        d3 = self.up3(b_last)
        d3 = self.dec3(torch.cat([d3, e3_last], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2_last], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1_last], dim=1))

        seg_logits = self.final_cls(d1)
        return seg_logits

    def forward(self, x: torch.Tensor, labels=None):
        """
        Args:
            x: [B, T, C, H, W] - Video frame sequence
            labels: [B, H, W] optional, for SGA prototype update during training

        Returns:
            If use_sadg and modules are attached:
                dict with 'seg_logits', 'bottleneck_features', 'serialized_tokens'
            Else:
                seg_logits tensor [B, num_classes, H, W]
        """
        B, T, C, H, W = x.shape

        # Reshape to process frame-by-frame through 2D encoder: [B*T, C, H, W]
        x_flat = x.view(B * T, C, H, W)

        b, e1, e2, e3 = self._encode(x_flat, H, W)
        _, C_b, H_b, W_b = b.shape

        # Temporal Mamba selective scan at bottleneck
        if self.use_mamba:
            b_seq = b.view(B, T, C_b, H_b, W_b)
            b_mamba = self.temporal_mamba(b_seq)
            b_last = b_mamba[:, -1]  # [B, 256, H/8, W/8]
        else:
            b_last = b.view(B, T, C_b, H_b, W_b)[:, -1]

        # --- SAGD Processing ---
        serialized_tokens = None
        if self.use_sadg and self.sas is not None:
            # SAS: Structure-Aware Serialization
            fwd_cds, rev_cds, fwd_gcs, rev_gcs = self.sas(b_last)

            # Average all 4 serialization views for the primary token stream
            serialized_tokens = (fwd_cds + rev_cds + fwd_gcs + rev_gcs) / 4.0  # [B, N, C]

            # SGA: Spectral Graph Alignment (train=EMA update, test=alignment)
            if self.sga is not None:
                serialized_tokens = self.sga(serialized_tokens, labels)

            # Reshape back to spatial feature map for decoder
            b_last = serialized_tokens.permute(0, 2, 1).reshape(B, C_b, H_b, W_b)

        # Extract last frame skip connections
        e3_last = e3.view(B, T, -1, H // 4, W // 4)[:, -1]
        e2_last = e2.view(B, T, -1, H // 2, W // 2)[:, -1]
        e1_last = e1.view(B, T, -1, H, W)[:, -1]

        # Decoder
        seg_logits = self._decode(b_last, e3_last, e2_last, e1_last)

        if self.use_sadg:
            return {
                'seg_logits': seg_logits,
                'seg': seg_logits,
                'bottleneck_features': b_last,
                'serialized_tokens': serialized_tokens,
            }

        return seg_logits

    def forward_domain(self, x: torch.Tensor, domain_id: int = 0):
        """
        Forward pass for a specific domain (used during multi-domain training).
        Returns bottleneck features for HDM processing.

        Args:
            x: [B, T, C, H, W]
            domain_id: domain index

        Returns:
            dict with 'seg_logits', 'bottleneck', 'serialized_tokens', 'skips'
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        b, e1, e2, e3 = self._encode(x_flat, H, W)
        _, C_b, H_b, W_b = b.shape

        if self.use_mamba:
            b_seq = b.view(B, T, C_b, H_b, W_b)
            b_mamba = self.temporal_mamba(b_seq)
            b_last = b_mamba[:, -1]
        else:
            b_last = b.view(B, T, C_b, H_b, W_b)[:, -1]

        # SAS serialization
        serialized_tokens = None
        if self.use_sadg and self.sas is not None:
            fwd_cds, rev_cds, fwd_gcs, rev_gcs = self.sas(b_last)
            serialized_tokens = (fwd_cds + rev_cds + fwd_gcs + rev_gcs) / 4.0

        # Skips for later decoding
        e3_last = e3.view(B, T, -1, H // 4, W // 4)[:, -1]
        e2_last = e2.view(B, T, -1, H // 2, W // 2)[:, -1]
        e1_last = e1.view(B, T, -1, H, W)[:, -1]

        return {
            'bottleneck': b_last,
            'serialized_tokens': serialized_tokens,
            'skips': (e3_last, e2_last, e1_last),
            'spatial_shape': (H_b, W_b),
        }

    def decode_from_tokens(self, tokens: torch.Tensor, skips, spatial_shape):
        """
        Decode from HDM-processed tokens back to segmentation logits.

        Args:
            tokens: [B, N, C] processed tokens
            skips: (e3_last, e2_last, e1_last)
            spatial_shape: (H_b, W_b) bottleneck spatial dims
        """
        B = tokens.shape[0]
        C_b = tokens.shape[2]
        H_b, W_b = spatial_shape

        b_last = tokens.permute(0, 2, 1).reshape(B, C_b, H_b, W_b)
        e3_last, e2_last, e1_last = skips

        # Slice skip connections to match batch size B
        e3_last = e3_last[:B]
        e2_last = e2_last[:B]
        e1_last = e1_last[:B]

        return self._decode(b_last, e3_last, e2_last, e1_last)
