"""
VivimGazer: VivimBackboneMultiFrame + GazerHead.

Wraps the Mamba3 multi-frame backbone and attaches the GAZER action policy head
on the Mamba3 bottleneck features (b_mamba [B, T, C_b, H_b, W_b]).

forward(x, mode):
  mode='decode'   -> existing inference path, [B, C, H, W] (last frame + FIFO)
  mode='gazer'    -> (logits [B, T, C, H, W], a_logits [B, T, 2], q [B, T, 1])
  mode='rollout'  -> (logits, GazerSample) using GazerHead.sample
  mode='encode'   -> b_mamba [B, T, C_b, H_b, W_b]

The segmentation decoder/encoder are byte-for-byte identical to the base
backbone, so existing checkpoints load with strict=False (GazerHead is the only
new sub-module). Inference (mode='decode') does not touch the GazerHead at all,
keeping real-time 120Hz FIFO streaming unchanged.
"""

import torch
import torch.nn as nn

from models.vivim_backbone_multiframe import VivimBackboneMultiFrame
from models.gazer_head import GazerHead, GazerSample


class VivimGazer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        base_channels=32,
        d_state=16,
        d_conv=4,
        expand=2,
        mamba_headdim=64,
        use_mamba=True,
        mamba_version="mamba3",
        temporal_window=5,
        use_fifo_cache=True,
        dec_base=None,
        enc_base=None,
        gazer_d_z=256,
        gazer_n_transformer_layers=2,
        gazer_pool_size=14,
        gazer_hidden=256,
        use_gazer=True,
        **kwargs,
    ):
        super().__init__()
        self.backbone = VivimBackboneMultiFrame(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mamba=use_mamba,
            mamba_version=mamba_version,
            mamba_headdim=mamba_headdim,
            temporal_window=temporal_window,
            use_fifo_cache=use_fifo_cache,
            dec_base=dec_base,
            enc_base=enc_base,
        )
        self.use_gazer = use_gazer
        self.bottleneck_channels = base_channels * 8
        if use_gazer:
            self.gazer = GazerHead(
                d_bottleneck=self.bottleneck_channels,
                d_z=gazer_d_z,
                n_transformer_layers=gazer_n_transformer_layers,
                pool_size=gazer_pool_size,
                hidden=gazer_hidden,
            )
        else:
            self.gazer = None

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        """Return b_mamba [B, T, C_b, H_b, W_b] from the Mamba3 bottleneck."""
        return self.backbone(x, encode_only=True)

    def forward(self, x, mode="decode", output_all_frames=False):
        """
        mode='decode'  -> [B, C, H, W] (inference, FIFO, GazerHead unused)
        mode='encode'  -> b_mamba [B, T, C_b, H_b, W_b]
        mode='gazer'   -> (logits [B,T,C,H,W], a_logits [B,T,2], q [B,T,1])
        mode='rollout' -> (logits [B,T,C,H,W], GazerSample)
        Single-pass: backbone decodes all T frames AND returns b_mamba.
        """
        if mode == "decode":
            return self.backbone(x, output_all_frames=False)

        if mode == "encode":
            return self.encode(x)

        if not self.use_gazer or self.gazer is None:
            return self.backbone(x, output_all_frames=output_all_frames)

        # Single forward: decode all frames + capture Mamba3 bottleneck features.
        logits, b_mamba = self.backbone(x, output_all_frames=True, return_b_mamba=True)
        if mode == "gazer":
            a_logits, q = self.gazer(b_mamba)
            return logits, a_logits, q
        if mode == "rollout":
            sample = self.gazer.sample(b_mamba)
            return logits, sample
        raise ValueError(f"Unknown mode: {mode}")
