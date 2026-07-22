"""
TemporalDecoder: UNet Decoder with Bottleneck ConvLSTM for temporal feature fusion.

Architecture:
    Frozen Encoder outputs skips for each of T frames.
    At the bottleneck (lowest resolution), a ConvLSTM fuses temporal information.
    The remaining decoder stages (transpconv + conv blocks + skip connections)
    are structurally identical to the original UNetDecoder but with fresh weights.

This module is designed to be a drop-in replacement for UNetDecoder when working
with temporal sequences.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Tuple, Type, Optional

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from convlstm import ConvLSTMCell


class TemporalDecoder(nn.Module):
    """
    UNet Decoder with a ConvLSTM cell at the bottleneck for temporal fusion.

    During training:
        Receives T sets of skip connections and fuses them at the bottleneck.
    During streaming inference:
        Receives 1 set of skip connections and updates internal ConvLSTM state.

    Args:
        encoder: The frozen PlainConvEncoder (used only for architecture metadata).
        num_classes: Number of segmentation classes (4 for OpenEDS).
        n_conv_per_stage: Number of conv blocks per decoder stage.
        deep_supervision: Whether to output intermediate predictions.
        bottleneck_hidden_dim: Hidden dim for the bottleneck ConvLSTM (default: same as bottleneck channels).
        convlstm_kernel_size: Kernel size for ConvLSTM (default: 3).
    """

    def __init__(
        self,
        encoder: Union[PlainConvEncoder, ResidualEncoder],
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool = True,
        bottleneck_hidden_dim: Optional[int] = None,
        convlstm_kernel_size: int = 3,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = None,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        # ─── Bottleneck ConvLSTM ───
        bottleneck_channels = encoder.output_channels[-1]  # e.g., 512
        if bottleneck_hidden_dim is None:
            bottleneck_hidden_dim = bottleneck_channels
        self.bottleneck_hidden_dim = bottleneck_hidden_dim

        self.bottleneck_convlstm = ConvLSTMCell(
            input_dim=bottleneck_channels,
            hidden_dim=bottleneck_hidden_dim,
            kernel_size=convlstm_kernel_size
        )

        # Projection layer if hidden_dim != bottleneck_channels
        if bottleneck_hidden_dim != bottleneck_channels:
            self.bottleneck_proj = nn.Conv2d(bottleneck_hidden_dim, bottleneck_channels, 1)
        else:
            self.bottleneck_proj = nn.Identity()

        # ─── Standard UNet Decoder stages (fresh weights, same architecture) ───
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        stages = []
        transpconvs = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv,
                stride_for_transpconv, bias=conv_bias
            ))

            stages.append(StackedConvBlocks(
                n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip,
                input_features_skip, encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, nonlin_first
            ))

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

        # Internal ConvLSTM state (for streaming inference)
        self._lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_state(self):
        """Reset ConvLSTM hidden state. Call at the start of a new sequence."""
        self._lstm_state = None

    def forward_single(self, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Process a single time step (streaming mode).

        Args:
            skips: List of encoder skip connections for ONE frame.
                   skips[-1] is the bottleneck, skips[0] is highest resolution.
        Returns:
            Segmentation output (or list if deep_supervision=True).
        """
        bottleneck = skips[-1]  # [B, 512, 7, 10]

        # Temporal fusion at bottleneck via ConvLSTM
        h, self._lstm_state = self.bottleneck_convlstm(bottleneck, self._lstm_state)
        lres_input = self.bottleneck_proj(h)

        # Standard UNet decoder path
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            skip = skips[-(s + 2)]

            # Handle spatial size mismatch (non-power-of-2 input dimensions)
            if x.shape[2:] != skip.shape[2:]:
                x = x[:, :, :skip.shape[2], :skip.shape[3]]

            x = torch.cat((x, skip), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def forward(
        self,
        skips_sequence: List[List[torch.Tensor]],
    ):
        """
        Process a sequence of T frames (training mode).

        Args:
            skips_sequence: List of T skip-connection lists.
                skips_sequence[t] = [skip_stage0, skip_stage1, ..., skip_bottleneck]

        Returns:
            Segmentation output for the LAST frame in the sequence.
        """
        self.reset_state()

        for t in range(len(skips_sequence)):
            output = self.forward_single(skips_sequence[t])

        # Return prediction for the last frame only
        return output
