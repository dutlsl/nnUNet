"""
TemporalUNet: Frozen Encoder + TemporalDecoder (Bottleneck ConvLSTM).

This model reuses the encoder weights from a trained nnUNet PlainConvUNet,
freezes them, and replaces the decoder with a TemporalDecoder that can
process sequences of T frames for temporally consistent segmentation.
"""
import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from temporal_decoder import TemporalDecoder


class TemporalUNet(nn.Module):
    """
    Temporal U-Net for sequential eye segmentation.

    Architecture:
        [Frozen PlainConvEncoder] → T sets of skip connections
                                         ↓
                               [TemporalDecoder (ConvLSTM at bottleneck)]
                                         ↓
                               Segmentation mask for frame t

    Args:
        pretrained_unet: A trained PlainConvUNet whose encoder weights are reused.
        num_classes: Number of segmentation classes (default: 4).
        deep_supervision: Whether to use deep supervision during training.
        bottleneck_hidden_dim: Hidden channels for bottleneck ConvLSTM.
        convlstm_kernel_size: Kernel size for ConvLSTM cell.
    """

    def __init__(
        self,
        pretrained_unet: PlainConvUNet,
        num_classes: int = 4,
        deep_supervision: bool = True,
        bottleneck_hidden_dim: Optional[int] = None,
        convlstm_kernel_size: int = 3,
    ):
        super().__init__()

        # ─── Frozen Encoder ───
        self.encoder = pretrained_unet.encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        # ─── Fresh Temporal Decoder ───
        # Extract decoder config from original UNetDecoder
        original_decoder = pretrained_unet.decoder
        n_conv_per_stage = []
        for stage in original_decoder.stages:
            # Each stage is a StackedConvBlocks; count its conv blocks
            n_conv_per_stage.append(len(stage.convs))

        self.decoder = TemporalDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=deep_supervision,
            bottleneck_hidden_dim=bottleneck_hidden_dim,
            convlstm_kernel_size=convlstm_kernel_size,
        )

        self.temporal_window = None  # Set during training

    def train(self, mode: bool = True):
        """Override to keep encoder always in eval mode (frozen BatchNorm)."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a temporal sequence.

        Args:
            x_sequence: [B, T, C, H, W] — T consecutive frames

        Returns:
            Segmentation prediction for the LAST frame.
        """
        B, T, C, H, W = x_sequence.shape

        # Extract skip connections for each frame (encoder is frozen, no grad)
        skips_sequence = []
        with torch.no_grad():
            for t in range(T):
                skips = self.encoder(x_sequence[:, t])  # List of skip tensors
                skips_sequence.append(skips)

        # Pass all T skip sets through temporal decoder
        return self.decoder(skips_sequence)

    def forward_streaming(self, x_frame: torch.Tensor) -> torch.Tensor:
        """
        Streaming inference: process one frame at a time.

        Call decoder.reset_state() at the start of each new sequence.

        Args:
            x_frame: [B, C, H, W] — single frame

        Returns:
            Segmentation prediction for this frame.
        """
        with torch.no_grad():
            skips = self.encoder(x_frame)

        return self.decoder.forward_single(skips)

    def reset_temporal_state(self):
        """Reset ConvLSTM state for a new sequence."""
        self.decoder.reset_state()

    @staticmethod
    def from_pretrained(
        model_folder: str,
        checkpoint_name: str = 'checkpoint_best.pth',
        num_classes: int = 4,
        deep_supervision: bool = True,
        bottleneck_hidden_dim: Optional[int] = None,
        convlstm_kernel_size: int = 3,
        device: torch.device = torch.device('cpu'),
    ) -> 'TemporalUNet':
        """
        Create a TemporalUNet from a trained nnUNet checkpoint.

        Args:
            model_folder: Path to nnUNet model folder (e.g., .../nnUNetTrainer__Plans__2d)
            checkpoint_name: Which checkpoint to load (default: checkpoint_best.pth)
            num_classes: Number of segmentation classes
            deep_supervision: Use deep supervision
            bottleneck_hidden_dim: ConvLSTM hidden dim (None = same as bottleneck channels)
            convlstm_kernel_size: ConvLSTM kernel size
            device: Device for model

        Returns:
            TemporalUNet with frozen encoder and fresh temporal decoder.
        """
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        from batchgenerators.utilities.file_and_folder_operations import load_json, join

        # Load plans and dataset info
        plans = load_json(join(model_folder, 'plans.json'))
        dataset_json = load_json(join(model_folder, 'dataset.json'))
        plans_manager = PlansManager(plans)
        config_manager = plans_manager.get_configuration('2d')

        # Find the fold directory with the checkpoint
        fold_dirs = [d for d in os.listdir(model_folder) if d.startswith('fold_')]
        fold_dir = join(model_folder, fold_dirs[0]) if fold_dirs else model_folder
        checkpoint_path = join(fold_dir, checkpoint_name)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Build network using the exact same method as nnUNetTrainer
        num_input_channels = determine_num_input_channels(
            plans_manager, config_manager, dataset_json
        )
        network = get_network_from_plans(
            config_manager.network_arch_class_name,
            config_manager.network_arch_init_kwargs,
            config_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_classes,
            allow_init=False,
            deep_supervision=deep_supervision,
        )

        # Load pretrained weights
        network.load_state_dict(checkpoint['network_weights'])
        network.to(device)

        # Create TemporalUNet
        model = TemporalUNet(
            pretrained_unet=network,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            bottleneck_hidden_dim=bottleneck_hidden_dim,
            convlstm_kernel_size=convlstm_kernel_size,
        ).to(device)

        print(f"TemporalUNet created from {checkpoint_path}")
        enc_params = sum(p.numel() for p in model.encoder.parameters())
        dec_params = sum(p.numel() for p in model.decoder.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Encoder params (frozen): {enc_params:,}")
        print(f"  Decoder params (trainable): {dec_params:,}")
        print(f"  Total trainable: {trainable:,}")

        return model
