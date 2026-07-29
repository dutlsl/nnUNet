import sys
sys.path.insert(0, '/home/iulab0/PycharmProjects/nnUNet')

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from models.vivim_backbone import VivimBackbone


class nnUNetTrainer_Vivim(nnUNetTrainer):
    """
    Modular nnUNet v2 extension for Vivim (Video Vision Mamba).
    Plugs custom backbone directly into nnUNet's modular architecture builder.
    """
    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False
    ):
        return VivimBackbone(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            base_channels=32,
            d_state=16,
            d_conv=4,
            expand=2,
            use_mamba=True,
        )
