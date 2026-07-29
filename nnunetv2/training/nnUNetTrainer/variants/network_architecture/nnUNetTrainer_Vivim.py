import os
import sys
import torch
from torch import nn
from datetime import datetime
import wandb

# Ensure project root is in sys.path
sys.path.insert(0, '/home/iulab0/PycharmProjects/nnUNet')

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from models.vivim_backbone import VivimBackbone


class VivimBackboneNNUNet(nn.Module):
    """
    nnUNet-compatible wrapper for Vivim (Video Vision Mamba).
    Handles both 4D [B, C, H, W] (single frame / 2D nnUNet) and 5D [B, T, C, H, W] (sequence) inputs.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.backbone = VivimBackbone(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mamba=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # 2D input from standard nnUNet DataLoader: [B, C, H, W] -> unsqueeze T=1 -> [B, 1, C, H, W]
            x_seq = x.unsqueeze(1)
            out = self.backbone(x_seq)  # [B, Num_Classes, H, W]
            return out
        elif x.ndim == 5:
            # 5D Sequence input: [B, T, C, H, W]
            out = self.backbone(x)
            return out
        else:
            raise ValueError(f"Invalid input tensor shape for VivimBackboneNNUNet: {x.shape}")


class nnUNetTrainer_Vivim(nnUNetTrainer):
    """
    Custom nnUNetTrainer integrating Vivim (Video Vision Mamba) as the native architecture inside nnUNet v2.
    Leverages nnUNet's automatic plan management, PolyLR scheduling, 250 iterations/epoch, validation logging, and WandB integration.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            run_name = f"nnUNet_Vivim_Mamba_fold{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[nnUNetTrainer_Vivim] Initializing WandB Sync for run: {run_name}", flush=True)
            wandb.init(
                project="eyeball-3d",
                name=run_name,
                config={
                    "trainer": self.__class__.__name__,
                    "plans": self.plans_manager.plans_name,
                    "configuration": self.configuration_name,
                    "fold": self.fold,
                    "num_epochs": self.num_epochs,
                    "initial_lr": self.initial_lr,
                }
            )

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.local_rank == 0:
            try:
                epoch_idx = self.current_epoch - 1
                train_loss = self.logger.get_value('train_losses', step=-1)
                val_loss = self.logger.get_value('val_losses', step=-1)
                mean_fg_dice = self.logger.get_value('mean_fg_dice', step=-1)
                ema_dice = self.logger.get_value('ema_fg_dice', step=-1)

                wandb.log({
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mean_fg_dice": mean_fg_dice,
                    "ema_pseudo_dice": ema_dice,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                })
            except Exception as e:
                print(f"[WandB Logging Error] {e}", flush=True)

    def on_train_end(self):
        super().on_train_end()
        if self.local_rank == 0:
            wandb.finish()

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False
    ) -> nn.Module:
        print("[nnUNetTrainer_Vivim] Natively building Vivim (Video Vision Mamba) Architecture inside nnUNet v2!", flush=True)
        return VivimBackboneNNUNet(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            base_channels=32,
            d_state=16,
            d_conv=4,
            expand=2,
        )
