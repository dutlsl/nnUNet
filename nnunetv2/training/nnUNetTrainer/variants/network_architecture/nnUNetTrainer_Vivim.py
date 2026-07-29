import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
import wandb

# Ensure project root is in sys.path
sys.path.insert(0, '/home/iulab0/PycharmProjects/nnUNet')

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from models.vivim_backbone import VivimBackbone
from datasets.openeds_dataset import OpenEDS400SequenceDataset


class VivimBackboneNNUNet(nn.Module):
    """
    nnUNet-compatible wrapper for Vivim (Video Vision Mamba).
    Natively receives 5D Temporal Sequences [B, T=3, C=1, H, W] for true Spatio-Temporal Mamba Scanning.
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
        if x.ndim == 5:
            # True 5D Temporal Sequence: [B, T=3, C=1, H, W] -> Temporal Mamba Bottleneck
            out = self.backbone(x)  # [B, Num_Classes, H, W]
            return out
        elif x.ndim == 4:
            x_seq = x.unsqueeze(1)
            out = self.backbone(x_seq)
            return out
        else:
            raise ValueError(f"Invalid input tensor shape for VivimBackboneNNUNet: {x.shape}")


class nnUNetSequenceDataLoaderWrapper:
    """
    Wraps OpenEDS400SequenceDataset to yield 5D Temporal Sequence batches [B, T=3, C, H, W]
    in the exact format expected by nnUNetTrainer ({'data': ..., 'target': ...}).
    """
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        return {
            'data': batch['images'],               # [B, T=3, 1, 448, 448] (연속 3프레임 시퀀스)
            'target': batch['label'].unsqueeze(1), # [B, 1, 448, 448] (정답 마스크)
        }


class nnUNetTrainer_Vivim(nnUNetTrainer):
    """
    Custom nnUNetTrainer integrating Vivim (Video Vision Mamba) with TRUE 3-frame Temporal Sequence DataLoader
    natively inside nnUNet v2.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def get_plain_dataloaders(self, initial_patch_size=None, dim=None):
        """
        Overrides nnUNet's default 2D DataLoader to yield TRUE 3-frame Temporal Sequences (T=3).
        """
        print("[nnUNetTrainer_Vivim] Loading True 3-frame Temporal Sequence DataLoader (T=3)!", flush=True)
        seq_root = "/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_Extracted"
        pseudo_root = "/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_PseudoLabels_400"

        train_ds = OpenEDS400SequenceDataset(
            image_dir=os.path.join(seq_root, 'train'),
            label_dir=os.path.join(pseudo_root, 'train'),
            temporal_window=3,
            crop=[120, 520, 0, 400],
            padded_resolution=[448, 448],
            mean=86.45,
            std=39.94,
        )

        val_ds = OpenEDS400SequenceDataset(
            image_dir=os.path.join(seq_root, 'validation'),
            label_dir=os.path.join(pseudo_root, 'validation'),
            temporal_window=3,
            crop=[120, 520, 0, 400],
            padded_resolution=[448, 448],
            mean=86.45,
            std=39.94,
        )

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        return nnUNetSequenceDataLoaderWrapper(train_loader), nnUNetSequenceDataLoaderWrapper(val_loader)

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            run_name = f"nnUNet_Vivim_Mamba_Temporal_fold{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[nnUNetTrainer_Vivim] Initializing WandB Sync for Temporal run: {run_name}", flush=True)
            wandb.init(
                project="eyeball-3d",
                name=run_name,
                config={
                    "trainer": self.__class__.__name__,
                    "plans": self.plans_manager.plans_name,
                    "configuration": self.configuration_name,
                    "fold": self.fold,
                    "temporal_window": 3,
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
        print("[nnUNetTrainer_Vivim] Natively building Vivim (Video Vision Mamba) Temporal Architecture inside nnUNet v2!", flush=True)
        return VivimBackboneNNUNet(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            base_channels=32,
            d_state=16,
            d_conv=4,
            expand=2,
        )
