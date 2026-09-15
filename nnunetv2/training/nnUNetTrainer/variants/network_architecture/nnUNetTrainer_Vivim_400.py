import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
import wandb

# Ensure project root is in sys.path
sys.path.insert(0, '/home/iulab1/PycharmProjects/nnUNet')

# PyTorch 2.5.1 + quack compatibility
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = None
if not hasattr(torch, 'float4_e2m1fn_x2'):
    torch.float4_e2m1fn_x2 = None

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from models.vivim_backbone import VivimBackbone
from datasets.openeds_dataset import OpenEDS400SequenceDataset
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_Vivim import VivimBackboneNNUNet, nnUNetSequenceDataLoaderWrapper


class nnUNetTrainer_Vivim_400(nnUNetTrainer):
    """
    Unified custom nnUNetTrainer integrating Vivim (Video Vision Mamba) with TRUE 400x400 (448 padded) Temporal Sequence DataLoader.
    Supports dynamic temporal window (T=3, T=5, T=7, etc.) via TEMPORAL_WINDOW env var.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.temporal_window = int(os.environ.get('TEMPORAL_WINDOW', '3'))

        # Ensure output folder is unique per T value and preserves 192 weights
        self.output_folder = self.output_folder + f"_400_T{self.temporal_window}"
        os.makedirs(self.output_folder, exist_ok=True)

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def get_plain_dataloaders(self, initial_patch_size=None, dim=None):
        """
        Overrides nnUNet's default 2D DataLoader to yield TRUE 400x400 (448 pad) Temporal Sequences (T=temporal_window).
        """
        print(f"[nnUNetTrainer_Vivim_400] Loading True {self.temporal_window}-frame 400x400 Temporal Sequence DataLoader (T={self.temporal_window})!", flush=True)
        seq_root = "/home/iulab1/PycharmProjects/nnUNet/Openedsdata2019/Sequence_Extracted"
        pseudo_root = "/home/iulab1/PycharmProjects/nnUNet/Openedsdata2019/Sequence_PseudoLabels_400"

        train_ds = OpenEDS400SequenceDataset(
            image_dir=os.path.join(seq_root, 'train'),
            label_dir=os.path.join(pseudo_root, 'train'),
            temporal_window=self.temporal_window,
        )

        val_ds = OpenEDS400SequenceDataset(
            image_dir=os.path.join(seq_root, 'validation'),
            label_dir=os.path.join(pseudo_root, 'validation'),
            temporal_window=self.temporal_window,
        )

        train_loader = DataLoader(train_ds, batch_size=self.configuration_manager.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.configuration_manager.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

        return nnUNetSequenceDataLoaderWrapper(train_loader), nnUNetSequenceDataLoaderWrapper(val_loader)

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            run_name = f"nnUNet_Vivim_Mamba_400_T{self.temporal_window}_fold{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[nnUNetTrainer_Vivim_400] Initializing WandB Sync for 400x400 T={self.temporal_window} run: {run_name}", flush=True)
            wandb.init(
                project="eyeball-3d",
                name=run_name,
                config={
                    "trainer": self.__class__.__name__,
                    "plans": self.plans_manager.plans_name,
                    "configuration": self.configuration_name,
                    "fold": self.fold,
                    "temporal_window": self.temporal_window,
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
        print("[nnUNetTrainer_Vivim_400] Natively building Vivim (Video Vision Mamba) Temporal Architecture inside nnUNet v2!", flush=True)
        return VivimBackboneNNUNet(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            base_channels=32,
            d_state=16,
            d_conv=4,
            expand=2,
        )
