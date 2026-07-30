import os
import sys
import torch
from torch import nn, autocast
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
from nnunetv2.utilities.helpers import dummy_context

# Ensure project root is in sys.path
sys.path.insert(0, '/home/iulab0/PycharmProjects/nnUNet')

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from models.vivim_backbone import VivimBackbone
from datasets.openeds_dataset import OpenEDS400SequenceDataset
from losses.weakmed_loss import DiceCELoss, WeakMedEyeballLoss
from utils.config import load_config, ConfigDict


class VivimBackboneNNUNet(nn.Module):
    """
    nnUNet-compatible wrapper for Vivim (Video Vision Mamba).
    Natively receives 5D Temporal Sequences [B, T=3, C=1, H, W] for true Spatio-Temporal Mamba Scanning.
    Optionally includes a separate eyeball segmentation head for WeakMEd supervision.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 use_eyeball_head: bool = False):
        super().__init__()
        self.use_eyeball_head = use_eyeball_head
        self.backbone = VivimBackbone(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mamba=True,
            use_eyeball_head=use_eyeball_head,
        )

    def forward(self, x: torch.Tensor):
        if x.ndim == 5:
            # True 5D Temporal Sequence: [B, T=3, C=1, H, W] -> Temporal Mamba Bottleneck
            out = self.backbone(x)
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
    When WeakMEd is enabled, also passes 'eyeball_bbox' through.
    """
    def __init__(self, dataloader: DataLoader, pass_eyeball_bbox: bool = False):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.pass_eyeball_bbox = pass_eyeball_bbox

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        result = {
            'data': batch['images'],               # [B, T=3, 1, 448, 448]
            'target': batch['label'].unsqueeze(1),  # [B, 1, 448, 448]
        }

        if self.pass_eyeball_bbox and 'eyeball_bbox' in batch:
            result['eyeball_bbox'] = batch['eyeball_bbox']  # [B, 4]

        return result


class nnUNetTrainer_Vivim(nnUNetTrainer):
    """
    Custom nnUNetTrainer integrating Vivim (Video Vision Mamba) with TRUE 3-frame Temporal Sequence DataLoader
    natively inside nnUNet v2.

    Supports WeakMEd eyeball supervision via config switch (cfg.ablation.use_weakmed).
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

        # Load config if available, otherwise use defaults
        self.cfg = self._load_experiment_config()
        self.use_weakmed = getattr(self.cfg.ablation, 'use_weakmed', False) if self.cfg else False
        self.use_eyeball_head = getattr(self.cfg.ablation, 'use_eyeball_head', False) if self.cfg else False

        # WeakMEd loss module (initialized lazily after config is loaded)
        self.weakmed_loss = None
        self.seg_loss = None

    def _load_experiment_config(self) -> ConfigDict:
        """Try to load experiment config from known paths."""
        config_candidates = [
            os.path.join('/home/iulab0/PycharmProjects/nnUNet/configs', 'weakmed_eyeball.yaml'),
            os.path.join('/home/iulab0/PycharmProjects/nnUNet/configs', 'baseline.yaml'),
        ]
        for path in config_candidates:
            if os.path.exists(path):
                try:
                    cfg = load_config(path)
                    print(f"[nnUNetTrainer_Vivim] Loaded config: {path}", flush=True)
                    return cfg
                except Exception as e:
                    print(f"[nnUNetTrainer_Vivim] Config load failed for {path}: {e}", flush=True)
        print("[nnUNetTrainer_Vivim] No config found, using defaults", flush=True)
        return ConfigDict({
            'ablation': {'use_mamba': True, 'use_weakmed': False, 'use_eyeball_head': False},
            'losses': {'seg': {'weight': 1.0}},
            'training': {'freeze_encoder_epochs': 0},
        })

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def initialize(self):
        """Override to set up WeakMEd loss after network is built."""
        super().initialize()

        # Segmentation loss (4-class DiceCE)
        self.seg_loss = DiceCELoss(num_classes=4)

        # WeakMEd eyeball loss
        if self.use_weakmed:
            weakmed_cfg = self.cfg.losses.weakmed if self.cfg else {}
            m2b_w = getattr(weakmed_cfg, 'm2b', {}).get('weight', 1.0) if hasattr(weakmed_cfg, 'm2b') else 1.0
            sc_w = getattr(weakmed_cfg, 'sc', {}).get('weight', 1.0) if hasattr(weakmed_cfg, 'sc') else 1.0
            sc_scale = getattr(weakmed_cfg, 'sc', {}).get('scale_factor', 0.5) if hasattr(weakmed_cfg, 'sc') else 0.5

            self.weakmed_loss = WeakMedEyeballLoss(
                m2b_weight=m2b_w,
                sc_weight=sc_w,
                sc_scale_factor=sc_scale,
            )
            print(f"[nnUNetTrainer_Vivim] WeakMEd loss initialized (M2B={m2b_w}, SC={sc_w})", flush=True)

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

        return (
            nnUNetSequenceDataLoaderWrapper(train_loader, pass_eyeball_bbox=self.use_weakmed),
            nnUNetSequenceDataLoaderWrapper(val_loader, pass_eyeball_bbox=self.use_weakmed),
        )

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            run_name = f"nnUNet_Vivim_Mamba_Temporal_fold{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if self.use_weakmed:
                run_name = f"WeakMEd_{run_name}"
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
                    "use_weakmed": self.use_weakmed,
                    "use_eyeball_head": self.use_eyeball_head,
                }
            )

    def on_train_epoch_start(self):
        """Staged backbone unfreeze: freeze encoder for first N epochs."""
        super().on_train_epoch_start()

        freeze_epochs = 0
        if self.cfg and hasattr(self.cfg, 'training'):
            freeze_epochs = getattr(self.cfg.training, 'freeze_encoder_epochs', 0)

        if freeze_epochs > 0 and hasattr(self.network, 'backbone'):
            backbone = self.network.backbone
            if self.current_epoch < freeze_epochs:
                # Freeze encoder layers
                for module_name in ['enc1', 'enc2', 'enc3', 'bottleneck']:
                    module = getattr(backbone, module_name, None)
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = False
                if self.current_epoch == 0:
                    print(f"[nnUNetTrainer_Vivim] Encoder FROZEN for first {freeze_epochs} epochs", flush=True)
            elif self.current_epoch == freeze_epochs:
                # Unfreeze all
                for param in self.network.parameters():
                    param.requires_grad = True
                print(f"[nnUNetTrainer_Vivim] Encoder UNFROZEN at epoch {self.current_epoch}", flush=True)

    def train_step(self, batch: dict) -> dict:
        """
        Override train_step to handle multi-task output (seg + eyeball)
        and WeakMEd loss computation.
        """
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            # Handle multi-task output
            if isinstance(output, dict):
                seg_logits = output['seg']
                eyeball_logits = output.get('eyeball', None)
            else:
                seg_logits = output
                eyeball_logits = None

            # Primary segmentation loss
            total_loss = self.seg_loss(seg_logits, target.squeeze(1).long())

            # WeakMEd eyeball loss
            weakmed_losses = {}
            if self.use_weakmed and eyeball_logits is not None and 'eyeball_bbox' in batch:
                bbox = batch['eyeball_bbox'].to(self.device, non_blocking=True)
                weakmed_result = self.weakmed_loss(eyeball_logits, bbox)
                weakmed_weight = 0.5
                if self.cfg and hasattr(self.cfg, 'losses') and hasattr(self.cfg.losses, 'weakmed'):
                    weakmed_weight = getattr(self.cfg.losses.weakmed, 'weight', 0.5)
                total_loss = total_loss + weakmed_weight * weakmed_result['total']
                weakmed_losses = {
                    'weakmed_m2b': weakmed_result['m2b'].item(),
                    'weakmed_sc': weakmed_result['sc'].item(),
                    'weakmed_total': weakmed_result['total'].item(),
                }

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        result = {'loss': total_loss.detach().cpu().numpy()}
        result.update(weakmed_losses)
        return result

    def validation_step(self, batch: dict) -> dict:
        """Override to handle multi-task output during validation."""
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.network(data)

                if isinstance(output, dict):
                    seg_logits = output['seg']
                else:
                    seg_logits = output

                val_loss = self.seg_loss(seg_logits, target.squeeze(1).long())

        # Compute pseudo dice for online evaluation
        predicted = seg_logits.argmax(dim=1)
        target_sq = target.squeeze(1)

        # Per-class dice (skip background)
        tp_hard = torch.zeros(3, device=self.device)  # 3 foreground classes
        fp_hard = torch.zeros(3, device=self.device)
        fn_hard = torch.zeros(3, device=self.device)
        for c in range(1, 4):  # Sclera, Iris, Pupil
            pred_c = (predicted == c).float()
            tgt_c = (target_sq == c).float()
            tp_hard[c - 1] = (pred_c * tgt_c).sum()
            fp_hard[c - 1] = (pred_c * (1 - tgt_c)).sum()
            fn_hard[c - 1] = ((1 - pred_c) * tgt_c).sum()

        return {
            'loss': val_loss.detach().cpu().numpy(),
            'tp_hard': tp_hard.detach().cpu().numpy(),
            'fp_hard': fp_hard.detach().cpu().numpy(),
            'fn_hard': fn_hard.detach().cpu().numpy(),
        }

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.local_rank == 0:
            try:
                epoch_idx = self.current_epoch - 1
                train_loss = self.logger.get_value('train_losses', step=-1)
                val_loss = self.logger.get_value('val_losses', step=-1)
                mean_fg_dice = self.logger.get_value('mean_fg_dice', step=-1)
                ema_dice = self.logger.get_value('ema_fg_dice', step=-1)

                log_dict = {
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mean_fg_dice": mean_fg_dice,
                    "ema_pseudo_dice": ema_dice,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                }
                wandb.log(log_dict)
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
        # Check if WeakMEd config exists to determine eyeball head
        use_eyeball = False
        config_path = '/home/iulab0/PycharmProjects/nnUNet/configs/weakmed_eyeball.yaml'
        if os.path.exists(config_path):
            try:
                cfg = load_config(config_path)
                use_eyeball = getattr(cfg.ablation, 'use_eyeball_head', False)
            except Exception:
                pass

        head_str = " + Eyeball Head" if use_eyeball else ""
        print(f"[nnUNetTrainer_Vivim] Building Vivim Temporal Architecture{head_str}!", flush=True)
        return VivimBackboneNNUNet(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            base_channels=32,
            d_state=16,
            d_conv=4,
            expand=2,
            use_eyeball_head=use_eyeball,
        )
