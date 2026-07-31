import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
import wandb

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from models.vivim_backbone import VivimBackbone
from datasets.openeds_dataset import OpenEDS400SequenceDataset
from losses.weakmed_loss import WeakMedSphereLoss
from utils.config import ConfigDict
from utils.metrics import compute_dice_score
from utils.sphere_metrics import compute_all_sphere_metrics


class VivimBackboneNNUNet(nn.Module):
    """
    nnUNet-compatible wrapper for Vivim (Video Vision Mamba).
    Natively receives 5D Temporal Sequences [B, T=3, C=1, H, W] for true Spatio-Temporal Mamba Scanning
    and incorporates the Parametric Sphere Head for eyeball estimation.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        sphere_head_cfg = {
            'hidden_dim': 128,
            'max_radius': 200.0,
            'min_radius': 30.0,
            'sharpness': 20.0,
            'sharpness_max': 50.0,
            'sharpness_anneal_epochs': 50,
        }
        self.backbone = VivimBackbone(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mamba=True,
            use_sphere_head=True,
            sphere_head_cfg=sphere_head_cfg,
        )

    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        return self.backbone(x)


class nnUNetSequenceDataLoaderWrapper:
    """
    Wraps OpenEDS400SequenceDataset to yield 5D Temporal Sequence batches [B, T=3, C, H, W]
    and auxiliary bounding box & circle labels for WeakMed & Sphere metrics.
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
            'data': batch['images'],                              # [B, T=3, 1, 448, 448]
            'target': batch['label'].unsqueeze(1),                # [B, 1, 448, 448]
            'eyeball_bbox': batch.get('eyeball_bbox', None),      # [B, 4]
            'eyeball_circle': batch.get('eyeball_circle', None),  # [B, 3]
        }


class nnUNetTrainer_Vivim(nnUNetTrainer):
    """
    Native nnUNet v2 Trainer for Vivim + Parametric Sphere Head + WeakMed Loss.

    - Natively follows nnUNet training schedule:
      - 1000 epochs (no early stopping)
      - PolyLRScheduler decay ((1 - epoch / 1000) ** 0.9)
      - 250 iterations/epoch (train), 50 iterations/epoch (val)
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 1e-3
        self.weight_decay = 1e-4

        # WeakMed Loss config
        self.weakmed_cfg = ConfigDict({
            'losses': {
                'seg_dice_ce': {'weight': 1.0},
                'weakmed_m2b': {'weight': 1.0},
                'sclera_containment': {'weight': 0.5, 'warmup_epochs': 10},
            },
            'ablation': {
                'use_weakmed': True,
                'use_sclera_containment': True,
            },
        })

    def configure_optimizers(self):
        """Use AdamW with lr=1e-3 and PolyLRScheduler to prevent gradient explosion on SphereHead."""
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def initialize_network(self):
        """Build network and load pre-trained seg head weights."""
        super().initialize_network()
        ckpt_path = "/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/vivim_weakmed_400x400/checkpoint_best.pth"
        if os.path.exists(ckpt_path):
            print(f"[nnUNetTrainer_Vivim] Loading pre-trained seg weights from {ckpt_path}...", flush=True)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)

            model_keys = set(self.network.state_dict().keys())
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('sphere_head.') or k.startswith('backbone.sphere_head.'):
                    continue
                if k in model_keys:
                    new_state_dict[k] = v
                elif f"backbone.{k}" in model_keys:
                    new_state_dict[f"backbone.{k}"] = v

            missing, unexpected = self.network.load_state_dict(new_state_dict, strict=False)
            print(f"[nnUNetTrainer_Vivim] Successfully loaded {len(new_state_dict)} pre-trained weights into VivimBackbone! Missing: {len(missing)}.", flush=True)

    def _build_loss(self):
        """Override nnUNet default loss with WeakMedSphereLoss."""
        return WeakMedSphereLoss(num_classes=self.label_manager.num_segmentation_heads)

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def get_dataloaders(self):
        """Overrides nnUNet's default DataLoader to yield 3-frame Temporal Sequences (T=3)."""
        print("[nnUNetTrainer_Vivim] Loading 3-frame Temporal Sequence DataLoader (T=3)!", flush=True)
        data_root = os.environ.get('DATA_ROOT', '/home/iulab0/PycharmProjects/nnUNet')
        seq_root = os.path.join(data_root, 'Openedsdata2019', 'Sequence_Extracted')
        pseudo_root = os.path.join(data_root, 'Openedsdata2019', 'Sequence_PseudoLabels_400')

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

    def train_step(self, batch: dict) -> dict:
        """Custom train step handling model dict output and WeakMedSphereLoss."""
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True).squeeze(1).long()
        eyeball_bbox = batch['eyeball_bbox'].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Sharpness annealing over first 50 epochs
        if hasattr(self.network, 'backbone') and hasattr(self.network.backbone, 'sphere_head'):
            k_init, k_max = 20.0, 50.0
            progress = min(1.0, self.current_epoch / 50.0)
            self.network.backbone.sphere_head.set_sharpness(k_init + (k_max - k_init) * progress)

        # Forward
        output = self.network(data)  # dict with 'seg_logits', 'sphere_params', 'sphere_mask'

        # Loss
        loss_dict = self.loss(output, target, eyeball_bbox, self.weakmed_cfg, current_epoch=self.current_epoch)
        l = loss_dict['total_loss']

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """Custom validation step computing Dice, Circle IoU, and nnUNet tp/fp/fn_hard metrics."""
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)  # [B, 1, H, W]
        target_squeezed = target.squeeze(1).long()
        eyeball_bbox = batch['eyeball_bbox'].to(self.device, non_blocking=True)
        eyeball_circle = batch['eyeball_circle'].to(self.device, non_blocking=True)

        with torch.no_grad():
            output = self.network(data)
            loss_dict = self.loss(output, target_squeezed, eyeball_bbox, self.weakmed_cfg, current_epoch=self.current_epoch)
            l = loss_dict['total_loss']

            seg_logits = output['seg_logits']
            axes = [0] + list(range(2, seg_logits.ndim))
            output_seg = seg_logits.argmax(1)[:, None]
            predicted_onehot = torch.zeros(seg_logits.shape, device=seg_logits.device, dtype=torch.float16)
            predicted_onehot.scatter_(1, output_seg, 1)

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_onehot, target, axes=axes, mask=None)

            tp_hard = tp.detach().cpu().numpy()[1:]  # Exclude background
            fp_hard = fp.detach().cpu().numpy()[1:]
            fn_hard = fn.detach().cpu().numpy()[1:]

            dice_dict = compute_dice_score(seg_logits, target_squeezed, num_classes=self.label_manager.num_segmentation_heads)
            sphere_m = compute_all_sphere_metrics(output['sphere_params'], eyeball_circle)

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'mean_dice': dice_dict['Mean_Dice'],
            'circle_iou': sphere_m['circle_iou_mean'],
        }

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            run_name = f"nnUNet_Vivim_WeakMedSphere_fold{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[nnUNetTrainer_Vivim] Initializing WandB Sync: {run_name}", flush=True)
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

                wandb.log({
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mean_fg_dice": mean_fg_dice,
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
        print("[nnUNetTrainer_Vivim] Natively building Vivim + Parametric Sphere Head inside nnUNet v2!", flush=True)
        return VivimBackboneNNUNet(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            base_channels=32,
            d_state=16,
            d_conv=4,
            expand=2,
        )
