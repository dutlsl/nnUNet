"""
Training script for 400x400 TemporalUNet (Frozen 400x400 Encoder + ConvLSTM Temporal Decoder) with WandB logging.

Uses 100% on-the-fly cropping [120:520, 0:400] for sequence images and pairs them with 400x400 pseudo-labels.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import GradScaler, autocast
from pathlib import Path
import wandb

sys.path.insert(0, os.path.dirname(__file__))

from temporal_unet import TemporalUNet
from sequence_dataset_400 import get_sequence_400_dataloaders
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from batchgenerators.utilities.file_and_folder_operations import load_json


# ─── Loss Functions ───

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.dice_loss(logits, target) + self.ce_loss(logits, target)


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, predictions: list, target: torch.Tensor) -> torch.Tensor:
        weights = [1.0 / (2 ** i) for i in range(len(predictions))]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        total_loss = 0
        for i, pred in enumerate(predictions):
            if pred.shape[2:] != target.shape[1:]:
                t_down = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=pred.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                t_down = target
            total_loss += weights[i] * self.loss_fn(pred, t_down)
        return total_loss


# ─── Training Function ───

def train_temporal_400(
    vanilla_checkpoint_path: str = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/VanillaUNet_400x400/checkpoint_best.pth',
    plans_path: str = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d/plans.json',
    sequence_root: str = '/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_Extracted',
    pseudo_label_root: str = '/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_PseudoLabels_400',
    output_dir: str = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/TemporalUNet_400x400',
    temporal_window: int = 3,
    batch_size: int = 4,
    num_epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    num_workers: int = 4,
):
    os.makedirs(output_dir, exist_ok=True)
    device_obj = torch.device(device)

    # Initialize Weights & Biases (wandb)
    wandb.init(
        project="nnUNet_400x400",
        name="TemporalUNet_ConvLSTM_400x400",
        config={
            "architecture": "TemporalUNet_ConvLSTM_Bottleneck",
            "input_resolution": "400x400 (On-The-Fly Crop)",
            "padded_resolution": "448x448",
            "temporal_window": temporal_window,
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": num_epochs,
            "num_classes": 4,
            "vanilla_checkpoint": vanilla_checkpoint_path,
        }
    )

    print(f"=== Starting 400x400 ConvLSTM TemporalUNet Training with WandB ===")
    print(f"Device: {device}")
    print(f"Vanilla Checkpoint: {vanilla_checkpoint_path}")
    print(f"Output Directory: {output_dir}")

    # Build 400x400 PlainConvUNet
    plans = load_json(plans_path)
    arch = plans['configurations']['2d']['architecture']
    base_unet = get_network_from_plans(
        arch['network_class_name'], arch['arch_kwargs'], arch['_kw_requires_import'],
        1, 4, allow_init=False, deep_supervision=True
    )

    # Load frozen encoder weights
    ckpt = torch.load(vanilla_checkpoint_path, map_location='cpu', weights_only=False)
    base_unet.load_state_dict(ckpt['state_dict'], strict=False)

    # Build TemporalUNet
    model = TemporalUNet(pretrained_unet=base_unet, num_classes=4, deep_supervision=True)
    model.to(device_obj)

    # Dataloaders
    loaders = get_sequence_400_dataloaders(
        sequence_root=sequence_root,
        pseudo_label_root=pseudo_label_root,
        temporal_window=temporal_window,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader = loaders['train']
    val_loader = loaders.get('validation', None)

    loss_fn = DeepSupervisionWrapper(DiceCELoss(num_classes=4))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device_obj)  # [B, T, 1, H, W]
            labels = batch['label'].to(device_obj)   # [B, H, W]

            optimizer.zero_grad()
            with autocast('cuda'):
                preds = model(images)
                loss = loss_fn(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['images'].to(device_obj)
                    labels = batch['label'].to(device_obj)
                    with autocast('cuda'):
                        preds = model(images)
                        loss = loss_fn(preds, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
        else:
            val_loss = train_loss

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:03d}/{num_epochs:03d}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_duration_sec": elapsed,
        })

        checkpoint_latest = os.path.join(output_dir, 'checkpoint_latest.pth')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss}, checkpoint_latest)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_best = os.path.join(output_dir, 'checkpoint_best.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss}, checkpoint_best)
            print(f"  -> Best TemporalUNet model saved! (Val Loss: {val_loss:.4f})")

    wandb.finish()
    print("\n=== TemporalUNet Training Completed Successfully! ===")


if __name__ == '__main__':
    train_temporal_400()
