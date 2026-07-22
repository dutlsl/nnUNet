"""
Training script for TemporalUNet (Frozen Encoder + ConvLSTM Temporal Decoder).

Usage:
    python train_temporal.py

This script:
  1. Loads the pretrained nnUNet encoder (frozen)
  2. Initializes a fresh TemporalDecoder with ConvLSTM at the bottleneck
  3. Trains the decoder on Sequence Dataset with pseudo-labels
  4. Uses Dice+CE loss with optional temporal consistency regularization
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from temporal_unet import TemporalUNet
from sequence_dataset import get_sequence_dataloaders


# ─── Loss Functions ───

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            target: [B, H, W] (integer class labels)
        """
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss (same as nnUNet default)."""

    def __init__(self, num_classes: int = 4, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, target):
        return self.dice_weight * self.dice_loss(logits, target) + \
               self.ce_weight * self.ce_loss(logits, target)


class DeepSupervisionWrapper(nn.Module):
    """Apply loss at multiple decoder resolutions with decreasing weights."""

    def __init__(self, loss_fn: nn.Module, weights: list = None):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, predictions: list, target: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            weights = [1.0 / (2 ** i) for i in range(len(predictions))]
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]
        else:
            weights = self.weights

        total_loss = 0
        for i, pred in enumerate(predictions):
            if pred.shape[2:] != target.shape[1:]:
                # Downsample target to match prediction resolution
                t_down = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=pred.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                t_down = target
            total_loss += weights[i] * self.loss_fn(pred, t_down)

        return total_loss


# ─── Training Loop ───

def train_temporal(
    model_folder: str,
    sequence_root: str,
    pseudo_label_root: str,
    output_dir: str,
    temporal_window: int = 3,
    batch_size: int = 4,
    num_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_temporal: float = 0.1,
    device: str = 'cuda',
    num_workers: int = 4,
    save_every: int = 25,
    checkpoint_name: str = 'checkpoint_best.pth',
):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'training_log.json')
    log_data = {'epochs': [], 'config': {
        'temporal_window': temporal_window, 'batch_size': batch_size,
        'num_epochs': num_epochs, 'lr': lr, 'lambda_temporal': lambda_temporal,
    }}

    # ─── 1. Build model ───
    print("Loading pretrained model and building TemporalUNet...")
    os.environ['nnUNet_raw'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results'

    model = TemporalUNet.from_pretrained(
        model_folder=model_folder,
        checkpoint_name=checkpoint_name,
        num_classes=4,
        deep_supervision=True,
        device=torch.device(device),
    )
    model.train()

    # ─── 2. Get normalization stats ───
    from batchgenerators.utilities.file_and_folder_operations import load_json, join
    plans = load_json(join(model_folder, 'plans.json'))
    fg_props = plans['foreground_intensity_properties_per_channel']['0']
    mean_val = fg_props['mean']
    std_val = fg_props['std']

    # ─── 3. Build dataloaders ───
    print("Building dataloaders...")
    loaders = get_sequence_dataloaders(
        sequence_root=sequence_root,
        pseudo_label_root=pseudo_label_root,
        mean=mean_val,
        std=std_val,
        temporal_window=temporal_window,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader = loaders['train']
    val_loader = loaders.get('validation', None)

    # ─── 4. Loss, optimizer, scheduler ───
    base_loss = DiceCELoss(num_classes=4)
    loss_fn = DeepSupervisionWrapper(base_loss)

    # Only optimize decoder parameters (encoder is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    best_val_loss = float('inf')
    print(f"\nStarting training: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ─── 5. Training loop ───
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)  # [B, T, 1, H, W]
            labels = batch['label'].to(device)    # [B, H, W]

            optimizer.zero_grad()

            with autocast():
                predictions = model(images)  # list of [B, 4, H_s, W_s] if deep_supervision

                if isinstance(predictions, list):
                    seg_loss = loss_fn(predictions, labels)
                else:
                    seg_loss = base_loss(predictions, labels)

                loss = seg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # ─── 6. Validation ───
        avg_val_loss = float('nan')
        if val_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['images'].to(device)
                    labels = batch['label'].to(device)
                    with autocast():
                        predictions = model(images)
                        if isinstance(predictions, list):
                            vloss = loss_fn(predictions, labels)
                        else:
                            vloss = base_loss(predictions, labels)
                    val_loss += vloss.item()
            avg_val_loss = val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, os.path.join(output_dir, 'checkpoint_best.pth'))
                print(f"  ★ New best val loss: {avg_val_loss:.4f}")

        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss if not np.isnan(avg_val_loss) else None,
            'lr': scheduler.get_last_lr()[0],
            'epoch_time_s': epoch_time,
        }
        log_data['epochs'].append(log_entry)

        print(f"  Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Save latest
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(output_dir, 'checkpoint_latest.pth'))

        # Save log
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == '__main__':
    train_temporal(
        model_folder='/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d',
        sequence_root='/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_Extracted',
        pseudo_label_root='/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_PseudoLabels',
        output_dir='/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/TemporalUNet_v1',
        temporal_window=3,
        batch_size=4,
        num_epochs=200,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_temporal=0.1,
        device='cuda',
        num_workers=4,
    )
