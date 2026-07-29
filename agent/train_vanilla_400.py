"""
Dynamic & Unbounded Pixel-Precision Training Script for 400x400 Native Vanilla nnUNet (PlainConvUNet).

Pixel-Level Precision Optimization Strategy:
1. 100% On-the-fly cropping [120:520, 0:400] in RAM.
2. min_epochs = 100: Guarantees at least 100 epochs of training before Early Stopping can trigger.
3. ReduceLROnPlateau: Dynamically decays LR by 0.5 whenever val_loss plateaus for 15 epochs, enabling subpixel edge refinement.
4. Conservative Early Stopping (patience = 60): Triggers ONLY after min_epochs and ONLY when val_loss has truly saturated across 60 consecutive epochs.
5. max_epochs = 2000: Unbounded safety ceiling so training is never artificially truncated.
"""

import os
import sys
import glob
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
import wandb

sys.path.insert(0, os.path.dirname(__file__))

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from batchgenerators.utilities.file_and_folder_operations import load_json


# ─── 400x400 On-The-Fly Dataset ───

class OpenEDS400OnTheFlyDataset(Dataset):
    """
    Dataset that loads 640x400 OpenEDS PNG images and NPY labels,
    and applies on-the-fly cropping [120:520, 0:400] in RAM.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        mean: float = 86.45,
        std: float = 39.94,
        target_size: int = 448,  # Pad 400->448 for clean 64-stride divisions
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.target_size = target_size
        
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.samples = []
        for img_path in self.image_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(label_dir, f"{basename}.npy")
            if os.path.exists(lbl_path):
                self.samples.append((img_path, lbl_path))

        print(f"OpenEDS400OnTheFlyDataset: Loaded {len(self.samples)} pairs from {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # Load original 640x400 image (Height=640, Width=400)
        img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        lbl = np.load(lbl_path).astype(np.int64)

        # On-The-Fly Crop [120:520, 0:400] -> (Height=400, Width=400)
        img_crop = img[120:520, 0:400]
        lbl_crop = lbl[120:520, 0:400]

        # Z-Score Normalization
        img_norm = (img_crop - self.mean) / (self.std + 1e-8)
        img_tensor = img_norm[None, :, :]  # [1, 400, 400]

        # Zero-pad from (400, 400) to (448, 448) for clean stride divisions
        pad_size = self.target_size - 400  # 48
        pad_top = pad_size // 2            # 24
        pad_bottom = pad_size - pad_top    # 24
        pad_left = pad_size // 2           # 24
        pad_right = pad_size - pad_left    # 24

        img_padded = np.pad(
            img_tensor,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

        lbl_padded = np.pad(
            lbl_crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

        return {
            'image': torch.from_numpy(img_padded).float(),  # [1, 448, 448]
            'label': torch.from_numpy(lbl_padded).long(),    # [448, 448]
        }


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
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.dice(logits, target) + self.ce(logits, target)


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


# ─── Main Training Script ───

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/VanillaUNet_400x400'
    os.makedirs(output_dir, exist_ok=True)

    min_epochs = 100
    max_epochs = 2000
    patience = 60
    batch_size = 16
    lr = 1e-3

    # Initialize Weights & Biases (wandb)
    wandb.init(
        project="nnUNet_400x400",
        name="VanillaUNet_PlainConvUNet_400x400_DynamicPlateau_Min100Epochs",
        config={
            "architecture": "PlainConvUNet_2D",
            "input_resolution": "400x400 (On-The-Fly Crop)",
            "padded_resolution": "448x448",
            "batch_size": batch_size,
            "learning_rate": lr,
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "patience": patience,
            "lr_scheduler": "ReduceLROnPlateau (factor=0.5, patience=15)",
            "num_classes": 4,
            "crop_slice": "[120:520, 0:400]",
        }
    )

    print(f"=== Starting Unbounded Dynamic 400x400 Vanilla nnUNet Training ===")
    print(f"Device: {device}")
    print(f"Min Epochs: {min_epochs} | Max Epochs Ceiling: {max_epochs} | Early Stopping Patience: {patience}")
    print(f"Output Directory: {output_dir}")

    # Load nnUNet 2D architecture specs
    plans_path = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d/plans.json'
    plans = load_json(plans_path)
    arch = plans['configurations']['2d']['architecture']

    # Instantiate 2D PlainConvUNet
    model = get_network_from_plans(
        arch['network_class_name'],
        arch['arch_kwargs'],
        arch['_kw_requires_import'],
        1,  # input_channels
        4,  # num_classes
        allow_init=True,
        deep_supervision=True
    )
    model.to(device)

    # Datasets and Loaders
    openeds_base = '/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Semantic_Segmentation_Dataset'
    train_ds = OpenEDS400OnTheFlyDataset(
        image_dir=os.path.join(openeds_base, 'train', 'images'),
        label_dir=os.path.join(openeds_base, 'train', 'labels'),
    )
    val_ds = OpenEDS400OnTheFlyDataset(
        image_dir=os.path.join(openeds_base, 'validation', 'images'),
        label_dir=os.path.join(openeds_base, 'validation', 'labels'),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Loss, Optimizer, Dynamic Scheduler, Scaler
    criterion = DeepSupervisionWrapper(DiceCELoss(num_classes=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                preds = model(images)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                with autocast('cuda'):
                    preds = model(images)
                    loss = criterion(preds, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:04d}/{max_epochs:04d}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_duration_sec": elapsed,
            "epochs_no_improve": epochs_no_improve,
        })

        # Save Checkpoint & Check Early Stopping
        checkpoint_latest = os.path.join(output_dir, 'checkpoint_latest.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'val_loss': val_loss}, checkpoint_latest)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_best = os.path.join(output_dir, 'checkpoint_best.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'val_loss': val_loss}, checkpoint_best)
            print(f"  -> Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement in Val Loss ({epochs_no_improve}/{patience})")

        # Early Stopping checks min_epochs constraint
        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"\n[Dynamic Early Stopping Triggered at Epoch {epoch}]")
            print(f"Guaranteed {min_epochs} min_epochs passed. Val Loss has not improved for {patience} consecutive epochs.")
            print(f"Best Val Loss achieved: {best_val_loss:.4f} at Epoch {epoch - patience}")
            break

    wandb.finish()
    print("\n=== Unbounded Dynamic Training Completed Successfully! ===")


if __name__ == '__main__':
    train()
