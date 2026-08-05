"""
Main training entrypoint for ViViM + Mamba3 research experiments.
Strictly complies with Research Code Generation Harness (.agents/skills/research-code-harness/SKILL.md).

Features:
- Config-driven (loads YAML via --config)
- Seed reproducibility (random, numpy, torch, cuda)
- Config snapshot & Best checkpoint saving
- WandB integration
- Mixed precision training (AMP)

Usage:
    python train.py --config configs/vivim_mamba3_192.yaml
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config, save_config, ConfigDict
from utils.seed import set_seed
from utils.metrics import compute_dice_score
from datasets.openeds_dataset import get_openeds_sequence_dataloaders
from models.vivim_backbone import VivimBackbone
from losses.weakmed_loss import WeakMedLoss


def parse_args():
    parser = argparse.ArgumentParser(description="ViViM Mamba3 Segmentation Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vivim_mamba3_192.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints and logs (defaults to nnUNet_results/<exp_name>)",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    max_iters: int,
    cfg: ConfigDict,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_dices = []

    iter_count = 0
    pbar = tqdm(dataloader, total=min(len(dataloader), max_iters), desc="Train Epoch", leave=False)

    for batch in pbar:
        if iter_count >= max_iters:
            break

        images = batch['images'].to(device)  # [B, T, 1, H, W]
        targets = batch['label'].to(device)  # [B, H, W]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(images)           # [B, Num_Classes, H, W]
            loss = criterion(logits, targets, cfg=cfg)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        dices = compute_dice_score(logits.detach(), targets, num_classes=cfg.model.num_classes)
        all_dices.append(dices['Mean_Dice'])

        iter_count += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{dices['Mean_Dice']:.4f}"})

    mean_loss = total_loss / max(1, iter_count)
    mean_dice = float(np.mean(all_dices)) if all_dices else 0.0
    return mean_loss, {'Mean_Dice': mean_dice}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_iters: int,
    cfg: ConfigDict,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    class_names = ['Background', 'Sclera', 'Iris', 'Pupil']
    class_dices = {name: [] for name in class_names}
    mean_dices = []

    iter_count = 0
    pbar = tqdm(dataloader, total=min(len(dataloader), max_iters), desc="Val Epoch", leave=False)

    for batch in pbar:
        if iter_count >= max_iters:
            break

        images = batch['images'].to(device)
        targets = batch['label'].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, targets, cfg=cfg)

        total_loss += loss.item()
        dices = compute_dice_score(logits, targets, num_classes=cfg.model.num_classes)

        for name in class_names:
            class_dices[name].append(dices[name])
        mean_dices.append(dices['Mean_Dice'])

        iter_count += 1

    avg_loss = total_loss / max(1, iter_count)
    metrics = {name: float(np.mean(class_dices[name])) for name in class_names}
    metrics['Mean_Dice'] = float(np.mean(mean_dices)) if mean_dices else 0.0
    return avg_loss, metrics


def main():
    args = parse_args()

    # 1. Load Configuration
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    print(f"[Train] Loaded config: {config_path}")

    # 2. Fix Random Seed
    seed = getattr(cfg.experiment, 'seed', 42)
    set_seed(seed)

    # 3. Setup Output Directory
    exp_name = getattr(cfg.experiment, 'name', 'vivim_mamba3_192')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(PROJECT_ROOT, "nnUNet_results", f"{exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save Config Snapshot
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    save_config(cfg, snapshot_path)
    print(f"[Train] Saved config snapshot to: {snapshot_path}")

    # 4. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")

    # 5. Dataloaders
    print("[Train] Initializing dataloaders...")
    dataloaders = get_openeds_sequence_dataloaders(cfg)
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('validation', dataloaders.get('val'))

    # 6. Model
    print("[Train] Building ViViM model...")
    mamba_cfg = cfg.model.mamba
    mamba_version = getattr(cfg.ablation, 'mamba_version', 'mamba3')
    model = VivimBackbone(
        in_channels=1,
        num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
        d_state=mamba_cfg.d_state,
        d_conv=mamba_cfg.d_conv,
        expand=mamba_cfg.expand,
        use_mamba=cfg.ablation.use_mamba,
        mamba_version=mamba_version,
        mamba_headdim=getattr(mamba_cfg, 'headdim', 64),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model trainable parameters: {total_params:,}")

    # 7. Optimizer & Scheduler & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=1e-6,
    )

    criterion = WeakMedLoss(num_classes=cfg.model.num_classes).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 8. WandB Logging (optional)
    use_wandb = False
    if hasattr(cfg, 'logging') and getattr(cfg.logging, 'wandb_project', None):
        try:
            import wandb
            run_name = f"{cfg.logging.wandb_run_prefix}_{timestamp}"
            wandb.init(
                project=cfg.logging.wandb_project,
                name=run_name,
                config=cfg.to_dict(),
            )
            use_wandb = True
            print(f"[Train] WandB initialized: {run_name}")
        except Exception as e:
            print(f"[Train] WandB initialization skipped: {e}")

    # 9. Training Loop
    best_val_dice = 0.0
    patience = getattr(cfg.training.early_stopping, 'patience', 100)
    patience_counter = 0

    max_train_iters = cfg.training.num_iterations_per_epoch
    max_val_iters = cfg.training.num_val_iterations_per_epoch

    print(f"[Train] Starting training for {cfg.training.epochs} epochs...")

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.time()

        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            max_iters=max_train_iters,
            cfg=cfg,
        )

        val_loss, val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_iters=max_val_iters,
            cfg=cfg,
        )

        scheduler.step()
        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        val_dice = val_metrics['Mean_Dice']

        # Console Log
        print(
            f"Epoch {epoch:04d}/{cfg.training.epochs:04d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} (Dice: {train_metrics['Mean_Dice']:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} "
            f"(Sclera: {val_metrics['Sclera']:.4f}, Iris: {val_metrics['Iris']:.4f}, Pupil: {val_metrics['Pupil']:.4f}) | "
            f"LR: {current_lr:.6f}"
        )

        # WandB Log
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/dice': train_metrics['Mean_Dice'],
                'val/loss': val_loss,
                'val/dice': val_dice,
                'val/sclera_dice': val_metrics['Sclera'],
                'val/iris_dice': val_metrics['Iris'],
                'val/pupil_dice': val_metrics['Pupil'],
                'learning_rate': current_lr,
            })

        # Save Best Checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            best_ckpt_path = os.path.join(output_dir, "checkpoint_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_metrics': val_metrics,
                'config': cfg.to_dict(),
            }, best_ckpt_path)
            print(f"  ★ New best Val Mean Dice: {val_dice:.4f}! Saved to {best_ckpt_path}")
        else:
            patience_counter += 1

        # Periodic Latest Checkpoint
        if epoch % 50 == 0:
            latest_ckpt_path = os.path.join(output_dir, "checkpoint_latest.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'config': cfg.to_dict(),
            }, latest_ckpt_path)

        if patience_counter >= patience:
            print(f"[Train] Early stopping triggered after {patience} epochs without improvement.")
            break

    print(f"\n{'='*60}")
    print(f"Training Complete! Best Val Mean Dice: {best_val_dice:.4f}")
    print(f"Results saved to: {output_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
