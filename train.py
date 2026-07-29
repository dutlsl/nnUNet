"""
Main Training Entrypoint for Eyeball 3D Research (weakMed Branch).
Strictly complies with the Research Code Generation Harness.

Usage:
    python train.py --config configs/weakmed.yaml
"""

import os
import sys
import time
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import wandb

# Insert project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from utils.seed import set_seed
from utils.config import load_config, save_config
from utils.metrics import compute_dice_score
from datasets.openeds_dataset import get_openeds_sequence_dataloaders
from models.vivim_backbone import VivimBackbone
from losses.weakmed_loss import WeakMedLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Eyeball 3D Research Training Entrypoint")
    parser.add_argument("--config", type=str, default="configs/weakmed.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


def build_model(cfg):
    backbone_type = cfg.model.backbone.lower()
    if backbone_type == "vivim":
        print(f"[Model] Initializing Vivim Backbone (Mamba: {cfg.ablation.use_mamba})...")
        model = VivimBackbone(
            in_channels=1,
            num_classes=cfg.model.num_classes,
            base_channels=cfg.model.base_channels,
            d_state=cfg.model.mamba.d_state,
            d_conv=cfg.model.mamba.d_conv,
            expand=cfg.model.mamba.expand,
            use_mamba=cfg.ablation.use_mamba,
        )
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")
    return model


def main():
    args = parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    cfg = load_config(args.config)

    # 2. Fix Random Seed for Reproducibility
    set_seed(cfg.experiment.seed)

    output_dir = os.path.join("nnUNet_results", cfg.experiment.name)
    os.makedirs(output_dir, exist_ok=True)

    # Save Config Snapshot
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    save_config(cfg, snapshot_path)

    # 3. Initialize WandB
    run_name = f"{cfg.logging.wandb_run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=cfg.logging.wandb_project,
        name=run_name,
        config=cfg.to_dict(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Training: {cfg.experiment.name} ===")
    print(f"Device: {device}")
    print(f"Config Snapshot Saved: {snapshot_path}")

    # 4. Dataloaders
    loaders = get_openeds_sequence_dataloaders(cfg)
    train_loader = loaders.get("train", None)
    val_loader = loaders.get("validation", None)

    if train_loader is None or len(train_loader) == 0:
        print("[Warning] Train DataLoader is empty or no valid samples found. Check dataset paths.")
        return

    # 5. Build Model & Loss
    model = build_model(cfg).to(device)
    loss_fn = WeakMedLoss(num_classes=cfg.model.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

    best_val_dice = 0.0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)  # [B, T, 1, 448, 448]
            labels = batch["label"].to(device)   # [B, 448, 448]

            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, labels, cfg=cfg)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_dices = []
        if val_loader is not None and len(val_loader) > 0:
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["images"].to(device)
                    labels = batch["label"].to(device)
                    preds = model(images)
                    metrics = compute_dice_score(preds, labels)
                    val_dices.append(metrics["Mean_Dice"])
            val_mean_dice = float(sum(val_dices) / len(val_dices))
        else:
            val_mean_dice = 0.0

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:03d}/{cfg.training.epochs:03d}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Val Mean Dice: {val_mean_dice:.4f} | LR: {current_lr:.6f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mean_dice": val_mean_dice,
            "learning_rate": current_lr,
            "epoch_duration_sec": elapsed,
        })

        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            ckpt_path = os.path.join(output_dir, "checkpoint_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mean_dice": val_mean_dice,
                "config": cfg.to_dict(),
            }, ckpt_path)
            print(f"  -> Best model saved! (Val Mean Dice: {val_mean_dice:.4f})")

    wandb.finish()
    print("\n=== Training Completed Successfully! ===")


if __name__ == "__main__":
    main()
