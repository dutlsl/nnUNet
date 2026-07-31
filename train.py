"""
Unified Training Script for Eyeball-3D Research.

Single entrypoint for all experiment variants via YAML config:
  - Baseline: Vivim + DiceCE supervised segmentation
  - WeakMed Sphere: + Parametric Sphere Head + M2B + Sclera Containment

Usage:
  python train.py --config configs/weakmed_sphere.yaml
  python train.py --config configs/weakmed_sphere.yaml --resume path/to/checkpoint.pth
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Resolve project root for imports
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config, save_config, ConfigDict
from utils.seed import set_seed
from utils.metrics import compute_dice_score
from utils.sphere_metrics import compute_all_sphere_metrics
from datasets.openeds_dataset import get_openeds_sequence_dataloaders
from models.vivim_backbone import VivimBackbone
from losses.weakmed_loss import WeakMedSphereLoss


def _resolve_env_vars(cfg_dict: dict) -> dict:
    """Recursively resolve ${ENV_VAR} patterns in config string values."""
    resolved = {}
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            resolved[k] = _resolve_env_vars(v)
        elif isinstance(v, str) and '${' in v:
            for env_key in os.environ:
                v = v.replace(f'${{{env_key}}}', os.environ[env_key])
            resolved[k] = v
        else:
            resolved[k] = v
    return resolved


def build_model(cfg) -> nn.Module:
    """Build VivimBackbone with optional SphereHead from config."""
    use_sphere = getattr(cfg.ablation, 'use_sphere_head', False)
    sphere_cfg = cfg.model.sphere_head if use_sphere else None

    model = VivimBackbone(
        in_channels=1,
        num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
        d_state=cfg.model.mamba.d_state,
        d_conv=cfg.model.mamba.d_conv,
        expand=cfg.model.mamba.expand,
        use_mamba=cfg.ablation.use_mamba,
        use_sphere_head=use_sphere,
        sphere_head_cfg=sphere_cfg,
    )
    return model


def load_pretrained_seg_weights(model: nn.Module, ckpt_path: str):
    """Load pre-trained seg head weights, skip sphere head (random init)."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[Train] No pretrained checkpoint found at '{ckpt_path}', training from scratch.")
        return

    print(f"[Train] Loading pretrained seg weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Filter out sphere_head keys (we want random init for sphere head)
    filtered = {k: v for k, v in state_dict.items()
                if not k.startswith('sphere_head.')}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[Train] Loaded {len(filtered)} params. "
          f"Missing (expected for sphere_head): {len(missing)}. "
          f"Unexpected: {len(unexpected)}.")


def update_sharpness(model: nn.Module, epoch: int, cfg):
    """Anneal sphere head sharpness from initial to max over configured epochs."""
    if not hasattr(model, 'sphere_head'):
        return

    sphere_cfg = cfg.model.sphere_head
    k_init = sphere_cfg.sharpness
    k_max = getattr(sphere_cfg, 'sharpness_max', k_init)
    anneal_epochs = getattr(sphere_cfg, 'sharpness_anneal_epochs', 50)

    if anneal_epochs <= 0 or k_max <= k_init:
        return

    progress = min(1.0, epoch / anneal_epochs)
    k_current = k_init + (k_max - k_init) * progress
    model.sphere_head.set_sharpness(k_current)


def train_one_epoch(model, dataloader_iter, dataloader, loss_fn, optimizer, cfg, epoch, device):
    """Run one training epoch with fixed number of iterations (nnUNet schedule)."""
    model.train()
    total_losses = {}
    num_iterations = getattr(cfg.training, 'num_iterations_per_epoch', len(dataloader))

    for _ in range(num_iterations):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        images = batch['images'].to(device)        # [B, T, 1, H, W]
        label = batch['label'].to(device)           # [B, H, W]
        eyeball_bbox = batch['eyeball_bbox'].to(device)  # [B, 4]

        optimizer.zero_grad()

        # Forward
        model_output = model(images)

        # Compute combined loss
        loss_dict = loss_fn(
            model_output=model_output,
            label=label,
            eyeball_bbox=eyeball_bbox,
            cfg=cfg,
            current_epoch=epoch,
        )

        loss = loss_dict['total_loss']
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

    avg_losses = {k: v / max(num_iterations, 1) for k, v in total_losses.items()}
    return avg_losses, dataloader_iter


@torch.no_grad()
def validate(model, dataloader_iter, dataloader, loss_fn, cfg, epoch, device):
    """Run validation with fixed number of iterations (nnUNet schedule)."""
    model.eval()
    total_losses = {}
    all_dices = []
    all_sphere_metrics = []
    num_val_iterations = getattr(cfg.training, 'num_val_iterations_per_epoch', len(dataloader))

    for _ in range(num_val_iterations):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        images = batch['images'].to(device)
        label = batch['label'].to(device)
        eyeball_bbox = batch['eyeball_bbox'].to(device)
        eyeball_circle = batch['eyeball_circle'].to(device)  # [B, 3]

        model_output = model(images)

        # Loss
        loss_dict = loss_fn(
            model_output=model_output,
            label=label,
            eyeball_bbox=eyeball_bbox,
            cfg=cfg,
            current_epoch=epoch,
        )

        for k, v in loss_dict.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

        # Seg Dice metrics
        dice = compute_dice_score(model_output['seg_logits'], label,
                                  num_classes=cfg.model.num_classes)
        all_dices.append(dice)

        # Sphere metrics (if available)
        if 'sphere_params' in model_output:
            sphere_m = compute_all_sphere_metrics(
                model_output['sphere_params'], eyeball_circle)
            all_sphere_metrics.append(sphere_m)

    avg_losses = {k: v / max(num_val_iterations, 1) for k, v in total_losses.items()}

    avg_dice = {}
    if all_dices:
        for key in all_dices[0]:
            avg_dice[key] = sum(d[key] for d in all_dices) / len(all_dices)

    avg_sphere = {}
    if all_sphere_metrics:
        for key in all_sphere_metrics[0]:
            avg_sphere[key] = sum(m[key] for m in all_sphere_metrics) / len(all_sphere_metrics)

    return avg_losses, avg_dice, avg_sphere, dataloader_iter


def main():
    parser = argparse.ArgumentParser(description='Eyeball-3D Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load and resolve config
    cfg = load_config(args.config)
    cfg_dict = _resolve_env_vars(cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg))
    cfg = ConfigDict(cfg_dict)

    # Seed
    set_seed(cfg.experiment.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    # Output directory
    output_dir = Path(os.environ.get('nnUNet_results',
                      str(PROJECT_ROOT / 'nnUNet_results'))) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    save_config(cfg, str(output_dir / 'config_snapshot.yaml'))
    print(f"[Train] Config saved to: {output_dir / 'config_snapshot.yaml'}")

    # Data
    print("[Train] Building dataloaders...")
    loaders = get_openeds_sequence_dataloaders(cfg)
    train_loader = loaders.get('train')
    val_loader = loaders.get('validation')

    if train_loader is None:
        print("[Train] ERROR: No training data found!")
        sys.exit(1)

    print(f"[Train] Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader) if val_loader else 0}")

    # Model
    print("[Train] Building model...")
    model = build_model(cfg)
    model = model.to(device)

    # Load pretrained weights (seg head only)
    pretrained_path = getattr(cfg.training, 'pretrained_checkpoint', '')
    if pretrained_path:
        load_pretrained_seg_weights(model, pretrained_path)

    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"[Train] Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_metric = ckpt.get('best_metric', 0.0)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss
    loss_fn = WeakMedSphereLoss(num_classes=cfg.model.num_classes)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # WandB (optional)
    wandb_run = None
    try:
        import wandb
        run_name = f"{cfg.logging.wandb_run_prefix}_{cfg.experiment.name}_{int(time.time())}"
        wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            name=run_name,
            config=cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg),
        )
        print(f"[Train] WandB initialized: {run_name}")
    except (ImportError, Exception) as e:
        print(f"[Train] WandB not available ({e}), logging to stdout only.")

    # Training loop
    patience_counter = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_start = time.time()

        # Update sharpness annealing
        update_sharpness(model, epoch, cfg)

        # Train
        train_losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, cfg, epoch, device)

        # Validate
        val_losses, val_dice, val_sphere = {}, {}, {}
        if val_loader is not None:
            val_losses, val_dice, val_sphere = validate(
                model, val_loader, loss_fn, cfg, epoch, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print summary
        lr_current = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch + 1}/{cfg.training.epochs}] "
              f"Time: {epoch_time:.1f}s | LR: {lr_current:.6f}")
        print(f"  Train Loss: {train_losses.get('total_loss', 0):.4f} "
              f"(seg={train_losses.get('seg_loss', 0):.4f}, "
              f"m2b={train_losses.get('m2b_loss', 0):.4f}, "
              f"contain={train_losses.get('containment_loss', 0):.4f})")

        if val_dice:
            print(f"  Val Dice: Sclera={val_dice.get('Sclera', 0):.4f}, "
                  f"Iris={val_dice.get('Iris', 0):.4f}, "
                  f"Pupil={val_dice.get('Pupil', 0):.4f}, "
                  f"Mean={val_dice.get('Mean_Dice', 0):.4f}")

        if val_sphere:
            print(f"  Val Sphere: IoU={val_sphere.get('circle_iou_mean', 0):.4f}, "
                  f"CenterErr={val_sphere.get('center_error_mean', 0):.1f}px, "
                  f"RadiusErr={val_sphere.get('radius_error_rel_mean', 0):.4f}")

        # WandB logging
        if wandb_run:
            log_dict = {}
            for k, v in train_losses.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_losses.items():
                log_dict[f'val/{k}'] = v
            for k, v in val_dice.items():
                log_dict[f'val/dice_{k}'] = v
            for k, v in val_sphere.items():
                log_dict[f'val/{k}'] = v
            log_dict['lr'] = lr_current
            if hasattr(model, 'sphere_head'):
                log_dict['sharpness'] = model.sphere_head.renderer.sharpness
            wandb_run.log(log_dict, step=epoch)

        # Best model checkpoint (by mean foreground dice)
        current_metric = val_dice.get('Mean_Dice', 0.0)
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'val_dice': val_dice,
                'val_sphere': val_sphere,
                'config': cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg),
            }
            ckpt_path = output_dir / 'checkpoint_best.pth'
            torch.save(ckpt, str(ckpt_path))
            print(f"  ★ New best! Mean Dice={best_metric:.4f} → saved to {ckpt_path}")
        else:
            patience_counter += 1

        # Early stopping
        if (epoch >= cfg.training.early_stopping.min_epochs
                and patience_counter >= cfg.training.early_stopping.patience):
            print(f"\n[Train] Early stopping at epoch {epoch + 1} "
                  f"(patience={cfg.training.early_stopping.patience})")
            break

    print(f"\n[Train] Training complete! Best Mean Dice: {best_metric:.4f}")
    print(f"[Train] Best checkpoint: {output_dir / 'checkpoint_best.pth'}")

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
