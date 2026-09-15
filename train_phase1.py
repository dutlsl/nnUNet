"""
Phase 1 Training: Sparse-Label (Strategy A)
- DualStream dataloader: labeled (Semantic GT) + unlabeled (Sequence)
- SparseLabelLoss: DiceCE for labeled, EntMin+M2B for unlabeled
- Inference architecture: IDENTICAL to base ViViM+Mamba3

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_phase1.py --config configs/phase1_sparse_t5.yaml
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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config, save_config, ConfigDict
from utils.seed import set_seed
from utils.metrics import compute_dice_score
from datasets.dual_stream_dataset import get_dual_stream_dataloaders

# Compatibility patch
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = None
if not hasattr(torch, 'float4_e2m1fn_x2'):
    torch.float4_e2m1fn_x2 = None

from models.vivim_backbone import VivimBackbone
from losses.sparse_label_loss import SparseLabelLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Sparse-Label Training")
    parser.add_argument("--config", type=str, default="configs/phase1_sparse_t5.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--precision", type=str, default=None, choices=["fp16", "bf16", "fp32"],
                        help="Precision mode: fp16, bf16, or fp32 (default: from config or bf16)")
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler,
                    max_iters, cfg, epoch, amp_dtype=torch.bfloat16, use_amp=True):
    model.train()
    total_loss = 0.0
    total_sup = 0.0
    total_unsup = 0.0
    class_names = ['Background', 'Sclera', 'Iris', 'Pupil']
    class_dices = {n: [] for n in class_names}
    mean_dices = []
    iter_count = 0
    total_iters = min(len(dataloader), max_iters)

    from tqdm import tqdm
    pbar = tqdm(
        total=total_iters, desc=f"[E{epoch:03d}]", dynamic_ncols=True, leave=False,
        bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )

    for batch in dataloader:
        if iter_count >= max_iters:
            break

        images = batch['images'].to(device)
        targets = batch['label'].to(device)
        has_label = batch['has_label'].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp and (device.type == 'cuda')):
            logits = model(images)
            loss, l_sup, l_unsup = criterion(logits, targets, has_label, cfg=cfg, epoch=epoch)

        if torch.isnan(loss) or torch.isinf(loss):
            pbar.write(f"  [WARNING] NaN/Inf loss ({loss.item()}) at epoch {epoch}, iter {iter_count}! Skipping step.")
            optimizer.zero_grad()
            iter_count += 1
            pbar.update(1)
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_sup += l_sup.item()
        total_unsup += l_unsup.item()

        if has_label.any():
            lab_logits = logits[has_label.bool()]
            lab_targets = targets[has_label.bool()]
            dices = compute_dice_score(lab_logits, lab_targets, num_classes=cfg.model.num_classes)
            for n in class_names:
                class_dices[n].append(dices[n])
            mean_dices.append(dices['Mean_Dice'])

        iter_count += 1
        pbar.update(1)
        cur_d = np.mean(mean_dices[-20:]) if mean_dices else 0.0
        cur_s = np.mean(class_dices['Sclera'][-20:]) if class_dices['Sclera'] else 0.0
        cur_i = np.mean(class_dices['Iris'][-20:]) if class_dices['Iris'] else 0.0
        cur_p = np.mean(class_dices['Pupil'][-20:]) if class_dices['Pupil'] else 0.0
        pbar.set_postfix_str(f"Loss:{loss.item():.4f} | Pupil:{cur_p:.3f} Iris:{cur_i:.3f} Sclera:{cur_s:.3f} Mean:{cur_d:.3f}")

    pbar.close()
    n = max(1, iter_count)
    metrics = {name: float(np.mean(class_dices[name])) if class_dices[name] else 0.0
               for name in class_names}
    metrics['Mean_Dice'] = float(np.mean(mean_dices)) if mean_dices else 0.0
    return total_loss / n, total_sup / n, total_unsup / n, metrics


def validate(model, dataloader, device, max_iters, cfg, amp_dtype=torch.bfloat16, use_amp=True):
    model.eval()
    total_loss = 0.0
    class_names = ['Background', 'Sclera', 'Iris', 'Pupil']
    class_dices = {n: [] for n in class_names}
    mean_dices = []
    iter_count = 0

    dice_ce = nn.CrossEntropyLoss()
    total_val_iters = min(len(dataloader), max_iters)
    from tqdm import tqdm
    pbar = tqdm(
        total=total_val_iters, desc="  [Validation]", dynamic_ncols=True, leave=False,
        bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp and (device.type == 'cuda')):
            for batch in dataloader:
                if iter_count >= max_iters:
                    break

                images = batch['images'].to(device)
                targets = batch['label'].to(device)

                logits = model(images)
                loss = dice_ce(logits, targets)
                total_loss += loss.item()

            dices = compute_dice_score(logits, targets, num_classes=cfg.model.num_classes)
            for n in class_names:
                class_dices[n].append(dices[n])
            mean_dices.append(dices['Mean_Dice'])
            iter_count += 1
            pbar.update(1)
            cur_vd = np.mean(mean_dices) if mean_dices else 0.0
            pbar.set_postfix_str(f"Loss: {loss.item():.4f} | Val Mean Dice: {cur_vd:.4f}")

    pbar.close()
    n = max(1, iter_count)
    metrics = {name: float(np.mean(class_dices[name])) if class_dices[name] else 0.0
               for name in class_names}
    metrics['Mean_Dice'] = float(np.mean(mean_dices)) if mean_dices else 0.0
    return total_loss / n, metrics


def main():
    # Disable terminal auto-wrap so lines never staircase/overflow on resize
    sys.stdout.write("[?7l")
    sys.stdout.flush()
    args = parse_args()
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    print(f"[Phase1] Loaded config: {config_path}")

    seed = getattr(cfg.experiment, 'seed', 42)
    set_seed(seed)

    exp_name = getattr(cfg.experiment, 'name', 'phase1_sparse')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "nnUNet_results", f"{exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    save_config(cfg, os.path.join(output_dir, "config_snapshot.yaml"))
    gpu_id = args.gpu
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"[Phase1] Device: {device}")

    # Dataloaders
    print("[Phase1] Building DualStream dataloaders...")
    loaders = get_dual_stream_dataloaders(cfg)
    train_loader = loaders['train']
    val_loader = loaders['validation']

    # Model (identical architecture to base)
    print("[Phase1] Building ViViM model...")
    mamba_cfg = cfg.model.mamba
    model = VivimBackbone(
        in_channels=1, num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
        d_state=mamba_cfg.d_state, d_conv=mamba_cfg.d_conv, expand=mamba_cfg.expand,
        use_mamba=cfg.ablation.use_mamba,
        mamba_version=getattr(cfg.ablation, 'mamba_version', 'mamba3'),
        mamba_headdim=getattr(mamba_cfg, 'headdim', 64),
    ).to(device)

    # Load pretrained weights
    pretrained = getattr(cfg.training, 'pretrained_checkpoint', None)
    if pretrained and os.path.isfile(pretrained):
        print(f'[Phase1] Loading pretrained: {pretrained}')
        ckpt = torch.load(pretrained, map_location='cpu')
        sd = ckpt.get('network_weights', ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
        clean = {k.replace('backbone.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(clean, strict=False)
        print(f'[Phase1] Loaded (missing={len(missing)}, unexpected={len(unexpected)})')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Phase1] Params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                   weight_decay=cfg.training.weight_decay)

    best_val_dice = 0.0
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        print(f"[Phase1] Resuming from checkpoint: {args.resume}")
        res_ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(res_ckpt['model_state_dict'])
        if 'optimizer_state_dict' in res_ckpt:
            optimizer.load_state_dict(res_ckpt['optimizer_state_dict'])
        start_epoch = res_ckpt.get('epoch', 0) + 1
        best_val_dice = res_ckpt.get('val_dice', 0.0)
        best_ckpt = os.path.join(output_dir, "checkpoint_best.pth")
        if os.path.isfile(best_ckpt):
            try:
                b_data = torch.load(best_ckpt, map_location='cpu')
                best_val_dice = max(best_val_dice, b_data.get('val_dice', 0.0))
            except Exception:
                pass
        print(f"[Phase1] Resumed at Epoch {start_epoch}, Best Val Dice: {best_val_dice:.4f}")

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group.get('lr', cfg.training.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6,
        last_epoch=start_epoch - 1 if start_epoch > 1 else -1
    )
    criterion = SparseLabelLoss(num_classes=cfg.model.num_classes).to(device)

    precision = args.precision or getattr(cfg.training, 'precision', 'bf16')
    if precision in ('bf16', 'bfloat16'):
        amp_dtype = torch.bfloat16
        use_amp = True
        use_scaler = False
    elif precision in ('fp16', 'float16'):
        amp_dtype = torch.float16
        use_amp = True
        use_scaler = True
    else:
        amp_dtype = torch.float32
        use_amp = False
        use_scaler = False

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler and (device.type == 'cuda'))
    print(f"[Phase1] Precision: {precision} (amp_dtype={amp_dtype}, use_amp={use_amp}, scaler_enabled={scaler.is_enabled()})")

    # WandB
    use_wandb = False
    if hasattr(cfg, 'logging') and getattr(cfg.logging, 'wandb_project', None):
        try:
            import wandb
            run_name = f"{cfg.logging.wandb_run_prefix}_{timestamp}"
            wandb.init(project=cfg.logging.wandb_project, name=run_name, config=cfg.to_dict())
            use_wandb = True
            print(f"[Phase1] WandB: {run_name}")
        except Exception as e:
            print(f"[Phase1] WandB skipped: {e}")

    patience = getattr(cfg.training.early_stopping, 'patience', 50)
    patience_counter = 0
    max_train_iters = cfg.training.num_iterations_per_epoch
    max_val_iters = cfg.training.num_val_iterations_per_epoch

    print(f"[Phase1] Training from epoch {start_epoch} to {cfg.training.epochs}...")
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss, sup_loss, unsup_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            max_train_iters, cfg, epoch, amp_dtype=amp_dtype, use_amp=use_amp)
        val_loss, val_metrics = validate(model, val_loader, device, max_val_iters, cfg,
                                         amp_dtype=amp_dtype, use_amp=use_amp)
        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        vd = val_metrics['Mean_Dice']

        print(f"\n[Epoch {epoch:03d} Done | {elapsed:.1f}s | LR: {lr:.6f}]", flush=True)
        print(f"  train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}", flush=True)
        print(f"  Val Dice: {vd:.4f} (Pupil:{val_metrics['Pupil']:.4f} Iris:{val_metrics['Iris']:.4f} Sclera:{val_metrics['Sclera']:.4f})", flush=True)

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'train/loss': train_loss, 'train/loss_sup': sup_loss,
                       'train/loss_unsup': unsup_loss, 'train/dice': train_metrics['Mean_Dice'],
                       'val/loss': val_loss, 'val/dice': vd,
                       'val/sclera': val_metrics['Sclera'], 'val/iris': val_metrics['Iris'],
                       'val/pupil': val_metrics['Pupil'], 'lr': lr})

        
        with open("logs/phase1_sparse_t5.log", "a") as lf:
            lf.write(f"\n[Epoch {epoch:03d} Done | {elapsed:.1f}s | LR: {lr:.6f}]\n")
            lf.write(f"  train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}\n")
            lf.write(f"  Val Dice: {vd:.4f} (Pupil:{val_metrics['Pupil']:.4f} Iris:{val_metrics['Iris']:.4f} Sclera:{val_metrics['Sclera']:.4f})\n")
            if vd > best_val_dice:
                lf.write(f"  ★ New Best Val Dice: {vd:.4f}\n")
    
        if vd > best_val_dice:
            best_val_dice = vd
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': vd, 'val_metrics': val_metrics, 'config': cfg.to_dict()},
                       os.path.join(output_dir, "checkpoint_best.pth"))
            print(f"  ★ New Best Val Dice: {vd:.4f} (Saved checkpoint_best.pth)", flush=True)
        else:
            patience_counter += 1
        import shutil; cols = shutil.get_terminal_size(fallback=(80, 24)).columns; print("-" * max(20, min(cols - 2, 90)), flush=True)

        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': vd, 'config': cfg.to_dict()},
                       os.path.join(output_dir, "checkpoint_latest.pth"))

        if patience_counter >= patience:
            print(f"[Phase1] Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}\nPhase1 Done! Best Val Dice: {best_val_dice:.4f}\nOutput: {output_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
