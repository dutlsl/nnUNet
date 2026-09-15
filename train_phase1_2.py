"""
Phase 1+2 Hybrid Training: DualStream + Multi-Frame Decoding (Strategy E = A + D)

Combines:
  - Phase 1 (Strategy A): DualStream labeled/unlabeled mixing, no pseudo-labels
  - Phase 2 (Strategy D): Multi-Frame decoding during training, temporal smoothness

Key design:
  - Training: VivimBackboneMultiFrame with output_all_frames=True → [B, T, C, H, W]
  - Inference: output_all_frames=False → [B, C, H, W] (identical to base, zero overhead)
  - DualStream data: labeled samples have label on last frame only, unlabeled have none
  - MultiFrameMaskedLoss: DiceCE on labeled frames + temporal smoothness + M2B on all

Usage:
    python train_phase1_2.py --config configs/phase1_2_hybrid_t5.yaml --gpu 1 --precision bf16
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config, save_config, ConfigDict
from utils.seed import set_seed
from utils.metrics import compute_dice_score
from datasets.dual_stream_dataset import (
    SemanticGTSequenceDataset, UnlabeledSequenceDataset,
    ConcatDualStreamDataset, DualStreamBatchSampler,
)

# PyTorch compatibility patch
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = None
if not hasattr(torch, 'float4_e2m1fn_x2'):
    torch.float4_e2m1fn_x2 = None

from models.vivim_backbone_multiframe import VivimBackboneMultiFrame
from losses.multiframe_loss import MultiFrameMaskedLoss


# ---------------------------------------------------------------------------
# Multi-Frame Adapter: converts DualStream single-label output to multi-frame
# ---------------------------------------------------------------------------
class MultiFrameAdapter(Dataset):
    """
    Wraps a DualStream dataset (SemanticGT + Unlabeled concat) to produce
    multi-frame format compatible with VivimBackboneMultiFrame + MultiFrameMaskedLoss.

    Input format  (from DualStream):
        images:    [T, 1, H, W]
        label:     [H, W]        (single label for last frame, or zeros)
        has_label: bool

    Output format (for multi-frame training):
        images:     [T, 1, H, W]
        labels:     [T, H, W]     (label at last frame if has_label, else all zeros)
        label_mask: [T]           ([0,...,0,1] if has_label, [0,...,0,0] if not)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        images = sample['images']       # [T, 1, H, W]
        label = sample['label']         # [H, W]
        has_label = sample['has_label'] # bool

        T = images.shape[0]
        H, W = label.shape

        labels = torch.zeros(T, H, W, dtype=torch.long)
        label_mask = torch.zeros(T, dtype=torch.float32)

        if has_label:
            labels[-1] = label       # Last frame carries the GT
            label_mask[-1] = 1.0

        return {
            'images': images,         # [T, 1, H, W]
            'labels': labels,         # [T, H, W]
            'label_mask': label_mask,  # [T]
        }


# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1+2 Hybrid: DualStream + Multi-Frame")
    parser.add_argument("--config", type=str, default="configs/phase1_2_hybrid_t5.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=1, help="CUDA device index")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--precision", type=str, default=None, choices=["bf16", "fp16", "fp32"],
                        help="Floating point precision")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler,
                    max_iters, cfg, epoch, amp_dtype=torch.bfloat16, use_amp=True):
    model.train()
    total_loss = 0.0
    total_sup = 0.0
    total_smooth = 0.0
    class_names = ['Background', 'Sclera', 'Iris', 'Pupil']
    class_dices = {n: [] for n in class_names}
    mean_dices = []
    iter_count = 0
    n_nan_skipped = 0
    total_iters = min(len(dataloader), max_iters)

    pbar = tqdm(
        total=total_iters, desc=f"[E{epoch:03d}]", dynamic_ncols=True, leave=False,
        bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )

    for batch in dataloader:
        if iter_count >= max_iters:
            break

        images = batch['images'].to(device)       # [B, T, 1, H, W]
        targets = batch['labels'].to(device)       # [B, T, H, W]
        label_mask = batch['label_mask'].to(device) # [B, T]

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp and (device.type == 'cuda')):
            logits = model(images, output_all_frames=True)  # [B, T, C, H, W]
            loss, l_sup, l_smooth = criterion(logits, targets, label_mask, cfg=cfg, epoch=epoch)

        if torch.isnan(loss) or torch.isinf(loss):
            pbar.write(f"  [WARNING] NaN/Inf loss at epoch {epoch}, iter {iter_count}! Skipping step.")
            optimizer.zero_grad()
            n_nan_skipped += 1
            iter_count += 1
            pbar.update(1)
            continue

        max_grad_norm = getattr(cfg.training, 'max_grad_norm', 1.0)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        total_sup += l_sup.item()
        total_smooth += l_smooth.item()

        # Dice on labeled frames only
        mask = label_mask.bool()
        if mask.any():
            lab_logits = logits[mask]
            lab_targets = targets[mask]
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
        pbar.set_postfix_str(f"Loss:{loss.item():.4f} | P:{cur_p:.3f} I:{cur_i:.3f} S:{cur_s:.3f} D:{cur_d:.3f}")

    pbar.close()
    if n_nan_skipped > 0:
        print(f"  [INFO] Skipped {n_nan_skipped} NaN/Inf iterations in epoch {epoch}")

    n = max(1, iter_count - n_nan_skipped)
    metrics = {name: float(np.mean(class_dices[name])) if class_dices[name] else 0.0
               for name in class_names}
    metrics['Mean_Dice'] = float(np.mean(mean_dices)) if mean_dices else 0.0
    return total_loss / n, total_sup / n, total_smooth / n, metrics


# ---------------------------------------------------------------------------
# Validation (single-frame, identical to inference)
# ---------------------------------------------------------------------------
def validate(model, dataloader, device, max_iters, cfg,
             amp_dtype=torch.bfloat16, use_amp=True):
    model.eval()
    total_loss = 0.0
    class_names = ['Background', 'Sclera', 'Iris', 'Pupil']
    class_dices = {n: [] for n in class_names}
    mean_dices = []
    iter_count = 0

    dice_ce = nn.CrossEntropyLoss()
    total_val_iters = min(len(dataloader), max_iters)
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

                # Inference mode: only last frame decoded
                logits = model(images, output_all_frames=False)
                loss = dice_ce(logits, targets)
                total_loss += loss.item()

                dices = compute_dice_score(logits, targets, num_classes=cfg.model.num_classes)
                for n in class_names:
                    class_dices[n].append(dices[n])
                mean_dices.append(dices['Mean_Dice'])
                iter_count += 1
                pbar.update(1)
                cur_vd = np.mean(mean_dices) if mean_dices else 0.0
                pbar.set_postfix_str(f"Loss: {loss.item():.4f} | Val Dice: {cur_vd:.4f}")

    pbar.close()
    n = max(1, iter_count)
    metrics = {name: float(np.mean(class_dices[name])) if class_dices[name] else 0.0
               for name in class_names}
    metrics['Mean_Dice'] = float(np.mean(mean_dices)) if mean_dices else 0.0
    return total_loss / n, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sys.stdout.write("")
    sys.stdout.flush()
    args = parse_args()
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    print(f"[Hybrid] Loaded config: {config_path}")

    seed = getattr(cfg.experiment, 'seed', 42)
    set_seed(seed)

    exp_name = getattr(cfg.experiment, 'name', 'phase1_2_hybrid')
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
    print(f"[Hybrid] Device: {device}")

    # ---- Data Loading: DualStream → MultiFrameAdapter ----
    sem_root = cfg.data.semantic_gt_root
    seq_root = cfg.data.sequence_root
    T = cfg.data.temporal_window
    bs = cfg.training.batch_size
    n_lab = getattr(cfg.training, 'n_labeled_per_batch', max(1, bs // 4))
    nw = cfg.training.num_workers
    crop = tuple(cfg.data.input_crop)
    pad_res = tuple(cfg.data.target_resolution)
    stride = getattr(cfg.data, 'unlabeled_stride', 3)

    print("[Hybrid] Building DualStream + MultiFrame training dataloaders...")
    lab_train = SemanticGTSequenceDataset(
        os.path.join(sem_root, 'train', 'images'),
        os.path.join(sem_root, 'train', 'labels'),
        os.path.join(seq_root, 'train'),
        temporal_window=T, crop=crop, padded_resolution=pad_res,
    )
    unlab_train = UnlabeledSequenceDataset(
        os.path.join(seq_root, 'train'),
        temporal_window=T, crop=crop, padded_resolution=pad_res, stride=stride,
    )
    combined = ConcatDualStreamDataset(lab_train, unlab_train)
    adapted = MultiFrameAdapter(combined)

    sampler = DualStreamBatchSampler(len(lab_train), len(unlab_train), n_lab, bs)
    train_loader = DataLoader(adapted, batch_sampler=sampler, num_workers=nw, pin_memory=True)

    print(f"[Hybrid] Train: {len(lab_train)} labeled + {len(unlab_train)} unlabeled = {len(combined)} total")
    print(f"[Hybrid] Batch: {bs} (labeled={n_lab}, unlabeled={bs - n_lab})")

    # Validation: labeled-only, single-frame output (uses SemanticGTSequenceDataset directly)
    print("[Hybrid] Building validation dataloader...")
    lab_val = SemanticGTSequenceDataset(
        os.path.join(sem_root, 'validation', 'images'),
        os.path.join(sem_root, 'validation', 'labels'),
        os.path.join(seq_root, 'validation'),
        temporal_window=T, crop=crop, padded_resolution=pad_res,
    )
    val_loader = DataLoader(lab_val, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=True, drop_last=False)

    # ---- Model ----
    print("[Hybrid] Building VivimBackboneMultiFrame model...")
    mamba_cfg = cfg.model.mamba
    model = VivimBackboneMultiFrame(
        in_channels=1, num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
        d_state=mamba_cfg.d_state, d_conv=mamba_cfg.d_conv, expand=mamba_cfg.expand,
        use_mamba=cfg.ablation.use_mamba,
        mamba_version=getattr(cfg.ablation, 'mamba_version', 'mamba3'),
        mamba_headdim=getattr(mamba_cfg, 'headdim', 64),
    ).to(device)

    # Load pretrained checkpoint (supports both Phase 1 and fold_1 formats)
    pretrained = getattr(cfg.training, 'pretrained_checkpoint', None)
    if pretrained and os.path.isfile(pretrained):
        print(f'[Hybrid] Loading pretrained: {pretrained}')
        ckpt = torch.load(pretrained, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt.get('network_weights',
             ckpt.get('state_dict', ckpt)))
        clean = {k.replace('backbone.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(clean, strict=False)
        print(f'[Hybrid] Loaded (missing={len(missing)}, unexpected={len(unexpected)})')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Hybrid] Params: {total_params:,}")

    # ---- Optimizer, Scheduler, Criterion ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                   weight_decay=cfg.training.weight_decay)

    best_val_dice = 0.0
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        print(f"[Hybrid] Resuming from checkpoint: {args.resume}")
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
        print(f"[Hybrid] Resumed at Epoch {start_epoch}, Best Val Dice: {best_val_dice:.4f}")

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group.get('lr', cfg.training.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6,
        last_epoch=start_epoch - 1 if start_epoch > 1 else -1
    )
    criterion = MultiFrameMaskedLoss(num_classes=cfg.model.num_classes).to(device)

    # ---- Precision ----
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
    print(f"[Hybrid] Precision: {precision} (amp_dtype={amp_dtype}, use_amp={use_amp}, "
          f"scaler_enabled={scaler.is_enabled()})")

    # ---- WandB ----
    use_wandb = False
    if hasattr(cfg, 'logging') and getattr(cfg.logging, 'wandb_project', None):
        try:
            import wandb
            run_name = f"{cfg.logging.wandb_run_prefix}_{timestamp}"
            wandb.init(project=cfg.logging.wandb_project, name=run_name, config=cfg.to_dict())
            use_wandb = True
            print(f"[Hybrid] WandB: {run_name}")
        except Exception as e:
            print(f"[Hybrid] WandB skipped: {e}")

    patience = getattr(cfg.training.early_stopping, 'patience', 500)
    patience_counter = 0
    max_train_iters = cfg.training.num_iterations_per_epoch
    max_val_iters = cfg.training.num_val_iterations_per_epoch

    log_file = "logs/phase1_2_hybrid_t5.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # ---- Training Loop ----
    print(f"[Hybrid] Training from epoch {start_epoch} to {cfg.training.epochs}...")
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss, sup_loss, smooth_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            max_train_iters, cfg, epoch, amp_dtype=amp_dtype, use_amp=use_amp)
        val_loss, val_metrics = validate(model, val_loader, device, max_val_iters, cfg,
                                         amp_dtype=amp_dtype, use_amp=use_amp)
        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        vd = val_metrics['Mean_Dice']

        print(f"\n[Epoch {epoch:03d} Done | {elapsed:.1f}s | LR: {lr:.6f}]", flush=True)
        print(f"  train_loss: {train_loss:.4f} (sup:{sup_loss:.4f} smooth:{smooth_loss:.4f}) "
              f"| val_loss: {val_loss:.4f}", flush=True)
        print(f"  Val Dice: {vd:.4f} (Pupil:{val_metrics['Pupil']:.4f} "
              f"Iris:{val_metrics['Iris']:.4f} Sclera:{val_metrics['Sclera']:.4f})", flush=True)

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'train/loss': train_loss, 'train/loss_sup': sup_loss,
                       'train/loss_smooth': smooth_loss, 'train/dice': train_metrics['Mean_Dice'],
                       'val/loss': val_loss, 'val/dice': vd,
                       'val/sclera': val_metrics['Sclera'], 'val/iris': val_metrics['Iris'],
                       'val/pupil': val_metrics['Pupil'], 'lr': lr})

        with open(log_file, "a") as lf:
            lf.write(f"\n[Epoch {epoch:03d} Done | {elapsed:.1f}s | LR: {lr:.6f}]\n")
            lf.write(f"  train_loss: {train_loss:.4f} (sup:{sup_loss:.4f} smooth:{smooth_loss:.4f}) "
                     f"| val_loss: {val_loss:.4f}\n")
            lf.write(f"  Val Dice: {vd:.4f} (Pupil:{val_metrics['Pupil']:.4f} "
                     f"Iris:{val_metrics['Iris']:.4f} Sclera:{val_metrics['Sclera']:.4f})\n")
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
        import shutil
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        print("-" * max(20, min(cols - 2, 90)), flush=True)

        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': vd, 'config': cfg.to_dict()},
                       os.path.join(output_dir, "checkpoint_latest.pth"))

        if patience_counter >= patience:
            print(f"[Hybrid] Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}\nHybrid Done! Best Val Dice: {best_val_dice:.4f}\nOutput: {output_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
