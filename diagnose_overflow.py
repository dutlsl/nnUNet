"""
Phase 1 Overflow Diagnosis: Run from epoch 0 with fine-grained triggers.
Catches the EXACT iteration where things start going wrong.
Monitors: logits, gradients, loss components, GradScaler, Mamba state, per-class val dice.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = None
if not hasattr(torch, 'float4_e2m1fn_x2'):
    torch.float4_e2m1fn_x2 = None

from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.metrics import compute_dice_score
from datasets.dual_stream_dataset import get_dual_stream_dataloaders
from models.vivim_backbone import VivimBackbone
from losses.sparse_label_loss import SparseLabelLoss


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm(2).item() ** 2
    return total ** 0.5


def mamba_state_stats(model):
    stats = {}
    for name, p in model.named_parameters():
        if 'mamba' in name.lower() or 'temporal' in name.lower():
            d = p.data.float()
            stats[name] = {
                'abs_max': d.abs().max().item(),
                'abs_mean': d.abs().mean().item(),
                'has_nan': bool(torch.isnan(d).any().item()),
                'has_inf': bool(torch.isinf(d).any().item()),
            }
    return stats


def logit_stats(logits):
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=1)
        log_p = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_p).sum(dim=1)
        return {
            'abs_max': logits.abs().max().item(),
            'mean': logits.float().mean().item(),
            'std': logits.float().std().item(),
            'entropy_mean': entropy.mean().item(),
            'entropy_min': entropy.min().item(),
            'prob_max': probs.max().item(),
            'prob_min': probs.min().item(),
        }


def per_class_val(model, val_loader, device, max_iters, num_classes):
    model.eval()
    class_names = ['Background', 'Sclera', 'Iris', 'Pupil']
    class_dices = {n: [] for n in class_names}
    mean_dices = []
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= max_iters:
                break
            images = batch['images'].to(device)
            targets = batch['label'].to(device)
            logits = model(images)

            # Check for NaN in val logits too
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                total_loss += float('nan')
            else:
                total_loss += ce(logits, targets).item()

            dices = compute_dice_score(logits, targets, num_classes=num_classes)
            for n in class_names:
                class_dices[n].append(dices[n])
            mean_dices.append(dices['Mean_Dice'])
            count += 1
    metrics = {n: float(np.mean(class_dices[n])) if class_dices[n] else 0.0 for n in class_names}
    metrics['Mean_Dice'] = float(np.mean(mean_dices)) if mean_dices else 0.0
    metrics['val_loss'] = total_loss / max(1, count)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_sparse_t5.yaml")
    parser.add_argument("--max_epochs", type=int, default=45)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(os.path.join(PROJECT_ROOT, args.config))
    set_seed(42)

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)

    loaders = get_dual_stream_dataloaders(cfg)
    train_loader = loaders['train']
    val_loader = loaders['validation']

    mamba_cfg = cfg.model.mamba
    model = VivimBackbone(
        in_channels=1, num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
        d_state=mamba_cfg.d_state, d_conv=mamba_cfg.d_conv, expand=mamba_cfg.expand,
        use_mamba=cfg.ablation.use_mamba,
        mamba_version=getattr(cfg.ablation, 'mamba_version', 'mamba3'),
        mamba_headdim=getattr(mamba_cfg, 'headdim', 64),
    ).to(device)

    pretrained = getattr(cfg.training, 'pretrained_checkpoint', None)
    if pretrained and os.path.isfile(pretrained):
        ckpt = torch.load(pretrained, map_location='cpu')
        sd = ckpt.get('network_weights', ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
        clean = {k.replace('backbone.', ''): v for k, v in sd.items()}
        model.load_state_dict(clean, strict=False)
        print(f"[DIAG] Loaded pretrained: {pretrained}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                   weight_decay=cfg.training.weight_decay)
    criterion = SparseLabelLoss(num_classes=cfg.model.num_classes).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    max_train_iters = cfg.training.num_iterations_per_epoch
    max_val_iters = cfg.training.num_val_iterations_per_epoch

    rampup_ep = getattr(cfg.training, 'unsup_rampup_epochs', 30)
    unsup_w = getattr(cfg.training, 'unsup_weight', 0.2)

    os.makedirs(os.path.join(PROJECT_ROOT, "logs", "diagnostics"), exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logpath = os.path.join(PROJECT_ROOT, "logs", "diagnostics", f"diag_{ts}.jsonl")
    print(f"[DIAG] Log: {logpath}")
    print(f"[DIAG] rampup_epochs={rampup_ep}, unsup_weight={unsup_w}")
    print(f"[DIAG] lr={cfg.training.lr}, GradScaler init={scaler.get_scale():.0f}")
    print(f"[DIAG] Starting from epoch 0...", flush=True)

    prev_loss = None
    fatal = False

    for epoch in range(1, args.max_epochs + 1):
        rampup = min(1.0, epoch / max(1, rampup_ep))
        model.train()
        ep_losses = []
        ep_sup = []
        ep_unsup = []
        iter_count = 0

        print(f"\n{'='*80}", flush=True)
        print(f"[EPOCH {epoch:03d}] rampup={rampup:.4f} scaler={scaler.get_scale():.0f}", flush=True)

        for batch in train_loader:
            if iter_count >= max_train_iters:
                break

            images = batch['images'].to(device)
            targets = batch['label'].to(device)
            has_label = batch['has_label'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=True):
                logits = model(images)
                loss, l_sup, l_unsup = criterion(logits, targets, has_label, cfg=cfg, epoch=epoch)

            ls = logit_stats(logits)
            scale_before = scaler.get_scale()

            is_bad = bool(torch.isnan(loss).item() or torch.isinf(loss).item())

            if not is_bad:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = grad_norm(model)
                scaler.step(optimizer)
                scaler.update()
                ep_losses.append(loss.item())
                ep_sup.append(l_sup.item())
                ep_unsup.append(l_unsup.item())
            else:
                gn = float('nan')
                optimizer.zero_grad()

            scale_after = scaler.get_scale()

            # === TRIGGERS ===
            triggers = []
            if is_bad:
                triggers.append("NaN/Inf loss")
            if ls['abs_max'] > 30.0:
                triggers.append(f"logit_abs_max={ls['abs_max']:.1f}")
            if not is_bad and gn > 50.0:
                triggers.append(f"grad_norm={gn:.1f}")
            if scale_after < scale_before * 0.5:
                triggers.append(f"scaler_halved {scale_before:.0f}->{scale_after:.0f}")
            if not is_bad and prev_loss is not None and loss.item() > prev_loss * 3.0 and loss.item() > 0.5:
                triggers.append(f"loss_spike {prev_loss:.4f}->{loss.item():.4f}")

            if not is_bad:
                prev_loss = loss.item()

            entry = {
                'epoch': epoch, 'iter': iter_count, 'rampup': round(rampup, 4),
                'loss': round(loss.item(), 6) if not is_bad else 'nan',
                'sup': round(l_sup.item(), 6), 'unsup': round(l_unsup.item(), 6),
                'grad_norm': round(gn, 4) if not is_bad else 'nan',
                'scaler': scale_after,
                'logit_abs_max': round(ls['abs_max'], 2),
                'logit_std': round(ls['std'], 4),
                'entropy_mean': round(ls['entropy_mean'], 4),
                'prob_max': round(ls['prob_max'], 6),
                'n_lab': int(has_label.sum().item()),
                'triggers': triggers,
            }
            with open(logpath, 'a') as f:
                f.write(json.dumps(entry) + '\n')

            if triggers:
                print(f"\n  🔴 TRIGGER E{epoch:03d}/I{iter_count:04d}: {triggers}", flush=True)
                print(f"     loss={entry['loss']} sup={l_sup.item():.4f} unsup={l_unsup.item():.4f}", flush=True)
                print(f"     logit_abs_max={ls['abs_max']:.2f} std={ls['std']:.4f} entropy={ls['entropy_mean']:.4f}", flush=True)
                print(f"     grad_norm={gn} scaler={scale_before:.0f}->{scale_after:.0f}", flush=True)
                print(f"     prob_max={ls['prob_max']:.6f} prob_min={ls['prob_min']:.8f}", flush=True)

                mstats = mamba_state_stats(model)
                print(f"     --- MAMBA STATE ---", flush=True)
                for mname, ms in mstats.items():
                    flag = "X" if ms['has_nan'] or ms['has_inf'] or ms['abs_max'] > 100 else " "
                    print(f"     [{flag}] {mname}: max={ms['abs_max']:.4f} mean={ms['abs_mean']:.6f} nan={ms['has_nan']} inf={ms['has_inf']}", flush=True)

                if is_bad:
                    print(f"\n  FATAL NaN at epoch {epoch} iter {iter_count}. STOPPING.", flush=True)
                    fatal = True
                    break
            elif iter_count % 50 == 0:
                print(f"  [I{iter_count:04d}] loss={loss.item():.4f} sup={l_sup.item():.4f} unsup={l_unsup.item():.4f} "
                      f"| logit_max={ls['abs_max']:.1f} grad={gn:.1f} scaler={scale_after:.0f} entropy={ls['entropy_mean']:.3f}", flush=True)

            iter_count += 1

        if fatal:
            break

        # === END OF EPOCH: FULL VAL WITH PER-CLASS DICE ===
        val_m = per_class_val(model, val_loader, device, max_val_iters, cfg.model.num_classes)
        avg_loss = np.mean(ep_losses) if ep_losses else float('nan')
        avg_sup = np.mean(ep_sup) if ep_sup else float('nan')
        avg_unsup = np.mean(ep_unsup) if ep_unsup else float('nan')

        print(f"\n  [EPOCH {epoch:03d} DONE]", flush=True)
        print(f"  train: loss={avg_loss:.4f} sup={avg_sup:.4f} unsup={avg_unsup:.4f}", flush=True)
        print(f"  val:   loss={val_m['val_loss']:.4f} | Mean={val_m['Mean_Dice']:.4f}", flush=True)
        print(f"         Pupil={val_m['Pupil']:.4f} Iris={val_m['Iris']:.4f} Sclera={val_m['Sclera']:.4f} BG={val_m['Background']:.4f}", flush=True)

        val_triggers = []
        if val_m['Pupil'] < 0.5:
            val_triggers.append(f"Pupil DEAD ({val_m['Pupil']:.4f})")
        if val_m['Iris'] < 0.5:
            val_triggers.append(f"Iris DEAD ({val_m['Iris']:.4f})")
        if val_m['Sclera'] < 0.5:
            val_triggers.append(f"Sclera DEAD ({val_m['Sclera']:.4f})")
        if val_m['Mean_Dice'] < 0.8 and epoch > 3:
            val_triggers.append(f"Mean Dice collapsed ({val_m['Mean_Dice']:.4f})")
        if np.isnan(val_m['val_loss']):
            val_triggers.append("val_loss is NaN")

        if val_triggers:
            print(f"\n  ** VAL TRIGGERS: {val_triggers}", flush=True)
            mstats = mamba_state_stats(model)
            print(f"     --- MAMBA STATE AT VAL TRIGGER ---", flush=True)
            for mname, ms in mstats.items():
                flag = "X" if ms['has_nan'] or ms['has_inf'] or ms['abs_max'] > 100 else " "
                print(f"     [{flag}] {mname}: max={ms['abs_max']:.4f} mean={ms['abs_mean']:.6f}", flush=True)

        val_entry = {
            'epoch': epoch, 'type': 'val',
            'train_loss': round(avg_loss, 6), 'train_sup': round(avg_sup, 6), 'train_unsup': round(avg_unsup, 6),
            'val_loss': round(val_m['val_loss'], 6) if not np.isnan(val_m['val_loss']) else 'nan',
            'dice_mean': round(val_m['Mean_Dice'], 4),
            'dice_pupil': round(val_m['Pupil'], 4),
            'dice_iris': round(val_m['Iris'], 4),
            'dice_sclera': round(val_m['Sclera'], 4),
            'dice_bg': round(val_m['Background'], 4),
            'rampup': round(rampup, 4),
            'val_triggers': val_triggers,
        }
        with open(logpath, 'a') as f:
            f.write(json.dumps(val_entry) + '\n')

        print(f"{'='*80}\n", flush=True)

    print(f"\n[DIAG] Complete. Log: {logpath}", flush=True)


if __name__ == '__main__':
    main()
