"""
MultiFrameMaskedLoss: Phase 2 loss for multi-frame decoding.

- Masked Supervised Loss: DiceCE only on frames with labels
- Temporal Smoothness Loss: L2 between adjacent frame predictions
- M2B Boundary on all frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce = self.ce_loss(logits, target)
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (probs * target_oh).sum((0, 2, 3))
        union = probs.sum((0, 2, 3)) + target_oh.sum((0, 2, 3))
        dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return ce + dice


class MultiFrameMaskedLoss(nn.Module):
    """
    Args for forward():
        logits: [B, T, C, H, W] — multi-frame predictions
        targets: [B, T, H, W] — labels (valid only where label_mask=1)
        label_mask: [B, T] — 1 if frame has label, 0 otherwise
    Returns:
        (total_loss, loss_sup, loss_smooth)
    """
    def __init__(self, num_classes=4, m2b_kernel_size=5):
        super().__init__()
        self.dice_ce = DiceCELoss(num_classes=num_classes)
        self.m2b_pool = nn.MaxPool2d(m2b_kernel_size, stride=1, padding=m2b_kernel_size // 2)

    def forward(self, logits, targets, label_mask, cfg=None, epoch=1):
        B, T, C, H, W = logits.shape
        dev = logits.device

        loss_sup = torch.tensor(0.0, device=dev)
        loss_smooth = torch.tensor(0.0, device=dev)
        loss_m2b = torch.tensor(0.0, device=dev)

        # --- 1. Masked supervised loss (only labeled frames) ---
        mask = label_mask.bool()  # [B, T]
        n_labeled = mask.sum().item()
        if n_labeled > 0:
            # Gather labeled frames
            lab_logits = logits[mask]   # [N_lab, C, H, W]
            lab_targets = targets[mask]  # [N_lab, H, W]
            loss_sup = self.dice_ce(lab_logits, lab_targets)

        # --- 2. Temporal smoothness loss (all adjacent pairs) ---
        if T > 1:
            # [B, T-1, C, H, W]
            probs_all = F.softmax(logits, dim=2)
            diff = probs_all[:, 1:] - probs_all[:, :-1]
            loss_smooth = (diff ** 2).mean()

        # --- 3. M2B boundary sharpness on all frames ---
        logits_flat = logits.view(B * T, C, H, W)
        probs_flat = F.softmax(logits_flat, dim=1)
        boundary = self.m2b_pool(probs_flat) - probs_flat
        loss_m2b = boundary.abs().mean()

        # Weights
        smooth_w = 1.0
        m2b_w = 1.0
        if cfg:
            smooth_w = getattr(cfg.losses, 'temporal_smooth', type('', (), {'weight': 1.0})).weight
            m2b_w = getattr(cfg.losses.weakmed_m2b, 'weight', 1.0)

        # Ramp-up for unsupervised losses
        rampup_ep = 30 if not cfg else getattr(cfg.training, 'unsup_rampup_epochs', 30)
        rampup = min(1.0, epoch / max(1, rampup_ep))

        total = loss_sup + rampup * (smooth_w * loss_smooth + m2b_w * loss_m2b)
        return total, loss_sup, loss_smooth
