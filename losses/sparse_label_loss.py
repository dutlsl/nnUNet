"""
SparseLabelLoss: Labeled/Unlabeled-Aware Loss for Phase 1 sparse-label training.
Strictly NO pseudo-labels (Strategy A).

- Supervised Loss: DiceCE on labeled samples
- Unsupervised Loss: M2B Boundary Sharpness on all predicted probability maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Combined Soft Dice + Cross Entropy Loss (identical to Phase 2)."""
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce = self.ce_loss(logits, target)
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * target_oh).sum(dims)
        union = probs.sum(dims) + target_oh.sum(dims)
        dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return ce + dice


class SparseLabelLoss(nn.Module):
    """
    Dual-path loss: supervised for labeled, self-supervised M2B for unlabeled.
    No pseudo-labels to prevent self-reinforcing gradient explosion.
    """
    def __init__(self, num_classes: int = 4, m2b_kernel_size: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_ce = DiceCELoss(num_classes=num_classes)
        self.m2b_pool = nn.MaxPool2d(kernel_size=m2b_kernel_size, stride=1,
                                      padding=m2b_kernel_size // 2)

    def forward(self, logits, target, has_label, cfg=None, epoch=1):
        dev = logits.device
        labeled_mask = has_label.bool()
        n_lab = labeled_mask.sum().item()

        loss_sup = torch.tensor(0.0, device=dev)

        # 1. Supervised loss on labeled samples
        if n_lab > 0:
            sup_logits = logits[labeled_mask]
            sup_target = target[labeled_mask]
            loss_sup = self.dice_ce(sup_logits, sup_target)

        # 2. Self-supervised M2B Boundary Sharpness (NO pseudo-labels)
        probs = F.softmax(logits, dim=1)
        boundary = self.m2b_pool(probs) - probs
        loss_m2b = boundary.abs().mean()

        m2b_w = 0.5
        if cfg and hasattr(cfg, 'losses') and hasattr(cfg.losses, 'weakmed_m2b'):
            m2b_w = getattr(cfg.losses.weakmed_m2b, 'weight', 0.5)

        loss_unsup = m2b_w * loss_m2b

        # Ramp-up
        rampup_ep = 30
        unsup_w = 0.2
        if cfg:
            rampup_ep = getattr(cfg.training, 'unsup_rampup_epochs', 30)
            unsup_w = getattr(cfg.training, 'unsup_weight', 0.2)
        rampup = min(1.0, epoch / max(1, rampup_ep))

        total = loss_sup + rampup * unsup_w * loss_unsup
        return total, loss_sup, loss_unsup
