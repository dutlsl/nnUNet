"""
WeakMed Loss Module: Includes M2B (Max-pooling to Boundary) Loss and Scale Consistency Loss.
Designed for weakly-supervised and fine-scale boundary refinement in medical video segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Combined Soft Dice + Cross Entropy Loss for 2D Segmentation."""
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, target)

        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()

        return ce + dice


class WeakMedLoss(nn.Module):
    def __init__(self, num_classes: int = 4, m2b_kernel_size: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.base_loss = DiceCELoss(num_classes=num_classes)
        self.m2b_pool = nn.MaxPool2d(kernel_size=m2b_kernel_size, stride=1, padding=m2b_kernel_size // 2)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, cfg=None) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            target: [B, H, W]
        """
        total_loss = self.base_loss(logits, target)

        if cfg is None or not hasattr(cfg, 'ablation') or not cfg.ablation.use_weakmed:
            return total_loss

        probs = F.softmax(logits, dim=1)

        # 1. M2B (Max-pooling to Boundary) Loss
        # Extracts boundary confidence gradients by comparing max-pooled probabilities to original
        boundary_probs = self.m2b_pool(probs) - probs
        m2b_loss = boundary_probs.abs().mean()
        total_loss += cfg.losses.weakmed_m2b.weight * m2b_loss

        # 2. Scale Consistency Loss
        # Downsamples logits by half and computes consistency loss against downsampled target
        down_logits = F.interpolate(logits, scale_factor=0.5, mode='bilinear', align_corners=False)
        down_target = F.interpolate(target.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).long()
        sc_loss = self.base_loss(down_logits, down_target)
        total_loss += cfg.losses.weakmed_sc.weight * sc_loss

        return total_loss
