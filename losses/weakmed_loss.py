"""
WeakMed Loss Module for Spherical Eyeball Semi-Supervised Training.

Implements:
1. DiceCELoss: Standard supervised 2D segmentation loss (Dice + CrossEntropy)
2. M2BLoss: Mask-to-Box transformation loss from WeakMed paper (CVPR 2025)
   - Projects predicted circle mask into box-aligned representation
   - Supervises against bounding box GT via BCE + Dice
3. ScleraContainmentLoss: Semi-supervised constraint
   - All sclera pixels must lie inside the predicted eyeball circle
   - Uses 2D seg output as weak supervision signal for sphere head

SC Loss (Scale Consistency) is intentionally excluded because:
  - SC was designed to resolve M2B's many-to-one ambiguity (multiple shapes → same bbox)
  - Our SphereHead outputs a circle parameterized by (cx, cy, r), so shape is fixed
  - No shape ambiguity exists → SC is unnecessary
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


class M2BLoss(nn.Module):
    """
    Mask-to-Box (M2B) transformation loss from the WeakMed paper.

    Given a predicted mask and GT bounding box:
    1. Extract the mask region within the bbox
    2. Project to 1D via max-pooling along rows and columns
    3. Back-project to 2D box-aligned representation via outer product (min)
    4. Supervise the box-aligned representation against a filled bbox GT (all 1s)

    This implementation operates only on bbox patches for memory efficiency
    and avoids in-place ops for clean autograd support.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        sphere_mask: torch.Tensor,
        eyeball_bbox: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute M2B loss between sphere mask and bounding box GT.

        Args:
            sphere_mask: [B, 1, H, W] differentiable circle mask in [0, 1]
            eyeball_bbox: [B, 4] GT bounding box (x1, y1, x2, y2)

        Returns:
            m2b_loss: scalar loss value (BCE + Dice within bbox)
        """
        B, _, H, W = sphere_mask.shape
        bce_sum = torch.tensor(0.0, device=sphere_mask.device)
        dice_sum = torch.tensor(0.0, device=sphere_mask.device)
        valid_count = 0

        for b in range(B):
            x1 = max(0, int(eyeball_bbox[b, 0].item()))
            y1 = max(0, int(eyeball_bbox[b, 1].item()))
            x2 = min(W, int(eyeball_bbox[b, 2].item()))
            y2 = min(H, int(eyeball_bbox[b, 3].item()))

            if x2 <= x1 or y2 <= y1:
                continue

            # Extract patch: [h, w]
            patch = sphere_mask[b, 0, y1:y2, x1:x2]
            h, w = patch.shape

            # M2B Projection (Eq. 1): max-pool along rows and columns
            P_w = patch.max(dim=0, keepdim=True).values  # [1, w]
            P_h = patch.max(dim=1, keepdim=True).values  # [h, 1]

            # M2B Back-projection (Eq. 2): outer product with min
            T_prime = torch.min(P_w.expand(h, w), P_h.expand(h, w))  # [h, w]

            # GT is all 1s inside bbox (box mask)
            gt = torch.ones_like(T_prime)

            # BCE
            bce_sum = bce_sum + F.binary_cross_entropy(
                T_prime.clamp(1e-7, 1 - 1e-7), gt, reduction='mean')

            # Dice
            intersection = (T_prime * gt).sum()
            union = T_prime.sum() + gt.sum()
            dice_sum = dice_sum + (1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth))

            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=sphere_mask.device, requires_grad=True)

        return (bce_sum + dice_sum) / valid_count


class ScleraContainmentLoss(nn.Module):
    """
    Semi-supervised containment constraint.

    Enforces that all sclera pixels (from 2D seg prediction) lie inside
    the predicted eyeball circle. This uses the 2D seg head's output as
    a weak supervisory signal for the sphere head without any GT eyeball label.

    L_contain = mean(max(0, sclera_mask - eyeball_mask)^2)

    If sclera extends beyond the circle boundary, the penalty is positive.
    This naturally encourages the circle to encompass all visible eye structures.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        sphere_mask: torch.Tensor,
        seg_logits: torch.Tensor,
        sclera_class_idx: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            sphere_mask: [B, 1, H, W] predicted circle mask in [0, 1]
            seg_logits: [B, C, H, W] segmentation logits
            sclera_class_idx: class index for sclera (default=1)

        Returns:
            containment_loss: scalar
        """
        # Derive sclera probability from seg predictions (detach to stop grad to seg head)
        seg_probs = F.softmax(seg_logits.detach(), dim=1)

        # Union of all foreground classes (Sclera ∪ Iris ∪ Pupil) = all visible eye parts
        # This is a stronger signal than sclera alone
        foreground_prob = seg_probs[:, 1:, :, :].sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Violation: foreground extends beyond sphere mask
        violation = F.relu(foreground_prob - sphere_mask)  # [B, 1, H, W]

        # Squared penalty for smooth gradients
        containment_loss = (violation ** 2).mean()

        return containment_loss


class WeakMedSphereLoss(nn.Module):
    """
    Combined loss for spherical eyeball WeakMed semi-supervised training.

    Components:
    1. seg_loss: DiceCE on 2D segmentation (fully supervised)
    2. m2b_loss: M2B transformation loss on sphere mask vs bbox GT (weakly supervised)
    3. containment_loss: Foreground containment in sphere (semi-supervised)

    Total = w_seg * L_seg + w_m2b * L_m2b + w_contain * L_contain
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.seg_loss_fn = DiceCELoss(num_classes=num_classes)
        self.m2b_loss_fn = M2BLoss()
        self.containment_loss_fn = ScleraContainmentLoss()

    def forward(
        self,
        model_output: dict,
        label: torch.Tensor,
        eyeball_bbox: torch.Tensor,
        cfg,
        current_epoch: int = 0,
    ) -> dict:
        """
        Args:
            model_output: dict from VivimBackbone with keys:
                'seg_logits': [B, C, H, W]
                'sphere_mask': [B, 1, H, W]  (if sphere head enabled)
                'sphere_params': [B, 3]       (if sphere head enabled)
            label: [B, H, W] GT segmentation mask
            eyeball_bbox: [B, 4] GT bounding box (x1, y1, x2, y2)
            cfg: config object with loss weights
            current_epoch: for warmup scheduling

        Returns:
            dict with 'total_loss' and individual loss components
        """
        losses = {}

        # 1. Supervised segmentation loss (always active)
        seg_logits = model_output['seg_logits']
        seg_loss = self.seg_loss_fn(seg_logits, label)
        w_seg = cfg.losses.seg_dice_ce.weight
        losses['seg_loss'] = seg_loss
        total = w_seg * seg_loss

        # 2. M2B Loss on sphere mask (if sphere head active)
        if 'sphere_mask' in model_output and cfg.ablation.use_weakmed:
            sphere_mask = model_output['sphere_mask']
            m2b_loss = self.m2b_loss_fn(sphere_mask, eyeball_bbox)
            w_m2b = cfg.losses.weakmed_m2b.weight
            losses['m2b_loss'] = m2b_loss
            total = total + w_m2b * m2b_loss

        # 3. Sclera Containment Loss (semi-supervised, with warmup)
        if ('sphere_mask' in model_output
                and cfg.ablation.use_sclera_containment
                and current_epoch >= cfg.losses.sclera_containment.warmup_epochs):
            containment_loss = self.containment_loss_fn(
                model_output['sphere_mask'],
                model_output['seg_logits'],
            )
            w_contain = cfg.losses.sclera_containment.weight
            losses['containment_loss'] = containment_loss
            total = total + w_contain * containment_loss

        losses['total_loss'] = total
        return losses
