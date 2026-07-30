"""
WeakMed Loss Module: Paper-accurate M2B (Mask-to-Box) Transform + Scale Consistency Loss.

Reference: "Rethinking Box Supervision: Bias-Free Weakly Supervised Medical Segmentation"
- M2B Transform: Eq. 2 — row/column marginal projection → box-aligned intersection
- Scale Consistency: Eq. 4 — symmetric KL divergence across input scales
- Total Loss: Eq. 5 — L_total = L_M2B + L_SC

Designed for weakly-supervised eyeball segmentation using sclera-derived bounding box labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Combined Binary Dice + BCE Loss for single-class mask supervision."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 1, H, W] — raw logits (pre-sigmoid)
            target: [B, 1, H, W] — binary target mask (0 or 1)
        """
        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()

        return bce_loss + dice_loss


class DiceCELoss(nn.Module):
    """Combined Soft Dice + Cross Entropy Loss for multi-class 2D Segmentation."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        ce = self.ce_loss(logits, target)

        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()

        return ce + dice


def mask_to_box_transform(pred_probs: torch.Tensor, bbox: torch.Tensor,
                          img_h: int, img_w: int) -> torch.Tensor:
    """
    Differentiable Mask-to-Box (M2B) transformation (WeakMEd Eq. 2).

    Projects predicted mask probabilities into box-aligned representations
    by computing row/column marginal projections and their intersection.

    Args:
        pred_probs: [B, 1, H, W] — sigmoid probabilities of eyeball prediction
        bbox: [B, 4] — (x1, y1, x2, y2) square bounding box coordinates (pixel)
        img_h: image height
        img_w: image width

    Returns:
        T: [B, 1, H, W] — box-aligned transformed mask
    """
    B = pred_probs.shape[0]
    T_out = torch.zeros_like(pred_probs)

    for b in range(B):
        x1, y1, x2, y2 = bbox[b].int().tolist()

        # Clamp to valid range
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        P_crop = pred_probs[b, 0, y1:y2, x1:x2]  # [h, w]
        h, w = P_crop.shape

        # Column-wise projection: max over height dimension → [1, w]
        P_w = P_crop.max(dim=0, keepdim=True).values  # [1, w]
        P_hat_w = P_w.expand(h, w)  # broadcast to [h, w]

        # Row-wise projection: max over width dimension → [h, 1]
        P_h = P_crop.max(dim=1, keepdim=True).values  # [h, 1]
        P_hat_h = P_h.expand(h, w)  # broadcast to [h, w]

        # Box-aligned intersection: T' = min(P_hat_w, P_hat_h)
        T_prime = torch.min(P_hat_w, P_hat_h)  # [h, w]

        T_out[b, 0, y1:y2, x1:x2] = T_prime

    return T_out


def create_box_mask(bbox: torch.Tensor, img_h: int, img_w: int,
                    device: torch.device) -> torch.Tensor:
    """
    Create binary box mask from bounding box coordinates.

    Args:
        bbox: [B, 4] — (x1, y1, x2, y2)
        img_h, img_w: image dimensions
        device: target device

    Returns:
        box_mask: [B, 1, H, W] — binary mask with 1 inside bbox
    """
    B = bbox.shape[0]
    box_mask = torch.zeros(B, 1, img_h, img_w, device=device)

    for b in range(B):
        x1, y1, x2, y2 = bbox[b].int().tolist()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        if x2 > x1 and y2 > y1:
            box_mask[b, 0, y1:y2, x1:x2] = 1.0

    return box_mask


class MaskToBoxLoss(nn.Module):
    """
    M2B Loss (WeakMEd Eq. 3): Supervise M2B-transformed predictions with ground-truth box masks.

    L_M2B = 0.5 * [BCE(T1, B) + BCE(T2, B)] + 0.5 * [Dice(T1, B) + Dice(T2, B)]

    In single-scale mode (when scale consistency handles the dual-scale),
    we compute M2B on the primary prediction only.
    """

    def __init__(self):
        super().__init__()
        self.dice_bce = DiceBCELoss()

    def forward(self, pred_probs: torch.Tensor, bbox: torch.Tensor,
                img_h: int, img_w: int) -> torch.Tensor:
        """
        Args:
            pred_probs: [B, 1, H, W] — sigmoid probabilities
            bbox: [B, 4] — (x1, y1, x2, y2) square bounding box
            img_h, img_w: spatial dimensions
        """
        # M2B transform: project prediction into box-aligned space
        T = mask_to_box_transform(pred_probs, bbox, img_h, img_w)

        # Ground-truth box mask
        B_mask = create_box_mask(bbox, img_h, img_w, pred_probs.device)

        # M2B is already in probability space [0,1], so we convert to logits for DiceBCE
        # Clamp to avoid log(0)
        T_clamped = T.clamp(1e-6, 1.0 - 1e-6)
        T_logits = torch.log(T_clamped / (1.0 - T_clamped))

        return self.dice_bce(T_logits, B_mask)


class ScaleConsistencyLoss(nn.Module):
    """
    Scale Consistency (SC) Loss (WeakMEd Eq. 4):
    Symmetric KL divergence between predictions at two input scales within bbox region.

    L_SC = Σ_{(i,j)∈Ω_B} [KL(P1 || P2) + KL(P2 || P1)] / (2|Ω_B|)
    """

    def __init__(self, scale_factor: float = 0.5):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, logits_full: torch.Tensor, bbox: torch.Tensor,
                img_h: int, img_w: int, forward_fn=None, data=None) -> torch.Tensor:
        """
        Args:
            logits_full: [B, 1, H, W] — eyeball logits at full scale
            bbox: [B, 4] — bounding box coordinates
            img_h, img_w: spatial dimensions
            forward_fn: callable that takes scaled input data → eyeball logits
            data: [B, T, C, H, W] or [B, C, H, W] — input data for re-inference

        If forward_fn and data are provided, runs inference at downscaled input.
        Otherwise, uses bilinear interpolation of logits as approximation.
        """
        P1 = torch.sigmoid(logits_full)

        if forward_fn is not None and data is not None:
            # True multi-scale: re-run model at different input scale
            if data.ndim == 5:
                B, T, C, H_in, W_in = data.shape
                data_flat = data.view(B * T, C, H_in, W_in)
                data_scaled = F.interpolate(
                    data_flat, scale_factor=self.scale_factor,
                    mode='bilinear', align_corners=False
                )
                data_scaled = data_scaled.view(B, T, C,
                                               data_scaled.shape[2],
                                               data_scaled.shape[3])
            else:
                data_scaled = F.interpolate(
                    data, scale_factor=self.scale_factor,
                    mode='bilinear', align_corners=False
                )

            with torch.no_grad():
                result_scaled = forward_fn(data_scaled)
                if isinstance(result_scaled, dict):
                    logits_scaled = result_scaled['eyeball']
                else:
                    logits_scaled = result_scaled

            # Upsample back to original resolution
            logits_scaled_up = F.interpolate(
                logits_scaled, size=(img_h, img_w),
                mode='bilinear', align_corners=False
            )
            P2 = torch.sigmoid(logits_scaled_up)
        else:
            # Approximate: downscale + upscale logits
            logits_down = F.interpolate(
                logits_full, scale_factor=self.scale_factor,
                mode='bilinear', align_corners=False
            )
            logits_up = F.interpolate(
                logits_down, size=(img_h, img_w),
                mode='bilinear', align_corners=False
            )
            P2 = torch.sigmoid(logits_up)

        # Create bbox region mask
        bbox_mask = create_box_mask(bbox, img_h, img_w, P1.device)

        # Compute symmetric KL divergence within bbox region
        # Clamp probabilities to avoid log(0)
        eps = 1e-6
        P1_c = P1.clamp(eps, 1.0 - eps)
        P2_c = P2.clamp(eps, 1.0 - eps)

        # Binary distribution: p and (1-p)
        kl_12 = P1_c * torch.log(P1_c / P2_c) + (1 - P1_c) * torch.log((1 - P1_c) / (1 - P2_c))
        kl_21 = P2_c * torch.log(P2_c / P1_c) + (1 - P2_c) * torch.log((1 - P2_c) / (1 - P1_c))

        sym_kl = 0.5 * (kl_12 + kl_21)

        # Mask to bbox region only
        masked_kl = sym_kl * bbox_mask
        num_pixels = bbox_mask.sum().clamp(min=1.0)

        return masked_kl.sum() / num_pixels


class WeakMedEyeballLoss(nn.Module):
    """
    WeakMed Eyeball Loss (Eq. 5): L_total = L_M2B + L_SC

    Applies M2B transform and Scale Consistency to train eyeball segmentation
    using only sclera-derived square bounding box annotations.
    """

    def __init__(self, m2b_weight: float = 1.0, sc_weight: float = 1.0,
                 sc_scale_factor: float = 0.5):
        super().__init__()
        self.m2b_loss = MaskToBoxLoss()
        self.sc_loss = ScaleConsistencyLoss(scale_factor=sc_scale_factor)
        self.m2b_weight = m2b_weight
        self.sc_weight = sc_weight

    def forward(self, eyeball_logits: torch.Tensor, bbox: torch.Tensor,
                forward_fn=None, data=None) -> dict:
        """
        Args:
            eyeball_logits: [B, 1, H, W] — raw eyeball head logits
            bbox: [B, 4] — sclera-derived square bounding box (x1, y1, x2, y2)
            forward_fn: optional callable for true multi-scale SC loss
            data: optional input data for multi-scale inference

        Returns:
            dict with 'total', 'm2b', 'sc' loss values
        """
        _, _, img_h, img_w = eyeball_logits.shape
        eyeball_probs = torch.sigmoid(eyeball_logits)

        # M2B Loss
        l_m2b = self.m2b_loss(eyeball_probs, bbox, img_h, img_w)

        # Scale Consistency Loss
        l_sc = self.sc_loss(eyeball_logits, bbox, img_h, img_w,
                            forward_fn=forward_fn, data=data)

        total = self.m2b_weight * l_m2b + self.sc_weight * l_sc

        return {
            'total': total,
            'm2b': l_m2b.detach(),
            'sc': l_sc.detach(),
        }
