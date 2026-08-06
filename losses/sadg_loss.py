"""
SAGD Loss Functions for Eyeball Segmentation.
Combines supervised segmentation loss with domain generalization losses.

1. DiceCE Loss: Standard segmentation loss (Dice + CrossEntropy)
2. Domain Consistency Loss: Enforces feature consistency across domains
3. Structural Contrastive Loss: Contrastive learning on serialized token features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DiceLoss(nn.Module):
    """Soft Dice Loss for multi-class foreground segmentation with partial annotation support."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4:
            target = target.squeeze(1)
        target = target.long().clamp(min=0, max=3)
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Mask out classes that are not present in the ground truth for each sample
        cls_present = (target_onehot.sum(dim=(2, 3)) > 0)[:, 1:]  # [B, num_classes - 1]
        fg_dice = dice[:, 1:]
        
        loss_per_cls = torch.where(cls_present, 1.0 - fg_dice, torch.tensor(0.0, device=pred.device))
        num_present = cls_present.float().sum()
        if num_present > 0:
            return loss_per_cls.sum() / num_present
        return 1.0 - fg_dice.mean()


class DiceCELoss(nn.Module):
    """Standard Dice + CrossEntropy loss for multi-class segmentation.
    
    Supports per-class weights for CE to handle background-dominant class imbalance.
    Default: [0.1, 1.0, 1.0, 1.0] — suppresses background gradient dominance.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 1.0,
        ce_weight: float = 0.3,
        class_weights: Optional[list] = None,
    ):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        # Default: background=0.1, foreground=1.0 to prevent mode collapse
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4:
            target = target.squeeze(1)
        target = target.long().clamp(min=0, max=3)
        loss_dice = self.dice(pred, target)
        # Ensure class_weights on same device as pred (SAGDLoss may not be .to(device)'d)
        w = self.class_weights.to(pred.device) if self.class_weights is not None else None
        loss_ce = F.cross_entropy(pred, target, weight=w)
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce


class DomainConsistencyLoss(nn.Module):
    """
    Token-Level Domain Consistency Loss (MMD-based).

    Aligns feature distributions across domains using Maximum Mean Discrepancy (MMD)
    on serialized tokens. This operates PURELY at the feature level —
    NO seg logits, NO labels from auxiliary domains (Swirski/LPW).

    Previous implementation used KL divergence on seg logits, which forced
    primary domain (4-class) predictions to match auxiliary (pupil-only) predictions,
    suppressing Iris/Sclera and causing mode collapse.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Compute Gaussian RBF kernel between token sets."""
        # x: [B, N, C], y: [B, M, C]
        # Sample to avoid OOM: take max 128 tokens from each
        N = min(x.shape[1], 128)
        M = min(y.shape[1], 128)
        if x.shape[1] > N:
            idx_x = torch.randperm(x.shape[1], device=x.device)[:N]
            x = x[:, idx_x]
        if y.shape[1] > M:
            idx_y = torch.randperm(y.shape[1], device=y.device)[:M]
            y = y[:, idx_y]

        # [B, N, 1, C] - [B, 1, M, C] -> [B, N, M]
        dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(dim=-1)
        return torch.exp(-dist / (2 * sigma ** 2))

    def forward(
        self,
        primary_tokens: torch.Tensor,
        auxiliary_tokens_list: list,
    ) -> torch.Tensor:
        """
        Token-level MMD domain alignment (no labels needed).

        Args:
            primary_tokens: [B, N, C] serialized tokens from primary domain
            auxiliary_tokens_list: List of [B_aux, N, C] tokens from auxiliary domains

        Returns:
            loss: scalar MMD domain consistency loss
        """
        if primary_tokens is None or not auxiliary_tokens_list:
            return torch.tensor(0.0, device=primary_tokens.device if primary_tokens is not None else 'cpu')

        total_loss = torch.tensor(0.0, device=primary_tokens.device)
        count = 0

        p = F.normalize(primary_tokens.float(), dim=-1)

        for aux_tokens in auxiliary_tokens_list:
            if aux_tokens is None or aux_tokens.shape[0] == 0:
                continue

            B_min = min(p.shape[0], aux_tokens.shape[0])
            p_sub = p[:B_min]
            a_sub = F.normalize(aux_tokens[:B_min].float(), dim=-1)

            # MMD: E[k(x,x)] + E[k(y,y)] - 2*E[k(x,y)]
            k_pp = self._gaussian_kernel(p_sub, p_sub).mean()
            k_aa = self._gaussian_kernel(a_sub, a_sub).mean()
            k_pa = self._gaussian_kernel(p_sub, a_sub).mean()

            mmd = k_pp + k_aa - 2 * k_pa
            total_loss = total_loss + mmd.clamp(min=0)
            count += 1

        return total_loss / max(count, 1)




class StructuralContrastiveLoss(nn.Module):
    """
    Structural Contrastive Loss.
    Encourages features at the same position in the serialized sequence
    (i.e., same structural role across domains) to be similar,
    while pushing apart features from different structural positions.
    """

    def __init__(self, temperature: float = 0.07, num_negatives: int = 256):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives

    def forward(
        self,
        primary_tokens: torch.Tensor,
        auxiliary_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            primary_tokens: [B, N, C] serialized tokens from primary domain
            auxiliary_tokens: [B, N, C] serialized tokens from auxiliary domain

        Returns:
            loss: scalar contrastive loss
        """
        B, N, C = primary_tokens.shape
        if auxiliary_tokens.shape[0] == 0 or N == 0:
            return torch.tensor(0.0, device=primary_tokens.device)

        B_min = min(B, auxiliary_tokens.shape[0])
        primary_tokens = primary_tokens[:B_min]
        auxiliary_tokens = auxiliary_tokens[:B_min]

        # FP32 forced: AMP FP16 with /temperature(0.07) causes overflow beyond FP16 max(65504)
        p_norm = F.normalize(primary_tokens.float(), dim=-1)  # [B, N, C]
        a_norm = F.normalize(auxiliary_tokens.float(), dim=-1)  # [B, N, C]

        # Sample anchor positions
        num_samples = min(self.num_negatives, N)
        indices = torch.randperm(N, device=primary_tokens.device)[:num_samples]  # [num_samples]

        anchors = p_norm[:, indices, :]  # [B_min, num_samples, C]

        # Compute cosine similarity, clamp to valid range BEFORE dividing by temperature
        # This prevents FP16 overflow: cosine ∈ [-1, 1], so max after /0.07 ≈ 14.3 (safe)
        sim_matrix = torch.bmm(anchors, a_norm.transpose(1, 2))  # [B_min, num_samples, N]
        sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)
        sim_matrix = sim_matrix / self.temperature

        # Target for anchor i is exact token position indices[i]
        labels = indices.unsqueeze(0).expand(B_min, -1)  # [B_min, num_samples]

        loss = F.cross_entropy(sim_matrix.reshape(-1, N), labels.reshape(-1))
        return loss


class SAGDLoss(nn.Module):
    """
    Combined SAGD loss:
    L_total = w_seg * L_DiceCE + warmup(epoch) * (w_dc * L_DC + w_sc * L_SC)

    DESIGN NOTE: Auxiliary domains (Swirski/LPW) are for FEATURE-LEVEL alignment only.
    - StructuralContrastiveLoss: token-level contrastive (no labels) ✅
    - DomainConsistencyLoss: KL on seg logits (DISABLED, weight=0) ❌
      Swirski/LPW have pupil-only labels → their decoded seg logits are background-dominant
      → KL pushes primary Iris/Sclera predictions toward background → mode collapse.

    Warmup: SADG auxiliary losses ramp from 0→1 over warmup_epochs to let
    supervised DiceCE converge first before introducing feature alignment.
    """

    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg.losses

        # Class weights for CE: suppress background gradient dominance
        class_weights = getattr(loss_cfg.seg, 'class_weights', None)
        if class_weights is not None:
            class_weights = list(class_weights)

        self.seg_loss = DiceCELoss(
            num_classes=cfg.model.num_classes,
            dice_weight=loss_cfg.seg.dice_weight,
            ce_weight=loss_cfg.seg.ce_weight,
            class_weights=class_weights,
        )
        self.domain_consistency_loss = DomainConsistencyLoss(
            temperature=loss_cfg.domain_consistency.temperature,
        )
        self.structural_contrastive_loss = StructuralContrastiveLoss(
            num_negatives=loss_cfg.structural_contrastive.num_negatives,
        )

        self.w_seg = loss_cfg.seg.weight
        self.w_dc = loss_cfg.domain_consistency.weight
        self.w_sc = loss_cfg.structural_contrastive.weight
        self.warmup_epochs = getattr(loss_cfg, 'warmup_epochs', 0)

    def forward(
        self,
        seg_logits: torch.Tensor,
        labels: torch.Tensor,
        primary_tokens: Optional[torch.Tensor] = None,
        auxiliary_tokens_list: Optional[list] = None,
        current_epoch: int = 0,
        **kwargs,  # absorb legacy auxiliary_logits_list if passed
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seg_logits: [B, C, H, W] segmentation predictions (primary domain)
            labels: [B, H, W] ground truth labels (primary domain ONLY — OpenEDS)
            primary_tokens: [B, N, C] primary domain serialized tokens
            auxiliary_tokens_list: List of [B_aux, N, C] auxiliary domain tokens
                                  (feature-level only, NO labels from Swirski/LPW)
            current_epoch: for warmup schedule

        Returns:
            dict with 'total', 'seg', 'domain_consistency', 'structural_contrastive'
        """
        # Segmentation loss (always computed, only on primary domain OpenEDS labels)
        loss_seg = self.seg_loss(seg_logits, labels)

        # Warmup factor: 0 → 1 over warmup_epochs
        if self.warmup_epochs > 0 and current_epoch < self.warmup_epochs:
            warmup_factor = float(current_epoch) / float(self.warmup_epochs)
        else:
            warmup_factor = 1.0

        # Domain consistency loss (token-level MMD — NO labels, NO seg logits)
        # Aligns feature distributions between primary and auxiliary domains purely
        # at the token level. Swirski/LPW contribute only features, not labels.
        loss_dc = torch.tensor(0.0, device=seg_logits.device)
        if self.w_dc > 0 and primary_tokens is not None and auxiliary_tokens_list:
            loss_dc = self.domain_consistency_loss(
                primary_tokens, auxiliary_tokens_list,
            )

        # Structural contrastive loss (feature-level, no labels involved — safe)
        loss_sc = torch.tensor(0.0, device=seg_logits.device)
        if primary_tokens is not None and auxiliary_tokens_list:
            for aux_tokens in auxiliary_tokens_list:
                if aux_tokens.shape[0] > 0:
                    loss_sc = loss_sc + self.structural_contrastive_loss(
                        primary_tokens, aux_tokens
                    )
            loss_sc = loss_sc / max(len(auxiliary_tokens_list), 1)

        # Total loss with warmup on auxiliary losses
        total = self.w_seg * loss_seg + warmup_factor * (
            self.w_dc * loss_dc + self.w_sc * loss_sc
        )

        return {
            'total': total,
            'seg': loss_seg,
            'domain_consistency': loss_dc,
            'structural_contrastive': loss_sc,
        }
