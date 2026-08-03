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
    """Soft Dice Loss for multi-class segmentation."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] raw logits
            target: [B, H, W] integer class labels
        """
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss for segmentation."""

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
    ):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_dice = self.dice(pred, target)
        loss_ce = self.ce(pred, target)
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce


class DomainConsistencyLoss(nn.Module):
    """
    Domain Consistency Loss.
    Enforces that features from different domains sharing the same semantic class
    should produce similar predictions. Uses soft KL divergence between
    domain-specific predictions.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        primary_logits: torch.Tensor,
        auxiliary_logits_list: list,
        primary_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            primary_logits: [B, C, H, W] predictions from primary domain (OpenEDS)
            auxiliary_logits_list: List of [B_aux, C, H, W] predictions from auxiliary domains
            primary_labels: [B, H, W] ground truth labels for primary domain

        Returns:
            loss: scalar domain consistency loss
        """
        if not auxiliary_logits_list:
            return torch.tensor(0.0, device=primary_logits.device)

        # Soft predictions from primary domain
        primary_soft = F.softmax(primary_logits / self.temperature, dim=1)

        total_loss = torch.tensor(0.0, device=primary_logits.device)
        count = 0

        for aux_logits in auxiliary_logits_list:
            if aux_logits.shape[0] == 0:
                continue

            # Match spatial dimensions
            if aux_logits.shape[2:] != primary_logits.shape[2:]:
                aux_logits = F.interpolate(
                    aux_logits,
                    size=primary_logits.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )

            aux_soft = F.softmax(aux_logits / self.temperature, dim=1)

            # Use minimum batch size
            B_min = min(primary_soft.shape[0], aux_soft.shape[0])
            p_soft = primary_soft[:B_min]
            a_soft = aux_soft[:B_min]

            # KL divergence (symmetric, averaged per pixel across channels)
            kl_pa = F.kl_div(
                torch.log(p_soft + 1e-8), a_soft, reduction='none'
            ).sum(dim=1).mean()
            kl_ap = F.kl_div(
                torch.log(a_soft + 1e-8), p_soft, reduction='none'
            ).sum(dim=1).mean()

            total_loss = total_loss + (kl_pa + kl_ap) / 2.0
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

        # Normalize features
        p_norm = F.normalize(primary_tokens, dim=-1)  # [B, N, C]
        a_norm = F.normalize(auxiliary_tokens, dim=-1)  # [B, N, C]

        # Sample anchor-positive pairs (same position = positive)
        num_samples = min(self.num_negatives, N)
        indices = torch.randperm(N, device=primary_tokens.device)[:num_samples]

        anchors = p_norm[:, indices, :]  # [B, num_samples, C]
        positives = a_norm[:, indices, :]  # [B, num_samples, C]

        # Positive similarity
        pos_sim = (anchors * positives).sum(dim=-1) / self.temperature  # [B, num_samples]

        # Negative: all other positions from auxiliary domain
        neg_features = a_norm  # [B, N, C]
        neg_sim = torch.bmm(
            anchors, neg_features.transpose(1, 2)
        ) / self.temperature  # [B, num_samples, N]

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, num_samples, 1+N]
        labels = torch.zeros(B_min, num_samples, dtype=torch.long, device=primary_tokens.device)

        loss = F.cross_entropy(logits.reshape(-1, 1 + N), labels.reshape(-1))
        return loss


class SAGDLoss(nn.Module):
    """
    Combined SAGD loss:
    L_total = w_seg * L_DiceCE + w_dc * L_DomainConsistency + w_sc * L_StructuralContrastive
    """

    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg.losses

        self.seg_loss = DiceCELoss(
            num_classes=cfg.model.num_classes,
            dice_weight=loss_cfg.seg.dice_weight,
            ce_weight=loss_cfg.seg.ce_weight,
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

    def forward(
        self,
        seg_logits: torch.Tensor,
        labels: torch.Tensor,
        primary_tokens: Optional[torch.Tensor] = None,
        auxiliary_tokens_list: Optional[list] = None,
        auxiliary_logits_list: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seg_logits: [B, C, H, W] segmentation predictions (primary domain)
            labels: [B, H, W] ground truth labels (primary domain)
            primary_tokens: [B, N, C] primary domain serialized tokens (for contrastive)
            auxiliary_tokens_list: List of [B_aux, N, C] auxiliary domain tokens
            auxiliary_logits_list: List of [B_aux, C, H, W] auxiliary domain predictions

        Returns:
            dict with 'total', 'seg', 'domain_consistency', 'structural_contrastive'
        """
        # Segmentation loss (always computed)
        loss_seg = self.seg_loss(seg_logits, labels)

        # Domain consistency loss
        loss_dc = torch.tensor(0.0, device=seg_logits.device)
        if auxiliary_logits_list:
            loss_dc = self.domain_consistency_loss(
                seg_logits, auxiliary_logits_list, labels
            )

        # Structural contrastive loss
        loss_sc = torch.tensor(0.0, device=seg_logits.device)
        if primary_tokens is not None and auxiliary_tokens_list:
            for aux_tokens in auxiliary_tokens_list:
                if aux_tokens.shape[0] > 0:
                    loss_sc = loss_sc + self.structural_contrastive_loss(
                        primary_tokens, aux_tokens
                    )
            loss_sc = loss_sc / max(len(auxiliary_tokens_list), 1)

        # Total loss
        total = self.w_seg * loss_seg + self.w_dc * loss_dc + self.w_sc * loss_sc

        return {
            'total': total,
            'seg': loss_seg,
            'domain_consistency': loss_dc,
            'structural_contrastive': loss_sc,
        }
