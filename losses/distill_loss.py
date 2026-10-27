"""
DistillLoss: annealed distillation of the GAZER RL-tuned policy into a
lighter student (step 4 of the 5-step plan).

Objective (user-chosen: "RL 보상 유지 전제 최대 경량화"):
shrink the model while keeping the per-pixel teacher agreement
(val/agreement_teacher) within -0.01 of the full model. The distillation
objective is the SAME per-pixel teacher-likelihood the GRPO reward uses
(plan §7.2 reward r(x) = ref_t[x, m(x)]) plus an exact per-pixel KL
anchor to the RL policy — i.e. the student anneals toward the RL policy
and the teacher simultaneously, so the RL reward (teacher agreement) is
the preserved quantity.

L = alpha_ce * CE( p_s, ref_soft )          # per-pixel cross-entropy
                                     # against the SAM3 landscape (soft)
    + alpha_kl * KL( p_s || p_t )            # exact per-pixel forward KL
    + alpha_tv * TV( logits_s )              # light spatial smoothing
      (TV only where the anneal weight > 0; it is the "annealing" term —
      it regularizes the lighter decoder as it is being annealed)

All terms are per-pixel means (same normalization as the GRPO loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_ce(logp: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Per-pixel cross-entropy against a soft teacher map, mean over pixels.

    logp:  [N, C, H, W] float32 log-softmax of the student (grad)
    soft:  [N, C, H, W] float (teacher soft4, sums to 1 over C; detached)
    """
    logp = logp.float()
    soft = soft.float().detach()
    return -(soft * logp).sum(dim=1).mean()


def kl_student_teacher(logp_s: torch.Tensor, logp_t: torch.Tensor) -> torch.Tensor:
    """Exact per-pixel forward KL( p_s || p_t ), mean over pixels.

    logp_s: [N, C, H, W] float32 (grad), logp_t: [N, C, H, W] (detached).
    """
    from losses.grpo_loss import pixel_kl
    return pixel_kl(logp_s, logp_t).mean()


def tv_loss(logits: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Total variation of the logit field (spatial regularizer).

    logits: [N, C, H, W] (grad).
    """
    lg = logits.float()
    d_h = (lg[:, :, 1:, :] - lg[:, :, :-1, :]).abs().mean()
    d_w = (lg[:, :, :, 1:] - lg[:, :, :, :-1]).abs().mean()
    return scale * (d_h + d_w)


class DistillLoss(nn.Module):
    def __init__(self, alpha_ce: float = 1.0, alpha_kl: float = 1.0,
                 alpha_tv: float = 1e-3):
        super().__init__()
        self.alpha_ce = float(alpha_ce)
        self.alpha_kl = float(alpha_kl)
        self.alpha_tv = float(alpha_tv)

    def forward(self, logp_s: torch.Tensor, logp_t: torch.Tensor,
                soft: torch.Tensor, logits_s: torch.Tensor = None) -> torch.Tensor:
        """soft=None -> the CE term is skipped (SAM3-free KL-only mode)."""
        loss = self.alpha_kl * kl_student_teacher(logp_s, logp_t)
        if soft is not None:
            loss = loss + self.alpha_ce * soft_ce(logp_s, soft)
        if logits_s is not None and self.alpha_tv > 0:
            loss = loss + self.alpha_tv * tv_loss(logits_s)
        return loss

    def terms(self, logp_s, logp_t, soft, logits_s):
        return {
            "ce": float(soft_ce(logp_s.detach(), soft).item())
            if soft is not None else 0.0,
            "kl": float(kl_student_teacher(logp_s.detach(), logp_t.detach()).item()),
            "tv": float(tv_loss(logits_s.detach()).item()) if logits_s is not None else 0.0,
        }
