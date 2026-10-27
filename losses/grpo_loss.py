"""
GRPOLoss: policy-gradient update for the GAZER segmentation policy
(policy = the segmentation backbone itself; action = per-pixel mask).

L_GRPO  = -mean( exp(logp - logp_ref) * A )

  logp / logp_ref:  [B, T] float32 — PIXEL-MEAN categorical log-prob of the
                    sampled mask under the current / reference policy
                    ((1/P) sum_x log p[x, m(x)]), so the ratio is O(1) and
                    exp() stays stable.

L_KL    = beta_kl * mean_t KL( pi_theta_t || pi_ref_t )

  EXACT per-pixel forward KL (the K3 single-sample estimator is only valid
  for the old 2-token action, not per-pixel masks):

      KL_t = (1/P) sum_x sum_c p_theta[x,c] * (log p_theta[x,c] - log p_ref[x,c])

  >= 0, log-stable, computed with gradients (it is part of the loss).

L_Q     = lambda_q * MSE(...)  (disabled: lambda_q = 0 in the config; the
                               quality head is not used in this design).
All ratios in float32. A (advantage) is detached (it is a reward constant).
"""

import torch
import torch.nn as nn


def pixel_kl(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
) -> torch.Tensor:
    """Exact per-pixel forward KL( pi || pi_ref ) per frame, then mean over frames.

    logp:     [N, C, H, W] float32 log-softmax of the CURRENT policy (grad)
    logp_ref: [N, C, H, W] float32 log-softmax of the REFERENCE (detached)
    Returns:  [N] float32, each >= 0 (up to float error).
    """
    logp = logp.float()
    logp_ref = logp_ref.float().detach()
    p = logp.exp()
    # (1/P) sum_x sum_c : divide by the pixel count ONLY (C stays summed).
    # A pixel-SUM here would scale the KL by H*W (~2e5) and let the KL term
    # dominate the loss (effective beta_kl ~ 1e4x) — plan §7.2 is per-pixel.
    return (p * (logp - logp_ref)).sum(dim=(1, 2, 3)) / (
        float(logp.shape[2]) * float(logp.shape[3]))


class GRPOLoss(nn.Module):
    def __init__(self, beta_kl: float = 0.05, lambda_q: float = 0.0):
        super().__init__()
        self.beta_kl = beta_kl
        self.lambda_q = lambda_q

    def forward(
        self,
        logp: torch.Tensor,
        logp_ref: torch.Tensor,
        advantage: torch.Tensor,
        kl: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        logp:      [B, T] float32 current policy (pixel-mean log-prob of the
                   sampled mask)
        logp_ref:  [B, T] float32 reference policy (EMA, detached)
        advantage: [B, T] float32 detached
        kl:        [B, T] float32 exact per-pixel KL(pi||pi_ref), with grad
        mask:      [B, T] optional float (1 = valid)
        """
        logp = logp.float()
        logp_ref = logp_ref.float().detach()
        adv = advantage.detach().float()
        kl = kl.float()

        ratio = torch.exp((logp - logp_ref).clamp(-20, 20))
        pg = -(ratio * adv)
        if mask is not None:
            m = mask.float()
            pg = (pg * m).sum() / (m.sum() + 1e-8)
            kl = (kl * m).sum() / (m.sum() + 1e-8)
        else:
            pg = pg.mean()
            kl = kl.mean()

        loss = pg + self.beta_kl * kl
        return loss

    @staticmethod
    def kl_divergence(logp: torch.Tensor, logp_ref: torch.Tensor) -> float:
        """Mean exact per-pixel KL as a python float (for logging).

        logp / logp_ref: [N, C, H, W] float32 log-softmaxes (no grad needed).
        """
        with torch.no_grad():
            v = pixel_kl(logp.detach(), logp_ref.detach())
            return float(v.mean().item())
