"""
GRPOLoss: policy-gradient update for the GAZER segmentation policy
(policy = the segmentation backbone itself; action = per-pixel mask).

L_GRPO  = -mean( exp(logp - logp_ref) * A )

  DENSE per-pixel (2026-10-27, GRPOLoss.dense — used by train_gazer.grpo_step):
    logp / logp_ref: [G, N, H, W] float32 — per-pixel categorical log-prob
                     of the sampled mask (categorical_logp_dense) under the
                     current / reference policy
    advantage:       [G, N, H, W] dense group-centered advantage
                     (dense_pixel_advantage, raw centering, no std norm)
  Legacy frame-scalar path (GRPOLoss.forward): [B, T] pixel-mean
  ((1/P) sum_x log p[x, m(x)]) so the ratio is O(1).

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
    def __init__(self, beta_kl: float = 0.05, lambda_q: float = 0.0,
                 beta_ent: float = 0.0):
        super().__init__()
        self.beta_kl = beta_kl
        self.lambda_q = lambda_q
        self.beta_ent = beta_ent

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

    def dense(
        self,
        logp: torch.Tensor,
        logp_ref: torch.Tensor,
        advantage: torch.Tensor,
        kl: torch.Tensor,
        entropy: torch.Tensor = None,
    ) -> torch.Tensor:
        """Dense per-pixel variant (replaces the pixel-mean path).

        logp / logp_ref: [G, N, H, W] float32 per-pixel log-prob of each
                         sampled mask (categorical_logp_dense), ref detached
        advantage:       [G, N, H, W] float32 detached dense group-centered
                         advantage (dense_pixel_advantage)
        kl:              [N] float32 exact per-pixel KL(pi||pi_ref), grad —
                         expanded to [G, N] for the shared KL term
        entropy:        scalar float32, GRAD — mean full-categorical
                         per-pixel entropy of the policy, computed in
                         grpo_step from the [B,T,C,H,W] log-softmax (the
                         gathered logp above carries only the sampled class
                         and is not a distribution, so the entropy cannot be
                         derived here). Used by the 2026-10-29 anti-collapse
                         term (see below).
        All three terms are a plain mean, so the per-pixel normalization
        carries over unchanged.
        """
        logp = logp.float()
        logp_ref = logp_ref.float().detach()
        adv = advantage.detach().float()

        ratio = torch.exp((logp - logp_ref).clamp(-20, 20))
        pg = -(ratio * adv).mean()
        # KL is per-frame (same for every sample g); expand and mean.
        kl = kl.float().mean()
        # 2026-10-29 anti-collapse term. Raw-centering advantage is exactly
        # 0 at confident pixels, so once the policy becomes near-one-hot the
        # G rollouts agree per pixel -> A = 0 everywhere and the policy
        # gradient vanishes (observed run 20261027_170622: ep1 VAL 0.7934,
        # then grad/norm=0.0000, entropy=0.0000 from ep2 on). Minimizing
        # -beta_ent * H(pi) maximizes the policy's per-pixel entropy, so
        # minority classes get probability mass back, minority rollouts
        # reappear, and the group variance (the only GRPO signal under raw
        # centering) recovers. beta_kl alone (0.05 -> 0.01) was too weak to
        # counter the one-hot pull.
        if entropy is not None and self.beta_ent > 0.0:
            return pg + self.beta_kl * kl - self.beta_ent * entropy.float()
        return pg + self.beta_kl * kl

    @staticmethod
    def kl_divergence(logp: torch.Tensor, logp_ref: torch.Tensor) -> float:
        """Mean exact per-pixel KL as a python float (for logging).

        logp / logp_ref: [N, C, H, W] float32 log-softmaxes (no grad needed).
        """
        with torch.no_grad():
            v = pixel_kl(logp.detach(), logp_ref.detach())
            return float(v.mean().item())
