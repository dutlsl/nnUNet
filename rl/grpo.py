"""
GRPO (Group Relative Policy Optimization) for the GAZER segmentation policy.

Policy  pi_theta = the segmentation BACKBONE itself (unfrozen, semantic
          fold_1 init). Per-frame action = a full segmentation mask
          m_t^g ~ Cat(p_t) (G independent samples; they all share ONE
          backbone forward, so VRAM ~= a single forward).
Teacher ref_t    = GT one-hot (has_gt frames) else SAM3 multi-prompt soft4.
          LIVE teacher / reward landscape, recomputed on-policy at every step
          (RAM-cached because the per-frame reference is a run constant);
          this is a fixed SIMULATOR, never a static supervision target
          (plan §7.0: reward-landscape fit != supervision-data fit).

Reward (per-pixel DENSE teacher-likelihood — better credit assignment than a
frame-scalar Dice, plan §7.2):
    r_t^g(x) = alpha * ref_t[x, m_t^g(x)]        teacher likelihood of the
                                                  sampled class at pixel x
    r_t^g    = (1/P) sum_x r_t^g(x)              pixel-mean scalar per frame

Advantage:
    G_t^g = sum_{tau>=t} gamma_disc^{tau-t} * r_tau^g   (discounted returns)
    A_t^g = clip((G_t^g - mean_g) / (std_g + 1e-2), +-5)
            group z-score over the G rollouts of one window (std floor, clip)

Loss (losses/grpo_loss.py):
    L = -mean_g exp(logp_t^g - logp_ref_t^g) * A_t^g
        + beta_kl * mean_t KL( pi_theta_t || pi_ref_t )   (exact per-pixel KL)
    logp_t^g    = (1/P) sum_x log_softmax(p_t)[x, m_t^g(x)]   (pixel-mean,
                                                               keeps exp stable)
    logp_ref_t^g = same gather on the frozen/EMA reference logits (no-grad)

All reward / advantage / logp tensors are float32.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def categorical_logp(
    logp: torch.Tensor,
    m: torch.Tensor,
) -> torch.Tensor:
    """Pixel-mean categorical log-prob of the sampled mask.

    logp: [N, C, H, W] float32 = log_softmax of the per-frame logits (C = 4)
    m:    [N, H, W] long sampled class per pixel (values in [0, C))
    Returns [N] float32 = (1/(H*W)) sum_x logp[n, m[n, x], x].
    Pixel-mean normalization keeps the value O(1) so exp(ratio) in the loss
    stays in a stable range (a raw pixel-SUM would be ~-1e6 -> exp underflow).
    """
    logp_pix = logp.permute(0, 2, 3, 1)                      # [N, H, W, C]
    gathered = logp_pix.gather(3, m.unsqueeze(-1)).squeeze(-1)  # [N, H, W]
    return gathered.mean(dim=(1, 2))


def teacher_likelihood_reward(
    ref_soft: torch.Tensor,
    m: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Dense per-pixel teacher likelihood of the sampled mask.

    ref_soft: [N, C, H, W] float32 soft teacher (GT one-hot or SAM3 soft4,
              sums to 1 over C)
    m:        [N, H, W] long sampled class per pixel
    Returns [N] float32 = alpha * (1/(H*W)) sum_x ref_soft[n, m[n, x], x].
    """
    ref_pix = ref_soft.permute(0, 2, 3, 1)                   # [N, H, W, C]
    gathered = ref_pix.gather(3, m.unsqueeze(-1)).squeeze(-1)  # [N, H, W]
    return alpha * gathered.mean(dim=(1, 2))


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """G_t = sum_{tau>=t} gamma^{tau-t} * R_tau.  rewards [N, T] -> [N, T]."""
    N, T = rewards.shape
    G = torch.zeros_like(rewards)
    run = torch.zeros(N, dtype=torch.float32, device=rewards.device)
    for t in range(T - 1, -1, -1):
        run = rewards[:, t] + gamma * run
        G[:, t] = run
    return G


def group_advantage(
    returns: torch.Tensor,
    adv_eps: float = 1e-2,
    adv_clip: float = 5.0,
) -> torch.Tensor:
    """Per-window group z-score advantage.

    returns: [B, G, T] float32 (G rollouts of each of B windows)
    A[b, g, t] = clip((G - mean_g) / (std_g + eps), +-adv_clip)
    """
    mean = returns.mean(dim=1, keepdim=True)
    std = returns.std(dim=1, keepdim=True)
    A = (returns - mean) / (std + adv_eps)
    return A.clamp(-adv_clip, adv_clip)


def compute_group_advantage(
    rewards: torch.Tensor,
    gamma: float,
    adv_eps: float = 1e-2,
    adv_clip: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """rewards [B, G, T] -> (returns [B, G, T], advantage [B, G, T])."""
    B, G, T = rewards.shape
    ret = discounted_returns(rewards.reshape(B * G, T), gamma).view(B, G, T)
    A = group_advantage(ret, adv_eps, adv_clip)
    return ret, A


def dice_agreement(pred: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Per-frame Dice between student softmax and a soft teacher map.
    (Kept for validation-time agreement metrics only.)
    pred:    [B, C, H, W] (logits or probs; softmax applied if logits)
    teacher: [B, C, H, W] soft, sums to 1 over C
    Returns [B] in [0, 1].
    """
    probs = F.softmax(pred, dim=1)
    smooth = 1e-5
    inter = (probs * teacher).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + teacher.sum(dim=(1, 2, 3))
    return (2.0 * inter + smooth) / (denom + smooth)
