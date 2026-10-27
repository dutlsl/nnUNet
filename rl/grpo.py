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

Advantage (DENSE per-pixel group centering, 2026-10-27):
    A_t^g(x) = clip(r_t^g(x) - mean_g r_t^g(x), +-adv_clip)
    The group mean is an unbiased per-pixel baseline (GRPO preserved
    pixel-wise). No std normalization: the semantic fold_1 init is
    near-deterministic (per-pixel entropy ~0.014 nats), so at a confident
    pixel all G samples agree and r - mean = 0 EXACTLY while the std is ~0 —
    a z-score with the std floor (adv_eps=1e-2) divided ~0 by the floor and
    emitted pure noise (measured group reward std 1.6e-4 => advantage std
    0.016, policy gradient ~ 0 from ep2 on). Raw centering gives A = 0 at
    confident pixels and O(1) only at uncertain (boundary) pixels.
    discounted_returns / group_advantage / compute_group_advantage are kept
    for the legacy frame-scalar path (and as reference implementations).

Loss (losses/grpo_loss.py, GRPOLoss.dense):
    L = -mean_g,x exp(logp_t^g(x) - logp_ref_t^g(x)) * A_t^g(x)
        + beta_kl * mean KL( pi_theta || pi_ref )  (exact per-pixel KL)
    logp_t^g(x)    = log_softmax(p_t)[x, m_t^g(x)]   (per-pixel, dense)
    logp_ref_t^g(x) = same gather on the frozen/EMA reference logits (no-grad)

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


def dense_pixel_advantage(
    rewards_pix: torch.Tensor,
    adv_clip: float = 5.0,
) -> torch.Tensor:
    """Per-pixel GROUP-CENTERED advantage (raw, no std normalization).

    rewards_pix: [G, *P] float (detached) — per-pixel reward of the G
    samples, e.g. [G, B, H, W] for one frame.

    A = clip(r - mean_g r, +-adv_clip)

    Why raw centering and not a z-score: at a confident pixel all G samples
    agree on the class, so r - mean = 0 EXACTLY and the std is ~0 — a
    z-score with a std floor (group_advantage) would divide a ~0 numerator
    by the floor and emit pure noise (measured: group reward std 1.6e-4 vs
    floor 1e-2 -> effective advantage std 0.016, policy gradient ~ 0). The
    raw centering gives A = 0 exactly where the policy is confident and
    O(1) only at uncertain (boundary) pixels — the gradient acts precisely
    where the semantic init disagrees with the teacher. The group mean is
    an unbiased per-pixel baseline (GRPO property preserved pixel-wise).
    """
    A = rewards_pix - rewards_pix.mean(dim=0, keepdim=True)
    return A.clamp(-adv_clip, adv_clip)


def categorical_logp_dense(
    logp: torch.Tensor,
    m: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel categorical log-prob of each sampled mask (WITH grad).

    logp: [B, C, H, W] float32 log-softmax (grad)
    m:    [G, B, H, W] long sampled class per pixel
    Returns [G, B, H, W] float32 = logp[g, b, m[g,b,x], x] per pixel.
    (The expand is a view; the gather is the only materialization.)
    """
    lp = logp.unsqueeze(0).expand(m.shape[0], *logp.shape)
    return lp.gather(2, m.unsqueeze(-1)).squeeze(-1)


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
