"""
GazerHead: autoregressive per-frame ACTION policy head (GAZER).

Consumes the Mamba3 bottleneck features b_mamba [B, T, C, H, W] and produces,
for each frame t:
  - action logits a_logits [B, T, 2] over {QUERY(=0, call teacher), STOP(=1)}
  - quality q_t [B, T, 1] in [0, 1] (expected teacher-student agreement)

Design mirrors AutoGaze (Attend Before Attention): a causal context encoder over
the temporal token stream plus an autoregressive (GRU) decoder that emits one
action token per frame. STOP forces all subsequent actions to STOP (masking).

All log-prob / sampling arithmetic is upcast to float32 (FP16-safe ratios).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_QUERY = 0
ACTION_STOP = 1


@dataclass
class GazerSample:
    """Rollout result for one (or a group of) sequences.
    actions: [B, T] long, values in {0=QUERY, 1=STOP}
    logp:    [B, T] float32 log-probability of each action under the policy
    q:       [B, T] float32 quality prediction in [0, 1]
    stop_index: [B] long, first STOP index per sequence (T if none)
    """
    actions: torch.Tensor
    logp: torch.Tensor
    q: torch.Tensor
    stop_index: torch.Tensor


class GazerHead(nn.Module):
    def __init__(
        self,
        d_bottleneck: int = 256,
        d_z: int = 256,
        n_transformer_layers: int = 2,
        pool_size: int = 14,
        hidden: int = 256,
    ):
        super().__init__()
        # 56/14 = 4 -> 4x4 = 16 tokens per frame.
        # AvgPool2d keeps channels first: [B*T, C, h, w] -> permute to token-major.
        self.pool_size = pool_size
        self.token_proj = nn.Sequential(
            nn.Linear(d_bottleneck, d_z),
            nn.LayerNorm(d_z),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_z, nhead=max(1, d_z // 64), dim_feedforward=2 * d_z,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)

        self.decoder_in = nn.Linear(d_z + 2 + 1, hidden)   # ctx_t + prev action + prev q
        self.gru = nn.GRUCell(hidden, hidden)
        self.action_head = nn.Linear(hidden, 2)
        self.quality_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self._hidden = hidden

    # ------------------------------------------------------------------
    def _tokenize(self, b_mamba: torch.Tensor) -> torch.Tensor:
        """[B, T, C, H, W] -> [B, T, N_tok, d_z] (H/W = 56 -> 4x4 = 16 tokens)"""
        B, T, C, H, W = b_mamba.shape
        x = b_mamba.reshape(B * T, C, H, W)
        x = F.adaptive_avg_pool2d(x, (H // self.pool_size, W // self.pool_size))
        x = x.flatten(2).permute(0, 2, 1)           # [B*T, N_tok, C]
        x = self.token_proj(x)                      # [B*T, N_tok, d_z]
        N_tok = x.shape[1]
        return x.view(B, T, N_tok, x.shape[-1])

    def _causal_mask(self, T: int, N_tok: int, device) -> torch.Tensor:
        L = T * N_tok
        m = torch.zeros(L, L, dtype=torch.bool, device=device)
        for t in range(T):
            for u in range(T):
                if u > t:
                    m[t * N_tok:(t + 1) * N_tok, u * N_tok:(u + 1) * N_tok] = True
        return m

    def _forward_frames(self, b_mamba: torch.Tensor) -> torch.Tensor:
        """[B, T, C, H, W] -> ctx [B, T, d_z] (causal transformer, frame-wise mean)."""
        z = self._tokenize(b_mamba)                 # [B, T, N, d_z]
        B, T, N, D = z.shape
        z_flat = z.reshape(B, T * N, D)
        # Causal mask: frame t's tokens may only attend to frames u <= t.
        neg = torch.full((z_flat.shape[1], z_flat.shape[1]), float("-inf"),
                         device=b_mamba.device, dtype=b_mamba.dtype)
        attn = torch.where(self._causal_mask(T, N, b_mamba.device), neg, torch.tensor(0.0))
        ctx = self.context_encoder(z_flat, mask=attn)
        return ctx.view(B, T, N, D).mean(dim=2)     # [B, T, d_z]

    def forward(self, b_mamba: torch.Tensor):
        """
        Args:
            b_mamba: [B, T, C, H, W]
        Returns:
            a_logits: [B, T, 2]
            q:        [B, T, 1] in [0, 1]
        """
        ctx = self._forward_frames(b_mamba)       # [B, T, d_z]
        B, T, D = ctx.shape
        h = torch.zeros(B, self.gru.hidden_size, device=b_mamba.device,
                        dtype=b_mamba.dtype)
        prev_a = torch.zeros(B, 2, device=b_mamba.device, dtype=b_mamba.dtype)
        prev_q = torch.zeros(B, 1, device=b_mamba.device, dtype=b_mamba.dtype)

        a_logits_list, q_list = [], []
        for t in range(T):
            inp = torch.cat([ctx[:, t], prev_a, prev_q], dim=-1)
            h = self.gru(self.decoder_in(inp), h)
            logits = self.action_head(h)            # [B, 2]
            q = self.quality_head(h)                # [B, 1]
            a_logits_list.append(logits)
            q_list.append(q)
            prev_a = F.one_hot(logits.argmax(-1), 2).to(b_mamba.dtype)
            prev_q = q                                   # keep [B, 1]

        a_logits = torch.stack(a_logits_list, dim=1)
        q = torch.stack(q_list, dim=1)
        return a_logits, q

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, b_mamba: torch.Tensor, temperature: float = 1.0,
               greedy: bool = False) -> GazerSample:
        """On-policy sampling of the action sequence.
        All log-prob math in float32.
        """
        ctx = self._forward_frames(b_mamba)
        B, T, D = ctx.shape
        dev = b_mamba.device
        h = torch.zeros(B, self.gru.hidden_size, device=dev, dtype=ctx.dtype)
        prev_a = torch.zeros(B, 2, device=dev, dtype=ctx.dtype)
        prev_q = torch.zeros(B, 1, device=dev, dtype=ctx.dtype)

        actions, logps, qs = [], [], []
        any_stopped = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(T):
            inp = torch.cat([ctx[:, t], prev_a, prev_q], dim=-1)
            h = self.gru(self.decoder_in(inp), h)
            logits = self.action_head(h).float()    # [B, 2] float32
            q = self.quality_head(h).float()        # [B, 1]

            if greedy:
                a = logits.argmax(-1)
            else:
                probs = F.softmax(logits / max(temperature, 1e-3), dim=-1)
                a = torch.multinomial(probs, 1).squeeze(-1)
            # Force STOP once STOP has been taken (or already stopped)
            a = torch.where(any_stopped, torch.ones_like(a), a)
            a = torch.where(a == ACTION_STOP, torch.ones_like(a), a)
            # log-prob of chosen action
            logp = F.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(1)).squeeze(1)
            actions.append(a)
            logps.append(logp)
            qs.append(q.squeeze(-1))
            any_stopped = any_stopped | (a == ACTION_STOP)
            prev_a = F.one_hot(a, 2).to(ctx.dtype)
            prev_q = q.squeeze(-1).unsqueeze(-1).to(ctx.dtype)

        actions_t = torch.stack(actions, dim=1)
        logp_t = torch.stack(logps, dim=1)
        q_t = torch.stack(qs, dim=1)
        # stop_index = first STOP index (T if never STOP)
        stop_idx = torch.where(
            (actions_t == ACTION_STOP).any(dim=1),
            (actions_t == ACTION_STOP).float().argmax(dim=1),
            torch.full((B,), T, dtype=torch.long, device=dev),
        )
        return GazerSample(actions=actions_t, logp=logp_t, q=q_t, stop_index=stop_idx)

    # ------------------------------------------------------------------
    def action_log_probs(self, b_mamba: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log-prob of a given action sequence under the policy (float32).
        Args:
            actions: [B, T] long
        Returns:
            [B, T] float32
        """
        ctx = self._forward_frames(b_mamba)
        B, T, D = ctx.shape
        dev = b_mamba.device
        h = torch.zeros(B, self.gru.hidden_size, device=dev, dtype=ctx.dtype)
        prev_a = torch.zeros(B, 2, device=dev, dtype=ctx.dtype)
        prev_q = torch.zeros(B, 1, device=dev, dtype=ctx.dtype)
        logps = []
        for t in range(T):
            inp = torch.cat([ctx[:, t], prev_a, prev_q], dim=-1)
            h = self.gru(self.decoder_in(inp), h)
            logits = self.action_head(h).float()
            q = self.quality_head(h).float()
            logp = F.log_softmax(logits, dim=-1).gather(1, actions[:, t].unsqueeze(1)).squeeze(1)
            logps.append(logp)
            prev_a = F.one_hot(actions[:, t], 2).to(ctx.dtype)
            prev_q = q.squeeze(-1).unsqueeze(-1).to(ctx.dtype)
        return torch.stack(logps, dim=1)
