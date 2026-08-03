"""
HDM-2D: Hierarchical Domain Modeling for Multi-Domain Mamba Processing.
Adapted from SAGD (Mamba Learns in Context) for eyeball segmentation.

Implements two-stage hierarchical Mamba processing:
1. ISM (Intra-domain Sequence Modeling): Each domain's tokens are processed
   independently through domain-specific Mamba blocks, learning domain-specific
   temporal dynamics.
2. IRF (Inter-domain Representation Fusion): Tokens from all domains are
   interleaved and processed through a shared Mamba block, enabling cross-domain
   knowledge transfer while preserving domain-specific features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import MambaLayer
from typing import List, Tuple


class IntraDomainMamba(nn.Module):
    """
    ISM: Independent Mamba block for a single domain.
    Processes serialized tokens from one domain through LayerNorm + Mamba + residual.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B_d, N, C] tokens from a single domain

        Returns:
            out: [B_d, N, C] domain-enhanced tokens
        """
        residual = x
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        out = residual + self.proj(x_mamba)
        return out


class InterDomainFusion(nn.Module):
    """
    IRF: Inter-domain Representation Fusion via interleaved Mamba.
    Interleaves token chunks from different domains to enable cross-domain
    information flow through the Mamba's sequential state.
    """

    def __init__(
        self,
        d_model: int,
        num_domains: int = 3,
        chunk_size: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.num_domains = num_domains
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.norm = nn.LayerNorm(d_model)
        self.shared_mamba = MambaLayer(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.proj = nn.Linear(d_model, d_model)

    def _interleave(self, domain_tokens: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        """
        Interleave chunks from multiple domains.

        Args:
            domain_tokens: List of [B_d, N, C] tensors (one per domain)

        Returns:
            interleaved: [B_min, N_total, C] interleaved tokens
            chunk_counts: Number of chunks per domain for de-interleaving
        """
        # Use minimum batch size across domains
        B_min = min(t.shape[0] for t in domain_tokens)
        domain_tokens = [t[:B_min] for t in domain_tokens]

        chunks_per_domain = []
        all_chunks = []

        for d_idx, tokens in enumerate(domain_tokens):
            N = tokens.shape[1]
            # Split into chunks along sequence dimension
            n_chunks = (N + self.chunk_size - 1) // self.chunk_size
            chunks = torch.split(tokens, self.chunk_size, dim=1)
            chunks_per_domain.append(len(chunks))
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append((d_idx, c_idx, chunk))

        # Interleave: round-robin across domains
        max_chunks = max(chunks_per_domain)
        interleaved_chunks = []
        for c_idx in range(max_chunks):
            for d_idx in range(len(domain_tokens)):
                if c_idx < chunks_per_domain[d_idx]:
                    # Find the matching chunk
                    for d, c, chunk in all_chunks:
                        if d == d_idx and c == c_idx:
                            interleaved_chunks.append(chunk)
                            break

        # Concatenate along sequence dimension
        interleaved = torch.cat(interleaved_chunks, dim=1)  # [B_min, N_total, C]
        return interleaved, chunks_per_domain

    def _deinterleave(
        self, interleaved: torch.Tensor, chunks_per_domain: List[int], original_lengths: List[int]
    ) -> List[torch.Tensor]:
        """
        De-interleave fused tokens back to per-domain sequences.

        Args:
            interleaved: [B, N_total, C]
            chunks_per_domain: chunks count per domain
            original_lengths: original sequence length per domain

        Returns:
            domain_tokens: List of [B, N_d, C] tensors
        """
        B = interleaved.shape[0]
        max_chunks = max(chunks_per_domain)
        num_domains = len(chunks_per_domain)

        # Split back into chunks
        chunk_list = []
        offset = 0
        for c_idx in range(max_chunks):
            for d_idx in range(num_domains):
                if c_idx < chunks_per_domain[d_idx]:
                    remaining = original_lengths[d_idx] - c_idx * self.chunk_size
                    chunk_len = min(self.chunk_size, remaining)
                    chunk = interleaved[:, offset:offset + chunk_len, :]
                    chunk_list.append((d_idx, chunk))
                    offset += chunk_len

        # Reassemble per domain
        domain_chunks = {d: [] for d in range(num_domains)}
        for d_idx, chunk in chunk_list:
            domain_chunks[d_idx].append(chunk)

        domain_tokens = []
        for d_idx in range(num_domains):
            if domain_chunks[d_idx]:
                domain_tokens.append(torch.cat(domain_chunks[d_idx], dim=1))
            else:
                domain_tokens.append(torch.zeros(B, 0, self.d_model, device=interleaved.device))

        return domain_tokens

    def forward(self, domain_tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            domain_tokens: List of [B_d, N, C] domain-specific token sequences

        Returns:
            fused_tokens: List of [B_min, N, C] cross-domain fused tokens
        """
        original_lengths = [t.shape[1] for t in domain_tokens]

        # Interleave
        interleaved, chunks_per_domain = self._interleave(domain_tokens)

        # Shared Mamba processing
        residual = interleaved
        x_norm = self.norm(interleaved)
        x_mamba = self.shared_mamba(x_norm)
        fused = residual + self.proj(x_mamba)

        # De-interleave
        fused_tokens = self._deinterleave(fused, chunks_per_domain, original_lengths)

        return fused_tokens


class HDM2D(nn.Module):
    """
    HDM-2D: Full Hierarchical Domain Modeling module.
    Combines ISM (per-domain Mamba) + IRF (cross-domain interleaved Mamba).

    During training: receives multi-domain serialized tokens, outputs domain-fused features.
    During inference (single domain): bypasses IRF, uses only ISM for the source domain.
    """

    def __init__(self, cfg):
        super().__init__()
        hdm_cfg = cfg.model.hdm
        d_model = cfg.model.base_channels * 8  # 256

        self.num_domains = hdm_cfg.num_domains

        # ISM: One Mamba block per domain
        self.ism_blocks = nn.ModuleList([
            IntraDomainMamba(
                d_model=d_model,
                d_state=hdm_cfg.mamba_d_state,
                d_conv=hdm_cfg.mamba_d_conv,
                expand=hdm_cfg.mamba_expand,
            )
            for _ in range(self.num_domains)
        ])

        # IRF: Shared cross-domain Mamba
        self.irf = InterDomainFusion(
            d_model=d_model,
            num_domains=self.num_domains,
            chunk_size=hdm_cfg.interleave_chunk_size,
            d_state=hdm_cfg.mamba_d_state,
            d_conv=hdm_cfg.mamba_d_conv,
            expand=hdm_cfg.mamba_expand,
        )

        # Final fusion projection
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        serialized_tokens: List[torch.Tensor],
        domain_ids: List[int],
    ) -> torch.Tensor:
        """
        Args:
            serialized_tokens: List of [B_d, N, C] serialized tokens per domain
                Each entry corresponds to one domain's batch of serialized features.
            domain_ids: List of domain indices (0=OpenEDS, 1=Swirski, 2=LPW)

        Returns:
            fused_output: [B_total, N, C] domain-fused tokens for primary domain
        """
        # ISM: Process each domain independently
        ism_outputs = []
        for tokens, d_id in zip(serialized_tokens, domain_ids):
            ism_out = self.ism_blocks[d_id](tokens)
            ism_outputs.append(ism_out)

        if self.training and len(ism_outputs) > 1:
            # IRF: Cross-domain fusion during training
            fused_list = self.irf(ism_outputs)
            # Return primary domain (OpenEDS = domain 0) fused output
            primary_output = self.fusion_norm(fused_list[0])
        else:
            # Inference or single-domain: use ISM output directly
            primary_output = self.fusion_norm(ism_outputs[0])

        return primary_output

    def forward_single_domain(self, tokens: torch.Tensor, domain_id: int = 0) -> torch.Tensor:
        """
        Single-domain forward pass (for inference or ablation).

        Args:
            tokens: [B, N, C] serialized tokens
            domain_id: which domain's ISM block to use

        Returns:
            output: [B, N, C]
        """
        ism_out = self.ism_blocks[domain_id](tokens)
        return self.fusion_norm(ism_out)
