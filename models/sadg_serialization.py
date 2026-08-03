"""
SAS-2D: Structure-Aware Serialization for 2D Feature Maps.
Adapted from SAGD (Mamba Learns in Context) for eyeball segmentation.

Implements two complementary serialization strategies:
1. CDS-2D (Centroid Distance Serialization): BFS-based ordering from pupil centroid
2. GCS-2D (Graph Cut Serialization): Heat-diffusion-based ordering via Laplacian eigenvectors

Both strategies provide structure-aware token orderings for Mamba sequential processing,
replacing naive raster-scan with anatomically meaningful sequences that respect
the concentric structure of the eye (pupil -> iris -> sclera -> background).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class CDS2D(nn.Module):
    """
    Centroid Distance Serialization (CDS-2D).
    Orders spatial tokens by BFS-like distance from the estimated pupil centroid.
    For eyeball images, this creates a concentric ordering:
    pupil center -> pupil boundary -> iris -> sclera -> background.
    """

    def __init__(self, feature_h: int, feature_w: int, num_neighbors: int = 8):
        super().__init__()
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.num_neighbors = num_neighbors
        self.num_tokens = feature_h * feature_w

        # Learnable centroid estimator: predicts (cy, cx) from feature statistics
        self.centroid_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),      # (cy_norm, cx_norm) in [0, 1]
            nn.Sigmoid(),
        )

        # Precompute grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(feature_h, dtype=torch.float32),
            torch.arange(feature_w, dtype=torch.float32),
            indexing='ij'
        )
        # [N, 2] grid positions
        self.register_buffer('grid_coords', torch.stack([
            grid_y.reshape(-1) / max(feature_h - 1, 1),
            grid_x.reshape(-1) / max(feature_w - 1, 1),
        ], dim=-1))

    def forward(self, features: torch.Tensor) -> torch.LongTensor:
        """
        Compute CDS ordering for each sample in the batch.

        Args:
            features: [B, C, H, W] bottleneck feature map

        Returns:
            order: [B, N] indices for serialization (N = H*W)
        """
        B = features.shape[0]
        device = features.device

        # Estimate centroid from features
        centroid = self.centroid_head(features)  # [B, 2] -> (cy_norm, cx_norm)

        # Compute distance from each grid position to the estimated centroid
        # grid_coords: [N, 2], centroid: [B, 2]
        grid = self.grid_coords.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
        cent = centroid.unsqueeze(1)  # [B, 1, 2]

        # Euclidean distance from centroid
        dist = torch.norm(grid - cent, dim=-1)  # [B, N]

        # Sort by distance -> concentric ordering from center outward
        order = torch.argsort(dist, dim=-1)  # [B, N]

        return order


class GCS2D(nn.Module):
    """
    Graph Cut Serialization (GCS-2D).
    Constructs an affinity graph over spatial tokens, computes the Laplacian,
    and orders tokens by the Fiedler vector (2nd smallest eigenvector).
    This produces a globally smooth serialization that respects feature similarity boundaries.
    """

    def __init__(
        self,
        feature_h: int,
        feature_w: int,
        diffusion_time: float = 2.0,
        num_eigenvectors: int = 32,
    ):
        super().__init__()
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.num_tokens = feature_h * feature_w
        self.diffusion_time = diffusion_time
        self.num_eigenvectors = min(num_eigenvectors, self.num_tokens)

        # Learnable affinity projection (reduces channel dim for efficient graph computation)
        self.affinity_proj = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Precompute spatial adjacency mask (8-connected grid)
        adj_mask = self._build_grid_adjacency(feature_h, feature_w)
        self.register_buffer('adj_mask', adj_mask)

    @staticmethod
    def _build_grid_adjacency(h: int, w: int) -> torch.Tensor:
        """Build 8-connected spatial adjacency mask for HxW grid."""
        n = h * w
        adj = torch.zeros(n, n, dtype=torch.bool)
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            nidx = ni * w + nj
                            adj[idx, nidx] = True
        return adj

    def forward(self, features: torch.Tensor) -> torch.LongTensor:
        """
        Compute GCS ordering via Fiedler vector of feature-weighted Laplacian.

        Args:
            features: [B, C, H, W] bottleneck feature map

        Returns:
            order: [B, N] indices for serialization
        """
        B = features.shape[0]
        device = features.device

        # Project features for affinity computation
        feat_proj = self.affinity_proj(features)  # [B, 64, H, W]
        feat_flat = feat_proj.reshape(B, feat_proj.shape[1], -1)  # [B, 64, N]
        feat_flat = feat_flat.permute(0, 2, 1)  # [B, N, 64]

        # Normalize features
        feat_norm = F.normalize(feat_flat, dim=-1)  # [B, N, 64]

        # Compute pairwise cosine similarity (affinity matrix)
        affinity = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]

        # Apply spatial adjacency mask + heat kernel
        adj_mask = self.adj_mask.unsqueeze(0).expand(B, -1, -1).to(device)
        affinity = affinity * adj_mask.float()
        affinity = torch.exp(affinity * self.diffusion_time)
        affinity = affinity * adj_mask.float()  # Zero out non-adjacent

        # Ensure symmetry and non-negative
        affinity = (affinity + affinity.transpose(1, 2)) / 2.0
        affinity = torch.clamp(affinity, min=0.0)

        # Compute graph Laplacian: L = D - W
        degree = affinity.sum(dim=-1)  # [B, N]
        degree_mat = torch.diag_embed(degree)  # [B, N, N]
        laplacian = degree_mat - affinity

        # Normalized Laplacian: D^{-1/2} L D^{-1/2}
        d_inv_sqrt = torch.diag_embed(1.0 / (torch.sqrt(degree) + 1e-8))
        laplacian_norm = torch.bmm(d_inv_sqrt, torch.bmm(laplacian, d_inv_sqrt))

        # Eigendecomposition — use Fiedler vector (2nd smallest eigenvector)
        # For N=576, this is computationally tractable
        laplacian_norm = (laplacian_norm + laplacian_norm.transpose(1, 2)) / 2.0
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_norm)
        except RuntimeError:
            # Fallback: add small diagonal perturbation for numerical stability
            laplacian_norm = laplacian_norm + 1e-6 * torch.eye(
                self.num_tokens, device=device
            ).unsqueeze(0)
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_norm)

        # Fiedler vector = 2nd eigenvector (index 1, since index 0 is constant)
        fiedler = eigenvectors[:, :, 1]  # [B, N]

        # Sort by Fiedler vector value -> graph-cut-aware serialization
        order = torch.argsort(fiedler, dim=-1)  # [B, N]

        return order


class StructureAwareSerializer(nn.Module):
    """
    SAS-2D: Combines CDS and GCS serializations.
    Produces 4 serialized sequences: [fwd_CDS, rev_CDS, fwd_GCS, rev_GCS]
    that are concatenated along the sequence dimension for Mamba processing.
    """

    def __init__(self, cfg):
        super().__init__()
        sas_cfg = cfg.model.sas
        feature_h = sas_cfg.feature_h
        feature_w = sas_cfg.feature_w

        self.cds = CDS2D(
            feature_h=feature_h,
            feature_w=feature_w,
            num_neighbors=sas_cfg.cds_num_neighbors,
        )
        self.gcs = GCS2D(
            feature_h=feature_h,
            feature_w=feature_w,
            diffusion_time=sas_cfg.gcs_diffusion_time,
            num_eigenvectors=sas_cfg.gcs_num_eigenvectors,
        )

        self.num_tokens = feature_h * feature_w
        self.cache_serialization = sas_cfg.cache_serialization
        self._cached_cds_order = None
        self._cached_gcs_order = None

    def clear_cache(self):
        """Clear cached serialization orders (call at epoch start)."""
        self._cached_cds_order = None
        self._cached_gcs_order = None

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Serialize feature map tokens using both CDS and GCS strategies.

        Args:
            features: [B, C, H, W] bottleneck feature map

        Returns:
            fwd_cds: [B, N, C] forward CDS-serialized tokens
            rev_cds: [B, N, C] reverse CDS-serialized tokens
            fwd_gcs: [B, N, C] forward GCS-serialized tokens
            rev_gcs: [B, N, C] reverse GCS-serialized tokens
        """
        B, C, H, W = features.shape
        N = H * W

        # Flatten spatial dims: [B, C, N] -> [B, N, C]
        tokens = features.reshape(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Compute serialization orders
        cds_order = self.cds(features)  # [B, N]
        gcs_order = self.gcs(features)  # [B, N]

        # Gather tokens in serialized order
        # Expand order for gathering: [B, N, 1] -> [B, N, C]
        cds_idx = cds_order.unsqueeze(-1).expand(-1, -1, C)
        gcs_idx = gcs_order.unsqueeze(-1).expand(-1, -1, C)

        fwd_cds = torch.gather(tokens, 1, cds_idx)
        rev_cds = torch.flip(fwd_cds, dims=[1])
        fwd_gcs = torch.gather(tokens, 1, gcs_idx)
        rev_gcs = torch.flip(fwd_gcs, dims=[1])

        return fwd_cds, rev_cds, fwd_gcs, rev_gcs
