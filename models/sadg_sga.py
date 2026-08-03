"""
SGA-2D: Spectral Graph Alignment for Test-Time Domain Adaptation.
Adapted from SAGD (Mamba Learns in Context) for eyeball segmentation.

Maintains a prototype bank learned during training (EMA update) and performs
spectral graph alignment at test time to align unseen domain features to
source domain prototypes.

During training: only updates prototype bank via EMA (minimal overhead).
During inference: performs full spectral alignment for domain adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SourcePrototypeBank(nn.Module):
    """
    Maintains EMA-updated class prototypes from source domain features.
    Each class has K prototypes to capture intra-class diversity.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_prototypes_per_class: int = 4,
        feature_dim: int = 256,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes_per_class
        self.feature_dim = feature_dim
        self.momentum = momentum

        # Prototype bank: [num_classes, K, C]
        self.register_buffer(
            'prototypes',
            torch.randn(num_classes, num_prototypes_per_class, feature_dim)
        )
        self.register_buffer(
            'prototype_initialized',
            torch.zeros(num_classes, dtype=torch.bool)
        )

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Update prototypes via EMA using labeled source domain features.

        Args:
            features: [B, N, C] token features
            labels: [B, H, W] segmentation labels (will be downsampled to match N)
            spatial_shape: (H_feat, W_feat) feature map dimensions
        """
        B, N, C = features.shape
        labels = labels[:B]

        if spatial_shape is not None:
            H_feat, W_feat = spatial_shape
        else:
            aspect_ratio = labels.shape[-1] / max(labels.shape[-2], 1)
            H_feat = max(int((N / aspect_ratio) ** 0.5), 1)
            W_feat = N // H_feat

        # Downsample labels to feature resolution
        labels_down = F.interpolate(
            labels.float().unsqueeze(1),
            size=(H_feat, W_feat),
            mode='nearest'
        ).squeeze(1).long()  # [B, H_feat, W_feat]
        labels_flat = labels_down.reshape(B, -1)  # [B, N]

        for cls_id in range(self.num_classes):
            # Gather all features belonging to this class
            cls_mask = (labels_flat == cls_id)  # [B, N]
            if not cls_mask.any():
                continue

            cls_features = features[cls_mask]  # [M, C]
            if cls_features.shape[0] == 0:
                continue

            # Simple K-means-style: split features into K groups
            K = self.num_prototypes
            M = cls_features.shape[0]

            if M < K:
                # Not enough samples: replicate
                new_protos = cls_features.mean(dim=0, keepdim=True).expand(K, -1)
            else:
                # Divide features evenly into K groups
                chunk_size = M // K
                new_protos = torch.stack([
                    cls_features[i * chunk_size:(i + 1) * chunk_size].mean(dim=0)
                    for i in range(K)
                ])

            # EMA update
            if self.prototype_initialized[cls_id]:
                self.prototypes[cls_id] = (
                    self.momentum * self.prototypes[cls_id]
                    + (1 - self.momentum) * new_protos
                )
            else:
                self.prototypes[cls_id] = new_protos
                self.prototype_initialized[cls_id] = True


class SpectralGraphAlignment(nn.Module):
    """
    SGA-2D: Test-time spectral graph alignment.
    Constructs affinity graphs for both target features and source prototypes,
    then aligns via spectral matching of graph Laplacian eigenvectors.
    """

    def __init__(
        self,
        num_classes = 4,
        feature_dim: int = 256,
        num_prototypes_per_class: int = 4,
        temperature: float = 0.1,
        spectral_k: int = 16,
        momentum: float = 0.999,
    ):
        super().__init__()
        if hasattr(num_classes, 'model'):
            cfg = num_classes
            num_classes = cfg.model.num_classes
            feature_dim = cfg.model.base_channels * 8
            sga_cfg = cfg.model.sga
            num_prototypes_per_class = sga_cfg.num_prototypes_per_class
            temperature = sga_cfg.alignment_temperature
            spectral_k = sga_cfg.spectral_k
            momentum = sga_cfg.prototype_momentum
        else:
            num_classes = int(num_classes)

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.spectral_k = spectral_k

        # Prototype bank
        self.prototype_bank = SourcePrototypeBank(
            num_classes=num_classes,
            num_prototypes_per_class=num_prototypes_per_class,
            feature_dim=feature_dim,
            momentum=momentum,
        )

        # Alignment projection
        self.align_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def _compute_graph_eigenvectors(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute top-k eigenvectors of the feature affinity graph Laplacian.

        Args:
            features: [M, C] normalized feature vectors
            k: number of eigenvectors

        Returns:
            eigvecs: [M, k] eigenvectors
        """
        M = features.shape[0]
        k = min(k, M)

        # Affinity matrix
        affinity = torch.mm(features, features.T)  # [M, M]
        affinity = torch.clamp(affinity, min=0.0)

        # Graph Laplacian
        degree = affinity.sum(dim=-1)
        d_inv_sqrt = 1.0 / (torch.sqrt(degree) + 1e-8)
        d_inv_sqrt_mat = torch.diag(d_inv_sqrt)
        laplacian_norm = torch.eye(M, device=features.device) - torch.mm(
            d_inv_sqrt_mat, torch.mm(affinity, d_inv_sqrt_mat)
        )

        # Symmetric for numerical stability
        laplacian_norm = (laplacian_norm + laplacian_norm.T) / 2.0

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_norm)
        except RuntimeError:
            laplacian_norm = laplacian_norm + 1e-6 * torch.eye(M, device=features.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_norm)

        return eigenvectors[:, :k]  # [M, k]

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, C] token features from the model
            labels: [B, H, W] ground truth labels (training only, for prototype update)
            spatial_shape: (H_feat, W_feat) optional spatial shape

        Returns:
            aligned_features: [B, N, C] spectrally aligned features
        """
        if self.training:
            # Training: only update prototypes, pass features through
            if labels is not None:
                self.prototype_bank.update(features.detach(), labels, spatial_shape=spatial_shape)
            return features

        # Test-time: spectral graph alignment
        B, N, C = features.shape
        device = features.device

        # Get all source prototypes: [num_classes * K, C]
        all_protos = self.prototype_bank.prototypes.reshape(-1, C)  # [num_classes*K, C]
        all_protos = F.normalize(all_protos, dim=-1)

        aligned_features = []
        for b in range(B):
            feat_b = features[b]  # [N, C]
            feat_b_norm = F.normalize(feat_b, dim=-1)

            # Compute spectral embeddings for target features
            target_eigvecs = self._compute_graph_eigenvectors(
                feat_b_norm, self.spectral_k
            )  # [N, k]

            # Compute spectral embeddings for source prototypes
            proto_eigvecs = self._compute_graph_eigenvectors(
                all_protos, min(self.spectral_k, all_protos.shape[0])
            )  # [P, k]

            # Compute alignment weights via spectral similarity
            # Pad proto_eigvecs if needed
            k_target = target_eigvecs.shape[1]
            k_proto = proto_eigvecs.shape[1]
            k_min = min(k_target, k_proto)

            spectral_sim = torch.mm(
                target_eigvecs[:, :k_min],
                proto_eigvecs[:, :k_min].T
            )  # [N, P]

            alignment_weights = F.softmax(spectral_sim / self.temperature, dim=-1)  # [N, P]

            # Align: weighted combination of prototypes
            proto_contribution = torch.mm(alignment_weights, all_protos)  # [N, C]

            # Blend aligned prototype information with original features
            feat_aligned = feat_b + self.align_proj(proto_contribution)
            aligned_features.append(feat_aligned)

        aligned_features = torch.stack(aligned_features)  # [B, N, C]
        return aligned_features
