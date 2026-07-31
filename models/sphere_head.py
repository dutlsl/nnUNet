"""
Parametric Sphere Head for Eyeball Estimation.
Regresses 3 scalar parameters (cx, cy, r) from bottleneck features
and renders a differentiable circle mask via sigmoid approximation.

Design rationale:
  - The eyeball is a perfect sphere, so its 2D projection is always a circle.
  - By constraining the output to (cx, cy, r), Box Collapse is structurally impossible.
  - The DifferentiableCircleRenderer creates a smooth, gradient-friendly mask
    using a sigmoid envelope: mask(x,y) = σ(−k · (dist(x,y) − r)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableCircleRenderer(nn.Module):
    """
    Renders a 2D circle mask from parameters (cx, cy, r) using a differentiable
    sigmoid boundary. Gradients flow back to the circle parameters.

    mask(x, y) = sigmoid(-k * (sqrt((x - cx)^2 + (y - cy)^2) - r))

    Args:
        sharpness: Initial sigmoid steepness (k). Higher = sharper boundary.
    """

    def __init__(self, sharpness: float = 20.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        r: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Args:
            cx: [B] center x coordinates (in pixel space)
            cy: [B] center y coordinates (in pixel space)
            r:  [B] radius (in pixel space)
            H:  output height
            W:  output width

        Returns:
            mask: [B, 1, H, W] differentiable circle mask in [0, 1]
        """
        device = cx.device
        dtype = cx.dtype

        # Create coordinate grids [H, W]
        yy = torch.arange(H, device=device, dtype=dtype)
        xx = torch.arange(W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')  # [H, W]

        # Expand for batch: [B, H, W]
        grid_x = grid_x.unsqueeze(0).expand(cx.shape[0], -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(cy.shape[0], -1, -1)

        # Compute signed distance from circle boundary
        # dist > 0 outside circle, dist < 0 inside circle
        dx = grid_x - cx.view(-1, 1, 1)
        dy = grid_y - cy.view(-1, 1, 1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # [B, H, W]
        signed_dist = dist - r.view(-1, 1, 1)  # positive outside, negative inside

        # Sigmoid envelope: inside circle → 1.0, outside → 0.0
        mask = torch.sigmoid(-self.sharpness * signed_dist)  # [B, H, W]

        return mask.unsqueeze(1)  # [B, 1, H, W]


class SphereHead(nn.Module):
    """
    Regresses eyeball sphere parameters (cx, cy, r) from encoder bottleneck features.
    Uses Global Average Pooling → MLP to produce 3 bounded scalar values.

    Output ranges:
        cx ∈ [0, W], cy ∈ [0, H] via sigmoid scaling
        r  ∈ [min_radius, max_radius] via sigmoid scaling

    Args:
        feature_dim: Channel dimension of bottleneck features.
        hidden_dim:  MLP hidden layer dimension.
        max_radius:  Maximum eyeball radius in pixels.
        min_radius:  Minimum eyeball radius in pixels (prevents degenerate solutions).
        sharpness:   Sigmoid steepness for circle rendering.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        max_radius: float = 200.0,
        min_radius: float = 30.0,
        sharpness: float = 20.0,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius

        # GAP → MLP → 3 raw outputs (cx_raw, cy_raw, r_raw)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # (cx_raw, cy_raw, r_raw)
        )

        self.renderer = DifferentiableCircleRenderer(sharpness=sharpness)

        # Initialize MLP so initial predictions are roughly centered
        self._init_weights()

    def _init_weights(self):
        """Initialize final layer bias so initial prediction is centered with moderate radius."""
        # sigmoid(0) = 0.5, so zero init gives center of image and mid-range radius
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        H: int,
        W: int,
    ) -> dict:
        """
        Args:
            features: [B, C, Hf, Wf] bottleneck feature map
            H: target mask height (e.g., 448)
            W: target mask width (e.g., 448)

        Returns:
            dict with:
                'params': [B, 3] tensor of (cx, cy, r)
                'mask':   [B, 1, H, W] differentiable circle mask
        """
        B = features.shape[0]

        # Global Average Pooling → [B, C, 1, 1] → [B, C]
        pooled = self.gap(features).view(B, -1)

        # MLP → [B, 3]
        raw = self.mlp(pooled)

        # Bounded activations
        cx = torch.sigmoid(raw[:, 0]) * W   # [0, W]
        cy = torch.sigmoid(raw[:, 1]) * H   # [0, H]
        r = torch.sigmoid(raw[:, 2]) * (self.max_radius - self.min_radius) + self.min_radius  # [min_r, max_r]

        params = torch.stack([cx, cy, r], dim=1)  # [B, 3]

        # Render differentiable circle mask
        mask = self.renderer(cx, cy, r, H, W)  # [B, 1, H, W]

        return {
            'params': params,
            'mask': mask,
        }

    def set_sharpness(self, k: float):
        """Update renderer sharpness (for annealing during training)."""
        self.renderer.sharpness = k
