"""
Unit tests for WeakMEd loss components:
1. Mask-to-Box (M2B) Transform — paper Eq. 2
2. Scale Consistency Loss — paper Eq. 4
3. WeakMedEyeballLoss — integrated loss
4. Eyeball Bounding Box extraction from segmentation labels
"""

import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, '/home/iulab0/PycharmProjects/nnUNet')

from losses.weakmed_loss import (
    mask_to_box_transform,
    create_box_mask,
    MaskToBoxLoss,
    ScaleConsistencyLoss,
    WeakMedEyeballLoss,
    DiceBCELoss,
    DiceCELoss,
)
from datasets.openeds_dataset import OpenEDS400SequenceDataset


class TestMaskToBoxTransform:
    """Test M2B transform (WeakMEd Eq. 2)."""

    def test_output_shape(self):
        """M2B output should match input shape."""
        B, H, W = 2, 64, 64
        pred = torch.rand(B, 1, H, W)
        bbox = torch.tensor([[10, 10, 50, 50], [5, 5, 60, 60]], dtype=torch.float32)
        out = mask_to_box_transform(pred, bbox, H, W)
        assert out.shape == (B, 1, H, W), f"Expected {(B, 1, H, W)}, got {out.shape}"

    def test_output_range(self):
        """M2B output should be in [0, 1] since it uses min of max-projected values."""
        pred = torch.rand(4, 1, 32, 32)
        bbox = torch.tensor([[5, 5, 25, 25]] * 4, dtype=torch.float32)
        out = mask_to_box_transform(pred, bbox, 32, 32)
        assert out.min() >= 0.0, f"M2B output min={out.min()}, expected >= 0"
        assert out.max() <= 1.0, f"M2B output max={out.max()}, expected <= 1"

    def test_zeros_outside_bbox(self):
        """M2B should produce zeros outside the bounding box."""
        pred = torch.ones(1, 1, 32, 32) * 0.8
        bbox = torch.tensor([[8, 8, 24, 24]], dtype=torch.float32)
        out = mask_to_box_transform(pred, bbox, 32, 32)
        # Outside bbox should be zero
        assert out[0, 0, 0, 0] == 0.0, "M2B should be zero outside bbox"
        assert out[0, 0, 31, 31] == 0.0, "M2B should be zero outside bbox"
        # Inside bbox should be non-zero
        assert out[0, 0, 16, 16] > 0.0, "M2B should be non-zero inside bbox"

    def test_uniform_prediction_fills_box(self):
        """When prediction is uniform inside bbox, M2B should also be uniform."""
        pred = torch.zeros(1, 1, 32, 32)
        pred[0, 0, 10:22, 10:22] = 0.9
        bbox = torch.tensor([[10, 10, 22, 22]], dtype=torch.float32)
        out = mask_to_box_transform(pred, bbox, 32, 32)
        # min(max_col, max_row) of uniform = min(0.9, 0.9) = 0.9
        interior = out[0, 0, 10:22, 10:22]
        assert torch.allclose(interior, torch.tensor(0.9), atol=1e-5), \
            f"Uniform prediction should give uniform M2B, got range [{interior.min()}, {interior.max()}]"

    def test_empty_bbox(self):
        """Empty/zero bbox should produce all-zero output."""
        pred = torch.rand(1, 1, 32, 32)
        bbox = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
        out = mask_to_box_transform(pred, bbox, 32, 32)
        assert out.sum() == 0.0, "Empty bbox should produce all-zero M2B output"


class TestCreateBoxMask:
    """Test box mask creation utility."""

    def test_box_mask_shape(self):
        bbox = torch.tensor([[5, 5, 20, 20]], dtype=torch.float32)
        mask = create_box_mask(bbox, 32, 32, torch.device('cpu'))
        assert mask.shape == (1, 1, 32, 32)

    def test_box_mask_values(self):
        bbox = torch.tensor([[5, 5, 15, 15]], dtype=torch.float32)
        mask = create_box_mask(bbox, 32, 32, torch.device('cpu'))
        assert mask[0, 0, 10, 10] == 1.0
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 20, 20] == 0.0


class TestScaleConsistencyLoss:
    """Test Scale Consistency loss (WeakMEd Eq. 4)."""

    def test_output_nonnegative(self):
        """KL divergence should be non-negative."""
        sc = ScaleConsistencyLoss(scale_factor=0.5)
        logits = torch.randn(2, 1, 32, 32)
        bbox = torch.tensor([[4, 4, 28, 28], [4, 4, 28, 28]], dtype=torch.float32)
        loss = sc(logits, bbox, 32, 32)
        assert loss >= 0.0, f"SC loss should be non-negative, got {loss}"

    def test_identical_predictions_zero_loss(self):
        """Identical predictions should give near-zero SC loss (when not using approx downscale)."""
        # This tests the approximate mode which downscales/upscales, so loss won't be exactly 0
        sc = ScaleConsistencyLoss(scale_factor=1.0)  # No actual scaling
        logits = torch.ones(2, 1, 32, 32) * 2.0
        bbox = torch.tensor([[4, 4, 28, 28], [4, 4, 28, 28]], dtype=torch.float32)
        loss = sc(logits, bbox, 32, 32)
        assert loss < 0.01, f"Same-scale SC should be near-zero, got {loss}"


class TestWeakMedEyeballLoss:
    """Test integrated WeakMEd loss."""

    def test_forward_returns_dict(self):
        """WeakMedEyeballLoss should return dict with total, m2b, sc keys."""
        wm = WeakMedEyeballLoss(m2b_weight=1.0, sc_weight=1.0)
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        bbox = torch.tensor([[4, 4, 28, 28], [4, 4, 28, 28]], dtype=torch.float32)
        result = wm(logits, bbox)
        assert 'total' in result
        assert 'm2b' in result
        assert 'sc' in result
        assert result['total'].requires_grad  # Must be differentiable

    def test_gradient_flow(self):
        """Gradients should flow through WeakMEd loss."""
        wm = WeakMedEyeballLoss()
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        bbox = torch.tensor([[4, 4, 28, 28], [4, 4, 28, 28]], dtype=torch.float32)
        result = wm(logits, bbox)
        result['total'].backward()
        assert logits.grad is not None, "Gradients should flow through WeakMEd loss"
        assert logits.grad.abs().sum() > 0, "Gradients should be non-zero"


class TestDiceCELoss:
    """Test multi-class DiceCE loss."""

    def test_output_positive(self):
        loss_fn = DiceCELoss(num_classes=4)
        logits = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        loss = loss_fn(logits, target)
        assert loss > 0, f"DiceCE loss should be positive, got {loss}"


class TestEyeballBBoxExtraction:
    """Test Sclera→Square BB extraction from segmentation labels."""

    def test_basic_extraction(self):
        """Sclera+Iris+Pupil union should produce a valid square BB."""
        label = np.zeros((100, 100), dtype=np.int64)
        label[30:60, 40:70] = 1  # Sclera
        label[35:55, 45:65] = 2  # Iris
        label[40:50, 50:60] = 3  # Pupil

        bbox = OpenEDS400SequenceDataset._extract_eyeball_bbox(label)
        x1, y1, x2, y2 = bbox.astype(int)

        # Should be square
        assert (x2 - x1) == (y2 - y1), "BB should be square"
        # Should contain all foreground
        assert y1 <= 30, f"y1={y1} should be <= 30"
        assert y2 >= 60, f"y2={y2} should be >= 60"
        assert x1 <= 40, f"x1={x1} should be <= 40"
        assert x2 >= 70, f"x2={x2} should be >= 70"

    def test_empty_label(self):
        """All-background label should return zero BB."""
        label = np.zeros((100, 100), dtype=np.int64)
        bbox = OpenEDS400SequenceDataset._extract_eyeball_bbox(label)
        assert np.allclose(bbox, [0, 0, 0, 0]), f"Empty label should give zero BB, got {bbox}"

    def test_small_region(self):
        """Small foreground region should still produce valid square BB."""
        label = np.zeros((100, 100), dtype=np.int64)
        label[50, 50] = 3  # Single pupil pixel
        bbox = OpenEDS400SequenceDataset._extract_eyeball_bbox(label)
        x1, y1, x2, y2 = bbox.astype(int)
        assert (x2 - x1) == (y2 - y1), "BB should be square"
        assert x2 > x1, "BB should have positive width"


class TestEyeballHead:
    """Test that VivimBackbone eyeball head works correctly."""

    def test_eyeball_head_output_shape(self):
        """Eyeball head should produce [B, 1, H, W] output."""
        from models.vivim_backbone import VivimBackbone

        model = VivimBackbone(
            in_channels=1, num_classes=4, base_channels=16,
            d_state=4, d_conv=2, expand=1,
            use_mamba=False, use_eyeball_head=True,
        )
        x = torch.randn(2, 3, 1, 64, 64)  # [B, T, C, H, W]
        out = model(x)

        assert isinstance(out, dict), "Output should be dict when eyeball head enabled"
        assert 'seg' in out, "Output should contain 'seg' key"
        assert 'eyeball' in out, "Output should contain 'eyeball' key"
        assert out['seg'].shape == (2, 4, 64, 64), f"Seg shape wrong: {out['seg'].shape}"
        assert out['eyeball'].shape == (2, 1, 64, 64), f"Eye shape wrong: {out['eyeball'].shape}"

    def test_no_eyeball_head_returns_tensor(self):
        """Without eyeball head, output should be a plain tensor."""
        from models.vivim_backbone import VivimBackbone

        model = VivimBackbone(
            in_channels=1, num_classes=4, base_channels=16,
            d_state=4, d_conv=2, expand=1,
            use_mamba=False, use_eyeball_head=False,
        )
        x = torch.randn(2, 3, 1, 64, 64)
        out = model(x)

        assert isinstance(out, torch.Tensor), "Output should be tensor when eyeball head disabled"
        assert out.shape == (2, 4, 64, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
