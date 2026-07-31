"""
Sphere/Circle evaluation metrics for eyeball estimation.
Computes geometric agreement between predicted and GT circle parameters.
"""

import torch
import numpy as np
import math
from typing import Dict


def _circle_intersection_area(cx1: float, cy1: float, r1: float,
                               cx2: float, cy2: float, r2: float) -> float:
    """
    Compute the intersection area of two circles using the standard
    lens-area formula.
    """
    d = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    # No overlap
    if d >= r1 + r2:
        return 0.0

    # One circle inside the other
    if d + min(r1, r2) <= max(r1, r2):
        return math.pi * min(r1, r2) ** 2

    # Partial overlap (lens area)
    r1_sq = r1 ** 2
    r2_sq = r2 ** 2
    d_sq = d ** 2

    alpha = math.acos((d_sq + r1_sq - r2_sq) / (2.0 * d * r1 + 1e-12))
    beta = math.acos((d_sq + r2_sq - r1_sq) / (2.0 * d * r2 + 1e-12))

    area = (r1_sq * (alpha - 0.5 * math.sin(2 * alpha)) +
            r2_sq * (beta - 0.5 * math.sin(2 * beta)))

    return area


def compute_circle_iou(pred_params: torch.Tensor,
                       gt_params: torch.Tensor) -> Dict[str, float]:
    """
    Compute IoU between predicted and GT circles.

    Args:
        pred_params: [B, 3] predicted (cx, cy, r)
        gt_params: [B, 3] ground truth (cx, cy, r)

    Returns:
        dict with 'circle_iou_mean' and per-sample 'circle_ious'
    """
    B = pred_params.shape[0]
    ious = []

    for b in range(B):
        cx1, cy1, r1 = pred_params[b].detach().cpu().tolist()
        cx2, cy2, r2 = gt_params[b].detach().cpu().tolist()

        if r1 < 1e-3 or r2 < 1e-3:
            ious.append(0.0)
            continue

        intersection = _circle_intersection_area(cx1, cy1, r1, cx2, cy2, r2)
        union = math.pi * r1 ** 2 + math.pi * r2 ** 2 - intersection

        iou = intersection / (union + 1e-12)
        ious.append(iou)

    return {
        'circle_iou_mean': float(np.mean(ious)) if ious else 0.0,
        'circle_ious': ious,
    }


def compute_circle_center_error(pred_params: torch.Tensor,
                                gt_params: torch.Tensor) -> Dict[str, float]:
    """
    Compute Euclidean distance between predicted and GT circle centers.

    Args:
        pred_params: [B, 3] predicted (cx, cy, r)
        gt_params: [B, 3] ground truth (cx, cy, r)

    Returns:
        dict with 'center_error_mean' (in pixels)
    """
    pred_centers = pred_params[:, :2]  # [B, 2]
    gt_centers = gt_params[:, :2]      # [B, 2]

    errors = torch.sqrt(((pred_centers - gt_centers) ** 2).sum(dim=1) + 1e-8)
    error_list = errors.detach().cpu().tolist()

    return {
        'center_error_mean': float(np.mean(error_list)),
        'center_errors': error_list,
    }


def compute_radius_error(pred_params: torch.Tensor,
                         gt_params: torch.Tensor) -> Dict[str, float]:
    """
    Compute relative radius error: |r_pred - r_gt| / r_gt.

    Args:
        pred_params: [B, 3] predicted (cx, cy, r)
        gt_params: [B, 3] ground truth (cx, cy, r)

    Returns:
        dict with 'radius_error_mean' (relative) and 'radius_error_abs_mean' (pixels)
    """
    pred_r = pred_params[:, 2]
    gt_r = gt_params[:, 2]

    abs_errors = (pred_r - gt_r).abs()
    rel_errors = abs_errors / (gt_r + 1e-8)

    abs_list = abs_errors.detach().cpu().tolist()
    rel_list = rel_errors.detach().cpu().tolist()

    return {
        'radius_error_abs_mean': float(np.mean(abs_list)),
        'radius_error_rel_mean': float(np.mean(rel_list)),
    }


def compute_all_sphere_metrics(pred_params: torch.Tensor,
                               gt_params: torch.Tensor) -> Dict[str, float]:
    """
    Compute all sphere/circle metrics at once.

    Args:
        pred_params: [B, 3] predicted (cx, cy, r)
        gt_params: [B, 3] ground truth (cx, cy, r)

    Returns:
        dict with all metrics (circle_iou_mean, center_error_mean,
                               radius_error_abs_mean, radius_error_rel_mean)
    """
    iou = compute_circle_iou(pred_params, gt_params)
    center = compute_circle_center_error(pred_params, gt_params)
    radius = compute_radius_error(pred_params, gt_params)

    return {
        'circle_iou_mean': iou['circle_iou_mean'],
        'center_error_mean': center['center_error_mean'],
        'radius_error_abs_mean': radius['radius_error_abs_mean'],
        'radius_error_rel_mean': radius['radius_error_rel_mean'],
    }
