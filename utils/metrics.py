import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4, smooth: float = 1e-5) -> Dict[str, float]:
    """
    Compute per-class and mean Dice scores.
    Args:
        pred: Argmax predicted class tensor [B, H, W] or logits [B, C, H, W]
        target: Ground truth class tensor [B, H, W]
    """
    if pred.ndim == 4:
        pred = pred.argmax(dim=1)

    class_names = ['Background', 'Sclera', 'Iris', 'Pupil']
    dices = {}

    for c in range(num_classes):
        p_c = (pred == c).float()
        t_c = (target == c).float()
        intersection = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dices[class_names[c]] = dice.item()

    dices['Mean_Dice'] = float(np.mean([dices[name] for name in class_names]))
    return dices
