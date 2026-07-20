"""
OpenEDS 2019 Test Set Evaluation Script
Calculates class-wise Dice, IoU, and Mean Dice on the test set predictions.
"""
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path

def compute_metrics(gt_dir, pred_dir):
    gt_files = sorted(list(Path(gt_dir).glob("*.nii.gz")))
    pred_files = sorted(list(Path(pred_dir).glob("*.nii.gz")))
    
    print(f"Found {len(gt_files)} ground truth files and {len(pred_files)} prediction files.")
    
    classes = {1: "Sclera", 2: "Iris", 3: "Pupil"}
    dice_scores = {c: [] for c in classes}
    iou_scores = {c: [] for c in classes}
    
    for gt_path in gt_files:
        pred_path = Path(pred_dir) / gt_path.name
        if not pred_path.exists():
            continue
            
        gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)
        pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
        
        for c_id in classes:
            gt_c = (gt == c_id)
            pred_c = (pred == c_id)
            
            intersection = np.logical_and(gt_c, pred_c).sum()
            total = gt_c.sum() + pred_c.sum()
            union = np.logical_or(gt_c, pred_c).sum()
            
            if total == 0:
                dice = 1.0 if intersection == 0 else 0.0
            else:
                dice = 2.0 * intersection / total
                
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
                
            dice_scores[c_id].append(dice)
            iou_scores[c_id].append(iou)
            
    summary = {}
    print("\n" + "="*50)
    print("OpenEDS 2019 Test Set Evaluation Summary")
    print("="*50)
    
    mean_dices = []
    mean_ious = []
    
    for c_id, c_name in classes.items():
        m_dice = np.mean(dice_scores[c_id])
        s_dice = np.std(dice_scores[c_id])
        m_iou = np.mean(iou_scores[c_id])
        
        mean_dices.append(m_dice)
        mean_ious.append(m_iou)
        
        summary[c_name] = {
            "Dice_mean": float(round(m_dice, 4)),
            "Dice_std": float(round(s_dice, 4)),
            "IoU_mean": float(round(m_iou, 4))
        }
        print(f"Class {c_id} ({c_name:7s}) -> Dice: {m_dice:.4f} ± {s_dice:.4f} | IoU: {m_iou:.4f}")
        
    overall_mdice = float(round(np.mean(mean_dices), 4))
    overall_miou = float(round(np.mean(mean_ious), 4))
    summary["Overall_mDice"] = overall_mdice
    summary["Overall_mIoU"] = overall_miou
    
    print("-"*50)
    print(f"Overall Mean Dice (mDice): {overall_mdice:.4f}")
    print(f"Overall Mean IoU  (mIoU) : {overall_miou:.4f}")
    print("="*50 + "\n")
    
    # Save metrics to JSON
    json_path = Path(pred_dir) / "test_evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation metrics saved to: {json_path}")
    
    return summary

if __name__ == "__main__":
    gt_dir = "/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw/Dataset600_OpenEDS2019/labelsTs"
    pred_dir = "/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d/test_predictions"
    compute_metrics(gt_dir, pred_dir)
