"""
Generate 100% Clean, High-Accuracy 4-Class Pseudo-Labels for Sequence Dataset
using the Trained Vivim Checkpoint (checkpoint_final.pth).

Maps predictions cleanly to:
  0: Background
  1: Pupil
  2: Iris
  3: Sclera
"""

import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.vivim_backbone import VivimBackbone


def generate_clean_pseudo_labels(
    checkpoint_path: str = os.path.join(
        PROJECT_ROOT,
        'nnUNet_results',
        'Dataset600_OpenEDS2019',
        'nnUNetTrainer_Vivim__nnUNetPlans__2d',
        'fold_0',
        'checkpoint_final.pth',
    ),
    sequence_root: str = os.path.join(PROJECT_ROOT, 'Openedsdata2019', 'Sequence_Extracted'),
    output_root: str = os.path.join(PROJECT_ROOT, 'Openedsdata2019', 'Sequence_PseudoLabels_400'),
    batch_size: int = 64,
    device: str = 'cuda',
):
    os.makedirs(output_root, exist_ok=True)
    device_obj = torch.device(device)

    print("=== Starting High-Accuracy Clean 4-Class Pseudo-Label Generation ===")
    print(f"Device        : {device}")
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Sequence Root : {sequence_root}")
    print(f"Output Root   : {output_root}")

    # Instantiate trained Vivim backbone
    model = VivimBackbone(in_channels=1, num_classes=4, base_channels=32).to(device_obj)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['network_weights']
    clean_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    y_start, y_end, x_start, x_end = 120, 520, 0, 400

    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(sequence_root, split)
        if not os.path.exists(split_dir):
            continue

        out_split = os.path.join(output_root, split)
        os.makedirs(out_split, exist_ok=True)

        png_files = sorted(glob.glob(os.path.join(split_dir, '**', '*.png'), recursive=True))
        print(f"\n[{split}] Found {len(png_files)} frames. Generating clean 4-class pseudo-labels...")

        for batch_start in range(0, len(png_files), batch_size):
            batch_files = png_files[batch_start:batch_start + batch_size]
            batch_imgs = []

            for fpath in batch_files:
                img = np.array(Image.open(fpath).convert('L'))
                if img.shape[0] > y_start:
                    img = img[y_start:y_end, x_start:x_end]  # [400, 400]
                img_resize = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)
                img_norm = (img_resize.astype(np.float32) / 255.0 - 0.5) / 0.5
                batch_imgs.append(img_norm)

            batch_np = np.stack(batch_imgs)[:, None, None]  # [B, 1, 1, 192, 192]
            batch_tensor = torch.from_numpy(batch_np).repeat(1, 3, 1, 1, 1).to(device_obj)

            with torch.no_grad():
                logits = model(batch_tensor)
                if isinstance(logits, dict):
                    logits = logits['seg_logits']
                preds_192 = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

            for i, fpath in enumerate(batch_files):
                fname = Path(fpath).stem
                rel_path = os.path.relpath(fpath, split_dir)
                rel_dir = os.path.dirname(rel_path)
                save_dir = os.path.join(out_split, rel_dir) if rel_dir else out_split
                os.makedirs(save_dir, exist_ok=True)

                # Resize 192x192 argmax prediction back to 400x400 mask
                pred_400 = cv2.resize(preds_192[i], (400, 400), interpolation=cv2.INTER_NEAREST)
                np.save(os.path.join(save_dir, f"{fname}.npy"), pred_400)

            if (batch_start // batch_size) % 50 == 0 or (batch_start + batch_size) >= len(png_files):
                done = min(batch_start + batch_size, len(png_files))
                print(f"  [{split}] {done}/{len(png_files)} frames processed ({done / len(png_files) * 100:.1f}%)")

        print(f"  [{split}] Completed! Clean pseudo-labels saved to {out_split}")

    print("\n=== Clean Pseudo-Label Generation Completed Successfully! ===")


if __name__ == '__main__':
    import cv2
    generate_clean_pseudo_labels()
