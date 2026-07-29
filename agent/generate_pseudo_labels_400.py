"""
Generate 400x400 Pseudo-labels for Sequence Dataset using the trained 400x400 Vanilla nnUNet model.

Applies 100% on-the-fly cropping [120:520, 0:400] to sequence frames and saves predicted 400x400 masks as .npy files.
"""

import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from batchgenerators.utilities.file_and_folder_operations import load_json


def generate_pseudo_labels_400(
    checkpoint_path: str = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/VanillaUNet_400x400/checkpoint_best.pth',
    plans_path: str = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d/plans.json',
    sequence_root: str = '/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_Extracted',
    output_root: str = '/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_PseudoLabels_400',
    batch_size: int = 32,
    device: str = 'cuda',
    max_samples: int = None,  # Useful for testing subset
):
    """Generate 400x400 pseudo-labels for sequence frames."""
    os.makedirs(output_root, exist_ok=True)
    device_obj = torch.device(device)

    print(f"=== Starting 400x400 Pseudo-Label Generation ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Sequence Root: {sequence_root}")
    print(f"Output Root: {output_root}")

    # Load 400x400 PlainConvUNet model
    plans = load_json(plans_path)
    arch = plans['configurations']['2d']['architecture']

    network = get_network_from_plans(
        arch['network_class_name'],
        arch['arch_kwargs'],
        arch['_kw_requires_import'],
        1, 4,
        allow_init=False, deep_supervision=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    network.to(device_obj)
    network.eval()

    mean_val = 86.45
    std_val = 39.94

    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(sequence_root, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split}: directory not found ({split_dir})")
            continue

        out_split = os.path.join(output_root, split)
        os.makedirs(out_split, exist_ok=True)

        png_files = sorted(glob.glob(os.path.join(split_dir, '**', '*.png'), recursive=True))
        if max_samples is not None:
            png_files = png_files[:max_samples]

        print(f"\n[{split}] Found {len(png_files)} frames. Generating 400x400 pseudo-labels...")

        for batch_start in range(0, len(png_files), batch_size):
            batch_files = png_files[batch_start:batch_start + batch_size]
            batch_imgs = []

            for fpath in batch_files:
                img = np.array(Image.open(fpath).convert('L'), dtype=np.float32)
                # On-the-fly crop [120:520, 0:400] -> (400, 400)
                img_crop = img[120:520, 0:400]
                img_norm = (img_crop - mean_val) / (std_val + 1e-8)
                batch_imgs.append(img_norm)

            batch_np = np.stack(batch_imgs)[:, None]  # [B, 1, 400, 400]
            # Zero-pad from (400, 400) to (448, 448) for clean 64-stride division
            batch_np = np.pad(batch_np, ((0, 0), (0, 0), (24, 24), (24, 24)), mode='constant', constant_values=0)
            batch_tensor = torch.from_numpy(batch_np).to(device_obj)

            with torch.no_grad():
                logits = network(batch_tensor)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                # Argmax and crop back from (448, 448) to (400, 400) by stripping padding (24:424, 24:424)
                masks = logits.argmax(dim=1)[:, 24:424, 24:424].cpu().numpy().astype(np.uint8)

            for i, fpath in enumerate(batch_files):
                fname = Path(fpath).stem
                rel_path = os.path.relpath(fpath, split_dir)
                rel_dir = os.path.dirname(rel_path)
                save_dir = os.path.join(out_split, rel_dir) if rel_dir else out_split
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f"{fname}.npy"), masks[i])

            if (batch_start // batch_size) % 50 == 0:
                done = min(batch_start + batch_size, len(png_files))
                print(f"  [{split}] {done}/{len(png_files)} frames processed")

        print(f"  [{split}] Done! Saved 400x400 masks to {out_split}")

    print("\n=== Pseudo-Label Generation Completed! ===")


if __name__ == '__main__':
    generate_pseudo_labels_400()
