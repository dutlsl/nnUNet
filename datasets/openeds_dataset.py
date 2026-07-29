"""
OpenEDS 400x400 Sequence Dataset for Temporal Segmentation Training.
Applies 100% on-the-fly cropping [120:520, 0:400] and zero-padding to 448x448.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Tuple


class OpenEDS400SequenceDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        temporal_window: int = 3,
        crop: List[int] = [120, 520, 0, 400],
        padded_resolution: List[int] = [448, 448],
        mean: float = 86.45,
        std: float = 39.94,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.crop = crop
        self.padded_resolution = padded_resolution
        self.mean = mean
        self.std = std
        self.image_dir = image_dir
        self.label_dir = label_dir

        all_pngs = sorted(glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True))
        if len(all_pngs) == 0:
            # Fallback to direct PNG search
            all_pngs = sorted(glob.glob(os.path.join(image_dir, '*.png')))

        if len(all_pngs) == 0:
            self.samples = []
            print(f"[OpenEDS400SequenceDataset] Warning: No PNG files found in {image_dir}")
            return

        # Map each file path and group by folder or consecutive frame ID
        self.samples: List[Tuple[List[str], str]] = []
        
        # Group PNG paths by their directory (subject folder)
        dir_to_files: Dict[str, List[str]] = {}
        for p in all_pngs:
            parent = os.path.dirname(p)
            if parent not in dir_to_files:
                dir_to_files[parent] = []
            dir_to_files[parent].append(p)

        for parent_dir, file_list in dir_to_files.items():
            sorted_files = sorted(file_list)
            if len(sorted_files) < temporal_window:
                continue
            for end_idx in range(temporal_window - 1, len(sorted_files)):
                start_idx = end_idx - temporal_window + 1
                window_files = sorted_files[start_idx:end_idx + 1]
                target_file = window_files[-1]
                fname = os.path.splitext(os.path.basename(target_file))[0]
                
                # Check for corresponding label .npy file
                rel_dir = os.path.relpath(parent_dir, image_dir)
                if rel_dir == '.':
                    label_path = os.path.join(label_dir, f"{fname}.npy")
                else:
                    label_path = os.path.join(label_dir, rel_dir, f"{fname}.npy")

                if os.path.exists(label_path):
                    self.samples.append((window_files, label_path))

        print(f"[OpenEDS400SequenceDataset] {len(self.samples)} sequence samples discovered in {image_dir} (T={temporal_window})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_files, label_path = self.samples[idx]
        y_start, y_end, x_start, x_end = self.crop
        pad_h = self.padded_resolution[0] - (y_end - y_start)
        pad_w = self.padded_resolution[1] - (x_end - x_start)
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2

        frames = []
        for fpath in window_files:
            img = np.array(Image.open(fpath).convert('L'), dtype=np.float32)
            img_crop = img[y_start:y_end, x_start:x_end]
            img_norm = (img_crop - self.mean) / (self.std + 1e-8)
            frames.append(img_norm)

        # [T, 1, H, W]
        frames_np = np.stack(frames)[:, None]
        # Pad to [T, 1, 448, 448]
        frames_np = np.pad(frames_np, ((0, 0), (0, 0), (pad_h_half, pad_h_half), (pad_w_half, pad_w_half)), mode='constant', constant_values=0)

        label = np.load(label_path).astype(np.int64)  # [400, 400]
        label_padded = np.pad(label, ((pad_h_half, pad_h_half), (pad_w_half, pad_w_half)), mode='constant', constant_values=0)

        return {
            'images': torch.from_numpy(frames_np).float(),  # [T, 1, 448, 448]
            'label': torch.from_numpy(label_padded).long(),   # [448, 448]
        }


def get_openeds_sequence_dataloaders(cfg) -> Dict[str, DataLoader]:
    loaders = {}
    seq_root = cfg.data.sequence_root
    pseudo_root = cfg.data.pseudo_label_root

    for split in ['train', 'validation']:
        img_dir = os.path.join(seq_root, split)
        lbl_dir = os.path.join(pseudo_root, split)

        if not os.path.isdir(img_dir):
            continue

        ds = OpenEDS400SequenceDataset(
            image_dir=img_dir,
            label_dir=lbl_dir,
            temporal_window=cfg.data.temporal_window,
            crop=list(cfg.data.crop),
            padded_resolution=list(cfg.data.padded_resolution),
            mean=cfg.data.normalize.mean,
            std=cfg.data.normalize.std,
        )

        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=(split == 'train'),
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    return loaders
