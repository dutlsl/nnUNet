"""
OpenEDS 400x400 Sequence Dataset for Temporal Segmentation Training.

Applies 100% on-the-fly cropping [120:520, 0:400] to sequence frames and pairs them
with 400x400 pseudo-label masks.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Tuple, List, Dict


class OpenEDS400SequenceDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        temporal_window: int = 3,
        mean: float = 86.45,
        std: float = 39.94,
        target_size: int = 448,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.mean = mean
        self.std = std
        self.target_size = target_size
        self.image_dir = image_dir
        self.label_dir = label_dir

        all_pngs = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if len(all_pngs) == 0:
            print(f"Warning: No PNG files found in {image_dir}")
            self.samples = []
            return

        frame_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in all_pngs]

        # Group consecutive IDs into subject sequences
        subjects: List[List[int]] = []
        current_subject = [frame_ids[0]]
        for i in range(1, len(frame_ids)):
            if frame_ids[i] == frame_ids[i - 1] + 1:
                current_subject.append(frame_ids[i])
            else:
                subjects.append(current_subject)
                current_subject = [frame_ids[i]]
        subjects.append(current_subject)

        # Sliding window samples
        self.samples: List[Tuple[List[int], int]] = []
        for subject_ids in subjects:
            if len(subject_ids) < temporal_window:
                continue
            for end_idx in range(temporal_window - 1, len(subject_ids)):
                start_idx = end_idx - temporal_window + 1
                window_ids = subject_ids[start_idx:end_idx + 1]
                label_id = window_ids[-1]
                label_path = os.path.join(label_dir, f"{label_id:012d}.npy")
                if os.path.exists(label_path):
                    self.samples.append((window_ids, label_id))

        print(f"OpenEDS400SequenceDataset: {len(self.samples)} samples ({len(subjects)} subjects, T={temporal_window})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_ids, label_id = self.samples[idx]

        frames = []
        for fid in window_ids:
            fpath = os.path.join(self.image_dir, f"{fid:012d}.png")
            img = np.array(Image.open(fpath).convert('L'), dtype=np.float32)
            # On-the-fly crop [120:520, 0:400] -> (400, 400)
            img_crop = img[120:520, 0:400]
            img_norm = (img_crop - self.mean) / (self.std + 1e-8)
            frames.append(img_norm)

        # [T, 1, 400, 400]
        frames_np = np.stack(frames)[:, None]

        # Zero-pad from (400, 400) to (448, 448)
        frames_np = np.pad(frames_np, ((0, 0), (0, 0), (24, 24), (24, 24)), mode='constant', constant_values=0)

        # Load 400x400 pseudo-label
        label_path = os.path.join(self.label_dir, f"{label_id:012d}.npy")
        label = np.load(label_path).astype(np.int64)  # [400, 400]
        label_padded = np.pad(label, ((24, 24), (24, 24)), mode='constant', constant_values=0)  # [448, 448]

        return {
            'images': torch.from_numpy(frames_np).float(),  # [T, 1, 448, 448]
            'label': torch.from_numpy(label_padded).long(),   # [448, 448]
        }


def get_sequence_400_dataloaders(
    sequence_root: str,
    pseudo_label_root: str,
    temporal_window: int = 3,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ['train', 'validation']:
        img_dir = os.path.join(sequence_root, split)
        lbl_dir = os.path.join(pseudo_label_root, split)

        if not os.path.isdir(img_dir):
            print(f"Warning: {img_dir} not found, skipping {split}")
            continue

        ds = OpenEDS400SequenceDataset(
            image_dir=img_dir,
            label_dir=lbl_dir,
            temporal_window=temporal_window,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    return loaders
