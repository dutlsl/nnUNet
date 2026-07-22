"""
OpenEDS Sequence Dataset for temporal segmentation training.

Handles the flat directory structure of the OpenEDS Sequence Dataset where
frames are not organized by subject but can be grouped by detecting
consecutive filename ranges (each subject has 600 consecutive frames).

Each sample returns:
    images: [T, 1, H, W] — T consecutive frames, Z-score normalized
    label:  [H, W]       — Pseudo-label mask for the LAST frame
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Tuple, List, Dict


class OpenEDSSequenceDataset(Dataset):
    """
    Dataset that yields temporal sequences of T consecutive frames.

    Automatically discovers subject boundaries by detecting gaps in
    consecutive filenames.

    Args:
        image_dir: Directory containing all PNG frames (flat structure).
        label_dir: Directory containing all .npy pseudo-label files (flat).
        temporal_window: Number of consecutive frames per sample (default: 3).
        mean: Z-score normalization mean.
        std: Z-score normalization std.
        target_h: Pad height to this value (448 for clean stride division).
        target_w: Pad width to this value (640).
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        temporal_window: int = 3,
        mean: float = 0.0,
        std: float = 1.0,
        target_h: int = 448,
        target_w: int = 640,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.mean = mean
        self.std = std
        self.target_h = target_h
        self.target_w = target_w
        self.image_dir = image_dir
        self.label_dir = label_dir

        # Discover all frames and group into subject sequences
        all_pngs = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if len(all_pngs) == 0:
            print(f"Warning: No PNG files found in {image_dir}")
            self.samples = []
            return

        # Extract integer IDs and group consecutive ranges into subjects
        frame_ids = []
        for p in all_pngs:
            stem = os.path.splitext(os.path.basename(p))[0]
            frame_ids.append(int(stem))

        # Detect subject boundaries (gap > 1 between consecutive IDs)
        subjects: List[List[int]] = []
        current_subject = [frame_ids[0]]
        for i in range(1, len(frame_ids)):
            if frame_ids[i] == frame_ids[i - 1] + 1:
                current_subject.append(frame_ids[i])
            else:
                subjects.append(current_subject)
                current_subject = [frame_ids[i]]
        subjects.append(current_subject)

        # Build sliding window samples
        self.samples: List[Tuple[List[int], int]] = []  # (frame_id_list, label_frame_id)
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

        print(f"OpenEDSSequenceDataset: {len(self.samples)} samples "
              f"({len(subjects)} subjects, T={temporal_window}, "
              f"frames/subject≈{len(subjects[0]) if subjects else 0})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_ids, label_id = self.samples[idx]

        # Load T consecutive frames
        frames = []
        for fid in window_ids:
            fpath = os.path.join(self.image_dir, f"{fid:012d}.png")
            img = np.array(Image.open(fpath).convert('L'), dtype=np.float32)
            img_norm = (img - self.mean) / (self.std + 1e-8)
            frames.append(img_norm)

        # Stack to [T, H, W] then add channel dim → [T, 1, H, W]
        frames_np = np.stack(frames)[:, None]

        # Pad to nearest multiple of 64 for clean stride divisions
        h, w = frames_np.shape[2], frames_np.shape[3]
        target_h = ((h + 63) // 64) * 64
        target_w = ((w + 63) // 64) * 64
        if h < target_h or w < target_w:
            pad_h = target_h - h
            pad_w = target_w - w
            frames_np = np.pad(frames_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                               mode='constant', constant_values=0)

        # Load pseudo-label
        label_path = os.path.join(self.label_dir, f"{label_id:012d}.npy")
        label = np.load(label_path).astype(np.int64)  # [H, W]
        if label.shape[0] < target_h or label.shape[1] < target_w:
            pad_h = target_h - label.shape[0]
            pad_w = target_w - label.shape[1]
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        return {
            'images': torch.from_numpy(frames_np).float(),  # [T, 1, H, W]
            'label': torch.from_numpy(label).long(),          # [H, W]
        }


def get_sequence_dataloaders(
    sequence_root: str,
    pseudo_label_root: str,
    mean: float,
    std: float,
    temporal_window: int = 3,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Create train and validation DataLoaders."""

    loaders = {}
    for split in ['train', 'validation']:
        img_dir = os.path.join(sequence_root, split)
        lbl_dir = os.path.join(pseudo_label_root, split)

        if not os.path.isdir(img_dir):
            print(f"Warning: {img_dir} not found, skipping {split}")
            continue

        ds = OpenEDSSequenceDataset(
            image_dir=img_dir,
            label_dir=lbl_dir,
            temporal_window=temporal_window,
            mean=mean,
            std=std,
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
