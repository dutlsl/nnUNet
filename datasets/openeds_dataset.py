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
from typing import Dict, List, Tuple, Optional


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


class RITnetPreprocessor:
    """
    RITnet preprocessing pipeline:
    Resize 192x192 → Gamma 0.8 → CLAHE 1.5 → Normalize [-1, 1]
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (192, 192),
        gamma: float = 0.8,
        clahe_clip_limit: float = 1.5,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
    ):
        self.target_size = target_size
        self.gamma = gamma
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        table = 255.0 * (np.linspace(0, 1, 256) ** gamma)
        self.gamma_table = table.astype(np.uint8)

    def __call__(self, img_uint8: np.ndarray) -> np.ndarray:
        if img_uint8.shape[:2] != self.target_size:
            img_uint8 = cv2.resize(img_uint8, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        img_gamma = cv2.LUT(img_uint8, self.gamma_table)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
        img_clahe = clahe.apply(img_gamma)
        img_float = img_clahe.astype(np.float32) / 255.0
        return (img_float - 0.5) / 0.5


class OpenEDS192SequenceDataset(Dataset):
    """
    OpenEDS 2019 Sequential Dataset with 192x192 Resize & RITnet Preprocessing (T=3).
    """
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        temporal_window: int = 3,
        preprocessor: Optional[RITnetPreprocessor] = None,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.preprocessor = preprocessor or RITnetPreprocessor()
        self.image_dir = image_dir
        self.label_dir = label_dir

        all_pngs = sorted(glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True))
        if len(all_pngs) == 0:
            all_pngs = sorted(glob.glob(os.path.join(image_dir, '*.png')))

        if len(all_pngs) == 0:
            self.samples = []
            print(f"[OpenEDS192SequenceDataset] Warning: No PNG files found in {image_dir}")
            return

        self.samples: List[Tuple[List[str], str]] = []
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
                
                rel_dir = os.path.relpath(parent_dir, image_dir)
                if rel_dir == '.':
                    label_path = os.path.join(label_dir, f"{fname}.npy")
                else:
                    label_path = os.path.join(label_dir, rel_dir, f"{fname}.npy")

                if os.path.exists(label_path):
                    self.samples.append((window_files, label_path))

        print(f"[OpenEDS192SequenceDataset] {len(self.samples)} sequence samples discovered in {image_dir} (T={temporal_window})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_files, label_path = self.samples[idx]
        frames = []
        for fpath in window_files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((192, 192), dtype=np.uint8)
            img_prep = self.preprocessor(img)
            frames.append(img_prep)

        frames_np = np.stack(frames)[:, None]  # [T, 1, 192, 192]
        label = np.load(label_path).astype(np.int64)  # [192, 192]

        return {
            'images': torch.from_numpy(frames_np).float(),  # [T, 1, 192, 192]
            'label': torch.from_numpy(label).long(),        # [192, 192]
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

        ds = OpenEDS192SequenceDataset(
            image_dir=img_dir,
            label_dir=lbl_dir,
            temporal_window=cfg.data.temporal_window,
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
