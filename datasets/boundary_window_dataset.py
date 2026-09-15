"""
BoundaryWindowDataset: Phase 2 dataset that samples temporal windows
spanning the Sequence-Semantic boundary so that labeled and unlabeled
frames coexist in the same window.

Returns:
  - images: [T, 1, H, W]
  - labels: [T, H, W]       (valid only where label_mask=1)
  - label_mask: [T]          (1=has label, 0=no label)
"""

import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


class RITnetPreprocessor400:
    def __init__(self, gamma=0.8, clahe_clip_limit=1.5, clahe_tile_grid=(8, 8)):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        table = 255.0 * (np.linspace(0, 1, 256) ** gamma)
        self.gamma_table = table.astype(np.uint8)

    def __call__(self, img_uint8):
        img_gamma = cv2.LUT(img_uint8, self.gamma_table)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
        img_clahe = clahe.apply(img_gamma)
        return (img_clahe.astype(np.float32) / 255.0 - 0.5) / 0.5


class BoundaryWindowDataset(Dataset):
    """
    Constructs temporal windows spanning Sequence → Semantic boundary.

    For each subject, the Sequence ends at frame N and Semantic starts at N+1.
    We create windows [N-T+1+k, ..., N+k] for k=0..num_offsets that straddle
    the boundary, mixing unlabeled Sequence frames with labeled Semantic frames.

    Also includes pure-labeled windows (all Semantic) and pure-unlabeled windows
    (all Sequence) for balance.
    """
    def __init__(self, semantic_image_dir, semantic_label_dir, sequence_image_dir,
                 temporal_window=5, crop=(120, 520, 0, 400),
                 padded_resolution=(448, 448), boundary_offsets=None):
        super().__init__()
        self.T = temporal_window
        self.crop = crop
        self.padded_resolution = padded_resolution
        self.preprocessor = RITnetPreprocessor400()

        # Build frame lookup for both datasets
        seq_pngs = sorted(glob.glob(os.path.join(sequence_image_dir, '*.png')))
        sem_pngs = sorted(glob.glob(os.path.join(semantic_image_dir, '*.png')))

        self.frame_lookup = {}  # frame_num -> (filepath, has_label, label_path_or_None)
        for fpath in seq_pngs:
            fn = int(os.path.splitext(os.path.basename(fpath))[0])
            self.frame_lookup[fn] = (fpath, False, None)

        for fpath in sem_pngs:
            fn = int(os.path.splitext(os.path.basename(fpath))[0])
            fname = os.path.splitext(os.path.basename(fpath))[0]
            lbl = os.path.join(semantic_label_dir, f"{fname}.npy")
            has = os.path.exists(lbl)
            self.frame_lookup[fn] = (fpath, has, lbl if has else None)

        all_fns = sorted(self.frame_lookup.keys())

        # Detect subject boundaries and find seq->sem transitions
        subjects = []
        cur = [all_fns[0]]
        for i in range(1, len(all_fns)):
            if all_fns[i] - all_fns[i - 1] > 1:
                subjects.append(cur)
                cur = []
            cur.append(all_fns[i])
        if cur:
            subjects.append(cur)

        # Build samples
        self.samples = []  # List of List[int] (frame numbers in window)

        if boundary_offsets is None:
            boundary_offsets = list(range(temporal_window))

        for subj_frames in subjects:
            if len(subj_frames) < temporal_window:
                continue

            # Find boundary: last unlabeled -> first labeled
            boundary_idx = None
            for i in range(len(subj_frames) - 1):
                fn = subj_frames[i]
                fn_next = subj_frames[i + 1]
                _, has_cur, _ = self.frame_lookup[fn]
                _, has_next, _ = self.frame_lookup[fn_next]
                if not has_cur and has_next:
                    boundary_idx = i
                    break

            if boundary_idx is not None:
                # Boundary-spanning windows
                for offset in boundary_offsets:
                    end_idx = boundary_idx + offset + 1
                    start_idx = end_idx - temporal_window
                    if start_idx >= 0 and end_idx <= len(subj_frames):
                        self.samples.append(subj_frames[start_idx:end_idx])

            # Pure unlabeled windows (every 5 frames)
            for i in range(0, len(subj_frames) - temporal_window + 1, 5):
                window = subj_frames[i:i + temporal_window]
                if all(not self.frame_lookup[fn][1] for fn in window):
                    self.samples.append(window)

            # Pure labeled windows (stride 1)
            labeled_fns = [fn for fn in subj_frames if self.frame_lookup[fn][1]]
            for i in range(len(labeled_fns) - temporal_window + 1):
                self.samples.append(labeled_fns[i:i + temporal_window])

        print(f"[BoundaryWindowDataset] {len(self.samples)} windows "
              f"(T={temporal_window}, {len(subjects)} subjects)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_fns = self.samples[idx]
        y0, y1, x0, x1 = self.crop
        crop_h, crop_w = y1 - y0, x1 - x0
        pad_h = self.padded_resolution[0] - crop_h
        pad_w = self.padded_resolution[1] - crop_w
        ph, pw = pad_h // 2, pad_w // 2

        frames = []
        labels = []
        mask = []

        for fn in window_fns:
            fpath, has_label, lbl_path = self.frame_lookup[fn]

            # Load and preprocess image
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((640, 400), dtype=np.uint8)
            img_crop = img[y0:y1, x0:x1]
            img_prep = self.preprocessor(img_crop)
            frames.append(img_prep)

            # Load label if available
            if has_label and lbl_path:
                lbl = np.load(lbl_path).astype(np.int64)
                if lbl.shape[0] >= y1:
                    lbl = lbl[y0:y1, x0:x1]
                labels.append(lbl)
                mask.append(1)
            else:
                labels.append(np.zeros((crop_h, crop_w), dtype=np.int64))
                mask.append(0)

        frames_np = np.stack(frames)[:, None]  # [T, 1, H, W]
        frames_np = np.pad(frames_np, ((0,0),(0,0),(ph,pad_h-ph),(pw,pad_w-pw)), mode='constant')

        labels_np = np.stack(labels)  # [T, H, W]
        labels_np = np.pad(labels_np, ((0,0),(ph,pad_h-ph),(pw,pad_w-pw)), mode='constant')

        return {
            'images': torch.from_numpy(frames_np).float(),
            'labels': torch.from_numpy(labels_np).long(),
            'label_mask': torch.tensor(mask, dtype=torch.bool),
        }


def get_boundary_window_dataloaders(cfg):
    sem_root = cfg.data.semantic_gt_root
    seq_root = cfg.data.sequence_root
    T = cfg.data.temporal_window
    bs = cfg.training.batch_size
    nw = cfg.training.num_workers
    crop = tuple(cfg.data.input_crop)
    pad_res = tuple(cfg.data.target_resolution)

    loaders = {}
    for split in ['train', 'validation']:
        ds = BoundaryWindowDataset(
            os.path.join(sem_root, split, 'images'),
            os.path.join(sem_root, split, 'labels'),
            os.path.join(seq_root, split),
            temporal_window=T, crop=crop, padded_resolution=pad_res,
        )
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == 'train'),
            num_workers=nw, pin_memory=True, drop_last=(split == 'train'),
        )
    return loaders
