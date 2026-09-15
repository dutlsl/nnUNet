"""
DualStreamDataset: Mixes labeled (Semantic GT) and unlabeled (Sequence) frames
in a single DataLoader for sparse-label training (Phase 1).

Each sample returns:
  - images: [T, 1, H, W]  (temporal window of grayscale frames)
  - label:  [H, W]         (4-class GT or zeros if unlabeled)
  - has_label: bool         (True if labeled sample)
"""

import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional
import random
import math


class RITnetPreprocessor400:
    """Gamma 0.8 -> CLAHE 1.5 -> Normalize [-1, 1]"""
    def __init__(self, gamma=0.8, clahe_clip_limit=1.5, clahe_tile_grid=(8, 8)):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        table = 255.0 * (np.linspace(0, 1, 256) ** gamma)
        self.gamma_table = table.astype(np.uint8)

    def __call__(self, img_uint8: np.ndarray) -> np.ndarray:
        img_gamma = cv2.LUT(img_uint8, self.gamma_table)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
        img_clahe = clahe.apply(img_gamma)
        img_float = img_clahe.astype(np.float32) / 255.0
        return (img_float - 0.5) / 0.5


def _load_and_preprocess(fpath, crop, preprocessor):
    """Load frame, crop [120:520, 0:400], preprocess."""
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((640, 400), dtype=np.uint8)
    y0, y1, x0, x1 = crop
    return preprocessor(img[y0:y1, x0:x1])


class SemanticGTSequenceDataset(Dataset):
    """
    Labeled frames from Semantic_Segmentation_Dataset with temporal context
    pulled from adjacent Sequence frames (1-frame gap).
    """
    def __init__(self, semantic_image_dir, semantic_label_dir, sequence_image_dir,
                 temporal_window=5, crop=(120, 520, 0, 400), padded_resolution=(448, 448)):
        super().__init__()
        self.T = temporal_window
        self.crop = crop
        self.padded_resolution = padded_resolution
        self.preprocessor = RITnetPreprocessor400()

        # Build lookup: frame_number -> filepath for Sequence frames
        seq_frames = sorted(glob.glob(os.path.join(sequence_image_dir, '*.png')))
        if not seq_frames:
            seq_frames = sorted(glob.glob(os.path.join(sequence_image_dir, '**', '*.png'), recursive=True))
        self.seq_lookup = {}
        for fpath in seq_frames:
            fname = os.path.splitext(os.path.basename(fpath))[0]
            try:
                self.seq_lookup[int(fname)] = fpath
            except ValueError:
                pass

        # Build samples
        sem_images = sorted(glob.glob(os.path.join(semantic_image_dir, '*.png')))
        self.samples = []
        for img_path in sem_images:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(semantic_label_dir, f"{fname}.npy")
            if os.path.exists(label_path):
                try:
                    self.samples.append((int(fname), img_path, label_path))
                except ValueError:
                    pass

        print(f"[SemanticGTSeqDataset] {len(self.samples)} labeled (T={temporal_window}, "
              f"seq_lookup={len(self.seq_lookup)} frames)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_num, img_path, label_path = self.samples[idx]
        y0, y1, x0, x1 = self.crop
        crop_h, crop_w = y1 - y0, x1 - x0
        pad_h = self.padded_resolution[0] - crop_h
        pad_w = self.padded_resolution[1] - crop_w
        ph, pw = pad_h // 2, pad_w // 2

        # Build temporal window: [frame_num - T + 1, ..., frame_num]
        frames = []
        for t_off in range(-(self.T - 1), 1):
            fn = frame_num + t_off
            if fn in self.seq_lookup:
                fpath = self.seq_lookup[fn]
            elif fn == frame_num:
                fpath = img_path
            else:
                fpath = img_path  # fallback
            frames.append(_load_and_preprocess(fpath, self.crop, self.preprocessor))

        frames_np = np.stack(frames)[:, None]  # [T, 1, 400, 400]
        frames_np = np.pad(frames_np, ((0,0),(0,0),(ph,pad_h-ph),(pw,pad_w-pw)), mode='constant')

        label = np.load(label_path).astype(np.int64)
        if label.shape[0] >= y1:  # full resolution label, crop it
            label = label[y0:y1, x0:x1]
        label_padded = np.pad(label, ((ph,pad_h-ph),(pw,pad_w-pw)), mode='constant')

        return {
            'images': torch.from_numpy(frames_np).float(),
            'label': torch.from_numpy(label_padded).long(),
            'has_label': True,
        }


class UnlabeledSequenceDataset(Dataset):
    """Unlabeled temporal windows from Sequence_Extracted."""
    def __init__(self, sequence_image_dir, temporal_window=5,
                 crop=(120, 520, 0, 400), padded_resolution=(448, 448), stride=3):
        super().__init__()
        self.T = temporal_window
        self.crop = crop
        self.padded_resolution = padded_resolution
        self.preprocessor = RITnetPreprocessor400()

        all_pngs = sorted(glob.glob(os.path.join(sequence_image_dir, '*.png')))
        if not all_pngs:
            all_pngs = sorted(glob.glob(os.path.join(sequence_image_dir, '**', '*.png'), recursive=True))

        # Parse frame numbers and detect subject boundaries
        frame_nums = []
        for fpath in all_pngs:
            fname = os.path.splitext(os.path.basename(fpath))[0]
            try:
                frame_nums.append((int(fname), fpath))
            except ValueError:
                pass
        frame_nums.sort()

        # Split by subject (gap > 1)
        subject_seqs = []
        if frame_nums:
            cur = [frame_nums[0][1]]
            for i in range(1, len(frame_nums)):
                if frame_nums[i][0] - frame_nums[i-1][0] > 1:
                    subject_seqs.append(cur)
                    cur = []
                cur.append(frame_nums[i][1])
            if cur:
                subject_seqs.append(cur)

        # Sliding window with stride
        self.samples = []
        for seq in subject_seqs:
            if len(seq) < temporal_window:
                continue
            for end_idx in range(temporal_window - 1, len(seq), stride):
                self.samples.append(seq[end_idx - temporal_window + 1:end_idx + 1])

        print(f"[UnlabeledSeqDataset] {len(self.samples)} unlabeled windows "
              f"(T={temporal_window}, stride={stride}, {len(subject_seqs)} subjects)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_files = self.samples[idx]
        y0, y1, x0, x1 = self.crop
        crop_h, crop_w = y1 - y0, x1 - x0
        pad_h = self.padded_resolution[0] - crop_h
        pad_w = self.padded_resolution[1] - crop_w
        ph, pw = pad_h // 2, pad_w // 2

        frames = [_load_and_preprocess(f, self.crop, self.preprocessor) for f in window_files]
        frames_np = np.stack(frames)[:, None]
        frames_np = np.pad(frames_np, ((0,0),(0,0),(ph,pad_h-ph),(pw,pad_w-pw)), mode='constant')

        dummy_label = np.zeros(self.padded_resolution, dtype=np.int64)
        return {
            'images': torch.from_numpy(frames_np).float(),
            'label': torch.from_numpy(dummy_label).long(),
            'has_label': False,
        }


class DualStreamBatchSampler(Sampler):
    """Yields batches with fixed labeled:unlabeled ratio."""
    def __init__(self, n_labeled, n_unlabeled, n_labeled_per_batch, batch_size, drop_last=True):
        self.n_labeled = n_labeled
        self.n_unlabeled = n_unlabeled
        self.n_lab = n_labeled_per_batch
        self.n_unlab = batch_size - n_labeled_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        lab_idx = list(range(self.n_labeled))
        unlab_idx = list(range(self.n_labeled, self.n_labeled + self.n_unlabeled))
        random.shuffle(lab_idx)
        random.shuffle(unlab_idx)

        # Cycle labeled to match unlabeled length
        n_batches = len(unlab_idx) // self.n_unlab
        lab_cycle = (lab_idx * (n_batches * self.n_lab // max(1, len(lab_idx)) + 2))

        li, ui = 0, 0
        for _ in range(n_batches):
            batch = lab_cycle[li:li+self.n_lab] + unlab_idx[ui:ui+self.n_unlab]
            li += self.n_lab
            ui += self.n_unlab
            yield batch

    def __len__(self):
        return self.n_unlabeled // max(1, self.n_unlab)


class ConcatDualStreamDataset(Dataset):
    """Concat labeled (0..N_lab-1) + unlabeled (N_lab..N_total-1)."""
    def __init__(self, labeled_ds, unlabeled_ds):
        self.labeled_ds = labeled_ds
        self.unlabeled_ds = unlabeled_ds
        self.n_labeled = len(labeled_ds)

    def __len__(self):
        return self.n_labeled + len(self.unlabeled_ds)

    def __getitem__(self, idx):
        if idx < self.n_labeled:
            return self.labeled_ds[idx]
        return self.unlabeled_ds[idx - self.n_labeled]


def get_dual_stream_dataloaders(cfg):
    """Build DualStream train loader + labeled-only val loader."""
    sem_root = cfg.data.semantic_gt_root
    seq_root = cfg.data.sequence_root
    T = cfg.data.temporal_window
    bs = cfg.training.batch_size
    n_lab = getattr(cfg.training, 'n_labeled_per_batch', max(1, bs // 4))
    nw = cfg.training.num_workers
    crop = tuple(cfg.data.input_crop)
    pad_res = tuple(cfg.data.target_resolution)
    stride = getattr(cfg.data, 'unlabeled_stride', 3)

    loaders = {}

    # Train: dual-stream
    lab_train = SemanticGTSequenceDataset(
        os.path.join(sem_root, 'train', 'images'),
        os.path.join(sem_root, 'train', 'labels'),
        os.path.join(seq_root, 'train'),
        temporal_window=T, crop=crop, padded_resolution=pad_res,
    )
    unlab_train = UnlabeledSequenceDataset(
        os.path.join(seq_root, 'train'),
        temporal_window=T, crop=crop, padded_resolution=pad_res, stride=stride,
    )
    combined = ConcatDualStreamDataset(lab_train, unlab_train)
    sampler = DualStreamBatchSampler(len(lab_train), len(unlab_train), n_lab, bs)
    loaders['train'] = DataLoader(combined, batch_sampler=sampler, num_workers=nw, pin_memory=True)

    # Validation: labeled only
    lab_val = SemanticGTSequenceDataset(
        os.path.join(sem_root, 'validation', 'images'),
        os.path.join(sem_root, 'validation', 'labels'),
        os.path.join(seq_root, 'validation'),
        temporal_window=T, crop=crop, padded_resolution=pad_res,
    )
    loaders['validation'] = DataLoader(lab_val, batch_size=bs, shuffle=False, num_workers=nw,
                                        pin_memory=True, drop_last=False)

    return loaders
