"""
SequenceGazerDataset: T-frame windows from the OpenEDS Sequence Dataset.

For each frame the dataset also reports whether a SEMANTIC ground-truth label
is available (frame number present in Semantic_Segmentation_Dataset/labels).
Semantic GT frames are free QUERY (cost 0); the remaining frames are queried
via SAM3 at teacher-call time. No point/box labels are ever used.

Returns per window:
  images:  [T, 1, 448, 448] float  (RITnet-preprocessed, padded) — student input
  gt:      [T, 448, 448] long      (0 where no GT)
  has_gt:  [T] bool
  native:  [T, H0, W0] uint8       (native-resolution crop) — for SAM3
  keys:    list[T] of frame numbers
"""

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

from datasets.openeds_dataset import RITnetPreprocessor


class SequenceGazerDataset(Dataset):
    def __init__(
        self,
        sequence_image_dir: str,
        semantic_label_root: Optional[str],
        temporal_window: int = 5,
        crop: Tuple[int, int, int, int] = (120, 520, 0, 400),
        padded_resolution: Tuple[int, int] = (448, 448),
        num_classes: int = 4,
        preprocessor: Optional[RITnetPreprocessor] = None,
    ):
        super().__init__()
        self.T = temporal_window
        self.crop = crop
        self.padded_resolution = padded_resolution
        self.num_classes = num_classes
        self.preprocessor = preprocessor or RITnetPreprocessor(
            target_size=(crop[1] - crop[0], crop[3] - crop[2]))
        self.semantic_label_root = semantic_label_root

        all_pngs = sorted(glob.glob(os.path.join(sequence_image_dir, '**', '*.png'),
                                    recursive=True)) or sorted(
            glob.glob(os.path.join(sequence_image_dir, '*.png')))
        if not all_pngs:
            raise FileNotFoundError(f"No sequence PNGs in {sequence_image_dir}")

        # Group frames by subject directory; a frame has GT iff its number has
        # a semantic label .npy.
        dir_to_files: Dict[str, List[str]] = {}
        for p in all_pngs:
            dir_to_files.setdefault(os.path.dirname(p), []).append(p)

        self.samples: List[Tuple[List[str], List[bool]]] = []
        for parent, files in dir_to_files.items():
            files = sorted(files)
            if len(files) < self.T:
                continue
            for end in range(self.T - 1, len(files)):
                win = files[end - self.T + 1: end + 1]
                has_gt = [self._has_gt(f) for f in win]
                self.samples.append((win, has_gt))
        print(f"[SequenceGazerDataset] {len(self.samples)} windows (T={self.T}) "
              f"from {sequence_image_dir}")

    def _has_gt(self, png_path: str) -> bool:
        if not self.semantic_label_root:
            return False
        frame = os.path.splitext(os.path.basename(png_path))[0]
        # semantic_label_root already points at .../labels
        return os.path.exists(os.path.join(self.semantic_label_root, f"{frame}.npy"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        win, has_gt = self.samples[idx]
        y0, y1, x0, x1 = self.crop
        crop_h, crop_w = y1 - y0, x1 - x0
        pad_h = self.padded_resolution[0] - crop_h
        pad_w = self.padded_resolution[1] - crop_w
        ph, pw = pad_h // 2, pad_w // 2

        images, native, gt, keys = [], [], [], []
        for i, fpath in enumerate(win):
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((640, 400), dtype=np.uint8)
            img_crop = img[y0:y1, x0:x1]                      # [crop_h, crop_w]
            images.append(self.preprocessor(img_crop))
            native.append(img_crop)
            frame = int(os.path.splitext(os.path.basename(fpath))[0])
            keys.append(frame)
            if has_gt[i]:
                lbl = np.load(os.path.join(self.semantic_label_root, f"{frame}.npy")).astype(np.int64)
                gt.append(lbl[y0:y1, x0:x1])
            else:
                gt.append(np.zeros((crop_h, crop_w), dtype=np.int64))

        images_np = np.stack(images)[:, None]
        images_np = np.pad(images_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
        gt_np = np.stack(gt)
        gt_np = np.pad(gt_np, ((0, 0), (ph, ph), (pw, pw)), mode="constant")
        native_np = np.stack(native)
        native_np = np.pad(native_np, ((0, 0), (ph, ph), (pw, pw)), mode="constant")

        return {
            "images": torch.from_numpy(images_np).float(),
            "gt": torch.from_numpy(gt_np).long(),
            "has_gt": torch.tensor(has_gt, dtype=torch.bool),
            "native": native_np,
            "keys": keys,
        }


def gazer_collate(batch):
    """Stack tensors, keep native as a numpy array and keys as a flat list."""
    out = {}
    for k in ("images", "gt", "has_gt"):
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["native"] = np.stack([b["native"] for b in batch], axis=0)   # [B,T,H0,W0] uint8
    out["keys"] = [k for b in batch for k in b["keys"]]
    return out


def get_sequence_gazer_dataloaders(cfg) -> Dict[str, DataLoader]:
    loaders = {}
    seq_root = cfg.data.sequence_root
    sem_root = getattr(cfg.data, "semantic_gt_root", None)
    T = cfg.data.temporal_window
    crop = tuple(cfg.data.input_crop)
    pad_res = tuple(cfg.data.target_resolution)
    bs = cfg.training.batch_size
    nw = cfg.training.num_workers

    for split in ["train", "validation"]:
        sem_label_root = None
        if sem_root:
            cand = os.path.join(sem_root, split, "labels")
            if os.path.isdir(cand):
                sem_label_root = cand
        ds = SequenceGazerDataset(
            os.path.join(seq_root, split),
            sem_label_root,
            temporal_window=T,
            crop=crop,
            padded_resolution=pad_res,
            num_classes=cfg.model.num_classes,
        )
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == "train"),
            num_workers=nw, pin_memory=False, drop_last=(split == "train"),
            collate_fn=gazer_collate)
    return loaders
