"""
Multi-Domain Dataset for SAGD Training.
Combines OpenEDS (4-class), Swirski (ellipse), and LPW (video segment labels)
into a unified DataLoader with domain-aware batching.

All domains share the same RITnet preprocessing pipeline:
  Resize 192x192 → Gamma 0.8 → CLAHE 1.5 → Normalize [-1,1]
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from typing import Dict, List, Tuple, Optional


import pickle

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.cache_sagd'))


class RITnetPreprocessor:
    """
    RITnet-compatible preprocessing pipeline.
    Gamma 0.8 → CLAHE 1.5 → Normalize [-1, 1]
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

        # Precompute gamma LUT
        table = 255.0 * (np.linspace(0, 1, 256) ** gamma)
        self.gamma_table = table.astype(np.uint8)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: grayscale image [H, W] uint8

        Returns:
            processed: [H', W'] float32 in [-1, 1]
        """
        # Resize to target
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                             interpolation=cv2.INTER_LINEAR)

        # Gamma correction
        img = cv2.LUT(img, self.gamma_table)

        # CLAHE (created on-the-fly to ensure multiprocessing DataLoader picklability)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )
        img = clahe.apply(img)

        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        return img


class OpenEDSDomainDataset(Dataset):
    """
    Domain 0: OpenEDS 2019 — 4-class segmentation labels.
    Applies RITnet preprocessing and resizes to 192x192.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        temporal_window: int = 3,
        original_crop: List[int] = None,
        preprocessor: Optional[RITnetPreprocessor] = None,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.original_crop = original_crop or [120, 520, 0, 400]
        self.preprocessor = preprocessor or RITnetPreprocessor()
        self.domain_id = 0

        os.makedirs(CACHE_DIR, exist_ok=True)
        split_name = os.path.basename(image_dir)
        cache_path = os.path.join(CACHE_DIR, f"openeds_{split_name}_T{temporal_window}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            all_pngs = sorted(glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True))
            if not all_pngs:
                all_pngs = sorted(glob.glob(os.path.join(image_dir, '*.png')))

            self.samples = []
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
                    label_path = os.path.join(label_dir, f"{fname}.npy")
                    if not os.path.exists(label_path):
                        label_path = os.path.join(label_dir, 'labels', f"{fname}.npy")
                    if not os.path.exists(label_path) and rel_dir != '.':
                        label_path = os.path.join(label_dir, rel_dir, f"{fname}.npy")
                    if not os.path.exists(label_path) and rel_dir != '.':
                        label_path = os.path.join(label_dir, rel_dir.replace('images', 'labels'), f"{fname}.npy")

                    if os.path.exists(label_path):
                        self.samples.append((window_files, label_path))

            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)

        print(f"[OpenEDS Domain] {len(self.samples)} samples (T={temporal_window})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_files, label_path = self.samples[idx]
        y_start, y_end, x_start, x_end = self.original_crop

        frames = []
        for fpath in window_files:
            img = np.array(Image.open(fpath).convert('L'))
            # Apply original crop first (640x400 -> 400x400)
            if img.shape[0] > y_start:
                img = img[y_start:y_end, x_start:x_end]
            # RITnet preprocessing (resize to 192x192 + gamma + CLAHE + normalize)
            img_processed = self.preprocessor(img)
            frames.append(img_processed)

        # [T, 1, 192, 192]
        frames_tensor = torch.from_numpy(np.stack(frames)[:, None]).float()

        # Load and resize label
        label = np.load(label_path).astype(np.int64)  # [400, 400]
        label_resized = cv2.resize(
            label.astype(np.uint8),
            (self.preprocessor.target_size[1], self.preprocessor.target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

        return {
            'images': frames_tensor,           # [T, 1, 192, 192]
            'label': torch.from_numpy(label_resized).long(),  # [192, 192]
            'domain_id': 0,
        }


class SwirskiDomainDataset(Dataset):
    """
    Domain 1: Swirski Dataset — pupil ellipse annotations -> binary mask.
    Generates pupil-only binary masks from ellipse parameters.
    """

    def __init__(
        self,
        root: str,
        subsets: List[str],
        temporal_window: int = 3,
        preprocessor: Optional[RITnetPreprocessor] = None,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.preprocessor = preprocessor or RITnetPreprocessor()
        self.domain_id = 1

        self.samples: List[Tuple[List[str], dict]] = []

        for subset in subsets:
            subset_dir = os.path.join(root, subset)
            frames_dir = os.path.join(subset_dir, 'frames')
            ellipse_file = os.path.join(subset_dir, 'pupil-ellipses.txt')

            if not os.path.exists(ellipse_file) or not os.path.isdir(frames_dir):
                print(f"[Swirski] Skipping {subset}: missing files")
                continue

            # Parse ellipse annotations
            ellipses = {}
            with open(ellipse_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) != 2:
                        continue
                    frame_id = int(parts[0].strip())
                    vals = [float(v.strip()) for v in parts[1].strip().split()]
                    if len(vals) == 5:
                        cx, cy, a, b, angle = vals
                        ellipses[frame_id] = {
                            'cx': cx, 'cy': cy,
                            'a': a, 'b': b,
                            'angle': angle,
                        }

            # Get all frame files
            frame_files = sorted(glob.glob(os.path.join(frames_dir, '*-eye.png')))
            if not frame_files:
                frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))

            # Build frame_id -> file mapping
            id_to_file = {}
            for fp in frame_files:
                fname = os.path.basename(fp)
                try:
                    fid = int(fname.split('-')[0])
                    id_to_file[fid] = fp
                except (ValueError, IndexError):
                    continue

            # Build temporal sequences from labeled frames
            labeled_ids = sorted(set(ellipses.keys()) & set(id_to_file.keys()))
            for i in range(len(labeled_ids)):
                target_id = labeled_ids[i]
                # Try to build a temporal window ending at target_id
                window_ids = []
                for j in range(i - temporal_window + 1, i + 1):
                    if 0 <= j < len(labeled_ids):
                        window_ids.append(labeled_ids[j])

                if len(window_ids) < temporal_window:
                    # Pad by repeating first frame
                    while len(window_ids) < temporal_window:
                        window_ids.insert(0, window_ids[0])

                window_files = [id_to_file[wid] for wid in window_ids]
                self.samples.append((window_files, ellipses[target_id]))

        print(f"[Swirski Domain] {len(self.samples)} samples (T={temporal_window})")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _ellipse_to_mask(
        ellipse: dict, img_shape: Tuple[int, int], target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Convert ellipse annotation to binary pupil mask at target_size."""
        h_orig, w_orig = img_shape
        h_target, w_target = target_size

        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        center = (int(ellipse['cx']), int(ellipse['cy']))
        axes = (int(ellipse['a']), int(ellipse['b']))
        angle = int(np.degrees(ellipse['angle']))

        cv2.ellipse(mask, center, axes, angle, 0, 360, color=3, thickness=-1)  # class 3 = pupil

        # Resize to target
        mask = cv2.resize(mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.int64)

    def __getitem__(self, idx):
        window_files, ellipse = self.samples[idx]

        frames = []
        img_shape = None
        for fpath in window_files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((192, 192), dtype=np.uint8)
            if img_shape is None:
                img_shape = img.shape[:2]
            img_processed = self.preprocessor(img)
            frames.append(img_processed)

        frames_tensor = torch.from_numpy(np.stack(frames)[:, None]).float()

        # Generate binary mask from ellipse
        label = self._ellipse_to_mask(
            ellipse, img_shape, self.preprocessor.target_size
        )

        return {
            'images': frames_tensor,
            'label': torch.from_numpy(label).long(),
            'domain_id': 1,
        }


class LPWDomainDataset(Dataset):
    """
    Domain 2: LPW (Labelled Pupils in the Wild) — segment label videos.
    Uses pre-extracted pupil+eyelid mask videos from 'Pupils in the wild improved'.
    """

    def __init__(
        self,
        video_root: str,
        segment_label_root: str,
        temporal_window: int = 3,
        preprocessor: Optional[RITnetPreprocessor] = None,
        skip_missing_labels: bool = True,
        extracted_video_root: Optional[str] = None,
        extracted_label_root: Optional[str] = None,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.preprocessor = preprocessor or RITnetPreprocessor()
        self.domain_id = 2
        self.skip_missing_labels = skip_missing_labels
        self.video_root = video_root
        self.extracted_video_root = extracted_video_root
        self.extracted_label_root = extracted_label_root

        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"lpw_T{temporal_window}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            self.samples = []
            label_files = {}
            if os.path.isdir(segment_label_root):
                for f in os.listdir(segment_label_root):
                    if f.endswith('_pupil.mp4'):
                        key = f.replace('_pupil.mp4', '')
                        pupil_path = os.path.join(segment_label_root, f)
                        eyelid_path = os.path.join(segment_label_root, f.replace('_pupil.mp4', '_eyelid.mp4'))
                        if os.path.exists(eyelid_path):
                            label_files[key] = (pupil_path, eyelid_path)

            for folder_name in sorted(os.listdir(video_root)):
                folder_path = os.path.join(video_root, folder_name)
                if not os.path.isdir(folder_path):
                    continue

                for video_file in sorted(os.listdir(folder_path)):
                    if not video_file.endswith('.avi'):
                        continue

                    file_id = video_file.replace('.avi', '')
                    label_key = f"folder-{folder_name}_file-{file_id}"

                    if label_key not in label_files:
                        if skip_missing_labels:
                            continue

                    video_path = os.path.join(folder_path, video_file)
                    pupil_label_path, eyelid_label_path = label_files[label_key]

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        continue
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    if n_frames < temporal_window:
                        continue

                    for frame_idx in range(temporal_window - 1, n_frames, 5):
                        self.samples.append(
                            (video_path, pupil_label_path, eyelid_label_path, frame_idx)
                        )

            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)

        print(f"[LPW Domain] {len(self.samples)} samples (T={temporal_window})")

    def __len__(self):
        return len(self.samples)

    def _read_video_frames(
        self, video_path: str, target_frame: int, count: int
    ) -> List[np.ndarray]:
        """Read `count` consecutive frames ending at target_frame."""
        # Try PNG extracted first (100x faster than video seek decoding)
        if self.extracted_video_root:
            try:
                rel = os.path.relpath(video_path, self.video_root)
                folder_id, fname = os.path.split(rel)
                file_id = os.path.splitext(fname)[0]
                png_dir = os.path.join(self.extracted_video_root, folder_id, file_id)

                start = max(0, target_frame - count + 1)
                frames = []
                if os.path.isdir(png_dir):
                    for idx in range(start, target_frame + 1):
                        png_path = os.path.join(png_dir, f"{idx:06d}.png")
                        if os.path.exists(png_path):
                            gray = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                            if gray is not None:
                                frames.append(gray)

                if len(frames) == count:
                    return frames
            except Exception:
                pass

        # Fallback to VideoCapture if PNGs not extracted yet
        cap = cv2.VideoCapture(video_path)
        start = max(0, target_frame - count + 1)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(count):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            else:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((192, 192), dtype=np.uint8))
        cap.release()

        while len(frames) < count:
            frames.insert(0, frames[0].copy())

        return frames[-count:]

    def _read_label_frame(
        self, pupil_path: str, eyelid_path: str, frame_idx: int
    ) -> np.ndarray:
        """Read a single frame from pupil/eyelid label videos and create a mask."""
        target_size = self.preprocessor.target_size

        # Try PNG extracted first
        if self.extracted_label_root:
            try:
                pupil_prefix = os.path.splitext(os.path.basename(pupil_path))[0]
                eyelid_prefix = os.path.splitext(os.path.basename(eyelid_path))[0]
                pupil_png = os.path.join(self.extracted_label_root, pupil_prefix, f"{frame_idx:06d}.png")
                eyelid_png = os.path.join(self.extracted_label_root, eyelid_prefix, f"{frame_idx:06d}.png")

                if os.path.exists(pupil_png) and os.path.exists(eyelid_png):
                    pupil_gray = cv2.imread(pupil_png, cv2.IMREAD_GRAYSCALE)
                    eyelid_gray = cv2.imread(eyelid_png, cv2.IMREAD_GRAYSCALE)

                    mask = np.zeros(target_size, dtype=np.int64)

                    if pupil_gray is not None:
                        if pupil_gray.shape[:2] != target_size:
                            pupil_gray = cv2.resize(pupil_gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                        mask[pupil_gray > 128] = 3

                    if eyelid_gray is not None:
                        if eyelid_gray.shape[:2] != target_size:
                            eyelid_gray = cv2.resize(eyelid_gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                        eyelid_region = (eyelid_gray > 128) & (mask != 3)
                        mask[eyelid_region] = 2

                    return mask
            except Exception:
                pass

        # Fallback to VideoCapture
        cap_pupil = cv2.VideoCapture(pupil_path)
        cap_pupil.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_p, frame_p = cap_pupil.read()
        cap_pupil.release()

        cap_eyelid = cv2.VideoCapture(eyelid_path)
        cap_eyelid.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_e, frame_e = cap_eyelid.read()
        cap_eyelid.release()

        mask = np.zeros(target_size, dtype=np.int64)

        if ret_p and frame_p is not None:
            pupil_gray = cv2.cvtColor(frame_p, cv2.COLOR_BGR2GRAY)
            pupil_gray = cv2.resize(pupil_gray, (target_size[1], target_size[0]),
                                     interpolation=cv2.INTER_NEAREST)
            mask[pupil_gray > 128] = 3

        if ret_e and frame_e is not None:
            eyelid_gray = cv2.cvtColor(frame_e, cv2.COLOR_BGR2GRAY)
            eyelid_gray = cv2.resize(eyelid_gray, (target_size[1], target_size[0]),
                                      interpolation=cv2.INTER_NEAREST)
            eyelid_region = (eyelid_gray > 128) & (mask != 3)
            mask[eyelid_region] = 2

        return mask

    def __getitem__(self, idx):
        video_path, pupil_path, eyelid_path, frame_idx = self.samples[idx]

        # Read temporal window of frames
        raw_frames = self._read_video_frames(video_path, frame_idx, self.temporal_window)

        frames = []
        for img in raw_frames:
            img_processed = self.preprocessor(img)
            frames.append(img_processed)

        frames_tensor = torch.from_numpy(np.stack(frames)[:, None]).float()

        # Read label
        label = self._read_label_frame(pupil_path, eyelid_path, frame_idx)

        return {
            'images': frames_tensor,
            'label': torch.from_numpy(label).long(),
            'domain_id': 2,
        }


def collate_multi_domain(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that groups samples by domain_id.
    Returns separate batches per domain for HDM processing.
    """
    domain_batches = {}
    for sample in batch:
        d_id = sample['domain_id']
        if d_id not in domain_batches:
            domain_batches[d_id] = {'images': [], 'label': [], 'domain_id': d_id}
        domain_batches[d_id]['images'].append(sample['images'])
        domain_batches[d_id]['label'].append(sample['label'])

    result = {}
    for d_id, data in domain_batches.items():
        result[d_id] = {
            'images': torch.stack(data['images']),
            'label': torch.stack(data['label']),
            'domain_id': d_id,
        }

    return result


class MultiDomainDataset(Dataset):
    """Wrapper dataset taking (domain_id, local_index) tuples."""
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, item):
        domain_id, local_idx = item
        return self.datasets[domain_id][local_idx]


class MultiDomainBatchSampler:
    """
    Yields batches where each domain has exactly `samples_per_domain` items per step.
    Guarantees B_min is constant (e.g., 2) across all domains.

    CRITICAL: Uses SEQUENTIAL cursor per domain (not random sampling).
    Each epoch, the full index range is shuffled once, then iterated sequentially.
    This preserves temporal locality within each OpenEDS/Swirski/LPW sequence
    window, which is essential for Vivim+Mamba's sequential state modeling.
    """
    def __init__(self, datasets: List[Dataset], samples_per_domain: int = 2, num_iterations: int = 250):
        self.datasets = datasets
        self.samples_per_domain = samples_per_domain
        self.num_iterations = num_iterations

    def __iter__(self):
        # Create epoch-level permuted index arrays per domain.
        # Permutation randomizes WHICH temporal windows appear, but iteration
        # order within a batch is sequential (consecutive cursor positions).
        domain_indices = []
        for ds in self.datasets:
            n = len(ds)
            if n == 0:
                domain_indices.append(np.array([], dtype=np.int64))
            else:
                perm = np.random.permutation(n)
                # Tile to ensure enough indices for all iterations
                needed = self.num_iterations * self.samples_per_domain
                if needed > n:
                    perm = np.tile(perm, (needed // n) + 1)
                domain_indices.append(perm)

        # Sequential cursors per domain
        cursors = [0] * len(self.datasets)

        for _ in range(self.num_iterations):
            batch_indices = []
            for domain_id, ds in enumerate(self.datasets):
                n = len(ds)
                if n == 0:
                    continue
                for _ in range(self.samples_per_domain):
                    idx = int(domain_indices[domain_id][cursors[domain_id]])
                    batch_indices.append((domain_id, idx))
                    cursors[domain_id] += 1
            yield batch_indices

    def __len__(self):
        return self.num_iterations


def get_multi_domain_dataloaders(
    cfg, batch_size: Optional[int] = None, num_iterations: int = 250
) -> Dict[str, DataLoader]:
    """
    Build multi-domain DataLoaders dynamically configured by nnUNetTrainer.

    Returns:
        dict with 'train' and 'validation' DataLoaders
    """
    if batch_size is None:
        batch_size = getattr(cfg.training, 'batch_size', 12)

    preprocessor = RITnetPreprocessor(
        target_size=tuple(cfg.data.input_resolution),
        gamma=cfg.data.preprocess.gamma,
        clahe_clip_limit=cfg.data.preprocess.clahe_clip_limit,
        clahe_tile_grid=tuple(cfg.data.preprocess.clahe_tile_grid),
    )

    temporal_window = cfg.data.temporal_window

    # Domain 0: OpenEDS
    openeds_cfg = cfg.data.openeds
    openeds_train = OpenEDSDomainDataset(
        image_dir=os.path.join(openeds_cfg.sequence_root, 'train'),
        label_dir=os.path.join(openeds_cfg.pseudo_label_root, 'train'),
        temporal_window=temporal_window,
        original_crop=list(openeds_cfg.original_crop),
        preprocessor=preprocessor,
    )
    openeds_val = OpenEDSDomainDataset(
        image_dir=os.path.join(openeds_cfg.sequence_root, 'validation'),
        label_dir=os.path.join(openeds_cfg.pseudo_label_root, 'validation'),
        temporal_window=temporal_window,
        original_crop=list(openeds_cfg.original_crop),
        preprocessor=preprocessor,
    )

    # Domain 1: Swirski
    swirski_cfg = cfg.data.swirski
    swirski_ds = SwirskiDomainDataset(
        root=swirski_cfg.root,
        subsets=list(swirski_cfg.subsets),
        temporal_window=temporal_window,
        preprocessor=preprocessor,
    )

    # Domain 2: LPW
    lpw_cfg = cfg.data.lpw
    lpw_ds = LPWDomainDataset(
        video_root=lpw_cfg.video_root,
        segment_label_root=lpw_cfg.segment_label_root,
        temporal_window=temporal_window,
        preprocessor=preprocessor,
        skip_missing_labels=lpw_cfg.skip_missing_labels,
        extracted_video_root=getattr(lpw_cfg, 'extracted_video_root', None),
        extracted_label_root=getattr(lpw_cfg, 'extracted_label_root', None),
    )

    domain_list = [openeds_train, swirski_ds, lpw_ds]
    multi_ds = MultiDomainDataset(domain_list)
    num_domains = len(domain_list)
    samples_per_domain = max(1, batch_size // num_domains)
    num_workers = getattr(cfg.training, 'num_workers', 0)

    batch_sampler = MultiDomainBatchSampler(
        domain_list,
        samples_per_domain=samples_per_domain,
        num_iterations=num_iterations,
    )

    loaders = {
        'train': DataLoader(
            multi_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_multi_domain,
        ),
        'validation': DataLoader(
            openeds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders
