"""
OpenEDS 2019 → nnUNetv2 Dataset Format Converter

Converts OpenEDS 2019 Semantic Segmentation dataset (PNG images + NPY labels)
into nnUNetv2's expected NIfTI format.

Dataset ID: 600 (Dataset600_OpenEDS2019)
- train/ (8,916) + validation/ (2,403) → imagesTr/ + labelsTr/ (11,319 total)
- Semantic_Segmentation_Test_Dataset/ (1,440) → imagesTs/ + labelsTs/

Labels:
  0: Background, 1: Sclera, 2: Iris, 3: Pupil
"""

import os
import glob
import json
import numpy as np
import nibabel as nib
from PIL import Image
from multiprocessing import Pool
from functools import partial


# --- Configuration ---
OPENEDS_BASE = "/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019"
SEG_DATASET = os.path.join(OPENEDS_BASE, "Semantic_Segmentation_Dataset")
TEST_DATASET = os.path.join(OPENEDS_BASE, "Semantic_Segmentation_Test_Dataset")

NNUNET_RAW = "/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw"
DATASET_NAME = "Dataset600_OpenEDS2019"
OUTPUT_DIR = os.path.join(NNUNET_RAW, DATASET_NAME)

IMAGES_TR = os.path.join(OUTPUT_DIR, "imagesTr")
LABELS_TR = os.path.join(OUTPUT_DIR, "labelsTr")
IMAGES_TS = os.path.join(OUTPUT_DIR, "imagesTs")
LABELS_TS = os.path.join(OUTPUT_DIR, "labelsTs")


def png_to_nifti(png_path: str) -> nib.Nifti1Image:
    """Load a PNG image and convert to NIfTI (H, W, 1) with identity affine."""
    img = Image.open(png_path).convert("L")  # Force single-channel grayscale
    arr = np.array(img, dtype=np.float32)     # (H, W)
    arr = arr[:, :, np.newaxis]               # (H, W, 1) for pseudo-3D
    nii = nib.Nifti1Image(arr, affine=np.eye(4))
    return nii


def npy_to_nifti(npy_path: str) -> nib.Nifti1Image:
    """Load a NPY label and convert to NIfTI (H, W, 1) with identity affine."""
    arr = np.load(npy_path).astype(np.uint8)  # (H, W), values 0-3
    # Ensure label shape matches image shape:
    # images are (H, W) from PIL, labels are stored as (H, W) in npy
    # Both should be (640, 400) based on analysis
    arr = arr[:, :, np.newaxis]               # (H, W, 1) for pseudo-3D
    nii = nib.Nifti1Image(arr, affine=np.eye(4))
    return nii


def convert_single_case(args):
    """Convert a single image-label pair. Used for multiprocessing."""
    img_path, lbl_path, case_id, out_img_dir, out_lbl_dir = args
    
    # Convert image
    img_nii = png_to_nifti(img_path)
    img_out = os.path.join(out_img_dir, f"{case_id}_0000.nii.gz")
    nib.save(img_nii, img_out)
    
    # Convert label (if exists)
    if lbl_path is not None and os.path.exists(lbl_path):
        lbl_nii = npy_to_nifti(lbl_path)
        lbl_out = os.path.join(out_lbl_dir, f"{case_id}.nii.gz")
        nib.save(lbl_nii, lbl_out)
    
    return case_id


def gather_cases(split_dir: str) -> list:
    """Gather all image-label pairs from a split directory."""
    img_dir = os.path.join(split_dir, "images")
    lbl_dir = os.path.join(split_dir, "labels")
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    cases = []
    for img_path in img_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, basename + ".npy")
        if not os.path.exists(lbl_path):
            lbl_path = None
        cases.append((img_path, lbl_path, basename))
    return cases


def main():
    # Create output directories
    for d in [IMAGES_TR, LABELS_TR, IMAGES_TS, LABELS_TS]:
        os.makedirs(d, exist_ok=True)
    
    # --- Gather training cases (train + validation) ---
    train_cases = gather_cases(os.path.join(SEG_DATASET, "train"))
    val_cases = gather_cases(os.path.join(SEG_DATASET, "validation"))
    all_train_cases = train_cases + val_cases
    
    print(f"Training cases: {len(train_cases)} (train) + {len(val_cases)} (val) = {len(all_train_cases)} total")
    
    # Prepare args for parallel processing
    # Case IDs: OpenEDS_NNNNNN format using original filename
    train_args = [
        (img, lbl, f"OpenEDS_{basename}", IMAGES_TR, LABELS_TR)
        for img, lbl, basename in all_train_cases
    ]
    
    # --- Gather test cases ---
    test_cases = gather_cases(TEST_DATASET)
    print(f"Test cases: {len(test_cases)}")
    
    test_args = [
        (img, lbl, f"OpenEDS_{basename}", IMAGES_TS, LABELS_TS)
        for img, lbl, basename in test_cases
    ]
    
    # --- Convert in parallel ---
    num_workers = min(8, os.cpu_count() or 4)
    print(f"Converting training data with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = pool.map(convert_single_case, train_args)
    print(f"  Converted {len(results)} training cases.")
    
    print(f"Converting test data with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results_ts = pool.map(convert_single_case, test_args)
    print(f"  Converted {len(results_ts)} test cases.")
    
    # --- Create dataset.json ---
    # Count actual training files that have both image and label
    num_training = len([a for a in train_args if a[1] is not None])
    
    dataset_json = {
        "channel_names": {
            "0": "grayscale_IR"
        },
        "labels": {
            "background": 0,
            "sclera": 1,
            "iris": 2,
            "pupil": 3
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    
    json_path = os.path.join(OUTPUT_DIR, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\nDataset conversion complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  dataset.json: {json_path}")
    print(f"  Training: {num_training} cases")
    print(f"  Test: {len(results_ts)} cases")


if __name__ == "__main__":
    main()
