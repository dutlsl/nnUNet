"""
End-to-End Sequence Dataset Gaze Tracking Pipeline (Batch CUDA Inference)
Extracts sequence frames, runs batch GPU nnUNet segmentation,
extracts geometric pupil/iris features, and applies temporal tracking.
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from acvl_utils.cropping_and_padding.padding import pad_nd_image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from gaze_estimation import extract_gaze_features_from_mask
from temporal_tracking import process_sequence_trajectory

os.environ['nnUNet_raw'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results'

def process_sequence_folder(img_dir: str, model_dir: str, output_base_dir: str, max_frames: int = 600, batch_size: int = 16):
    dir_name = Path(img_dir).name
    print(f"\n==========================================")
    print(f"Processing Sequence Directory: {dir_name}")
    print(f"==========================================")
    
    out_dir = Path(output_base_dir) / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Predictor & Network on CUDA
    device = torch.device('cuda')
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )
    
    network = predictor.network.to(device)
    network.eval()
    
    # 2. Collect image paths
    png_paths = sorted(list(Path(img_dir).glob("*.png")))
    if max_frames and max_frames > 0:
        png_paths = png_paths[:max_frames]
        
    print(f"Found {len(png_paths)} sequence PNG frames to process in batch mode.")
    
    results = []
    
    for i in range(0, len(png_paths), batch_size):
        batch_paths = png_paths[i:i+batch_size]
        batch_imgs = []
        
        for p in batch_paths:
            img = Image.open(p).convert("L")
            img_np = np.array(img, dtype=np.float32)
            
            mean_val = img_np.mean()
            std_val = img_np.std()
            if std_val > 1e-8:
                img_norm = (img_np - mean_val) / std_val
            else:
                img_norm = img_np - mean_val
                
            img_padded, slicer = pad_nd_image(img_norm[None, None], new_shape=(448, 640), return_slicer=True, mode='constant')
            batch_imgs.append(img_padded)
            
        # Stack into batch tensor (B, 1, H, W)
        batch_np = np.concatenate(batch_imgs, axis=0)
        batch_tensor = torch.from_numpy(batch_np).to(device)
        
        with torch.no_grad():
            logits = network(batch_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred_masks = logits.argmax(dim=1)[:, slicer[2], slicer[3]].cpu().numpy().astype(np.uint8)
            
        for b_idx in range(len(batch_paths)):
            global_idx = i + b_idx
            mask = pred_masks[b_idx]
            
            gaze_feat = extract_gaze_features_from_mask(mask)
            gaze_feat["frame_id"] = global_idx
            gaze_feat["filename"] = batch_paths[b_idx].name
            results.append(gaze_feat)
            
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(png_paths):
            print(f"Processed {min(i + batch_size, len(png_paths))}/{len(png_paths)} frames...")
            
    # 3. Stage 3: Temporal Tracking and Classification
    print("\nApplying Stage 3 Temporal Tracking & Classification...")
    tracked_results = process_sequence_trajectory(results, fps=200.0)
    
    # Save CSV
    df = pd.DataFrame(tracked_results)
    csv_path = out_dir / f"{dir_name}_gaze_trajectory.csv"
    df.to_csv(csv_path, index=False)
    
    # Event Statistics
    event_counts = df["event"].value_counts().to_dict()
    print(f"\n==========================================")
    print(f"Sequence Gaze Tracking Summary ({dir_name}):")
    print(f"==========================================")
    print(f"  Total Frames Processed : {len(df)}")
    print(f"  Event Distribution     : {event_counts}")
    print(f"  Trajectory CSV Path    : {csv_path}")
    print(f"==========================================\n")
    
    return tracked_results

if __name__ == "__main__":
    model_dir = "/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d"
    val_dir = "/tmp/Validation_Part_1"
    out_dir = "/home/iulab0/PycharmProjects/nnUNet/gaze_results"
    
    if os.path.exists(val_dir):
        process_sequence_folder(val_dir, model_dir, out_dir, max_frames=600, batch_size=16)
