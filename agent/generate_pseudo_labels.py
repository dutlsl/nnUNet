"""
Generate pseudo-labels for the Sequence Dataset using the trained nnUNet model.
Runs inference on all 91,200 sequence frames and saves predicted masks as .npy files.
"""
import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from temporal_unet import TemporalUNet


def generate_pseudo_labels(
    model_folder: str,
    sequence_root: str,
    output_root: str,
    checkpoint_name: str = 'checkpoint_best.pth',
    batch_size: int = 32,
    device: str = 'cuda',
):
    """Generate pseudo-labels for all sequence frames."""

    os.makedirs(output_root, exist_ok=True)

    # Load model (we only need the frozen encoder+original decoder for per-frame inference)
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from batchgenerators.utilities.file_and_folder_operations import load_json, join
    from acvl_utils.cropping_and_padding.padding import pad_nd_image

    plans = load_json(join(model_folder, 'plans.json'))
    dataset_json = load_json(join(model_folder, 'dataset.json'))
    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration('2d')
    num_input_channels = determine_num_input_channels(plans_manager, config_manager, dataset_json)

    # Compute normalization stats from plans
    foreground_intensity_properties = plans['foreground_intensity_properties_per_channel']['0']
    mean_val = foreground_intensity_properties['mean']
    std_val = foreground_intensity_properties['std']

    fold_dirs = [d for d in os.listdir(model_folder) if d.startswith('fold_')]
    fold_dir = join(model_folder, fold_dirs[0])
    checkpoint_path = join(fold_dir, checkpoint_name)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    network = get_network_from_plans(
        config_manager.network_arch_class_name,
        config_manager.network_arch_init_kwargs,
        config_manager.network_arch_init_kwargs_req_import,
        num_input_channels, 4,
        allow_init=False, deep_supervision=False,
    )
    network.load_state_dict(checkpoint['network_weights'])
    network.to(device)
    network.eval()

    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(sequence_root, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split}: directory not found")
            continue

        out_split = os.path.join(output_root, split)
        os.makedirs(out_split, exist_ok=True)

        # Collect all PNG files
        png_files = sorted(glob.glob(os.path.join(split_dir, '**', '*.png'), recursive=True))
        print(f"\n[{split}] Found {len(png_files)} frames. Generating pseudo-labels...")

        # Process in batches
        for batch_start in range(0, len(png_files), batch_size):
            batch_files = png_files[batch_start:batch_start + batch_size]
            batch_imgs = []

            for fpath in batch_files:
                img = np.array(Image.open(fpath).convert('L'), dtype=np.float32)
                img_norm = (img - mean_val) / (std_val + 1e-8)
                batch_imgs.append(img_norm)

            # Stack batch
            batch_np = np.stack(batch_imgs)[:, None]  # [B, 1, H, W]
            batch_tensor = torch.from_numpy(batch_np).to(device)
            orig_h, orig_w = batch_tensor.shape[2], batch_tensor.shape[3]

            # Pad to nearest multiple of 64 for clean stride divisions
            # Sequence images are 640x400, need padding on W dimension
            def _ceil_div(a, b):
                return (a + b - 1) // b * b

            target_h = _ceil_div(orig_h, 64)  # 640 → 640 (already aligned)
            target_w = _ceil_div(orig_w, 64)  # 400 → 448

            pad_h = target_h - orig_h
            pad_w = target_w - orig_w
            if pad_h > 0 or pad_w > 0:
                batch_tensor = torch.nn.functional.pad(
                    batch_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0
                )

            with torch.no_grad():
                logits = network(batch_tensor)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                # Crop back to original size
                masks = logits.argmax(dim=1)[:, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)

            # Save each mask
            for i, fpath in enumerate(batch_files):
                fname = Path(fpath).stem
                # Preserve subject directory structure
                rel_path = os.path.relpath(fpath, split_dir)
                rel_dir = os.path.dirname(rel_path)
                save_dir = os.path.join(out_split, rel_dir) if rel_dir else out_split
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f"{fname}.npy"), masks[i])

            if (batch_start // batch_size) % 50 == 0:
                done = min(batch_start + batch_size, len(png_files))
                print(f"  [{split}] {done}/{len(png_files)} frames processed")

        print(f"  [{split}] Done! Saved to {out_split}")


if __name__ == '__main__':
    os.environ['nnUNet_raw'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results'

    generate_pseudo_labels(
        model_folder='/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d',
        sequence_root='/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_Extracted',
        output_root='/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_PseudoLabels',
        batch_size=32,
        device='cuda',
    )
