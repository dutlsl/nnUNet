"""
Master Pipeline Chaining Script for 400x400 Native Architecture.

Sequence of Operations:
1. Monitor running train_vanilla_400.py process until Early Stopping completes.
2. Verify /home/iulab0/PycharmProjects/nnUNet/nnUNet_results/VanillaUNet_400x400/checkpoint_best.pth exists.
3. Run 400x400 Pseudo-label Generation (generate_pseudo_labels_400.py).
4. Run 400x400 ConvLSTM TemporalUNet Training with WandB (train_temporal_400.py).
"""

import os
import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from generate_pseudo_labels_400 import generate_pseudo_labels_400
from train_temporal_400 import train_temporal_400


def find_vanilla_pid():
    """Find PID of train_vanilla_400.py process using pgrep/ps."""
    try:
        res = subprocess.check_output(["pgrep", "-f", "train_vanilla_400.py"]).decode().strip()
        pids = [int(p) for p in res.split() if p.isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


def is_process_running(pid):
    """Check if process with PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    print("=== 400x400 Master Chaining Pipeline Initialized ===", flush=True)
    
    # Step 1: Monitor train_vanilla_400.py
    pid = find_vanilla_pid()
    if pid is not None:
        print(f"Monitoring 400x400 Vanilla nnUNet training process (PID: {pid})...", flush=True)
        while is_process_running(pid):
            time.sleep(15)
        print(f"Vanilla nnUNet training process (PID: {pid}) has completed.", flush=True)
    else:
        print("No active train_vanilla_400.py process found. Proceeding to checkpoint verification.", flush=True)

    # Step 2: Checkpoint Verification
    ckpt_path = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/VanillaUNet_400x400/checkpoint_best.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Critical Error: {ckpt_path} not found! Cannot proceed with pseudo-label generation.")
    
    print(f"\n[Step 1 Verified] Found best 400x400 Vanilla model: {ckpt_path}", flush=True)

    # Step 3: Pseudo-Label Generation
    print("\n[Step 2/3] Launching 400x400 Pseudo-Label Generation...", flush=True)
    generate_pseudo_labels_400(
        checkpoint_path=ckpt_path,
        sequence_root='/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_Extracted',
        output_root='/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_PseudoLabels_400',
        batch_size=32,
        device='cuda',
    )

    # Step 4: TemporalUNet Training with WandB
    print("\n[Step 3/3] Launching 400x400 ConvLSTM TemporalUNet Training with WandB...", flush=True)
    train_temporal_400(
        vanilla_checkpoint_path=ckpt_path,
        sequence_root='/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_Extracted',
        pseudo_label_root='/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/Sequence_PseudoLabels_400',
        output_dir='/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/TemporalUNet_400x400',
        temporal_window=3,
        batch_size=4,
        num_epochs=150,
        lr=1e-3,
        device='cuda',
    )

    print("\n=== Master Chaining Pipeline Finished Completely! ===", flush=True)


if __name__ == '__main__':
    main()
