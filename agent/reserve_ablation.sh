#!/bin/bash
# reserve_ablation.sh
# Waiting for GPU 0 main training process (PID 28635) to finish

TARGET_PID=28635
LOG_FILE="/home/iulab1/PycharmProjects/nnUNet/logs/sagd_sas_only_gpu1.log"

echo "[RESERVE] [$(date)] Waiting for PID $TARGET_PID to finish..."

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 30
done

echo "[RESERVE] [$(date)] PID $TARGET_PID has finished. Starting Module 1 SAS-only ablation training on GPU 1..."

# Environment variables setup
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=online
export SAGD_CONFIG_FILE=sadg_sas_only.yaml
export NNUNET_PROJECT_ROOT=/home/iulab1/PycharmProjects/nnUNet
export nnUNet_raw=/home/iulab1/PycharmProjects/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/home/iulab1/PycharmProjects/nnUNet/nnUNet_preprocessed
export nnUNet_results=/home/iulab1/PycharmProjects/nnUNet/nnUNet_results

# Execute training using the virtual environment python interpreter
/home/iulab1/PycharmProjects/nnUNet/nnunet_env/bin/python -u -c "
import os, sys
sys.path.insert(0, '/home/iulab1/PycharmProjects/nnUNet')
sys.argv = ['nnUNetv2_train', 'Dataset600_OpenEDS2019', '2d', '0', '-tr', 'nnUNetTrainer_Vivim_SADG']
from nnunetv2.run.run_training import run_training_entry
run_training_entry()
" > "$LOG_FILE" 2>&1

echo "[RESERVE] [$(date)] Ablation training launched."
