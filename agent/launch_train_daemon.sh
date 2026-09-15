#!/bin/bash
set -e

PY="/home/iulab1/PycharmProjects/nnUNet/nnunet_env/bin/python"
DIR="/home/iulab1/PycharmProjects/nnUNet"

mkdir -p "$DIR/logs"

echo "=== Cleaning previous logs ==="
> "$DIR/logs/phase1_sparse_t5.log"
> "$DIR/logs/phase2_multiframe_t5.log"

echo "=== Launching Phase 1 on GPU 0 (Full Daemon) ==="
CUDA_VISIBLE_DEVICES=0 setsid $PY -u "$DIR/train_phase1.py" --config "$DIR/configs/phase1_sparse_t5.yaml" --gpu 0 < /dev/null > "$DIR/logs/phase1_sparse_t5.log" 2>&1 &

echo "=== Launching Phase 2 on GPU 1 (Full Daemon) ==="
CUDA_VISIBLE_DEVICES=1 setsid $PY -u "$DIR/train_phase2.py" --config "$DIR/configs/phase2_multiframe_t5.yaml" --gpu 0 < /dev/null > "$DIR/logs/phase2_multiframe_t5.log" 2>&1 &

disown -a

sleep 2
echo "=== Launched Processes ==="
ps -eo pid,ppid,tty,stat,cmd | grep -E "train_phase[12]" | grep -v grep
