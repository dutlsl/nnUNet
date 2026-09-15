#!/bin/bash
cd /home/iulab1/PycharmProjects/nnUNet
TMUX=/home/iulab1/.local/bin/tmux

$TMUX kill-session -t train 2>/dev/null || true
$TMUX kill-session -t phase1 2>/dev/null || true
$TMUX kill-session -t phase2 2>/dev/null || true
$TMUX kill-session -t dashboard 2>/dev/null || true

$TMUX new-session -d -s train -x 200 -y 50 "CUDA_VISIBLE_DEVICES=0 /home/iulab1/PycharmProjects/nnUNet/nnunet_env/bin/python train_phase1.py --config configs/phase1_sparse_t5.yaml --gpu 0"
$TMUX split-window -h -t train:0 "CUDA_VISIBLE_DEVICES=1 /home/iulab1/PycharmProjects/nnUNet/nnunet_env/bin/python train_phase2.py --config configs/phase2_multiframe_t5.yaml --gpu 0"
$TMUX split-window -v -p 35 -t train:0.0 "watch -n 1 /home/iulab1/PycharmProjects/nnUNet/agent/show_system_resources.sh"
echo "TRAIN_LAUNCHED_SUCCESSFULLY"
