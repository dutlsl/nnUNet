#!/bin/bash
SESSION="monitor"

tmux has-session -t $SESSION 2>/dev/null
if [ $? -eq 0 ]; then
    tmux attach-session -t $SESSION
    exit 0
fi

# 1. Window 0: dashboard
tmux new-session -d -s $SESSION -n "dashboard" "tail -f -n 1000 /home/iulab1/PycharmProjects/nnUNet/logs/train_400_t3_gpu0.log | sed -u 's/\r/\n/g' | grep --line-buffered -E '^Epoch [0-9]+|★ New best'"
tmux split-window -h -t $SESSION:0 "tail -f -n 1000 /home/iulab1/PycharmProjects/nnUNet/logs/train_400_t5_gpu1.log | sed -u 's/\r/\n/g' | grep --line-buffered -E '^Epoch [0-9]+|★ New best'"
tmux split-window -v -t $SESSION:0.0 "watch -n 1 nvidia-smi"
tmux split-window -v -t $SESSION:0.1 "nvidia-smi dmon -s pucm -d 2"
tmux select-layout -t $SESSION:0 tiled

# 2. Separate Fullscreen Windows
tmux new-window -t $SESSION -n "nvidia-smi" "watch -n 1 nvidia-smi"
tmux new-window -t $SESSION -n "sys-top" "top -d 2"
tmux new-window -t $SESSION -n "log-t3" "tail -f -n 1000 /home/iulab1/PycharmProjects/nnUNet/logs/train_400_t3_gpu0.log | sed -u 's/\r/\n/g' | grep --line-buffered -E '^Epoch [0-9]+|★ New best'"
tmux new-window -t $SESSION -n "log-t5" "tail -f -n 1000 /home/iulab1/PycharmProjects/nnUNet/logs/train_400_t5_gpu1.log | sed -u 's/\r/\n/g' | grep --line-buffered -E '^Epoch [0-9]+|★ New best'"

tmux select-window -t $SESSION:0
tmux attach-session -t $SESSION
