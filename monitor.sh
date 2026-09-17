#!/bin/bash
# Shortcut to attach to the live monitor tmux session or run directly
if /home/iulab1/.local/bin/tmux has-session -t monitor 2>/dev/null; then
    exec /home/iulab1/.local/bin/tmux attach -t monitor
else
    exec /home/iulab1/PycharmProjects/nnUNet/nnunet_env/bin/python /home/iulab1/PycharmProjects/nnUNet/agent/live_monitor.py
fi
