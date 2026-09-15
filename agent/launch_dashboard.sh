#!/bin/bash
TMUX_BIN="/home/iulab1/.local/bin/tmux"
DIR="/home/iulab1/PycharmProjects/nnUNet"

# Kill existing dashboard session if present
$TMUX_BIN kill-session -t dashboard 2>/dev/null || true

# 1. Create session with Phase 1 log (will become top-left pane)
$TMUX_BIN new-session -d -s dashboard -x 200 -y 55 "cd $DIR && tail -n 30 -f logs/phase1_sparse_t5.log"

# 2. Split vertically for bottom pane (Server status: 42% height)
$TMUX_BIN split-window -v -p 42 -t dashboard:0 "cd $DIR && watch -n 1 /home/iulab1/PycharmProjects/nnUNet/agent/show_system_resources.sh"

# 3. Split top pane horizontally for Phase 2 log (50% width)
$TMUX_BIN split-window -h -p 50 -t dashboard:0.0 "cd $DIR && tail -n 30 -f logs/phase2_multiframe_t5.log"

# 4. Set pane titles and border formatting
$TMUX_BIN set-option -t dashboard pane-border-status top
$TMUX_BIN set-option -t dashboard pane-border-format " #{pane_title} "
$TMUX_BIN set-option -t dashboard pane-border-style fg=colour240
$TMUX_BIN set-option -t dashboard pane-active-border-style fg=colour39

$TMUX_BIN select-pane -t dashboard:0.0 -T " ⚡ [Phase 1: GPU 0 | Loss 분기 + DualStream] "
$TMUX_BIN select-pane -t dashboard:0.1 -T " 🚀 [Phase 2: GPU 1 | Multi-Frame 디코딩] "
$TMUX_BIN select-pane -t dashboard:0.2 -T " 🖥️ [Server Status: Dual RTX 3090, CPU, RAM] "

echo "Dashboard session created successfully."
