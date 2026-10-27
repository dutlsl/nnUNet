#!/usr/bin/env bash
# GAZER watchdog -- keeps the GRPO training alive across IDE restarts,
# crashes, OOM kills, and machine reboots. A system cron job (not the IDE)
# invokes this, so it keeps running after the terminal / IDE is closed.
#
# Behaviour (single source of truth for "ensure the run is running"):
#   * If train_gazer.py is running            -> no-op.
#   * If it is NOT running and it did NOT
#     finish gracefully (no done marker)      -> relaunch, detached via setsid
#       so it survives the IDE/agent process group being killed.
#   * If it finished gracefully (done marker) -> leave it; do not relaunch.
#
# The launched run touches logs/.gazer_done ONLY on a clean exit (the
# `&& touch` runs only when python returns 0). A crash/kill leaves the
# marker absent, so the next cron fire relaunches automatically.
set -u
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

REPO=/home/iulab1/PycharmProjects/nnUNet
LOG=$REPO/logs/gazer_grpo_backbone_t5.log
DONE=$REPO/logs/.gazer_done
HOLD=$REPO/logs/.gazer_hold
CFG=configs/gazer_grpo_t5.yaml
PY=$REPO/nnunet_env/bin/python
LOCK=/tmp/.gazer_watchdog.lock

mkdir -p "$REPO/logs"

# Single instance: if another watchdog is mid-launch, back off.
exec 200>"$LOCK"
flock -n 200 || exit 0

# Ensure the persistent event logger runs (setsid-detached, PPID=1; it
# tails the training log and appends critical lines, timestamped, to
# logs/gazer_events.log). Kept alive even while training is running, so
# the IDE-restart-surviving event history has no gaps.
pgrep -f "gazer_event_logger\.sh" >/dev/null || \
  setsid bash "$REPO/tools/gazer_event_logger.sh" </dev/null >/dev/null 2>&1 &

# Already running? -> nothing to do.
pgrep -f "nnunet_env/bin/python.*train_gazer\.py.*$CFG" >/dev/null && exit 0

# Finished cleanly? -> leave it.
[ -f "$DONE" ] && exit 0

# Manual hold: .gazer_hold present -> pause auto-launch (used while the
# GPUs are in a smoke test or the agent is mid-diagnosis). Remove it to
# re-arm the watchdog.
[ -f "$HOLD" ] && exit 0

# (Re)launch fully detached: setsid puts it in its own session so a SIGKILL
# of the IDE/agent process group cannot reach it. On a clean exit it writes
# the done marker; on any crash/kill the marker stays absent.
setsid bash -c "cd '$REPO' && \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u train_gazer.py --config $CFG --gpu 0 \
  >> '$LOG' 2>&1 && touch '$DONE'" </dev/null >/dev/null 2>&1 &

exit 0
