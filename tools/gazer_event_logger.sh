#!/usr/bin/env bash
# GAZER persistent event tracker.
#
# The Grok session's monitors die with the IDE. This logger is the OS-level
# replacement: setsid-detached (PPID=1), launched by the watchdog (cron
# every 15 min, @reboot), it tails the training log and appends every
# critical line, timestamped, to logs/gazer_events.log. That file is the
# complete event history across IDE restarts / agent absences.
#
# Idempotent: flock on /tmp/.gazer_event_logger.lock; the watchdog only
# (re)launches it when no instance is running.
set -u
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

REPO=/home/iulab1/PycharmProjects/nnUNet
LOG=$REPO/logs/gazer_grpo_backbone_t5.log
EV=$REPO/logs/gazer_events.log
LOCK=/tmp/.gazer_event_logger.lock

mkdir -p "$REPO/logs"
exec 200>"$LOCK"
flock -n 200 || exit 0

PATTERN='VAL |reward-landscape cache complete|Traceback|[Nn][Aa][Nn]|early stop|\[GAZER\] done|worker died|RuntimeError|CUDA out of memory|Killed|\[phase0\] cache '

# tail -F keeps following across rename/rotation; if the pipeline ever
# breaks the outer loop restarts it.
while true; do
  if [ -f "$LOG" ]; then
    tail -n 0 -F "$LOG" 2>/dev/null \
      | grep --line-buffered -E "$PATTERN" \
      | while IFS= read -r line; do
          printf '%s %s\n' "$(date '+%F %T')" "$line" >> "$EV"
        done
  fi
  sleep 5
done
