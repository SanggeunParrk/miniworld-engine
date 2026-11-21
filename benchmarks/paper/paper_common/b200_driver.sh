#!/bin/bash
# b200_driver.sh [GPUS...] : start one lane per GPU (default 0 1 2 3, never 4-7) and wait for them; writes $PAPER/b200_driver.done at the end.
#   nohup paper_common/b200_driver.sh > b200_driver.log 2>&1 &     (pids in b200_driver.pids: kill exactly those to stop)
PAPER=$(cd "$(dirname "$0")/.." && pwd)
GPUS=("${@:-0 1 2 3}"); read -ra GPUS <<< "${GPUS[*]}"
: > "$PAPER/b200_driver.pids"; rm -f "$PAPER/b200_driver.done"
for g in "${GPUS[@]}"; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${used:-0}" -gt 2000 ]; then echo "driver: GPU $g busy ($used MiB): not used"; continue; fi
  setsid bash "$PAPER/paper_common/b200_lane.sh" "$g" > "$PAPER/b200_lane$g.log" 2>&1 < /dev/null &
  echo $! >> "$PAPER/b200_driver.pids"; echo "driver: lane GPU $g pid $!"
  sleep 20   # stagger the starts (JIT / cache locks)
done
wait
date -Is > "$PAPER/b200_driver.done"
