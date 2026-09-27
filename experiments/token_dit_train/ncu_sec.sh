#!/bin/bash
# ncu_sec.sh <kernel regex> <script args...>: SOL, warp-state and scheduler sections of one launch (text)
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
cd /home/psk6950/miniworld-engine-dit2/experiments/token_dit_train
K=$1; shift
/usr/local/cuda-12.9/bin/ncu -k regex:$K -s 1 -c 1 --section SpeedOfLight --section WarpStateStats --section SchedulerStats \
  --section Occupancy --section ComputeWorkloadAnalysis python -W ignore "$@" 2>&1 | grep -E "^\s+[A-Z][A-Za-z /\.-]+\s{2,}" | sed 's/^/sec: /'
