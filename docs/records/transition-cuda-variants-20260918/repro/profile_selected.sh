#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
prefix="$root/selected-$2-D$1"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda-12.9/bin/ncu --profile-from-start off --section SpeedOfLight --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --cache-control none --force-overwrite -o "$prefix" bash "$root/env.sh" "$root/profile_selected.py" --d "$1" --variant "$2"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda-12.9/bin/ncu --import "$prefix.ncu-rep" --page details > "$prefix.txt"
