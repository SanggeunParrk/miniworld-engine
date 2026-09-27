#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda-12.9/bin/ncu --profile-from-start off --section SpeedOfLight --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --cache-control none --force-overwrite -o "$root/gate-full-D128" bash "$root/env.sh" "$root/profile_gate.py"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /usr/local/cuda-12.9/bin/ncu --import "$root/gate-full-D128.ncu-rep" --page details > "$root/gate-full-D128.txt"
bash "$root/runtime_probe.sh"
