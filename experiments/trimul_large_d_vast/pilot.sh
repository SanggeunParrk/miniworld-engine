#!/usr/bin/env bash
set -euo pipefail
export LD_PRELOAD=/workspace/envs/engine/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so.12:/workspace/envs/engine/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12
mkdir -p /workspace/vast-results/trimul-large-d
python -u experiments/trimul_large_d_vast/pilot.py --width "$1" --length "$2" "${@:3}" > "/workspace/vast-results/trimul-large-d/pilot-D$1-L$2.log" 2>&1
