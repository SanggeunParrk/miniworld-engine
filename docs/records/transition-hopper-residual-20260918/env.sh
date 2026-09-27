#!/bin/bash
set -euo pipefail
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src:/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine
export MAX_JOBS=4
export TORCH_EXTENSIONS_DIR=/home/psk6950/MiniWorld/runs/transition_h100_residual_20260918/torch_extensions
export MINIWORLD_MATHDX_HOME=/home/psk6950/mathdx_dl/extracted/nvidia/mathdx
if [ "${1:-}" = "--memcheck" ]; then
  shift
  exec /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode=90 python -u "$@"
fi
exec python -u "$@"
