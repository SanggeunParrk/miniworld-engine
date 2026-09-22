#!/bin/bash
set -euo pipefail
ulimit -c 0
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
export MINIWORLD_ANTHROPIC_ROOT=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream
export PYTHONPATH=/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src:$MINIWORLD_ANTHROPIC_ROOT/common/opt_core
export TORCH_EXTENSIONS_DIR=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/extensions
export MODEL_OPT_JIT_ROOT=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/jit
export TRITON_CACHE_DIR=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/triton
export MAX_JOBS=4
export OPT_CORE_CELL_CENSUS_PRINT=0
export TRIMUL_NATIVE_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/native_rebuilt/build
export MINIWORLD_MATHDX_HOME=/home/psk6950/mathdx_dl/extracted/nvidia/mathdx
exec "$@"
