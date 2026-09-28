#!/usr/bin/env bash
set -euo pipefail
cd /home/psk6950/miniworld-engine
test -n "${SLURM_JOB_ID:-}" || { echo 'Run through Slurm, not the login node.' >&2; exit 2; }
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/experiments/transition_wide_fusion:$PWD/experiments/transition_shapes_vast"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$PWD/.bench/transition-wide-local/cache/extensions"
export MINIWORLD_ENGINE_JIT_ROOT="$PWD/.bench/transition-wide-local/cache/native"
export TRITON_CACHE_DIR="$PWD/.bench/transition-wide-local/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.bench/transition-wide-local/cache/inductor"
export MINIWORLD_TRANSITION_WIDE_SAVE_H=0
mkdir -p "$TORCH_EXTENSIONS_DIR" "$MINIWORLD_ENGINE_JIT_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
mode=${1:-profile_stages}
shift || true
exec python -u "experiments/transition_wide_fusion/${mode}.py" "$@"
