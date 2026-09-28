#!/bin/bash
set -euo pipefail
root=/home/psk6950/miniworld-engine
envroot=/home/psk6950/MiniWorld/.pixi/envs/cu128
export PATH="$envroot/bin:/usr/local/cuda-12.9/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH="$envroot/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root/src"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export CUTLASS_PATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2
export TORCH_EXTENSIONS_DIR="$root/experiments/triattn_local_d128/cache/extensions"
export TORCHINDUCTOR_CACHE_DIR="$root/experiments/triattn_local_d128/cache/inductor"
export TRITON_CACHE_DIR="$root/experiments/triattn_local_d128/cache/triton"
export MINIWORLD_ENGINE_JIT_ROOT="$root/experiments/triattn_local_d128/cache/native"
cd "$root"
exec "$@"
