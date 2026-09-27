#!/bin/bash
# usage: run_b200.sh <script.py> [args...]   (B200, colleague's account: every cache under the personal workspace, GPU 6 only)
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950
export TMPDIR=$W/.tmp XDG_CACHE_HOME=$W/.cache MPLCONFIGDIR=$W/.cache/matplotlib PYTHONNOUSERSITE=1
export TRITON_CACHE_DIR=$W/.cache/triton_dit TORCH_EXTENSIONS_DIR=$W/.cache/torch_extensions_dit
export MINIWORLD_ENGINE_JIT_ROOT=$W/.cache/miniworld_engine_jit_dit CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$W/mw-dit/src:$W/refs/uplifting-biomolecular-modeling/common/opt_core
export OPT_CORE_CELL_CENSUS_PRINT=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
source $W/miniworld-engine/.venv/bin/activate
cd "$(dirname "$0")"
python -W ignore "$@"
