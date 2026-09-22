#!/bin/bash
# usage: run.sh <args to bench.py>   (compute node; node01)
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/psk6950/miniworld-engine-dit/src:/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/upstream/common/opt_core
export OPT_CORE_CELL_CENSUS_PRINT=0
export TRITON_CACHE_DIR=/home/psk6950/MiniWorld/runs/diffusion_vs_anthropic_20260922/triton_build_cache
cd /home/psk6950/miniworld-engine-dit/experiments/token_dit_fused
python -W ignore bench.py "$@"
