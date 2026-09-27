#!/bin/bash
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/upstream/common/opt_core
export OPT_CORE_CELL_CENSUS_PRINT=0 PYTHONWARNINGS=ignore
cd /home/psk6950/miniworld-engine-dit2/experiments/token_dit_overlap/vs_anthropic
python -W ignore "$@"
