#!/bin/bash
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/psk6950/miniworld-engine-dit2/src:/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/upstream/common/opt_core
export OPT_CORE_CELL_CENSUS_PRINT=0
export TRITON_CACHE_DIR=/home/psk6950/MiniWorld/runs/diffusion_vs_anthropic_20260922/triton_build_cache
cd /home/psk6950/miniworld-engine-dit2/experiments/token_dit_overlap
/usr/local/cuda-12.9/bin/ncu --target-processes all --nvtx --nvtx-include "prof/" --csv --page raw \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_sectors_op_read.sum \
    python -W ignore ncu_step.py "$@"
