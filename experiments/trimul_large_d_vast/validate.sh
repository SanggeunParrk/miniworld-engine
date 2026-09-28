#!/usr/bin/env bash
set -euo pipefail
width=${1:?width}; length=${2:?length}
# Prefer the cu128 wheel runtime used by the frozen Lt selections.
export LD_PRELOAD=/workspace/envs/engine/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so.12:/workspace/envs/engine/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12
root=/workspace/experiments/trimul-large-d/runs/trimul_d256_bwd_sol90_stage2_20260923
out=/workspace/vast-results/trimul-large-d
mkdir -p "$out"
export SLURM_JOB_ID="vast-D${width}-L${length}"
if [[ "$width" == 256 ]]; then
 script=validate_d256_pool_checkpoint.py
 export SLURM_ARRAY_TASK_ID=$((length==768))
else
 script=validate_wide_checkpoint24.py
 export SLURM_ARRAY_TASK_ID=$(( (width==512)*2+(length==768) ))
fi
python -u "$root/$script" > "$out/validate-D${width}-L${length}.log" 2>&1
cp "$root"/validation-*"$SLURM_JOB_ID".json "$out/"
