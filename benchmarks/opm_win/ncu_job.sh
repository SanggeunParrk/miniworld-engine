#!/bin/bash
# ncu_job.sh TAG ROW L S BACKENDS...   prologue / epilogue kernels of the cuda and triton paths (cuBLAS skipped), a few sections
D=$(cd "$(dirname "$0")" && pwd); TAG=$1; shift
ncu --kernel-name regex:"opm_epilogue_kernel|opm_prologue_tma_wide|_opm_epilogue_kernel|_opm_layernorm_gemm_kernel" --launch-skip 0 --launch-count 40 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats --section Occupancy --section LaunchStats --section SchedulerStats \
  --csv --page details python $D/kprof.py "$@" --iters 1 > $D/logs/ncu_$TAG.csv 2> $D/logs/ncu_$TAG.err
echo "exit $?"
