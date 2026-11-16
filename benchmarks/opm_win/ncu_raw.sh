#!/bin/bash
# ncu_raw.sh TAG ROW L S BACKEND   raw metrics (stall reasons, pipes, L2 / DRAM) of the prologue / epilogue kernels
D=$(cd "$(dirname "$0")" && pwd); TAG=$1; shift
M=gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed
M+=,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed
M+=,sm__inst_executed.sum,smsp__warps_active.avg,smsp__issue_active.avg.pct_of_peak_sustained_elapsed
for r in barrier branch_resolving dispatch_stall drain gmma imc_miss lg_throttle long_scoreboard math_pipe_throttle membar mio_throttle misc no_instruction not_selected selected short_scoreboard sleeping tex_throttle wait; do M+=,smsp__warps_issue_stalled_$r.avg; done
ncu --kernel-name regex:"opm_epilogue_kernel|opm_prologue_tma_wide" --launch-skip 2 --launch-count 2 --metrics $M --csv --page raw python $D/kprof.py "$@" --iters 1 > $D/logs/ncuraw_$TAG.csv 2> $D/logs/ncuraw_$TAG.err
echo "exit $?"
