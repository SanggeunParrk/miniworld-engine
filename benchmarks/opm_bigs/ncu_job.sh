#!/bin/bash
# ncu_job.sh TAG ROW L S BACKEND [KERNEL_REGEX]
D=/home/psk6950/miniworld-engine-scratch/wt-opm-bigs/benchmarks/opm_bigs
M=gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,lts__t_bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,sm__throughput.avg.pct_of_peak_sustained_elapsed
K=${6:-regex:.}
/usr/local/cuda/bin/ncu --clock-control none --profile-from-start off -k "$K" --metrics $M --csv -o $D/logs/ncu_$1 -f $D/job.sh python $D/prof.py $2 $3 $4 $5 1
/usr/local/cuda/bin/ncu --import $D/logs/ncu_$1.ncu-rep --csv --page raw 2>/dev/null | python3 -c "pass" 2>/dev/null
