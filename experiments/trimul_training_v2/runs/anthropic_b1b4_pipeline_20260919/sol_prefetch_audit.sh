#!/bin/bash
set -euo pipefail
P=runs/anthropic_b1b4_pipeline_20260919
E=runs/anthropic_adoption_20260919/env.sh
export PYTHONNOUSERSITE=1
for tool in memcheck racecheck synccheck; do
 for length in 64 384; do
  timeout -k 10s 480s bash "$E" compute-sanitizer --tool "$tool" --kernel-name kns=dual_b1b4 --error-exitcode 3 python -u -B "$P/sanitize_experiment.py" --source dual_ln_prefetch --length "$length" --count 132 --part 2 > "$P/sol-prefetch-$tool-L$length.log" 2>&1
  printf 'RESULT %s L%s passed\n' "$tool" "$length"
 done
done
for length in 64 384; do
 timeout -k 10s 180s bash "$E" compute-sanitizer --tool memcheck --error-exitcode 3 python -u -B "$P/sanitize_experiment.py" --source dual_ln_prefetch --length "$length" --count 132 --part 1 > "$P/sol-prefetch-split-memcheck-L$length.log" 2>&1
 printf 'RESULT split-memcheck L%s passed\n' "$length"
done
metrics=gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,lts__t_sectors.sum
for length in 384 768; do
 timeout -k 5s 300s bash "$E" ncu --clock-control none --cache-control none --replay-mode application --metrics "$metrics" --profile-from-start off -k regex:dual_b1b4 --force-overwrite -o "$P/sol-prefetch-warm-L$length" python -u -B "$P/profile_experiment.py" --source dual_ln_prefetch --length "$length" > "$P/sol-prefetch-warm-L$length.log" 2>&1
 bash "$E" ncu --import "$P/sol-prefetch-warm-L$length.ncu-rep" --page raw --csv > "$P/sol-prefetch-warm-L$length.csv"
 printf 'RESULT warm-NCU L%s passed\n' "$length"
done
timeout -k 5s 300s bash "$E" ncu --set full --import-source yes -k regex:dual_b1b4 --profile-from-start off --force-overwrite -o "$P/sol-prefetch-full-L768" python -u -B "$P/profile_experiment.py" --source dual_ln_prefetch --length 768 > "$P/sol-prefetch-full-L768.log" 2>&1
for length in 384 768; do
 bash "$E" ncu --import "$P/sol-prefetch-full-L$length.ncu-rep" --page raw --csv > "$P/sol-prefetch-full-L$length.csv"
done
bash "$E" ncu --import "$P/sol-prefetch-full-L384.ncu-rep" --page source --print-source sass --csv > "$P/sol-prefetch-source.csv"
printf 'RESULT full-NCU exports passed\n'
