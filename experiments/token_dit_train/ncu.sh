#!/bin/bash
# ncu.sh <kernel regex> <script args...>: DRAM / L2 / tensor metrics of the last launch matching the regex
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
cd /home/psk6950/miniworld-engine-dit2/experiments/token_dit_train
K=$1; shift
/usr/local/cuda-12.9/bin/ncu -k regex:$K -s 1 -c 1 --csv --page raw \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct \
  python -W ignore "$@" | tail -n 3 | python3 -c "
import sys, csv
M = {'gpu__time_duration', 'dram__bytes_read', 'dram__bytes_write', 'lts__t_bytes', 'lts__throughput', 'gpu__dram_throughput', 'sm__pipe_tensor_cycles_active', 'sm__inst_executed_pipe_xu', 'lts__t_sector_hit_rate'}
rows = list(csv.reader(sys.stdin))
h, u, v = rows[0], rows[1], rows[2]
for a, b, c in zip(h, u, v):
    if a.split('.')[0] in M and (a.endswith('.sum') or a.endswith('.pct') or 'pct_of_peak' in a and a.split('.')[1] == 'avg'): print(f'ncu: {a:70s} {c:>16s} {b}')
"
