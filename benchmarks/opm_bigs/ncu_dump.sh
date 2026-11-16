#!/bin/bash
ENV=/home/psk6950/miniworld-engine/.pixi/envs/default; export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
for f in "$@"; do echo "== $f"; /usr/local/cuda/bin/ncu --import $f --page raw --csv 2>/dev/null | python -c "
import sys,csv
r=list(csv.reader(sys.stdin)); h=r[0]
for row in r[2:]:
    d=dict(zip(h,row)); print(d['Kernel Name'][:36], {k.split('__')[-1][:34] if False else k[:60]:v for k,v in d.items() if k in ('gpu__time_duration.sum','sm__cycles_elapsed.avg.per_second','sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active','dram__throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','lts__throughput.avg.pct_of_peak_sustained_elapsed','lts__t_sector_hit_rate.pct','launch__grid_size','sm__warps_active.avg.pct_of_peak_sustained_active')})
"; done
