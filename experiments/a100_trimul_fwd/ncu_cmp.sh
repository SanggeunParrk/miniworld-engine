#!/bin/bash
# ncu_cmp.sh <kernel-regex> <tag> <script args...>: metrics summary of one launch (after 3 warm) of a kernel from prof_contract.py
S=$HOME/.cache/miniworld-a100/ncu; mkdir -p $S; K=$1; TAG=$2; shift 2
/usr/local/cuda-12.9/bin/ncu -k regex:$K -c 1 -s 3 --section SpeedOfLight --section SourceCounters --import-source yes \
  --metrics regex:smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio,smsp__inst_executed_pipe_tensor.sum,smsp__inst_executed.sum,smsp__cycles_active.avg,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,launch__grid_size,launch__block_size,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.sum \
  -f -o $S/$TAG python prof_contract.py "$@" >/dev/null 2>&1
/usr/local/cuda-12.9/bin/ncu --import $S/$TAG.ncu-rep --page raw --csv 2>/dev/null > $S/$TAG.raw.csv
/usr/local/cuda-12.9/bin/ncu --import $S/$TAG.ncu-rep --page source --csv --print-source sass 2>/dev/null > $S/$TAG.sass.csv
python3 - "$S/$TAG" <<'PY'
import csv, sys
raw = list(csv.reader(open(sys.argv[1] + ".raw.csv"))); h, v = raw[0], raw[2]
g = lambda k: v[h.index(k)] if k in h else "?"
print(g("Kernel Name")[:90])
for k in ["gpu__time_duration.sum","smsp__cycles_active.avg","smsp__inst_executed.sum","smsp__inst_executed_pipe_tensor.sum","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
          "launch__registers_per_thread","launch__grid_size","launch__block_size","launch__occupancy_limit_registers","dram__bytes_read.sum","dram__bytes_write.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed","lts__t_sectors.sum","l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"]:
    print(f"  {k:62s} {g(k)}")
st = sorted(((k.split("stalled_")[1].split("_per")[0], float(v[i] or 0)) for i, k in enumerate(h) if k.endswith("_per_issue_active.ratio") and "issue_stalled" in k and "not_issued" not in k), key=lambda x: -x[1])
print("  stalls:", ", ".join(f"{k} {x:.2f}" for k, x in st[:8]))
PY
