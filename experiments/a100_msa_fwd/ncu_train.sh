#!/bin/bash
# ncu_op.sh <kernel-regex> <op> <L> [tag] [bench -D args...]: Nsight Compute summary (stalls, pipes, memory, instruction mix) of one launch after 3 warm ones.
S=$HOME/.cache/miniworld-a100/ncu; mkdir -p $S; TAG=${4:-$2-$3}; EXTRA="${@:5}"
NCU=/usr/local/cuda-12.9/bin/ncu
$NCU -k regex:$1 -c 1 -s 3 --section SourceCounters --section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis --import-source yes \
  --metrics regex:smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio,smsp__inst_executed_pipe_tensor.sum,smsp__inst_executed.sum,smsp__cycles_active.avg,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,lts__t_sector_hit_rate.pct \
  -f -o $S/$TAG python bench_train.py --op $2 --length $3 --no-time --repeat 5 $EXTRA >/dev/null 2>&1
$NCU --import $S/$TAG.ncu-rep --page raw --csv 2>/dev/null > $S/$TAG.raw.csv
$NCU --import $S/$TAG.ncu-rep --page source --csv --print-source sass 2>/dev/null > $S/$TAG.sass.csv
python3 - "$S/$TAG" <<'PY'
import csv, sys, collections
base = sys.argv[1]
raw = list(csv.reader(open(base + ".raw.csv")))
h, v = raw[0], raw[2]
get = lambda k: v[h.index(k)] if k in h else "?"
print("duration_us", get("gpu__time_duration.sum"), "| clk", get("sm__cycles_elapsed.avg.per_second"), "| inst", get("smsp__inst_executed.sum"),
      "| tensor inst", get("smsp__inst_executed_pipe_tensor.sum"), "| dram R/W", get("dram__bytes_read.sum"), get("dram__bytes_write.sum"),
      "| L2 bytes", get("lts__t_bytes.sum"), "hit%", get("lts__t_sector_hit_rate.pct"), "| smem conflicts", get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"))
print("regs", get("launch__registers_per_thread"), "| warps active %", get("sm__warps_active.avg.pct_of_peak_sustained_active"),
      "| tensor active %", get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"))
st = sorted(((k.split("stalled_")[1].split("_per")[0], float(v[i] or 0)) for i, k in enumerate(h) if "issue_stalled" in k and k.endswith("_per_issue_active.ratio") and "not_issued" not in k), key=lambda x: -x[1])
print("stalls/issue:", ", ".join(f"{k} {x:.2f}" for k, x in st[:9]))
rows = list(csv.reader(open(base + ".sass.csv")))
hd = rows[1]; iS, iE = hd.index("Source"), hd.index("Instructions Executed")
ops = collections.Counter(); tot = 0
for r in rows[2:]:
    try: n = int(r[iE])
    except Exception: continue
    tok = r[iS].split(); op = (tok[1] if tok and tok[0].startswith("@") else (tok[0] if tok else "?")).split(".")[0]
    ops[op] += n; tot += n
print("ops:", ", ".join(f"{o} {100*n/tot:.1f}%" for o, n in ops.most_common(14)))
PY
