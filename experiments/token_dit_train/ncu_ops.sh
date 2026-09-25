#!/bin/bash
# ncu_ops.sh <kernel regex> <script args...>: executed SASS instructions per opcode (one launch), sorted
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
cd /home/psk6950/miniworld-engine-dit2/experiments/token_dit_train
K=$1; shift
/usr/local/cuda-12.9/bin/ncu -k regex:$K -s 1 -c 1 --metrics sass__inst_executed_per_opcode --print-metric-instances details \
  python -W ignore "$@" > logs/ncu_ops_raw.txt 2>&1
python3 - <<'PY'
import re
t = open("logs/ncu_ops_raw.txt").read()
m = re.search(r"sass__inst_executed_per_opcode.*?\((.*?)\)", t, re.S)
pairs = re.findall(r"([A-Z0-9_.]+):\s*([\d,]+)", m.group(1)) if m else []
tot = sum(int(v.replace(",", "")) for _, v in pairs)
for k, v in sorted(pairs, key=lambda p: -int(p[1].replace(",", ""))):
    n = int(v.replace(",", ""))
    print(f"ops: {k:12s} {n:>14,d} {100 * n / tot:6.2f} %")
print(f"ops: total {tot:,d}")
PY
