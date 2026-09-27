#!/bin/bash
# ncu_lines.sh <kernel-regex> <variant> <L> <tag>: ncu_stalls.sh (with PROG) + stall samples aggregated per CUDA source line
bash "$(dirname "$0")/ncu_stalls.sh" "$1" "$2" "$3" "$4"
S=$HOME/.cache/miniworld-a100/ncu
/usr/local/cuda-12.9/bin/ncu --import $S/$4.ncu-rep --page source --csv --print-source cuda,sass > $S/$4.mix.csv 2>/dev/null
python3 - "$S/$4.mix.csv" <<'PY'
import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
def num(x):
    try: return float(x)
    except Exception: return 0.0
fl = collections.OrderedDict(); fn = "?"
for r in rows:
    if not r: continue
    if r[0] == "File Path": fn = r[1].split('/')[-1]; continue
    if r[0] in ("Function Name", "Line No"): continue
    if r[0] != "" and len(r) > 7: fl[(fn, r[0], r[1].strip())] = [num(r[4]), num(r[7])]
tot = sum(v[0] for v in fl.values()) or 1; ti = sum(v[1] for v in fl.values()) or 1
for k, v in sorted(fl.items(), key=lambda kv: -kv[1][0])[:24]:
    print(k[0][:14].ljust(14), k[1].rjust(4), f"{100*v[0]/tot:5.1f}% inst {100*v[1]/ti:5.1f}%", k[2][:96])
PY
