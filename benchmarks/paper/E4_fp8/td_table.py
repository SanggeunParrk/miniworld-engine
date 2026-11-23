"""Pivot table of fp8_pair.py token DiT records: default / FP8 speed ratio per (module, widths, S) x length. >1 = FP8 faster.

    python td_table.py records.jsonl [more.jsonl ...]     (a '*' marks a case the FP8 route did not fire on: ratio ~1, default kernels)
"""

import collections
import json
import statistics
import sys

rows = collections.defaultdict(dict)
lens = set()
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        k = r["kwargs"]
        key = f'{r["module"].replace("TokenDiT", "")} dc{k["d_cond"]} dp{k["d_pair"]} h{k["n_head"]} S{r["samples"]}'
        L = r["length"]
        lens.add(L)
        rows[key][L] = (r["ratio_default_over_fp8"], r["fp8_route_fired"])
lens = sorted(lens)
print("| case | " + " | ".join(f"L{L}" for L in lens) + " |")
print("|---|" + "---|" * len(lens))
worst = []
for key in sorted(rows):
    cells = []
    for L in lens:
        v = rows[key].get(L)
        cells.append("" if v is None else f"{v[0]:.3f}" + ("" if v[1] else "*"))
        if v is not None and v[1]:
            worst.append(v[0])
    print(f"| {key} | " + " | ".join(cells) + " |")
if worst:
    print(f"\nfired cases: {len(worst)}, min {min(worst):.3f}, median {statistics.median(worst):.3f}, max {max(worst):.3f}, below 1.0: {sum(w < 1 for w in worst)}")
