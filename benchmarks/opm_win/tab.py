"""Fair-protocol table: per (row, L, S) the backend / PyTorch ratio of each stack (cool: median of reps; sustained: mean of the rotated
reps), and the lead of each engine backend over Anthropic = (Anthropic / PyTorch_kit) / (backend / PyTorch_engine) (> 1: faster).
   python tab.py KIT.jsonl ENGINE.jsonl[:label] ...   (labels name the engine record sets, e.g. before / after)"""
import json
import statistics
import sys
from collections import defaultdict


def load(path):
    d = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r.get("status") != "ok":
            d[(r["row"], r["length"], r["depth"])][r["backend"]] = None
            continue
        d[(r["row"], r["length"], r["depth"])][r["backend"]] = (statistics.median(r["cool"]), statistics.fmean(r["sus"]))
    return d


kit = load(sys.argv[1])
sets = []
for arg in sys.argv[2:]:
    path, _, label = arg.partition(":")
    sets.append((label or path, load(path)))
cols = []
for label, d in sets:
    for b in ("cuda", "triton"):
        cols.append((label, b))
print("| row | L | S | Anthropic/PT cool | sus | " + " | ".join(f"{lab} {b}/PT cool | sus | lead vs Anthropic cool | sus" for lab, b in cols) + " |")
print("|" + "---|" * (5 + 4 * len(cols)))
keys = sorted(set(kit) | {k for _, d in sets for k in d})
for k in keys:
    a = kit.get(k, {})
    ar = None
    if a.get("anthropic") and a.get("pytorch"):
        ar = (a["anthropic"][0] / a["pytorch"][0], a["anthropic"][1] / a["pytorch"][1])
    cells = [str(k[0]), str(k[1]), str(k[2]), f"{ar[0]:.3f}" if ar else "-", f"{ar[1]:.3f}" if ar else "-"]
    for label, b in cols:
        e = dict(sets)[label].get(k, {})
        if e.get(b) and e.get("pytorch"):
            er = (e[b][0] / e["pytorch"][0], e[b][1] / e["pytorch"][1])
            cells += [f"{er[0]:.3f}", f"{er[1]:.3f}"]
            cells += [f"{ar[0] / er[0]:.3f}", f"{ar[1] / er[1]:.3f}"] if ar else ["-", "-"]
        else:
            cells += ["-"] * 4
    print("| " + " | ".join(cells) + " |")
