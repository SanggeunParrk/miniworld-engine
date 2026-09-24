"""Parse `nsys stats -r cuda_gpu_trace --format csv` of graph_trace.py: take the LAST replay, classify kernels, and
report per-block time per kernel class plus the gaps (start of kernel i+1 - end of kernel i; negative = overlap)."""
import csv, sys, collections
rows = [r for r in csv.DictReader(open(sys.argv[1]))]
NB = 24
def cls(n):
    if "attn_kernel" in n: return "attention core"
    if "resgate_adaln" in n: return "resgate+AdaLN rows"
    if "GemmGated" in n: return "expand+SwiGLU GEMM"
    if "quackgemm" in n or "nvjet" in n or "gemm" in n.lower(): return "plain GEMM"
    return "other"
k = [r for r in rows if r.get("Name")]
key_start = next(c for c in k[0] if c.startswith("Start"))
key_dur = next(c for c in k[0] if c.startswith("Duration"))
k.sort(key=lambda r: int(r[key_start]))
# one replay = everything between two consecutive first-GEMM-of-the-step markers; use the last full replay
n_per = len(k) // int(sys.argv[2])
last = k[-n_per:]
t0 = int(last[0][key_start]); t1 = max(int(r[key_start]) + int(r[key_dur]) for r in last)
per = collections.defaultdict(float); gaps = collections.defaultdict(list)
for i, r in enumerate(last):
    per[cls(r["Name"])] += int(r[key_dur]) / 1e3 / NB
    if i + 1 < len(last):
        g = int(last[i + 1][key_start]) - (int(r[key_start]) + int(r[key_dur]))
        gaps[(cls(r["Name"]), cls(last[i + 1]["Name"]))].append(g / 1e3)
wall = (t1 - t0) / 1e3 / NB
print(f"kernels per replay {len(last)}; wall {wall:.1f} us/block; sum of kernel durations {sum(per.values()):.1f} us/block")
for c, t in sorted(per.items(), key=lambda x: -x[1]):
    print(f"  {c:<22} {t:7.1f} us/block  {100 * t / wall:5.1f} %")
print("gaps between consecutive kernels (us, median / min / max; negative = the next one started before this ended):")
for (x, y), v in sorted(gaps.items(), key=lambda x: -len(x[1])):
    v = sorted(v)
    print(f"  {x:<20} -> {y:<22} n={len(v):3d}  {v[len(v)//2]:6.2f} / {v[0]:6.2f} / {v[-1]:6.2f}  total {sum(v)/NB:6.2f} us/block")
