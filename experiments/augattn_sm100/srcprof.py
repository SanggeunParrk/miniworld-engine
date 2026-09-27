"""ncu SASS samples aggregated per CUDA source line (file, line) through nvdisasm line info.  python srcprof.py <cubin> <sass.csv> [top]"""
import csv, re, sys, collections
cub, sass = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
import subprocess
dis = subprocess.run(["nvdisasm", "--print-line-info", cub], capture_output=True, text=True).stdout
cur = None; off2 = {}
for l in dis.split("\n"):
    m = re.search(r"//## File \"([^\"]+)\", line (\d+)", l)
    if m: cur = (m.group(1).split("/")[-1], int(m.group(2)))
    m2 = re.match(r"\s*/\*([0-9a-f]{4,})\*/", l)
    if m2 and cur: off2[int(m2.group(1), 16)] = cur
rows = list(csv.reader(open(sass)))
h = rows[1] if rows[0][0] == "Kernel Name" else rows[0]
body = [r for r in rows if len(r) == len(h) and r[0].startswith("0x")]
S = h.index("Warp Stall Sampling (All Samples)")
cols = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
base = int(body[0][0], 16)
smp = collections.Counter(); why = collections.defaultdict(collections.Counter)
for r in body:
    k = off2.get(int(r[0], 16) - base, ("?", 0))
    smp[k] += int(r[S])
    for c in cols: why[k][c[6:]] += int(r[h.index(c)] or 0)
srcs = {}
tot = sum(smp.values())
for (f, ln), v in smp.most_common(top):
    if f not in srcs:
        try: srcs[f] = open("src/" + f).read().split("\n")
        except OSError: srcs[f] = []
    txt = srcs[f][ln - 1].strip()[:78] if 0 < ln <= len(srcs[f]) else ""
    w = " ".join(f"{a}:{b}" for a, b in why[(f, ln)].most_common(3) if b)
    print(f"{v:6d} {100*v/tot:5.1f}% {f}:{ln:<4d} [{w}] {txt}")
