"""Turn gemm_sweep.py logs into a PLAIN_CFGS candidate list: the union of each shape's top configs, so _pick_mm can
find every per-shape winner without racing the whole grid at runtime."""
import re
import sys
from collections import OrderedDict

TOP = 3
pat = re.compile(r"quack tM(\d+) tN(\d+) cl(\d+)x(\d+) pp(\d)|quack tN(\d+) cl(\d+)x(\d+) pp(\d)")
best = OrderedDict()
shape = None
for path in sys.argv[1:]:
    seen = {}
    for line in open(path):
        if line and line[0].isalpha() and ":" in line:
            shape = line.split(":")[0]
            seen[shape] = 0
            continue
        m = pat.search(line)
        if not m or shape is None or seen.get(shape, TOP) >= TOP:
            continue
        g = m.groups()
        cfg = (int(g[0]), int(g[1]), int(g[2]), int(g[3]), bool(int(g[4]))) if g[0] else \
              (128, int(g[5]), int(g[6]), int(g[7]), bool(int(g[8])))
        best[cfg] = best.get(cfg, 0) + 1
        seen[shape] += 1
for cfg in best:
    print(f"{cfg},")
