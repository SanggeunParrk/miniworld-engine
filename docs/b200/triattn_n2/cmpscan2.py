# cmpscan2.py A.pt B.pt : per class / width summary of a bitscan comparison
import sys, torch, collections
a = torch.load(sys.argv[1]); b = torch.load(sys.argv[2])
g = collections.defaultdict(lambda: [0, 0, 0.0])
for k in a:
    if k.endswith("|repeat") or k not in b: continue
    cls = k.split("-{")[0] + " " + k.split("-{")[1].split("}")[0]
    d = (a[k] - b[k]).abs().max().item()
    g[cls][0] += 1; g[cls][1] += int(d == 0.0); g[cls][2] = max(g[cls][2], d)
for c, (n, e, w) in g.items(): print(f"{c:70s} cases {n} bitwise-equal {e} worst {w}")
