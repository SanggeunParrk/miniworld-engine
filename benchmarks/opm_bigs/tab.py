import json, sys, collections
def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("status") == "ok": d[(r["row"], r["length"], r["depth"], r["backend"])] = r["median_ms"]
    return d
new = load(sys.argv[1]); old = {}
for p in sys.argv[2:]: old.update(load(p))
print("row  L    S      pytorch  triton | cuda_old  cuda_new | new/triton")
for k in sorted({k[:3] for k in new}):
    g = lambda d, b: d.get(k + (b,))
    f = lambda x: f"{x:8.3f}" if x else "     n/a"
    c, t = g(new, "cuda"), g(new, "triton")
    print(k[0], f"{k[1]:4d} {k[2]:6d}", f(g(new, "pytorch")), f(t), "|", f(g(old, "cuda")), f(c), "|", f"{t/c:.2f}x" if c and t else "")
