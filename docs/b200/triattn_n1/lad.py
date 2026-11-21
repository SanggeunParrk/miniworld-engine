import json, glob, sys, collections, math
R, T, out = sys.argv[1], sys.argv[2], sys.argv[3]
def key(c): return json.dumps({"module": c["module"], "dtype": c["dtype"], "mode": c["mode"], "length": c.get("length"), "kwargs": c["kwargs"], "samples": c.get("samples"), "depth": c.get("depth")}, sort_keys=True)
new = {}; newpt = {}
for f in glob.glob(R + "/cuda_*.jsonl"):
    for l in open(f):
        t = json.loads(l)
        if t.get("status") != "ok" or t.get("timing") != "graph": continue
        k = key(t)
        (new if t["backend"] == "cuda" else newpt if t["backend"] == "pytorch" else {})[k] = t["median_ms"]
tok = {"b200_sm100": collections.defaultdict(dict), "h100_sm90": collections.defaultdict(dict)}
for card in tok:
    for m in ("TriangleAttention", "BidirectionalTriangleAttention"):
        for l in open(f"{T}/{card}/{m}.jsonl"):
            t = json.loads(l)
            c = t["case"]; r = t.get("result", {})
            if t["kind"] != "speed" or r.get("status") != "ok" or (r.get("timed_as") or "graph") != "graph": continue
            tok[card][key(c)][c["backend"]] = r["median_ms"]
lad, miss, n = [], [], 0
per = collections.defaultdict(list)
for k, cu in new.items():
    c = json.loads(k); b = tok["b200_sm100"].get(k, {}); h = tok["h100_sm90"].get(k, {}).get("cuda")
    pt = newpt.get(k, b.get("pytorch")); tr = b.get("triton")
    n += 1
    small = (c["length"] or 0) <= 256
    for nm, slow in (("triton", tr), ("pytorch", pt)):
        if slow is not None and cu > slow * 1.02: lad.append((round(cu / slow, 3), nm, "smallL" if small else "REAL", c["module"], c["mode"], c["length"], c["kwargs"]))
    if h is not None:
        r = cu / h; per[c["module"], c["mode"]].append(r)
        if h >= 0.020 and r > 1 / 1.5: miss.append((round(r, 3), c["module"], c["mode"], c["length"], c["kwargs"], round(cu, 4), round(h, 4)))
with open(out, "w") as fh:
    fh.write(f"cases measured {n}\nLADDER violations (cuda slower than rung by >2%): {len(lad)}\n")
    for x in sorted(lad, reverse=True): fh.write(" ".join(map(str, x)) + "\n")
    fh.write(f"\nH100 1.5x misses (H100 >= 20 us, ratio > 0.667): {len(miss)}\n")
    for x in sorted(miss, reverse=True): fh.write(" ".join(map(str, x)) + "\n")
    for k, v in per.items(): fh.write(f"{k} cases {len(v)} geomean {math.exp(sum(map(math.log, v))/len(v)):.3f} worst {max(v):.3f}\n")
print(open(out).read()[:6000])
