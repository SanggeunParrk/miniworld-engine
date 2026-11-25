# h100ratio.py TAG : Transition eval cuda, B200 measure.speed records (out/m_TAG_eval_*.jsonl) vs the H100 cuda tokens; the 1.5x rule with the 20 us exemption
import collections, glob, json, math, sys
tag = sys.argv[1]
h = {}
for line in open("src/foldjet_engine/judge/tokens/h100_sm90/Transition.jsonl"):
    t = json.loads(line); c = t["case"]; r = t.get("result", {})
    if t["kind"] != "speed" or c["backend"] != "cuda" or r.get("status") != "ok" or c["mode"] != "eval": continue
    h[(c["row"], c["dtype"], c["length"], c.get("depth"))] = r["median_ms"]
rows = []
for f in sorted(glob.glob(f"out/m_{tag}_eval_*.jsonl")):
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("backend") != "cuda" or r.get("status") != "ok" or r["mode"] != "eval": continue
        k = (r["row"], r["dtype"], r["length"], r.get("depth"))
        if k in h: rows.append((k, r["median_ms"], h[k], r["kwargs"]["d_hidden"], r["kwargs"]["n"]))
T = 1 / 1.5
print("key\tD\tn\tb200_ms\th100_ms\tratio\tclass")
elig = [x for x in rows if x[2] >= 0.020]
for k, b, hh, D, n in sorted(rows):
    cl = "exempt" if hh < 0.020 else ("slower" if b / hh > 1 else ("miss" if b / hh > T else "ok"))
    print(f"{k}\t{D}\t{n}\t{b:.4f}\t{hh:.4f}\t{b/hh:.3f}\t{cl}")
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else 0
print(f"SUMMARY compared {len(rows)} exempt {len(rows)-len(elig)} eligible {len(elig)} miss(>0.667) {sum(b/hh>T for _,b,hh,_,_ in elig)} slower {sum(b/hh>1 for _,b,hh,_,_ in elig)} worst {max((b/hh for _,b,hh,_,_ in elig),default=0):.3f} geomean {g([b/hh for _,b,hh,_,_ in elig]):.3f}")
