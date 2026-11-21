# solrun.py TAG [TAG2]: SOL% of the cuda training records _s90/spd_TAG_L*.jsonl (sol.py of the SOL study); TAG2: before -> after
import glob, json, subprocess, sys, statistics as st
SOL = "../sol-triattn/sol.py"; PK = "../sol-triattn/peaks.json"
def run(tag):
    tok = f"_s90/tok_{tag}.jsonl"; n = 0
    with open(tok, "w") as f:
        for fn in sorted(glob.glob(f"_s90/spd_{tag}_L*.jsonl")):
            for l in open(fn):
                try: r = json.loads(l)
                except Exception: continue
                if r.get("status") != "ok" or r.get("backend") != "cuda": continue
                c = {k: r[k] for k in ("module", "row", "kwargs", "length", "mode", "dtype")}; c["backend"] = "cuda"
                f.write(json.dumps({"kind": "speed", "case": c, "result": {"status": "ok", "median_ms": r["median_ms"]}}) + "\n"); n += 1
    subprocess.run([sys.executable, SOL, PK, tok, f"_s90/sol_{tag}.tsv", f"_s90/sol_{tag}.md"], check=True, stdout=subprocess.DEVNULL)
    rows = {}
    for i, l in enumerate(open(f"_s90/sol_{tag}.tsv")):
        if i == 0: h = l.split(); continue
        d = dict(zip(h, l.rstrip("\n").split("\t")))
        rows[(int(d["row"]), d["starting"], int(d["L"]))] = d
    return rows
a = run(sys.argv[1]); b = run(sys.argv[2]) if len(sys.argv) > 2 else None
def cnt(R):
    s = [float(d["sol"]) * 100 for d in R.values()]
    return f"n {len(s)} min/med/max {min(s):.1f}/{st.median(s):.1f}/{max(s):.1f}  >=90 {sum(x>=90 for x in s)} >=70 {sum(x>=70 for x in s)} >=50 {sum(x>=50 for x in s)}"
print("A", sys.argv[1], cnt(a))
if b: print("B", sys.argv[2], cnt(b))
for grp, sel in (("row 72 (D64 H4 dh16)", lambda k: k[0] == 72), ("row 241", lambda k: k[0] == 241), ("row 242", lambda k: k[0] == 242), ("row 243 (D128)", lambda k: k[0] == 243),
                 ("row 244 (D256)", lambda k: k[0] == 244), ("row 245 (D384)", lambda k: k[0] == 245), ("row 188 bias-only", lambda k: k[0] == 188), ("dh32 self-att (241-245)", lambda k: k[0] >= 241)):
    for nm, R in (("A", a), ("B", b)):
        if R is None: continue
        s = [float(d["sol"]) * 100 for k, d in R.items() if sel(k)]
        print(f"  {grp:26s} {nm} min/med/max {min(s):5.1f}/{st.median(s):5.1f}/{max(s):5.1f}")
print("per case (row start L: t_meas us, t_sol us, bind, strategy, SOL%)" + (" -> B t_meas, SOL%" if b else ""))
for k in sorted(a):
    d = a[k]; x = f"  r{k[0]} {'S' if d['starting']=='True' else 'E'} L{k[2]:4d}: {float(d['t_meas'])*1e6:8.1f} us  sol {float(d['t_sol'])*1e6:7.1f}  {d['bind']:6s} {d['strategy'][:6]:6s} {float(d['sol'])*100:5.1f}%"
    if b and k in b: x += f"  ->  {float(b[k]['t_meas'])*1e6:8.1f} us {float(b[k]['sol'])*100:5.1f}%"
    print(x)
