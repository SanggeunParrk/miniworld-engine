"""SOL table of Transition INFERENCE (eval) from measure.speed records (backend cuda).  usage: python sol_inf.py PEAKS.json FLOOR_US 'glob' [...]
Formulas = docs/b200/sol_transition/sol_transition.py (inference): FLOPs 6 M D H, bytes 2 M D b + 12 D H + 8 D, MUFU M H; t_sol = max.
Extra: fp16 'kind::f16' SOL for float32 (fp16 tensor peak; numerically exact only where the fp16-operand path is accuracy-equal, see the doc),
and SOL% against t_sol + FLOOR (measured floor of one CUDA-graph replay with the same flush, `floor.py`: empty graph / 1 kernel)."""
import glob, json, statistics, sys

P = json.load(open(sys.argv[1])); FLOOR = float(sys.argv[2])
SUS = {"master": 1365e12, "float32": 1290e12}  # sustained cuBLAS square 8192 GEMM after 1 s of load (b200cfg/ceil.py): bf16 1365, fp16 1290 (fp32 path runs fp16 operands), tf32 716
mu = 16 * P["sms"] * P["clk_ghz"] * 1e9
tens = {"master": P["bf16"] * 1e12, "float32": P["tf32"] * 1e12}; fp16 = P["fp16"] * 1e12; bw = P["bw"] * 1e9
rec = {}
for f in sorted({g for a in sys.argv[3:] for g in glob.glob(a)}):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("backend") != "cuda" or r.get("status") != "ok" or r["dtype"] not in tens or r["mode"] != "eval": continue
        rec[(r["row"], r["dtype"], r["length"], r.get("depth"))] = r
out = []
for k, r in sorted(rec.items()):
    row, dt, L, depth = k; D = r["kwargs"]["d_hidden"]; n = r["kwargs"]["n"]; st = r["stream"]; S = r.get("samples", 1) or 1
    M = S * L * L if st == "token_pair" else (S * (depth or 1) * L if st == "msa_token" else S * L)
    H = n * D; b = 2 if dt == "master" else 4
    fl = 6 * M * D * H; by = 2 * M * D * b + 12 * D * H + 8 * D
    tt = fl / tens[dt] * 1e6; tm = by / bw * 1e6; tu = M * H / mu * 1e6
    ts = max(tt, tm, tu); bound = "tensor" if ts == tt else ("mem" if ts == tm else "mufu")
    t = r["median_ms"] * 1e3
    t16 = max(fl / fp16 * 1e6, tm, tu) if dt == "float32" else None
    out.append(dict(row=row, dt=dt, st=st, L=L, D=D, n=n, M=M, t=t, ts=ts, bound=bound, pct=100 * ts / t, pctf=100 * (ts + FLOOR) / t, pcts=100 * max(fl / SUS[dt] * 1e6, tm, tu) / t,
                    pct16=None if t16 is None else 100 * t16 / t, t16=t16))
print("row\tdtype\tstream\tL\tD\tn\tM\tt_us\tt_sol_us\tbound\tSOL%\tSOL%+floor\tt_sol_fp16\tSOL%fp16\tSOL%sustained")
for o in out:
    print("\t".join(str(x) for x in (o["row"], o["dt"], o["st"], o["L"], o["D"], o["n"], o["M"], f"{o['t']:.2f}", f"{o['ts']:.2f}", o["bound"], f"{o['pct']:.1f}",
                                  f"{o['pctf']:.1f}", "" if o["t16"] is None else f"{o['t16']:.2f}", "" if o["pct16"] is None else f"{o['pct16']:.1f}", f"{o['pcts']:.1f}")))
print("\n# summary (cases, SOL% min / median / max; count >= 90 / 70 / 50)")
for dt in ("master", "float32", None):
    for key, nm in (("pct", "SOL%"), ("pctf", "SOL%+floor"), ("pcts", "SOL%sustained-peak")):
        v = [o[key] for o in out if dt is None or o["dt"] == dt]
        if v: print(f"# {dt or 'all'} {nm}: n={len(v)} min {min(v):.1f} med {statistics.median(v):.1f} max {max(v):.1f}; >=90 {sum(x>=90 for x in v)} >=70 {sum(x>=70 for x in v)} >=50 {sum(x>=50 for x in v)}")
v = [o["pct16"] for o in out if o["pct16"] is not None]
print(f"# float32 vs fp16 kind::f16 peak: n={len(v)} min {min(v):.1f} med {statistics.median(v):.1f} max {max(v):.1f}; >=90 {sum(x>=90 for x in v)} >=70 {sum(x>=70 for x in v)} >=50 {sum(x>=50 for x in v)}")
print("\n# lowest 15 (SOL%)")
for o in sorted(out, key=lambda o: o["pct"])[:15]: print(f"# row {o['row']} {o['dt']} L{o['L']} D{o['D']} n{o['n']} t {o['t']:.2f} t_sol {o['ts']:.2f} {o['bound']} {o['pct']:.1f}% (+floor {o['pctf']:.1f}%)")
