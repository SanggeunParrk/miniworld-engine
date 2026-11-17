"""End-level table. usage: an3.py MODE[,MODE] DIR [DIR...]   (all dirs are merged: bf16_t* and <mode>_t* files {target}.{case}.pt; json beside them)
per target: bf16 trunk seeds n, pair RMSD max / mean, ipTM range | per mode: n same-seed RMSD max / mean, max |d ipTM|, inside = max same-seed RMSD <= max(1.1 x max bf16 pair RMSD, 0.5 A);
flip fraction (same-seed RMSD > 1.1 x max bf16 pair RMSD) and overflow / peak statistics from the jsons."""
import sys, glob, json, itertools, os, statistics as st, torch
modes = sys.argv[1].split(","); dirs = sys.argv[2:]
def kabsch(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    u, s, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.linalg.det(u @ vt)); D = torch.diag(torch.tensor([1, 1, d]))
    return ((a @ (u @ D @ vt) - b) ** 2).sum(-1).mean().sqrt().item()
files = {}
for d in dirs:
    for p in glob.glob(f"{d}/*.pt"):
        t, c = os.path.basename(p)[:-3].split("."); files.setdefault(t, {})[c] = p
def ld(p):
    d = torch.load(p); return d["coords"].reshape(-1, 3), (None if d["iptm"] is None else float(d["iptm"].float().mean()))
def info(p):
    j = p[:-3] + ".json"
    return json.load(open(j)) if os.path.exists(j) else {}
tot = {m: [0, 0] for m in modes}; ovf = {m: 0 for m in modes}; redo = {m: 0 for m in modes}; ncase = {m: 0 for m in modes}; peaks = {m: {} for m in modes}; lsc = {m: [99, -99] for m in modes}
print("target | bf16 seeds n, pair RMSD max/mean (A), ipTM range | per mode: n, same-seed RMSD max/mean, per-seed list, max|d ipTM|, inside")
for t in sorted(files):
    C = files[t]; bf = sorted(c for c in C if c.startswith("bf16_t"))
    if len(bf) < 2: continue
    D = {n: ld(C[n]) for n in bf}
    sp = [kabsch(D[a][0], D[b][0]) for a, b in itertools.combinations(bf, 2)]
    ip = [D[n][1] for n in bf if D[n][1] is not None]
    thr = max(1.1 * max(sp), 0.5)
    row = f"{t[:4]} | bf16 n={len(bf)} {max(sp):.2f}/{st.mean(sp):.2f} ipTM [{min(ip):.3f},{max(ip):.3f}]" if ip else f"{t[:4]} | bf16 n={len(bf)} {max(sp):.2f}/{st.mean(sp):.2f}"
    for m in modes:
        fs = sorted(c for c in C if c.startswith(m + "_t") and "bf16_t" + c.split("_t")[-1] in D)
        if not fs: row += f" | {m}: -"; continue
        same = []; dip = []
        for n in fs:
            b = "bf16_t" + n.split("_t")[-1]; c, i = ld(C[n]); same.append(kabsch(c, D[b][0]))
            if i is not None and D[b][1] is not None: dip.append(i - D[b][1])
            j = info(C[n]); ncase[m] += 1; ovf[m] += j.get("overflow_events", 0); redo[m] += j.get("redo", 0)
            if j.get("log2scale_min") is not None: lsc[m] = [min(lsc[m][0], j["log2scale_min"]), max(lsc[m][1], j["log2scale_max"])]
            for k, v in j.get("peak", {}).items():
                a = peaks[m].setdefault(k, [0, 0, 0.0, 0.0, 0]); a[0] += v["calls"]; a[1] += v["n_eq1"]; a[4] += v["n_nan"]
                if v["peak_frac_max"] is not None: a[2] = max(a[2], v["peak_frac_max"]); a[3] = max(a[3], v["peak_frac_p99"])
        ok = max(same) <= thr; tot[m][0] += ok; tot[m][1] += 1
        row += f" | {m}: n={len(fs)} {max(same):.2f}/{st.mean(same):.2f} [{' '.join(f'{x:.2f}' for x in same)}] dip {max(abs(x) for x in dip) if dip else float('nan'):.3f} {'in' if ok else 'OUT'}"
    print(row, flush=True)
print("inside / targets:", {m: f"{a}/{b}" for m, (a, b) in tot.items()})
for m in modes:
    print(f"overflow [{m}]: cases {ncase[m]}, cases with a saturation flag {ovf[m]}, redo {redo[m]}, log2(scale) range over sites {lsc[m]}")
    for k, a in sorted(peaks[m].items()): print(f"   peak [{m}] {k}: calls {a[0]}, calibration/sat (frac=1) {a[1]}, max peak frac of range {a[2]:.3f}, max p99 {a[3]:.3f}, nan {a[4]}")
