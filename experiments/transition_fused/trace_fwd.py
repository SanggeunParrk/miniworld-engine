"""Timeline of the fused forward from the TRACE build: per warpgroup, the cycles each phase of a chunk and a tile take, and how
far apart the two warpgroups of one SM run.

  python trace_fwd.py --length 384 [--cubin build/transition_fwd_trace.cubin]
"""
import argparse, statistics, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402

D, H, ROWS, NCTA, NCH = 128, 512, 128, 132, 8
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--cubin", default=str(HERE / "build/transition_fwd_trace.cubin"))
p.add_argument("--ctas", type=int, default=4); a = p.parse_args()
M = a.length ** 2; tiles = M // ROWS; dev = "cuda"
torch.manual_seed(2319)
x = torch.randn(M, D, device=dev, dtype=torch.bfloat16)
gamma = (torch.rand(D, device=dev) + 0.5).contiguous(); beta = (torch.randn(D, device=dev) * 0.1).contiguous()
wa = (torch.randn(H, D, device=dev) * D ** -0.5).to(torch.bfloat16).contiguous()
wb = (torch.randn(H, D, device=dev) * D ** -0.5).to(torch.bfloat16).contiguous()
ws = (torch.randn(D, H, device=dev) * H ** -0.5).to(torch.bfloat16).contiguous(); wst = ws.t().contiguous()
k = drv.Kernel(a.cubin, "transition_fwd_fused", 231424)
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
out = torch.empty_like(x)
maps = (tm(x, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]),
        tm(wst, [D, H], D * 2, [64, 64]), tm(out, [D, M], D * 2, [64, 64]))
xn = torch.empty_like(x); rstd = torch.zeros(M, device=dev, dtype=torch.float32); c1 = torch.empty(M, device=dev, dtype=torch.float32)
for _ in range(5):
    k((NCTA, 1, 1), (256, 1, 1), *maps, gamma, beta, xn, out, rstd, c1, int(M), int(tiles), 1e-5)
torch.cuda.synchronize()
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
ev[0].record(); k((NCTA, 1, 1), (256, 1, 1), *maps, gamma, beta, xn, out, rstd, c1, int(M), int(tiles), 1e-5); ev[1].record()
torch.cuda.synchronize()
print(f"single launch (events): {ev[0].elapsed_time(ev[1]) * 1e3:.1f} us")
T = rstd.view(torch.int32).cpu().numpy().astype("int64") & 0xFFFFFFFF
per_cta_tiles = (tiles + NCTA - 1) // NCTA
def st(cta, wg, i, s): return int(T[(cta * 2 + wg) * 1024 + 64 * i + s])
cal = []
ph = {k_: [] for k_ in ("x_wait", "LN", "w_wait", "G1_wait", "SwiGLU", "G2_issue", "epilogue", "tile")}
skew = []
for cta in range(a.ctas):
    nt = min(len(range(cta, tiles, NCTA)), 15)
    for wg in (0, 1):
        for i in range(1, nt - 1):                            # steady-state tiles only
            ph["x_wait"].append(st(cta, wg, i, 1) - st(cta, wg, i, 0))
            ph["LN"].append(st(cta, wg, i, 2) - st(cta, wg, i, 1))
            for j in range(NCH):
                b = 3 + 4 * j
                ph["w_wait"].append(st(cta, wg, i, b + 1) - st(cta, wg, i, b) )     # includes the G1 issue itself
                ph["G1_wait"].append(st(cta, wg, i, b + 2) - st(cta, wg, i, b + 1))
                ph["SwiGLU"].append(st(cta, wg, i, b + 3) - st(cta, wg, i, b + 2))
                nxt = st(cta, wg, i, b + 4) if j + 1 < NCH else st(cta, wg, i, 35)
                ph["G2_issue"].append(nxt - st(cta, wg, i, b + 3))
            ph["epilogue"].append(st(cta, wg, i + 1, 0) - st(cta, wg, i, 35))
            ph["tile"].append(st(cta, wg, i + 1, 0) - st(cta, wg, i, 0))
        c0_, g0_, c1_, g1_ = (int(T[(cta * 2 + wg) * 1024 + 1000 + q]) for q in range(4))
        cal.append(((c1_ - c0_) & 0xFFFFFFFF, (g1_ - g0_) & 0xFFFFFFFF))
        for i in range(1, nt - 1):
            for j in range(NCH):
                skew.append(st(cta, 1, i, 3 + 4 * j + 1) - st(cta, 0, i, 3 + 4 * j + 1))
print(f"L{a.length}: {tiles} tiles, ~{tiles / NCTA:.2f} per CTA; cycles (MEAN over steady-state tiles of CTAs 0..{a.ctas - 1}, both warpgroups)")
tile = statistics.mean(ph["tile"])
for k_, v in ph.items():
    m = statistics.mean(v); n = NCH if k_ in ("w_wait", "G1_wait", "SwiGLU", "G2_issue") else 1
    print(f"  {k_:>9s}: {m:7.0f} cyc  x{n} = {m * n:7.0f}  ({100 * m * n / tile:5.1f} % of a tile)")
print(f"  G1-issue skew wg1 - wg0: median {statistics.median(skew):.0f} cyc, |skew| p90 {sorted(abs(s) for s in skew)[int(.9 * len(skew))]:.0f}")
cyc, ns = (sum(c for c, _ in cal), sum(g for _, g in cal))
f = cyc / ns
print(f"  %clock rate from %globaltimer: {f:.3f} GHz (loop span {statistics.mean(g for _, g in cal) / 1e3:.1f} us per warpgroup)")
print(f"  tile = {tile:.0f} %clock = {tile / f / 1e3:.2f} us; x {tiles / NCTA:.2f} tiles = {tile / f / 1e3 * tiles / NCTA:.1f} us")
print(f"  tensor floor of a tile: 6*128*D*H / 4096 FLOP/SM-clk = {6*128*D*H/4096:.0f} SM cycles -> {6*128*D*H/4096/1.755e3:.2f} us at 1.755 GHz")

A = (rstd.view(torch.int32).cpu().numpy().astype("int64") & 0xFFFFFFFF)[M - 8 * 264:].reshape(264, 8)
entry, lstart, lend, smid = A[:, 0], A[:, 2], A[:, 4], A[:, 1]
t0 = entry.min()
print("all CTAs (both warpgroups), us after the first CTA's entry:")
for name, v in (("entry", entry), ("loop start", lstart), ("loop end", lend)):
    v = (v - t0) / 1e3
    print(f"  {name:>10s}: min {v.min():7.1f}  median {sorted(v)[len(v)//2]:7.1f}  max {v.max():7.1f}")
span = (lend - lstart) / 1e3
print(f"  loop span : min {span.min():7.1f}  median {sorted(span)[len(span)//2]:7.1f}  max {span.max():7.1f}")
late = sorted(range(264), key=lambda q: -(entry[q] - t0))[:6]
print("  latest entries: " + ", ".join(f"cta {q // 2} wg {q % 2} sm {smid[q]} +{(entry[q] - t0) / 1e3:.1f}" for q in late))
print("per-tile timeline, CTA 0 (both warpgroups), us from loop start (%clock / measured rate):")
for wg in (0, 1):
    base = (cta0 := 0)
    c_start = int(T[(0 * 2 + wg) * 1024 + 1000])
    nt = len(range(0, tiles, NCTA))
    row = []
    for i in range(min(nt, 15)):
        top, ln_done, epi = st(0, wg, i, 0), st(0, wg, i, 2), st(0, wg, i, 35)
        row.append(f"t{i}: top {((top - c_start) & 0xFFFFFFFF) / f / 1e3:6.1f} LN {((ln_done - top) & 0xFFFFFFFF) / f / 1e3:5.2f} "
                   f"chunks {((epi - ln_done) & 0xFFFFFFFF) / f / 1e3:5.2f}")
    print(f"  wg {wg}:"); [print("    " + r) for r in row]
    c_end = int(T[(0 * 2 + wg) * 1024 + 1002])
    print(f"    loop end {((c_end - c_start) & 0xFFFFFFFF) / f / 1e3:6.1f}")
