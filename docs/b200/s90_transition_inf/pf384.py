# pf384.py n,M: tfwd_d384 built with -DPROBE=1 launched directly (inference); per-CTA stamps (median over CTAs, us from the earliest tile-0 start)
import sys, statistics, torch
from foldjet_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from foldjet_engine.kernels.transition.cuda.fused_sm100a import build_cubins
D = 384
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
names = {8: "MMA expand starts tile", 14: "last expand issued tile", 20: "LN x landed tile", 26: "LN done tile", 32: "SW last chunk done tile", 38: "epi out_full tile", 44: "epi end tile"}
for c in sys.argv[1:2]:
    n, M = (int(v) for v in c.split(",")); H = n * D
    torch.manual_seed(0)
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    k = fw._Launch(x, H)
    p = build_cubins(f"d{D}n{n}", (("fwdp", "widths/tfwd_d384.cu", (f"-DHID={H}", "-DPROBE=1", *[f"-D{a}" for a in sys.argv[2:]])),))
    k.ext.load_cubin(f"d{D}n{n}_fwdp", p["fwdp"])
    wa, wb = (torch.randn(H, D, device="cuda") * D ** -0.5).bfloat16(), (torch.randn(H, D, device="cuda") * D ** -0.5).bfloat16()
    ws = (torch.randn(D, H, device="cuda") * H ** -0.5).bfloat16()
    g, b = torch.ones(D, device="cuda"), torch.zeros(D, device="cuda")
    out = torch.empty_like(x); mout = k.rows(out)
    ph = torch.empty(1, 1, device="cuda", dtype=torch.bfloat16); rstd = torch.empty(1, device="cuda")
    st = torch.zeros(k.grid * 64, dtype=torch.int64, device="cuda")
    args = (k.rows(x), k.tmap(wa, D, H, 64, 32), k.tmap(wb, D, H, 64, 32), k.tmap(ws, H, D), mout, mout, g, b, rstd, st, ph, ph, ph, k.tiles, 1e-5, 0)
    for _ in range(4):
        flush.zero_(); st.zero_(); k.run("fwdp", "transition_fwd_d384_sm100", k.grid, args); torch.cuda.synchronize()
    t = st.view(k.grid, 64).cpu().double()
    t0 = t[:, 8][t[:, 8] > 0].min()
    print(f"== n {n} M {M}: grid {k.grid} tiles {k.tiles}")
    for base, nm in names.items():
        row = []
        for i in range(5):
            col = [(v - t0) / 1000 for v in t[:, base + i].tolist() if v > 0]
            row.append(f"{statistics.median(col):7.2f}" if col else "   --  ")
        print(f"   {nm:28s}", " ".join(row))
