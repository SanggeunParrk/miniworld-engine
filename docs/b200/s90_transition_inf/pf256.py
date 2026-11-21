# pf256.py n,M : tfwd_d256 inference one-tile build (fwdi1 flags) + -DPROBE=1, launched as the host does; stamps of tile 0 (us from the earliest CTA start, median / max over leader CTAs)
import sys, statistics, torch
from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from miniworld_engine.kernels.transition.cuda.fused_sm100a import build_cubins
D = 256
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
names = [(0, "kernel start"), (1, "setup done"), (4, "MMA expand c0 starts"), (5, "MMA expand c1 starts"), (52, "squeeze c0 waits h_full ok"), (100, "LN: x landed"), (106, "LN done"), (112, "epi sees out_full"), (118, "epi released out_empty"), (124, "epi end")]
for c in sys.argv[1:]:
    n, M = (int(v) for v in c.split(",")); H = n * D
    torch.manual_seed(0)
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    k = fw._Launch(x, H)
    fl = [f for f in fw.FWD_INFER_FLAGS[(D, n)] if f != "-DX2=1"] if k.tiles <= k.grid else list(fw.FWD_INFER_FLAGS[(D, n)])
    p = build_cubins(f"d{D}n{n}", (("fwdp", "widths/tfwd_d256.cu", (f"-DHID={H}", "-DPROBE=1", *fl)),))
    k.ext.load_cubin(f"d{D}n{n}_fwdp", p["fwdp"])
    wa, wb = (torch.randn(H, D, device="cuda") * D ** -0.5).bfloat16(), (torch.randn(H, D, device="cuda") * D ** -0.5).bfloat16()
    ws = (torch.randn(D, H, device="cuda") * H ** -0.5).bfloat16()
    g, b = torch.ones(D, device="cuda"), torch.zeros(D, device="cuda")
    out = torch.empty_like(x); mout = k.rows(out)
    rstd = torch.empty(1, device="cuda")
    st = torch.zeros(k.grid * 136, dtype=torch.int64, device="cuda")
    args = (k.rows(x), k.tmap(wa, D, H), k.tmap(wb, D, H), k.tmap(ws, H, D, 64, D // 2), mout, mout, g, b, rstd, st, k.tiles, 1e-5, 0, x, out)
    for _ in range(4):
        flush.zero_(); st.zero_(); k.run("fwdp", "transition_fwd_d256_sm100", k.grid, args, block=512); torch.cuda.synchronize()
    s = st.view(k.grid, 136).cpu().double()
    t0 = s[:, 0][s[:, 0] > 0].min()
    print(f"== bf16 D {D} n {n} M {M}: grid {k.grid} tiles {k.tiles} ({k.tiles / k.grid * 2:.2f} a pair-CTA) flags {fl}", flush=True)
    for idx, nm in names:
        col = [(v - t0) / 1000 for v in s[:, idx].tolist() if v > 0]
        if col: print(f"   {nm:30s} med {statistics.median(col):7.2f}  max {max(col):7.2f}  min {min(col):7.2f}")
