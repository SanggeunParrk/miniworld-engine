# pfh16.py D,M ... : fp32 tg_fused_h16 built with -DPROBE=1 launched directly (inference); per-CTA stamps of tile 0, us from the earliest CTA start (median / max over CTAs)
import sys, statistics, torch
from miniworld_engine.kernels.transition.cuda import fused_generic_sm100a as fg
from miniworld_engine.kernels.transition.cuda.fused_sm100a import build_cubins
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
names = [(1, "kernel start"), (0, "setup done (TMEM, barriers)"), (2, "first weight box issued"), (3, "first Ws box issued"), (14, "LN0 warps: x staged"), (15, "LN0 stats done"), (16, "LN0 xn stored"),
         (11, "LN: x staged (tile 0)"), (7, "LN: stats done (tile 0)"), (8, "LN: xn in TMEM, xfull"), (4, "MMA sees xfull"), (20, "expand c0 start"), (21, "expand c1 start"), (52, "SwiGLU c0 abfull"), (53, "SwiGLU c1 abfull"),
         (36, "squeeze c0 start"), (5, "last squeeze issued"), (9, "epilogue sees yfull"), (84, "epilogue tile 0 end"), (10, "kernel end")]
for c in sys.argv[1:]:
    D, M = (int(v) for v in c.split(","))
    n = 2; H = n * D
    torch.manual_seed(0)
    x = torch.randn(M, D, device="cuda")
    k = fg._K(x, True)
    import os
    flags = ["-DPROBE=1", f"-DD={D}", *os.environ.get("PF_FLAGS", "").split()] + (["-DEP3=1"] if D == 128 and M >= fg.EP3_MIN_ROWS else [])
    p = build_cubins("probe_h16" + os.environ.get("PF_TAG", ""), ((f"fp{D}", "generic/tg_fused_h16.cu", tuple(flags)),))
    k.ext.load_cubin(f"probe_h16_fp{D}", p[f"fp{D}"])
    wa, wb = (torch.randn(H, D, device="cuda") * D ** -0.5), (torch.randn(H, D, device="cuda") * D ** -0.5)
    ws = torch.randn(D, H, device="cuda") * H ** -0.5
    g, b = torch.ones(D, device="cuda"), torch.zeros(D, device="cuda")
    wa16, wb16, ws16, sc = fg._h16_forms(g, b, wa, wb, ws)
    out = torch.empty_like(x)
    t = k.ext.tmap; v = lambda w: w.view(torch.bfloat16)
    grid = max(1, min(k.nsm, -(-M // 128)))
    st = torch.zeros(grid * 96, dtype=torch.int64, device="cuda")
    args = [t(x, D, M, 32, 128), t(v(wa16), D, H, 64, 64), t(v(wb16), D, H, 64, 64), t(v(ws16), H, D, 64, D), t(out, D, M, 32, 32), x, g, b, sc, 1e-5, M, H, out, st, out, 0]
    for _ in range(4):
        flush.zero_(); st.zero_(); k.ext.launch_kernel(f"probe_h16_fp{D}", "tg_fused_h16_sm100", grid, 448, 1, list(args), False); torch.cuda.synchronize()
    s = st.view(grid, 96).cpu().double()
    t0 = s[:, 1][s[:, 1] > 0].min()
    print(f"== fp32 D {D} n 2 M {M}: grid {grid} tiles {-(-M // 128)} ({-(-M // 128) / grid:.2f} a CTA)", flush=True)
    for idx, nm in names:
        col = [(v_ - t0) / 1000 for v_ in s[:, idx].tolist() if v_ > 0]
        if col: print(f"   {nm:30s} med {statistics.median(col):7.2f}  max {max(col):7.2f}  min {min(col):7.2f}")
