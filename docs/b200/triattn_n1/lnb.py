import sys, torch
from pathlib import Path
from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir as BB
from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
ensure_cuda_home()
src = Path(BB.__file__).parent / "b200_sources"
new = load_extension(name="n1_lnbwd_t4", sources=[str(src / "lnbwd_sm100.cu")], extra_cuda_cflags=[*host_flags(), *BB._FLAGS, *gencodes("100a")], verbose=False)
old = BB.glue(); bo = BB.bo()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)[n // 2]
C = 128
for L in [int(x) for x in sys.argv[1:]]:
    M = L * L; torch.manual_seed(0)
    x = torch.randn(M, C, device="cuda", dtype=torch.bfloat16); dy = torch.randn_like(x)
    w = torch.rand(C, device="cuda") + 0.5; mean = torch.randn(M, device="cuda") * .1; rstd = torch.rand(M, device="cuda") + 0.5
    for dt in (torch.bfloat16, torch.float32):
        dyln = torch.randn(M, C, device="cuda", dtype=dt)
        for tr in (False, True):
            oo = old.ln_bwd(dyln, x, dy, w, mean, rstd, L, tr); o = (oo[0], bo.bo_colsum(oo[1]))
            t0 = tm(lambda: bo.bo_colsum(old.ln_bwd(dyln, x, dy, w, mean, rstd, L, tr)[1]))
            res = f"L{L} {str(dt)[6:]} tr={int(tr)}: old {t0:.1f} (colsum-in 2112 blk)"
            for gm in (1, 2):
                n = new.ln_bwd128(dyln, x, dy, w, mean, rstd, L, tr, gm)
                eq = torch.equal(o[0], n[0]); n2 = new.ln_bwd128(dyln, x, dy, w, mean, rstd, L, tr, gm); det = torch.equal(n[1], n2[1]) and torch.equal(n[0], n2[0])
                ps = (o[1].double() - n[1].double()).abs().max() / o[1].double().abs().max()
                res += f" | gm{gm} {tm(lambda: new.ln_bwd128(dyln, x, dy, w, mean, rstd, L, tr, gm)):.1f} eq={int(eq)} det={int(det)} dpart {float(ps):.1e}"
            print(res, flush=True)
