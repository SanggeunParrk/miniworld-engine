import sys, torch
from pathlib import Path
from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir as BB
from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
ensure_cuda_home()
src = Path(BB.__file__).parent / "b200_sources"
new = load_extension(name="n1_tl_t", sources=[str(src / "lnbwd_sm100.cu")], extra_cuda_cflags=[*host_flags(), *BB._FLAGS, *gencodes("100a")], verbose=False)
old = BB.ext()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)[n // 2]
cfgs = {256: [(16, 4, 2), (16, 5, 2), (16, 3, 2), (32, 2, 2), (32, 3, 2)], 128: [(16, 4, 2), (16, 5, 2), (16, 6, 2), (16, 8, 2), (32, 3, 2), (32, 4, 2), (32, 5, 2)]}
for L in [int(x) for x in sys.argv[1:]]:
    LL = L * L; torch.manual_seed(0)
    for K in (256, 128):
        g = torch.randn(K, LL, device="cuda", dtype=torch.bfloat16); o = torch.randn(K, LL, device="cuda", dtype=torch.bfloat16)
        wo = torch.randn(128, K, device="cuda", dtype=torch.bfloat16) * .1; x = torch.randn(LL, 128, device="cuda", dtype=torch.bfloat16)
        for su in (False, True):
            u0 = torch.empty(K, LL, device="cuda", dtype=torch.bfloat16) if su else None; u1 = torch.empty_like(u0) if su else None
            out0 = torch.empty_like(x); out1 = torch.empty_like(x)
            old.tail(g, o, wo, x, out0, u0); t0 = tm(lambda: old.tail(g, o, wo, x, out0, u0))
            res = f"L{L} K{K} {'train' if su else 'eval'}: shared {t0:.1f}"
            for (kch, ns, gm) in cfgs[K]:
                try:
                    new.tail_b200(g, o, wo, x, out1, u1, kch, ns, gm)
                    eq = torch.equal(out0, out1) and (not su or torch.equal(u0, u1))
                    res += f" | kch{kch} ns{ns} {tm(lambda: new.tail_b200(g, o, wo, x, out1, u1, kch, ns, gm)):.1f} eq={int(eq)}"
                except Exception as e: res += f" | kch{kch} ns{ns} ERR {str(e)[:60]}"
            print(res, flush=True)
