import sys, torch
from pathlib import Path
from miniworld_engine.kernels.triangle_attention.cuda import b200_bidir as BB
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
ensure_cuda_home()
src = Path(BB.__file__).parent / "b200_sources"
new = load_extension(name="n1_lnf_t", sources=[str(src / "lnbwd_sm100.cu")], extra_cuda_cflags=[*host_flags(), *BB._FLAGS, *gencodes("100a")], verbose=False)
old = BB.glue()
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
    x = torch.randn(M, C, device="cuda", dtype=torch.bfloat16) * 3 + 1
    w = torch.rand(C, device="cuda") + 0.5; b = torch.randn(C, device="cuda")
    o = old.ln_fwd(x, w, b, 1e-5, L, False, False); t0 = tm(lambda: old.ln_fwd(x, w, b, 1e-5, L, False, False))
    res = f"L{L}: old {t0:.1f}"
    for gm in (2, 4, 8):
        n = new.ln_fwd128(x, w, b, 1e-5, gm); eq = all(torch.equal(a, c) for a, c in zip(o, n))
        res += f" | gm{gm} {tm(lambda: new.ln_fwd128(x, w, b, 1e-5, gm)):.1f} eq={int(eq)}"
    print(res, flush=True)
