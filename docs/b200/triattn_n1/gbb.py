import sys, torch
from pathlib import Path
from miniworld_engine.kernels.triangle_attention.cuda import b200_bidir as BB
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
ensure_cuda_home()
src = Path(BB.__file__).parent / "b200_sources"
new = load_extension(name="n1_gb_t", sources=[str(src / "lnbwd_sm100.cu")], extra_cuda_cflags=[*host_flags(), *BB._FLAGS, *gencodes("100a")], verbose=False)
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
for L in [int(x) for x in sys.argv[1:]]:
    LL = L * L; torch.manual_seed(0)
    dy = torch.randn(LL, 128, device="cuda", dtype=torch.bfloat16); wo = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * .1
    big = torch.randn(260, LL, device="cuda", dtype=torch.bfloat16)           # planes as slices of one [260, LL] tensor (the module layout)
    gp = big[128:256]; o = torch.randn(128, LL, device="cuda", dtype=torch.bfloat16)
    dg0 = torch.empty(260, LL, device="cuda", dtype=torch.bfloat16); dg1 = torch.empty(260, LL, device="cuda", dtype=torch.bfloat16)
    r0 = old.gate_bwd_fused(dy, wo, gp, o, dg0[128:256]); t0 = tm(lambda: old.gate_bwd_fused(dy, wo, gp, o, dg0[128:256]))
    res = f"L{L}: old {t0:.1f}"
    for gm in (2, 3):
        try:
            r1 = new.gate_bwd128(dy, wo, gp, o, dg1[128:256], gm)
            eq = torch.equal(r0, r1) and torch.equal(dg0[128:256], dg1[128:256])
            res += f" | gm{gm} {tm(lambda: new.gate_bwd128(dy, wo, gp, o, dg1[128:256], gm)):.1f} eq={int(eq)}"
        except Exception as e: res += f" | gm{gm} ERR {str(e)[:80]}"
    print(res, flush=True)
