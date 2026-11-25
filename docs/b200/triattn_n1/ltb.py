import sys, torch
from pathlib import Path
from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir as BB
from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
ensure_cuda_home()
e = load_extension(name="n1_ltmm_t", sources=[sys.argv[1]], extra_cuda_cflags=[*host_flags(), *BB._FLAGS, *gencodes("100a")], verbose=False)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)[n // 2]
H, dh = 4, 32
for L in [int(x) for x in sys.argv[2:]]:
    LL = L * L; torch.manual_seed(0)
    dos = torch.randn(H, dh * L, L, device="cuda", dtype=torch.bfloat16); c = torch.randn(H, dh * L, L, device="cuda", dtype=torch.bfloat16)
    dt = torch.randn(LL, 128, device="cuda", dtype=torch.bfloat16); u = torch.randn(128, LL, device="cuda", dtype=torch.bfloat16)
    dc = torch.randn(260, LL, device="cuda", dtype=torch.bfloat16); y = torch.randn(LL, 128, device="cuda", dtype=torch.bfloat16)
    cases = {
        "dP": (lambda: torch.bmm(dos.transpose(1, 2), c, out_dtype=torch.float32), lambda: e.lt_mm(dos.transpose(1, 2), c, dos)),
        "dWo": (lambda: torch.mm(dt.t(), u.t(), out_dtype=torch.float32), lambda: e.lt_mm(dt.t(), u.t(), dt)),
        "dW": (lambda: torch.mm(dc, y, out_dtype=torch.float32), lambda: e.lt_mm(dc, y, dc)),
    }
    for nm, (ref, new) in cases.items():
        r = ref(); n1 = new(); n2 = new(); torch.cuda.synchronize()
        err = float((r.double() - n1.double()).abs().max() / r.double().abs().max()); det = torch.equal(n1, n2)
        print(f"L{L} {nm}: torch {tm(ref):.1f}  tuned {tm(new):.1f}  relerr {err:.1e} det {det}", flush=True)
