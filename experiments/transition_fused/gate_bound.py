"""Practical bound for the gate stage: cuBLAS on the same GEMMs writing the same bytes (M x 3H bf16), no gate math."""
import statistics, torch
def t(fn, reps=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
bf = torch.bfloat16
for L in (384,):
    M = L * L
    for D in (256, 384, 512):
        H = 4 * D
        xn = torch.randn(M, D, device="cuda", dtype=bf); dy = torch.randn(M, D, device="cuda", dtype=bf)
        w2 = torch.randn(2 * H, D, device="cuda", dtype=bf); ws = torch.randn(D, H, device="cuda", dtype=bf)
        w3 = torch.randn(3 * H, D, device="cuda", dtype=bf)
        o2 = torch.empty(M, 2 * H, device="cuda", dtype=bf); o1 = torch.empty(M, H, device="cuda", dtype=bf); o3 = torch.empty(M, 3 * H, device="cuda", dtype=bf)
        a = t(lambda: torch.mm(xn, w2.t(), out=o2)); b = t(lambda: torch.mm(dy, ws, out=o1)); c = t(lambda: torch.mm(xn, w3.t(), out=o3))
        fl = 2 * M * D * 3 * H
        print(f"D{D} L{L}: cuBLAS ab {a:.1f} + dh {b:.1f} = {a+b:.1f} us ({100*fl/989e12/((a+b)*1e-6):.1f} %) | one GEMM N=3H {c:.1f} us ({100*fl/989e12/(c*1e-6):.1f} %) | write-only floor {3*M*H*2/3.35e12*1e6:.0f} us")
