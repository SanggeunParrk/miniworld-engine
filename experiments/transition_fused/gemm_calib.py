"""What cuBLAS reaches on the Transition's two GEMM shapes, per width: the practical ceiling for a custom GEMM here.
GEMM1 = [M, D] x [D, 8D] (the packed a|b expand), GEMM2 = [M, 4D] x [4D, D] (+ x via addmm, the squeeze + residual)."""
import statistics, torch
PEAK = 989e12
def t(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
        for _ in range(reps): fn()
        b.record(); torch.cuda.synchronize(); o.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(o)
for L in (384, 768):
    M = L * L
    for D in (64, 128, 256, 384, 512):
        x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(8 * D, D, device="cuda", dtype=torch.bfloat16)
        h = torch.randn(M, 4 * D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(D, 4 * D, device="cuda", dtype=torch.bfloat16)
        u1 = t(lambda: x @ w1.t()); f1 = 2 * M * D * 8 * D
        u2 = t(lambda: torch.addmm(x, h, w2.t())); f2 = 2 * M * 4 * D * D
        print(f"L{L} D{D:4d}: GEMM1 {u1:8.1f} us {100 * f1 / PEAK / (u1 * 1e-6):5.1f} % | GEMM2+res {u2:8.1f} us {100 * f2 / PEAK / (u2 * 1e-6):5.1f} % "
              f"| sum {u1 + u2:8.1f} us vs fwd floor {24 * M * D * D / PEAK * 1e6:7.1f}", flush=True)
        del x, w1, h, w2; torch.cuda.empty_cache()
