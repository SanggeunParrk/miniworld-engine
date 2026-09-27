"""cuBLAS contraction forms for the TriMul planes: per 128-channel half, NT (a b^T), TN (a^T b), NN, and one 256-channel TN call."""
import statistics, sys, torch
def t(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    r = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n): g.replay()
        b.record(); b.synchronize(); r.append(a.elapsed_time(b) / n * 1000)
    return statistics.median(r)
for L in (384, 768):
    a = torch.randn(256, L, L, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a); x = torch.empty_like(a)
    h = 128
    nt = t(lambda: torch.bmm(a[:h], b[:h].transpose(1, 2), out=x[:h]))
    tn = t(lambda: torch.bmm(a[h:].transpose(1, 2), b[h:], out=x[h:]))
    nn = t(lambda: torch.bmm(a[:h], b[:h], out=x[:h]))
    both = t(lambda: (torch.bmm(a[:h], b[:h].transpose(1, 2), out=x[:h]), torch.bmm(a[h:].transpose(1, 2), b[h:], out=x[h:])))
    tn256 = t(lambda: torch.bmm(a.transpose(1, 2), b, out=x))
    print(f"L{L}: half NT {nt:.1f}  half TN {tn:.1f}  half NN {nn:.1f} | current NT+TN {both:.1f}  single TN(256) {tn256:.1f} us")
