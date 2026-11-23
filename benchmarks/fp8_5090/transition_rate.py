"""5090 default Transition (D128 n4 inference): time and achieved TFLOPS vs the mma peaks (bf16 218, e4m3 459 measured in p0_rates_gemm). python transition_rate.py rows ..."""
import sys, torch
import miniworld_engine
from miniworld_engine.modules import Transition
flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
def bench(f, n=3):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): flush.zero_(); f()
    gf = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n): flush.zero_()
    def t(gr):
        gr.replay(); torch.cuda.synchronize(); ts = []
        for _ in range(7):
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        return sorted(ts)[3]
    return (t(g) - t(gf)) / n * 1e3
for d in (128, 256):
    for L in map(int, sys.argv[1:]):
        m = Transition(d, n=4, master_weights=True, backend="cuda").cuda().eval()
        x = torch.randn(1, L, L, d, device="cuda", dtype=torch.bfloat16)
        def f():
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return m(x)
        us = bench(f); T = L * L
        fl = T * (2 * d * 8 * d + 2 * 4 * d * d)
        print(f"D{d} L{L} rows {T}: {us:.0f} us  {fl/us/1e6:.0f} TFLOPS (bf16 peak 218, e4m3 459)  bytes/row-floor {T*d*4/1e6/us*1e3:.0f} GB/s", flush=True)
