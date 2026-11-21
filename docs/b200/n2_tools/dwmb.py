# dwmb.py M : swa_embed_dwf / colsum / featbwd alone (graph, CUPTI kernel us, L2 flushed): full grid, dense-only grid (40 CTAs), hot groups only
import statistics, sys, torch
from torch.profiler import profile, ProfilerActivity
from miniworld_engine.kernels.swa_dit.cuda import sm100
M = int(sys.argv[1]); dev = torch.device("cuda"); torch.manual_seed(0)
K = sm100.embed_glue(0)
dp = torch.randn(M, 128, device=dev); pos = torch.randn(1, M, 3, device=dev)
chg = torch.randint(-1, 2, (1, M), device=dev); el = torch.randint(1, 20, (1, M), device=dev); mask = torch.ones(1, M, device=dev, dtype=torch.uint8)
dw = torch.empty(128, 133, device=dev); gb = torch.randn(2, M, 128, device=dev); out = torch.empty(2, 128, device=dev)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
def bench(fn, name, key):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(20):
        flush.zero_(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr: g.replay(); torch.cuda.synchronize()
        ts += [e.time_range.end - e.time_range.start for e in pr.events() if key in e.name]
    print(f"{name:28s} {statistics.median(ts):6.1f} us", flush=True)
bench(lambda: K.dwf((21 * 8, 1, 1), (1024, 1, 1), dp, pos, chg, el, mask, dw, M, 1, 0), "dwf full (esm)", "dwf")
bench(lambda: K.dwf((5 * 8, 1, 1), (1024, 1, 1), dp, pos, chg, el, mask, dw, M, 1, 0), "dwf dense only", "dwf")
bench(lambda: K.dwf((6 * 8, 1, 1), (1024, 1, 1), dp, pos, chg, el, mask, dw, M, 0, 0), "dwf legacy", "dwf")
bench(lambda: K.colsum((2, 8, 1), (1024, 1, 1), gb, out, M), "colsum", "colsum")
