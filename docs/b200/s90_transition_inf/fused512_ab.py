# fold_ab.py D,n,M ... : graph median (L2 flushed) of the Transition cuda inference call with the fold (FOLD_LN) on / off, alternated 3 times in one process
import statistics, sys, torch
from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from miniworld_engine.modules import Transition
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def graph(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g

for c in sys.argv[1:]:
    D, n, M = (int(v) for v in c.split(","))
    torch.manual_seed(0)
    m = Transition(D, n=n, backend="cuda").cuda().eval()
    for p in m.parameters():
        if p.dim() > 1: torch.nn.init.normal_(p, 0, D ** -0.5)
    x = torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16)
    gs = {}
    for fold in (True, False):  # True = fused 512
        fw.FWD512 = fold
        def fn():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): return m(x)
        gs[fold] = graph(fn)
    res = {True: [], False: []}
    for rep in range(3):
        for fold in (True, False):  # True = fused 512
            ts = []
            for _ in range(40):
                flush.zero_(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); gs[fold].replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
            res[fold].append(statistics.median(ts))
    print(f"{c:22s} fused {[round(v, 1) for v in res[True]]}  base {[round(v, 1) for v in res[False]]}  ratio {statistics.median(res[True]) / statistics.median(res[False]):.3f}", flush=True)
