# d256_ab.py n,M ... : D 256 inference graph median (L2 flushed), 3-D boxes (D256_3D) vs per-K-block boxes, alternated 3 times in one process; outputs compared
import statistics, sys, torch
from foldjet_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from foldjet_engine.modules import Transition
D = 256
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def graph(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g
ON = fw.D256_3D
for c in sys.argv[1:]:
    n, M = (int(v) for v in c.split(","))
    torch.manual_seed(0)
    m = Transition(D, n=n, backend="cuda").cuda().eval()
    for p in m.parameters():
        if p.dim() > 1: torch.nn.init.normal_(p, 0, D ** -0.5)
    x = torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16)
    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): return m(x)
    gs, ys = {}, {}
    for on in (True, False):
        fw.D256_3D = ON if on else ()
        ys[on] = run().clone(); gs[on] = graph(run)
    print(f"{D},{c:14s} max|diff| {(ys[True].float() - ys[False].float()).abs().max().item():.3e}", end="  ", flush=True)
    res = {True: [], False: []}
    for rep in range(3):
        for on in (True, False):
            ts = []
            for _ in range(40):
                flush.zero_(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); gs[on].replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
            res[on].append(statistics.median(ts))
    print(f"3d {[round(v, 1) for v in res[True]]}  boxes {[round(v, 1) for v in res[False]]}  ratio {statistics.median(res[True]) / statistics.median(res[False]):.3f}", flush=True)
