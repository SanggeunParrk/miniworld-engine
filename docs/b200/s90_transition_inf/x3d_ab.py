# x3d_ab.py D,n,M ... : D 384 inference, graph median (L2 flushed) with the 3-D x box (X3D_D384 True) vs the 12 boxes, alternated 3 times in one process; outputs must be bit-identical
import statistics, sys, torch
from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from miniworld_engine.kernels.transition.cuda.fused_sm100a import build_cubins
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
    H = n * D
    torch.manual_seed(0)
    m = Transition(D, n=n, backend="cuda").cuda().eval()
    for p in m.parameters():
        if p.dim() > 1: torch.nn.init.normal_(p, 0, D ** -0.5)
    x = torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16)
    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): return m(x)
    k = fw._Launch(x[0], H)
    # the routed (X3D) cubins are loaded by the first call; the old build is registered under the SAME names for the B graphs
    fw.X3D_D384 = True; fw.W4D_D384 = True; run(); torch.cuda.synchronize()
    pn = build_cubins(f"d{D}n{n}", (("fwdi0", "widths/tfwd_d384.cu", (f"-DHID={H}", "-DFOLD=1")), ("fwd0", "widths/tfwd_d384.cu", (f"-DHID={H}",))))
    gs, ys = {}, {}
    for on in (True, False):
        fw.X3D_D384 = on; fw.W4D_D384 = on
        if not on:
            k.ext.load_cubin(f"d{D}n{n}_fwdi", pn["fwdi0"]); k.ext.load_cubin(f"d{D}n{n}_fwd", pn["fwd0"])
        ys[on] = run().clone(); gs[on] = graph(run)
    print(f"{c:18s} max|diff| {(ys[True].float() - ys[False].float()).abs().max().item():.3e}", end="  ", flush=True)
    res = {True: [], False: []}
    for rep in range(3):
        for on in (True, False):
            ts = []
            for _ in range(40):
                flush.zero_(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); gs[on].replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
            res[on].append(statistics.median(ts))
    print(f"x3d {[round(v, 1) for v in res[True]]}  boxes {[round(v, 1) for v in res[False]]}  ratio {statistics.median(res[True]) / statistics.median(res[False]):.3f}", flush=True)
