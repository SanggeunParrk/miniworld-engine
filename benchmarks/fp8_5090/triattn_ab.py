"""TriAttn default vs e4m3 route, same process, alternating, CUDA graph, 128 MB L2 flush; error vs the float64 module.
python triattn_ab.py [--mask] [--rounds N] d,h,L ...   (d_pair, heads of 32, L)"""
import copy, sys, torch
import miniworld_engine
from miniworld_engine import fp8
from miniworld_engine.modules import TriangleAttention
args = sys.argv[1:]
use_mask = "--mask" in args; args = [a for a in args if a != "--mask"]
rounds = 6
if "--rounds" in args:
    i = args.index("--rounds"); rounds = int(args[i + 1]); del args[i:i + 2]
cases = [tuple(map(int, c.split(","))) for c in args]
flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
def graph_time(f, n=2):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): flush.zero_(); f()
    gf = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n): flush.zero_()
    def t(gr):
        gr.replay(); torch.cuda.synchronize(); ts = []
        for _ in range(5):
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        return sorted(ts)[len(ts) // 2]
    return (t(g) - t(gf)) / n * 1e3
print("arch", fp8.arch(), "mask", use_mask, flush=True)
for d, h, L in cases:
    torch.manual_seed(0)
    m = TriangleAttention(d, n_head=h, master_weights=True, backend="cuda").cuda().eval()
    for p in m.parameters():
        if p.dim() > 1: p.data.normal_(0, 0.1)
    x = torch.randn(1, L, L, d, device="cuda", dtype=torch.bfloat16)
    mk = torch.ones(1, L, device="cuda", dtype=torch.bool)
    if use_mask: mk[0, -L // 10:] = False
    def run():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return m(x, mk)
    fp8.disable(); y0 = run().float()
    fp8.enable("safe", model="af3"); fp8.reset_route_stats(); y1 = run().float(); y1 = run().float(); st = fp8.route_stats()["triattn"]; chk = fp8.check()
    if L <= 512:
        m64 = copy.deepcopy(m).double(); m64.set_backend("pytorch")
        with torch.no_grad(): ref = m64(x.double(), mk).double()
        def err(y):
            u = ref - x.double(); e = y.double() - ref
            return (e.pow(2).mean().sqrt() / u.pow(2).mean().sqrt()).item(), (e.abs().max() / u.abs().max()).item()
        e0, e1 = err(y0), err(y1)
    else:
        e0 = e1 = (float("nan"),) * 2
    ta, tb = [], []
    for r in range(rounds):
        fp8.disable(); ta.append(graph_time(run))
        fp8.enable("safe", model="af3"); run(); tb.append(graph_time(run))
    ta_, tb_ = sorted(ta)[len(ta) // 2], sorted(tb)[len(tb) // 2]
    print(f"d{d} h{h} L{L} | default upd-err rms {e0[0]:.3e} max {e0[1]:.3e} | fp8 rms {e1[0]:.3e} max {e1[1]:.3e} | fired pv_front={st.get('pv_front')} sat={chk} | default {ta_:.0f} us fp8 {tb_:.0f} us ratio {ta_/tb_:.3f} (min-based {min(ta)/min(tb):.3f})", flush=True)
