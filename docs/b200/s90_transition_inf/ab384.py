# ab384.py FLAG... -- CASE...: graph median (L2 flushed) of Transition cuda inference at D,n,M,bf16 cases with the D 384 inference build flags FLAG... (-DSTAG=3 ...)
import statistics, sys
import torch
from foldjet_engine.kernels.transition.cuda import fused_wide_sm100a as fw
a = sys.argv[1:]; i = a.index("--"); fw.D384_INFER_FLAGS = tuple(a[:i]); cases = a[i + 1:]
from foldjet_engine.modules import Transition
torch.backends.cuda.matmul.allow_tf32 = True
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def graph(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g
for c in cases:
    D, n, M = (int(v) for v in c.split(","))
    torch.manual_seed(0)
    m = Transition(D, n=n, backend="cuda").cuda()
    for p in m.parameters():
        if p.dim() > 1: torch.nn.init.normal_(p, 0, D ** -0.5)
    x = torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    m.eval()
    def fn():
        with torch.no_grad(), ac: return m(x)
    g = graph(fn)
    ts = []
    for _ in range(60):
        flush.zero_(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
    print(f"{' '.join(a[:i]) or 'base':14s} {c:18s} median {statistics.median(ts):9.1f} us", flush=True)
