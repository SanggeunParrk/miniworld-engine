import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/snu_hwle/psk/mw-bidir/src")
import torch
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication as TMB)
from miniworld_kernels.modules.triangle_multiplication import TriangleMultiplication as TM
from miniworld_kernels.modules import ImplementationType

torch.manual_seed(0)
which = sys.argv[1]          # "cute" | "cueq"
L = int(sys.argv[2])
N = 100
d = 128; dev = "cuda"

def rnd(m):
    for n, p in m.named_parameters():
        if n.endswith("ln_pair.weight") or n.endswith("ln_out.weight"): p.data.normal_(1.0, 0.05)
        elif n.endswith(".bias"): p.data.normal_(0.0, 0.02)
        else: p.data.normal_(0.0, 0.05)
    return m

pair = torch.randn(1, L, L, d, device=dev, dtype=torch.bfloat16)
if which == "cute":
    m = rnd(TMB(d_pair=d, d_hidden=d, implementation=ImplementationType("cute")).to(dev)).to(torch.bfloat16).eval()
    fn = lambda: m(pair)
else:
    om = rnd(TM(d, outgoing=True, implementation=ImplementationType("cuequivariance")).to(dev)).to(torch.bfloat16).eval()
    im = rnd(TM(d, outgoing=False, implementation=ImplementationType("cuequivariance")).to(dev)).to(torch.bfloat16).eval()
    fn = lambda: om(pair) + im(pair)

with torch.no_grad():
    for _ in range(10): fn()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(8): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): _ = fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("REPLAY")
    for _ in range(N): g.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print(f"done {which} L={L} N={N}")
