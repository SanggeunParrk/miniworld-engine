import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/snu_hwle/psk/mw-bidir/src")
import torch
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication as TMB)
from miniworld_kernels.modules.triangle_multiplication import TriangleMultiplication as TM
from miniworld_kernels.modules import ImplementationType
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training_b200 import BidirTriMulB200Train
torch.manual_seed(0)
which = sys.argv[1]; L = int(sys.argv[2]); N = 50; d = 128; dev = "cuda"
def rnd(m):
    for n, p in m.named_parameters():
        if n.endswith("ln_pair.weight") or n.endswith("ln_out.weight"): p.data.normal_(1.0, 0.05)
        elif n.endswith(".bias"): p.data.normal_(0.0, 0.02)
        else: p.data.normal_(0.0, 0.05)
    return m
pair = torch.randn(1, L, L, d, device=dev, dtype=torch.bfloat16).requires_grad_(True)
cot = torch.randn(1, L, L, d, device=dev, dtype=torch.bfloat16)
base = rnd(TMB(d_pair=d, d_hidden=d, implementation=ImplementationType("pytorch")).to(dev))
if which == "cute":
    m = BidirTriMulB200Train(base).to(dev).to(torch.bfloat16)
    def step():
        for p in m.parameters(): p.grad = None
        y = m(pair); (y * cot).sum().backward()
else:
    om = rnd(TM(d, outgoing=True, implementation=ImplementationType("cuequivariance")).to(dev)).to(torch.bfloat16)
    im = rnd(TM(d, outgoing=False, implementation=ImplementationType("cuequivariance")).to(dev)).to(torch.bfloat16)
    def step():
        for mm in (om, im):
            for p in mm.parameters(): p.grad = None
        y = om(pair) + im(pair); (y * cot).sum().backward()
for _ in range(5): step()
torch.cuda.synchronize()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(5): step()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): step()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("REPLAY")
for _ in range(N): g.replay()
torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
print(f"done {which} L={L} N={N}")
