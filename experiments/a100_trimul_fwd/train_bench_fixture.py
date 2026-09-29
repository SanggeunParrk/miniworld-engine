"""The training fixture of train_bench.py (the baseline's module and shapes, dropout row scale fixed)."""
import torch
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication


def init_(m):
    torch.manual_seed(1234)
    with torch.no_grad():
        for n, t in m.named_parameters():
            if t.ndim >= 2: t.normal_(std=t.shape[-1] ** -.5)
            elif "weight" in n: t.copy_(1 + .1 * torch.randn_like(t))
            else: t.normal_(std=.05)
    return m


def make(var, L, p=0.25):
    mod = (BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH, p_drop=p) if var == "bidir"
           else TriangleMultiplication(128, implementation=I.PYTORCH, p_drop=p))
    mod = init_(mod.cuda().bfloat16()).train()
    torch.manual_seed(90323)
    z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.rand(1, L, device="cuda") > .1
    ds = ((torch.rand(1, 1, L, 128, device="cuda") > p).to(torch.bfloat16) / (1 - p)).to(torch.bfloat16)
    dy = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    return mod, z, mask, ds, dy
