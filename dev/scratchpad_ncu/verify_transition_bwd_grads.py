"""Grad correctness: Version A (save_xn=False, recompute, now stacked) & B (save_xn=True)
vs a pure-torch transition reference. All grads must cos ~1.0."""
import torch
from miniworld_kernels import kernels

dev = "cuda"; torch.manual_seed(0)
d, n, eps = 128, 4, 1e-5
M = 512 * 512

def mk():
    x = torch.randn(M, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    wa = (torch.randn(n*d, d, device=dev)/d**0.5).bfloat16().requires_grad_()
    wb = (torch.randn(n*d, d, device=dev)/d**0.5).bfloat16().requires_grad_()
    ws = (torch.randn(d, n*d, device=dev)/(n*d)**0.5).bfloat16().requires_grad_()
    g = (torch.randn(d, device=dev)*0.1+1).bfloat16().requires_grad_()
    b = (torch.randn(d, device=dev)*0.1).bfloat16().requires_grad_()
    return x, g, b, wa, wb, ws

base = mk()
gy = torch.randn(M, d, device=dev, dtype=torch.bfloat16)

def clone(args):
    return tuple(t.detach().clone().requires_grad_() for t in args)

def ref(args):
    x, g, b, wa, wb, ws = args
    xf = x.float()
    mean = xf.mean(-1, keepdim=True); var = xf.var(-1, unbiased=False, keepdim=True)
    xn = ((xf-mean)*torch.rsqrt(var+eps))*g.float()+b.float()
    a = xn @ wa.float().T; bb = xn @ wb.float().T
    h = (a*torch.sigmoid(a))*bb
    return (h @ ws.float().T)

def run_fused(args, save_xn):
    x, g, b, wa, wb, ws = args
    out = kernels.triton_transition_fused(x, g, b, wa, wb, ws, n, eps, save_xn=save_xn)
    return out

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

names = ["dx", "dg", "db", "dWa", "dWb", "dWs"]
# reference grads
aR = clone(base); ref(aR).mul(gy.float()).sum().backward()
gR = [t.grad for t in aR]

for save_xn in (True, False):
    a = clone(base)
    run_fused(a, save_xn).mul(gy).sum().backward()
    gT = [t.grad for t in a]
    tag = "B(save_xn)" if save_xn else "A(recompute,stacked)"
    print(f"--- Version {tag} vs torch ref ---")
    for nm, t, r in zip(names, gT, gR):
        print(f"   cos {nm:4s} = {cos(t, r):.6f}")
