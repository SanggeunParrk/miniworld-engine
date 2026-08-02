"""Training fwd+bwd A/B: save_xn=True (Version B, save xn) vs save_xn=False (Version A, recompute xn).
Measures the full step (forward + .backward()) for the transition fused training path."""
import torch, time
from miniworld_engine import kernels

dev = "cuda"
torch.manual_seed(0)
d, n, eps = 128, 4, 1e-5

def make(M):
    x = torch.randn(M, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    wa = (torch.randn(n*d, d, device=dev) / d**0.5).bfloat16().requires_grad_()
    wb = (torch.randn(n*d, d, device=dev) / d**0.5).bfloat16().requires_grad_()
    ws = (torch.randn(d, n*d, device=dev) / (n*d)**0.5).bfloat16().requires_grad_()
    g = (torch.randn(d, device=dev)*0.1 + 1).bfloat16().requires_grad_()
    b = (torch.randn(d, device=dev)*0.1).bfloat16().requires_grad_()
    return x, g, b, wa, wb, ws

def step(args, save_xn):
    x, g, b, wa, wb, ws = args
    for t in args:
        if t.grad is not None: t.grad = None
    out = kernels.triton_transition_fused(x, g, b, wa, wb, ws, n, eps, save_xn=save_xn)
    out.sum().backward()
    return out

def bench(M, save_xn, iters=50):
    args = make(M)
    for _ in range(15): step(args, save_xn)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): step(args, save_xn)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1e3

for L in (512, 768, 1024):
    M = L*L
    tB = bench(M, True)
    tA = bench(M, False)
    print(f"L={L:5d} M={M:8d}  save_xn=True(B)={tB:.3f}ms  save_xn=False(A,recompute)={tA:.3f}ms  A/B={tB/tA:.3f}x")
