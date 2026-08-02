"""Training fwd+bwd: (a) old Version B (save_xn=True, triton fwd), (b) Version A + triton fwd,
(c) Version A + CUDA b2b fwd (new single path). + grad correctness of (c) vs torch."""
import os, time, torch
from miniworld_engine import kernels

dev = "cuda"; torch.manual_seed(0)
d, n, eps = 128, 4, 1e-5

def mk(M):
    x = torch.randn(M, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    wa = (torch.randn(n*d, d, device=dev)/d**0.5).bfloat16().requires_grad_()
    wb = (torch.randn(n*d, d, device=dev)/d**0.5).bfloat16().requires_grad_()
    ws = (torch.randn(d, n*d, device=dev)/(n*d)**0.5).bfloat16().requires_grad_()
    g = (torch.randn(d, device=dev)*0.1+1).bfloat16().requires_grad_()
    b = (torch.randn(d, device=dev)*0.1).bfloat16().requires_grad_()
    return [x, g, b, wa, wb, ws]

def step(args, save_xn):
    for t in args:
        t.grad = None
    out = kernels.triton_transition_fused(args[0], args[1], args[2], args[3], args[4], args[5], n, eps, save_xn=save_xn)
    out.sum().backward()

def bench(M, save_xn, cuda_b2b, iters=50):
    os.environ["MINIWORLD_TRANSITION_CUDA_B2B"] = cuda_b2b
    args = mk(M)
    for _ in range(15): step(args, save_xn)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): step(args, save_xn)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1e3

# correctness: Version A + cuda fwd vs torch
def ref(args):
    x,g,b,wa,wb,ws = args
    xf=x.float(); mean=xf.mean(-1,keepdim=True); var=xf.var(-1,unbiased=False,keepdim=True)
    xn=((xf-mean)*torch.rsqrt(var+eps))*g.float()+b.float()
    a=xn@wa.float().T; bb=xn@wb.float().T; h=(a*torch.sigmoid(a))*bb
    return h@ws.float().T
def cos(a,b): return torch.nn.functional.cosine_similarity(a.float().flatten(),b.float().flatten(),0).item()
os.environ["MINIWORLD_TRANSITION_CUDA_B2B"]="1"
base=mk(512*512); gy=torch.randn(512*512,d,device=dev,dtype=torch.bfloat16)
def clone(a): return [t.detach().clone().requires_grad_() for t in a]
aR=clone(base); ref(aR).mul(gy.float()).sum().backward(); gR=[t.grad for t in aR]
aC=clone(base); kernels.triton_transition_fused(aC[0],aC[1],aC[2],aC[3],aC[4],aC[5],n,eps,save_xn=False).mul(gy).sum().backward()
print("=== grad correctness: Version A + CUDA fwd vs torch ===")
for nm,t,r in zip(["dx","dg","db","dWa","dWb","dWs"], [t.grad for t in aC], gR):
    print(f"   cos {nm:4s} = {cos(t,r):.6f}")

print("=== training step ms ===")
for L in (512, 768, 1024):
    M=L*L
    tB = bench(M, True,  "0")   # old Version B (triton fwd, save xn)
    tAt= bench(M, False, "0")   # Version A + triton fwd
    tAc= bench(M, False, "1")   # Version A + CUDA b2b fwd (new default)
    print(f"L={L:5d}  B(save_xn,triton)={tB:.3f}  A+triton={tAt:.3f}  A+CUDAfwd={tAc:.3f}ms  | new/oldB={tAc/tB:.3f}x")
