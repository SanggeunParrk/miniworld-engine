"""b2b single-launch forward vs champion forward vs compile (CUDA graph), atom."""
import torch, torch.nn.functional as F
import ct_b2b_ext as ext
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
import sys; sys.path.insert(0,"/home/psk6950/miniworld-kernels/_ct_cutlass")
from miniworld_engine.kernels.conditioned_transition.triton.inference import cond_transition_inference

def cos(a,b):
    a=a.flatten().double(); b=b.flatten().double(); return (a@b/(a.norm()*b.norm()+1e-20)).item()

def ref(x,cond,wa,wb,ws,wsc,bsc):
    a=x@wa.t(); b=x@wb.t(); h=F.silu(a)*b; out=h@ws.t(); scale=cond@wsc.t()+bsc
    return torch.sigmoid(scale)*out

def mk(M):
    K,ND,D,DC=128,256,128,384
    g=torch.Generator("cuda").manual_seed(0); f=lambda *s: torch.randn(*s,device="cuda",generator=g)
    return (f(M,K),f(M,DC),f(ND,K)*.05,f(ND,K)*.05,f(D,ND)*.05,f(D,DC)*.05,f(D))

def gb(fn, M, it=200, wu=20):
    t=mk(M)
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn(*t)
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn(*t)
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it*1e3

def cb(fn, M, it=200, wu=30):
    t=mk(M)
    for _ in range(wu): fn(*t)
    torch.cuda.synchronize()
    a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(it): fn(*t)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it*1e3

refc = torch.compile(ref, mode="reduce-overhead")
print(f"{'M':>6} | {'b2b_us':>8} {'triton_inf':>10} {'compile':>8} | {'cos':>8} | vs_triton vs_comp")
for M in (2048,4096,8192):
    t=mk(M)
    y=ext.b2b_forward(*t); c=cos(y, ref(*t))
    b2b=gb(lambda *a: ext.b2b_forward(*a), M)
    trit=gb(lambda *a: cond_transition_inference(*a), M)
    comp=cb(refc, M)
    print(f"{M:>6} | {b2b:8.1f} {trit:10.1f} {comp:8.1f} | {c:8.5f} | {trit/b2b:7.2f}x {comp/b2b:6.2f}x")
print("B2B BENCH DONE")
