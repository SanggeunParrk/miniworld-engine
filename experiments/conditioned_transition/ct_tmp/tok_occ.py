import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.composed import cond_transition_inference_composed as fwd

def cos(a,b):
    a=a.double().reshape(-1); b=b.double().reshape(-1); return (a@b/(a.norm()*b.norm()+1e-12)).item()
def ref(x,cond,Wa,Wb,Ws,Wsc,bsc):
    h=F.silu(x@Wa.t())*(x@Wb.t()); out=h@Ws.t(); s=cond@Wsc.t()+bsc; return torch.sigmoid(s)*out
def mk(M,d,n=2,dc=384):
    g=torch.Generator('cuda').manual_seed(0); f=lambda *s: torch.randn(*s,device='cuda',dtype=torch.float32,generator=g)
    return f(M,d),f(M,dc),f(n*d,d)/d**.5,f(n*d,d)/d**.5,f(d,n*d)/(n*d)**.5,f(d,dc)/dc**.5,torch.full((d,),-2.,device='cuda')
def graph_bench(args, it=200):
    # warmup (triggers autotune) then CUDA-graph capture+replay -> no launch overhead
    for _ in range(30): fwd(*args)
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): _=fwd(*args)
    torch.cuda.synchronize()
    for _ in range(30): g.replay()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1e3
print(f"torch {torch.__version__} {torch.cuda.get_device_name()}  [CUDA-graph]")
print(f"{'L':>5} {'d':>4} | {'cos':>9} {'us':>8}   (prior token CUDA-graph: 384=29.8 512=32.6 768=57.0 1024=60.7)")
for L in (384,512,768,1024):
    a=mk(L,768); r=ref(*a); y=fwd(*a); c=cos(y,r); t=graph_bench(a)
    print(f"{L:>5} {768:>4} | {c:9.6f} {t:7.1f}")
print("DONE")
