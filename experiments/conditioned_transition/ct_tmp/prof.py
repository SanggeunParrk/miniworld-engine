import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.composed import _expand_swiglu, _squeeze_gate
def mk(M,d,n=2,dc=384):
    g=torch.Generator('cuda').manual_seed(0); f=lambda *s: torch.randn(*s,device='cuda',dtype=torch.float32,generator=g)
    return f(M,d),f(M,dc),f(n*d,d)/d**.5,f(n*d,d)/d**.5,f(d,n*d)/(n*d)**.5,f(d,dc)/dc**.5,torch.full((d,),-2.,device='cuda')
def gbench(fn,it=300):
    for _ in range(30): fn()
    torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): _=fn()
    torch.cuda.synchronize()
    for _ in range(30): g.replay()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1e3
PEAK_BW=3.35e3  # GB/s H100 SXM
PEAK_TF32=494.0 # TFLOP/s dense
print(f"torch {torch.__version__} {torch.cuda.get_device_name()}  [per-kernel CUDA-graph + roofline]")
print(f"{'L':>5} {'kernel':>8} {'us':>7} {'GB/s':>8} {'%BW':>6} {'TFLOP/s':>8} {'%cmp':>6}")
for L in (384,768,1024):
    M=L; d=768; n=2; ND=n*d; DC=384
    x,cond,Wa,Wb,Ws,Wsc,bsc = mk(M,d)
    # expand: read x(M*d)+Wa,Wb(2*ND*d)+write h(M*ND); flops 2*(M*ND*d)*2
    t_e=gbench(lambda: _expand_swiglu(x,Wa,Wb))
    h=_expand_swiglu(x,Wa,Wb)
    bytes_e=(M*d + 2*ND*d + M*ND)*4; flop_e=2*(M*ND*d)*2
    # squeeze: read h(M*ND)+Ws(d*ND)+cond(M*DC)+Wsc(d*DC)+write y(M*d); flops 2*(M*d*ND)+2*(M*d*DC)
    t_s=gbench(lambda: _squeeze_gate(h,cond,Ws,Wsc,bsc))
    bytes_s=(M*ND + d*ND + M*DC + d*DC + M*d)*4; flop_s=2*(M*d*ND)+2*(M*d*DC)
    for nm,t,by,fl in (("expand",t_e,bytes_e,flop_e),("squeeze",t_s,bytes_s,flop_s)):
        gbs=by/(t*1e-6)/1e9; tf=fl/(t*1e-6)/1e12
        print(f"{L:>5} {nm:>8} {t:7.1f} {gbs:8.0f} {100*gbs/PEAK_BW:5.0f}% {tf:8.0f} {100*tf/PEAK_TF32:5.0f}%")
print("DONE")
