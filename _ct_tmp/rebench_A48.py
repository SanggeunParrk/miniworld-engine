import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.composed import cond_transition_inference_composed as fwd2k  # 1+2|3+4+5
from miniworld_kernels.kernels.conditioned_transition.triton.inference import cond_transition_inference as b2b      # single-kernel (atom)
A=48
def cos(a,b):
    a=a.double().reshape(-1); b=b.double().reshape(-1); return (a@b/(a.norm()*b.norm()+1e-12)).item()
def ref(x,c,Wa,Wb,Ws,Wsc,bsc):
    h=F.silu(x@Wa.t())*(x@Wb.t()); o=h@Ws.t(); s=c@Wsc.t()+bsc; return torch.sigmoid(s)*o
def mk(M,d,n=2,dc=384):
    g=torch.Generator('cuda').manual_seed(0); f=lambda *s: torch.randn(*s,device='cuda',dtype=torch.float32,generator=g)
    return f(M,d),f(M,dc),f(n*d,d)/d**.5,f(n*d,d)/d**.5,f(d,n*d)/(n*d)**.5,f(d,dc)/dc**.5,torch.full((d,),-2.,device='cuda')
def gbench(fn,it=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): _=fn()
    torch.cuda.synchronize()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1e3
def lbench(fn,it=50):
    for _ in range(15): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it*1e3
print(f"torch {torch.__version__} {torch.cuda.get_device_name()}  A={A}  [CUDA-graph fwd, us]")
print(f"{'stream':>6} {'L':>5} {'M=A*L':>8} {'d':>4} | {'2kern':>9} {'b2b':>9} {'eager':>9} {'compile':>9} | {'2k/comp':>8} {'b2b/comp':>8} {'cos':>9}")
cfg=[("atom",128,(2048,4096,8192)),("token",768,(384,512,768,1024))]
for stream,d,Ls in cfg:
    for L in Ls:
        M=A*L; x,c,Wa,Wb,Ws,Wsc,bsc=mk(M,d); r=ref(x,c,Wa,Wb,Ws,Wsc,bsc)
        y2=fwd2k(x,c,Wa,Wb,Ws,Wsc,bsc); cc=cos(y2,r)
        t2=gbench(lambda: fwd2k(x,c,Wa,Wb,Ws,Wsc,bsc))
        tb=float('nan')
        if d<=128:
            try: tb=gbench(lambda: b2b(x,c,Wa,Wb,Ws,Wsc,bsc))
            except Exception: tb=float('nan')
        te=gbench(lambda: ref(x,c,Wa,Wb,Ws,Wsc,bsc))
        cf=torch.compile(ref,mode="reduce-overhead")
        for _ in range(5): cf(x,c,Wa,Wb,Ws,Wsc,bsc)
        tc=lbench(lambda: cf(x,c,Wa,Wb,Ws,Wsc,bsc))
        bs=f"{tb:8.1f}" if tb==tb else f"{'n/a':>8}"
        bc=f"{tc/tb:7.2f}x" if tb==tb else f"{'n/a':>8}"
        print(f"{stream:>6} {L:>5} {M:>8} {d:>4} | {t2:8.1f} {bs} {te:8.1f} {tc:8.1f} | {tc/t2:6.2f}x {bc} {cc:9.6f}")
print("DONE")
