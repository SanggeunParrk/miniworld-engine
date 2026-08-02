"""Clean stats re-bench at real M=48*L: ours(cuBLAS+fused) vs CUTLASS vs compile, fwd+bwd.
7 timed reps each -> median + min/max spread, to quantify variance."""
import statistics, torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_engine.kernels.conditioned_transition.triton.training import cond_transition_train, set_forward_mode
from ct_full import cond_transition_train_full
from miniworld_engine.kernels.conditioned_transition.triton.train_12_345 import cond_transition_train_12_345 as fused_dgrad
A=48
def ref(x,cond,Wa,Wb,Ws,Wsc,bsc):
    h=F.silu(x@Wa.t())*(x@Wb.t()); o=h@Ws.t(); s=cond@Wsc.t()+bsc; return torch.sigmoid(s)*o
def make(M,d,n=2,dc=384):
    g=torch.Generator('cuda').manual_seed(0); f=lambda *s: torch.randn(*s,device='cuda',dtype=torch.float32,generator=g)
    ts=(f(M,d),f(M,dc),f(n*d,d)/d**.5,f(n*d,d)/d**.5,f(d,n*d)/(n*d)**.5,f(d,dc)/dc**.5,torch.full((d,),-2.,device='cuda'))
    return tuple(t.detach().requires_grad_(True) for t in ts)
def fb(fn,t): y=fn(*t); return torch.autograd.grad(y,t,torch.ones_like(y))
def gstats(make_t,fn,it=200,wu=30,reps=7):
    t=make_t(); s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fb(fn,t)
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fb(fn,t)
    for _ in range(wu): g.replay()
    torch.cuda.synchronize(); out=[]
    for _ in range(reps):
        a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
        for _ in range(it): g.replay()
        b.record(); torch.cuda.synchronize(); out.append(a.elapsed_time(b)/it*1e3)
    return statistics.median(out),min(out),max(out)
def cstats(fn,make_t,it=200,wu=30,reps=7):
    t=make_t()
    def call(): y=fn(*t); return torch.autograd.grad(y,t,torch.ones_like(y))
    for _ in range(wu): call()
    torch.cuda.synchronize(); out=[]
    for _ in range(reps):
        a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
        for _ in range(it): call()
        b.record(); torch.cuda.synchronize(); out.append(a.elapsed_time(b)/it*1e3)
    return statistics.median(out),min(out),max(out)
def sp(lo,hi,med): return 100*(hi-lo)/med
set_forward_mode("auto")
STREAMS=[("atom",128,(48*2048,48*4096,48*8192)),("token",768,(48*384,48*512,48*768,48*1024))]
shapes=[(s,d,M) for s,d,Ms in STREAMS for M in Ms]
print(f"torch {torch.__version__} {torch.cuda.get_device_name()}  A={A}  fwd+bwd us [median over 7 reps, ±=spread%]")
refc=torch.compile(ref,mode="reduce-overhead")
print(f"{'stream':>6} {'M':>7} {'d':>4} | {'fusedDG':>9} {'±%':>4} | {'cuBLASdg':>9} {'±%':>4} | {'compile':>9} {'±%':>4} | {'fDG/comp':>9} {'fDG/cuB':>8}")
for s,d,M in shapes:
    fm,flo,fhi=gstats(lambda: make(M,d),fused_dgrad)
    om,olo,ohi=gstats(lambda: make(M,d),cond_transition_train)
    pm,plo,phi=cstats(refc,lambda: make(M,d))
    print(f"{s:>6} {M:>7} {d:>4} | {fm:9.1f} {sp(flo,fhi,fm):3.0f}% | {om:9.1f} {sp(olo,ohi,om):3.0f}% | {pm:9.1f} {sp(plo,phi,pm):3.0f}% | {pm/fm:8.2f}x {om/fm:7.2f}x")
    torch.cuda.synchronize(); torch.cuda.empty_cache()
print("STATS DONE")
