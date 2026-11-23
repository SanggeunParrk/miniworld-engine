import sys, copy, torch
import miniworld_engine
from miniworld_engine import fp8
from miniworld_engine.modules import TriangleMultiplication
cases=[tuple(map(int,c.split(","))) for c in sys.argv[1:]]  # d,L,outgoing(1/0)
flush=torch.empty(128<<20,dtype=torch.uint8,device="cuda")
def graph_time(f,n=3):
    f(); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): flush.zero_(); f()
    gf=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n): flush.zero_()
    def t(gr):
        gr.replay(); torch.cuda.synchronize(); ts=[]
        for _ in range(7):
            e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();gr.replay();e1.record();torch.cuda.synchronize();ts.append(e0.elapsed_time(e1))
        return sorted(ts)[len(ts)//2]
    return (t(g)-t(gf))/n*1e3
print("arch",fp8.arch())
for d,L,out in cases:
    torch.manual_seed(0)
    m=TriangleMultiplication(d,outgoing=bool(out),master_weights=True,backend="cuda").cuda().eval()
    for p in m.parameters(): p.data.normal_(0,0.1) if p.dim()>1 else None
    x=torch.randn(1,L,L,d,device="cuda",dtype=torch.bfloat16); mk=torch.ones(1,L,device="cuda",dtype=torch.bool); mk[0,-L//10:]=False
    def run():
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16): return m(x,mk)
    fp8.disable(); y0=run().float()
    fp8.enable("safe"); fp8.reset_route_stats(); y1=run().float(); y1=run().float(); st=fp8.route_stats()["trimul"]; chk=fp8.check()
    # fp64 reference
    m64=copy.deepcopy(m).double(); m64.set_backend("pytorch")
    with torch.no_grad(): ref=m64(x.double(),mk).double()
    def err(y): 
        e=(y.double()-ref); return (e.pow(2).mean().sqrt()/ref.pow(2).mean().sqrt()).item(), (e.abs().max()/ref.abs().max()).item()
    e0,e1=err(y0),err(y1)
    # timing: A-B alternate
    ta=[];tb=[]
    for r in range(3):
        fp8.disable(); 
        ta.append(graph_time(run))
        fp8.enable("safe"); run(); 
        tb.append(graph_time(run))
    print(f"d{d} L{L} {'out' if out else 'in '} | default rel-rms {e0[0]:.2e} max {e0[1]:.2e} | fp8 rel-rms {e1[0]:.2e} max {e1[1]:.2e} | route {st} sat={chk} | default {min(ta):.0f} us fp8 {min(tb):.0f} us ratio {min(ta)/min(tb):.3f}",flush=True)
