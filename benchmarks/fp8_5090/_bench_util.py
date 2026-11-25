import torch
flush=torch.empty(128<<20,dtype=torch.uint8,device="cuda")
def bench(f,n=2):
    f(); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): flush.zero_(); f()
    gf=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n): flush.zero_()
    def t(gr):
        gr.replay(); torch.cuda.synchronize(); ts=[]
        for _ in range(5):
            e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();gr.replay();e1.record();torch.cuda.synchronize();ts.append(e0.elapsed_time(e1))
        return min(ts)
    return (t(g)-t(gf))/n*1e3
import foldjet_engine
from foldjet_engine.modules import TriangleMultiplication
for d in (128,256):
  for L in (256,512,1024,2048):
    try:
        m=TriangleMultiplication(d,master_weights=True,backend="cuda").cuda().eval()
        x=torch.randn(1,L,L,d,device="cuda",dtype=torch.bfloat16); mk=torch.ones(1,L,device="cuda",dtype=torch.bool)
        def f():
            with torch.no_grad(), torch.autocast("cuda",torch.bfloat16): m(x,mk)
        print("TriMul default d",d,"L",L,f"{bench(f):.0f} us",flush=True)
    except Exception as e: print("TriMul",d,L,"ERR",repr(e)[:300]); break
