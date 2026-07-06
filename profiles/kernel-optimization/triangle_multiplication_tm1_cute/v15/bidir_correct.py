import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/snu_hwle/psk/miniworld-kernels/src")
import torch
from miniworld_kernels.modules.triangle_multiplication.reference import TriangleMultiplicationReference
from miniworld_kernels.modules.triangle_multiplication import TriangleMultiplication
from miniworld_kernels.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
from miniworld_kernels.modules import ImplementationType
torch.manual_seed(0); d=128; dev="cuda"; L=1024
def rnd(m):
    for n,p in m.named_parameters():
        if n.endswith("ln_pair.weight") or n.endswith("ln_out.weight"): p.data.normal_(1.0,0.05)
        elif n.endswith(".bias"): p.data.normal_(0.0,0.02)
        else: p.data.normal_(0.0,0.05)
    return m
def score(y,r):
    e=(y-r).abs(); return torch.nn.functional.cosine_similarity(y.flatten(),r.flatten(),dim=0).item(),(e.mean()/r.abs().mean()).item(),e.max().item()
def graph_cos(m,pair):
    with torch.no_grad():
        eager=m(pair,None).clone(); si=pair.clone()
        s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(8): _=m(si,None)
        torch.cuda.current_stream().wait_stream(s)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out=m(si,None)
        si.copy_(pair); g.replay(); torch.cuda.synchronize()
        return score(out.float(),eager.float())
pair=torch.randn(1,L,L,d,device=dev)
print(f"=== SINGLE-DIR L={L} ===")
ref=rnd(TriangleMultiplicationReference(d).to(dev))
with torch.inference_mode(): yref=ref(pair.float(),None).float()
m=TriangleMultiplication(d,implementation=ImplementationType("cute")).to(dev); m.load_state_dict(ref.state_dict()); m=m.to(torch.bfloat16).eval()
with torch.inference_mode(): y=m(pair.to(torch.bfloat16),None).float()
c,r,mx=score(y,yref); print(f"  eager vs fp32ref: cos={c:.6f} relmean={r:.3e} maxabs={mx:.3e}")
gc,gr,gmx=graph_cos(m,pair.to(torch.bfloat16)); print(f"  graph vs eager:   cos={gc:.6f} relmean={gr:.3e} maxabs={gmx:.3e}")
print(f"=== BIDIR L={L} ===")
bref=rnd(BidirectionalTriangleMultiplication(d,implementation=ImplementationType("pytorch")).to(dev))
with torch.inference_mode(): ybref=bref(pair.float(),None).float()
bm=BidirectionalTriangleMultiplication(d,implementation=ImplementationType("cute")).to(dev); bm.load_state_dict(bref.state_dict()); bm=bm.to(torch.bfloat16).eval()
with torch.inference_mode(): yb=bm(pair.to(torch.bfloat16),None).float()
c,r,mx=score(yb,ybref); print(f"  eager vs fp32ref: cos={c:.6f} relmean={r:.3e} maxabs={mx:.3e}")
gc,gr,gmx=graph_cos(bm,pair.to(torch.bfloat16)); print(f"  graph vs eager:   cos={gc:.6f} relmean={gr:.3e} maxabs={gmx:.3e}")
