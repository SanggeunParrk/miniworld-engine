import sys, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/snu_hwle/psk/miniworld-kernels/src")
import triton
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication.module import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.baseline_dtv1_bidir import fused_bidirectional_dtv1
DEV="cuda"; DT=torch.bfloat16; d=128
def cos(a,b): a,b=a.float().flatten(),b.float().flatten(); return (a@b/(a.norm()*b.norm()+1e-20)).item()

def graph_bench(fn, pair):
    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(5): fn(pair)
    torch.cuda.current_stream().wait_stream(st)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn(pair)
    def run(): g.replay()
    return triton.testing.do_bench(run, warmup=30, rep=100, return_mode="median"), out

class DtV1(nn.Module):
    def __init__(self,base): super().__init__(); self.b=base; self.h=base.d_hidden
    def forward(self,pair):
        b=self.b
        return fused_bidirectional_dtv1(pair,None,norm_in_weight=b.ln_pair.weight,norm_in_bias=b.ln_pair.bias,
            p_in_weight=torch.cat([b.to_left.weight,b.to_right.weight],dim=0),
            g_in_weight=torch.cat([b.to_left_gate.weight,b.to_right_gate.weight],dim=0),
            norm_out_weight=b.ln_out.weight,norm_out_bias=b.ln_out.bias,
            p_out_weight=b.to_out.weight,g_out_weight=b.to_gate.weight,h=self.h,eps=1e-5)
class Cueq2(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.o=TriangleMultiplication(d_pair=d,d_hidden=d,outgoing=True,implementation=ImplementationType.CUEQUIVARIANCE).to(DEV,DT)
        self.i=TriangleMultiplication(d_pair=d,d_hidden=d,outgoing=False,implementation=ImplementationType.CUEQUIVARIANCE).to(DEV,DT)
    def forward(self,pair): return self.o(pair)+self.i(pair)

for L in [384,768,1024]:
    torch.manual_seed(0)
    base=BidirectionalTriangleMultiplication(d_pair=d,d_hidden=d,implementation=ImplementationType.PYTORCH).to(DEV)
    for lin in (base.to_left,base.to_left_gate,base.to_right,base.to_right_gate,base.to_gate,base.to_out):
        nn.init.normal_(lin.weight,std=d**-0.5)
    base=base.to(DT)
    cute=BidirectionalTriangleMultiplication(d_pair=d,d_hidden=d,implementation=ImplementationType.CUTE).to(DEV)
    cute.load_state_dict(base.state_dict()); cute=cute.to(DT)
    pair=torch.randn(1,L,L,d,device=DEV,dtype=DT)
    with torch.inference_mode():
        yref=base(pair)
        res={}
        for name,mod in (("cute",cute),):
            try:
                ms,out=graph_bench(mod,pair); res[name]=(ms,cos(yref,out))
            except Exception as e: res[name]=(float("nan"),str(e)[:60])
        for name,mk in (("cuequiv",lambda:Cueq2(d)),("dtv1",lambda:DtV1(base))):
            try:
                mod=mk(); 
                ms,out=graph_bench(mod,pair); res[name]=(ms,cos(yref,out))
            except Exception as e:
                try: ms=triton.testing.do_bench(lambda:mk()(pair),warmup=20,rep=50,return_mode="median"); res[name]=(ms,-9)
                except Exception as e2: res[name]=(float("nan"),str(e2)[:50])
    line=f"L={L}: "
    for k in ("cute","cuequiv","dtv1"):
        m,c=res.get(k,(float("nan"),0)); line+=f"{k}={m:.4f}ms(cos={c if isinstance(c,float) else c}) "
    print(line,flush=True)
