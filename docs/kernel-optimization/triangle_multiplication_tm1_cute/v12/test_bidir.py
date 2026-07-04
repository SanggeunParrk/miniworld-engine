import sys, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/snu_hwle/psk/miniworld-kernels/src")
from miniworld_kernels.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import _load_cute_fns
torch.manual_seed(0)
DEV="cuda"; d=128; h=d; L=int(sys.argv[1]) if len(sys.argv)>1 else 256
base=BidirectionalTriangleMultiplication(d_pair=d,d_hidden=h,implementation=ImplementationType.PYTORCH).to(DEV)
for lin in (base.to_left,base.to_left_gate,base.to_right,base.to_right_gate,base.to_gate,base.to_out):
    nn.init.normal_(lin.weight,std=d**-0.5)
base=base.to(torch.bfloat16)
pair=torch.randn(1,L,L,d,device=DEV,dtype=torch.bfloat16)
with torch.inference_mode():
    y_ref=base(pair).float()
    # fp32 ref
    base_f=BidirectionalTriangleMultiplication(d_pair=d,d_hidden=h,implementation=ImplementationType.PYTORCH).to(DEV)
    base_f.load_state_dict(base.state_dict()); base_f=base_f.float()
    y_f32=base_f(pair.float())

tm1f,tm2f,fused_ln_mask,lnt=_load_cute_fns()
def cute_bidir(pair):
    b,l1,l2,dd=pair.shape; M=b*l1*l2
    o=lnt(pair.reshape(M,dd), base.ln_pair.weight, base.ln_pair.bias, eps=1e-5, layout="nd->nd")
    x=(o[0] if isinstance(o,tuple) else o).view(b,l1,l2,dd)
    def half(sl):
        return tm1f(x, base.to_left.weight[sl].T.contiguous(), base.to_left_gate.weight[sl].T.contiguous(),
                       base.to_right.weight[sl].T.contiguous(), base.to_right_gate.weight[sl].T.contiguous(),
                       out_layout="bdll_sm100")
    lo,ro=half(slice(0,h)); li,ri=half(slice(h,2*h))
    O=torch.einsum("bdik,bdjk->bdij",lo,ro)   # outgoing
    I=torch.einsum("bdki,bdkj->bdij",li,ri)   # incoming
    tri=torch.cat([O,I],dim=1)                 # [B,2h,L,L]
    tri_dbn=tri.permute(1,0,2,3).reshape(2*h,b,l1*l2)
    oo=lnt(tri_dbn, base.ln_out.weight, base.ln_out.bias, eps=1e-5, layout="dbn->bnd")
    outn=(oo[0] if isinstance(oo,tuple) else oo).view(b,l1,l2,2*h)
    gate=torch.sigmoid(x.reshape(M,dd) @ base.to_gate.weight.T)
    outp=outn.reshape(M,2*h) @ base.to_out.weight.T
    return (gate*outp).view(b,l1,l2,dd)
with torch.inference_mode():
    y_cute=cute_bidir(pair).float()
def cos(a,b): a,b=a.flatten(),b.flatten(); return (a@b/(a.norm()*b.norm()+1e-20)).item()
print(f"L={L}: cos(cute, bf16-ref)={cos(y_cute,y_ref):.6f}  cos(cute, fp32-ref)={cos(y_cute,y_f32):.6f}  cos(bf16ref, fp32ref)={cos(y_ref,y_f32):.6f}",flush=True)
