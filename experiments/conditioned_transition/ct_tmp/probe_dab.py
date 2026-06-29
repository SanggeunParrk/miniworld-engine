import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.training import _swiglu_bwd_packed, _swiglu

dev="cuda"; g=torch.Generator(dev).manual_seed(1)
f=lambda *s: torch.randn(*s,device=dev,dtype=torch.float32,generator=g)
M,ND=300,256
# make a,b as views of a contiguous (M,2ND) like the real path
ab = f(M,2*ND)
a,b = ab[:,:ND], ab[:,ND:]
dh = f(M,ND)

# reference da/db
sa = torch.sigmoid(a)
silu = a*sa
silu_p = sa + silu*(1-sa)
da_ref = dh*b*silu_p
db_ref = dh*silu

dab = _swiglu_bwd_packed(a,b,dh)
da = dab[:,:ND]; db = dab[:,ND:]
def cos(x,y):
    x=x.double().reshape(-1); y=y.double().reshape(-1)
    return (x@y/(x.norm()*y.norm()+1e-12)).item()
print("cos da", cos(da,da_ref), "max", (da-da_ref).abs().max().item())
print("cos db", cos(db,db_ref), "max", (db-db_ref).abs().max().item())

# also test fwd swiglu on views
h = _swiglu(a,b); h_ref = F.silu(a)*b
print("cos h", cos(h,h_ref))
print("DONE")
