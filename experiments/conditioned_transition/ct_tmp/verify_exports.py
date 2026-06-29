"""Confirm exports import and the rewritten dgemm path is still correct."""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition import (
    cond_transition_train_fused, ConditionedTransitionTailFusedFunction,
)
from miniworld_kernels.kernels.conditioned_transition.triton import set_wgrad_backend

def cos(a,b):
    a=a.double().reshape(-1); b=b.double().reshape(-1)
    return (a@b/(a.norm()*b.norm()+1e-12)).item()

def ref(x,cond,Wa,Wb,Ws,Wsc,bsc):
    a=x@Wa.t(); b=x@Wb.t(); h=F.silu(a)*b; out=h@Ws.t()
    return torch.sigmoid(cond@Wsc.t()+bsc)*out

dev="cuda"; g=torch.Generator(dev).manual_seed(3)
f=lambda *s: torch.randn(*s,device=dev,generator=g)
for d,M in [(128,2048),(768,512)]:
    n=2; dc=384
    x=f(M,d).requires_grad_(True); cond=f(M,dc).requires_grad_(True)
    Wa=(f(n*d,d)/d**0.5).requires_grad_(True); Wb=(f(n*d,d)/d**0.5).requires_grad_(True)
    Ws=(f(d,n*d)/(n*d)**0.5).requires_grad_(True); Wsc=(f(d,dc)/dc**0.5).requires_grad_(True)
    bsc=torch.full((d,),-2.0,device=dev,requires_grad=True)
    t=(x,cond,Wa,Wb,Ws,Wsc,bsc)
    yr=ref(*t); gr=torch.autograd.grad(yr,t,torch.ones_like(yr))
    for be in ("cublas","triton"):
        set_wgrad_backend(be)
        yo=cond_transition_train_fused(*t); go=torch.autograd.grad(yo,t,torch.ones_like(yo))
        worst=min([cos(yo,yr)]+[cos(a,b) for a,b in zip(go,gr)])
        print(f"d={d} M={M} wgrad={be:>6} worst_cos={worst:.5f} {'OK' if worst>=0.999 else 'FAIL'}")
print("DONE")
