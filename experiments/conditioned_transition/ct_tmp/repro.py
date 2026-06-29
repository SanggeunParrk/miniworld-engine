import torch
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.inference import cond_transition_inference
dev="cuda"; g=torch.Generator(dev).manual_seed(0)
f=lambda *s: torch.randn(*s,device=dev,dtype=torch.float32,generator=g)
M,d,n,dc=2048,128,2,384
x=f(M,d);cond=f(M,dc);Wa=f(n*d,d);Wb=f(n*d,d);Ws=f(d,n*d);Wsc=f(d,dc);bsc=torch.full((d,),-2.0,device=dev)
y=cond_transition_inference(x,cond,Wa,Wb,Ws,Wsc,bsc)
print("OK", y.shape)
