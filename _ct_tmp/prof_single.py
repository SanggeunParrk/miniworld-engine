import torch, triton
torch.backends.cuda.matmul.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton import composed as C
# pin ONE config each (avoid autotune sweep so ncu captures the real kernel cleanly)
C._expand_swiglu_kernel.configs=[triton.Config({"BLOCK_M":64,"BLOCK_N":64,"BLOCK_K":64},num_warps=4,num_stages=3)]
C._squeeze_gate_kernel.configs=[triton.Config({"BLOCK_M":64,"BLOCK_D":64,"BLOCK_K":64},num_warps=4,num_stages=3)]
g=torch.Generator('cuda').manual_seed(0); f=lambda *s: torch.randn(*s,device='cuda',dtype=torch.float32,generator=g)
M,d,n,dc=768,768,2,384; ND=n*d
x,cond,Wa,Wb,Ws,Wsc,bsc=f(M,d),f(M,dc),f(ND,d)/d**.5,f(ND,d)/d**.5,f(d,ND)/ND**.5,f(d,dc)/dc**.5,torch.full((d,),-2.,device='cuda')
h=C._expand_swiglu(x,Wa,Wb); C._squeeze_gate(h,cond,Ws,Wsc,bsc); torch.cuda.synchronize()  # warmup/compile
# profiled calls:
h=C._expand_swiglu(x,Wa,Wb); C._squeeze_gate(h,cond,Ws,Wsc,bsc); torch.cuda.synchronize()
print("OK")
