import torch
from miniworld_engine.kernels.transition.cuda.variants import extension
x=torch.randn(129,128,device='cuda',dtype=torch.bfloat16);wa=torch.randn(512,128,device='cuda',dtype=x.dtype);wb=torch.randn_like(wa);ws=torch.randn(128,512,device='cuda',dtype=x.dtype);dh=torch.randn(129,512,device='cuda',dtype=x.dtype)
c=dict(bk=128,bn=32,bo=64,mgroups=1,ngroups=1,stages=2,min_blocks=1)
e=extension('full_k',128,c);e.forward(x,x,wa,wb,ws);e.gate_backward(x,wa,wb,dh);torch.cuda.synchronize();print('PASS native core',flush=True)
