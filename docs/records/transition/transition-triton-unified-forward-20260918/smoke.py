import torch
from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton.b2b_residual import transition_b2b_residual
settings.configure(engine_backend='triton',autotune_miss_cap=3)
x=torch.randn(1,129,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
g=torch.ones(128,device='cuda',requires_grad=True);b=torch.zeros_like(g,requires_grad=True)
wa=torch.randn(512,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);wb=torch.randn_like(wa,requires_grad=True);ws=torch.randn(128,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=transition_b2b_residual(x,g,b,wa,wb,ws,1e-5);y.sum().backward();print('PASS',y.shape)
