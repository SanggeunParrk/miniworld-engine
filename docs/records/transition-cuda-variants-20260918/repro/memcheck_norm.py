import torch
from miniworld_engine.kernels.transition.cuda.variants import norm_extension
x=torch.randn(129,128,device='cuda',dtype=torch.bfloat16);g=torch.ones(128,device='cuda');b=torch.zeros_like(g)
e=norm_extension();y,mu,rs=e.forward(x,g,b,1e-5,4);e.backward(y,x,g,mu,rs,y,4,4,8,256,4);torch.cuda.synchronize()
print('PASS norm',flush=True)
for line in open('/proc/self/maps'):
 if 'libcudart' in line:print(line.strip())
