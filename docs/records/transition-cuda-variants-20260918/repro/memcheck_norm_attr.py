from pathlib import Path
from hashlib import sha256
import torch
from miniworld_engine.kernels._nvcc import ensure_cuda_home,host_flags,gencodes,load_extension
ensure_cuda_home();src=Path(__file__).with_name('norm_attr_probe.cu')
e=load_extension(name='norm_attr_probe_'+sha256(src.read_bytes()).hexdigest()[:12],sources=[str(src)],extra_cflags=['-std=c++17'],extra_cuda_cflags=[*host_flags(),'-std=c++17','-O3','--use_fast_math','-lineinfo',*gencodes('90a')],verbose=False)
x=torch.randn(129,128,device='cuda',dtype=torch.bfloat16);g=torch.ones(128,device='cuda');b=torch.zeros_like(g)
y,mu,rs=e.forward(x,g,b,1e-5,4);e.backward(y,x,g,mu,rs,y,4,4,8,256,4);torch.cuda.synchronize();print('PASS norm attributes',flush=True)
