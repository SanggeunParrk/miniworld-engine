import argparse,torch
from miniworld_engine.kernels.transition.cuda.variants import extension
p=argparse.ArgumentParser();p.add_argument('--variant',default='full_k');p.add_argument('--d',type=int,default=128);a=p.parse_args();d=a.d
c=dict(bk=(1<<(d-1).bit_length()) if a.variant=='full_k' else 64,bn=32,bo=64,mgroups=1,ngroups=2 if d>=384 else 1,stages=2,min_blocks=1)
e=extension(a.variant,d,c);m=384**2;x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);wa=torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5;wb=torch.randn_like(wa)*d**-.5;dh=torch.randn(m,4*d,device='cuda',dtype=x.dtype)
for _ in range(5):e.gate_backward(x,wa,wb,dh)
torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();e.gate_backward(x,wa,wb,dh);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
