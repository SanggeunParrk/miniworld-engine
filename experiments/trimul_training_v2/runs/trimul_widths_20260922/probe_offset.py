import torch,triton,sys,json,os
from pathlib import Path
R=Path(__file__).resolve().parent
variant=sys.argv[1];mod=__import__('dual_'+variant);k=mod._input_dual_bwd_kernel.fn
m=768**2;kg=512;kp=4096;n=512;bm=32;bn=64
with torch.no_grad():
 # Only one output tile is launched; input strides preserve the failing full shape.
 f=torch.full((kp,m),.03125,device='cuda',dtype=torch.bfloat16).t();g=torch.zeros((bm,kg),device='cuda',dtype=torch.bfloat16);w=torch.zeros((kg,n),device='cuda',dtype=torch.bfloat16);v=torch.full((kp,n),1/4096,device='cuda',dtype=torch.bfloat16);out=torch.empty((bm,n),device='cuda',dtype=torch.bfloat16)
 k[(1,)](g,f,w,v,out,m,kg,kp,n,*g.stride(),*f.stride(),*w.stride(),*v.stride(),BLOCK_M1=bm,BLOCK_N=bn,BLOCK_K=64,GROUP_M=1,shape_key=0,num_warps=4,num_stages=2)
 torch.cuda.synchronize();got=out[:bm,:bn];assert torch.equal(got,torch.full_like(got,.03125)),got
 print('OFFSET_PROBE_PASS',variant,flush=True)
