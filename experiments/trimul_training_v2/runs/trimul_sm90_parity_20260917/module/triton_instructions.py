import json
from pathlib import Path
import torch,triton
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import _bidir_front_kernel
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import _output_f567_kernel
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import _input_dual_bwd_kernel
from miniworld_engine.kernels._tiles import tile_grid
root=Path('module/triton-ptx');root.mkdir(exist_ok=True)
kw=dict(device='cuda',dtype=torch.bfloat16)
m=16384;n=128;k=128;h=256;length=128
x=torch.randn(m,k,**kw);w=torch.randn(k,4*h,**kw);l=torch.empty(h,m,**kw);r=torch.empty_like(l);pre=torch.empty(4*h,m,**kw);mask=torch.ones(m,**kw)
compiled={}
compiled['front']=_bidir_front_kernel.fn[(triton.cdiv(m,64),)](x,w,l,r,pre,mask,m,m,K=k,H2=h,shape_key=0,SAVE_PREACT=True,BLOCK_M1=64,BLOCK_K_D=64,BLOCK_K_H2=64,num_warps=8,num_stages=4)
norm=torch.randn(m,h,**kw);wp=torch.randn(n,h,**kw);wg=torch.randn(k,n,**kw);y=torch.empty(m,n,**kw);p=torch.empty_like(y);g=torch.empty_like(y);res=torch.randn_like(y);ds=torch.ones(length,n,**kw)
compiled['f567']=_output_f567_kernel.fn[tile_grid(m,n,64,64)](norm,x,wp,wg,p,g,y,res,ds,m,length,h,k,n,*wp.stride(),*wg.stride(),BLOCK_M1=64,BLOCK_N=64,BLOCK_K=64,GROUP_M=1,shape_key=0,num_warps=4,num_stages=2)
f=torch.randn(8*k,m,**kw).t();wt=wg.t();v=torch.randn(8*k,n,**kw)
compiled['dual_bwd']=_input_dual_bwd_kernel.fn[tile_grid(m,n,64,128)](x,f,wt,v,y,m,k,8*k,n,*x.stride(),*f.stride(),*wt.stride(),*v.stride(),BLOCK_M1=64,BLOCK_N=128,BLOCK_K=64,GROUP_M=1,shape_key=0,num_warps=4,num_stages=3)
report={}
for name,c in compiled.items():
 s=c.asm['ptx'];(root/(name+'.ptx')).write_text(s)
 report[name]={a:sum(a in line for line in s.splitlines()) for a in ('wgmma.mma_async','mma.sync','cp.async.bulk.tensor')}
Path('module/triton-instructions.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
