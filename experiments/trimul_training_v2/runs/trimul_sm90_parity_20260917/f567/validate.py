import json, torch
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import output_f567_impl, _COMPILE_CACHE
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import _output_f567_kernel

def triton(a,x,wp,wg,r,d,L,c):
 m,kp=a.shape;kg,n=wg.shape
 y,p,g=(torch.empty_like(r) for _ in range(3))
 _output_f567_kernel.fn[((m+c['BLOCK_M1']-1)//c['BLOCK_M1']*((n+c['BLOCK_N']-1)//c['BLOCK_N']),)](a,x,wp,wg,p,g,y,r,d,m,L,kp,kg,n,*wp.stride(),*wg.stride(),**c,shape_key=0)
 return y,p,g

def inputs(m,kp,kg,n,L,layout):
 a=torch.randn(m,kp,device='cuda',dtype=torch.bfloat16)*.2
 x=torch.randn(m,kg,device='cuda',dtype=torch.bfloat16)*.2
 wp=torch.randn(n,kp,device='cuda',dtype=torch.bfloat16)*.2
 wg=torch.randn(n,kg,device='cuda',dtype=torch.bfloat16).t()*.2
 if layout: wp=wp.t().contiguous().t(); wg=wg.contiguous()
 r=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
 d=(torch.rand(L,n,device='cuda')>.2).bfloat16()*1.25
 return a,x,wp,wg,r,d,L

def main():
 torch.manual_seed(42); results=[]
 configs=[(64,32,16,4,4,2),(64,64,32,2,4,3),(64,128,64,1,4,4),(128,32,32,8,8,2),(128,64,64,2,8,3),(64,64,128,1,4,2)]
 for shape in [(513,256,128,128,17),(135,80,152,96,13)]:
  for layout in [0,1]:
   for vals in configs:
    c=dict(zip(('BLOCK_M1','BLOCK_N','BLOCK_K','GROUP_M','num_warps','num_stages'),vals)); inp=inputs(*shape,layout)
    y=output_f567_impl(*inp,c); ref=triton(*inp,c)
    torch.cuda.synchronize()
    errs=[((v.float()-r.float()).norm()/r.float().norm()).item() for v,r in zip(y,ref)]
    assert max(errs)<1.e-4,(shape,layout,c,errs)
    row=dict(shape=shape,layout=layout,config=c,rel_l2=errs); results.append(row); print(json.dumps(row),flush=True)
 Path(__file__).with_name('validation.json').write_text(json.dumps(results,indent=2))
 print('ALL PASS',len(results),flush=True)
 for fn in list(_COMPILE_CACHE.values())[:1]:
  print('COMPILEDATTR',dir(fn),flush=True)
if __name__=='__main__':main()
