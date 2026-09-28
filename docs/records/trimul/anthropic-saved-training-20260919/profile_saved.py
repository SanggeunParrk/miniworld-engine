import sys,torch,json
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_saved import front,output,build,_kernel,front_default,output_default,front_candidates,output_candidates
R=Path('/home/psk6950/MiniWorld/runs/anthropic_saved_training_20260919')
n,c,h=768,128,256;kw=dict(device='cuda',dtype=torch.bfloat16)
xn=torch.randn(1,n,n,c,**kw);ws=[torch.randn(c,h,**kw)/c**.5 for _ in range(4)];mask=torch.ones((1,n,n),**kw)
norm=torch.randn(n*n,h,**kw);wp=torch.randn(c,h,**kw)/h**.5;wg=torch.randn(c,c,**kw)/c**.5
res=torch.randn(n*n,c,**kw);ds=(torch.rand(n,c,device='cuda')>.25).bfloat16()/.75
for _ in range(2):front(xn,*ws,pair_mask=mask);output(norm,xn.reshape(n*n,c),wp,wg,res,ds,n)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
front(xn,*ws,pair_mask=mask);output(norm,xn.reshape(n*n,c),wp,wg,res,ds,n)
torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
if 'manifest' in sys.argv:
 records=[]
 for h in (128,256):
  for kind,cfg in [('front',front_default(h)),('output',output_default(h))]:
   k=_kernel(kind,c,h,cfg,0);p=build(kind,c,h,cfg)
   records.append(dict(kind=kind,C=c,H=h,config=cfg,cubin=str(p),attrs=k.attrs(),space_size=len(list(front_candidates(c,h) if kind=='front' else output_candidates(c,h)))))
 (R/'binary-evidence.json').write_text(json.dumps(records,indent=2));print(json.dumps(records),flush=True)
