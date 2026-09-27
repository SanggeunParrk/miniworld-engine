import sys,torch,json,gc
from pathlib import Path
sys.path.insert(0,'/home/psk6950/MiniWorld/runs/anthropic_trimul_training_20260919')
from bench_k3 import graph,paired
from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_saved import front,output,_kernel,front_smem,output_smem
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import bidir_front_triton
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train
R=Path('/home/psk6950/MiniWorld/runs/anthropic_saved_training_20260919')
records=json.loads((R/"component-times.json").read_text()) if (R/"component-times.json").exists() else []
for n in [384,768]:
 c,h=128,256;kw=dict(device='cuda',dtype=torch.bfloat16)
 torch.manual_seed(85)
 xn=torch.randn(1,n,n,c,**kw);ws=[torch.randn(c,h,**kw)/c**.5 for _ in range(4)]
 m=(torch.rand(1,n,n,device='cuda')>.2).bfloat16()
 norm=torch.randn(n*n,h,**kw);wp=torch.randn(c,h,**kw)/h**.5;wg=torch.randn(c,c,**kw)/c**.5
 ds=(torch.rand(n,c,device='cuda')>.25).bfloat16()/.75;res=torch.randn(n*n,c,**kw)
 for kind in ['front','output']:
  if any(r['N']==n and r['kind']==kind for r in records):continue
  if kind=='front':
   configs=[(2,64,4,2,0),(2,64,8,2,0),(1,128,4,2,0),(4,64,4,2,0),(2,64,6,1,0),(2,64,4,1,1)]
   ref=lambda:bidir_front_triton(xn,*ws,pair_mask=m)
   fn=lambda cfg:front(xn,*ws,pair_mask=m,config=cfg)
  else:
   configs=[(2,64,4,1,232,1),(1,64,6,1,232,1),(1,128,4,2,232,1),(1,64,4,1,232,1),(2,64,4,1,240,1),(2,64,4,2,232,1)]
   ref=lambda:output_f567_train(norm,xn.reshape(n*n,c),wp,wg,res,ds,n)
   fn=lambda cfg:output(norm,xn.reshape(n*n,c),wp,wg,res,ds,n,config=cfg)
  if kind=='output':configs=[cfg[:5] for cfg in configs]  # remove obsolete LN scheduling axis
  gs={'triton':graph(ref)};attrs={}
  for cfg in configs:
   (front_smem if kind=='front' else output_smem)(c,h,cfg)
   key=','.join(map(str,cfg));gs[key]=graph(lambda cfg=cfg:fn(cfg))
   a,b=gs[key][1][0],gs['triton'][1][0]
   err=((a.float()-b.float()).norm()/b.float().norm()).item();assert err<.003,(cfg,err)
   attrs[key]=_kernel(kind,c,h,cfg,0).attrs()
   print('BUILT',kind,n,cfg,attrs[key],flush=True)
  times=paired(gs,reps=20,rounds=6)
  row=dict(N=n,H=h,kind=kind,times=times,attrs=attrs);records.append(row)
  (R/'component-times.json').write_text(json.dumps(records,indent=2))
  print('RESULT',json.dumps(row),flush=True)
  del gs;gc.collect();torch.cuda.empty_cache()
