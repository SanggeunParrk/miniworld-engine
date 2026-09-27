import argparse,json,torch
from pathlib import Path
from loader import extension
p=argparse.ArgumentParser();p.add_argument('--file',required=True);p.add_argument('--d',type=int);p.add_argument('--production',action='store_true');p.add_argument('--limit',type=int,default=100);a=p.parse_args();r=Path(__file__).parent
rows=json.loads((r/a.file).read_text());count=0
for t in rows:
 if t.get('status') not in ('ok','built') or (a.d and t['D']!=a.d):continue
 d=t['D']
 if a.production:
  from miniworld_engine.kernels.transition.cuda.variants import extension as production
  e=production(t['variant'],d,t['config'])
 else:e=extension(t)
 for m in (65,129,257):
  x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);wa=torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5;wb=torch.randn_like(wa)*d**-.5;ws=torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5;dh=torch.randn(m,4*d,device='cuda',dtype=x.dtype)
  for _ in range(3):
   if t['direction']=='forward':y=e.forward(x,x,wa,wb,ws)
   else:y=e.gate_backward(x,wa,wb,dh)
  torch.cuda.synchronize()
 print('CHECKED',t['D'],t['variant'],t['direction'],t['config'],flush=True);count+=1
 if count>=a.limit:break
print('COMPLETED',count,flush=True)
