import argparse
import json
from short_common import *
from front_n128 import build
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
leaves,dy,mask,ds,ref,triton,names=setup(a.width,384)
r={'D':a.width,'L':384,'complete':False,'trials':[], 'sources':{f:sha(Path(__file__).with_name(f)) for f in ('front_n128.py','check_front_n128.py','short_common.py')}}
dest=OUT/f'front-n128-D{a.width}.json'
def save():dest.write_text(json.dumps(r,indent=2))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=training(a.width)(leaves,mask,ds,dy);original=plan.f.front
 y,g=plan();expected=[t.clone() for t in (y,*g)]
 saved_tensors=(original.xn,original.ab,plan.p.input_stats if a.width==256 else plan.pre)
 saved=[t.clone() for t in saved_tensors]
 base,_=capture(plan)
 configs=((1,64,2,2,1,2),(2,64,2,2,1,2)) if a.width!=384 else ((1,64,2,3,1,2),(2,64,2,3,1,2))
 for cfg in configs:
  row={'cfg':cfg};r['trials'].append(row);save();print('BUILD',cfg,flush=True)
  try:op=build(plan,cfg)
  except (RuntimeError,ValueError) as e:
   row['build_error']=str(e)[-1600:];save();print('BUILD_REJECT',str(e)[-200:],flush=True);continue
  plan.f.front=op
  for t in saved_tensors:t.fill_(float('nan'))
  y,g=plan();torch.cuda.synchronize()
  exact=[torch.equal(x,y) for x,y in zip(saved_tensors,saved)]
  es={n:error(x,y) for n,x,y in zip(names,(y,*g),expected)}
  row.update(saved_bitwise=exact,errors=es,cubin=str(op.cubin),sha256=sha(op.cubin),occupancy=op.occupancy)
  if all(exact) and strict(es):
   cg,_=capture(plan);row['times']=paired({'baseline':base,'candidate':cg});del cg
   row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us']
  save();print('N128',cfg,exact,row.get('speedup'),flush=True)
  plan.f.front=original
 r['complete']=True;save()
