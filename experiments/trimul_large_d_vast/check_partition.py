import argparse
import json
from short_common import *
from partition_front import build
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
leaves,dy,mask,ds,ref,triton,names=setup(a.width,384)
r={'D':a.width,'L':384,'complete':False,'trials':[], 'sources':{f:sha(Path(__file__).with_name(f)) for f in ('partition_front.py','check_partition.py','short_common.py')}}
dest=OUT/f'partition-D{a.width}.json'
def save(): dest.write_text(json.dumps(r,indent=2))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=training(a.width)(leaves,mask,ds,dy);original=plan.f.front
 y,g=plan();expected=[t.clone() for t in (y,*g)]
 saved=[t.clone() for t in (original.xn,original.ab,plan.pre)]
 base,_=capture(plan)
 for parts in (2,4,8):
  op=build(plan,parts);plan.f.front=op
  for t in (op.xn,op.ab,plan.pre):t.fill_(float('nan'))
  y,g=plan();torch.cuda.synchronize()
  exact=[torch.equal(x,y) for x,y in zip((op.xn,op.ab,plan.pre),saved)]
  es={n:error(x,y) for n,x,y in zip(names,(y,*g),expected)}
  row={'parts':parts,'saved_bitwise':exact,'errors':es,'cubin':str(op.cubin),'sha256':sha(op.cubin)}
  if all(exact) and strict(es):
   cg,_=capture(plan);row['times']=paired({'baseline':base,'candidate':cg});del cg
   row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us']
  r['trials'].append(row);save();print('PARTITION',parts,exact,row.get('speedup'),flush=True)
  plan.f.front=original
 r['complete']=True;save()
