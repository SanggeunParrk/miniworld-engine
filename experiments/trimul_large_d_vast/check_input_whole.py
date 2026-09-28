import argparse
import json
from short_common import *
from input_whole_tma import WholeInput
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
leaves,dy,mask,ds,ref,triton,names=setup(a.width,384)
r={'D':a.width,'L':384,'complete':False,'trials':[]}
dest=OUT/f'input-whole-D{a.width}.json'
def save():dest.write_text(json.dumps(r,indent=2))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=training(a.width)(leaves,mask,ds,dy);original=plan.dx.reduce_only
 y,g=plan();expected=[t.clone() for t in (y,*g)]
 base,_=capture(plan)
 for l2 in ('128B','256B'):
  op=WholeInput(plan,l2);plan.dx.reduce_only=op
  y,g=plan();torch.cuda.synchronize()
  es={n:error(x,y) for n,x,y in zip(names,(y,*g),expected)}
  row={'l2':l2,'errors':es,'cubin':str(op.cubin),'sha256':sha(op.cubin)}
  if strict(es):
   graph,_=capture(plan);row['times']=paired({'baseline':base,'candidate':graph},75);del graph
   row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us']
  r['trials'].append(row);save();print('INPUT_WHOLE',l2,es,row.get('speedup'),flush=True)
  plan.dx.reduce_only=original
 r['complete']=True;save()
