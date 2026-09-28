import argparse
import json
from short_common import *
from short_weights import attach
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
leaves,dy,mask,ds,ref,triton,names=setup(a.width,384)
r={'D':a.width,'L':384,'complete':False,'trials':[]}
dest=OUT/f'weights-D{a.width}.json'
def save():dest.write_text(json.dumps(r,indent=2))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=training(a.width)(leaves,mask,ds,dy)
 y,g=plan();expected=[t.clone() for t in (y,*g)]
 base,_=capture(plan)
 for splits in (2,4,6,8,12,16):
  op=attach(plan,splits);fast=[]
  for index in op.indices:
   op.index=index
   op();plan.dx.reduce_only();torch.cuda.synchronize()
   es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected[1:]) if n.startswith('dW')}
   row={'splits':splits,'index':index,'errors':es,'strict':strict(es),'algo':list(op.heuristics[index].algo.data)}
   if row['strict']:
    graph,_=capture(op);tm=paired({'op':graph},11)['op']['median_us'];del graph
    row['op_us']=tm;fast.append(row)
   r['trials'].append(row)
  for row in sorted(fast,key=lambda v:v['op_us'])[:2]:
   op.index=row['index'];cy,cg=plan()
   row['full_errors']={n:error(x,y) for n,x,y in zip(names,(cy,*cg),expected)}
   if strict(row['full_errors']):
    graph,_=capture(plan);row['times']=paired({'baseline':base,'candidate':graph},51);del graph
    row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us']
   print('WEIGHT',splits,row['index'],row.get('speedup'),flush=True)
  save()
 r['complete']=True;save()
