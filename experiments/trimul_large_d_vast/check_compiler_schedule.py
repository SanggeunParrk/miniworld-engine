import argparse
import json
from short_common import *
from compiler_schedule import build
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
leaves,dy,mask,ds,ref,triton,names=setup(a.width,384)
r={'D':a.width,'L':384,'complete':False,'trials':[]}
dest=OUT/f'compiler-schedule-D{a.width}.json'
def save():dest.write_text(json.dumps(r,indent=2))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=training(a.width)(leaves,mask,ds,dy)
 y,g=plan();expected=[t.clone() for t in (y,*g)]
 saved_tensors=(plan.f.front.xn,plan.f.front.ab,plan.p.input_stats if a.width==256 else plan.pre)
 saved=[t.clone() for t in saved_tensors];base,_=capture(plan)
 for comp in (('front',) if a.width==256 else ('front','contract')):
  old=plan.f.front if comp=='front' else plan.contract_gp
  for level in (0,2,8,10):
   print('BUILD_SCHEDULE',comp,level,flush=True)
   op=build(plan,comp,level)
   if comp=='front':plan.f.front=op
   else:plan.contract_gp=op
   y,g=plan();torch.cuda.synchronize()
   es={n:error(x,y) for n,x,y in zip(names,(y,*g),expected)}
   row={'component':comp,'level':level,'errors':es,'saved_bitwise':[torch.equal(x,y) for x,y in zip(saved_tensors,saved)],'registers':op.registers,'local_bytes':op.local_bytes,'cubin':str(op.cubin),'sha256':sha(op.cubin)}
   if strict(es) and all(row['saved_bitwise']):
    graph,_=capture(plan);row['times']=paired({'baseline':base,'candidate':graph},75);del graph
    row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us']
   r['trials'].append(row);save();print('SCHEDULE',comp,level,row.get('speedup'),op.registers,op.local_bytes,flush=True)
   if comp=='front':plan.f.front=old
   else:plan.contract_gp=old
 r['complete']=True;save()
