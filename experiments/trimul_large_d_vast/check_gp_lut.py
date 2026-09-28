import argparse
import json
from short_common import *
from gp_lut import LookupGP
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
leaves,dy,mask,ds,ref,triton,names=setup(a.width,384)
r={'D':a.width,'L':384,'complete':False,'trials':[]}
dest=OUT/f'gp-lut-D{a.width}.json'
def save():dest.write_text(json.dumps(r,indent=2))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=training(a.width)(leaves,mask,ds,dy);original=plan.contract_gp
 y,g=plan();expected=[t.clone() for t in (y,*g)];gp=plan.p.gp_all.clone()
 base,_=capture(plan)
 for kind in ('shared','global'):
  print('BUILD_LUT',kind,flush=True);op=LookupGP(plan,kind);plan.contract_gp=op
  plan.p.gp_all.fill_(float('nan'));y,g=plan();torch.cuda.synchronize()
  es={n:error(x,y) for n,x,y in zip(names,(y,*g),expected)}
  row={'kind':kind,'gp_bitwise':torch.equal(plan.p.gp_all,gp),'all_patterns_exact':op.all_patterns_exact,'errors':es,'occupancy':op.occupancy,'registers':op.registers,'local_bytes':op.local_bytes,'cubin':str(op.cubin),'sha256':sha(op.cubin)}
  if row['gp_bitwise'] and strict(es):
   graph,_=capture(plan);row['times']=paired({'baseline':base,'candidate':graph},75);del graph
   row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us']
  r['trials'].append(row);save();print('LUT',kind,row['gp_bitwise'],row.get('speedup'),row['local_bytes'],flush=True)
  plan.contract_gp=original
 r['complete']=True;save()
