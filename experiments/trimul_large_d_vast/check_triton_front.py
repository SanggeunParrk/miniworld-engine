import json
from short_common import *
from triton_saved_front import TritonFront
leaves,dy,mask,ds,ref,triton,names=setup(512,384)
r={'D':512,'L':384,'complete':False,'trials':[]}
dest=OUT/'triton-front-D512.json'
def save():dest.write_text(json.dumps(r,indent=2))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=training(512)(leaves,mask,ds,dy);original=plan.f.front
 y,g=plan();expected=[t.clone() for t in (y,*g)];saved_tensors=(original.xn,original.ab,plan.pre);saved=[t.clone() for t in saved_tensors]
 base,_=capture(plan)
 for cfg in ((128,64,64,4,3),(128,128,64,4,3),(128,128,64,8,3),(128,128,32,4,3),(256,128,64,8,3)):
  print('BUILD_TRITON',cfg,flush=True);op=TritonFront(plan,*cfg);plan.f.front=op
  for t in saved_tensors:t.fill_(float('nan'))
  try:y,g=plan();torch.cuda.synchronize()
  except Exception as e:
   r['trials'].append({'cfg':cfg,'error':str(e)[-1600:]});save();plan.f.front=original;continue
  exact=[torch.equal(x,y) for x,y in zip(saved_tensors,saved)]
  es={n:error(x,y) for n,x,y in zip(names,(y,*g),expected)}
  row={'cfg':cfg,'saved_bitwise':exact,'errors':es}
  if all(exact) and strict(es):
   graph,_=capture(plan);row['times']=paired({'baseline':base,'candidate':graph},75);del graph
   row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us']
  r['trials'].append(row);save();print('TRITON_FRONT',cfg,exact,es.get('dx'),row.get('speedup'),flush=True)
  plan.f.front=original
 r['complete']=True;save()
