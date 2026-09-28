"""Same-runtime direct Triton comparison; scopes use separate graph pools."""
import argparse
import gc
import json
from short_common import *
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--kind',default='baseline');a=p.parse_args()
leaves,dy,mask,ds,ref,triton,names=setup(a.width,384)
import torch._functorch.config as fc
fc.donated_buffer=False
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
r={'D':a.width,'L':384,'kind':a.kind,'complete':False,'times':{},'script_sha256':sha(__file__)}
dest=OUT/f'vs-triton-{a.kind}-D{a.width}.json'
def save():dest.write_text(json.dumps(r,indent=2))
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=training(a.width)(leaves,mask,ds,dy)
 if a.kind=='split4':
  from input_split import attach
  attach(plan,4,0)
 elif a.kind.startswith('partition'):
  from partition_front import build
  plan.f.front=build(plan,int(a.kind.replace('partition','')))
 cy,cg=plan();expected=[t.clone() for t in (cy,*cg)]
 tls=tuple(t.detach().clone().requires_grad_(True) for t in leaves)
 compiled=torch.compile(triton,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
 def tfwd():
  with torch.enable_grad():return compiled(*tls,mask,ds)
 def tfull():
  with torch.enable_grad():
   y=tfwd();return y,torch.autograd.grad(y,tls,dy)
 bs=torch.cuda.Stream();bs.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(bs):
  sy=tfwd();ty,tg=tfull()
 torch.cuda.current_stream().wait_stream(bs)
 frozen=[ty.clone(),*[t.clone() for t in tg]]
 es={n:error(x,y) for n,x,y in zip(names,expected,frozen)}
 r['triton_errors']=es;save()
 assert all(v<(.005 if n=='y' else .01) for n,v in es.items()),es
 def tbwd():
  with torch.enable_grad():return torch.autograd.grad(sy,tls,dy,retain_graph=True)
 for scope,native,baseline in (('backward',plan.backward,tbwd),('full',plan,tfull)):
  bs.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(bs):
   for _ in range(3):baseline()
  torch.cuda.current_stream().wait_stream(bs)
  tgph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(tgph,stream=bs):to=baseline()
  tgph.replay();torch.cuda.synchronize()
  actual=to if scope=='backward' else (to[0],*to[1])
  wanted=frozen[1:] if scope=='backward' else frozen
  ge=[error(x,y) for x,y in zip(actual,wanted)];assert max(ge)<5e-6,ge
  ng,_=capture(native)
  r['times'][scope]=paired({'triton':tgph,'native':ng},75)
  r.setdefault('speedup',{})[scope]=r['times'][scope]['triton']['median_us']/r['times'][scope]['native']['median_us']
  r.setdefault('graph_errors',{})[scope]=ge;save()
  print('TRITON',a.width,a.kind,scope,r['speedup'][scope],flush=True)
  del tgph,ng,to
  if scope=='backward':sy=None
  torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
 r['complete']=True;save()
