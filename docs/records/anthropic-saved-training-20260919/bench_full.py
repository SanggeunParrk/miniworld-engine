import sys,torch,json,gc
from pathlib import Path
sys.path.insert(0,'/home/psk6950/MiniWorld/runs/anthropic_trimul_training_20260919')
from check_full import setup,rel
from bench_k3 import graph,paired
R=Path('/home/psk6950/MiniWorld/runs/anthropic_saved_training_20260919')
records=[]
for n in (384,768):
 leaves,call,mask,ds=setup(n,'bidir');dy=torch.randn_like(leaves[0]);gs={};saved={}
 for backend in ('triton','anthropic_cuda','anthropic_saved'):
  shapes=[]
  def pack(t):shapes.append((tuple(t.shape),str(t.dtype)));return t
  with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):y=call(backend)
  saved[backend]=shapes
  del y  # release eager autograd nodes before capture on the side stream
  def train(backend=backend):
   y=call(backend);return y,torch.autograd.grad(y,leaves,dy)
  gs[backend+'_fwd']=graph(lambda backend=backend:call(backend))
  gs[backend+'_train']=graph(train)
 assert saved['triton']==saved['anthropic_saved']
 yr,rg=gs['triton_train'][1];y,g=gs['anthropic_saved_train'][1]
 errors=[rel(a,b) for a,b in zip((y,*g),(yr,*rg))];assert max(errors)<.003,errors
 times=paired(gs,reps=12,rounds=8)
 row=dict(N=n,B=1,C=128,H=256,dropout=.25,save_policy_identical=True,relative_l2=errors,times=times,saved_tensor_shapes=saved['anthropic_saved'])
 records.append(row);(R/'full-times.json').write_text(json.dumps(records,indent=2))
 print('RESULT',json.dumps(row),flush=True)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  y=call('anthropic_saved');torch.cuda.synchronize()
 p.export_chrome_trace(str(R/f'fwd-L{n}.json'))
 del gs,leaves,call,mask,ds,dy,y,yr,g,rg;gc.collect();torch.cuda.empty_cache()
