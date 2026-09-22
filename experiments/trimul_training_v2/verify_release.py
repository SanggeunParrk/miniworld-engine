"""Relocated-checkout GPU smoke: independent PyTorch formula, every gradient."""
import sys, json, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/trimul_cuda_widths_opt_20260923'))
# A packaged runner must never fall back to the original workspaces.
def audit(event,args):
 if event=='open' and isinstance(args[0],str):
  path=args[0]
  if path.startswith(('/home/psk6950/MiniWorld/runs/','/home/psk6950/miniworld-engine-tbwd/','/home/psk6950/miniworld-engine-k1k3/')):
   raise RuntimeError('Unpackaged dependency: '+path)
sys.addaudithook(audit)
import torch
from fixture import setup
from trimul_training_current import bidirectional_trimul_cuda
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
D=(64,128,256,384,512)[a.index//2];L=(384,768)[a.index%2]
leaves,dy,mask,ds,ref,_,names=setup(D,L)
y=bidirectional_trimul_cuda(*leaves,mask,ds);grads=torch.autograd.grad(y,leaves,dy)
ref=torch.compile(ref,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
ry=ref(*leaves,mask,ds);rg=torch.autograd.grad(ry,leaves,dy)
checks={}
for name,t,v in zip(names,(y,*grads),(ry,*rg)):
 error=float((t.float()-v.float()).norm()/v.float().norm().clamp_min(1e-20))
 checks[name]=error
 assert bool(t.isfinite().all()) and error<(.005 if name=='y' else .01),(name,error)
print('RELEASE_VALIDATION '+json.dumps(dict(D=D,L=L,checks=checks,pass_all=True)),flush=True)
