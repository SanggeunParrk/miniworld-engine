from width_autograd import *
import importlib.util
_spec=importlib.util.spec_from_file_location("all_width_current",R.parent/"trimul_training_current.py");_entry=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_entry)
bidirectional_trimul_cuda=_entry.bidirectional_trimul_cuda
from fixture import setup
import argparse,os
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--length',type=int,required=True);a=p.parse_args();D,N=a.width,a.length
leaves,dy,mask,ds,ref,_,names=setup(D,N);ref=torch.compile(ref,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
y=bidirectional_trimul_cuda(*leaves,mask,ds);grads=torch.autograd.grad(y,leaves,dy);ry=ref(*leaves,mask,ds);rg=torch.autograd.grad(ry,leaves,dy)
checks={}
for name,t,v in zip(names,(y,*grads),(ry,*rg)):
 err=float((t.float()-v.float()).norm()/v.float().norm().clamp_min(1e-20));checks[name]=dict(relative_l2=err,finite=bool(t.isfinite().all()));assert checks[name]['finite'] and err<(.005 if name=='y' else .01),(name,checks[name])
# Two outstanding forward calls must retain separate activations.
y1=bidirectional_trimul_cuda(*leaves,mask,ds);y2=bidirectional_trimul_cuda(*leaves,mask,ds);gg=torch.autograd.grad((y1,y2),leaves,(dy,dy))
errs=[float((g.float()-2*r.float()).norm()/(2*r.float()).norm().clamp_min(1e-20)) for g,r in zip(gg,grads)];assert max(errs)<.004,errs
# PyTorch version checks detect changes between forward and backward.
y3=bidirectional_trimul_cuda(*leaves,mask,ds)
with torch.no_grad():leaves[1].add_(.001)
try:torch.autograd.grad(y3,leaves,dy);raise AssertionError('missing saved tensor version check')
except RuntimeError as e:assert 'modified by an inplace operation' in str(e),str(e)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),checks=checks,multiple_forward_error=errs,version_check=True,complete=True)
(R/f'autograd-D{D}-L{N}.json').write_text(json.dumps(record,indent=2));print('AUTOGRAD PASS',D,N,max(v['relative_l2'] for v in checks.values()),flush=True)
