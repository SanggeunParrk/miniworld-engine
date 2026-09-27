import json,types,torch
from pathlib import Path
from baseline_loader import load
import miniworld_engine.kernels.transition.cuda.variants as n
r=Path(__file__).parent;prior=r.parent/'transition_cuda_variants_20260918';selected=json.loads((r/'final-selections.json').read_text());norms=json.loads((prior/'norm-selections.json').read_text());production=n.extension;rows=[]
relative=lambda a,b:((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()
for d in (128,256,384,512):
 for v in ('full_k','streamed_k'):
  t=json.loads((prior/f'tune-{v}-D{d}.json').read_text());old=types.SimpleNamespace(forward=load(t['best_forward']['extension']).forward,gate_backward=load(t['best_backward']['extension']).gate_backward)
  fc,bc=(next(x['config'] for x in selected if x['D']==d and x['variant']==v and x['direction']==q) for q in ('forward','backward'))
  torch.manual_seed(20919);x=torch.randn(1,129,d,device='cuda',dtype=torch.bfloat16,requires_grad=True);g=torch.rand(d,device='cuda',requires_grad=True);b=torch.randn_like(g,requires_grad=True);wa=(torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5).requires_grad_();wb=(torch.randn_like(wa)*d**-.5).requires_grad_();ws=(torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5).requires_grad_();leaves=(x,g,b,wa,wb,ws);dy=torch.randn_like(x)
  def call():return n.transition(*leaves,variant=v,forward_config=fc,backward_config=bc,norm_config=norms[str(d)]['best_norm']['config'])
  n.extension=lambda *args:old;ref=call();rg=torch.autograd.grad(ref,leaves,dy)
  n.extension=production;out=call();gr=torch.autograd.grad(out,leaves,dy);errors={k:relative(a,b) for k,a,b in zip(('y','dx','dg','db','dwa','dwb','dws'),(out,*gr),(ref,*rg))};assert max(errors.values())<=.0001,errors
  row=dict(D=d,variant=v,relative_errors=errors,bitwise=all(torch.equal(a,b) for a,b in zip((out,*gr),(ref,*rg))));rows.append(row);print('PASS',json.dumps(row),flush=True)
(r/'baseline-gradient-parity.json').write_text(json.dumps(rows,indent=2)+'\n')
