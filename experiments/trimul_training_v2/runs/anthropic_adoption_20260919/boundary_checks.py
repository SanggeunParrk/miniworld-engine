import copy,json,math,torch
from pathlib import Path
from miniworld_engine.modules import TriangleMultiplication,TriangleAttention,Transition
from miniworld_engine.integrations import anthropic as A
R=Path(__file__).resolve().parent
torch.manual_seed(943)
results=[]
def init(m):
 m=m.cuda().eval()
 for n,p in m.named_parameters():
  p.data=p.data.to(torch.bfloat16 if p.ndim>1 else torch.float32)
  with torch.no_grad():p.normal_(0,1/math.sqrt(p.shape[-1])) if p.ndim>1 else (p.fill_(1) if n.endswith('weight') else p.zero_())
 return m
def error(y,r):return float((y.float()-r.float()).square().mean().sqrt()/r.float().square().mean().sqrt().clamp_min(1e-12))
for kind in ('trimul-out','trimul-in','attn-start','attn-end','transition'):
 for masked in (False,True):
  cls,kw=(TriangleMultiplication,dict(outgoing=kind.endswith('out'),anthropic_row='native_rebuilt')) if kind.startswith('trimul') else ((TriangleAttention,dict(starting=kind.endswith('start'),n_head=4,d_hidden=128,anthropic_row='block:triattn_native')) if kind.startswith('attn') else (Transition,dict(anthropic_row='v2')))
  try:
   m=init(cls(128,implementation='anthropic',**kw));ref=copy.deepcopy(m).float()
   from miniworld_engine.modules.dispatch import KernelBackend
   ref._backend=KernelBackend.PYTORCH
   x=torch.randn(2,384,384,128,device='cuda',dtype=torch.bfloat16);before=x.clone()
   mask=torch.ones(2,384,device='cuda',dtype=torch.bool);mask[0,::7]=False;mask[1,::11]=False
   args=() if kind=='transition' else (mask if masked else None,)
   with torch.no_grad():
    y=m(x,*args);r=ref(x.float(),*args);err=error(y,r);uerr=error(y.float()-x.float(),r-x.float())
    assert torch.equal(x,before),'input mutated'
    assert torch.isfinite(y).all() and err<.03 and uerr<.03,(err,uerr)
    # Weight cache must repack on actual mutation, not just preserve output shape.
    if not masked:
     old=m._anthropic_weights
     m.to_out.weight.mul_(.5) if kind!='transition' else m.squeeze.weight.mul_(.5)
     m(x,*args)
     assert m._anthropic_weights is not old,'stale packed weights'
   results.append(dict(case=kind,masked=masked,status='passed',rel_rms=err,update_rel_rms=uerr))
  except Exception as e:results.append(dict(case=kind,masked=masked,status='failed',reason=str(e)))
  print(results[-1],flush=True)
(R/'boundary-checks.json').write_text(json.dumps(results,indent=2))
assert all(r['status']=='passed' for r in results)
