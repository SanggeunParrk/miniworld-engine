import argparse,json
from pathlib import Path
import torch
from miniworld_engine import settings
from comparison import matched

p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
settings.configure(engine_backend='auto',autotune_miss_cap=24)
root=Path(__file__).parent;d=a.width
configs={b:json.loads((root/f'selected{suffix}-D{d}.json').read_text()) for b,suffix in [('cuda','-cuda'),('triton_b2b','')]}
torch.manual_seed(951)
x=torch.randn(1,256,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
g=torch.rand(d,device='cuda',requires_grad=True);beta=torch.randn_like(g,requires_grad=True)
wa=(torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16)/d**.5).requires_grad_()
wb=(torch.randn_like(wa)/d**.5).requires_grad_()
ws=(torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)/d**.5).requires_grad_()
with torch.no_grad():g[0]=0
leaves=(x,g,beta,wa,wb,ws);dy=torch.randn_like(x);result={}
for mode in ('inference','training'):
 observed={}
 for backend in ('cuda','triton_b2b'):
    with torch.set_grad_enabled(mode=='training'):
        y=matched(*leaves,1e-5,backend=backend,config=configs[backend][mode]['config'])
        grads=torch.autograd.grad(y,leaves,dy) if mode=='training' else ()
    observed[backend]=(y,*grads)
 errors=[]
 for left,right in zip(observed['cuda'],observed['triton_b2b']):
    rel=((left.float()-right.float()).norm()/left.float().norm().clamp_min(1e-12)).item()
    assert torch.isfinite(right).all() and rel<.01,rel
    errors.append(rel)
 result[mode]=errors
with torch.no_grad():ws.zero_()
for backend in ('cuda','triton_b2b'):
 y=matched(*leaves,1e-5,backend=backend,config=configs[backend]['training']['config'])
 dx,dg,db=torch.autograd.grad(y,(x,g,beta),dy)
 torch.testing.assert_close(y,x,atol=0,rtol=0)
 torch.testing.assert_close(dx,dy,atol=0,rtol=0)
 assert torch.count_nonzero(dg)==0 and torch.count_nonzero(db)==0
result['zero_projection_residual']='exact'
(root/f'validation-D{d}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
