import argparse, json, types
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.modules import BidirectionalTriangleMultiplication as Module

p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=128);p.add_argument('--compile',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset())
torch.manual_seed(72)
base=Module(128,implementation='triton').cuda().bfloat16().train()
with torch.no_grad():
    for name,value in base.named_parameters():
        if 'ln_' not in name:value.normal_(std=128**-.5)
state=base.state_dict();L=a.length
x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
dy=torch.randn_like(x);mask=torch.rand(1,L,device='cuda')>.2
scale=(torch.rand(1,1,L,128,device='cuda')>.25).to(x.dtype)/.75
report={'length':L,'compiled':a.compile,'comparisons':[],'passed':False}
for zero in (False,True):
    drop=torch.zeros_like(scale) if zero else scale
    reference=None
    for names in ((),('front',),('f567',),('dual_bwd',),('front','f567','dual_bwd')):
        settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(names))
        model=Module(128,implementation='triton').cuda().bfloat16().train();model.load_state_dict(state)
        def fixed_scale(self,pair,p):return drop
        model._make_drop_row_scale=types.MethodType(fixed_scale,model)
        fn=torch.compile(model,dynamic=False,fullgraph=True,options={'triton.cudagraphs':False}) if a.compile else model
        inp=x.detach().clone().requires_grad_(True)
        y=fn(inp,mask);y.backward(dy)
        outputs={'y':y,'dx':inp.grad,**{n:v.grad for n,v in model.named_parameters()}}
        assert all(v is not None and torch.isfinite(v).all() for v in outputs.values())
        if reference is None:reference={k:v.detach().float().clone() for k,v in outputs.items()}
        errors={k:float((v.float()-reference[k]).norm()/reference[k].norm().clamp_min(1e-8)) for k,v in outputs.items()}
        row={'kernels':names,'zero_scale':zero,'relative_l2':errors,'max_relative_l2':max(errors.values())}
        report['comparisons'].append(row);print(json.dumps(row),flush=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        assert max(errors.values())<1e-4,row
        if zero:
            assert torch.equal(y,inp) and torch.equal(inp.grad,dy)
            assert all(torch.count_nonzero(v.grad)==0 for v in model.parameters())
report['passed']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
