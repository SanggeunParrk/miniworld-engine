import argparse
import copy
import json
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.modules import Transition
from common import capture,paired,rel,identity,wide
from selected import TransitionCandidate,CONFIGS

p=argparse.ArgumentParser()
p.add_argument('--width',type=int,choices=(128,256,384,512),required=True)
p.add_argument('--length',type=int,required=True)
p.add_argument('--repeats',type=int,default=100)
p.add_argument('--no-pytorch',action='store_true')
args=p.parse_args()
d,L=args.width,args.length
out=Path('.bench/transition-wide-local/qualification')
out.mkdir(parents=True,exist_ok=True)
dest=out/f'D{d}-L{L}.json'
torch.manual_seed(811+d+L)
settings.configure(engine_backend='auto',transition_residual_fusion=True,transition_fused_sm90a=True)
base=Transition(d,n=4,implementation='miniworld').cuda().bfloat16()
with torch.no_grad():
    for n,param in base.named_parameters():
        if param.ndim==2:
            param.normal_(std=param.shape[-1]**-.5)
        elif n=='ln_in.weight':
            param.copy_(1+.2*torch.randn_like(param))
        else:
            param.normal_(std=.2)
# D128 is a comparison row using its existing native kernel; the wide
# experiment does not implement or replace that path.
candidate_cls=Transition if d==128 else TransitionCandidate
cand=candidate_cls(d,n=4,implementation='miniworld').cuda().bfloat16()
cand.load_state_dict(base.state_dict())
mods={'baseline':base,'candidate':cand}
if not args.no_pytorch:
    ref=Transition(d,n=4,implementation='pytorch').cuda().bfloat16()
    ref.load_state_dict(base.state_dict())
    mods['pytorch']=ref
x=torch.randn((1,L,L,d),device='cuda',dtype=torch.bfloat16).requires_grad_()
dy=torch.randn_like(x)
names=['y','dx',*dict(base.named_parameters())]

def step(mod):
    y=mod(x)
    outputs=(y,*torch.autograd.grad(y,(x,*mod.parameters()),dy))
    # Keep numerical outputs, not old AccumulateGrad nodes bound to another stream.
    return tuple(t.detach() for t in outputs)

result=dict(D=d,L=L,identity=identity(),config=CONFIGS.get(d,f'existing D{d} CUDA'),complete=False,
            scope='actual module, compiled, fresh full F+B and all five parameter gradients')
def save():
    dest.write_text(json.dumps(result,indent=2))
if d==128:
    from miniworld_engine.kernels.transition.cuda import fused_sm90a
    original_entry=fused_sm90a.transition_fused_sm90a
    calls=[]
    def observed(*a,**kw):
        calls.append(1)
        return original_entry(*a,**kw)
    fused_sm90a.transition_fused_sm90a=observed
    try:
        expected=tuple(t.detach().clone() for t in step(base))
    finally:
        fused_sm90a.transition_fused_sm90a=original_entry
    assert calls,'D128 module did not dispatch to the native kernel'
    result['dispatch_entry']='fused_sm90a.transition_fused_sm90a'
else:
    expected=tuple(t.detach().clone() for t in step(base))
actual=step(cand)
result['candidate_errors']={n:rel(g,w) for n,g,w in zip(names,actual,expected)}
save()
assert max(result['candidate_errors'].values())<1e-4,result['candidate_errors']
del actual,expected
graphs,outputs,compiled={},{},{}
for name,mod in mods.items():
    compiled[name]=torch.compile(mod,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    graphs[name],outputs[name]=capture(lambda:step(compiled[name]))
    # Inductor changes BF16 rounding in the PyTorch reference's fused math;
    # graph replay must match a fresh call to the same compiled program.
    fresh=step(compiled[name])
    err={n:rel(g,w) for n,g,w in zip(names,outputs[name],fresh)}
    result.setdefault('compiled_graph_errors',{})[name]=err
    save()
    assert max(err.values())<1e-5,(name,err)
    del fresh
result['times']=paired(graphs,args.repeats)
print('TIMES',d,L,{k:t['median_ms'] for k,t in result['times'].items()},flush=True)
if 'pytorch' in outputs:
    result['pytorch_vs_candidate_errors']={n:rel(g,w) for n,g,w in zip(names,outputs['candidate'],outputs['pytorch'])}
save()
with torch.no_grad():
    x.mul_(.75).add_(.125)
    dy.mul_(-.875)
    for param in base.parameters():
        param.mul_(.875)
    for name,mod in mods.items():
        if name!='baseline':
            mod.load_state_dict(base.state_dict())
for name,graph in graphs.items():
    graph.replay()
    torch.cuda.synchronize()
    fresh=step(compiled[name])
    err={n:rel(g,w) for n,g,w in zip(names,outputs[name],fresh)}
    result.setdefault('changed_graph_errors',{})[name]=err
    save()
    assert max(err.values())<1e-5,(name,err)
    del fresh
result['changed_candidate_errors']={n:rel(g,w) for n,g,w in zip(names,outputs['candidate'],outputs['baseline'])}
assert max(result['changed_candidate_errors'].values())<1e-4,result['changed_candidate_errors']
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    graphs['candidate'].replay()
    torch.cuda.synchronize()
result['candidate_kernel_names']=sorted({e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA})
if d>=384:
    assert any('_ln_residual_persistent' in name for name in result['candidate_kernel_names'])
result['complete']=True
save()
print('PASS',dest,flush=True)
