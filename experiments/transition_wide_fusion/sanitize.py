"""New-kernel sanitizer entry; full mode also exercises actual module F+B."""
import argparse
import gc
import json
from pathlib import Path
import torch
from common import inputs,rel,capture,identity,_transition_ln_bwd
from ln_residual import ln_residual
from selected import CONFIGS,TransitionCandidate
from miniworld_engine.modules import Transition
from miniworld_engine import settings

p=argparse.ArgumentParser()
p.add_argument('--width',type=int,required=True)
p.add_argument('--length',type=int,default=768)
p.add_argument('--full',action='store_true')
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
d,L=args.width,args.length
# Eager autograd may keep leaf nodes alive across calls. Build the fixtures and
# capture on one side stream so their stream identity cannot cross to default.
work_stream=torch.cuda.Stream()
torch.cuda.set_stream(work_stream)
def capture_here(fn):
    for _ in range(3):
        outputs=fn()
        del outputs
        gc.collect()
        torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=work_stream):
        outputs=fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph,outputs
v=inputs(d,L)
x,gamma,beta,wa,wb,ws,dy=v
result=dict(D=d,L=L,identity=identity(),full=args.full,complete=False)
if args.full:
    settings.configure(engine_backend='auto',transition_residual_fusion=True,transition_fused_sm90a=True)
    base=Transition(d,n=4,implementation='miniworld').cuda().bfloat16()
    cand=TransitionCandidate(d,n=4,implementation='miniworld').cuda().bfloat16()
    with torch.no_grad():
        for param,value in zip(base.parameters(),(gamma,beta,wa,wb,ws)):
            param.copy_(value)
    cand.load_state_dict(base.state_dict())
    # Match the measured compiled module path. AOTAutograd releases saved
    # intermediates on the same schedule as the qualification benchmark.
    x=x.reshape(1,L,L,d)
    dy=dy.reshape_as(x)
    x.requires_grad_()
    base=torch.compile(base,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    cand=torch.compile(cand,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    result['compiled']=True
    def run(mod):
        y=mod(x)
        outputs=(y,*torch.autograd.grad(y,(x,*mod.parameters()),dy))
        return tuple(t.detach() for t in outputs)
    # Reference outputs live on the CPU, and are computed BEFORE capture.
    # This avoids keeping a graph pool while allocating another full F+B.
    expected=tuple(t.cpu() for t in run(base))
    original_x=x.detach().clone()
    original_dy=dy.clone()
    original_params=[p.detach().clone() for p in cand.parameters()]
    def change():
        with torch.no_grad():
            x.mul_(.75).add_(.1)
            dy.mul_(-.5)
            for param in cand.parameters():
                param.mul_(.875)
    change()
    changed_expected=tuple(t.cpu() for t in run(cand))
    with torch.no_grad():
        x.copy_(original_x)
        dy.copy_(original_dy)
        for param,original in zip(cand.parameters(),original_params):
            param.copy_(original)
    del original_x,original_dy,original_params,base
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print('REFERENCES_READY',torch.cuda.memory_allocated(),flush=True)
    graph,outputs=capture_here(lambda:run(cand))
    result['errors']=[rel(g.cpu(),w) for g,w in zip(outputs,expected)]
    assert max(result['errors'])<1e-4,result['errors']
    del expected
    change()
    graph.replay()
    torch.cuda.synchronize()
    result['changed_graph_errors']=[rel(g.cpu(),w) for g,w in zip(outputs,changed_expected)]
    assert max(result['changed_graph_errors'])<1e-5,result['changed_graph_errors']

else:
    # Random dXn probes the epilogue independently of all unchanged GEMMs.
    dxn=torch.randn_like(x)
    xx=x.float()
    rs=torch.rsqrt(xx.var(-1,unbiased=False)+1e-5)
    c1=xx.mean(-1)*rs
    del xx
    def run():
        return ln_residual(dxn,x,dy,gamma,rs,c1,**CONFIGS[d])[:3]
    expected=tuple(t.clone() for t in run())
    graph,outputs=capture_here(run)
    result['graph_errors']=[rel(g,w) for g,w in zip(outputs,expected)]
    assert max(result['graph_errors'])==0,result['graph_errors']
    dxn.mul_(-.5)
    dy.mul_(.875)
    gamma.mul_(.75)
    graph.replay()
    torch.cuda.synchronize()
    fresh=run()
    result['changed_graph_errors']=[rel(g,w) for g,w in zip(outputs,fresh)]
    assert max(result['changed_graph_errors'])==0,result['changed_graph_errors']
    # Zero upstream gradients must give exact zero for every output.
    dxn.zero_()
    dy.zero_()
    fresh=run()
    assert all(torch.count_nonzero(t)==0 for t in fresh)
result['complete']=True
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(result,indent=2))
print('PASS',args.out,flush=True)
