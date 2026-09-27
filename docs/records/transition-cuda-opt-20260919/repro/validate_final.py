"""Production entry: reference values, six gradients, identity, compile and graph."""
import argparse,json,torch
from pathlib import Path
from miniworld_engine import settings
from miniworld_engine.kernels.transition.cuda.variants import transition
p=argparse.ArgumentParser();p.add_argument('--d',type=int);p.add_argument('--variant');p.add_argument('--quick',action='store_true');a=p.parse_args();r=Path(__file__).parent;rows=[]
settings.configure(engine_backend='auto',autotune_miss_cap=3)
# Build reference, leaves and compiled autograd on the same non-default stream.
test_stream=torch.cuda.Stream();test_stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(test_stream)
selected=json.loads((r/'final-selections.json').read_text());prior=r.parent/'transition_cuda_variants_20260918';norms=json.loads((prior/'norm-selections.json').read_text())
def relative(x,y):return ((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12)).item()
for d in ([a.d] if a.d else (128,256,384,512)):
 for v in ([a.variant] if a.variant else ('full_k','streamed_k')):
  fc,bc=(next(t['config'] for t in selected if t['D']==d and t['variant']==v and t['direction']==dr) for dr in ('forward','backward'))
  for m in ((129,) if a.quick else (65,129,257)):
   torch.manual_seed(8519+d+m);x=torch.randn(1,m,d,device='cuda',dtype=torch.bfloat16,requires_grad=True);g=torch.rand(d,device='cuda',requires_grad=True);b=torch.randn_like(g,requires_grad=True)
   with torch.no_grad():g[0]=0
   wa=(torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5).requires_grad_();wb=(torch.randn_like(wa)*d**-.5).requires_grad_();ws=(torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5).requires_grad_();leaves=(x,g,b,wa,wb,ws);dy=torch.randn_like(x)
   def call(x):return transition(x,g,b,wa,wb,ws,variant=v,forward_config=fc,backward_config=bc,norm_config=norms[str(d)]['best_norm']['config'])
   xn=torch.nn.functional.layer_norm(x.float(),(d,),g,b,1e-5).bfloat16();aa=xn.float()@wa.float().T;bb=xn.float()@wb.float().T;h=(torch.nn.functional.silu(aa)*bb).bfloat16();ref=(h.float()@ws.float().T).bfloat16()+x;rg=torch.autograd.grad(ref,leaves,dy)
   out=call(x);gg=torch.autograd.grad(out,leaves,dy);errors={k:relative(u,w) for k,u,w in zip(('y','dx','dg','db','dwa','dwb','dws'),(out,*gg),(ref,*rg))};assert max(errors.values())<.02,errors
   if m==129:
    side=torch.cuda.current_stream()
    with torch.cuda.stream(side):
     compiled=torch.compile(call,dynamic=False,fullgraph=True)
     for _ in range(3):cy=compiled(x);cg=torch.autograd.grad(cy,leaves,dy)
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph,stream=side):gy=compiled(x);gr=torch.autograd.grad(gy,leaves,dy)
     graph.replay()
    torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
    assert max(relative(u,w) for u,w in zip((gy,*gr),(cy,*cg)))<.001
   with torch.no_grad():ws.zero_()
   out=call(x);dx,dg,db=torch.autograd.grad(out,(x,g,b),dy);torch.testing.assert_close(out,x,rtol=0,atol=0);torch.testing.assert_close(dx,dy,rtol=0,atol=0);assert torch.count_nonzero(dg)==torch.count_nonzero(db)==0
   rows.append(dict(D=d,variant=v,M=m,errors=errors,compile_graph=m==129,identity='exact'));(r/f'validation-{a.d}-{a.variant}.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS',json.dumps(rows[-1]),flush=True)
print('ALL PASS',len(rows),flush=True)
