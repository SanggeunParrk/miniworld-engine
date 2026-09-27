"""Independent graph timings of training fwd, retained-graph bwd, full step, and raw cuBLAS."""
import argparse,gc,json,socket,statistics
from pathlib import Path
import torch,triton
import torch._functorch.config
# Retained-graph backward timing must not consume donated forward buffers.
torch._functorch.config.donated_buffer=False
from miniworld_engine import settings
from miniworld_engine.kernels.transition.cuda.variants import transition
from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b
from miniworld_engine.kernels.transition.triton.residual import transition_residual
p=argparse.ArgumentParser();p.add_argument('--d',type=int,required=True);p.add_argument('--length',type=int,required=True);a=p.parse_args();root=Path(__file__).parent;d=a.d;m=a.length**2
settings.configure(engine_backend='triton',autotune_miss_cap=24,transition_residual_fusion=True)
records={v:json.loads((root/f'tune-{v}-D{d}.json').read_text()) for v in ('streamed_k','full_k')}
for v in records:records[v]['best_norm']=json.loads((root/'norm-selections.json').read_text())[str(a.d)]['best_norm']
result=dict(node=socket.gethostname(),D=d,L=a.length,mode='static fullgraph compile plus CUDA Graph; autograd.grad, no optimizer or gradient accumulation; donated_buffer=False for retained backward',rows=[])
side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(side)
def bench(f):
 for _ in range(3):f()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=side):
  for _ in range(3):f()
 for _ in range(3):graph.replay()
 times=[]
 for _ in range(3):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
  for _ in range(20):graph.replay()
  end.record();end.synchronize();times.append(start.elapsed_time(end)/60)
 return statistics.median(times)
for arm in ['triton_split','triton:streamed_k','cuda:streamed_k','triton:full_k','cuda:full_k']:
 torch.compiler.reset();gc.collect();torch.cuda.empty_cache();torch.manual_seed(918)
 x=torch.randn(1,a.length,a.length,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 g=(torch.rand(d,device='cuda')+.5).requires_grad_();b=(torch.randn_like(g)*.1).requires_grad_()
 wa=(torch.randn(4*d,d,device='cuda',dtype=x.dtype)*d**-.5).requires_grad_();wb=(torch.randn_like(wa)*d**-.5).requires_grad_();ws=(torch.randn(d,4*d,device='cuda',dtype=x.dtype)*(4*d)**-.5).requires_grad_()
 leaves=(x,g,b,wa,wb,ws);dy=torch.randn_like(x)
 def call(x,g,b,wa,wb,ws):
  if arm=='triton_split':return transition_residual(x,g,b,wa,wb,ws,1e-5)
  backend,v=arm.split(':');r=records[v]
  if backend=='triton':return transition_wide_b2b(x,g,b,wa,wb,ws,1e-5,config=r['best_triton_forward']['config'])
  return transition(x,g,b,wa,wb,ws,variant=v,forward_config=r['best_forward']['config'],backward_config=r['best_backward']['config'],norm_config=r['best_norm']['config'])
 compiled=torch.compile(call,dynamic=False,fullgraph=True)
 def fwd():return compiled(*leaves)
 def train():return torch.autograd.grad(fwd(),leaves,dy)
 actual=fwd();actual_grads=torch.autograd.grad(actual,leaves,dy)
 reference=transition_residual(*leaves,1e-5);reference_grads=torch.autograd.grad(reference,leaves,dy)
 errors={name:((u.float()-v.float()).norm()/v.float().norm().clamp_min(1e-12)).item() for name,u,v in zip(('y','dx','dg','db','dwa','dwb','dws'),(actual,*actual_grads),(reference,*reference_grads))}
 assert max(errors.values())<.02,errors
 del actual,actual_grads,reference,reference_grads
 train();torch.cuda.synchronize();gc.collect()
 base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();out=train();torch.cuda.synchronize()
 peak=torch.cuda.max_memory_allocated();del out
 row=dict(arm=arm,relative_errors=errors,training_base_allocated_bytes=base,training_peak_allocated_bytes=peak,training_peak_extra_bytes=peak-base,training_fwd_ms=bench(fwd),training_total_ms=bench(train))
 y=fwd();row['bwd_ms']=bench(lambda:torch.autograd.grad(y,leaves,dy,retain_graph=True));del y
 with torch.no_grad():row['inference_fwd_ms']=bench(fwd)
 result['rows'].append(row);print(json.dumps(row),flush=True)
 del compiled,leaves,x,g,b,wa,wb,ws,dy;gc.collect();torch.cuda.empty_cache()
 (root/f'breakdown-D{d}-L{a.length}.json').write_text(json.dumps(result,indent=2)+'\n')
# Same dimensions and layouts as the four common backward GEMMs. Isolated timings,
# not a claim their sum equals module backward time or NCU replay duration.
xn=torch.randn(m,d,device='cuda',dtype=torch.bfloat16);dy=torch.randn_like(xn);h=torch.randn(m,4*d,device='cuda',dtype=xn.dtype);dab=torch.randn(m,8*d,device='cuda',dtype=xn.dtype);wa=torch.randn(4*d,d,device='cuda',dtype=xn.dtype);wb=torch.randn_like(wa);ws=torch.randn(d,4*d,device='cuda',dtype=xn.dtype);wab=torch.cat((wa,wb))
result['isolated_common_backward_ms']={k:bench(f) for k,f in [('dh',lambda:torch.mm(dy,ws)),('dWs',lambda:torch.mm(dy.T,h)),('dWab',lambda:torch.mm(dab.T,xn)),('dxn',lambda:torch.mm(dab,wab)),('weight_cat',lambda:torch.cat((wa,wb)))]}
(root/f'breakdown-D{d}-L{a.length}.json').write_text(json.dumps(result,indent=2)+'\n')
