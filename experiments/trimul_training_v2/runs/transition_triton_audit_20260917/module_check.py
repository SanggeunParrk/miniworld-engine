import copy,json,torch
from pathlib import Path
from miniworld_engine import settings
settings.configure(engine_backend='triton',compile_wrap='custom_op')
from miniworld_engine.modules import Transition
from triton.testing import do_bench_cudagraph
results=[]
stream=torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
 for dtype,shape in [(torch.bfloat16,(1,384,384,128)),(torch.bfloat16,(1,384,384)),(torch.bfloat16,(1,768,768)),(torch.float32,(1,37,96))]:
  torch.manual_seed(71);d=shape[-1]
  m=Transition(d,implementation='triton').cuda().to(dtype)
  with torch.no_grad():
   for v in m.parameters():
    if v.ndim==2:v.normal_(std=d**-.5)
  x=torch.randn(shape,device='cuda',dtype=dtype,requires_grad=True);dy=torch.randn_like(x)
  def grads():return [x.grad.clone()]+[p.grad.clone() for p in m.parameters()]
  def clear():
   m.zero_grad(set_to_none=True);x.grad=None
  m.train();y=m(x);y.backward(dy);expected=[y.detach(),*grads()];clear()
  compiled=torch.compile(m,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
  for _ in range(3):
   clear();y=compiled(x);y.backward(dy)
  actual=[y.detach(),*grads()]
  errors=[]
  for a,b in zip(actual,expected):
   err=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12));errors.append(err)
   assert err < (1e-3 if dtype==torch.bfloat16 else 1e-5),errors
  clear();torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
   y=compiled(x);y.backward(dy)
  graph.replay();graph.replay();torch.cuda.synchronize()
  for a,b in zip([y.detach(),*grads()],expected):
   err=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
   assert err < (1e-3 if dtype==torch.bfloat16 else 1e-5),err
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(100):graph.replay()
  end.record();end.synchronize();train=start.elapsed_time(end)/100
  m.eval()
  with torch.no_grad():
   ref=m(x);out=compiled(x);torch.testing.assert_close(out,ref,rtol=0 if dtype==torch.bfloat16 else 3e-5,atol=0 if dtype==torch.bfloat16 else 5e-6)
   inf=do_bench_cudagraph(lambda:compiled(x),rep=150)
  row={'shape':shape,'dtype':str(dtype),'relative_errors':errors,'train_graph_ms':train,'infer_graph_ms':inf};results.append(row);print(row,flush=True)
 Path('/home/psk6950/MiniWorld/runs/transition_triton_audit_20260917/module-check.json').write_text(json.dumps(results,indent=2))
