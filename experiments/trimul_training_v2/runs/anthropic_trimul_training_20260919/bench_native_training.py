import importlib.util,json,gc,statistics,time
from pathlib import Path
import torch
from miniworld_engine.integrations.anthropic_training import triangle_multiplication_training,native_ops
E=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine')
spec=importlib.util.spec_from_file_location('checks', E/'tests/numerics/test_trimul_anthropic_native_training_gpu.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
R=Path('/home/psk6950/MiniWorld/runs/anthropic_trimul_training_20260919')
torch.backends.cuda.matmul.allow_tf32=False

def bench(fn):
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(2):fn()
  graph=torch.cuda.CUDAGraph()
  torch.cuda.reset_peak_memory_stats()
  with torch.cuda.graph(graph,stream=stream): out=fn()
  for _ in range(3):graph.replay()
  samples=[]
  for _ in range(15):
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   start.record();graph.replay();end.record();end.synchronize()
   samples.append(start.elapsed_time(end)*1000)
  peak=torch.cuda.max_memory_allocated()/2**30
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
 return {'median_us':statistics.median(samples),'min_us':min(samples),'max_us':max(samples),'peak_allocated_GiB':peak}

records=[]
for n in [384,768]:
 for direction in ['outgoing','bidirectional']:
  x,w,m,ds=checks.setup(n,direction);dy=torch.randn_like(x);leaves=(x,*w.values())
  def raw():return triangle_multiplication_training(x,m,weights=w,direction=direction)
  def fwd():return x+raw()*ds
  def train():
   y=fwd();g=torch.autograd.grad(y,leaves,dy);return y,g
  # Accuracy uses random nonzero weights and all 11 gradients, at actual target L.
  y,g=train();yr=x+checks.reference(x,w,m,direction)*ds
  gr=torch.autograd.grad(yr,leaves,dy)
  errors={k:checks.relative(a,b) for k,a,b in zip(['output','input',*w.keys()],[y,*g],[yr,*gr])}
  assert max(errors.values())<.01,errors
  del y,g,yr,gr
  times={}
  if direction=='outgoing':
   ops=native_ops();cache={}
   def original():
    with torch.no_grad():return ops.trimul(x,m,direction='outgoing',weights=w,residual=False,cache=cache)
   with torch.no_grad():torch.testing.assert_close(raw(),original(),rtol=0,atol=0)
   times['original_native_update_cached_weights']=bench(original)
   del cache
  times['native_training_forward_with_live_pack_dropout_residual']=bench(fwd)
  times['native_training_forward_backward_reference_bwd']=bench(train)
  # Capture actual launched names for proving full original forward route.
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   fwd();torch.cuda.synchronize()
  prof.export_chrome_trace(str(R/f'native-training-fwd-L{n}-{direction}.json'))
  row={'N':n,'direction':direction,'C':128,'H':256 if direction=='bidirectional' else 128,'B':1,'dropout':.25,'timing':'CUDA Graph replay, no optimizer/RNG','accuracy_relative_l2':errors,'times':times}
  records.append(row);(R/'native-training-times.json').write_text(json.dumps(records,indent=2))
  print(json.dumps(row),flush=True)
  del x,w,m,ds,dy,leaves;gc.collect();torch.cuda.empty_cache()
