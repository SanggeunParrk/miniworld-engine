import argparse,json,gc,time,socket,hashlib
from pathlib import Path
import torch
from miniworld_engine import settings
from benchmarks.runners import bench
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--width',type=int,default=128);a=p.parse_args()
settings.configure(autotune_miss_cap=24)
class Fabric:
 @staticmethod
 def setup_module(model):
  # Identical nonzero squeeze weights for implementation and reference. Stock fixture is zero-init.
  gen=torch.Generator(device='cuda').manual_seed(987)
  with torch.no_grad():
   for layer in model.layers:layer.squeeze.weight.normal_(std=a.width**-.5,generator=gen)
  return model
 @staticmethod
 def backward(y,dy):y.backward(dy)
result={'node':socket.gethostname(),'length':a.length,'width':a.width,'device':torch.cuda.get_device_name(),'fixture':'bench_module_transition; deterministic nonzero squeeze adapter','compile':'static dynamic=False; official fixture fullgraph=False; graph count recorded per row','cudagraph':'manual','autotune_miss_cap':24,'native':'runtime selected cache or declared default; no claim of full tuning','rows':[]}
original=bench.measured_result
active=None
traces={}
def measure(**kw):
 row=original(**kw)
 for t in kw['grad_to_none']:t.grad=None
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  with torch.no_grad() if not kw['is_train'] else torch.enable_grad():kw['func']()
  torch.cuda.synchronize()
 traces[active]=[{'kernel':e.name,'us':e.device_time_total} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
 return row
bench.measured_result=measure
for repeat in range(2):
 for mode in ('inference','training'):
  arms=[('triton','triton',True),('legacy','auto',False),('h100','auto',True)]
  if repeat:arms.reverse()
  for name,backend,fuse in arms:
   settings.configure(engine_backend=backend,transition_residual_fusion=fuse)
   torch.compiler.reset();gc.collect();torch.cuda.empty_cache()
   conf=bench.BenchConfig(target='transition',level='module',mode=mode,metric='time',compile=True,cudagraph='manual',precision='bf16-mixed',d_pair=a.width,n_layers=1,min_seq_len=a.length,max_seq_len=a.length)
   active=f'{repeat}-{mode}-{name}'
   print('START',active,flush=True)
   side=bench.capture_stream();side.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(side):
    row=bench.bench_module_transition(conf,a.length,'triton',Fabric())._asdict()
   torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
   result['rows'].append({'repeat':repeat,'mode':mode,'backend':name,**row})
   print('RESULT',json.dumps(result['rows'][-1]),flush=True)
   result['traces']=traces
   (Path(__file__).parent/f'bench-L{a.length}-D{a.width}.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
