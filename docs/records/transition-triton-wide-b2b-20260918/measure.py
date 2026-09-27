import argparse,gc,json,socket,types
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b, transition_wide_b2b_fused_ln
from benchmarks.runners import bench
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--width',type=int,default=128);p.add_argument('--repeats',type=int,default=2);p.add_argument('--auto-only',action='store_true');a=p.parse_args()
settings.configure(autotune_miss_cap=24,transition_residual_fusion=True,transition_h100_residual=True)
active=None
# Force the candidate at D256 as well; the production auto rule is measured
# separately and need not promote a candidate which loses there.
tuned=json.loads((Path(__file__).parent/f'selected-D{a.width}.json').read_text())
tuned_ln=json.loads((Path(__file__).parent/f'selected-ln-D{a.width}.json').read_text())
def b2b_forward(self,x):
 fn=transition_wide_b2b_fused_ln if active=='b2b_ln' else transition_wide_b2b
 return fn(x,self.ln_in.weight,self.ln_in.bias,self.expand_a.weight,self.expand_b.weight,self.squeeze.weight,self.ln_in.eps,config=(tuned_ln if active=='b2b_ln' else tuned)['config'])
class Fabric:
 @staticmethod
 def setup_module(model):
  gen=torch.Generator(device='cuda').manual_seed(987)
  with torch.no_grad():
   for layer in model.layers:
    layer.squeeze.weight.normal_(std=a.width**-.5,generator=gen)
    if layer._backend != KernelBackend.PYTORCH:
     if active in ('b2b','b2b_ln'):layer.forward=types.MethodType(b2b_forward,layer)
  return model
 @staticmethod
 def backward(y,dy):y.backward(dy)
result={'node':socket.gethostname(),'length':a.length,'width':a.width,'device':torch.cuda.get_device_name(),'fixture':'bench_module_transition; deterministic nonzero squeeze; explicit full-output Triton b2b; b2b_ln also fuses normalization','compile':'dynamic=False, fullgraph=False, observed graph count per row','cudagraph':'manual','b2b_config':tuned['config'],'native':'cache or declared default; full search not performed','autotune_miss_cap':24,'rows':[]}
for repeat in range(a.repeats):
 for mode in ['inference','training']:
  arms=['triton','auto','b2b','b2b_ln']
  if a.auto_only:arms=['triton','auto']
  if repeat%2:arms.reverse()
  for active in arms:
   settings.configure(engine_backend='auto' if active=='auto' else 'triton',transition_h100_save_xn=(active=='auto'))
   torch.compiler.reset();gc.collect();torch.cuda.empty_cache()
   conf=bench.BenchConfig(target='transition',level='module',mode=mode,metric='time',compile=True,cudagraph='manual',precision='bf16-mixed',d_pair=a.width,n_layers=1,min_seq_len=a.length,max_seq_len=a.length)
   print('START',repeat,mode,active,flush=True)
   side=bench.capture_stream();side.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(side):row=bench.bench_module_transition(conf,a.length,'triton',Fabric())._asdict()
   torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
   result['rows'].append({'repeat':repeat,'mode':mode,'backend':active,**row})
   print('RESULT',json.dumps(result['rows'][-1]),flush=True)
   (Path(__file__).parent/f'bench-L{a.length}-D{a.width}{"-auto" if a.auto_only else ""}.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
