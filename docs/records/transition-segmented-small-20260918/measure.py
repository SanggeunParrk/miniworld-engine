import argparse,gc,json,socket,types
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.modules.dispatch import KernelBackend

from benchmarks.runners import bench
from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b, transition_wide_b2b_fused_ln
from miniworld_engine.kernels.transition.triton.b2b_residual import transition_b2b_residual
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--width',type=int,default=128);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
settings.configure(autotune_miss_cap=24,transition_residual_fusion=True,transition_h100_residual=True)
active=None
selected=json.loads((Path(__file__).parent/f'selected-D{a.width}.json').read_text())
def wide_forward(self,x):
 args=(x,self.ln_in.weight,self.ln_in.bias,self.expand_a.weight,self.expand_b.weight,self.squeeze.weight,self.ln_in.eps)
 if active=='old_b2b':return transition_b2b_residual(*args)
 if active=='segmented_separate':return transition_wide_b2b(*args,config=selected['separate']['config'])
 return transition_wide_b2b_fused_ln(*args,config=selected['fused_'+mode]['config'])
class Fabric:
 @staticmethod
 def setup_module(model):
  gen=torch.Generator(device='cuda').manual_seed(987)
  with torch.no_grad():
   for layer in model.layers:
    layer.squeeze.weight.normal_(std=a.width**-.5,generator=gen)
    if layer._backend != KernelBackend.PYTORCH and active!='split':
     layer.forward=types.MethodType(wide_forward,layer)

  return model
 @staticmethod
 def backward(y,dy):y.backward(dy)
result={'node':socket.gethostname(),'length':a.length,'width':a.width,'device':torch.cuda.get_device_name(),'fixture':'bench_module_transition; default dispatch; BF16 activations, FP32 affine, identical split backward; nonzero squeeze','compile':'dynamic=False, fullgraph=False, observed graph count per row','cudagraph':'manual','autotune_miss_cap':24,'rows':[]}
for repeat in range(a.repeats):
 for mode in ['inference','training']:
  arms=['split','old_b2b','segmented_separate','segmented_fused']
  if repeat%2:arms.reverse()
  for active in arms:
   settings.configure(engine_backend='triton',transition_triton_b2b=active=='b2b')
   torch.compiler.reset();gc.collect();torch.cuda.empty_cache()
   conf=bench.BenchConfig(target='transition',level='module',mode=mode,metric='time',compile=True,cudagraph='manual',precision='bf16-mixed',d_pair=a.width,n_layers=1,min_seq_len=a.length,max_seq_len=a.length)
   print('START',repeat,mode,active,flush=True)
   side=bench.capture_stream();side.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(side):row=bench.bench_module_transition(conf,a.length,'triton',Fabric())._asdict()
   torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
   result['rows'].append({'repeat':repeat,'mode':mode,'backend':active,**row})
   print('RESULT',json.dumps(result['rows'][-1]),flush=True)
   (Path(__file__).parent/f'bench-L{a.length}-D{a.width}.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
