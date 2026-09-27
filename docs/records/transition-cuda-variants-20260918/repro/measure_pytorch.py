"""Static compiled module harness, nonzero squeeze, same BF16/FP32 contract."""
import argparse,gc,hashlib,json,socket,types
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.transition.module import Transition
from miniworld_engine.kernels.transition.cuda.variants import transition
from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b
from miniworld_engine.kernels.transition.triton.residual import transition_residual
from benchmarks.runners import bench
p=argparse.ArgumentParser();p.add_argument('--d',type=int,required=True);p.add_argument('--length',type=int,required=True);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
root=Path(__file__).parent
records={v:json.loads((root/f'tune-{v}-D{a.d}.json').read_text()) for v in ('streamed_k','full_k')}
for v in records:records[v]['best_norm']=json.loads((root/'norm-selections.json').read_text())[str(a.d)]['best_norm']
settings.configure(engine_backend='auto',autotune_miss_cap=24,transition_residual_fusion=True)
active=None

def new_forward(self,x):
 args=(x,self.ln_in.weight,self.ln_in.bias,self.expand_a.weight,self.expand_b.weight,self.squeeze.weight,self.ln_in.eps)
 if active=='triton_split':return transition_residual(*args)
 backend,variant=active.split(':');r=records[variant]
 if backend=='triton':return transition_wide_b2b(*args,config=r['best_triton_forward']['config'])
 return transition(*args,variant=variant,forward_config=r['best_forward']['config'],backward_config=r['best_backward']['config'],norm_config=r['best_norm']['config'])

class Fabric:
 @staticmethod
 def setup_module(model):
  gen=torch.Generator(device='cuda').manual_seed(987)
  with torch.no_grad():
   for index,layer in enumerate(model.layers):
    layer.squeeze.weight.normal_(std=a.d**-.5,generator=gen)
    assert layer.ln_in.weight.dtype==torch.float32
    if active.startswith('cuda:'):
     variant=active.split(':')[1];r=records[variant]
     native=Transition(a.d,implementation='cuda',cuda_variant=variant,
         cuda_forward_config=r['best_forward']['config'],cuda_backward_config=r['best_backward']['config'],
         cuda_norm_config=tuple(r['best_norm']['config'])).to(device=layer.squeeze.weight.device)
     native.load_state_dict(layer.state_dict());native.train(layer.training);model.layers[index]=native
    elif layer._backend!=KernelBackend.PYTORCH:layer.forward=types.MethodType(new_forward,layer)
  return model
 @staticmethod
 def backward(y,dy):y.backward(dy)

result=dict(node=socket.gethostname(),D=a.d,L=a.length,fixture='bench_module_transition',compile='dynamic=False, fullgraph=False',cudagraph='manual',precision='BF16 activation/weights; FP32 LN affine',squeeze='nonzero seed987',config_scope='L384 bounded seed winners reused at L768',configs={v:{k:r[k] for k in ('best_forward','best_backward','best_norm','best_triton_forward')} for v,r in records.items()},rows=[])
for repeat in range(a.repeats):
 for mode in ('inference',):
  for active in (['pytorch'][::(-1 if repeat%2 else 1)]):
   torch.compiler.reset();gc.collect();torch.cuda.empty_cache()
   conf=bench.BenchConfig(target='transition',level='module',mode=mode,metric='time',compile=True,cudagraph='manual',precision='bf16-mixed',d_pair=a.d,n_layers=1,min_seq_len=a.length,max_seq_len=a.length)
   print('START',repeat,mode,active,flush=True)
   side=bench.capture_stream();side.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(side):row=bench.bench_module_transition(conf,a.length,'pytorch',Fabric())._asdict()
   torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
   result['rows'].append(dict(repeat=repeat,mode=mode,arm=active,**row))
   (root/f'pytorch-D{a.d}-L{a.length}.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
   print('RESULT',json.dumps(result['rows'][-1],default=str),flush=True)
