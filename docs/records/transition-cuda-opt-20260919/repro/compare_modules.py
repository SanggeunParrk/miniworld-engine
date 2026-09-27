import argparse,gc,json,types,importlib.util,statistics
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.transition.module import Transition
import miniworld_engine.kernels.transition.cuda.variants as native
from miniworld_engine.kernels.transition.triton.wide_b2b import transition_wide_b2b
from miniworld_engine.kernels.transition.triton.residual import transition_residual
from benchmarks.runners import bench
p=argparse.ArgumentParser();p.add_argument('--d',type=int,required=True);p.add_argument('--length',type=int,required=True);p.add_argument('--repeats',type=int,default=2);a=p.parse_args();r=Path(__file__).parent;prior=r.parent/'transition_cuda_variants_20260918'
selected=json.loads((r/'final-selections.json').read_text());records={v:json.loads((prior/f'tune-{v}-D{a.d}.json').read_text()) for v in ('full_k','streamed_k')}
norm=json.loads((prior/'norm-selections.json').read_text())[str(a.d)]['best_norm']['config'];production_extension=native.extension;active='';loaded={}
def dl(path):
 if path not in loaded:
  s=importlib.util.spec_from_file_location(Path(path).stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);loaded[path]=m
 return loaded[path]
def configs(v,new):
 if not new:return records[v]['best_forward']['config'],records[v]['best_backward']['config']
 return tuple(next(t['config'] for t in selected if t['D']==a.d and t['variant']==v and t['direction']==d) for d in ('forward','backward'))
def ext(v,d,c):
 if active.startswith('old:'):
  t=records[v];return types.SimpleNamespace(forward=dl(t['best_forward']['extension']).forward,gate_backward=dl(t['best_backward']['extension']).gate_backward)
 return production_extension(v,d,c)
native.extension=ext
settings.configure(engine_backend='auto',autotune_miss_cap=24,transition_residual_fusion=True)
def fwd(self,x):
 args=(x,self.ln_in.weight,self.ln_in.bias,self.expand_a.weight,self.expand_b.weight,self.squeeze.weight,self.ln_in.eps)
 if active=='triton_best':
  if a.d>=384:return transition_residual(*args)
  return transition_wide_b2b(*args,config=records['full_k']['best_triton_forward']['config'])
 arm,v=active.split(':');fc,bc=configs(v,arm=='new')
 return native.transition(*args,variant=v,forward_config=fc,backward_config=bc,norm_config=norm)
class Fabric:
 @staticmethod
 def setup_module(model):
  gen=torch.Generator(device='cuda').manual_seed(987)
  with torch.no_grad():
   for i,layer in enumerate(model.layers):
    layer.squeeze.weight.normal_(std=a.d**-.5,generator=gen)
    assert layer.ln_in.weight.dtype==torch.float32
    assert layer.expand_a.weight.dtype==torch.bfloat16
    if layer._backend==KernelBackend.PYTORCH:continue
    if active.startswith(('old:','new:')):
     arm,v=active.split(':');fc,bc=configs(v,arm=='new')
     replacement=Transition(a.d,implementation='cuda',cuda_variant=v,cuda_forward_config=fc,cuda_backward_config=bc,cuda_norm_config=tuple(norm)).to(device=layer.squeeze.weight.device)
     replacement.load_state_dict(layer.state_dict());replacement.train(layer.training);model.layers[i]=replacement
    else:layer.forward=types.MethodType(fwd,layer)
  return model
 @staticmethod
 def backward(y,dy):y.backward(dy)
result=dict(D=a.d,L=a.length,reference_fixture='untouched_pytorch',fixture='bench_module_transition',compile='static',cudagraph='manual',precision='BF16 activations/weights; FP32 LN affine',rows=[])
for rep in range(a.repeats):
 for mode in ('inference','training'):
  arms=['old:full_k','new:full_k','old:streamed_k','new:streamed_k','triton_best']
  if rep%2:arms=arms[::-1]
  for active in arms:
   torch.compiler.reset();gc.collect();torch.cuda.empty_cache()
   cfg=bench.BenchConfig(target='transition',level='module',mode=mode,metric='time',compile=True,cudagraph='manual',precision='bf16-mixed',d_pair=a.d,n_layers=1,min_seq_len=a.length,max_seq_len=a.length)
   print('START',rep,mode,active,flush=True)
   side=bench.capture_stream();side.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(side):row=bench.bench_module_transition(cfg,a.length,'triton',Fabric())._asdict()
   torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
   result['rows'].append(dict(repeat=rep,mode=mode,arm=active,**row));(r/f'final-module-D{a.d}-L{a.length}.json').write_text(json.dumps(result,indent=2,default=str)+'\n');print('RESULT',json.dumps(result['rows'][-1],default=str),flush=True)
