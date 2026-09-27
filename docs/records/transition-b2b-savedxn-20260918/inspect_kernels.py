import json,socket
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.modules import Transition
from miniworld_engine.kernels.transition.hopper import transition_residual_hopper
settings.configure(autotune_miss_cap=24,engine_backend='auto',transition_residual_fusion=True)
torch.manual_seed(42)
side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
result={'node':socket.gethostname(),'length':768,'width':128,'kind':'torch.profiler single compiled forward/backward, not graph timing','traces':{}}
with torch.cuda.stream(side):
 model=Transition(128).cuda().bfloat16()
 with torch.no_grad():model.squeeze.weight.normal_(std=128**-.5)
 x=torch.randn(1,768,768,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);dy=torch.randn_like(x)
 leaves=(x,*model.parameters())
 funcs={}
 for name,saved in [('recompute',False),('saved',True)]:
  def fn(x,saved=saved):
   return transition_residual_hopper(x,model.ln_in.weight,model.ln_in.bias,model.expand_a.weight,model.expand_b.weight,model.squeeze.weight,save_xn=saved)
  funcs[name]=torch.compile(fn,fullgraph=True,dynamic=False)
 for name,fn in funcs.items():
  for _ in range(3):
   for t in leaves:t.grad=None
   fn(x).backward(dy)
 torch.cuda.synchronize()
 for name,fn in funcs.items():
  for t in leaves:t.grad=None
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
   fn(x).backward(dy)
   torch.cuda.synchronize()
  events=[{'kernel':e.name,'us':e.device_time_total} for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA and not e.name.startswith('##')]
  assert any('_transition_expand_gatebwd_kernel' in e['kernel'] for e in events),events
  result['traces'][name]=events
  print(name,[(e['kernel'][:75],round(e['us'],2)) for e in events],flush=True)
torch.cuda.current_stream().wait_stream(side)
(Path(__file__).parent/'profile-L768-D128.json').write_text(json.dumps(result,indent=2)+'\n')
