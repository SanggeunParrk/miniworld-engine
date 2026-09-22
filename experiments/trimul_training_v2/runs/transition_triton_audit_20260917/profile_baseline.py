import argparse,json,copy
from pathlib import Path
import torch
import miniworld_engine
from miniworld_engine import settings
settings.configure(engine_backend='triton',compile_wrap='custom_op')
from miniworld_engine.modules import Transition
p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=128);p.add_argument('--length',type=int,default=384);p.add_argument('--layout',choices=['pair','token'],default='pair');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(917);torch.backends.cuda.matmul.allow_tf32=False
shape=(1,a.length,a.length,a.width) if a.layout=='pair' else (1,a.length,a.width)
m=Transition(a.width,implementation='triton').cuda().bfloat16()
with torch.no_grad():
 for v in m.parameters():
  if v.ndim==2:v.normal_(std=a.width**-.5)
x=torch.randn(shape,device='cuda',dtype=torch.bfloat16,requires_grad=True);dy=torch.randn_like(x)
compiled=torch.compile(m,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
result={'package':miniworld_engine.__file__,'shape':shape,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'cases':{}}
for mode in ['inference','training']:
 m.train(mode=='training')
 for arm,model in [('eager',m),('compile',compiled)]:
  for _ in range(3):
   if mode=='training':
    m.zero_grad(set_to_none=True);x.grad=None;y=model(x);y.backward(dy)
   else:
    with torch.no_grad():y=model(x)
  torch.cuda.synchronize()
  for phase in (['forward','backward'] if mode=='training' else ['forward']):
   m.zero_grad(set_to_none=True);x.grad=None
   if phase=='backward':y=model(x)
   torch.cuda.synchronize()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
    if phase=='backward':y.backward(dy)
    elif mode=='training':y=model(x)
    else:
     with torch.no_grad():y=model(x)
    torch.cuda.synchronize()
   name=f'{mode}-{arm}-{phase}'
   prof.export_chrome_trace(str(a.out/(name+'.trace.json')))
   cuda=[{'name':e.name,'us':e.device_time_total} for e in prof.events() if str(e.device_type).endswith('CUDA')]
   cpu=[{'name':e.key,'count':e.count,'shapes':str(e.input_shapes),'device_us':e.device_time_total,'memory':e.device_memory_usage} for e in prof.key_averages(group_by_input_shape=True) if any(s in e.key for s in ['aten::cat','aten::add','aten::mm','aten::clone','aten::copy','aten::contiguous','miniworld'])]
   result['cases'][name]={'kernels':cuda,'operators':cpu}
   print(name,[(e['name'],round(e['us'],2)) for e in cuda],flush=True)
(a.out/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
