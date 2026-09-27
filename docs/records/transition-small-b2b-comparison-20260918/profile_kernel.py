import argparse,json
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.kernels.transition.triton.wide_b2b import launch
from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd,_ext
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--backend',choices=['cuda','triton'],required=True);a=p.parse_args()
settings.configure(autotune_miss_cap=24)
root=Path(__file__).parent;d=a.width;m=384**2
suffix='-cuda' if a.backend=='cuda' else ''
config=json.loads((root/f'selected{suffix}-D{d}.json').read_text())['inference']['config']
torch.manual_seed(678)
x=torch.randn(m,d,device='cuda',dtype=torch.bfloat16)
gamma=torch.rand(d,device='cuda',dtype=torch.bfloat16)+.5;beta=torch.randn_like(gamma)*.1
wa=torch.randn(4*d,d,device='cuda',dtype=torch.bfloat16)/d**.5
wb=torch.randn_like(wa)/d**.5
ws=torch.randn(d,4*d,device='cuda',dtype=torch.bfloat16)/d**.5
rs,c1=stats_triton(x,1e-5);out=torch.empty_like(x);empty=x.new_empty(0)
def call():
 if a.backend=='cuda':return transition_b2b_fwd(x,rs,c1,gamma,beta,wa,wb,ws,config=config)
 return launch(x,x,gamma,beta,rs,c1,wa,wb,ws,config=config,normalize=True,out=out,xn_out=empty)
result=call();call();torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart();call();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
metadata=dict(D=d,backend=a.backend,config=config,torch=torch.__version__,cuda=torch.version.cuda)
if a.backend=='triton':
 kernel=result[2]
 metadata.update(registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared)
 for ext in ('ptx','ttgir','cubin'):
  data=kernel.asm[ext];path=root/f'triton-D{d}.{ext}'
  if isinstance(data,bytes):path.write_bytes(data)
  else:path.write_text(data)
else:
 metadata['extension_path']=_ext('b2b',d,config).__file__
(root/f'metadata-{a.backend}-D{d}.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata),flush=True)
