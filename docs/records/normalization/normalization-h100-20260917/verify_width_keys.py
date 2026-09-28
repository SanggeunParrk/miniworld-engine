import json
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import cache
from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd,_transition_ln_bwd_kernel
settings.configure(engine_backend='triton',transition_lnbwd_cuda=False)
rows=[]
def observe(*args,**kwargs):
 amap=dict(zip(_transition_ln_bwd_kernel.arg_names,args));amap.update(kwargs)
 rows.append({'K':amap['K'],'shape_key':amap['shape_key'],'bucket':cache.bucket_of_autotuner(_transition_ln_bwd_kernel,amap)})
 # This probe verifies launch keys only. Numeric correctness was checked separately.
 return None
_transition_ln_bwd_kernel.run=observe
for width in (128,256,384,512,768):
 x=torch.zeros(384, width,device='cuda',dtype=torch.bfloat16);g=torch.ones(width,device='cuda');rs=torch.ones(384,device='cuda');c=torch.zeros_like(rs)
 _transition_ln_bwd(x,x,rs,c,g)
assert len({r['bucket'] for r in rows})==5,rows
print(json.dumps(rows,indent=2));Path(__file__).with_name('transition-width-keys.json').write_text(json.dumps(rows,indent=2)+'\n')
