import sys,json,torch
from experiment import *
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'))
from bench_k3 import graph,paired
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_materialize
from miniworld_engine.autotune.shape_key import both_key
rows=[]
for n in (384,768):
 for trans,c in ((False,128),(True,256)):
  x=torch.randn((c,n*n) if trans else (n*n,c),device='cuda',dtype=torch.bfloat16)
  g=torch.rand(c,device='cuda');b=torch.randn_like(g)
  gs={'triton':graph(lambda:_ln_materialize(x.t() if trans else x,g,b,1e-5,shape_key=both_key(n*n)))}
  gs.update({f'anthropic-tma-store{bulk}':graph(lambda bulk=bulk:ln_tma(x,g,b,trans=trans,bulk=bulk)) for bulk in (0,1)})
  rows.append(dict(N=n,C=c,transposed=trans,times=paired(gs,reps=30,rounds=8)))
  print('DONE',n,c,flush=True)
(R/'ln-reference.json').write_text(json.dumps(rows,indent=2))
